/// News & Sentiment Module
/// =======================
/// Fetches news from Serper (Google Search API), NewsAPI.org, and Reddit.
/// Uses Claude LLM for intelligent sentiment analysis that captures
/// geopolitical events, wars, macro trends, and modern market influences.
/// Falls back to word-based scoring when LLM is unavailable.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use chrono::{Utc, Duration};
use crate::llm;

// ════════════════════════════════════════
// Sentiment word lists (fallback)
// ════════════════════════════════════════

const POSITIVE_WORDS: &[&str] = &[
    "buy", "bull", "bullish", "up", "rise", "growth", "strong", "beat",
    "exceed", "upgrade", "calls", "surge", "rally", "gain", "profit",
    "outperform", "breakout", "rocket", "moon", "long",
];

const NEGATIVE_WORDS: &[&str] = &[
    "sell", "bear", "bearish", "down", "fall", "drop", "weak", "miss",
    "downgrade", "puts", "crash", "dump", "loss", "plunge", "decline",
    "underperform", "short", "tank", "collapse", "bankruptcy",
];

// ════════════════════════════════════════
// Types
// ════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SentimentData {
    pub symbol: String,
    pub date: String,
    pub news_score: f64,
    pub reddit_mentions: i64,
    pub reddit_score: f64,
    pub combined_score: f64,
    pub article_count: i64,
    #[serde(default)]
    pub llm_analysis: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NewsApiResponse {
    #[serde(default)]
    articles: Vec<NewsArticle>,
}

#[derive(Debug, Deserialize, Clone)]
struct NewsArticle {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RedditResponse {
    #[serde(default)]
    data: RedditData,
}

#[derive(Debug, Deserialize, Default)]
struct RedditData {
    #[serde(default)]
    children: Vec<RedditChild>,
}

#[derive(Debug, Deserialize)]
struct RedditChild {
    data: RedditPost,
}

#[derive(Debug, Deserialize)]
struct RedditPost {
    #[serde(default)]
    title: String,
    #[serde(default)]
    selftext: String,
}

// ════════════════════════════════════════
// Serper API types (Google Search)
// ════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct SerperResponse {
    #[serde(default)]
    organic: Vec<SerperResult>,
    #[serde(default)]
    news: Vec<SerperNewsResult>,
}

#[derive(Debug, Deserialize)]
struct SerperResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    snippet: String,
}

#[derive(Debug, Deserialize)]
struct SerperNewsResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    date: Option<String>,
}

// ════════════════════════════════════════
// Database
// ════════════════════════════════════════

pub fn create_sentiment_table(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS news_sentiment (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            date            TEXT NOT NULL,
            news_score      REAL DEFAULT 0.0,
            reddit_mentions INTEGER DEFAULT 0,
            reddit_score    REAL DEFAULT 0.0,
            combined_score  REAL DEFAULT 0.0,
            article_count   INTEGER DEFAULT 0,
            llm_analysis    TEXT DEFAULT NULL,
            fetched_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol ON news_sentiment(symbol, date);
        "
    )?;
    // Add llm_analysis column if table already exists without it
    let _ = conn.execute("ALTER TABLE news_sentiment ADD COLUMN llm_analysis TEXT DEFAULT NULL", []);
    Ok(())
}

pub fn store_sentiment(conn: &Connection, data: &SentimentData) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO news_sentiment (symbol, date, news_score, reddit_mentions, reddit_score, combined_score, article_count, llm_analysis)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            data.symbol, data.date, data.news_score,
            data.reddit_mentions, data.reddit_score,
            data.combined_score, data.article_count,
            data.llm_analysis,
        ],
    )?;
    Ok(())
}

pub fn get_sentiment_history(conn: &Connection, symbol: &str, days: i64) -> Vec<SentimentData> {
    let cutoff = (Utc::now() - Duration::days(days)).format("%Y-%m-%d").to_string();
    let mut stmt = match conn.prepare(
        "SELECT symbol, date, news_score, reddit_mentions, reddit_score, combined_score, article_count, llm_analysis
         FROM news_sentiment WHERE symbol = ?1 AND date >= ?2 ORDER BY date DESC"
    ) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    stmt.query_map(params![symbol, cutoff], |row| {
        Ok(SentimentData {
            symbol: row.get(0)?,
            date: row.get(1)?,
            news_score: row.get(2)?,
            reddit_mentions: row.get(3)?,
            reddit_score: row.get(4)?,
            combined_score: row.get(5)?,
            article_count: row.get(6)?,
            llm_analysis: row.get(7).ok(),
        })
    })
    .ok()
    .map(|rows| rows.filter_map(|r| r.ok()).collect())
    .unwrap_or_default()
}

/// Get latest sentiment for a symbol (for ML feature extraction)
pub fn get_recent_sentiment(conn: &Connection, symbol: &str, days: i64) -> Vec<SentimentData> {
    get_sentiment_history(conn, symbol, days)
}

/// Check if we already have sentiment for this symbol today
pub fn has_today_sentiment(conn: &Connection, symbol: &str) -> bool {
    let today = Utc::now().format("%Y-%m-%d").to_string();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM news_sentiment WHERE symbol = ?1 AND date = ?2",
            params![symbol, today],
            |row| row.get(0),
        )
        .unwrap_or(0);
    count > 0
}

// ════════════════════════════════════════
// Sentiment scoring (word-based fallback)
// ════════════════════════════════════════

fn score_text(text: &str) -> f64 {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let total = words.len() as f64;
    if total == 0.0 {
        return 0.0;
    }

    let pos = words.iter().filter(|w| POSITIVE_WORDS.contains(w)).count() as f64;
    let neg = words.iter().filter(|w| NEGATIVE_WORDS.contains(w)).count() as f64;

    ((pos - neg) / total).clamp(-1.0, 1.0)
}

// ════════════════════════════════════════
// Serper API (Google Search for news)
// ════════════════════════════════════════

/// Fetch recent news via Serper (Google Search API) for broader coverage.
/// Returns a list of (title, snippet) pairs.
pub async fn fetch_serper_news(
    client: &reqwest::Client,
    symbol: &str,
    serper_key: &str,
) -> Result<Vec<(String, String)>, String> {
    let query = clean_symbol_for_search(symbol);
    let asset_type = if symbol.ends_with("=X") { "forex currency" }
        else if symbol.ends_with(".L") { "stock UK" }
        else if matches!(symbol, "bitcoin"|"ethereum"|"solana"|"ripple"|"dogecoin"|"cardano"|"avalanche-2"|"chainlink"|"polkadot"|"near"|"sui"|"aptos"|"arbitrum"|"the-open-network"|"uniswap"|"tron"|"litecoin"|"shiba-inu"|"stellar"|"matic-network") { "cryptocurrency" }
        else { "stock market" };

    let search_query = format!("{} {} news analysis 2026", query, asset_type);

    let body = serde_json::json!({
        "q": search_query,
        "num": 10,
        "type": "news"
    });

    let resp = client
        .post("https://google.serper.dev/news")
        .header("X-API-KEY", serper_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| format!("Serper request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Serper returned {}", resp.status()));
    }

    let data: SerperResponse = resp.json().await
        .map_err(|e| format!("Serper parse error: {}", e))?;

    let mut results: Vec<(String, String)> = Vec::new();

    // Prioritise news results
    for item in &data.news {
        results.push((item.title.clone(), item.snippet.clone()));
    }
    // Also include organic results
    for item in &data.organic {
        results.push((item.title.clone(), item.snippet.clone()));
    }

    Ok(results)
}

// ════════════════════════════════════════
// Claude LLM Sentiment Analysis
// ════════════════════════════════════════

/// Use Claude to analyse news articles and return a sentiment score + reasoning.
/// This captures geopolitical events, wars, macro influences, sector dynamics —
/// things that word-counting completely misses.
pub async fn analyse_sentiment_with_llm(
    client: &reqwest::Client,
    provider: &llm::LlmProvider,
    symbol: &str,
    headlines: &[(String, String)],
    reddit_texts: &[String],
) -> Result<(f64, String), String> {
    if headlines.is_empty() && reddit_texts.is_empty() {
        return Ok((0.0, "No news data available".to_string()));
    }

    let clean_sym = clean_symbol_for_search(symbol);

    // Build the news digest for Claude
    let mut news_digest = String::new();
    news_digest.push_str("=== Recent News Headlines & Snippets ===\n");
    for (i, (title, snippet)) in headlines.iter().take(15).enumerate() {
        news_digest.push_str(&format!("{}. {} — {}\n", i + 1, title, snippet));
    }

    if !reddit_texts.is_empty() {
        news_digest.push_str("\n=== Reddit Discussion Snippets ===\n");
        for (i, text) in reddit_texts.iter().take(10).enumerate() {
            // Truncate long reddit posts
            let truncated = if text.len() > 300 { &text[..300] } else { text };
            news_digest.push_str(&format!("{}. {}\n", i + 1, truncated));
        }
    }

    let system_prompt = format!(
        "You are a senior financial analyst specialising in market sentiment analysis. \
         Your job is to analyse news and social media about {} ({}) and produce a precise \
         sentiment score.\n\n\
         IMPORTANT: Consider ALL of the following factors:\n\
         - Geopolitical events (wars, sanctions, trade disputes, elections)\n\
         - Macroeconomic indicators (inflation, interest rates, employment)\n\
         - Sector-specific dynamics (regulation, competition, innovation)\n\
         - Market momentum and technical sentiment\n\
         - Social media buzz and retail investor sentiment\n\
         - Supply chain disruptions or commodity price impacts\n\
         - Central bank policy and monetary decisions\n\n\
         RESPOND IN EXACTLY THIS FORMAT (no other text):\n\
         SCORE: <number from -1.0 to 1.0>\n\
         ANALYSIS: <2-3 sentence summary of key factors>\n\n\
         Score guide:\n\
         -1.0 = Extremely bearish (imminent crisis/collapse)\n\
         -0.5 = Bearish (significant negative catalysts)\n\
          0.0 = Neutral (mixed or no clear direction)\n\
         +0.5 = Bullish (significant positive catalysts)\n\
         +1.0 = Extremely bullish (strong positive momentum)",
        clean_sym, symbol
    );

    let user_message = format!(
        "Analyse the following news and social data for {} and provide your sentiment score:\n\n{}",
        clean_sym, news_digest
    );

    let response = llm::chat(client, provider, &system_prompt, &user_message).await?;

    // Parse the response
    parse_llm_sentiment(&response)
}

/// Parse Claude's response into (score, analysis)
fn parse_llm_sentiment(response: &str) -> Result<(f64, String), String> {
    let mut score: Option<f64> = None;
    let mut analysis = String::new();

    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("SCORE:") {
            let score_str = trimmed.trim_start_matches("SCORE:").trim();
            score = score_str.parse::<f64>().ok();
        } else if trimmed.starts_with("ANALYSIS:") {
            analysis = trimmed.trim_start_matches("ANALYSIS:").trim().to_string();
        }
    }

    match score {
        Some(s) => Ok((s.clamp(-1.0, 1.0), if analysis.is_empty() { response.to_string() } else { analysis })),
        None => {
            // Fallback: try to extract any number from the response
            for word in response.split_whitespace() {
                if let Ok(n) = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '-' && c != '.').parse::<f64>() {
                    if (-1.0..=1.0).contains(&n) {
                        return Ok((n, response.to_string()));
                    }
                }
            }
            Err(format!("Could not parse LLM sentiment from: {}", &response[..response.len().min(200)]))
        }
    }
}

// ════════════════════════════════════════
// NewsAPI fetching
// ════════════════════════════════════════

pub async fn fetch_news_articles(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Result<Vec<NewsArticle>, String> {
    let query = clean_symbol_for_search(symbol);
    let from = (Utc::now() - Duration::days(3)).format("%Y-%m-%d").to_string();

    let url = format!(
        "https://newsapi.org/v2/everything?q={} stock&from={}&sortBy=relevancy&pageSize=20&apiKey={}",
        query, from, api_key,
    );

    let resp = client.get(&url).send().await
        .map_err(|e| format!("NewsAPI request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("NewsAPI returned {}", resp.status()));
    }

    let data: NewsApiResponse = resp.json().await
        .map_err(|e| format!("NewsAPI parse error: {}", e))?;

    Ok(data.articles)
}

/// Fetch news and score with word-based method (fallback)
pub async fn fetch_news_sentiment(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Result<(f64, i64), String> {
    let articles = fetch_news_articles(client, symbol, api_key).await?;
    let article_count = articles.len() as i64;
    if article_count == 0 {
        return Ok((0.0, 0));
    }

    let total_score: f64 = articles.iter().map(|a| {
        let title = a.title.as_deref().unwrap_or("");
        let desc = a.description.as_deref().unwrap_or("");
        score_text(&format!("{} {}", title, desc))
    }).sum();

    let avg_score = total_score / article_count as f64;
    Ok((avg_score.clamp(-1.0, 1.0), article_count))
}

// ════════════════════════════════════════
// Reddit fetching
// ════════════════════════════════════════

pub async fn fetch_reddit_posts(
    client: &reqwest::Client,
    symbol: &str,
) -> Result<(i64, f64, Vec<String>), String> {
    let query = clean_symbol_for_search(symbol);
    let subreddits = ["wallstreetbets", "investing", "stocks"];

    let mut total_mentions: i64 = 0;
    let mut total_score: f64 = 0.0;
    let mut texts: Vec<String> = Vec::new();

    for sub in &subreddits {
        let url = format!(
            "https://www.reddit.com/r/{}/search.json?q={}&sort=new&limit=25&restrict_sr=on&t=day",
            sub, query,
        );

        let resp = client
            .get(&url)
            .header("User-Agent", "rust-invest-bot/1.0")
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(_) => continue,
        };

        if !resp.status().is_success() {
            continue;
        }

        let data: RedditResponse = match resp.json().await {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mentions = data.data.children.len() as i64;
        total_mentions += mentions;

        for child in &data.data.children {
            let text = format!("{} {}", child.data.title, child.data.selftext);
            total_score += score_text(&text);
            texts.push(text);
        }
    }

    let avg_score = if total_mentions > 0 {
        (total_score / total_mentions as f64).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Ok((total_mentions, avg_score, texts))
}

/// Legacy Reddit fetch (for backward compat)
pub async fn fetch_reddit_sentiment(
    client: &reqwest::Client,
    symbol: &str,
) -> Result<(i64, f64), String> {
    let (mentions, score, _texts) = fetch_reddit_posts(client, symbol).await?;
    Ok((mentions, score))
}

// ════════════════════════════════════════
// Combined fetch with LLM analysis
// ════════════════════════════════════════

/// Full pipeline: Serper + NewsAPI + Reddit → Claude LLM analysis → store
pub async fn fetch_and_store_sentiment(
    client: &reqwest::Client,
    conn: &Connection,
    symbol: &str,
    newsapi_key: &str,
) -> Result<SentimentData, String> {
    let today = Utc::now().format("%Y-%m-%d").to_string();
    let serper_key = std::env::var("SERPER_API_KEY").ok();
    let llm_provider = llm::load_provider();

    // 1. Fetch news from Serper (Google Search) if key available
    let serper_headlines: Vec<(String, String)> = if let Some(ref key) = serper_key {
        fetch_serper_news(client, symbol, key).await.unwrap_or_default()
    } else {
        vec![]
    };

    // 2. Fetch from NewsAPI
    let newsapi_articles = fetch_news_articles(client, symbol, newsapi_key).await.unwrap_or_default();
    let newsapi_headlines: Vec<(String, String)> = newsapi_articles.iter().map(|a| {
        (
            a.title.clone().unwrap_or_default(),
            a.description.clone().unwrap_or_default(),
        )
    }).collect();

    // 3. Fetch from Reddit
    let (reddit_mentions, reddit_word_score, reddit_texts) =
        fetch_reddit_posts(client, symbol).await.unwrap_or((0, 0.0, vec![]));

    // Combine all headlines (Serper first, then NewsAPI — deduplicate by title)
    let mut all_headlines: Vec<(String, String)> = serper_headlines;
    for (title, desc) in &newsapi_headlines {
        if !all_headlines.iter().any(|(t, _)| t == title) {
            all_headlines.push((title.clone(), desc.clone()));
        }
    }

    let article_count = all_headlines.len() as i64;

    // 4. Analyse with Claude LLM if available
    let (news_score, llm_analysis) = if let Some(ref provider) = llm_provider {
        match analyse_sentiment_with_llm(client, provider, symbol, &all_headlines, &reddit_texts).await {
            Ok((score, analysis)) => {
                println!("    LLM sentiment for {}: {:.2} — {}", symbol, score, &analysis[..analysis.len().min(80)]);
                (score, Some(analysis))
            }
            Err(e) => {
                eprintln!("    LLM analysis failed for {}: {}, falling back to word-based", symbol, e);
                let word_score = score_headlines_word_based(&all_headlines);
                (word_score, None)
            }
        }
    } else {
        // Fallback: word-based scoring
        let word_score = score_headlines_word_based(&all_headlines);
        (word_score, None)
    };

    // Combined score: LLM news score 60%, reddit word-based 40%
    let combined = if article_count > 0 && reddit_mentions > 0 {
        news_score * 0.6 + reddit_word_score * 0.4
    } else if article_count > 0 {
        news_score
    } else {
        reddit_word_score
    };

    let data = SentimentData {
        symbol: symbol.to_string(),
        date: today,
        news_score,
        reddit_mentions,
        reddit_score: reddit_word_score,
        combined_score: combined.clamp(-1.0, 1.0),
        article_count,
        llm_analysis,
    };

    store_sentiment(conn, &data).map_err(|e| format!("DB error: {}", e))?;
    Ok(data)
}

/// Score headlines using word-based method (fallback)
fn score_headlines_word_based(headlines: &[(String, String)]) -> f64 {
    if headlines.is_empty() {
        return 0.0;
    }
    let total_score: f64 = headlines.iter().map(|(title, desc)| {
        score_text(&format!("{} {}", title, desc))
    }).sum();
    (total_score / headlines.len() as f64).clamp(-1.0, 1.0)
}

// ════════════════════════════════════════
// ML Feature extraction
// ════════════════════════════════════════

/// Extract sentiment features for ML models.
/// Returns (news_sentiment_3d, reddit_mentions_norm, reddit_sentiment, sentiment_momentum).
/// All default to 0.0 if no data.
pub fn extract_sentiment_features(conn: &Connection, symbol: &str) -> (f64, f64, f64, f64) {
    let data = get_recent_sentiment(conn, symbol, 7);

    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // news_sentiment_3d: 3-day rolling average
    let recent_3 = &data[..data.len().min(3)];
    let news_3d = if !recent_3.is_empty() {
        recent_3.iter().map(|d| d.news_score).sum::<f64>() / recent_3.len() as f64
    } else {
        0.0
    };

    // reddit_mentions_norm: normalise to 0-1 range (cap at 200 mentions)
    let latest = &data[0];
    let reddit_norm = (latest.reddit_mentions as f64 / 200.0).min(1.0);

    // reddit_sentiment: direct score
    let reddit_sent = latest.reddit_score;

    // sentiment_momentum: today vs yesterday
    let momentum = if data.len() >= 2 {
        data[0].combined_score - data[1].combined_score
    } else {
        0.0
    };

    (news_3d, reddit_norm, reddit_sent, momentum)
}

// ════════════════════════════════════════
// Helpers
// ════════════════════════════════════════

fn clean_symbol_for_search(symbol: &str) -> String {
    let clean = symbol
        .replace("=X", "")
        .replace(".L", "")
        .replace("-", " ");
    // Map crypto IDs to common names
    match clean.as_str() {
        "bitcoin" => "Bitcoin BTC".to_string(),
        "ethereum" => "Ethereum ETH".to_string(),
        "solana" => "Solana SOL".to_string(),
        "ripple" => "Ripple XRP".to_string(),
        "dogecoin" => "Dogecoin DOGE".to_string(),
        "cardano" => "Cardano ADA".to_string(),
        "avalanche 2" => "Avalanche AVAX".to_string(),
        "chainlink" => "Chainlink LINK".to_string(),
        "polkadot" => "Polkadot DOT".to_string(),
        "near" => "NEAR Protocol".to_string(),
        "sui" => "Sui crypto".to_string(),
        "aptos" => "Aptos APT".to_string(),
        "arbitrum" => "Arbitrum ARB".to_string(),
        "the open network" => "Toncoin TON".to_string(),
        "uniswap" => "Uniswap UNI".to_string(),
        "tron" => "TRON TRX".to_string(),
        "litecoin" => "Litecoin LTC".to_string(),
        "shiba inu" => "Shiba Inu SHIB".to_string(),
        "stellar" => "Stellar XLM".to_string(),
        "matic network" => "Polygon MATIC".to_string(),
        _ => clean,
    }
}
