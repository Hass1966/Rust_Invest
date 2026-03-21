/// News & Sentiment Module
/// =======================
/// Fetches news from NewsAPI.org and Reddit, computes simple word-based
/// sentiment scores, and stores in the news_sentiment table.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use chrono::{Utc, Duration};

// ════════════════════════════════════════
// Sentiment word lists
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
}

#[derive(Debug, Deserialize)]
struct NewsApiResponse {
    #[serde(default)]
    articles: Vec<NewsArticle>,
}

#[derive(Debug, Deserialize)]
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
            fetched_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol ON news_sentiment(symbol, date);
        "
    )
}

pub fn store_sentiment(conn: &Connection, data: &SentimentData) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO news_sentiment (symbol, date, news_score, reddit_mentions, reddit_score, combined_score, article_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            data.symbol, data.date, data.news_score,
            data.reddit_mentions, data.reddit_score,
            data.combined_score, data.article_count,
        ],
    )?;
    Ok(())
}

pub fn get_sentiment_history(conn: &Connection, symbol: &str, days: i64) -> Vec<SentimentData> {
    let cutoff = (Utc::now() - Duration::days(days)).format("%Y-%m-%d").to_string();
    let mut stmt = match conn.prepare(
        "SELECT symbol, date, news_score, reddit_mentions, reddit_score, combined_score, article_count
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

// ════════════════════════════════════════
// Sentiment scoring
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
// NewsAPI fetching
// ════════════════════════════════════════

pub async fn fetch_news_sentiment(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Result<(f64, i64), String> {
    // Clean symbol for search query
    let query = symbol.replace("=X", "").replace(".L", "");
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

    let article_count = data.articles.len() as i64;
    if article_count == 0 {
        return Ok((0.0, 0));
    }

    let total_score: f64 = data.articles.iter().map(|a| {
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

pub async fn fetch_reddit_sentiment(
    client: &reqwest::Client,
    symbol: &str,
) -> Result<(i64, f64), String> {
    let query = symbol.replace("=X", "").replace(".L", "");
    let subreddits = ["wallstreetbets", "investing", "stocks"];

    let mut total_mentions: i64 = 0;
    let mut total_score: f64 = 0.0;

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
        }
    }

    let avg_score = if total_mentions > 0 {
        (total_score / total_mentions as f64).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Ok((total_mentions, avg_score))
}

// ════════════════════════════════════════
// Combined fetch for one symbol
// ════════════════════════════════════════

pub async fn fetch_and_store_sentiment(
    client: &reqwest::Client,
    conn: &Connection,
    symbol: &str,
    newsapi_key: &str,
) -> Result<SentimentData, String> {
    let today = Utc::now().format("%Y-%m-%d").to_string();

    // Fetch from both sources
    let (news_score, article_count) = fetch_news_sentiment(client, symbol, newsapi_key)
        .await
        .unwrap_or((0.0, 0));

    let (reddit_mentions, reddit_score) = fetch_reddit_sentiment(client, symbol)
        .await
        .unwrap_or((0, 0.0));

    // Combined: weight news 60%, reddit 40%
    let combined = if article_count > 0 && reddit_mentions > 0 {
        news_score * 0.6 + reddit_score * 0.4
    } else if article_count > 0 {
        news_score
    } else {
        reddit_score
    };

    let data = SentimentData {
        symbol: symbol.to_string(),
        date: today,
        news_score,
        reddit_mentions,
        reddit_score,
        combined_score: combined.clamp(-1.0, 1.0),
        article_count,
    };

    store_sentiment(conn, &data).map_err(|e| format!("DB error: {}", e))?;
    Ok(data)
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
