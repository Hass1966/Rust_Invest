/// Tiingo — Supplementary news sentiment data source
/// ==================================================
/// Provides additional news articles alongside NewsAPI and Serper.
/// Tiingo's news endpoint returns articles with pre-computed sentiment tags.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TiingoArticle {
    pub title: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "publishedDate")]
    pub published_date: Option<String>,
    pub source: Option<String>,
}

/// Fetch news articles from Tiingo for a given symbol
/// Returns Vec<(title, description)> matching the pattern used by Serper/NewsAPI
pub async fn fetch_news(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    // Clean up the symbol for Tiingo (remove =X suffix for FX, .L for UK)
    let tiingo_symbol = if symbol.ends_with("=X") {
        // FX pairs: EURUSD=X -> eurusd
        symbol.trim_end_matches("=X").to_lowercase()
    } else if symbol.ends_with(".L") {
        // UK stocks: HSBA.L -> hsba
        symbol.trim_end_matches(".L").to_lowercase()
    } else {
        symbol.to_lowercase()
    };

    let from_date = (chrono::Utc::now() - chrono::Duration::days(3))
        .format("%Y-%m-%dT%H:%M:%S")
        .to_string();

    let url = format!(
        "https://api.tiingo.com/tiingo/news?tickers={}&startDate={}&limit=20&token={}",
        tiingo_symbol, from_date, api_key
    );

    let resp = client
        .get(&url)
        .header("Content-Type", "application/json")
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        return Err(format!("Tiingo API error: HTTP {}", status).into());
    }

    let articles: Vec<TiingoArticle> = resp.json().await?;

    let results: Vec<(String, String)> = articles
        .into_iter()
        .filter_map(|a| {
            let title = a.title.unwrap_or_default();
            let desc = a.description.unwrap_or_default();
            if title.is_empty() {
                None
            } else {
                Some((title, desc))
            }
        })
        .collect();

    Ok(results)
}

/// Fetch Tiingo news and return as headline strings for sentiment pipeline
pub async fn fetch_headlines(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Vec<(String, String)> {
    match fetch_news(client, symbol, api_key).await {
        Ok(articles) => {
            if !articles.is_empty() {
                println!("    [Tiingo] {} articles for {}", articles.len(), symbol);
            }
            articles
        }
        Err(e) => {
            println!("    [Tiingo] {} for {} — continuing without", e, symbol);
            Vec::new()
        }
    }
}
