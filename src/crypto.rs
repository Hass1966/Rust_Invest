use crate::models::CoinData;
use serde::Deserialize;

// ── Historical data types ──

#[derive(Debug, Deserialize)]
pub struct MarketChart {
    pub prices: Vec<[f64; 2]>,        // [timestamp, price]
    pub total_volumes: Vec<[f64; 2]>,  // [timestamp, volume]
}

// ── API functions ──

pub async fn fetch_top_coins(client: &reqwest::Client) -> Result<Vec<CoinData>, Box<dyn std::error::Error>> {
    let url = "https://api.coingecko.com/api/v3/coins/markets\
        ?vs_currency=usd\
        &order=market_cap_desc\
        &per_page=20\
        &page=1\
        &sparkline=false";

    let response = client
        .get(url)
        .header("User-Agent", "RustInvest/0.1")
        .send()
        .await?;

    let body = response.text().await?;
    let coins: Vec<CoinData> = serde_json::from_str(&body)?;
    Ok(coins)
}

pub async fn fetch_history(
    client: &reqwest::Client,
    coin_id: &str,
    days: u32,
) -> Result<MarketChart, Box<dyn std::error::Error>> {
    let url = format!(
        "https://api.coingecko.com/api/v3/coins/{}/market_chart?vs_currency=usd&days={}",
        coin_id, days
    );

    let response = client
        .get(&url)
        .header("User-Agent", "RustInvest/0.1")
        .send()
        .await?;

    let body = response.text().await?;
    let chart: MarketChart = serde_json::from_str(&body)?;
    Ok(chart)
}