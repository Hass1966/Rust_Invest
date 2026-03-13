use crate::models::CoinData;
use serde::Deserialize;

// ── Historical data types ──

#[derive(Debug, Deserialize)]
pub struct MarketChart {
    pub prices: Vec<[f64; 2]>,        // [timestamp, price]
    pub total_volumes: Vec<[f64; 2]>,  // [timestamp, volume]
}

// ── CoinGecko ID → Binance symbol mapping ──

fn coingecko_to_binance(coin_id: &str) -> Option<&'static str> {
    match coin_id {
        "bitcoin"           => Some("BTCUSDT"),
        "ethereum"          => Some("ETHUSDT"),
        "solana"            => Some("SOLUSDT"),
        "ripple"            => Some("XRPUSDT"),
        "dogecoin"          => Some("DOGEUSDT"),
        "cardano"           => Some("ADAUSDT"),
        "avalanche-2"       => Some("AVAXUSDT"),
        "chainlink"         => Some("LINKUSDT"),
        "polkadot"          => Some("DOTUSDT"),
        "near"              => Some("NEARUSDT"),
        "sui"               => Some("SUIUSDT"),
        "aptos"             => Some("APTUSDT"),
        "arbitrum"          => Some("ARBUSDT"),
        "the-open-network"  => Some("TONUSDT"),
        "uniswap"           => Some("UNIUSDT"),
        "tron"              => Some("TRXUSDT"),
        "litecoin"          => Some("LTCUSDT"),
        "shiba-inu"         => Some("SHIBUSDT"),
        "stellar"           => Some("XLMUSDT"),
        "matic-network"     => Some("MATICUSDT"),
        _ => None,
    }
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

/// Fetch historical daily OHLCV from Binance public API (no key required).
/// Converts CoinGecko coin_id to Binance symbol, fetches klines, and
/// returns data in the same MarketChart format the rest of the codebase expects.
pub async fn fetch_history(
    client: &reqwest::Client,
    coin_id: &str,
    days: u32,
) -> Result<MarketChart, Box<dyn std::error::Error>> {
    let binance_symbol = coingecko_to_binance(coin_id)
        .ok_or_else(|| format!("No Binance mapping for CoinGecko id '{}'", coin_id))?;

    let limit = days.min(1000); // Binance max per request is 1000

    let url = format!(
        "https://api.binance.com/api/v3/klines?symbol={}&interval=1d&limit={}",
        binance_symbol, limit
    );

    let response = client
        .get(&url)
        .header("User-Agent", "RustInvest/0.1")
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Binance returned HTTP {}: {}", status, body).into());
    }

    // Binance klines: each element is an array of mixed types
    // [open_time, open, high, low, close, volume, close_time, ...]
    let body = response.text().await?;
    let klines: Vec<Vec<serde_json::Value>> = serde_json::from_str(&body)?;

    let mut prices = Vec::with_capacity(klines.len());
    let mut total_volumes = Vec::with_capacity(klines.len());

    for kline in &klines {
        if kline.len() < 6 { continue; }

        let open_time_ms = kline[0].as_f64().unwrap_or(0.0);
        let close_price: f64 = kline[4].as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| kline[4].as_f64())
            .unwrap_or(0.0);
        let volume: f64 = kline[5].as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| kline[5].as_f64())
            .unwrap_or(0.0);

        prices.push([open_time_ms, close_price]);
        total_volumes.push([open_time_ms, volume]);
    }

    if prices.is_empty() {
        return Err(format!("Binance returned 0 klines for {}", binance_symbol).into());
    }

    println!("    [Binance] {} klines for {} ({})", prices.len(), coin_id, binance_symbol);

    Ok(MarketChart { prices, total_volumes })
}