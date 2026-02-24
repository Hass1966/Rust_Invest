use serde::Deserialize;

// ── Crypto (CoinGecko) ──

#[derive(Debug, Deserialize, Clone)]
pub struct CoinData {
    pub id: String,
    pub symbol: String,
    pub name: String,
    pub current_price: f64,
    pub price_change_percentage_24h: Option<f64>,
    pub market_cap_rank: Option<u32>,
    pub market_cap: Option<f64>,
    pub total_volume: Option<f64>,
    pub high_24h: Option<f64>,
    pub low_24h: Option<f64>,
}

// ── Stocks (Alpha Vantage) ──

#[derive(Debug, Deserialize, Clone)]
pub struct StockQuote {
    #[serde(rename = "Global Quote")]
    pub global_quote: GlobalQuote,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GlobalQuote {
    #[serde(rename = "01. symbol")]
    pub symbol: String,
    #[serde(rename = "02. open")]
    pub open: String,
    #[serde(rename = "03. high")]
    pub high: String,
    #[serde(rename = "04. low")]
    pub low: String,
    #[serde(rename = "05. price")]
    pub price: String,
    #[serde(rename = "06. volume")]
    pub volume: String,
    #[serde(rename = "08. previous close")]
    pub previous_close: String,
    #[serde(rename = "09. change")]
    pub change: String,
    #[serde(rename = "10. change percent")]
    pub change_percent: String,
}