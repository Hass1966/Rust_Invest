use serde::Deserialize;

pub struct StockInfo {
    pub symbol: &'static str,
    pub name: &'static str,
}

pub const STOCK_LIST: &[StockInfo] = &[
    StockInfo { symbol: "SPY", name: "S&P 500 ETF" },
    StockInfo { symbol: "QQQ", name: "Nasdaq 100 ETF" },
    StockInfo { symbol: "DIA", name: "Dow Jones ETF" },
    StockInfo { symbol: "AAPL", name: "Apple" },
    StockInfo { symbol: "MSFT", name: "Microsoft" },
    StockInfo { symbol: "GOOGL", name: "Google" },
    StockInfo { symbol: "AMZN", name: "Amazon" },
    StockInfo { symbol: "NVDA", name: "Nvidia" },
    StockInfo { symbol: "META", name: "Meta" },
    StockInfo { symbol: "TSLA", name: "Tesla" },
];

/// FX currency pairs — fetched from Yahoo Finance (ticker format: EURUSD=X)
pub const FX_LIST: &[StockInfo] = &[
    StockInfo { symbol: "EURUSD=X", name: "EUR/USD" },
    StockInfo { symbol: "GBPUSD=X", name: "GBP/USD" },
    StockInfo { symbol: "JPY=X",    name: "USD/JPY" },
    StockInfo { symbol: "AUDUSD=X", name: "AUD/USD" },
    StockInfo { symbol: "CHF=X",    name: "USD/CHF" },
];

// ── Yahoo Finance response types ──

#[derive(Debug, Deserialize)]
pub struct YahooResponse {
    pub chart: ChartData,
}

#[derive(Debug, Deserialize)]
pub struct ChartData {
    pub result: Option<Vec<ChartResult>>,
    pub error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChartResult {
    pub meta: MetaData,
    pub timestamp: Option<Vec<i64>>,
    pub indicators: Indicators,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetaData {
    pub symbol: String,
    pub regular_market_price: Option<f64>,
    pub previous_close: Option<f64>,
    pub regular_market_day_high: Option<f64>,
    pub regular_market_day_low: Option<f64>,
    pub regular_market_volume: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct Indicators {
    pub quote: Vec<QuoteData>,
}

#[derive(Debug, Deserialize)]
pub struct QuoteData {
    pub open: Option<Vec<Option<f64>>>,
    pub high: Option<Vec<Option<f64>>>,
    pub low: Option<Vec<Option<f64>>>,
    pub close: Option<Vec<Option<f64>>>,
    pub volume: Option<Vec<Option<u64>>>,
}

// ── Simplified quote for display ──

pub struct StockQuoteResult {
    pub symbol: String,
    pub price: f64,
    pub change: f64,
    pub change_percent: f64,
    pub high: f64,
    pub low: f64,
    pub volume: u64,
}

// ── API functions ──

pub async fn fetch_quote(
    client: &reqwest::Client,
    symbol: &str,
) -> Result<StockQuoteResult, Box<dyn std::error::Error>> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range=1d",
        symbol
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")
        .send()
        .await?;

    let text = resp.text().await?;
    let data: YahooResponse = serde_json::from_str(&text)?;

    let result = data.chart.result
        .ok_or("No chart result")?
        .into_iter()
        .next()
        .ok_or("Empty result")?;

    let meta = &result.meta;
    let price = meta.regular_market_price.unwrap_or(0.0);
    let prev_close = meta.previous_close.unwrap_or(price);
    let change = price - prev_close;
    let change_pct = if prev_close != 0.0 {
        (change / prev_close) * 100.0
    } else {
        0.0
    };

    Ok(StockQuoteResult {
        symbol: meta.symbol.clone(),
        price,
        change,
        change_percent: change_pct,
        high: meta.regular_market_day_high.unwrap_or(0.0),
        low: meta.regular_market_day_low.unwrap_or(0.0),
        volume: meta.regular_market_volume.unwrap_or(0),
    })
}

pub async fn fetch_history(
    client: &reqwest::Client,
    symbol: &str,
    range: &str,  // "1y", "6mo", "3mo", etc.
) -> Result<Vec<(i64, f64, Option<u64>)>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range={}",
        symbol, range
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")
        .send()
        .await?;

    let text = resp.text().await?;
    let data: YahooResponse = serde_json::from_str(&text)?;

    let result = data.chart.result
        .ok_or("No chart result")?
        .into_iter()
        .next()
        .ok_or("Empty result")?;

    let timestamps = result.timestamp.unwrap_or_default();
    let closes = result.indicators.quote.first()
        .and_then(|q| q.close.as_ref())
        .cloned()
        .unwrap_or_default();
    let volumes = result.indicators.quote.first()
        .and_then(|q| q.volume.as_ref())
        .cloned()
        .unwrap_or_default();

    let mut points = Vec::new();
    for (i, ts) in timestamps.iter().enumerate() {
        if let Some(Some(close)) = closes.get(i) {
            let vol = volumes.get(i).and_then(|v| *v);
            points.push((*ts, *close, vol));
        }
    }

    Ok(points)
}