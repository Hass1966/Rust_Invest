use crate::models::{StockQuote, GlobalQuote};
use tokio::time::{sleep, Duration};

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

pub async fn fetch_quote(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Result<GlobalQuote, Box<dyn std::error::Error>> {
    let url = format!(
        "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={}&apikey={}",
        symbol, api_key
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "RustInvest/0.1")
        .send()
        .await?;

    let text = resp.text().await?;
    let quote: StockQuote = serde_json::from_str(&text)?;
    Ok(quote.global_quote)
}

pub async fn fetch_all_quotes(
    client: &reqwest::Client,
    api_key: &str,
) -> Vec<(String, String, Option<GlobalQuote>)> {
    let mut results = Vec::new();

    for stock in STOCK_LIST {
        sleep(Duration::from_secs(15)).await;

        let quote = fetch_quote(client, stock.symbol, api_key).await.ok();
        results.push((
            stock.symbol.to_string(),
            stock.name.to_string(),
            quote,
        ));
    }

    results
}