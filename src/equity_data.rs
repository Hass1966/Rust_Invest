/// Equity Data Fetchers — Earnings Calendar Proximity
/// ====================================================
/// Fetches earnings calendar data from Alpha Vantage to compute
/// days-to-next-earnings and days-since-last-earnings features.
///
/// These features capture the well-known earnings drift effect:
///   - Stocks tend to drift up before earnings (anticipation)
///   - Large moves happen at earnings (event risk)
///   - Post-earnings drift persists for days
///
/// Source: Alpha Vantage EARNINGS_CALENDAR endpoint

use reqwest;
use chrono::NaiveDate;

/// Earnings dates for a stock symbol
#[derive(Debug, Clone)]
pub struct EarningsCalendar {
    pub symbol: String,
    /// Past and future earnings dates sorted ascending
    pub dates: Vec<NaiveDate>,
}

/// Fetch upcoming and recent earnings dates from Alpha Vantage
pub async fn fetch_earnings_calendar(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
) -> Result<EarningsCalendar, Box<dyn std::error::Error>> {
    // Alpha Vantage earnings endpoint returns CSV
    let url = format!(
        "https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={}&horizon=12month&apikey={}",
        symbol, api_key
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "RustInvest/1.0")
        .send()
        .await?;

    let text = resp.text().await?;
    let mut dates = Vec::new();

    // Parse CSV: first line is header, subsequent lines have reportDate
    for (i, line) in text.lines().enumerate() {
        if i == 0 { continue; } // skip header
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 3 {
            // reportDate is typically the 3rd field
            if let Ok(date) = NaiveDate::parse_from_str(fields[2], "%Y-%m-%d") {
                dates.push(date);
            } else if let Ok(date) = NaiveDate::parse_from_str(fields[0], "%Y-%m-%d") {
                dates.push(date);
            }
        }
    }

    dates.sort();
    dates.dedup();

    println!("  [EARNINGS] Fetched {} earnings dates for {}", dates.len(), symbol);

    Ok(EarningsCalendar {
        symbol: symbol.to_string(),
        dates,
    })
}

/// Compute days-to-next-earnings and days-since-last-earnings for a given date.
/// Returns (days_to_next, days_since_last) — both as positive f64.
/// If no data available, returns (90.0, 90.0) as neutral values.
pub fn earnings_proximity(
    calendar: &EarningsCalendar,
    date: &NaiveDate,
) -> (f64, f64) {
    if calendar.dates.is_empty() {
        return (90.0, 90.0);
    }

    // Find the nearest future earnings date
    let days_to_next = calendar.dates.iter()
        .filter(|d| *d >= date)
        .next()
        .map(|d| (*d - *date).num_days() as f64)
        .unwrap_or(90.0);

    // Find the most recent past earnings date
    let days_since_last = calendar.dates.iter()
        .rev()
        .filter(|d| *d <= date)
        .next()
        .map(|d| (*date - *d).num_days() as f64)
        .unwrap_or(90.0);

    (days_to_next, days_since_last)
}

/// Batch-fetch earnings calendars for multiple stock symbols.
/// Rate-limited: Alpha Vantage allows 5 calls/minute on free tier.
pub async fn fetch_all_earnings(
    client: &reqwest::Client,
    symbols: &[&str],
    api_key: &str,
) -> Vec<EarningsCalendar> {
    let mut calendars = Vec::new();

    for (i, symbol) in symbols.iter().enumerate() {
        match fetch_earnings_calendar(client, symbol, api_key).await {
            Ok(cal) => calendars.push(cal),
            Err(e) => {
                println!("  [EARNINGS] Failed for {}: {}", symbol, e);
                calendars.push(EarningsCalendar {
                    symbol: symbol.to_string(),
                    dates: Vec::new(),
                });
            }
        }

        // Rate limit: Alpha Vantage free tier = 5 calls/min
        if i < symbols.len() - 1 && (i + 1) % 4 == 0 {
            println!("  [EARNINGS] Rate limit pause (Alpha Vantage free tier)...");
            tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
        }
    }

    calendars
}
