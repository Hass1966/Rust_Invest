/// Polygon.io — Primary price data source
/// ========================================
/// Replaces Yahoo Finance as primary data source for US stocks and FX.
/// Falls back to Yahoo Finance if Polygon rate limits are hit.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PolygonAggResponse {
    status: Option<String>,
    #[serde(rename = "resultsCount")]
    results_count: Option<usize>,
    results: Option<Vec<PolygonBar>>,
}

#[derive(Debug, Deserialize)]
struct PolygonBar {
    t: i64,   // timestamp (ms)
    c: f64,   // close
    v: Option<f64>,  // volume
}

#[derive(Debug, Deserialize)]
struct PolygonSnapshotResponse {
    status: Option<String>,
    ticker: Option<PolygonTickerSnapshot>,
}

#[derive(Debug, Deserialize)]
struct PolygonTickerSnapshot {
    day: Option<PolygonDayData>,
    #[serde(rename = "prevDay")]
    prev_day: Option<PolygonDayData>,
}

#[derive(Debug, Deserialize)]
struct PolygonDayData {
    c: Option<f64>,
    h: Option<f64>,
    l: Option<f64>,
    v: Option<f64>,
}

/// Fetch daily OHLCV history from Polygon.io
/// Returns Vec<(unix_timestamp_secs, close_price, volume_opt)> matching Yahoo format
pub async fn fetch_history(
    client: &reqwest::Client,
    symbol: &str,
    api_key: &str,
    from_date: &str,
    to_date: &str,
) -> Result<Vec<(i64, f64, Option<u64>)>, Box<dyn std::error::Error>> {
    // Polygon uses plain tickers for stocks, C: prefix for FX
    let poly_symbol = if symbol.ends_with("=X") {
        // Convert EURUSD=X -> C:EURUSD
        let pair = symbol.trim_end_matches("=X");
        format!("C:{}", pair)
    } else if symbol.ends_with(".L") {
        // UK stocks not supported on Polygon — caller should fall back to Yahoo
        return Err("UK stocks (.L suffix) not supported on Polygon".into());
    } else {
        symbol.to_string()
    };

    let url = format!(
        "https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?adjusted=true&sort=asc&limit=50000&apiKey={}",
        poly_symbol, from_date, to_date, api_key
    );

    let resp = client.get(&url).send().await?;
    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status == reqwest::StatusCode::FORBIDDEN {
        return Err(format!("Polygon rate limit hit (HTTP {})", status).into());
    }

    let text = resp.text().await?;
    let data: PolygonAggResponse = serde_json::from_str(&text)
        .map_err(|e| format!("Polygon parse error: {} — body: {}", e, &text[..200.min(text.len())]))?;

    if data.status.as_deref() == Some("ERROR") {
        return Err(format!("Polygon API error for {}", symbol).into());
    }

    let bars = data.results.unwrap_or_default();
    let mut points = Vec::with_capacity(bars.len());
    for bar in &bars {
        let ts_secs = bar.t / 1000; // Polygon returns ms timestamps
        let vol = bar.v.map(|v| v as u64);
        points.push((ts_secs, bar.c, vol));
    }

    Ok(points)
}

/// Fetch history with automatic Yahoo Finance fallback
pub async fn fetch_history_with_fallback(
    client: &reqwest::Client,
    symbol: &str,
    polygon_key: Option<&str>,
    yahoo_range: &str,
) -> Result<Vec<(i64, f64, Option<u64>)>, Box<dyn std::error::Error>> {
    // Try Polygon first if key available
    if let Some(api_key) = polygon_key {
        if !api_key.is_empty() {
            let to_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
            let from_date = match yahoo_range {
                "7y" => (chrono::Utc::now() - chrono::Duration::days(2555)).format("%Y-%m-%d").to_string(),
                "5y" => (chrono::Utc::now() - chrono::Duration::days(1825)).format("%Y-%m-%d").to_string(),
                "2y" => (chrono::Utc::now() - chrono::Duration::days(730)).format("%Y-%m-%d").to_string(),
                "1y" => (chrono::Utc::now() - chrono::Duration::days(365)).format("%Y-%m-%d").to_string(),
                _ => (chrono::Utc::now() - chrono::Duration::days(2555)).format("%Y-%m-%d").to_string(),
            };

            match fetch_history(client, symbol, api_key, &from_date, &to_date).await {
                Ok(points) if !points.is_empty() => {
                    println!("    [Polygon] {} data points for {}", points.len(), symbol);
                    return Ok(points);
                }
                Ok(_) => {
                    println!("    [Polygon] Empty response for {}, falling back to Yahoo", symbol);
                }
                Err(e) => {
                    println!("    [Polygon] {} — falling back to Yahoo", e);
                }
            }
        }
    }

    // Fallback to Yahoo Finance
    let points = crate::stocks::fetch_history(client, symbol, yahoo_range).await?;
    println!("    [Yahoo] {} data points for {}", points.len(), symbol);
    Ok(points)
}
