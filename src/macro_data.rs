/// Macro Data Fetchers — VIX, DXY, Treasury Yields, FRED Series
/// ==============================================================
/// Fetches macroeconomic indicators used as features for ALL assets.
///
/// Sources:
///   - Yahoo Finance: ^VIX (volatility), DX-Y.NYB (US dollar), ^TNX (10Y yield)
///   - FRED API: T10Y2Y (10Y-2Y spread), FEDFUNDS (fed funds rate)
///
/// All data points become features in the feature vector via features.rs.

use reqwest;
use serde::Deserialize;

// ════════════════════════════════════════
// Yahoo Finance tickers for macro data
// ════════════════════════════════════════

/// Fetch daily close for a Yahoo Finance ticker (e.g., ^VIX, DX-Y.NYB, ^TNX)
/// Returns Vec<(timestamp_secs, close_price)> sorted ascending.
pub async fn fetch_yahoo_daily(
    client: &reqwest::Client,
    ticker: &str,
    range: &str,
) -> Result<Vec<(i64, f64)>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range={}",
        ticker, range
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;
    let chart = &body["chart"]["result"][0];
    let timestamps = chart["timestamp"]
        .as_array()
        .ok_or("no timestamps")?;
    let closes = chart["indicators"]["quote"][0]["close"]
        .as_array()
        .ok_or("no closes")?;

    let mut data = Vec::new();
    for (ts, close) in timestamps.iter().zip(closes.iter()) {
        if let (Some(t), Some(c)) = (ts.as_i64(), close.as_f64()) {
            data.push((t, c));
        }
    }

    println!("  [MACRO] Fetched {} data points for {}", data.len(), ticker);
    Ok(data)
}

/// Fetch VIX daily close (fear index)
pub async fn fetch_vix(client: &reqwest::Client, range: &str) -> Result<Vec<(i64, f64)>, Box<dyn std::error::Error>> {
    fetch_yahoo_daily(client, "%5EVIX", range).await
}

/// Fetch DXY (US Dollar Index) daily close
pub async fn fetch_dxy(client: &reqwest::Client, range: &str) -> Result<Vec<(i64, f64)>, Box<dyn std::error::Error>> {
    fetch_yahoo_daily(client, "DX-Y.NYB", range).await
}

/// Fetch 10-Year Treasury yield daily close
pub async fn fetch_tnx(client: &reqwest::Client, range: &str) -> Result<Vec<(i64, f64)>, Box<dyn std::error::Error>> {
    fetch_yahoo_daily(client, "%5ETNX", range).await
}

// ════════════════════════════════════════
// FRED API — Federal Reserve Economic Data
// ════════════════════════════════════════

#[derive(Deserialize, Debug)]
struct FredResponse {
    observations: Vec<FredObservation>,
}

#[derive(Deserialize, Debug)]
struct FredObservation {
    date: String,
    value: String,
}

/// Fetch a FRED series by ID (e.g., T10Y2Y, FEDFUNDS).
/// Returns Vec<(date_string, value)> sorted ascending.
/// Requires FRED_API_KEY env var.
pub async fn fetch_fred_series(
    client: &reqwest::Client,
    series_id: &str,
    api_key: &str,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json&sort_order=asc&observation_start=2019-01-01",
        series_id, api_key
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "RustInvest/1.0")
        .send()
        .await?;

    let body: FredResponse = resp.json().await?;

    let mut data = Vec::new();
    for obs in &body.observations {
        if obs.value != "." {
            if let Ok(val) = obs.value.parse::<f64>() {
                data.push((obs.date.clone(), val));
            }
        }
    }

    println!("  [FRED] Fetched {} observations for {}", data.len(), series_id);
    Ok(data)
}

/// Fetch 10Y-2Y yield spread (T10Y2Y series)
/// Positive = normal yield curve, negative = inverted (recession signal)
pub async fn fetch_yield_spread(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    fetch_fred_series(client, "T10Y2Y", api_key).await
}

/// Fetch Fed Funds rate (FEDFUNDS series)
pub async fn fetch_fed_funds_rate(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    fetch_fred_series(client, "FEDFUNDS", api_key).await
}

/// Merged macro context with all macro indicators aligned by date
#[derive(Debug, Clone, Default)]
pub struct MacroIndicators {
    /// VIX daily close values
    pub vix: Vec<f64>,
    /// DXY (US Dollar Index) daily close
    pub dxy: Vec<f64>,
    /// 10-Year Treasury yield
    pub tnx: Vec<f64>,
    /// 10Y-2Y yield spread (from FRED)
    pub yield_spread: Vec<f64>,
    /// Fed funds rate (from FRED, monthly → forward-filled to daily)
    pub fed_funds: Vec<f64>,
}

/// Fetch all macro indicators. Falls back to empty vecs on failure.
pub async fn fetch_all_macro(
    client: &reqwest::Client,
    fred_api_key: Option<&str>,
    range: &str,
) -> MacroIndicators {
    let mut result = MacroIndicators::default();

    // Yahoo Finance tickers (parallel)
    let (vix_res, dxy_res, tnx_res) = tokio::join!(
        fetch_vix(client, range),
        fetch_dxy(client, range),
        fetch_tnx(client, range),
    );

    if let Ok(data) = vix_res {
        result.vix = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [MACRO] VIX fetch failed, using empty");
    }

    if let Ok(data) = dxy_res {
        result.dxy = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [MACRO] DXY fetch failed, using empty");
    }

    if let Ok(data) = tnx_res {
        result.tnx = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [MACRO] TNX fetch failed, using empty");
    }

    // FRED series (requires API key)
    if let Some(key) = fred_api_key {
        let (spread_res, ffr_res) = tokio::join!(
            fetch_yield_spread(client, key),
            fetch_fed_funds_rate(client, key),
        );

        if let Ok(data) = spread_res {
            result.yield_spread = data.iter().map(|(_, v)| *v).collect();
        } else {
            println!("  [FRED] Yield spread fetch failed");
        }

        if let Ok(data) = ffr_res {
            // Fed funds is monthly — forward-fill to daily granularity
            result.fed_funds = data.iter().map(|(_, v)| *v).collect();
        } else {
            println!("  [FRED] Fed funds rate fetch failed");
        }
    }

    result
}
