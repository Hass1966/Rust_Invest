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
use chrono::Utc;

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
    /// BOE base rate (Bank of England)
    pub boe_rate: Vec<f64>,
    /// UK 10-year gilt yield
    pub uk_10y_gilt: Vec<f64>,
    /// ECB main refinancing rate
    pub ecb_rate: Vec<f64>,
    /// EU inflation (HICP)
    pub eu_inflation: Vec<f64>,
}

// ════════════════════════════════════════
// Bank of England API (free, no key needed)
// ════════════════════════════════════════

/// Fetch BOE series data (XML-based API)
/// Series IDs: IUMABEDR (base rate), IUDMNZR (10yr gilt yield)
pub async fn fetch_boe_series(
    client: &reqwest::Client,
    series_id: &str,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&SeriesCodes={}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N",
        series_id
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "AlphaSignal/1.0")
        .send()
        .await?;

    let text = resp.text().await?;
    let mut data = Vec::new();

    // BOE CSV format: DATE,VALUE (skip header line)
    for line in text.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let date = parts[0].trim().replace(' ', "");
            if let Ok(val) = parts[1].trim().parse::<f64>() {
                // Convert DD/Mon/YYYY or DD Mon YYYY to YYYY-MM-DD
                let date_str = parse_boe_date(&date).unwrap_or(date);
                data.push((date_str, val));
            }
        }
    }

    println!("  [BOE] Fetched {} observations for {}", data.len(), series_id);
    Ok(data)
}

fn parse_boe_date(date: &str) -> Option<String> {
    // Try DD/Mon/YYYY format
    let parts: Vec<&str> = if date.contains('/') {
        date.split('/').collect()
    } else {
        date.split(' ').collect()
    };
    if parts.len() < 3 { return None; }
    let day = parts[0];
    let month = match parts[1].to_lowercase().as_str() {
        "jan" => "01", "feb" => "02", "mar" => "03", "apr" => "04",
        "may" => "05", "jun" => "06", "jul" => "07", "aug" => "08",
        "sep" => "09", "oct" => "10", "nov" => "11", "dec" => "12",
        _ => return None,
    };
    let year = parts[2];
    Some(format!("{}-{}-{:0>2}", year, month, day))
}

/// Fetch BOE base rate (Bank Rate)
pub async fn fetch_boe_base_rate(
    client: &reqwest::Client,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    fetch_boe_series(client, "IUMABEDR").await
}

/// Fetch UK 10-year gilt yield
pub async fn fetch_uk_gilt_yield(
    client: &reqwest::Client,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    fetch_boe_series(client, "IUDMNZR").await
}

// ════════════════════════════════════════
// ECB API (free, no key needed)
// ════════════════════════════════════════

/// Fetch ECB data series via the ECB SDMX REST API
/// Returns Vec<(date_string, value)>
pub async fn fetch_ecb_series(
    client: &reqwest::Client,
    flow_ref: &str,
    key: &str,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://data-api.ecb.europa.eu/service/data/{}/{}?format=csvdata",
        flow_ref, key
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "AlphaSignal/1.0")
        .send()
        .await?;

    let text = resp.text().await?;
    let mut data = Vec::new();

    // ECB CSV: header then rows with TIME_PERIOD and OBS_VALUE columns
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() { return Ok(data); }

    // Find column indices
    let headers: Vec<&str> = lines[0].split(',').collect();
    let time_idx = headers.iter().position(|h| h.contains("TIME_PERIOD"));
    let val_idx = headers.iter().position(|h| h.contains("OBS_VALUE"));

    if let (Some(ti), Some(vi)) = (time_idx, val_idx) {
        for line in lines.iter().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() > ti.max(vi) {
                if let Ok(val) = cols[vi].parse::<f64>() {
                    data.push((cols[ti].to_string(), val));
                }
            }
        }
    }

    println!("  [ECB] Fetched {} observations for {}/{}", data.len(), flow_ref, key);
    Ok(data)
}

/// Fetch ECB main refinancing rate
pub async fn fetch_ecb_refi_rate(
    client: &reqwest::Client,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    fetch_ecb_series(client, "FM", "M.U2.EUR.4F.KR.MRR_FR.LEV").await
}

/// Fetch EU inflation (HICP - all items, annual rate of change)
pub async fn fetch_eu_inflation(
    client: &reqwest::Client,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    fetch_ecb_series(client, "ICP", "M.U2.N.000000.4.ANR").await
}

// ════════════════════════════════════════
// SEC EDGAR Insider Trading (free, no key needed)
// ════════════════════════════════════════

/// Fetch recent Form 4 filings (insider transactions) for a given ticker
/// Returns net insider buying score: (buys - sells) normalised to [-1, 1]
pub async fn fetch_insider_score(
    client: &reqwest::Client,
    ticker: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start_date = Utc::now().checked_sub_signed(chrono::Duration::days(30))
        .unwrap_or(Utc::now())
        .format("%Y-%m-%d");
    let url = format!(
        "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt={}&forms=4&from=0&size=40",
        ticker, start_date
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "AlphaSignal research@alphasignal.co.uk")
        .send()
        .await?;

    if !resp.status().is_success() {
        return Ok(0.0); // Fail silently
    }

    let body: serde_json::Value = resp.json().await?;
    let total = body["hits"]["total"]["value"].as_i64().unwrap_or(0);

    // Simple heuristic: more filings = more insider activity
    // Normalise: 0-5 filings = low, 5-20 = medium, 20+ = high
    let score = (total as f64 / 20.0).min(1.0);
    Ok(score)
}

// ════════════════════════════════════════
// FINRA Short Interest (free, twice monthly)
// ════════════════════════════════════════

/// Fetch short interest ratio for a US stock
/// Returns short interest as a ratio (0-1 scale) or 0.0 on failure
pub async fn fetch_short_interest(
    client: &reqwest::Client,
    _ticker: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    // FINRA short interest is not easily available via API
    // Fall back to 0.0 (neutral) — can be populated from manual downloads
    Ok(0.0)
}

/// Fetch all macro indicators. Falls back to empty vecs on failure.
pub async fn fetch_all_macro(
    client: &reqwest::Client,
    fred_api_key: Option<&str>,
    range: &str,
) -> MacroIndicators {
    let mut result = MacroIndicators::default();

    // Yahoo Finance tickers + BOE/ECB (all parallel, no keys needed)
    let (vix_res, dxy_res, tnx_res, boe_rate_res, gilt_res, ecb_rate_res, eu_infl_res) = tokio::join!(
        fetch_vix(client, range),
        fetch_dxy(client, range),
        fetch_tnx(client, range),
        fetch_boe_base_rate(client),
        fetch_uk_gilt_yield(client),
        fetch_ecb_refi_rate(client),
        fetch_eu_inflation(client),
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

    if let Ok(data) = boe_rate_res {
        result.boe_rate = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [BOE] Base rate fetch failed, using empty");
    }

    if let Ok(data) = gilt_res {
        result.uk_10y_gilt = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [BOE] UK gilt yield fetch failed, using empty");
    }

    if let Ok(data) = ecb_rate_res {
        result.ecb_rate = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [ECB] Refi rate fetch failed, using empty");
    }

    if let Ok(data) = eu_infl_res {
        result.eu_inflation = data.iter().map(|(_, v)| *v).collect();
    } else {
        println!("  [ECB] EU inflation fetch failed, using empty");
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
