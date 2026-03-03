/// Crypto-Specific Feature Engineering
/// ====================================
/// Adds features predictive for cryptocurrency markets.
///
/// Stock models have 83 features (VIX, sector ETFs, treasuries).
/// Crypto models fail with only 14 generic technical indicators.
/// This module adds crypto-native signals:
///
/// 1. Fear & Greed Index (sentiment)
/// 2. Bitcoin Dominance (market structure)  
/// 3. Funding Rates (derivatives positioning)
/// 4. Cross-crypto correlations
/// 5. Relative strength vs BTC
/// 6. Crypto volatility regime ("crypto VIX")

use std::collections::HashMap;

/// All crypto-specific features for one day
#[derive(Debug, Clone)]
pub struct CryptoFeatureRow {
    pub date: String,
    pub fear_greed_value: f64,
    pub fear_greed_class: f64,
    pub fear_greed_7d_avg: f64,
    pub fear_greed_momentum: f64,
    pub btc_dominance: f64,
    pub btc_dominance_change_7d: f64,
    pub total_market_cap_change_pct: f64,
    pub funding_rate: f64,
    pub funding_rate_7d_avg: f64,
    pub funding_rate_extreme: f64,
    pub btc_return_lag1: f64,
    pub btc_eth_corr_30d: f64,
    pub altcoin_season_score: f64,
    pub crypto_vix_proxy: f64,
    pub asset_vs_btc_ratio_20d: f64,
    pub asset_vs_btc_momentum: f64,
}

impl CryptoFeatureRow {
    /// Return features as named vector for ML pipeline
    pub fn to_feature_vec(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("fear_greed_value", self.fear_greed_value),
            ("fear_greed_class", self.fear_greed_class),
            ("fear_greed_7d_avg", self.fear_greed_7d_avg),
            ("fear_greed_momentum", self.fear_greed_momentum),
            ("btc_dominance", self.btc_dominance),
            ("btc_dominance_change_7d", self.btc_dominance_change_7d),
            ("total_mcap_change_pct", self.total_market_cap_change_pct),
            ("funding_rate", self.funding_rate),
            ("funding_rate_7d_avg", self.funding_rate_7d_avg),
            ("funding_rate_extreme", self.funding_rate_extreme),
            ("btc_return_lag1", self.btc_return_lag1),
            ("btc_eth_corr_30d", self.btc_eth_corr_30d),
            ("altcoin_season_score", self.altcoin_season_score),
            ("crypto_vix_proxy", self.crypto_vix_proxy),
            ("asset_vs_btc_ratio_20d", self.asset_vs_btc_ratio_20d),
            ("asset_vs_btc_momentum", self.asset_vs_btc_momentum),
        ]
    }

    pub fn feature_count() -> usize { 16 }

    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "fear_greed_value", "fear_greed_class", "fear_greed_7d_avg",
            "fear_greed_momentum", "btc_dominance", "btc_dominance_change_7d",
            "total_mcap_change_pct", "funding_rate", "funding_rate_7d_avg",
            "funding_rate_extreme", "btc_return_lag1", "btc_eth_corr_30d",
            "altcoin_season_score", "crypto_vix_proxy",
            "asset_vs_btc_ratio_20d", "asset_vs_btc_momentum",
        ]
    }
}

// ============================================================================
// Fear & Greed Index — alternative.me API (free, no key)
// ============================================================================

pub fn fetch_fear_greed_history(days: usize) -> Result<Vec<(String, f64, String)>, Box<dyn std::error::Error>> {
    let url = format!("https://api.alternative.me/fng/?limit={}&format=json", days);
    let response = reqwest::blocking::get(&url)?;
    let json: serde_json::Value = response.json()?;
    let mut history = Vec::new();
    if let Some(data) = json.get("data") {
        if let Some(arr) = data.as_array() {
            for entry in arr {
                let value: f64 = entry.get("value")
                    .and_then(|v| v.as_str())
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(50.0);
                let classification: String = entry.get("value_classification")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Neutral")
                    .to_string();
                let timestamp: i64 = entry.get("timestamp")
                    .and_then(|v| v.as_str())
                    .and_then(|v| v.parse::<i64>().ok())
                    .unwrap_or(0);
                let date = timestamp_to_date(timestamp);
                history.push((date, value, classification));
            }
        }
    }
    history.reverse();
    println!("    [CryptoFeatures] Fetched {} days of Fear & Greed data", history.len());
    Ok(history)
}

fn encode_fear_greed_class(class: &str) -> f64 {
    match class {
        "Extreme Fear" => 0.0, "Fear" => 1.0, "Neutral" => 2.0,
        "Greed" => 3.0, "Extreme Greed" => 4.0, _ => 2.0,
    }
}

// ============================================================================
// Bitcoin Dominance — CoinGecko API (free, no key)
// ============================================================================

pub fn fetch_btc_dominance() -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let url = "https://api.coingecko.com/api/v3/global";
    let response = reqwest::blocking::get(url)?;
    let json: serde_json::Value = response.json()?;
    let data = json.get("data").ok_or("No data field")?;
    let btc_dominance: f64 = data.get("market_cap_percentage")
        .and_then(|m| m.as_object())
        .and_then(|m| m.get("btc"))
        .and_then(|v| v.as_f64())
        .unwrap_or(50.0);
    let mcap_change_24h: f64 = data.get("market_cap_change_percentage_24h_usd")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let total_volume: f64 = data.get("total_volume")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("usd"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let mcap_change_24h = data.get("market_cap_change_percentage_24h_usd")
        .and_then(|v| v.as_f64()).unwrap_or(0.0);
    let total_volume = data.get("total_volume")
        .and_then(|v| v.get("usd")).and_then(|v| v.as_f64()).unwrap_or(0.0);
    println!("    [CryptoFeatures] BTC dominance: {:.1}%, mcap change 24h: {:.2}%",
        btc_dominance, mcap_change_24h);
    Ok((btc_dominance, mcap_change_24h, total_volume))
}

// ============================================================================
// Binance Funding Rates — Binance Futures API (free, no key)
// ============================================================================

fn to_binance_perp_symbol(symbol: &str) -> Option<&'static str> {
    match symbol.to_uppercase().as_str() {
        "BTC" | "BITCOIN" => Some("BTCUSDT"),
        "ETH" | "ETHEREUM" => Some("ETHUSDT"),
        "XRP" | "RIPPLE" => Some("XRPUSDT"),
        "BNB" | "BINANCECOIN" => Some("BNBUSDT"),
        _ => None,
    }
}

pub fn fetch_funding_rates(symbol: &str, limit: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let perp_symbol = to_binance_perp_symbol(symbol)
        .ok_or_else(|| format!("No Binance perp for {}", symbol))?;
    let url = format!("https://fapi.binance.com/fapi/v1/fundingRate?symbol={}&limit={}", perp_symbol, limit);
    let response = reqwest::blocking::get(&url)?;
    let json: serde_json::Value = response.json()?;
    let mut rates = Vec::new();
    if let Some(data) = json.as_array() {
        for entry in data {
            let rate: f64 = entry.get("fundingRate")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            rates.push(rate);
        }
    }
    println!("    [CryptoFeatures] Fetched {} funding rates for {}", rates.len(), perp_symbol);
    Ok(rates)
}

// ============================================================================
// Cross-Crypto Computed Features
// ============================================================================

pub fn rolling_correlation(series_a: &[f64], series_b: &[f64], window: usize) -> Vec<f64> {
    let n = series_a.len().min(series_b.len());
    let mut correlations = Vec::with_capacity(n);
    for i in 0..n {
        if i < window - 1 { correlations.push(0.0); continue; }
        let a_w = &series_a[i + 1 - window..=i];
        let b_w = &series_b[i + 1 - window..=i];
        let mean_a: f64 = a_w.iter().sum::<f64>() / window as f64;
        let mean_b: f64 = b_w.iter().sum::<f64>() / window as f64;
        let (mut cov, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
        for j in 0..window {
            let (da, db) = (a_w[j] - mean_a, b_w[j] - mean_b);
            cov += da * db; var_a += da * da; var_b += db * db;
        }
        let denom = (var_a * var_b).sqrt();
        correlations.push(if denom > 1e-10 { cov / denom } else { 0.0 });
    }
    correlations
}

pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    let mut vols = Vec::with_capacity(returns.len());
    for i in 0..returns.len() {
        if i < window - 1 { vols.push(0.0); continue; }
        let w = &returns[i + 1 - window..=i];
        let mean: f64 = w.iter().sum::<f64>() / window as f64;
        let variance: f64 = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
        vols.push(variance.sqrt() * (365.0_f64).sqrt()); // annualised for crypto
    }
    vols
}

pub fn altcoin_season_score(btc_prices: &[f64], alt_prices_map: &HashMap<String, Vec<f64>>, window: usize) -> Vec<f64> {
    let n = btc_prices.len();
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        if i < window { scores.push(0.5); continue; }
        let btc_ret = (btc_prices[i] / btc_prices[i - window]) - 1.0;
        let mut beating = 0;
        let mut total = 0;
        for (_sym, prices) in alt_prices_map.iter() {
            if prices.len() > i && i >= window {
                let alt_ret = (prices[i] / prices[i - window]) - 1.0;
                if alt_ret > btc_ret { beating += 1; }
                total += 1;
            }
        }
        scores.push(if total > 0 { beating as f64 / total as f64 } else { 0.5 });
    }
    scores
}

pub fn relative_strength_vs_btc(asset_prices: &[f64], btc_prices: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = asset_prices.len().min(btc_prices.len());
    let mut ratios = Vec::with_capacity(n);
    let mut momentum = Vec::with_capacity(n);
    for i in 0..n {
        ratios.push(if btc_prices[i] > 0.0 { asset_prices[i] / btc_prices[i] } else { 1.0 });
    }
    for i in 0..n {
        if i < window { momentum.push(0.0); }
        else if ratios[i - window].abs() > 1e-10 {
            momentum.push((ratios[i] / ratios[i - window]) - 1.0);
        } else { momentum.push(0.0); }
    }
    // Z-score normalise ratios
    let mut normalised = Vec::with_capacity(n);
    for i in 0..n {
        if i < window { normalised.push(0.0); continue; }
        let w = &ratios[i + 1 - window..=i];
        let mean: f64 = w.iter().sum::<f64>() / window as f64;
        let std: f64 = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0)).sqrt();
        normalised.push(if std > 1e-10 { (ratios[i] - mean) / std } else { 0.0 });
    }
    (normalised, momentum)
}

// ============================================================================
// Main enrichment function
// ============================================================================

/// Enrich crypto features for all crypto assets.
/// Called from features.rs after basic technical features are computed.
pub fn enrich_crypto_features(
    crypto_symbols: &[&str],
    crypto_prices: &HashMap<String, Vec<f64>>,
    crypto_returns: &HashMap<String, Vec<f64>>,
    crypto_dates: &HashMap<String, Vec<String>>,
) -> HashMap<String, Vec<CryptoFeatureRow>> {
    println!("\n  === Crypto-Specific Feature Engineering ===");

    // 1. Fear & Greed (shared)
    let fear_greed = fetch_fear_greed_history(365).unwrap_or_else(|e| {
        eprintln!("    [CryptoFeatures] Warning: Fear & Greed fetch failed: {}", e);
        Vec::new()
    });
    let fg_by_date: HashMap<String, (f64, String)> = fear_greed.iter()
        .map(|(d, v, c)| (d.clone(), (*v, c.clone()))).collect();

    // 2. BTC dominance (shared)
    let (btc_dom, mcap_change, _) = fetch_btc_dominance().unwrap_or((50.0, 0.0, 0.0));

    // 3. Funding rates (per asset)
    let mut funding_map: HashMap<String, Vec<f64>> = HashMap::new();
    for sym in crypto_symbols {
        let rates = fetch_funding_rates(sym, 90).unwrap_or_default();
        funding_map.insert(sym.to_string(), rates);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // 4. Cross-crypto
    let btc_returns = crypto_returns.get("BTC").or_else(|| crypto_returns.get("bitcoin")).cloned().unwrap_or_default();
    let eth_returns = crypto_returns.get("ETH").or_else(|| crypto_returns.get("ethereum")).cloned().unwrap_or_default();
    let btc_prices = crypto_prices.get("BTC").or_else(|| crypto_prices.get("bitcoin")).cloned().unwrap_or_default();
    let btc_eth_corr = rolling_correlation(&btc_returns, &eth_returns, 30);
    let crypto_vix = rolling_volatility(&btc_returns, 30);
    let alt_prices: HashMap<String, Vec<f64>> = crypto_prices.iter()
        .filter(|(k, _)| { let kl = k.to_lowercase(); kl != "btc" && kl != "bitcoin" && kl != "usdt" && kl != "tether" })
        .map(|(k, v)| (k.clone(), v.clone())).collect();
    let alt_season = altcoin_season_score(&btc_prices, &alt_prices, 30);

    // 5. Build per-asset rows
    let mut result: HashMap<String, Vec<CryptoFeatureRow>> = HashMap::new();
    for sym in crypto_symbols {
        let dates = match crypto_dates.get(*sym) { Some(d) => d, None => continue };
        let prices = match crypto_prices.get(*sym) { Some(p) => p, None => continue };
        let n = dates.len();
        let (rs_ratio, rs_momentum) = relative_strength_vs_btc(prices, &btc_prices, 20);
        let funding = funding_map.get(*sym).cloned().unwrap_or_default();

        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let (fg_val, fg_cls) = fg_by_date.get(&dates[i]).cloned().unwrap_or((50.0, "Neutral".to_string()));
            let fg_class = encode_fear_greed_class(&fg_cls);
            let fg_7d = if i >= 7 {
                let mut s = 0.0; let mut c = 0;
                for j in i.saturating_sub(6)..=i { if let Some((v, _)) = fg_by_date.get(&dates[j]) { s += v; c += 1; } }
                if c > 0 { s / c as f64 } else { fg_val }
            } else { fg_val };
            let fg_mom = if i >= 7 { fg_val - fg_by_date.get(&dates[i-7]).map(|(v,_)| *v).unwrap_or(50.0) } else { 0.0 };
            let fr = if !funding.is_empty() { funding[i.min(funding.len()-1)] } else { 0.0 };
            let fr_7d = if !funding.is_empty() && i >= 7 {
                let s = i.saturating_sub(6).min(funding.len().saturating_sub(1));
                let e = (i+1).min(funding.len());
                let w = &funding[s..e]; w.iter().sum::<f64>() / w.len() as f64
            } else { fr };
            let btc_r1 = if i > 0 && btc_returns.len() > i-1 { btc_returns[i-1] } else { 0.0 };

            rows.push(CryptoFeatureRow {
                date: dates[i].clone(),
                fear_greed_value: fg_val / 100.0,
                fear_greed_class: fg_class / 4.0,
                fear_greed_7d_avg: fg_7d / 100.0,
                fear_greed_momentum: fg_mom / 100.0,
                btc_dominance: btc_dom / 100.0,
                btc_dominance_change_7d: 0.0,
                total_market_cap_change_pct: mcap_change / 100.0,
                funding_rate: fr,
                funding_rate_7d_avg: fr_7d,
                funding_rate_extreme: if fr.abs() > 0.01 { 1.0 } else { 0.0 },
                btc_return_lag1: btc_r1,
                btc_eth_corr_30d: if btc_eth_corr.len() > i { btc_eth_corr[i] } else { 0.0 },
                altcoin_season_score: if alt_season.len() > i { alt_season[i] } else { 0.5 },
                crypto_vix_proxy: if crypto_vix.len() > i { crypto_vix[i] } else { 0.0 },
                asset_vs_btc_ratio_20d: if rs_ratio.len() > i { rs_ratio[i] } else { 0.0 },
                asset_vs_btc_momentum: if rs_momentum.len() > i { rs_momentum[i] } else { 0.0 },
            });
        }
        println!("    [CryptoFeatures] {} — {} rows, {} crypto features", sym, rows.len(), CryptoFeatureRow::feature_count());
        result.insert(sym.to_string(), rows);
    }
    result
}

// ============================================================================
// Utility
// ============================================================================

fn timestamp_to_date(ts: i64) -> String {
    let days = ts / 86400;
    let mut y: i64 = 1970;
    let mut rem = days;
    loop {
        let dy = if is_leap(y) { 366 } else { 365 };
        if rem < dy { break; }
        rem -= dy; y += 1;
    }
    let md = if is_leap(y) { [31,29,31,30,31,30,31,31,30,31,30,31] } else { [31,28,31,30,31,30,31,31,30,31,30,31] };
    let mut m = 0;
    for (idx, &d) in md.iter().enumerate() { if rem < d { m = idx; break; } rem -= d; }
    format!("{:04}-{:02}-{:02}", y, m+1, rem+1)
}

fn is_leap(y: i64) -> bool { (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 }
