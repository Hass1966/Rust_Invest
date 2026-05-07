/// Daily Portfolio Tracker
/// ========================
/// Runs once per day (at market close, ~21:00 UTC) as part of the
/// serve binary's hourly scheduler.
///
/// Logic:
///   1. Read today's current signals (already in the in-memory cache)
///   2. Get the Sharpe-weighted allocations from the last train run (Part 1)
///   3. For each allocated asset: if yesterday's signal was BUY or HOLD,
///      apply today's price return to that allocation slice.
///      If SELL, that slice sits in cash (0% return).
///   4. Compute weighted portfolio return for the day.
///   5. Compound from yesterday's portfolio_value (or the backtest seed on day 1).
///   6. Write one row to daily_portfolio.
///
/// The seed value = the Sharpe-weighted final_value from the most recent
/// training run — that is, Part 1's answer to "what is £100k worth today
/// after 3 years following this strategy?"

use std::collections::HashMap;
use crate::{db, model_store, sector, market_regime, backtest_compare};

/// One signal entry stored in the daily ledger
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DailySignalEntry {
    pub asset: String,
    pub signal: String,
    pub price: f64,
    pub weight: f64,
    pub price_return: f64,   // actual price return that day
    pub contribution: f64,   // weight × price_return (0 if SELL)
}

/// Run the end-of-day portfolio update.
///
/// `signals` — the current enriched signals (from serve's in-memory cache)
/// `db_path` — path to SQLite database
///
/// Returns the new portfolio value, or None if there's nothing to update
/// (e.g. today already has an entry, or no seed data exists yet).
pub fn run_daily_update(
    signals: &HashMap<String, crate::enriched_signals::EnrichedSignal>,
    db_path: &str,
) -> Option<f64> {
    let database = db::Database::new(db_path).ok()?;
    let model_version = model_store::MODEL_VERSION;

    // Always update today's row (hourly refresh — INSERT OR REPLACE handles duplicates)
    // This means the tracker updates every hour with the latest signals and prices
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

    // Get the Sharpe-weighted allocations (weights per asset)
    let allocations = database.get_portfolio_allocations(model_version, "sharpe").ok()?;
    if allocations.is_empty() {
        println!("  [DailyTracker] No Sharpe allocations found — run train first");
        return None;
    }

    // Get the seed value: yesterday's portfolio value, or the backtest final value on day 1
    let (seed_value, yesterday_value) = match database.get_latest_daily_portfolio().ok()? {
        Some(prev) => (prev.seed_value, prev.portfolio_value),
        None => {
            // Day 1: seed from backtest
            let seed = database.get_backtest_seed_value(model_version).ok()??;
            println!("  [DailyTracker] Day 1! Seeding from backtest: £{:.0}", seed);
            (seed, seed)
        }
    };

    // Compute today's weighted return
    // For each allocated asset:
    //   - If signal is BUY or HOLD: apply the latest price return
    //   - If signal is SELL: contribution = 0 (sitting in cash)
    let total_weight: f64 = allocations.iter().map(|a| a.weight).sum();

    // Load yesterday's signals to detect signal changes for tx cost
    let prev_signals: HashMap<String, String> = database.get_latest_daily_portfolio().ok()
        .flatten()
        .and_then(|row| row.signals_json.as_deref()
            .and_then(|j| serde_json::from_str::<Vec<DailySignalEntry>>(j).ok()))
        .map(|entries| entries.into_iter().map(|e| (e.asset, e.signal)).collect())
        .unwrap_or_default();

    let mut entries: Vec<DailySignalEntry> = Vec::new();
    let mut weighted_return = 0.0_f64;

    for alloc in &allocations {
        let normalised_weight = if total_weight > 0.0 { alloc.weight / total_weight } else { 0.0 };

        let signal_entry = signals.get(&alloc.asset);
        let signal = signal_entry.map(|s| s.signal.as_str()).unwrap_or("HOLD");
        let current_price = signal_entry.map(|s| s.price).unwrap_or(0.0);

        // Get yesterday's closing price from stock_history to compute return
        let price_return = compute_price_return(&database, &alloc.asset, current_price);

        // Deduct transaction cost on signal changes (round-trip: 10 bps stocks)
        let prev_sig = prev_signals.get(&alloc.asset).map(|s| s.as_str()).unwrap_or("HOLD");
        let signal_changed = prev_sig != signal;
        let tx = if signal_changed {
            if alloc.asset.ends_with(".L") || alloc.asset.ends_with(".DE") || alloc.asset.ends_with(".PA") {
                0.0030  // 30bps for UK/EU stocks (stamp duty averaged)
            } else {
                backtest_compare::tx_cost("stock")  // 10bps for US stocks
            }
        } else { 0.0 };

        // Confidence-tiered position sizing (0–1 scale):
        // High confidence (>0.6) → full weight (1.0x)
        // Medium confidence (0.3-0.6) → half weight (0.5x)
        // Low confidence (0.1-0.3) → quarter weight (0.25x)
        // Very low (<0.1) → skip (0x, same as cash)
        let confidence = signal_entry.map(|s| s.technical.confidence).unwrap_or(0.0);
        let confidence_multiplier = if confidence > 0.6 {
            1.0
        } else if confidence > 0.3 {
            0.5
        } else if confidence > 0.1 {
            0.25
        } else {
            0.0
        };

        let contribution = match signal {
            "SELL" => -tx * normalised_weight, // still pay tx cost if switching to SELL
            _ => normalised_weight * (price_return * confidence_multiplier - tx),
        };

        weighted_return += contribution;

        entries.push(DailySignalEntry {
            asset: alloc.asset.clone(),
            signal: signal.to_string(),
            price: current_price,
            weight: (normalised_weight * 1000.0).round() / 10.0, // as pct
            price_return: (price_return * 10000.0).round() / 100.0, // as pct
            contribution: (contribution * 10000.0).round() / 100.0,
        });
    }

    // Compound: new value = yesterday_value × (1 + weighted_return)
    let new_value = yesterday_value * (1.0 + weighted_return);
    let daily_return_pct = weighted_return * 100.0;
    let cumulative_return_pct = (new_value - seed_value) / seed_value * 100.0;

    // Serialise signal entries for the ledger
    let signals_json = serde_json::to_string(&entries).unwrap_or_default();

    let row = db::DailyPortfolioRow {
        date: today.clone(),
        seed_value,
        portfolio_value: (new_value * 100.0).round() / 100.0,
        daily_return: (daily_return_pct * 100.0).round() / 100.0,
        cumulative_return: (cumulative_return_pct * 100.0).round() / 100.0,
        signals_json: Some(signals_json),
        model_version: model_version as i64,
    };

    if let Err(e) = database.upsert_daily_portfolio(&row) {
        println!("  [DailyTracker] Failed to write row: {}", e);
        return None;
    }

    println!(
        "  [DailyTracker] {} | £{:.0} | {:+.2}% today | {:+.2}% cumulative",
        today, new_value, daily_return_pct, cumulative_return_pct
    );

    Some(new_value)
}

/// Compute the price return for an asset using the last two closing prices
/// stored in stock_history (or fx_history / crypto_history).
/// Falls back to 0.0 if history is unavailable.
fn compute_price_return(database: &db::Database, asset: &str, current_price: f64) -> f64 {
    // If we have a live current_price, use it against yesterday's stored price
    if current_price > 0.0 {
        // Try stock history
        if let Ok(points) = database.get_stock_history(asset) {
            if let Some(last) = points.last() {
                if last.price > 0.0 {
                    return (current_price - last.price) / last.price;
                }
            }
        }
        // Try FX history
        if let Ok(points) = database.get_fx_history(asset) {
            if let Some(last) = points.last() {
                if last.price > 0.0 {
                    return (current_price - last.price) / last.price;
                }
            }
        }
        // Try crypto history
        if let Ok(points) = database.get_coin_history(asset) {
            if let Some(last) = points.last() {
                if last.price > 0.0 {
                    return (current_price - last.price) / last.price;
                }
            }
        }
    }
    0.0
}

/// Build the JSON response payload for the /api/v1/portfolio/daily-tracker endpoint
pub fn build_api_response(
    db_path: &str,
) -> serde_json::Value {
    let database = match db::Database::new(db_path) {
        Ok(d) => d,
        Err(_) => return serde_json::json!({"has_data": false, "note": "Database unavailable"}),
    };

    let rows = database.get_daily_portfolio(90).unwrap_or_default(); // last 90 days

    if rows.is_empty() {
        return serde_json::json!({
            "has_data": false,
            "note": "Daily tracking begins on first run after training. Check back tomorrow."
        });
    }

    // Rows come back newest-first; reverse for the chart
    let mut chart_rows: Vec<&db::DailyPortfolioRow> = rows.iter().collect();
    chart_rows.reverse();

    let latest = rows.first().unwrap(); // newest
    let oldest = chart_rows.first().unwrap(); // oldest

    // Parse today's signal entries from JSON
    let today_signals: Vec<serde_json::Value> = latest.signals_json
        .as_deref()
        .and_then(|j| serde_json::from_str::<Vec<DailySignalEntry>>(j).ok())
        .map(|entries| entries.iter().map(|e| serde_json::json!({
            "asset": e.asset,
            "signal": e.signal,
            "weight": e.weight,
            "price_return": e.price_return,
            "contribution": e.contribution,
        })).collect())
        .unwrap_or_default();

    // Build equity curve for the chart
    let equity_curve: Vec<serde_json::Value> = chart_rows.iter().map(|r| serde_json::json!({
        "date": r.date,
        "value": r.portfolio_value,
        "daily_return": r.daily_return,
    })).collect();

    // Compute model accuracy: % of days where signal direction matched actual return
    let (correct, total) = chart_rows.iter().fold((0usize, 0usize), |(c, t), r| {
        if let Some(j) = &r.signals_json {
            if let Ok(entries) = serde_json::from_str::<Vec<DailySignalEntry>>(j) {
                let day_correct = entries.iter().filter(|e| {
                    let return_positive = e.price_return > 0.0;
                    (e.signal == "BUY" && return_positive)
                        || (e.signal == "SELL" && !return_positive)
                        || e.signal == "HOLD"
                }).count();
                let day_total = entries.len();
                return (c + day_correct, t + day_total);
            }
        }
        (c, t)
    });
    let accuracy = if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 };

    let days_tracked = rows.len();
    let inception_date = oldest.date.clone();

    serde_json::json!({
        "has_data": true,
        "seed_value": latest.seed_value,
        "current_value": latest.portfolio_value,
        "daily_return": latest.daily_return,
        "cumulative_return": latest.cumulative_return,
        "inception_date": inception_date,
        "last_updated": latest.date,
        "days_tracked": days_tracked,
        "model_accuracy_pct": (accuracy * 10.0).round() / 10.0,
        "today_signals": today_signals,
        "equity_curve": equity_curve,
    })
}

/// A proposed allocation entry for the rebalance endpoint
#[derive(Debug, Clone, serde::Serialize)]
pub struct RebalanceEntry {
    pub asset: String,
    pub sector: String,
    pub signal: String,
    pub confidence: f64,
    pub sector_multiplier: f64,
    pub raw_weight: f64,
    pub final_weight_pct: f64,
}

/// Compute optimal sector-weighted allocation from current signals + regime.
///
/// Steps:
///   1. Get sector scores (weight_multiplier 0.5x-1.5x)
///   2. Get regime → exposure caps (BULL=80%, BEAR=35%, etc.)
///   3. Filter to BUY signals only, ranked by confidence × sector_multiplier
///   4. Apply inverse-volatility weighting + 15% single-position cap
///   5. Return proposed allocation
pub fn compute_rebalance(
    signals: &HashMap<String, crate::enriched_signals::EnrichedSignal>,
    regime: Option<&market_regime::MarketRegimeState>,
    db_path: &str,
) -> serde_json::Value {
    // Regime-based maximum equity exposure
    let max_exposure = match regime.map(|r| &r.regime) {
        Some(market_regime::MarketRegime::Bull) => 0.80,
        Some(market_regime::MarketRegime::Neutral) => 0.65,
        Some(market_regime::MarketRegime::EarlyWarning) => 0.50,
        Some(market_regime::MarketRegime::Bear) => 0.35,
        Some(market_regime::MarketRegime::Crisis) => 0.20,
        None => 0.65,
    };

    let regime_label = regime.map(|r| r.regime.to_string()).unwrap_or_else(|| "UNKNOWN".to_string());

    // Build sector score map
    let signal_inputs: Vec<sector::SignalInput> = signals.values().map(|s| sector::SignalInput {
        asset: s.asset.clone(),
        asset_class: s.asset_class.clone(),
        signal: s.signal.clone(),
        confidence: s.technical.confidence,
        probability_up: s.models.get("gbt").map(|m| m.probability_up).unwrap_or(50.0),
    }).collect();

    let sector_scores = sector::calculate_sector_scores(&signal_inputs);
    let sector_map: HashMap<String, f64> = sector_scores.iter()
        .map(|s| (s.label.clone(), s.weight_multiplier))
        .collect();

    // Filter to BUY signals, compute raw weights
    let mut buy_signals: Vec<(&String, &crate::enriched_signals::EnrichedSignal, f64)> = signals.iter()
        .filter(|(_, s)| s.signal == "BUY")
        .map(|(k, s)| {
            let sec = sector::classify_sector_with_class(&s.asset, &s.asset_class);
            let multiplier = sector_map.get(sec.label()).copied().unwrap_or(1.0);
            let raw_weight = s.technical.confidence * multiplier;
            (k, s, raw_weight)
        })
        .collect();

    // Sort by raw weight descending
    buy_signals.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    if buy_signals.is_empty() {
        return serde_json::json!({
            "regime": regime_label,
            "max_exposure_pct": max_exposure * 100.0,
            "cash_pct": 100.0,
            "allocations": [],
            "note": "No BUY signals — 100% cash position recommended"
        });
    }

    // Inverse-volatility weighting: use confidence as proxy for conviction
    // (actual vol data would require DB lookups for each asset's history)
    let total_raw: f64 = buy_signals.iter().map(|(_, _, w)| w).sum();

    let mut entries: Vec<RebalanceEntry> = Vec::new();
    let single_cap = 0.15; // 15% max per position

    for (_, sig, raw_w) in &buy_signals {
        let sec = sector::classify_sector_with_class(&sig.asset, &sig.asset_class);
        let multiplier = sector_map.get(sec.label()).copied().unwrap_or(1.0);
        let norm_weight = if total_raw > 0.0 { raw_w / total_raw * max_exposure } else { 0.0 };
        let capped_weight = norm_weight.min(single_cap);

        entries.push(RebalanceEntry {
            asset: sig.asset.clone(),
            sector: sec.label().to_string(),
            signal: sig.signal.clone(),
            confidence: (sig.technical.confidence * 10.0).round() / 10.0,
            sector_multiplier: multiplier,
            raw_weight: (*raw_w * 100.0).round() / 100.0,
            final_weight_pct: (capped_weight * 1000.0).round() / 10.0,
        });
    }

    // Renormalise after capping to use full max_exposure
    let total_capped: f64 = entries.iter().map(|e| e.final_weight_pct).sum();
    if total_capped > 0.0 {
        let scale = (max_exposure * 100.0) / total_capped;
        for e in &mut entries {
            e.final_weight_pct = (e.final_weight_pct * scale * 10.0).round() / 10.0;
            // Re-enforce single cap after renormalisation
            if e.final_weight_pct > single_cap * 100.0 {
                e.final_weight_pct = (single_cap * 100.0 * 10.0).round() / 10.0;
            }
        }
    }

    let invested_pct: f64 = entries.iter().map(|e| e.final_weight_pct).sum();
    let cash_pct = 100.0 - invested_pct;

    serde_json::json!({
        "regime": regime_label,
        "max_exposure_pct": (max_exposure * 100.0).round(),
        "invested_pct": (invested_pct * 10.0).round() / 10.0,
        "cash_pct": (cash_pct * 10.0).round() / 10.0,
        "positions": entries.len(),
        "allocations": entries,
    })
}
