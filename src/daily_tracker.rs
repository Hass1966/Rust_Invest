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
use crate::{db, model_store};

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

    let mut entries: Vec<DailySignalEntry> = Vec::new();
    let mut weighted_return = 0.0_f64;

    for alloc in &allocations {
        let normalised_weight = if total_weight > 0.0 { alloc.weight / total_weight } else { 0.0 };

        let signal_entry = signals.get(&alloc.asset);
        let signal = signal_entry.map(|s| s.signal.as_str()).unwrap_or("HOLD");
        let current_price = signal_entry.map(|s| s.price).unwrap_or(0.0);

        // Get yesterday's closing price from stock_history to compute return
        let price_return = compute_price_return(&database, &alloc.asset, current_price);

        // Apply return only if invested (BUY or HOLD); cash earns nothing
        let contribution = match signal {
            "SELL" => 0.0,
            _ => normalised_weight * price_return,
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
    // Try stock history first
    if let Ok(points) = database.get_stock_history(asset) {
        if points.len() >= 2 {
            let prev = points[points.len() - 2].price;
            let last = points[points.len() - 1].price;
            if prev > 0.0 {
                return (last - prev) / prev;
            }
        }
        // Only one point — use current_price vs stored
        if let Some(last) = points.last() {
            if last.price > 0.0 && current_price > 0.0 {
                return (current_price - last.price) / last.price;
            }
        }
    }

    // Try FX history
    if let Ok(points) = database.get_fx_history(asset) {
        if points.len() >= 2 {
            let prev = points[points.len() - 2].price;
            let last = points[points.len() - 1].price;
            if prev > 0.0 {
                return (last - prev) / prev;
            }
        }
    }

    // Try crypto history
    if let Ok(points) = database.get_coin_history(asset) {
        if points.len() >= 2 {
            let prev = points[points.len() - 2].price;
            let last = points[points.len() - 1].price;
            if prev > 0.0 {
                return (last - prev) / prev;
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
