/// export_csv — Export signal, portfolio and accuracy data to CSV files
/// =====================================================================
/// Reads from SQLite and model store, writes to exports/ directory.
/// Usage: cargo run --release --bin export_csv

use rust_invest::{db, model_store};
use std::fs;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let export_dir = "exports";
    fs::create_dir_all(export_dir)?;

    let database = db::Database::new("rust_invest.db")?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         ALPHA SIGNAL — CSV EXPORT                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    export_signals_all(&database, export_dir)?;
    export_daily_tracker(&database, export_dir)?;
    export_accuracy_by_asset(&database, export_dir)?;
    export_holdings_performance(&database, export_dir)?;

    println!("\n  All exports written to {}/", export_dir);
    Ok(())
}

/// 1. signals_all.csv — every signal snapshot
fn export_signals_all(db: &db::Database, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/signals_all.csv", dir);
    let mut f = fs::File::create(&path)?;

    writeln!(f, "date,asset,asset_class,signal,confidence,linreg_prob,logreg_prob,gbt_prob,ensemble_prob,trained_date,model_accuracy")?;

    // Use signal_history for per-model probabilities
    let signals = db.get_signal_history_all(100_000)?;

    // Get trained date from backtest results
    let version = model_store::MODEL_VERSION;
    let backtest = db.get_backtest_results(version).unwrap_or_default();
    let accuracy_map: std::collections::HashMap<String, f64> = backtest.iter()
        .map(|b| (b.asset.clone(), b.win_rate))
        .collect();

    // Get trained date from model manifest
    let manifest0 = model_store::load_manifest().ok();
    let trained_date = manifest0.as_ref()
        .map(|m| m.generated_at.clone())
        .unwrap_or_default();

    for s in &signals {
        let date = &s.timestamp[..10.min(s.timestamp.len())];
        let ensemble = match (s.linreg_prob, s.logreg_prob, s.gbt_prob) {
            (Some(l), Some(lg), Some(g)) => format!("{:.4}", (l + lg + g) / 3.0),
            _ => String::new(),
        };
        let acc = accuracy_map.get(&s.asset).map(|a| format!("{:.1}", a)).unwrap_or_default();

        writeln!(f, "{},{},{},{},{:.2},{},{},{},{},{},{}",
            date, s.asset, s.asset_class, s.signal_type,
            s.confidence,
            s.linreg_prob.map(|v| format!("{:.4}", v)).unwrap_or_default(),
            s.logreg_prob.map(|v| format!("{:.4}", v)).unwrap_or_default(),
            s.gbt_prob.map(|v| format!("{:.4}", v)).unwrap_or_default(),
            ensemble,
            &trained_date,
            acc,
        )?;
    }

    println!("  signals_all.csv         — {} rows", signals.len());
    Ok(())
}

/// 2. daily_tracker.csv — portfolio value over time
fn export_daily_tracker(db: &db::Database, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/daily_tracker.csv", dir);
    let mut f = fs::File::create(&path)?;

    writeln!(f, "date,portfolio_value_gbp,daily_pct_change,cumulative_pct_change,signals_followed")?;

    let rows = db.get_daily_portfolio(10_000)?;
    // Reverse to chronological order (DB returns DESC)
    let rows_sorted: Vec<_> = rows.into_iter().rev().collect();

    for row in &rows_sorted {
        // Count signals from signals_json if available
        let n_signals = row.signals_json.as_ref()
            .and_then(|j| serde_json::from_str::<Vec<serde_json::Value>>(j).ok())
            .map(|v| v.len())
            .unwrap_or(0);

        writeln!(f, "{},{:.2},{:.4},{:.4},{}",
            row.date, row.portfolio_value, row.daily_return, row.cumulative_return, n_signals
        )?;
    }

    println!("  daily_tracker.csv       — {} rows", rows_sorted.len());
    Ok(())
}

/// 3. accuracy_by_asset.csv — accuracy breakdown per asset
fn export_accuracy_by_asset(db: &db::Database, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/accuracy_by_asset.csv", dir);
    let mut f = fs::File::create(&path)?;

    writeln!(f, "asset,asset_class,total_signals,buy_signals,sell_signals,hold_signals,buy_accuracy,sell_accuracy,overall_accuracy,avg_confidence,trained_date")?;

    let all_signals = db.get_signal_history_all(100_000)?;

    // Group by asset
    let mut by_asset: std::collections::BTreeMap<String, Vec<&db::SignalHistoryRow>> = std::collections::BTreeMap::new();
    for s in &all_signals {
        by_asset.entry(s.asset.clone()).or_default().push(s);
    }

    let manifest = model_store::load_manifest().ok();
    let trained_date = manifest.as_ref()
        .map(|m| m.generated_at.clone())
        .unwrap_or_default();

    for (asset, signals) in &by_asset {
        let asset_class = signals.first().map(|s| s.asset_class.as_str()).unwrap_or("stock");
        let total = signals.len();
        let buys: Vec<_> = signals.iter().filter(|s| s.signal_type == "BUY").collect();
        let sells: Vec<_> = signals.iter().filter(|s| s.signal_type == "SELL").collect();
        let holds: Vec<_> = signals.iter().filter(|s| s.signal_type == "HOLD").collect();

        let buy_correct = buys.iter().filter(|s| s.was_correct == Some(true)).count();
        let buy_resolved = buys.iter().filter(|s| s.was_correct.is_some()).count();
        let sell_correct = sells.iter().filter(|s| s.was_correct == Some(true)).count();
        let sell_resolved = sells.iter().filter(|s| s.was_correct.is_some()).count();

        let total_correct = buy_correct + sell_correct;
        let total_resolved = buy_resolved + sell_resolved;

        let buy_acc = if buy_resolved > 0 { buy_correct as f64 / buy_resolved as f64 * 100.0 } else { 0.0 };
        let sell_acc = if sell_resolved > 0 { sell_correct as f64 / sell_resolved as f64 * 100.0 } else { 0.0 };
        let overall_acc = if total_resolved > 0 { total_correct as f64 / total_resolved as f64 * 100.0 } else { 0.0 };

        let avg_conf = if total > 0 {
            signals.iter().map(|s| s.confidence).sum::<f64>() / total as f64
        } else { 0.0 };

        writeln!(f, "{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.2},{}",
            asset, asset_class, total, buys.len(), sells.len(), holds.len(),
            buy_acc, sell_acc, overall_acc, avg_conf, &trained_date
        )?;
    }

    println!("  accuracy_by_asset.csv   — {} assets", by_asset.len());
    Ok(())
}

/// 4. holdings_performance.csv — current user holdings with P&L
fn export_holdings_performance(db: &db::Database, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/holdings_performance.csv", dir);
    let mut f = fs::File::create(&path)?;

    writeln!(f, "asset,shares,start_date,start_price_usd,current_price_usd,value_gbp,pnl_gbp,pnl_pct,current_signal,signal_date")?;

    let holdings = db.get_user_holdings()?;

    // Approximate GBP/USD rate (fetch latest GBPUSD=X)
    let gbp_rate = db.get_latest_fx_price("GBPUSD=X")
        .ok().flatten().unwrap_or(1.26);

    for h in &holdings {
        let start_price = get_price_at_date(db, &h.symbol, &h.asset_class, &h.start_date);
        let current_price = get_latest_price(db, &h.symbol, &h.asset_class);

        let value_usd = h.quantity * current_price;
        let cost_usd = h.quantity * start_price;
        let value_gbp = value_usd / gbp_rate;
        let pnl_gbp = (value_usd - cost_usd) / gbp_rate;
        let pnl_pct = if cost_usd > 0.0 { (current_price - start_price) / start_price * 100.0 } else { 0.0 };

        // Get latest signal for this asset
        let (signal, signal_date) = db.get_last_unresolved_signal(&h.symbol)
            .ok().flatten()
            .map(|s| (s.signal_type, s.timestamp[..10.min(s.timestamp.len())].to_string()))
            .unwrap_or_else(|| {
                // Try signal_snapshots
                db.get_signal_history(&h.symbol, 1)
                    .ok()
                    .and_then(|v| v.into_iter().next())
                    .map(|s| (s.signal.clone(), s.timestamp[..10.min(s.timestamp.len())].to_string()))
                    .unwrap_or(("N/A".to_string(), String::new()))
            });

        writeln!(f, "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
            h.symbol, h.quantity, h.start_date,
            start_price, current_price, value_gbp, pnl_gbp, pnl_pct,
            signal, signal_date
        )?;
    }

    println!("  holdings_performance.csv — {} holdings", holdings.len());
    Ok(())
}

fn get_price_at_date(db: &db::Database, symbol: &str, asset_class: &str, date: &str) -> f64 {
    match asset_class {
        "stock" => db.get_stock_price_at_date(symbol, date)
            .ok().flatten().map(|(_, p)| p).unwrap_or(0.0),
        "fx" => db.get_fx_price_at_date(symbol, date)
            .ok().flatten().map(|(_, p)| p).unwrap_or(0.0),
        "crypto" => db.get_crypto_price_at_date(symbol, date)
            .ok().flatten().map(|(_, p)| p).unwrap_or(0.0),
        _ => 0.0,
    }
}

fn get_latest_price(db: &db::Database, symbol: &str, asset_class: &str) -> f64 {
    match asset_class {
        "stock" => db.get_latest_stock_price(symbol).ok().flatten().unwrap_or(0.0),
        "fx" => db.get_latest_fx_price(symbol).ok().flatten().unwrap_or(0.0),
        "crypto" => db.get_latest_crypto_price(symbol).ok().flatten().unwrap_or(0.0),
        _ => 0.0,
    }
}
