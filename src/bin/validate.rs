/// validate — Retrospective Signal Validation
/// ============================================
/// Simulates "what if you had followed the model's signals for the last N days"
/// using historical price data already in the DB and saved model weights.
///
/// This is NOT a backtest — it uses the CURRENT saved models (trained on all
/// available data up to the last train run) and replays them against the last
/// N days of prices, computing what signals would have been generated each day
/// and what the actual next-day return was.
///
/// This answers: "Is the model performing as claimed right now?"
///
/// Usage:
///   cargo run --release --bin validate
///   cargo run --release --bin validate -- --days 14 --capital 10000
///   cargo run --release --bin validate -- --days 7

use rust_invest::*;
use std::collections::HashMap;
use chrono::{NaiveDate, Duration};

// ── CLI args ──

struct Config {
    days: usize,
    capital: f64,
    db_path: String,
    verbose: bool,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut days = 14_usize;
        let mut capital = 10_000.0_f64;
        let mut db_path = "rust_invest.db".to_string();
        let mut verbose = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--days" | "-d" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        days = v.parse().unwrap_or(14);
                    }
                }
                "--capital" | "-c" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        capital = v.parse().unwrap_or(10_000.0);
                    }
                }
                "--db" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        db_path = v.clone();
                    }
                }
                "--verbose" | "-v" => verbose = true,
                _ => {}
            }
            i += 1;
        }

        Config { days, capital, db_path, verbose }
    }
}

// ── Result types ──

#[derive(Debug, Clone)]
struct DayResult {
    date: String,
    portfolio_value: f64,
    daily_return_pct: f64,
    signals: Vec<AssetSignalResult>,
    correct_signals: usize,
    total_signals: usize,
}

#[derive(Debug, Clone)]
struct AssetSignalResult {
    asset: String,
    asset_class: String,
    signal: String,
    actual_return_pct: f64,
    weight: f64,
    contribution_pct: f64,
    was_correct: bool,
}

// ── Main ──

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_args();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         ALPHA SIGNAL — VALIDATION (Retrospective Simulation)   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!("  Period:  Last {} trading days", cfg.days);
    println!("  Capital: £{:.0}", cfg.capital);
    println!("  DB:      {}\n", cfg.db_path);

    let database = db::Database::new(&cfg.db_path)?;
    let model_version = model_store::MODEL_VERSION;

    // Load Sharpe-weighted allocations from last train run
    let allocations = database.get_portfolio_allocations(model_version, "sharpe")?;
    if allocations.is_empty() {
        eprintln!("  ERROR: No Sharpe allocations found. Run train first.");
        std::process::exit(1);
    }

    println!("  Loaded {} asset allocations from last training run", allocations.len());

    // Normalise weights to sum to 1.0
    let total_weight: f64 = allocations.iter().map(|a| a.weight).sum();
    let weights: HashMap<String, f64> = allocations.iter()
        .map(|a| (a.asset.clone(), a.weight / total_weight))
        .collect();

    // Build market context (used for feature generation)
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?
        .iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker).unwrap_or_default();
        market_histories.insert(ticker.to_string(), prices);
    }
    let market_context = features::build_market_context(&market_histories);

    // Load all price history for each allocated asset
    println!("\n  Loading price history for {} assets...", weights.len());
    let mut asset_prices: HashMap<String, Vec<(String, f64)>> = HashMap::new(); // (date, price)

    for alloc in &allocations {
        let symbol = &alloc.asset;

        // Try stocks first, then FX
        let points = if stocks::STOCK_LIST.iter().any(|s| s.symbol == symbol.as_str()) {
            database.get_stock_history(symbol)
                .unwrap_or_default()
                .into_iter()
                .map(|p| (p.timestamp[..10].to_string(), p.price))
                .collect::<Vec<_>>()
        } else if stocks::FX_LIST.iter().any(|s| s.symbol == symbol.as_str()) {
            database.get_fx_history(symbol)
                .unwrap_or_default()
                .into_iter()
                .map(|p| (p.timestamp[..10].to_string(), p.price))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        if !points.is_empty() {
            asset_prices.insert(symbol.clone(), points);
        }
    }

    // Determine the trading days to simulate
    // We need prices for days D and D+1 to compute actual returns
    // So we take the last (days + 2) days of data, use days for signals
    let today = chrono::Utc::now().date_naive();
    let start_date = today - Duration::days((cfg.days as i64) + 5); // +5 for weekends

    // Collect all unique trading dates from the price data
    let mut all_dates: Vec<String> = {
        let first_asset = asset_prices.values().next().unwrap();
        first_asset.iter()
            .map(|(d, _)| d.clone())
            .filter(|d| {
                let nd = NaiveDate::parse_from_str(d, "%Y-%m-%d").unwrap_or(NaiveDate::MIN);
                nd >= start_date && nd < today
            })
            .collect()
    };
    all_dates.sort();
    all_dates.dedup();

    // Take only the last N trading days
    if all_dates.len() > cfg.days {
        all_dates = all_dates[all_dates.len() - cfg.days..].to_vec();
    }

    if all_dates.is_empty() {
        eprintln!("  ERROR: No price data found for the requested period.");
        eprintln!("  Make sure train has been run and stock_history is populated.");
        std::process::exit(1);
    }

    println!("  Simulating {} trading days: {} → {}\n",
        all_dates.len(),
        all_dates.first().unwrap(),
        all_dates.last().unwrap()
    );

    // ── Day-by-day simulation ──

    let mut portfolio_value = cfg.capital;
    let mut day_results: Vec<DayResult> = Vec::new();
    let mut total_correct = 0_usize;
    let mut total_signals = 0_usize;

    // Build price lookup: asset → date → price
    let price_lookup: HashMap<String, HashMap<String, f64>> = asset_prices.iter()
        .map(|(asset, points)| {
            let by_date: HashMap<String, f64> = points.iter()
                .cloned()
                .collect();
            (asset.clone(), by_date)
        })
        .collect();

    // Determine asset classes
    let asset_class: HashMap<String, &str> = allocations.iter()
        .map(|a| {
            let cls = if stocks::STOCK_LIST.iter().any(|s| s.symbol == a.asset.as_str()) {
                "stock"
            } else if stocks::FX_LIST.iter().any(|s| s.symbol == a.asset.as_str()) {
                "fx"
            } else {
                "crypto"
            };
            (a.asset.clone(), cls)
        })
        .collect();

    for (day_idx, date) in all_dates.iter().enumerate() {
        // Find the next trading day for actual return computation
        let next_date = all_dates.get(day_idx + 1).cloned().unwrap_or_else(|| {
            // For the last day, use today's price
            today.format("%Y-%m-%d").to_string()
        });

        let mut day_weighted_return = 0.0_f64;
        let mut day_signals: Vec<AssetSignalResult> = Vec::new();
        let mut day_correct = 0_usize;
        let mut day_total = 0_usize;

        for alloc in &allocations {
            let symbol = &alloc.asset;
            let weight = weights[symbol];
            let cls = asset_class.get(symbol).copied().unwrap_or("stock");

            // Get price on signal date and next date
            let price_today = price_lookup.get(symbol)
                .and_then(|m| m.get(date.as_str()))
                .copied();
            let price_next = price_lookup.get(symbol)
                .and_then(|m| m.get(next_date.as_str()))
                .copied();

            let (price_today, price_next) = match (price_today, price_next) {
                (Some(t), Some(n)) => (t, n),
                _ => continue, // skip if no data for this day
            };

            let actual_return = (price_next - price_today) / price_today;

            // Generate the signal for this date using prices UP TO this date
            // (no lookahead — we slice the history to only include data available on this date)
            let signal = generate_signal_for_date(
                symbol, cls, date, &asset_prices, &market_context, &database, cfg.verbose,
            ).unwrap_or("HOLD".to_string());

            // Apply return: BUY/HOLD → invested, SELL → cash
            let contribution = match signal.as_str() {
                "SELL" => 0.0,
                _ => weight * actual_return,
            };

            day_weighted_return += contribution;

            // Was the signal correct?
            let was_correct = match signal.as_str() {
                "BUY" => actual_return > 0.0,
                "SELL" => actual_return < 0.0,
                "HOLD" => true, // HOLD is always "correct" — it's a neutral call
                _ => false,
            };

            // Only count BUY/SELL for accuracy (HOLD is always excluded from accuracy calc)
            if signal != "HOLD" {
                if was_correct { day_correct += 1; }
                day_total += 1;
            }

            day_signals.push(AssetSignalResult {
                asset: symbol.clone(),
                asset_class: cls.to_string(),
                signal,
                actual_return_pct: actual_return * 100.0,
                weight: weight * 100.0,
                contribution_pct: contribution * 100.0,
                was_correct,
            });
        }

        // Compound
        portfolio_value *= 1.0 + day_weighted_return;
        total_correct += day_correct;
        total_signals += day_total;

        day_results.push(DayResult {
            date: date.clone(),
            portfolio_value,
            daily_return_pct: day_weighted_return * 100.0,
            signals: day_signals,
            correct_signals: day_correct,
            total_signals: day_total,
        });
    }

    // ── Print Results ──

    print_results(&day_results, cfg.capital, total_correct, total_signals, &cfg);

    Ok(())
}

// ── Signal generation for a specific historical date ──

fn generate_signal_for_date(
    symbol: &str,
    asset_class: &str,
    date: &str,
    asset_prices: &HashMap<String, Vec<(String, f64)>>,
    market_context: &features::MarketContext,
    _database: &db::Database,
    verbose: bool,
) -> Option<String> {
    let points_full = asset_prices.get(symbol)?;

    // Slice to only prices available UP TO AND INCLUDING this date (no lookahead)
    let cutoff_points: Vec<analysis::PricePoint> = points_full.iter()
        .filter(|(d, _)| d.as_str() <= date)
        .map(|(d, p)| analysis::PricePoint {
            timestamp: d.clone(),
            price: *p,
            volume: None,
        })
        .collect();

    if cutoff_points.len() < 100 { return None; }

    let prices: Vec<f64> = cutoff_points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = cutoff_points.iter().map(|_| None).collect();
    let timestamps: Vec<String> = cutoff_points.iter().map(|p| p.timestamp.clone()).collect();

    let samples = match asset_class {
        "stock" | "fx" => features::build_rich_features(
            &prices, &volumes, &timestamps,
            Some(market_context), asset_class,
            if asset_class == "stock" { features::sector_etf_for(symbol) } else { None },
            None, None,
        ),
        _ => return None,
    };

    if samples.is_empty() { return None; }

    // Run inference using saved models
    let wf = infer_with_saved_models_quiet(symbol, &samples, verbose)?;

    let result = analysis::analyse_coin(symbol, &cutoff_points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(
        symbol,
        &wf,
        result.current_price,
        result.rsi_14.unwrap_or(50.0),
        trend,
    );

    Some(signal.signal.clone())
}

/// Inference without the verbose per-asset logging (cleaner output for validate)
fn infer_with_saved_models_quiet(
    symbol: &str,
    samples: &[ml::Sample],
    verbose: bool,
) -> Option<ensemble::WalkForwardResult> {
    if samples.is_empty() { return None; }

    let n_features = samples[0].features.len();

    let linreg_saved = model_store::load_weights(symbol, "linreg").ok()?;
    let logreg_saved = model_store::load_weights(symbol, "logreg").ok()?;
    let (gbt_saved, gbt_classifier) = model_store::load_gbt(symbol).ok()?;

    let feat = &samples.last().unwrap().features;

    let lin_feat = normalise_features(feat, &linreg_saved.norm_means, &linreg_saved.norm_stds);
    let raw_lin: f64 = linreg_saved.bias + linreg_saved.weights.iter().zip(lin_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
    let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);

    let log_feat = normalise_features(feat, &logreg_saved.norm_means, &logreg_saved.norm_stds);
    let log_z: f64 = logreg_saved.bias + logreg_saved.weights.iter().zip(log_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
    let log_prob = (1.0 / (1.0 + (-log_z).exp())).clamp(0.15, 0.85);

    let gbt_feat = normalise_features(feat, &gbt_saved.norm_means, &gbt_saved.norm_stds);
    let gbt_prob = gbt_classifier.predict_proba(&gbt_feat).clamp(0.15, 0.85);

    let lin_acc = linreg_saved.meta.walk_forward_accuracy;
    let log_acc = logreg_saved.meta.walk_forward_accuracy;
    let gbt_acc = gbt_saved.meta.walk_forward_accuracy;

    if verbose {
        println!("    {} | LinR={:.1}% LogR={:.1}% GBT={:.1}% | p: {:.2} {:.2} {:.2}",
            symbol, lin_acc, log_acc, gbt_acc, lin_prob, log_prob, gbt_prob);
    }

    Some(ensemble::WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: lin_acc,
        logistic_accuracy: log_acc,
        gbt_accuracy: gbt_acc,
        lstm_accuracy: 50.0,
        gru_accuracy: 50.0,
        rf_accuracy: 50.0,
        n_folds: 1,
        total_test_samples: 0,
        linear_recent: lin_acc,
        logistic_recent: log_acc,
        gbt_recent: gbt_acc,
        lstm_recent: 50.0,
        gru_recent: 50.0,
        rf_recent: 50.0,
        final_linear_prob: lin_prob,
        final_logistic_prob: log_prob,
        final_gbt_prob: gbt_prob,
        final_lstm_prob: 0.5,
        final_gru_prob: 0.5,
        final_rf_prob: 0.5,
        gbt_importance: Vec::new(),
        n_features,
        has_lstm: false,
        has_gru: false,
        has_rf: false,
        stacking_weights: None,
        val_log_loss: None,
    })
}

fn normalise_features(features: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    features.iter().enumerate().map(|(i, &f)| {
        let mean = means.get(i).copied().unwrap_or(0.0);
        let std = stds.get(i).copied().unwrap_or(1.0);
        if std == 0.0 { f - mean } else { (f - mean) / std }
    }).collect()
}

// ── Pretty-print results ──

fn print_results(
    days: &[DayResult],
    starting_capital: f64,
    total_correct: usize,
    total_signals: usize,
    _cfg: &Config,
) {
    let final_value = days.last().map(|d| d.portfolio_value).unwrap_or(starting_capital);
    let total_return = (final_value - starting_capital) / starting_capital * 100.0;
    let accuracy = if total_signals > 0 {
        total_correct as f64 / total_signals as f64 * 100.0
    } else { 0.0 };

    // Colour helpers (ANSI)
    let green  = |s: String| format!("\x1b[92m{}\x1b[0m", s);
    let red    = |s: String| format!("\x1b[91m{}\x1b[0m", s);
    let yellow = |s: String| format!("\x1b[93m{}\x1b[0m", s);
    let cyan   = |s: String| format!("\x1b[96m{}\x1b[0m", s);
    let dim    = |s: String| format!("\x1b[2m{}\x1b[0m", s);
    let bold   = |s: String| format!("\x1b[1m{}\x1b[0m", s);

    let fmt_pct = |v: f64| -> String {
        if v >= 0.0 { green(format!("+{:.2}%", v)) }
        else { red(format!("{:.2}%", v)) }
    };

    println!("{}", bold("━━━ DAY BY DAY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".to_string()));
    println!("{:<12} {:>10} {:>10} {:>8} {:>8}",
        dim("Date".to_string()),
        dim("Value".to_string()),
        dim("Day P&L".to_string()),
        dim("Acc".to_string()),
        dim("Signals".to_string()),
    );
    println!("{}", dim("─".repeat(60)));

    for day in days {
        let val_str = format!("£{:.0}", day.portfolio_value);
        let acc_str = if day.total_signals > 0 {
            let a = day.correct_signals as f64 / day.total_signals as f64 * 100.0;
            if a >= 60.0 { green(format!("{:.0}%", a)) }
            else if a >= 50.0 { yellow(format!("{:.0}%", a)) }
            else { red(format!("{:.0}%", a)) }
        } else {
            dim("—".to_string())
        };

        println!("  {}  {:>10}  {}  {}  {}",
            cyan(day.date.clone()),
            val_str,
            fmt_pct(day.daily_return_pct),
            acc_str,
            dim(format!("{}/{}", day.correct_signals, day.total_signals)),
        );

        // Per-asset detail on each day
        for sig in &day.signals {
            let sig_col = match sig.signal.as_str() {
                "BUY"  => green("BUY ".to_string()),
                "SELL" => red("SELL".to_string()),
                _      => yellow("HOLD".to_string()),
            };
            let correct_icon = if sig.signal == "HOLD" {
                dim("  ·".to_string())
            } else if sig.was_correct {
                green("  ✓".to_string())
            } else {
                red("  ✗".to_string())
            };
            println!("    {} {} {:>8}  actual {} contrib {}{}",
                sig_col,
                format!("{:<12}", sig.asset),
                dim(format!("{:.1}%w", sig.weight)),
                fmt_pct(sig.actual_return_pct),
                fmt_pct(sig.contribution_pct),
                correct_icon,
            );
        }
        println!();
    }

    // ── Summary ──

    println!("{}", bold("━━━ SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".to_string()));
    println!();
    println!("  Starting capital : £{:.0}", starting_capital);
    println!("  Final value      : {}", cyan(format!("£{:.2}", final_value)));
    println!("  Total return     : {}", fmt_pct(total_return));
    println!("  Period           : {} trading days", days.len());
    println!();

    let verdict = if accuracy >= 60.0 {
        green("✓ MODEL PERFORMING AS EXPECTED".to_string())
    } else if accuracy >= 50.0 {
        yellow("⚠ MODEL MARGINAL — WATCH CLOSELY".to_string())
    } else {
        red("✗ MODEL BELOW RANDOM — INVESTIGATE".to_string())
    };

    println!("  Signal accuracy  : {}  (BUY/SELL only, {}/{})",
        if accuracy >= 60.0 { green(format!("{:.1}%", accuracy)) }
        else if accuracy >= 50.0 { yellow(format!("{:.1}%", accuracy)) }
        else { red(format!("{:.1}%", accuracy)) },
        total_correct,
        total_signals,
    );
    println!("  Verdict          : {}", verdict);
    println!();

    // Per-asset breakdown
    println!("{}", bold("━━━ PER-ASSET ACCURACY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".to_string()));
    println!();

    // Aggregate per asset
    let mut asset_stats: HashMap<String, (usize, usize, f64)> = HashMap::new(); // (correct, total, cum_contribution)
    for day in days {
        for sig in &day.signals {
            let entry = asset_stats.entry(sig.asset.clone()).or_insert((0, 0, 0.0));
            entry.2 += sig.contribution_pct;
            if sig.signal != "HOLD" {
                entry.1 += 1;
                if sig.was_correct { entry.0 += 1; }
            }
        }
    }

    let mut asset_list: Vec<(String, (usize, usize, f64))> = asset_stats.into_iter().collect();
    asset_list.sort_by(|a, b| b.1.2.partial_cmp(&a.1.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("  {:<14} {:>8} {:>10} {:>12}",
        dim("Asset".to_string()),
        dim("Accuracy".to_string()),
        dim("Signals".to_string()),
        dim("Contribution".to_string()),
    );
    println!("  {}", dim("─".repeat(50)));

    for (asset, (correct, total, contribution)) in &asset_list {
        let acc_str = if *total > 0 {
            let a = *correct as f64 / *total as f64 * 100.0;
            if a >= 60.0 { green(format!("{:.0}%", a)) }
            else if a >= 50.0 { yellow(format!("{:.0}%", a)) }
            else { red(format!("{:.0}%", a)) }
        } else {
            dim("HOLD only".to_string())
        };

        println!("  {:<14} {:>8} {:>10} {}",
            format!("{:<14}", asset),
            acc_str,
            dim(format!("{}/{}", correct, total)),
            fmt_pct(*contribution),
        );
    }

    println!();
    println!("  Note: HOLD signals excluded from accuracy. Contribution = weighted return.");
    println!("        Run with --verbose to see per-day model probabilities.");
    println!();
}
