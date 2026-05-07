//! Walk-Forward Backtester
//!
//! Generates out-of-sample signals using expanding-window retraining.
//! For each quarterly test window, models are trained ONLY on prior data,
//! ensuring zero lookahead bias.
//!
//! Output: reports/walkforward_backtest.json

use rust_invest::{analysis, db, features, ml, gbt, lgbm, ensemble, stocks, lstm, ridge, gru, backtest_compare};
use std::collections::HashMap;
use chrono::NaiveDate;

/// A quarterly test window definition
struct TestWindow {
    train_end: NaiveDate,   // last date of training data (exclusive of test)
    test_start: NaiveDate,
    test_end: NaiveDate,
}

/// Individual signal record for the output
#[derive(serde::Serialize)]
struct WFSignal {
    date: String,
    asset: String,
    asset_class: String,
    signal: String,
    entry_price: f64,
    exit_price: Option<f64>,
    pct_return: Option<f64>,
    was_correct: Option<bool>,
    train_window_end: String,
    buy_probability: f64,
    confidence: f64,
    ridge_return: f64,
    lgbm_return: f64,
    gru_return: f64,
    ensemble_return: f64,
    // Legacy fields for backward compat with dashboard
    linreg_prob: f64,
    logreg_prob: f64,
    gbt_prob: f64,
    lgbm_prob: f64,
    lstm_prob: f64,
    regime_prob: f64,
}

/// Per-window summary
#[derive(serde::Serialize)]
struct WindowSummary {
    train_end: String,
    test_start: String,
    test_end: String,
    signals_generated: usize,
    buy_accuracy: f64,
    sell_accuracy: f64,
}

/// Overall summary
#[derive(serde::Serialize, Clone)]
struct Summary {
    total_signals: usize,
    buy_accuracy: f64,
    sell_accuracy: f64,
    expected_value_bps: f64,
    profit_factor: f64,
    sharpe_ratio: f64,
    max_drawdown_pct: f64,
}

/// Full output
#[derive(serde::Serialize)]
struct WalkForwardOutput {
    generated_at: String,
    windows: Vec<WindowSummary>,
    signals: Vec<WFSignal>,
    summary: Summary,
}

fn define_test_windows() -> Vec<TestWindow> {
    let mut windows = Vec::new();
    // Q1 2020 through Q4 2025: 24 quarterly windows (extended for COVID crash + recovery)
    let quarters = [
        ("2020-01-02", "2020-03-31"),  // Q1 2020 (COVID crash)
        ("2020-04-01", "2020-06-30"),  // Q2 2020 (V-recovery)
        ("2020-07-01", "2020-09-30"),
        ("2020-10-01", "2020-12-31"),
        ("2021-01-04", "2021-03-31"),
        ("2021-04-01", "2021-06-30"),
        ("2021-07-01", "2021-09-30"),
        ("2021-10-01", "2021-12-31"),
        ("2022-01-03", "2022-03-31"),
        ("2022-04-01", "2022-06-30"),
        ("2022-07-01", "2022-09-30"),
        ("2022-10-03", "2022-12-30"),
        ("2023-01-03", "2023-03-31"),
        ("2023-04-03", "2023-06-30"),
        ("2023-07-03", "2023-09-29"),
        ("2023-10-02", "2023-12-29"),
        ("2024-01-02", "2024-03-28"),
        ("2024-04-01", "2024-06-28"),
        ("2024-07-01", "2024-09-30"),
        ("2024-10-01", "2024-12-31"),
        ("2025-01-02", "2025-03-31"),
        ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-09-30"),
        ("2025-10-01", "2025-12-31"),
    ];

    for (test_s, test_e) in quarters {
        let test_start = NaiveDate::parse_from_str(test_s, "%Y-%m-%d").unwrap();
        let test_end = NaiveDate::parse_from_str(test_e, "%Y-%m-%d").unwrap();
        // Training data ends the day before test starts
        let train_end = test_start - chrono::Duration::days(1);
        windows.push(TestWindow { train_end, test_start, test_end });
    }
    windows
}

/// Find the index of the first sample whose timestamp is >= target_date
fn find_date_index(timestamps: &[String], target_date: &NaiveDate) -> Option<usize> {
    let target_str = target_date.format("%Y-%m-%d").to_string();
    timestamps.iter().position(|ts| {
        let date_part = &ts[..10.min(ts.len())];
        date_part >= target_str.as_str()
    })
}

/// Generate a signal from regression model return predictions.
/// Uses asset-class-specific return thresholds.
fn signal_from_returns(
    _symbol: &str,
    ensemble_return: f64,
    asset_class: &str,
) -> (String, f64, f64) {
    let threshold = match asset_class {
        "crypto" => 1.0,
        "fx" => 0.2,
        _ => 0.5,
    };

    let signal = if ensemble_return > threshold {
        "BUY"
    } else if ensemble_return < -threshold {
        "SELL"
    } else {
        "HOLD"
    };

    let signal_strength = (ensemble_return.abs() / 2.0).min(1.0);
    // Backward-compatible probability mapping
    let ensemble_prob = 0.5 + ensemble_return.clamp(-5.0, 5.0) / 10.0;

    (signal.to_string(), ensemble_prob, signal_strength)
}

/// Minimum move threshold for "correct" classification
fn min_threshold(asset_class: &str) -> f64 {
    match asset_class {
        "crypto" => 1.0,
        "fx" => 0.2,
        _ => 0.5,
    }
}

/// Train 3 regression models on a slice of samples and generate predictions for test samples.
/// Models: Ridge, LightGBM Regressor, GRU Regression
/// Uses inverse-MAE weighting for ensemble combination.
/// Returns Vec<(signal, ensemble_prob, confidence, ridge_ret, lgbm_ret, gru_ret, ensemble_ret)>
fn train_and_predict(
    symbol: &str,
    asset_class: &str,
    train_samples: &[ml::Sample],
    test_samples: &[ml::Sample],
) -> Vec<(String, f64, f64, f64, f64, f64, f64)> {
    if train_samples.len() < 50 || test_samples.is_empty() {
        return Vec::new();
    }

    let n_feat = train_samples[0].features.len();

    // Normalise training data
    let mut train_data = train_samples.to_vec();
    let (means, stds) = ml::normalise(&mut train_data);

    // Recency weights
    let recency = ensemble::compute_recency_weights(train_data.len());

    // Split: train on first 85%, validate on last 15%
    let val_start = (train_data.len() as f64 * 0.85) as usize;

    let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
    let y_train: Vec<f64> = train_data.iter().map(|s| s.label).collect();
    let (x_t, x_v) = x_train.split_at(val_start);
    let (y_t, y_v) = y_train.split_at(val_start);

    // ── Model 1: Ridge Regression ──
    let ridge_model = match ridge::RidgeRegression::train(x_t, y_t, 10.0) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };

    // ── Model 2: LightGBM Regressor ──
    let lgbm_config = lgbm::LGBMRegressorConfig::default();
    let lgbm_model = lgbm::LightGBMRegressor::train(
        x_t, y_t, Some(&recency[..x_t.len()]),
        Some(x_v), Some(y_v), &lgbm_config,
    ).ok();

    // ── Model 3: GRU Regression (top 30 features) ──
    let feature_names_list = features::active_feature_names();
    let feature_name_refs: Vec<&str> = feature_names_list.iter().map(|s| s.as_str()).collect();
    let importance = if let Some(ref lgbm) = lgbm_model {
        lgbm.feature_importance(&feature_name_refs)
    } else {
        // Use a dummy GBT for feature importance if no LGBM
        let y_binary: Vec<f64> = y_t.iter().map(|&y| if y > 0.0 { 1.0 } else { 0.0 }).collect();
        let gbt_config = gbt::GBTConfig::default();
        let gbt_tmp = gbt::GradientBoostedClassifier::train_weighted(
            x_t, &y_binary, None, None, None, gbt_config
        );
        gbt_tmp.feature_importance(&feature_name_refs)
    };
    let top_feature_indices: Vec<usize> = importance.iter()
        .take(30.min(importance.len()))
        .filter_map(|(name, _)| feature_names_list.iter().position(|n| n == name))
        .collect();
    let gru_indices = if top_feature_indices.len() >= 20 { Some(top_feature_indices.as_slice()) } else { None };
    let gru_input_size = gru_indices.map_or(n_feat, |idx| idx.len());

    let gru_config = gru::GRURegressionConfig {
        input_size: gru_input_size,
        hidden_size: 48,
        seq_length: 10,
        learning_rate: 0.001,
        epochs: 30, // reduced for walk-forward speed
        batch_size: 64,
        weight_decay: 0.03,
        ..gru::GRURegressionConfig::default()
    };
    let gru_model = gru::GRURegressionModel::new(gru_config).ok().and_then(|mut model| {
        let gru_val_start = val_start.saturating_sub(gru_config.seq_length * 5);
        if gru_val_start < 50 { return None; }
        match model.train_regression(&train_data[..gru_val_start], &train_data[gru_val_start..val_start], gru_indices) {
            Ok(_) => Some(model),
            Err(_) => None,
        }
    });

    // ── Evaluate on validation set to get MAE weights ──
    let val_data = &train_data[val_start..];
    let mut ridge_mae_sum = 0.0_f64;
    let mut lgbm_mae_sum = 0.0_f64;
    let mut gru_mae_sum = 0.0_f64;
    let mut ridge_count = 0usize;
    let mut lgbm_count = 0usize;
    let mut gru_count = 0usize;

    // Build GRU sequences for validation
    let val_seqs = lstm::build_sequences_regression(val_data, 10, gru_indices);

    for (vi, sample) in val_data.iter().enumerate() {
        let actual = sample.label;

        let ridge_pred = ridge_model.predict(&sample.features);
        ridge_mae_sum += (ridge_pred - actual).abs();
        ridge_count += 1;

        if let Some(ref model) = lgbm_model {
            let lgbm_pred = model.predict_return(&sample.features);
            lgbm_mae_sum += (lgbm_pred - actual).abs();
            lgbm_count += 1;
        }

        if let Some(ref model) = gru_model {
            if vi < val_seqs.len() {
                if let Ok(gru_pred) = model.predict_return(&val_seqs[vi].features) {
                    gru_mae_sum += (gru_pred - actual).abs();
                    gru_count += 1;
                }
            }
        }
    }

    let ridge_mae = if ridge_count > 0 { ridge_mae_sum / ridge_count as f64 } else { f64::MAX };
    let lgbm_mae = if lgbm_count > 0 { lgbm_mae_sum / lgbm_count as f64 } else { f64::MAX };
    let gru_mae = if gru_count > 0 { gru_mae_sum / gru_count as f64 } else { f64::MAX };

    // Inverse-MAE weights
    let ridge_w = if ridge_mae < f64::MAX { 1.0 / ridge_mae.max(0.01) } else { 0.0 };
    let lgbm_w = if lgbm_mae < f64::MAX { 1.0 / lgbm_mae.max(0.01) } else { 0.0 };
    let gru_w = if gru_mae < f64::MAX { 1.0 / gru_mae.max(0.01) } else { 0.0 };
    let total_w = ridge_w + lgbm_w + gru_w;

    // ── Generate predictions for test samples ──
    let mut test_norm: Vec<ml::Sample> = Vec::with_capacity(test_samples.len());
    for sample in test_samples {
        let mut features = sample.features.clone();
        for j in 0..features.len() {
            if j < means.len() && j < stds.len() && stds[j] > 1e-10 {
                features[j] = (features[j] - means[j]) / stds[j];
            }
        }
        test_norm.push(ml::Sample { features, label: sample.label });
    }
    let test_seqs = lstm::build_sequences_regression(&test_norm, 10, gru_indices);

    let mut results = Vec::with_capacity(test_samples.len());
    for (ti, sample) in test_norm.iter().enumerate() {
        let ridge_ret = ridge_model.predict(&sample.features);

        let lgbm_ret = if let Some(ref model) = lgbm_model {
            model.predict_return(&sample.features)
        } else { 0.0 };

        let gru_ret = if let Some(ref model) = gru_model {
            if ti < test_seqs.len() {
                model.predict_return(&test_seqs[ti].features).unwrap_or(0.0)
            } else { 0.0 }
        } else { 0.0 };

        let ensemble_return = if total_w > 0.0 {
            (ridge_w * ridge_ret + lgbm_w * lgbm_ret + gru_w * gru_ret) / total_w
        } else {
            ridge_ret // fallback to ridge only
        };

        let (signal, ensemble_prob, confidence) = signal_from_returns(symbol, ensemble_return, asset_class);
        results.push((signal, ensemble_prob, confidence, ridge_ret, lgbm_ret, gru_ret, ensemble_return));
    }

    results
}

fn main() {
    // Lock file to prevent duplicate runs
    let lockfile = "/tmp/walkforward.lock";
    if std::path::Path::new(lockfile).exists() {
        // Check if the PID in the lockfile is still running
        if let Ok(pid_str) = std::fs::read_to_string(lockfile) {
            if let Ok(pid) = pid_str.trim().parse::<u32>() {
                let proc_path = format!("/proc/{}", pid);
                if std::path::Path::new(&proc_path).exists() {
                    eprintln!("ERROR: Another walk-forward backtest is already running (PID {})", pid);
                    std::process::exit(1);
                }
            }
        }
        // Stale lock file — remove it
        let _ = std::fs::remove_file(lockfile);
    }
    std::fs::write(lockfile, std::process::id().to_string()).expect("Failed to create lock file");

    // Ensure lock file is cleaned up on exit
    struct LockGuard;
    impl Drop for LockGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file("/tmp/walkforward.lock");
        }
    }
    let _lock = LockGuard;

    println!("\n══════════════════════════════════════");
    println!("  Walk-Forward Backtester");
    println!("  Zero Lookahead Bias");
    println!("══════════════════════════════════════\n");

    let database = db::Database::new("rust_invest.db").expect("Failed to open database");

    // Build market context
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    if let Ok(spy) = database.get_stock_history("SPY") {
        market_histories.insert("SPY".to_string(), spy.iter().map(|p| p.price).collect());
    }
    for ticker in features::MARKET_TICKERS {
        if let Ok(prices) = database.get_market_prices(ticker) {
            market_histories.insert(ticker.to_string(), prices);
        }
    }
    market_histories.insert("HY_SPREAD".to_string(), database.get_market_prices("HY_SPREAD").unwrap_or_default());
    market_histories.insert("BREAKEVEN_5Y".to_string(), database.get_market_prices("BREAKEVEN_5Y").unwrap_or_default());
    let market_context = features::build_market_context(&market_histories);

    let windows = define_test_windows();
    let mut all_signals: Vec<WFSignal> = Vec::new();
    let mut window_summaries: Vec<WindowSummary> = Vec::new();

    // ── Process stocks ──
    println!("━━━ STOCKS ━━━\n");
    for stock in stocks::STOCK_LIST {
        let points = match database.get_stock_history(stock.symbol) {
            Ok(p) if p.len() >= 300 => p,
            _ => continue,
        };
        process_asset(
            stock.symbol, "stock", &points, &market_context,
            features::sector_etf_for(stock.symbol),
            &windows, &mut all_signals, &mut window_summaries,
        );
    }

    // ── FX and Crypto descoped (separate project) ──
    println!("\n━━━ FX/CRYPTO SKIPPED (descoped) ━━━\n");

    // ── Deduplicate window summaries (aggregate across assets) ──
    let mut agg_windows: Vec<WindowSummary> = Vec::new();
    for w in &windows {
        let train_end_str = w.train_end.format("%Y-%m-%d").to_string();
        let test_start_str = w.test_start.format("%Y-%m-%d").to_string();
        let test_end_str = w.test_end.format("%Y-%m-%d").to_string();

        let window_sigs: Vec<&WFSignal> = all_signals.iter()
            .filter(|s| s.train_window_end == train_end_str)
            .collect();

        let buy_sigs: Vec<&&WFSignal> = window_sigs.iter().filter(|s| s.signal == "BUY").collect();
        let sell_sigs: Vec<&&WFSignal> = window_sigs.iter().filter(|s| s.signal == "SELL" || s.signal == "SHORT").collect();
        let buy_correct = buy_sigs.iter().filter(|s| s.was_correct == Some(true)).count();
        let sell_correct = sell_sigs.iter().filter(|s| s.was_correct == Some(true)).count();

        agg_windows.push(WindowSummary {
            train_end: train_end_str,
            test_start: test_start_str,
            test_end: test_end_str,
            signals_generated: window_sigs.len(),
            buy_accuracy: if !buy_sigs.is_empty() { buy_correct as f64 / buy_sigs.len() as f64 * 100.0 } else { 0.0 },
            sell_accuracy: if !sell_sigs.is_empty() { sell_correct as f64 / sell_sigs.len() as f64 * 100.0 } else { 0.0 },
        });
    }

    // ── Compute overall summary ──
    let total = all_signals.len();
    let buy_resolved: Vec<&WFSignal> = all_signals.iter().filter(|s| s.signal == "BUY" && s.was_correct.is_some()).collect();
    let sell_resolved: Vec<&WFSignal> = all_signals.iter().filter(|s| (s.signal == "SELL" || s.signal == "SHORT") && s.was_correct.is_some()).collect();
    let buy_correct = buy_resolved.iter().filter(|s| s.was_correct == Some(true)).count();
    let sell_correct = sell_resolved.iter().filter(|s| s.was_correct == Some(true)).count();

    // Expected value (directional returns in bps)
    let actionable: Vec<&WFSignal> = all_signals.iter()
        .filter(|s| s.signal != "HOLD" && s.pct_return.is_some())
        .collect();
    let returns: Vec<f64> = actionable.iter().map(|s| {
        let pct = s.pct_return.unwrap();
        match s.signal.as_str() {
            "BUY" => pct,
            _ => -pct, // SELL/SHORT: negative price = profit
        }
    }).collect();

    let expected_value_bps = if !returns.is_empty() {
        returns.iter().sum::<f64>() / returns.len() as f64 * 100.0
    } else { 0.0 };

    let winners: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losers: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let profit_factor = if losers > 0.0 { winners / losers } else if winners > 0.0 { 99.99 } else { 0.0 };

    // Sharpe ratio from daily equity returns
    let sharpe = compute_sharpe(&all_signals);
    let max_dd = compute_max_drawdown(&all_signals);

    let summary = Summary {
        total_signals: total,
        buy_accuracy: if !buy_resolved.is_empty() { buy_correct as f64 / buy_resolved.len() as f64 * 100.0 } else { 0.0 },
        sell_accuracy: if !sell_resolved.is_empty() { sell_correct as f64 / sell_resolved.len() as f64 * 100.0 } else { 0.0 },
        expected_value_bps,
        profit_factor,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_dd,
    };

    let output = WalkForwardOutput {
        generated_at: chrono::Utc::now().to_rfc3339(),
        windows: agg_windows,
        signals: all_signals,
        summary: summary.clone(),
    };

    // Write to file
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize");
    std::fs::create_dir_all("reports").ok();
    std::fs::write("reports/walkforward_backtest.json", &json).expect("Failed to write report");

    println!("\n══════════════════════════════════════");
    println!("  Walk-Forward Backtest Complete");
    println!("  Total signals: {}", total);
    println!("  BUY accuracy: {:.1}%", summary.buy_accuracy);
    println!("  SELL accuracy: {:.1}%", summary.sell_accuracy);
    println!("  Expected value: {:.1} bps/signal", summary.expected_value_bps);
    println!("  Profit factor: {:.2}x", summary.profit_factor);
    println!("  Sharpe ratio: {:.2}", summary.sharpe_ratio);
    println!("  Max drawdown: {:.1}%", summary.max_drawdown_pct);
    println!("  Output: reports/walkforward_backtest.json");
    println!("══════════════════════════════════════\n");
}

fn process_asset(
    symbol: &str,
    asset_class: &str,
    points: &[analysis::PricePoint],
    market_context: &features::MarketContext,
    sector_etf: Option<&str>,
    windows: &[TestWindow],
    all_signals: &mut Vec<WFSignal>,
    _window_summaries: &mut Vec<WindowSummary>,
) {
    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
    let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

    let samples = features::build_rich_features_ext(
        &prices, &volumes, &timestamps, Some(market_context), asset_class, sector_etf, None, None, None, None,
    );
    if samples.len() < 100 { return; }

    // samples[j] corresponds to prices[260+j] because features.rs:619 starts at index 260
    let sample_offset = 260_usize;

    // Build a date → price map for exit price lookup
    let date_prices: HashMap<String, f64> = timestamps.iter().zip(prices.iter())
        .map(|(ts, p)| (ts[..10.min(ts.len())].to_string(), *p))
        .collect();

    // Sort dates for next-day lookups
    let mut sorted_dates: Vec<String> = date_prices.keys().cloned().collect();
    sorted_dates.sort();

    let mut asset_signals = 0;

    for window in windows {
        // Find timestamp indices for train/test split
        let train_end_ts = match find_date_index(&timestamps, &window.test_start) {
            Some(idx) => idx,
            None => continue,
        };
        let test_end_ts = match find_date_index(&timestamps, &(window.test_end + chrono::Duration::days(1))) {
            Some(idx) => idx,
            None => timestamps.len(),
        };

        // Convert timestamp indices to sample indices (samples start at offset 260)
        let train_end_si = train_end_ts.saturating_sub(sample_offset);
        let test_end_si = test_end_ts.saturating_sub(sample_offset).min(samples.len());

        // Minimum 252 trading days of training data
        if train_end_si < 252 { continue; }
        if train_end_si >= samples.len() { continue; }

        // Embargo: skip 5 samples between train and test to prevent autocorrelation leakage
        const EMBARGO: usize = 5;
        let test_start_si = (train_end_si + EMBARGO).min(samples.len());
        if test_start_si >= test_end_si { continue; }

        let train_samples = &samples[..train_end_si];
        let test_samples = &samples[test_start_si..test_end_si];

        if test_samples.is_empty() { continue; }

        let predictions = train_and_predict(symbol, asset_class, train_samples, test_samples);

        for (i, (signal, ensemble_prob, confidence, ridge_ret, lgbm_ret, gru_ret, ensemble_ret)) in predictions.iter().enumerate() {
            let ts_idx = sample_offset + test_start_si + i;
            if ts_idx >= timestamps.len() { break; }
            let date = timestamps[ts_idx][..10.min(timestamps[ts_idx].len())].to_string();
            let entry_price = prices[ts_idx];

            // Find next trading day's price for exit
            let next_day_idx = sorted_dates.iter().position(|d| d > &date);
            let exit_price = next_day_idx.and_then(|idx| date_prices.get(&sorted_dates[idx]).copied());

            let (pct_return, was_correct) = if let Some(exit) = exit_price {
                // Deduct round-trip transaction cost (10 bps for stocks)
                let tx_cost_pct = backtest_compare::tx_cost(asset_class) * 100.0;
                let raw_pct = (exit - entry_price) / entry_price * 100.0;
                let pct = if signal.as_str() != "HOLD" { raw_pct - tx_cost_pct } else { raw_pct };
                let threshold = min_threshold(asset_class);
                let correct = match signal.as_str() {
                    "BUY" => pct > threshold,
                    "SELL" | "SHORT" => pct < -threshold,
                    "HOLD" => pct.abs() < threshold,
                    _ => false,
                };
                (Some(pct), Some(correct))
            } else {
                (None, None)
            };

            // Backward-compatible prob mapping for dashboard
            let mapped_prob = 0.5 + ensemble_ret.clamp(-5.0, 5.0) / 10.0;

            all_signals.push(WFSignal {
                date,
                asset: symbol.to_string(),
                asset_class: asset_class.to_string(),
                signal: signal.clone(),
                entry_price,
                exit_price,
                pct_return,
                was_correct,
                train_window_end: window.train_end.format("%Y-%m-%d").to_string(),
                buy_probability: *ensemble_prob,
                confidence: *confidence,
                ridge_return: *ridge_ret,
                lgbm_return: *lgbm_ret,
                gru_return: *gru_ret,
                ensemble_return: *ensemble_ret,
                // Legacy fields — mapped from ensemble return for dashboard compat
                linreg_prob: mapped_prob,
                logreg_prob: mapped_prob,
                gbt_prob: mapped_prob,
                lgbm_prob: mapped_prob,
                lstm_prob: mapped_prob,
                regime_prob: mapped_prob,
            });
            asset_signals += 1;
        }
    }

    if asset_signals > 0 {
        println!("  {} — {} out-of-sample signals", symbol, asset_signals);
    }
}

/// Compute annualised Sharpe ratio from the walk-forward signals
fn compute_sharpe(signals: &[WFSignal]) -> f64 {
    let actionable: Vec<&WFSignal> = signals.iter()
        .filter(|s| s.signal != "HOLD" && s.pct_return.is_some())
        .collect();
    if actionable.len() < 2 { return 0.0; }

    let returns: Vec<f64> = actionable.iter().map(|s| {
        let pct = s.pct_return.unwrap() / 100.0;
        match s.signal.as_str() {
            "BUY" => pct,
            _ => -pct,
        }
    }).collect();

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();
    if std < 1e-10 { return 0.0; }

    let rf_daily = 0.045 / 252.0;
    ((mean - rf_daily) / std) * (252.0_f64).sqrt()
}

/// Compute max drawdown from the walk-forward signals (cumulative equity)
fn compute_max_drawdown(signals: &[WFSignal]) -> f64 {
    let actionable: Vec<&WFSignal> = signals.iter()
        .filter(|s| s.signal != "HOLD" && s.pct_return.is_some())
        .collect();
    if actionable.is_empty() { return 0.0; }

    let mut equity = 100.0;
    let mut peak = equity;
    let mut max_dd = 0.0;

    for s in &actionable {
        let pct = s.pct_return.unwrap() / 100.0;
        let ret = match s.signal.as_str() {
            "BUY" => pct,
            _ => -pct,
        };
        equity *= 1.0 + ret;
        if equity > peak { peak = equity; }
        let dd = (peak - equity) / peak * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    max_dd
}
