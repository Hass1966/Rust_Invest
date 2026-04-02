//! Walk-Forward Backtester
//!
//! Generates out-of-sample signals using expanding-window retraining.
//! For each quarterly test window, models are trained ONLY on prior data,
//! ensuring zero lookahead bias.
//!
//! Output: reports/walkforward_backtest.json

use rust_invest::{analysis, db, features, ml, gbt, ensemble, stocks};
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
    linreg_prob: f64,
    logreg_prob: f64,
    gbt_prob: f64,
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
    // Q1 2022 through Q4 2025: 16 quarterly windows
    let quarters = [
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

/// Compute binary cross-entropy (log-loss) for a single prediction
fn log_loss_single(predicted: f64, actual_up: bool) -> f64 {
    let p = predicted.clamp(1e-7, 1.0 - 1e-7);
    let y = if actual_up { 1.0 } else { 0.0 };
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// Generate a signal from 3 model probabilities using inverse-loss weighted voting
fn signal_from_probs(
    symbol: &str,
    lin_prob: f64,
    log_prob: f64,
    gbt_prob: f64,
    model_weights: &[f64; 3], // [lin_w, log_w, gbt_w] normalised to sum to 1
) -> (String, f64) {
    let ensemble_prob = model_weights[0] * lin_prob
        + model_weights[1] * log_prob
        + model_weights[2] * gbt_prob;

    let (buy_thresh, sell_thresh) = ensemble::get_signal_threshold(symbol);
    let short_thresh = 1.0 - buy_thresh;

    let mut ups = 0usize;
    if lin_prob > 0.5 { ups += 1; }
    if log_prob > 0.5 { ups += 1; }
    if gbt_prob > 0.5 { ups += 1; }
    let downs = 3 - ups;

    let signal = if ensemble_prob > buy_thresh {
        "BUY"
    } else if ensemble_prob < short_thresh && downs >= 2 && (ensemble_prob - 0.5).abs() > (buy_thresh - 0.5) {
        "SHORT"
    } else if ensemble_prob < sell_thresh && downs >= 2 {
        "SELL"
    } else {
        "HOLD"
    };

    (signal.to_string(), ensemble_prob)
}

/// Minimum move threshold for "correct" classification
fn min_threshold(asset_class: &str) -> f64 {
    match asset_class {
        "crypto" => 1.0,
        "fx" => 0.2,
        _ => 0.5,
    }
}

/// Train 3 models on a slice of samples and generate predictions for test samples.
/// Uses inverse validation log-loss weighting for ensemble combination.
/// Returns Vec<(signal, lin_prob, log_prob, gbt_prob)> for each test sample.
fn train_and_predict(
    symbol: &str,
    train_samples: &[ml::Sample],
    test_samples: &[ml::Sample],
) -> Vec<(String, f64, f64, f64)> {
    if train_samples.len() < 50 || test_samples.is_empty() {
        return Vec::new();
    }

    let n_feat = train_samples[0].features.len();

    // Normalise training data
    let mut train_data = train_samples.to_vec();
    let (means, stds) = ml::normalise(&mut train_data);

    // Class weights
    let vol_threshold = features::compute_volatility_threshold(&train_data);
    let (w_down, w_up) = features::compute_class_weights(&train_data, vol_threshold);

    // Recency weights with class adjustment
    let recency = ensemble::compute_recency_weights(train_data.len());
    let weights: Vec<f64> = train_data.iter().zip(recency.iter()).map(|(s, &rw)| {
        let cw = if s.label > 0.0 { w_up } else { w_down };
        rw * cw
    }).collect();

    // Split: train on first 85%, validate on last 15% for model weighting
    let val_start = (train_data.len() as f64 * 0.85) as usize;

    // Train LinReg on first 85%, reserve last 15% for validation log-loss weighting
    let mut lin = ml::LinearRegression::new(n_feat);
    lin.train_weighted(&train_data[..val_start], Some(&weights[..val_start]), 0.005, 3000);

    // Train LogReg
    let mut log = ml::LogisticRegression::new(n_feat);
    log.train_weighted(&train_data[..val_start], Some(&weights[..val_start]), 0.01, 3000);

    // Train GBT (with its own early-stopping validation)
    let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
    let y_train: Vec<f64> = train_data.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
    let (x_t, x_v) = x_train.split_at(val_start);
    let (y_t, y_v) = y_train.split_at(val_start);
    let gbt_recency = &weights[..x_t.len()];
    let gbt_config = gbt::GBTConfig {
        n_trees: 80,
        learning_rate: 0.08,
        tree_config: gbt::TreeConfig { max_depth: 4, min_samples_leaf: 8, min_samples_split: 16 },
        subsample_ratio: 0.8,
        early_stopping_rounds: Some(8),
    };
    let gbt_model = gbt::GradientBoostedClassifier::train_weighted(
        x_t, y_t, Some(gbt_recency), Some(x_v), Some(y_v), gbt_config
    );

    // ── Compute inverse validation log-loss weights ──
    // Evaluate each model on the held-out validation set (last 15% of training)
    let val_data = &train_data[val_start..];
    let mut lin_loss_sum = 0.0_f64;
    let mut log_loss_sum = 0.0_f64;
    let mut gbt_loss_sum = 0.0_f64;

    for sample in val_data {
        let actual_up = sample.label > 0.0;
        let lin_pred = lin.predict(&sample.features);
        let lin_prob = (1.0 / (1.0 + (-lin_pred).exp())).clamp(0.01, 0.99);
        let log_prob = log.predict_probability(&sample.features).clamp(0.01, 0.99);
        let gbt_prob = gbt_model.predict_proba(&sample.features).clamp(0.01, 0.99);

        lin_loss_sum += log_loss_single(lin_prob, actual_up);
        log_loss_sum += log_loss_single(log_prob, actual_up);
        gbt_loss_sum += log_loss_single(gbt_prob, actual_up);
    }

    let n_val = val_data.len().max(1) as f64;
    let lin_avg_loss = (lin_loss_sum / n_val).max(0.01);
    let log_avg_loss = (log_loss_sum / n_val).max(0.01);
    let gbt_avg_loss = (gbt_loss_sum / n_val).max(0.01);

    // Inverse-loss weights, normalised to sum to 1
    let inv_lin = 1.0 / lin_avg_loss;
    let inv_log = 1.0 / log_avg_loss;
    let inv_gbt = 1.0 / gbt_avg_loss;
    let inv_total = inv_lin + inv_log + inv_gbt;
    let model_weights = [inv_lin / inv_total, inv_log / inv_total, inv_gbt / inv_total];

    // Predict on test samples using the 85%-trained models with inverse-loss weights
    let mut results = Vec::with_capacity(test_samples.len());
    for sample in test_samples {
        let mut features = sample.features.clone();
        for j in 0..features.len() {
            if j < means.len() && j < stds.len() && stds[j] > 1e-10 {
                features[j] = (features[j] - means[j]) / stds[j];
            }
        }

        let lin_pred = lin.predict(&features);
        let lin_prob = 1.0 / (1.0 + (-lin_pred).exp());
        let log_prob = log.predict_probability(&features);
        let gbt_prob = gbt_model.predict_proba(&features);

        let (signal, _) = signal_from_probs(symbol, lin_prob, log_prob, gbt_prob, &model_weights);
        results.push((signal, lin_prob, log_prob, gbt_prob));
    }

    results
}

fn main() {
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

    // ── Process FX ──
    println!("\n━━━ FX ━━━\n");
    for fx in stocks::FX_LIST {
        let points = match database.get_fx_history(fx.symbol) {
            Ok(p) if p.len() >= 300 => p,
            _ => continue,
        };
        process_asset(
            fx.symbol, "fx", &points, &market_context,
            Some(fx.symbol),
            &windows, &mut all_signals, &mut window_summaries,
        );
    }

    // ── Process crypto ──
    println!("\n━━━ CRYPTO ━━━\n");
    if let Ok(coin_ids) = database.get_all_coin_ids() {
        for coin_id in &coin_ids {
            let points = match database.get_coin_history(coin_id) {
                Ok(p) if p.len() >= 300 => p,
                _ => continue,
            };
            process_asset(
                coin_id, "crypto", &points, &market_context,
                None,
                &windows, &mut all_signals, &mut window_summaries,
            );
        }
    }

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

    // Build a date → price map for exit price lookup
    let date_prices: HashMap<String, f64> = timestamps.iter().zip(prices.iter())
        .map(|(ts, p)| (ts[..10.min(ts.len())].to_string(), *p))
        .collect();

    // Sort dates for next-day lookups
    let mut sorted_dates: Vec<String> = date_prices.keys().cloned().collect();
    sorted_dates.sort();

    let mut asset_signals = 0;

    for window in windows {
        // Find sample indices for train/test split
        let train_end_idx = match find_date_index(&timestamps, &window.test_start) {
            Some(idx) => idx,
            None => continue,
        };
        let test_end_idx = match find_date_index(&timestamps, &(window.test_end + chrono::Duration::days(1))) {
            Some(idx) => idx,
            None => timestamps.len(),
        };

        // Minimum 252 trading days of training data
        if train_end_idx < 252 { continue; }
        if train_end_idx >= samples.len() { continue; }
        let test_end_capped = test_end_idx.min(samples.len());
        if train_end_idx >= test_end_capped { continue; }

        let train_samples = &samples[..train_end_idx];
        let test_samples = &samples[train_end_idx..test_end_capped];

        if test_samples.is_empty() { continue; }

        let predictions = train_and_predict(symbol, train_samples, test_samples);

        for (i, (signal, lin_prob, log_prob, gbt_prob)) in predictions.iter().enumerate() {
            let sample_idx = train_end_idx + i;
            if sample_idx >= timestamps.len() { break; }
            let date = timestamps[sample_idx][..10.min(timestamps[sample_idx].len())].to_string();
            let entry_price = prices[sample_idx];

            // Find next trading day's price for exit
            let next_day_idx = sorted_dates.iter().position(|d| d > &date);
            let exit_price = next_day_idx.and_then(|idx| date_prices.get(&sorted_dates[idx]).copied());

            let (pct_return, was_correct) = if let Some(exit) = exit_price {
                let pct = (exit - entry_price) / entry_price * 100.0;
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
                linreg_prob: *lin_prob,
                logreg_prob: *log_prob,
                gbt_prob: *gbt_prob,
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
