//! Walk-Forward Backtester
//!
//! Generates out-of-sample signals using expanding-window retraining.
//! For each quarterly test window, models are trained ONLY on prior data,
//! ensuring zero lookahead bias.
//!
//! Output: reports/walkforward_backtest.json

use rust_invest::{analysis, db, features, ml, gbt, lgbm, ensemble, stocks, lstm, regime};
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

/// Generate a signal from 6 model probabilities using inverse-loss weighted voting
/// Now includes quality gating, dead zone, tighter SHORT rules, and consensus adjustment
fn signal_from_probs(
    symbol: &str,
    probs: &[f64; 6], // [lin, log, gbt, lgbm, lstm, regime]
    weights: &[f64; 6], // normalised to sum to 1
    accuracies: &[f64; 6], // per-model validation accuracy (0-100 scale)
) -> (String, f64) {
    let ensemble_prob: f64 = probs.iter().zip(weights.iter())
        .map(|(p, w)| p * w)
        .sum();

    // ── Quality gating: skip signals when models have no edge ──
    let active_accuracies: Vec<f64> = accuracies.iter().zip(weights)
        .filter(|(_, &w)| w > 0.0)
        .map(|(&a, _)| a)
        .collect();
    let best = active_accuracies.iter().cloned().fold(0.0_f64, f64::max);
    let avg = if !active_accuracies.is_empty() {
        active_accuracies.iter().sum::<f64>() / active_accuracies.len() as f64
    } else { 0.0 };

    if best < 52.0 || (best < 55.0 && avg < 50.0) {
        return ("HOLD".to_string(), ensemble_prob);
    }

    // ── Dead zone: marginal probabilities → HOLD ──
    if ensemble_prob > 0.47 && ensemble_prob < 0.53 {
        return ("HOLD".to_string(), ensemble_prob);
    }

    let n_models = probs.iter().zip(weights.iter())
        .filter(|(_, &w)| w > 0.0)
        .count();
    let ups = probs.iter().zip(weights.iter())
        .filter(|(&p, &w)| w > 0.0 && p > 0.5)
        .count();
    let downs = n_models - ups;
    let majority = (n_models + 1) / 2;

    // ── Consensus threshold adjustment: relax when all models agree ──
    let (base_buy, base_sell) = ensemble::get_signal_threshold(symbol);
    let (buy_thresh, sell_thresh) = if ups == n_models && n_models >= 3 {
        (base_buy - 0.02, base_sell + 0.02)
    } else if downs == n_models && n_models >= 3 {
        (base_buy - 0.02, base_sell + 0.02)
    } else {
        (base_buy, base_sell)
    };

    // ── SHORT: require very strong conviction (supermajority + tighter threshold) ──
    let short_thresh = 1.0 - buy_thresh - 0.03; // 0.42 instead of 0.45
    let supermajority = ((n_models as f64 * 2.0 / 3.0).ceil() as usize).max(majority);

    let signal = if ensemble_prob > buy_thresh && ups >= majority {
        "BUY"
    } else if ensemble_prob < short_thresh && downs >= supermajority
        && (ensemble_prob - 0.5).abs() > (buy_thresh - 0.5) {
        "SHORT"
    } else if ensemble_prob < sell_thresh && downs >= majority {
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

/// Train 6 models on a slice of samples and generate predictions for test samples.
/// Models: LinReg, LogReg, GBT, LightGBM, LSTM, RegimeEnsemble
/// Uses inverse validation log-loss weighting for ensemble combination.
/// Returns (Vec<(signal, ensemble_prob, lin_prob, log_prob, gbt_prob, lgbm_prob, lstm_prob, regime_prob)>, model_accuracies)
fn train_and_predict(
    symbol: &str,
    train_samples: &[ml::Sample],
    test_samples: &[ml::Sample],
) -> (Vec<(String, f64, f64, f64, f64, f64, f64, f64)>, [f64; 6]) {
    if train_samples.len() < 50 || test_samples.is_empty() {
        return (Vec::new(), [0.0; 6]);
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

    // ── Model 1: LinReg ──
    let mut lin = ml::LinearRegression::new(n_feat);
    lin.train_weighted(&train_data[..val_start], Some(&weights[..val_start]), 0.005, 3000);

    // ── Model 2: LogReg ──
    let mut log = ml::LogisticRegression::new(n_feat);
    log.train_weighted(&train_data[..val_start], Some(&weights[..val_start]), 0.01, 3000);

    // ── Model 3: GBT ──
    let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
    let y_train: Vec<f64> = train_data.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
    let (x_t, x_v) = x_train.split_at(val_start);
    let (y_t, y_v) = y_train.split_at(val_start);
    let gbt_recency = &weights[..x_t.len()];
    let gbt_config = gbt::GBTConfig::default();
    let gbt_model = gbt::GradientBoostedClassifier::train_weighted(
        x_t, y_t, Some(gbt_recency), Some(x_v), Some(y_v), gbt_config
    );

    // ── Model 4: LightGBM ──
    let lgbm_config = lgbm::LGBMConfig::default();
    let lgbm_model = lgbm::LightGBMClassifier::train(
        x_t, y_t, Some(&weights[..x_t.len()]),
        Some(x_v), Some(y_v), &lgbm_config,
    ).ok();

    // ── Model 5: LSTM (top 30 features, reduced epochs for speed) ──
    let feature_names_list = features::active_feature_names();
    let feature_name_refs: Vec<&str> = feature_names_list.iter().map(|s| s.as_str()).collect();
    // Use LightGBM importance if available, fall back to GBT importance
    let gbt_importance = if let Some(ref lgbm) = lgbm_model {
        lgbm.feature_importance(&feature_name_refs)
    } else {
        gbt_model.feature_importance(&feature_name_refs)
    };
    let top_feature_indices: Vec<usize> = gbt_importance.iter()
        .take(30.min(gbt_importance.len()))
        .filter_map(|(name, _)| feature_names_list.iter().position(|n| n == name))
        .collect();
    let lstm_indices = if top_feature_indices.len() >= 20 { Some(top_feature_indices.as_slice()) } else { None };
    let lstm_input_size = lstm_indices.map_or(n_feat, |idx| idx.len());

    let lstm_model = lstm::LSTMModel::new(lstm::LSTMModelConfig {
        input_size: lstm_input_size,
        hidden_size: 64,
        seq_length: 10,
        learning_rate: 0.0005,
        epochs: 20, // reduced from 50 for walk-forward speed
        batch_size: 32,
    }).ok().and_then(|mut model| {
        let lstm_val_start = val_start.saturating_sub(10); // need seq_length buffer
        if lstm_val_start < 50 { return None; }
        match model.train(&train_data[..lstm_val_start], &train_data[lstm_val_start..], lstm_indices) {
            Ok(result) if result.final_val_loss.is_finite() => Some(model),
            _ => None,
        }
    });

    // ── Model 6: RegimeEnsemble ──
    let regime_model = if train_data.len() >= 100 {
        Some(regime::RegimeEnsemble::train(&train_data[..val_start]))
    } else {
        None
    };

    // ── Compute inverse validation log-loss weights for all 6 models ──
    let val_data = &train_data[val_start..];
    let mut losses = [0.0_f64; 6]; // [lin, log, gbt, lgbm, lstm, regime]
    let mut correct_counts = [0usize; 6];
    let mut total_counts = [0usize; 6];
    let mut lstm_correct = 0usize;
    let mut lstm_total = 0usize;

    // Build LSTM sequences from validation data for batch evaluation
    let val_seqs = lstm::build_sequences_with_subset(val_data, 10, lstm_indices);

    for (vi, sample) in val_data.iter().enumerate() {
        let actual_up = sample.label > 0.0;
        let lin_pred = lin.predict(&sample.features);
        let lin_prob = (1.0 / (1.0 + (-lin_pred).exp())).clamp(0.01, 0.99);
        let log_prob = log.predict_probability(&sample.features).clamp(0.01, 0.99);
        let gbt_prob = gbt_model.predict_proba(&sample.features).clamp(0.01, 0.99);

        losses[0] += log_loss_single(lin_prob, actual_up);
        losses[1] += log_loss_single(log_prob, actual_up);
        losses[2] += log_loss_single(gbt_prob, actual_up);
        // Track per-model accuracy
        if (lin_prob > 0.5) == actual_up { correct_counts[0] += 1; }
        total_counts[0] += 1;
        if (log_prob > 0.5) == actual_up { correct_counts[1] += 1; }
        total_counts[1] += 1;
        if (gbt_prob > 0.5) == actual_up { correct_counts[2] += 1; }
        total_counts[2] += 1;

        // LightGBM
        if let Some(ref model) = lgbm_model {
            let lgbm_prob = model.predict_proba(&sample.features).clamp(0.01, 0.99);
            losses[3] += log_loss_single(lgbm_prob, actual_up);
            if (lgbm_prob > 0.5) == actual_up { correct_counts[3] += 1; }
            total_counts[3] += 1;
        }

        // LSTM: only evaluate if we have a matching sequence
        if let Some(ref model) = lstm_model {
            if vi < val_seqs.len() {
                if let Ok(lstm_p) = model.predict_proba(&val_seqs[vi].features) {
                    let lstm_prob = lstm_p.clamp(0.01, 0.99);
                    losses[4] += log_loss_single(lstm_prob, actual_up);
                    if (lstm_prob > 0.5) == actual_up { lstm_correct += 1; correct_counts[4] += 1; }
                    lstm_total += 1;
                    total_counts[4] += 1;
                }
            }
        }

        // RegimeEnsemble
        if let Some(ref model) = regime_model {
            let regime_prob = model.predict_proba(&sample.features).clamp(0.01, 0.99);
            losses[5] += log_loss_single(regime_prob, actual_up);
            if (regime_prob > 0.5) == actual_up { correct_counts[5] += 1; }
            total_counts[5] += 1;
        }
    }

    let n_val = val_data.len().max(1) as f64;
    let lstm_accuracy = if lstm_total > 0 { lstm_correct as f64 / lstm_total as f64 * 100.0 } else { 0.0 };

    // Compute per-model validation accuracies (0-100 scale)
    let model_accuracies: [f64; 6] = std::array::from_fn(|i| {
        if total_counts[i] > 0 { correct_counts[i] as f64 / total_counts[i] as f64 * 100.0 } else { 0.0 }
    });
    // Gate LSTM: only include if accuracy >= 54%
    let lstm_useful = lstm_model.is_some() && lstm_accuracy >= 54.0;

    // Compute inverse-loss weights for all active models
    let mut inv_weights = [0.0_f64; 6];
    for i in 0..3 {
        inv_weights[i] = 1.0 / (losses[i] / n_val).max(0.01);
    }
    // LightGBM
    if lgbm_model.is_some() {
        inv_weights[3] = 1.0 / (losses[3] / n_val).max(0.01);
    }
    if lstm_useful {
        let lstm_n = lstm_total.max(1) as f64;
        inv_weights[4] = 1.0 / (losses[4] / lstm_n).max(0.01);
        // Bonus if accuracy >= 58%
        if lstm_accuracy >= 58.0 { inv_weights[4] *= 1.5; }
    }
    if regime_model.is_some() {
        inv_weights[5] = 1.0 / (losses[5] / n_val).max(0.01);
    }
    let inv_total: f64 = inv_weights.iter().sum();
    let model_weights: [f64; 6] = if inv_total > 0.0 {
        [
            inv_weights[0] / inv_total,
            inv_weights[1] / inv_total,
            inv_weights[2] / inv_total,
            inv_weights[3] / inv_total,
            inv_weights[4] / inv_total,
            inv_weights[5] / inv_total,
        ]
    } else {
        [1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, 0.0, 0.0]
    };

    // ── Generate predictions for test samples ──
    // Pre-build LSTM sequences from test samples
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
    let test_seqs = lstm::build_sequences_with_subset(&test_norm, 10, lstm_indices);

    let mut results = Vec::with_capacity(test_samples.len());
    for (ti, sample) in test_norm.iter().enumerate() {
        let lin_pred = lin.predict(&sample.features);
        let lin_prob = (1.0 / (1.0 + (-lin_pred).exp())).clamp(0.15, 0.85);
        let log_prob = log.predict_probability(&sample.features).clamp(0.15, 0.85);
        let gbt_prob = gbt_model.predict_proba(&sample.features).clamp(0.15, 0.85);

        let lgbm_prob = if let Some(ref model) = lgbm_model {
            model.predict_proba(&sample.features).clamp(0.15, 0.85)
        } else { 0.5 };

        let lstm_prob = if lstm_useful {
            if let Some(ref model) = lstm_model {
                if ti < test_seqs.len() {
                    model.predict_proba(&test_seqs[ti].features).unwrap_or(0.5).clamp(0.15, 0.85)
                } else { 0.5 }
            } else { 0.5 }
        } else { 0.5 };

        let regime_prob = if let Some(ref model) = regime_model {
            model.predict_proba(&sample.features).clamp(0.15, 0.85)
        } else { 0.5 };

        let probs = [lin_prob, log_prob, gbt_prob, lgbm_prob, lstm_prob, regime_prob];
        let (signal, ensemble_prob) = signal_from_probs(symbol, &probs, &model_weights, &model_accuracies);
        results.push((signal, ensemble_prob, lin_prob, log_prob, gbt_prob, lgbm_prob, lstm_prob, regime_prob));
    }

    (results, model_accuracies)
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

        let (predictions, _model_accuracies) = train_and_predict(symbol, train_samples, test_samples);

        for (i, (signal, ensemble_prob, lin_prob, log_prob, gbt_prob, lgbm_prob, lstm_prob, regime_prob)) in predictions.iter().enumerate() {
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

            // Confidence: distance from 0.5, capped at 1.0
            let confidence = ((*ensemble_prob - 0.5).abs() * 2.0).min(1.0);

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
                confidence,
                linreg_prob: *lin_prob,
                logreg_prob: *log_prob,
                gbt_prob: *gbt_prob,
                lgbm_prob: *lgbm_prob,
                lstm_prob: *lstm_prob,
                regime_prob: *regime_prob,
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
