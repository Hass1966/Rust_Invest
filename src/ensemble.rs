/// Ensemble System — Buy / Hold / Sell Signal with Confidence
/// ==========================================================
/// Four models in the ensemble:
///   1. Linear Regression (fast, interpretable baseline)
///   2. Logistic Regression (probabilistic classification)
///   3. Gradient Boosted Trees (nonlinear feature interactions)
///   4. LSTM (sequence patterns via candle-nn)
///
/// Two walk-forward modes:
///   walk_forward_samples() — takes pre-built Sample vectors from features.rs (for stocks)
///   walk_forward()         — takes raw prices/volumes, builds basic features (for crypto)

use crate::ml::{self, Sample};
use crate::gbt::{self, GBTConfig, TreeConfig, GradientBoostedClassifier};
use crate::lstm::{self, LSTMModelConfig, LSTMWalkForwardResult};
use serde::Deserialize;
use std::collections::HashMap;

// ════════════════════════════════════════
// Ensemble Overrides — per-asset model selection
// ════════════════════════════════════════

#[derive(Deserialize, Debug, Clone)]
pub struct EnsembleOverride {
    pub use_linreg: bool,
    pub use_logreg: bool,
    pub use_gbt: bool,
    #[serde(default)]
    pub reason: String,
}

impl Default for EnsembleOverride {
    fn default() -> Self {
        Self { use_linreg: true, use_logreg: true, use_gbt: true, reason: String::new() }
    }
}

/// Load ensemble overrides from config/ensemble_overrides.json
pub fn load_ensemble_overrides() -> HashMap<String, EnsembleOverride> {
    let path = "config/ensemble_overrides.json";
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            match serde_json::from_str::<HashMap<String, EnsembleOverride>>(&contents) {
                Ok(mut map) => {
                    // Remove comment key
                    map.remove("_comment");
                    println!("  [Ensemble] Loaded {} overrides from {}", map.len(), path);
                    map
                }
                Err(e) => {
                    println!("  [Ensemble] Failed to parse {}: {}", path, e);
                    HashMap::new()
                }
            }
        }
        Err(_) => {
            println!("  [Ensemble] No overrides file found, using defaults");
            HashMap::new()
        }
    }
}

/// Get the override for a specific symbol (falls back to "default" then all-enabled)
pub fn get_override(overrides: &HashMap<String, EnsembleOverride>, symbol: &str) -> EnsembleOverride {
    overrides.get(symbol)
        .or_else(|| overrides.get("default"))
        .cloned()
        .unwrap_or_default()
}

// ════════════════════════════════════════
// Walk-Forward on pre-built samples (RICH FEATURES)
// ════════════════════════════════════════

/// Walk-forward evaluation results
pub struct WalkForwardResult {
    pub symbol: String,
    pub linear_accuracy: f64,
    pub logistic_accuracy: f64,
    pub gbt_accuracy: f64,
    pub lstm_accuracy: f64,
    pub n_folds: usize,
    pub total_test_samples: usize,
    pub linear_recent: f64,
    pub logistic_recent: f64,
    pub gbt_recent: f64,
    pub lstm_recent: f64,
    pub final_linear_prob: f64,
    pub final_logistic_prob: f64,
    pub final_gbt_prob: f64,
    pub final_lstm_prob: f64,
    pub gbt_importance: Vec<(String, f64)>,
    pub n_features: usize,
    pub has_lstm: bool,
}

/// Walk-forward on pre-built Sample vectors (used with rich features)
pub fn walk_forward_samples(
    symbol: &str,
    samples: &[Sample],
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<WalkForwardResult> {
    if samples.len() < train_window + test_window + 10 {
        println!("  {} — not enough samples for walk-forward ({}, need {})",
            symbol, samples.len(), train_window + test_window);
        return None;
    }

    let n_features = samples[0].features.len();
    println!("  {} — walk-forward on {} samples × {} features", symbol, samples.len(), n_features);

    let mut total_lin_correct = 0_usize;
    let mut total_log_correct = 0_usize;
    let mut total_gbt_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;

    let mut last_lin_correct = 0_usize;
    let mut last_log_correct = 0_usize;
    let mut last_gbt_correct = 0_usize;
    let mut last_fold_size = 0_usize;

    let mut last_lin_prob = 0.5;
    let mut last_log_prob = 0.5;
    let mut last_gbt_prob = 0.5;
    let mut last_gbt_importance = Vec::new();

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        // Clone this fold's data for normalisation
        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;
        let test_len = test_end - train_end;

        let (train_data, test_data) = fold_samples.split_at_mut(train_len);

        let (means, stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &means, &stds);

        // Train all 3 pointwise models
        let mut lin = ml::LinearRegression::new(n_features);
        lin.train(train_data, 0.005, 3000);

        let mut log = ml::LogisticRegression::new(n_features);
        log.train(train_data, 0.01, 3000);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);

        let gbt_config = GBTConfig {
            n_trees: 80,
            learning_rate: 0.08,
            tree_config: TreeConfig {
                max_depth: 4,
                min_samples_leaf: 8,
                min_samples_split: 16,
            },
            subsample_ratio: 0.8,
            early_stopping_rounds: Some(8),
        };

        let gbt = GradientBoostedClassifier::train(
            x_t, y_t, Some(x_v), Some(y_v), gbt_config,
        );

        // Evaluate
        let mut fold_lin = 0;
        let mut fold_log = 0;
        let mut fold_gbt = 0;

        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            if (lin.predict(&s.features) > 0.0) == actual_up { fold_lin += 1; }
            if log.predict_direction(&s.features) == actual_up { fold_log += 1; }
            if gbt.predict_direction(&s.features) == actual_up { fold_gbt += 1; }
        }

        total_lin_correct += fold_lin;
        total_log_correct += fold_log;
        total_gbt_correct += fold_gbt;
        total_tested += test_len;
        n_folds += 1;

        last_lin_correct = fold_lin;
        last_log_correct = fold_log;
        last_gbt_correct = fold_gbt;
        last_fold_size = test_len;

        // Final predictions (sigmoid + clamp)
        if let Some(last_sample) = test_data.last() {
            let raw_lin = lin.predict(&last_sample.features);
            last_lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);
            last_log_prob = log.predict_probability(&last_sample.features).clamp(0.15, 0.85);
            last_gbt_prob = gbt.predict_proba(&last_sample.features).clamp(0.15, 0.85);
        }

        let feat_names: Vec<String> = {
            let rich = crate::features::feature_names();
            if n_features == rich.len() {
                rich
            } else {
                (0..n_features).map(|i| format!("Feature_{}", i)).collect()
            }
        };
        let feat_refs: Vec<&str> = feat_names.iter().map(|s| s.as_str()).collect();
        last_gbt_importance = gbt.feature_importance(&feat_refs);

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        return None;
    }

    let linear_acc = total_lin_correct as f64 / total_tested as f64 * 100.0;
    let logistic_acc = total_log_correct as f64 / total_tested as f64 * 100.0;
    let gbt_acc = total_gbt_correct as f64 / total_tested as f64 * 100.0;

    let linear_recent = last_lin_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let logistic_recent = last_log_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let gbt_recent = last_gbt_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    walk-forward: {} folds, {} test samples", n_folds, total_tested);
    println!("      LinReg: {:.1}% (recent: {:.1}%)", linear_acc, linear_recent);
    println!("      LogReg: {:.1}% (recent: {:.1}%)", logistic_acc, logistic_recent);
    println!("      GBT:    {:.1}% (recent: {:.1}%)", gbt_acc, gbt_recent);

    // Run LSTM walk-forward (separate because it uses sequences, not individual samples)
    let lstm_config = LSTMModelConfig {
        input_size: n_features,
        hidden_size: 32,
        seq_length: 20,
        learning_rate: 0.001,
        epochs: 40,
        batch_size: 32,
    };

    let lstm_result = lstm::walk_forward_lstm(
        symbol, samples, &lstm_config,
        train_window, test_window, step,
    );

    let (lstm_acc, lstm_recent_acc, lstm_prob, has_lstm) = match &lstm_result {
        Some(r) => (r.overall_accuracy, r.recent_accuracy, r.final_prob, true),
        None => (50.0, 50.0, 0.5, false),
    };

    Some(WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: linear_acc,
        logistic_accuracy: logistic_acc,
        gbt_accuracy: gbt_acc,
        lstm_accuracy: lstm_acc,
        n_folds,
        total_test_samples: total_tested,
        linear_recent,
        logistic_recent,
        gbt_recent,
        lstm_recent: lstm_recent_acc,
        final_linear_prob: last_lin_prob,
        final_logistic_prob: last_log_prob,
        final_gbt_prob: last_gbt_prob,
        final_lstm_prob: lstm_prob,
        gbt_importance: last_gbt_importance,
        n_features,
        has_lstm,
    })
}

// ════════════════════════════════════════
// Walk-Forward on raw prices (BASIC FEATURES — for crypto)
// ════════════════════════════════════════

/// Walk-forward using basic features built from raw price/volume data
pub fn walk_forward(
    symbol: &str,
    prices: &[f64],
    volumes: &[Option<f64>],
    _market_prices: Option<&[f64]>,
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<WalkForwardResult> {
    let samples = gbt::build_extended_features(prices, volumes);

    if samples.len() < train_window + test_window + 10 {
        println!("  {} — not enough data for walk-forward ({} samples, need {})",
            symbol, samples.len(), train_window + test_window);
        return None;
    }

    // For crypto with basic features, run without LSTM (not enough data for sequences)
    walk_forward_samples_no_lstm(symbol, &samples, train_window, test_window, step)
}

/// Walk-forward without LSTM (for crypto / basic features)
fn walk_forward_samples_no_lstm(
    symbol: &str,
    samples: &[Sample],
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<WalkForwardResult> {
    if samples.len() < train_window + test_window + 10 {
        return None;
    }

    let n_features = samples[0].features.len();
    println!("  {} — walk-forward on {} samples × {} features", symbol, samples.len(), n_features);

    let mut total_lin_correct = 0_usize;
    let mut total_log_correct = 0_usize;
    let mut total_gbt_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;

    let mut last_lin_correct = 0_usize;
    let mut last_log_correct = 0_usize;
    let mut last_gbt_correct = 0_usize;
    let mut last_fold_size = 0_usize;

    let mut last_lin_prob = 0.5;
    let mut last_log_prob = 0.5;
    let mut last_gbt_prob = 0.5;
    let mut last_gbt_importance = Vec::new();

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;
        let test_len = test_end - train_end;

        let (train_data, test_data) = fold_samples.split_at_mut(train_len);

        let (means, stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &means, &stds);

        let mut lin = ml::LinearRegression::new(n_features);
        lin.train(train_data, 0.005, 3000);

        let mut log = ml::LogisticRegression::new(n_features);
        log.train(train_data, 0.01, 3000);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);

        let gbt_config = GBTConfig {
            n_trees: 80,
            learning_rate: 0.08,
            tree_config: TreeConfig {
                max_depth: 4,
                min_samples_leaf: 8,
                min_samples_split: 16,
            },
            subsample_ratio: 0.8,
            early_stopping_rounds: Some(8),
        };

        let gbt = GradientBoostedClassifier::train(
            x_t, y_t, Some(x_v), Some(y_v), gbt_config,
        );

        let mut fold_lin = 0;
        let mut fold_log = 0;
        let mut fold_gbt = 0;

        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            if (lin.predict(&s.features) > 0.0) == actual_up { fold_lin += 1; }
            if log.predict_direction(&s.features) == actual_up { fold_log += 1; }
            if gbt.predict_direction(&s.features) == actual_up { fold_gbt += 1; }
        }

        total_lin_correct += fold_lin;
        total_log_correct += fold_log;
        total_gbt_correct += fold_gbt;
        total_tested += test_len;
        n_folds += 1;

        last_lin_correct = fold_lin;
        last_log_correct = fold_log;
        last_gbt_correct = fold_gbt;
        last_fold_size = test_len;

        if let Some(last_sample) = test_data.last() {
            let raw_lin = lin.predict(&last_sample.features);
            last_lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);
            last_log_prob = log.predict_probability(&last_sample.features).clamp(0.15, 0.85);
            last_gbt_prob = gbt.predict_proba(&last_sample.features).clamp(0.15, 0.85);
        }

        let feat_names: Vec<String> = (0..n_features).map(|i| format!("Feature_{}", i)).collect();
        let feat_refs: Vec<&str> = feat_names.iter().map(|s| s.as_str()).collect();
        last_gbt_importance = gbt.feature_importance(&feat_refs);

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        return None;
    }

    let linear_acc = total_lin_correct as f64 / total_tested as f64 * 100.0;
    let logistic_acc = total_log_correct as f64 / total_tested as f64 * 100.0;
    let gbt_acc = total_gbt_correct as f64 / total_tested as f64 * 100.0;

    let linear_recent = last_lin_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let logistic_recent = last_log_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let gbt_recent = last_gbt_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    walk-forward: {} folds, {} test samples", n_folds, total_tested);
    println!("      LinReg: {:.1}% (recent: {:.1}%)", linear_acc, linear_recent);
    println!("      LogReg: {:.1}% (recent: {:.1}%)", logistic_acc, logistic_recent);
    println!("      GBT:    {:.1}% (recent: {:.1}%)", gbt_acc, gbt_recent);

    Some(WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: linear_acc,
        logistic_accuracy: logistic_acc,
        gbt_accuracy: gbt_acc,
        lstm_accuracy: 50.0,
        n_folds,
        total_test_samples: total_tested,
        linear_recent,
        logistic_recent,
        gbt_recent,
        lstm_recent: 50.0,
        final_linear_prob: last_lin_prob,
        final_logistic_prob: last_log_prob,
        final_gbt_prob: last_gbt_prob,
        final_lstm_prob: 0.5,
        gbt_importance: last_gbt_importance,
        n_features,
        has_lstm: false,
    })
}

// ════════════════════════════════════════
// Ensemble & Signal
// ════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub symbol: String,
    pub signal: String,
    pub confidence: f64,
    pub ensemble_prob: f64,
    pub linear_prob: f64,
    pub logistic_prob: f64,
    pub gbt_prob: f64,
    pub lstm_prob: f64,
    pub linear_weight: f64,
    pub logistic_weight: f64,
    pub gbt_weight: f64,
    pub lstm_weight: f64,
    pub models_agree: usize,
    pub n_models: usize,
    pub walk_forward_accuracy: f64,
    pub signal_quality: String,
    pub current_price: f64,
    pub rsi: f64,
    pub sma_trend: String,
    pub has_lstm: bool,
}

/// Per-asset signal thresholds — noisy assets need higher confidence
pub fn get_signal_threshold(symbol: &str) -> (f64, f64) {
    match symbol {
        "TSLA" | "DIA" => (0.62, 0.38),
        "QQQ" | "AMZN" | "GOOGL" => (0.60, 0.40),
        "MSFT" => (0.58, 0.42),
        "META" | "NVDA" | "AAPL" | "SPY" => (0.57, 0.43),
        "USDJPY=X" | "AUDUSD=X" | "EURUSD=X" | "GBPUSD=X" | "USDCHF=X" => (0.55, 0.45),
        _ => (0.57, 0.43),
    }
}

pub fn ensemble_signal(
    symbol: &str,
    wf: &WalkForwardResult,
    current_price: f64,
    rsi: f64,
    sma_trend: &str,
) -> TradingSignal {
    ensemble_signal_with_override(symbol, wf, current_price, rsi, sma_trend, &EnsembleOverride::default())
}

pub fn ensemble_signal_with_override(
    symbol: &str,
    wf: &WalkForwardResult,
    current_price: f64,
    rsi: f64,
    sma_trend: &str,
    ov: &EnsembleOverride,
) -> TradingSignal {
    let best_overall = wf.linear_accuracy
        .max(wf.logistic_accuracy)
        .max(wf.gbt_accuracy)
        .max(if wf.has_lstm { wf.lstm_accuracy } else { 0.0 });

    let avg_accuracy = if wf.has_lstm {
        (wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy + wf.lstm_accuracy) / 4.0
    } else {
        (wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy) / 3.0
    };

    let (signal_quality, can_signal) = if best_overall >= 55.0 {
        ("HIGH".to_string(), true)
    } else if best_overall >= 52.0 && avg_accuracy >= 50.0 {
        ("MEDIUM".to_string(), true)
    } else if best_overall >= 50.5 {
        ("LOW".to_string(), true)
    } else {
        ("NO EDGE".to_string(), false)
    };

    // Accuracy-squared weighting, masked by ensemble override
    let lin_weight = if ov.use_linreg { (wf.linear_recent / 100.0).powi(2) } else { 0.0 };
    let log_weight = if ov.use_logreg { (wf.logistic_recent / 100.0).powi(2) } else { 0.0 };
    let gbt_weight = if ov.use_gbt { (wf.gbt_recent / 100.0).powi(2) } else { 0.0 };
    let lstm_weight = if wf.has_lstm { (wf.lstm_recent / 100.0).powi(2) } else { 0.0 };

    let total_weight = lin_weight + log_weight + gbt_weight + lstm_weight;

    let (lw, logw, gw, lstmw) = if total_weight > 0.0 {
        (
            lin_weight / total_weight,
            log_weight / total_weight,
            gbt_weight / total_weight,
            lstm_weight / total_weight,
        )
    } else if wf.has_lstm {
        (0.25, 0.25, 0.25, 0.25)
    } else {
        (1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0)
    };

    let ensemble_prob = lw * wf.final_linear_prob
        + logw * wf.final_logistic_prob
        + gw * wf.final_gbt_prob
        + lstmw * wf.final_lstm_prob;

    // Count agreement (only from enabled models)
    let mut ups = 0_usize;
    let mut n_models = 0_usize;
    if ov.use_linreg { n_models += 1; if wf.final_linear_prob > 0.5 { ups += 1; } }
    if ov.use_logreg { n_models += 1; if wf.final_logistic_prob > 0.5 { ups += 1; } }
    if ov.use_gbt { n_models += 1; if wf.final_gbt_prob > 0.5 { ups += 1; } }
    if wf.has_lstm { n_models += 1; if wf.final_lstm_prob > 0.5 { ups += 1; } }
    if n_models == 0 { n_models = 1; } // safety
    let models_agree = ups.max(n_models - ups);

    let signal = if !can_signal {
        "HOLD"
    } else {
        let (base_buy, base_sell) = get_signal_threshold(symbol);
        // Relax slightly when all models agree
        let (buy_thresh, sell_thresh) = if models_agree == n_models {
            (base_buy - 0.02, base_sell + 0.02)
        } else {
            (base_buy, base_sell)
        };

        if ensemble_prob > buy_thresh { "BUY" }
        else if ensemble_prob < sell_thresh { "SELL" }
        else { "HOLD" }
    };

    let confidence = if !can_signal {
        0.0
    } else {
        let raw = (ensemble_prob - 0.5).abs() * 200.0;
        let accuracy_cap = (avg_accuracy - 50.0).max(0.0) * 2.0;
        raw.min(accuracy_cap).min(best_overall - 50.0)
    };

    let wf_accuracy = wf.linear_accuracy * lw
        + wf.logistic_accuracy * logw
        + wf.gbt_accuracy * gw
        + wf.lstm_accuracy * lstmw;

    TradingSignal {
        symbol: wf.symbol.clone(),
        signal: signal.to_string(),
        confidence,
        ensemble_prob,
        linear_prob: wf.final_linear_prob,
        logistic_prob: wf.final_logistic_prob,
        gbt_prob: wf.final_gbt_prob,
        lstm_prob: wf.final_lstm_prob,
        linear_weight: lw,
        logistic_weight: logw,
        gbt_weight: gw,
        lstm_weight: lstmw,
        models_agree,
        n_models,
        walk_forward_accuracy: wf_accuracy,
        signal_quality,
        current_price,
        rsi,
        sma_trend: sma_trend.to_string(),
        has_lstm: wf.has_lstm,
    }
}

// ════════════════════════════════════════
// Console Output
// ════════════════════════════════════════

pub fn print_signals(signals: &[TradingSignal]) {
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                          TRADING SIGNALS — Ensemble Consensus (4 Models)                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<8} {:>8} {:>8} {:>6}  {:>6} {:>6} {:>6} {:>6}  {:>5} {:>7} {:>6} {:>7} ║",
        "Symbol", "Price", "Signal", "Conf%", "LinR", "LogR", "GBT", "LSTM", "Agree", "WF Acc", "RSI", "Quality");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for s in signals {
        let signal_icon = match s.signal.as_str() {
            "BUY" => "▲ BUY ",
            "SELL" => "▼ SELL",
            _ => "● HOLD",
        };
        let lstm_str = if s.has_lstm {
            format!("{:>5.1}%", s.lstm_prob * 100.0)
        } else {
            "  n/a".to_string()
        };
        println!("║ {:<8} {:>8.2} {} {:>5.1}%  {:>5.1}% {:>5.1}% {:>5.1}% {}  {}/{}   {:>5.1}% {:>5.1} {:>7} ║",
            s.symbol, s.current_price, signal_icon, s.confidence,
            s.linear_prob * 100.0, s.logistic_prob * 100.0, s.gbt_prob * 100.0,
            lstm_str,
            s.models_agree, s.n_models, s.walk_forward_accuracy, s.rsi, s.signal_quality);
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Quality: HIGH (>55%) = trustworthy | MEDIUM (52-55%) = marginal edge");
    println!("           LOW (50-52%) = barely above chance | NO EDGE (<50%) = forced HOLD");
    println!("  Conf% = distance from 50/50, capped by walk-forward accuracy\n");
}

// ════════════════════════════════════════
// HTML Dashboard
// ════════════════════════════════════════

pub fn signals_html(signals: &[TradingSignal]) -> String {
    let mut html = String::new();

    html.push_str("<h2>Trading Signals — Ensemble Consensus (4 Models)</h2>\n");
    html.push_str("<p>Four ML models (Linear Regression, Logistic Regression, Gradient Boosted Trees, LSTM) \
        trained with walk-forward evaluation on 80+ features (technical, volume, volatility, momentum, \
        calendar, market context, lagged, statistical), weighted by recent accuracy. \
        Market context features lagged by 1 day to prevent look-ahead bias.</p>\n");

    html.push_str("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:15px;margin:20px 0;'>\n");

    for s in signals {
        let (signal_color, signal_bg, signal_icon) = match s.signal.as_str() {
            "BUY"  => ("#00e676", "#1b3329", "▲"),
            "SELL" => ("#ff5252", "#3d1f1f", "▼"),
            _      => ("#ffd740", "#3d3520", "●"),
        };

        let (quality_color, quality_bg) = match s.signal_quality.as_str() {
            "HIGH"    => ("#00e676", "#1b3329"),
            "MEDIUM"  => ("#ffd740", "#3d3520"),
            "LOW"     => ("#ff9800", "#3d2e10"),
            _         => ("#ff5252", "#3d1f1f"),
        };

        let conf_width = (s.confidence * 3.0).min(100.0);

        let lstm_row = if s.has_lstm {
            format!(
                "<tr><td style='padding:4px;'>LSTM (Sequence)</td>\
                 <td style='text-align:right;padding:4px;'>{:.1}%</td>\
                 <td style='text-align:right;padding:4px;'>{:.0}%</td>\
                 <td style='text-align:right;padding:4px;'>{}</td></tr>\n",
                s.lstm_prob * 100.0, s.lstm_weight * 100.0,
                if s.lstm_prob > 0.5 { "<span class='positive'>▲</span>" } else { "<span class='negative'>▼</span>" },
            )
        } else {
            "<tr><td style='padding:4px;color:#555;'>LSTM</td>\
             <td style='text-align:right;padding:4px;color:#555;'>n/a</td>\
             <td style='text-align:right;padding:4px;color:#555;'>—</td>\
             <td style='text-align:right;padding:4px;color:#555;'>—</td></tr>\n".to_string()
        };

        html.push_str(&format!(r#"<div class='card' style='border-left:4px solid {};'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <h3 style='margin:0;display:inline;'>{}</h3>
      <span style='background:{};color:{};padding:2px 8px;border-radius:4px;font-size:10px;margin-left:8px;'>{}</span>
    </div>
    <span style='background:{};color:{};padding:6px 16px;border-radius:6px;font-size:18px;font-weight:bold;'>
      {} {}
    </span>
  </div>
  <div style='margin:12px 0;'>
    <div style='display:flex;justify-content:space-between;font-size:13px;color:#888;'>
      <span>Confidence</span><span>{:.1}%</span>
    </div>
    <div style='background:#0d1b2a;border-radius:4px;height:8px;margin-top:4px;'>
      <div style='width:{:.0}%;background:{};height:100%;border-radius:4px;'></div>
    </div>
  </div>
  <div style='font-size:13px;color:#aaa;margin:10px 0;'>
    <div>Price: <strong style='color:#e0e0e0;'>${:.2}</strong> &nbsp; RSI: <strong style='color:#e0e0e0;'>{:.1}</strong> &nbsp; Trend: <strong style='color:#e0e0e0;'>{}</strong></div>
  </div>
  <table style='width:100%;font-size:12px;margin-top:8px;'>
    <tr style='border-bottom:1px solid #1e3a5f;'>
      <th style='text-align:left;padding:4px;color:#888;font-weight:normal;'>Model</th>
      <th style='text-align:right;padding:4px;color:#888;font-weight:normal;'>P(Up)</th>
      <th style='text-align:right;padding:4px;color:#888;font-weight:normal;'>Weight</th>
      <th style='text-align:right;padding:4px;color:#888;font-weight:normal;'>Vote</th>
    </tr>
    <tr><td style='padding:4px;'>Linear Regression</td>
        <td style='text-align:right;padding:4px;'>{:.1}%</td>
        <td style='text-align:right;padding:4px;'>{:.0}%</td>
        <td style='text-align:right;padding:4px;'>{}</td></tr>
    <tr><td style='padding:4px;'>Logistic Regression</td>
        <td style='text-align:right;padding:4px;'>{:.1}%</td>
        <td style='text-align:right;padding:4px;'>{:.0}%</td>
        <td style='text-align:right;padding:4px;'>{}</td></tr>
    <tr style='background:#16213e;'><td style='padding:4px;'>Gradient Boosted Trees</td>
        <td style='text-align:right;padding:4px;'>{:.1}%</td>
        <td style='text-align:right;padding:4px;'>{:.0}%</td>
        <td style='text-align:right;padding:4px;'>{}</td></tr>
    {}
    <tr style='border-top:2px solid #1e3a5f;'>
      <td style='padding:6px 4px;font-weight:bold;'>Ensemble</td>
      <td style='text-align:right;padding:6px 4px;font-weight:bold;color:{};'>{:.1}%</td>
      <td></td>
      <td style='text-align:right;padding:6px 4px;font-weight:bold;color:{};'>{}/{} agree</td>
    </tr>
  </table>
  <div style='font-size:11px;color:#555;margin-top:8px;'>Walk-forward accuracy: {:.1}%</div>
</div>
"#,
            signal_color,
            s.symbol,
            quality_bg, quality_color, s.signal_quality,
            signal_bg, signal_color,
            signal_icon, s.signal,
            s.confidence, conf_width, signal_color,
            s.current_price, s.rsi, s.sma_trend,
            s.linear_prob * 100.0, s.linear_weight * 100.0,
            if s.linear_prob > 0.5 { "<span class='positive'>▲</span>" } else { "<span class='negative'>▼</span>" },
            s.logistic_prob * 100.0, s.logistic_weight * 100.0,
            if s.logistic_prob > 0.5 { "<span class='positive'>▲</span>" } else { "<span class='negative'>▼</span>" },
            s.gbt_prob * 100.0, s.gbt_weight * 100.0,
            if s.gbt_prob > 0.5 { "<span class='positive'>▲</span>" } else { "<span class='negative'>▼</span>" },
            lstm_row,
            signal_color, s.ensemble_prob * 100.0,
            signal_color, s.models_agree, s.n_models,
            s.walk_forward_accuracy,
        ));
    }

    html.push_str("</div>\n");

    // Summary table
    html.push_str("<h3>Signal Summary</h3>\n<table>\n");
    html.push_str("<tr><th>Symbol</th><th>Signal</th><th>Confidence</th>\
        <th>P(Up)</th><th>Agreement</th><th>WF Accuracy</th><th>Quality</th><th>RSI</th><th>Trend</th></tr>\n");

    for s in signals {
        let sig_class = match s.signal.as_str() {
            "BUY" => "signal-bullish",
            "SELL" => "signal-bearish",
            _ => "signal-neutral",
        };
        let qual_class = match s.signal_quality.as_str() {
            "HIGH" => "signal-bullish",
            "MEDIUM" => "signal-neutral",
            _ => "signal-bearish",
        };

        html.push_str(&format!(
            "<tr><td>{}</td><td><span class='{}'>{}</span></td>\
             <td>{:.1}%</td><td>{:.1}%</td><td>{}/{}</td>\
             <td>{:.1}%</td><td><span class='{}'>{}</span></td><td>{:.1}</td><td>{}</td></tr>\n",
            s.symbol, sig_class, s.signal,
            s.confidence, s.ensemble_prob * 100.0, s.models_agree, s.n_models,
            s.walk_forward_accuracy, qual_class, s.signal_quality,
            s.rsi, s.sma_trend,
        ));
    }
    html.push_str("</table>\n");

    html.push_str("<p><em>Signal thresholds: BUY when P(Up) &gt; 55-60%, SELL when P(Up) &lt; 40-45% \
        (adaptive based on model agreement). Confidence = distance from 50/50, \
        capped by walk-forward accuracy. Market context lagged 1 day (no look-ahead bias). \
        LSTM uses 20-day sequences for temporal pattern detection. \
        Assets with NO EDGE are forced to HOLD. Not financial advice.</em></p>\n");

    html
}
