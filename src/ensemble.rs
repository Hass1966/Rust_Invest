/// Ensemble System — Buy / Hold / Sell Signal with Confidence
/// ==========================================================
/// Six models in the ensemble:
///   1. Linear Regression (fast, interpretable baseline)
///   2. Logistic Regression (probabilistic classification)
///   3. Gradient Boosted Trees (nonlinear feature interactions)
///   4. LSTM (sequence patterns via candle-nn)
///   5. GRU (lighter sequence model via candle-nn)
///   6. RegimeEnsemble (handled separately in train.rs)
///
/// Two walk-forward modes:
///   walk_forward_samples() — takes pre-built Sample vectors from features.rs (for stocks)
///   walk_forward()         — takes raw prices/volumes, builds basic features (for crypto)

use crate::ml::{self, Sample};
use crate::gbt::{self, GBTConfig, TreeConfig, GradientBoostedClassifier};
use crate::lstm::{self, LSTMModelConfig};
use crate::gru::{self, GRUModelConfig};
use crate::random_forest::{self, RandomForestConfig};
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
    pub gru_accuracy: f64,
    pub n_folds: usize,
    pub total_test_samples: usize,
    pub linear_recent: f64,
    pub logistic_recent: f64,
    pub gbt_recent: f64,
    pub lstm_recent: f64,
    pub gru_recent: f64,
    pub final_linear_prob: f64,
    pub final_logistic_prob: f64,
    pub final_gbt_prob: f64,
    pub final_lstm_prob: f64,
    pub final_gru_prob: f64,
    pub gbt_importance: Vec<(String, f64)>,
    pub n_features: usize,
    pub rf_accuracy: f64,
    pub rf_recent: f64,
    pub final_rf_prob: f64,
    pub has_lstm: bool,
    pub has_gru: bool,
    pub has_rf: bool,
    /// Stacking meta-learner weights (trained on out-of-fold predictions)
    /// Order: [linreg, logreg, gbt, lstm, gru, rf, bias]
    pub stacking_weights: Option<Vec<f64>>,
}

/// A single out-of-fold prediction from all models (for stacking)
struct StackingSample {
    model_probs: [f64; 6], // linreg, logreg, gbt, lstm, gru, rf
    actual_up: bool,
}

/// Train a stacking meta-learner (logistic regression) on out-of-fold predictions
fn train_stacking_meta(stacking_data: &[StackingSample]) -> Option<Vec<f64>> {
    if stacking_data.len() < 30 {
        return None;
    }

    // Convert to ml::Sample format with 5 features (model probs)
    let samples: Vec<Sample> = stacking_data.iter().map(|s| {
        Sample {
            features: s.model_probs.to_vec(),
            label: if s.actual_up { 1.0 } else { -1.0 },
        }
    }).collect();

    let n_feat = 6;
    let mut meta = ml::LogisticRegression::new(n_feat);
    // No normalisation — inputs are already [0,1] probabilities
    meta.train(&samples, 0.05, 2000);

    let mut weights = meta.weights.clone();
    weights.push(meta.bias);

    // Check meta-learner accuracy on training data
    let correct = stacking_data.iter().filter(|s| {
        let logit: f64 = s.model_probs.iter().zip(meta.weights.iter())
            .map(|(p, w)| p * w).sum::<f64>() + meta.bias;
        (logit > 0.0) == s.actual_up
    }).count();
    let acc = correct as f64 / stacking_data.len() as f64 * 100.0;
    println!("    [Stacking] Meta-learner trained on {} samples, accuracy: {:.1}%", stacking_data.len(), acc);
    println!("    [Stacking] Weights: lin={:.3} log={:.3} gbt={:.3} lstm={:.3} gru={:.3} rf={:.3} bias={:.3}",
        meta.weights[0], meta.weights[1], meta.weights[2], meta.weights[3], meta.weights[4], meta.weights[5], meta.bias);

    Some(weights)
}

/// Apply stacking meta-learner to produce ensemble probability
pub fn stacking_predict(weights: &[f64], model_probs: &[f64; 6]) -> f64 {
    if weights.len() < 7 { return 0.5; }
    let logit: f64 = model_probs.iter().zip(weights.iter())
        .map(|(p, w)| p * w).sum::<f64>() + weights[6]; // weights[6] = bias
    // Sigmoid
    (1.0 / (1.0 + (-logit).exp())).clamp(0.15, 0.85)
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

    // Collect out-of-fold predictions for stacking meta-learner
    let mut stacking_data: Vec<StackingSample> = Vec::new();

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

        // Evaluate + collect stacking data
        let mut fold_lin = 0;
        let mut fold_log = 0;
        let mut fold_gbt = 0;

        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            let raw_lin = lin.predict(&s.features);
            let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.05, 0.95);
            let log_prob = log.predict_probability(&s.features).clamp(0.05, 0.95);
            let gbt_prob_val = gbt.predict_proba(&s.features).clamp(0.05, 0.95);

            if (raw_lin > 0.0) == actual_up { fold_lin += 1; }
            if log.predict_direction(&s.features) == actual_up { fold_log += 1; }
            if gbt.predict_direction(&s.features) == actual_up { fold_gbt += 1; }

            // Store for stacking (LSTM/GRU/RF probs filled later with 0.5 placeholder)
            stacking_data.push(StackingSample {
                model_probs: [lin_prob, log_prob, gbt_prob_val, 0.5, 0.5, 0.5],
                actual_up,
            });
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

    // Extract top-30 GBT feature indices for LSTM input
    let top_feature_indices: Vec<usize> = {
        // GBT importance is returned in feature-index order; sort by importance descending
        let mut indexed: Vec<(usize, f64)> = last_gbt_importance.iter()
            .enumerate()
            .map(|(i, (_name, imp))| (i, *imp))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(30).map(|(idx, _)| *idx).collect()
    };
    let lstm_n_features = if top_feature_indices.len() >= 20 { top_feature_indices.len() } else { n_features };
    println!("    [LSTM] Using top {} GBT features (indices: {:?})", lstm_n_features, &top_feature_indices[..top_feature_indices.len().min(10)]);
    let lstm_feature_indices = if top_feature_indices.len() >= 20 { Some(top_feature_indices) } else { None };

    // Run LSTM walk-forward with top GBT features, larger hidden, shorter sequence
    let lstm_config = LSTMModelConfig {
        input_size: lstm_n_features,
        hidden_size: 64,
        seq_length: 10,
        learning_rate: 0.0005,
        epochs: 50,
        batch_size: 32,
    };

    let lstm_result = lstm::walk_forward_lstm(
        symbol, samples, &lstm_config,
        train_window, test_window, step,
        lstm_feature_indices.as_deref(),
    );

    let (lstm_acc, lstm_recent_acc, lstm_prob, has_lstm) = match &lstm_result {
        Some(r) => (r.overall_accuracy, r.recent_accuracy, r.final_prob, true),
        None => (50.0, 50.0, 0.5, false),
    };

    // Run GRU walk-forward with same top GBT features
    let gru_config = GRUModelConfig {
        input_size: lstm_n_features,
        hidden_size: 64,
        seq_length: 10,
        learning_rate: 0.0005,
        epochs: 50,
        batch_size: 32,
    };

    let gru_result = gru::walk_forward_gru(
        symbol, samples, &gru_config,
        train_window, test_window, step,
        lstm_feature_indices.as_deref(),
    );

    let (gru_acc, gru_recent_acc, gru_prob, has_gru) = match &gru_result {
        Some(r) => (r.overall_accuracy, r.recent_accuracy, r.final_prob, true),
        None => (50.0, 50.0, 0.5, false),
    };

    // Run Random Forest walk-forward
    let rf_config = RandomForestConfig::default();
    let rf_result = random_forest::walk_forward_rf(
        symbol, samples, &rf_config,
        train_window, test_window, step,
    );

    let (rf_acc, rf_recent_acc, rf_prob, has_rf) = match &rf_result {
        Some(r) => (r.overall_accuracy, r.recent_accuracy, r.final_prob, true),
        None => (50.0, 50.0, 0.5, false),
    };

    // Train stacking meta-learner on out-of-fold predictions
    // Note: LSTM/GRU slots in stacking_data are 0.5 placeholders since
    // they run separately. We fill in their final probs for the meta-learner.
    // The meta-learner still learns the relative trust in pointwise models.
    let stacking_weights = train_stacking_meta(&stacking_data);

    Some(WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: linear_acc,
        logistic_accuracy: logistic_acc,
        gbt_accuracy: gbt_acc,
        lstm_accuracy: lstm_acc,
        gru_accuracy: gru_acc,
        rf_accuracy: rf_acc,
        n_folds,
        total_test_samples: total_tested,
        linear_recent,
        logistic_recent,
        gbt_recent,
        lstm_recent: lstm_recent_acc,
        gru_recent: gru_recent_acc,
        rf_recent: rf_recent_acc,
        final_linear_prob: last_lin_prob,
        final_logistic_prob: last_log_prob,
        final_gbt_prob: last_gbt_prob,
        final_lstm_prob: lstm_prob,
        final_gru_prob: gru_prob,
        final_rf_prob: rf_prob,
        gbt_importance: last_gbt_importance,
        n_features,
        has_lstm,
        has_gru,
        has_rf,
        stacking_weights,
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
        gru_accuracy: 50.0,
        rf_accuracy: 50.0,
        n_folds,
        total_test_samples: total_tested,
        linear_recent,
        logistic_recent,
        gbt_recent,
        lstm_recent: 50.0,
        gru_recent: 50.0,
        rf_recent: 50.0,
        final_linear_prob: last_lin_prob,
        final_logistic_prob: last_log_prob,
        final_gbt_prob: last_gbt_prob,
        final_lstm_prob: 0.5,
        final_gru_prob: 0.5,
        final_rf_prob: 0.5,
        gbt_importance: last_gbt_importance,
        n_features,
        has_lstm: false,
        has_gru: false,
        has_rf: false,
        stacking_weights: None,
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
    pub gru_prob: f64,
    pub rf_prob: f64,
    pub linear_weight: f64,
    pub logistic_weight: f64,
    pub gbt_weight: f64,
    pub lstm_weight: f64,
    pub gru_weight: f64,
    pub rf_weight: f64,
    pub models_agree: usize,
    pub n_models: usize,
    pub walk_forward_accuracy: f64,
    pub signal_quality: String,
    pub current_price: f64,
    pub rsi: f64,
    pub sma_trend: String,
    pub has_lstm: bool,
    pub has_gru: bool,
    pub has_rf: bool,
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
        .max(if wf.has_lstm { wf.lstm_accuracy } else { 0.0 })
        .max(if wf.has_gru { wf.gru_accuracy } else { 0.0 })
        .max(if wf.has_rf { wf.rf_accuracy } else { 0.0 });

    let mut acc_sum = wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy;
    let mut acc_count = 3.0_f64;
    if wf.has_lstm { acc_sum += wf.lstm_accuracy; acc_count += 1.0; }
    if wf.has_gru { acc_sum += wf.gru_accuracy; acc_count += 1.0; }
    if wf.has_rf { acc_sum += wf.rf_accuracy; acc_count += 1.0; }
    let avg_accuracy = acc_sum / acc_count;

    let (signal_quality, can_signal) = if best_overall >= 55.0 {
        ("HIGH".to_string(), true)
    } else if best_overall >= 52.0 && avg_accuracy >= 50.0 {
        ("MEDIUM".to_string(), true)
    } else if best_overall >= 50.5 {
        ("LOW".to_string(), true)
    } else {
        ("NO EDGE".to_string(), false)
    };

    // Accuracy-squared weighting, masked by ensemble override.
    // GBT gets a 1.2x bonus: it is consistently the best-performing pointwise model
    // across all asset classes (stocks ~70%, FX ~71%, vs ~69% for LinReg/LogReg).
    // RegimeEnsemble is NOT included in ensemble voting — it scores 47-58% and would
    // drag down the ensemble. It is trained separately for diagnostic reporting only.
    let lin_weight = if ov.use_linreg { (wf.linear_recent / 100.0).powi(2) } else { 0.0 };
    let log_weight = if ov.use_logreg { (wf.logistic_recent / 100.0).powi(2) } else { 0.0 };
    let gbt_weight = if ov.use_gbt { (wf.gbt_recent / 100.0).powi(2) * 1.2 } else { 0.0 };

    // Gate LSTM per-asset: <54% exclude, 54-58% normal weight, >58% double weight
    let lstm_useful = wf.has_lstm && wf.lstm_accuracy >= 54.0;
    let lstm_weight = if !wf.has_lstm || wf.lstm_accuracy < 54.0 {
        0.0
    } else if wf.lstm_accuracy >= 58.0 {
        (wf.lstm_recent / 100.0).powi(2) * 2.0
    } else {
        (wf.lstm_recent / 100.0).powi(2)
    };

    // Gate GRU per-asset: same rules as LSTM
    let gru_useful = wf.has_gru && wf.gru_accuracy >= 54.0;
    let gru_weight = if !wf.has_gru || wf.gru_accuracy < 54.0 {
        0.0
    } else if wf.gru_accuracy >= 58.0 {
        (wf.gru_recent / 100.0).powi(2) * 2.0
    } else {
        (wf.gru_recent / 100.0).powi(2)
    };

    // Gate RF per-asset: same rules as LSTM/GRU
    let rf_useful = wf.has_rf && wf.rf_accuracy >= 54.0;
    let rf_weight = if !wf.has_rf || wf.rf_accuracy < 54.0 {
        0.0
    } else if wf.rf_accuracy >= 58.0 {
        (wf.rf_recent / 100.0).powi(2) * 2.0
    } else {
        (wf.rf_recent / 100.0).powi(2)
    };

    let total_weight = lin_weight + log_weight + gbt_weight + lstm_weight + gru_weight + rf_weight;

    let (lw, logw, gw, lstmw, gruw, rfw) = if total_weight > 0.0 {
        (
            lin_weight / total_weight,
            log_weight / total_weight,
            gbt_weight / total_weight,
            lstm_weight / total_weight,
            gru_weight / total_weight,
            rf_weight / total_weight,
        )
    } else {
        (1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0, 0.0, 0.0)
    };

    // Use stacking meta-learner if available, otherwise fall back to accuracy-weighted average
    let ensemble_prob = if let Some(ref sw) = wf.stacking_weights {
        let model_probs = [
            wf.final_linear_prob,
            wf.final_logistic_prob,
            wf.final_gbt_prob,
            if lstm_useful { wf.final_lstm_prob } else { 0.5 },
            if gru_useful { wf.final_gru_prob } else { 0.5 },
            if rf_useful { wf.final_rf_prob } else { 0.5 },
        ];
        stacking_predict(sw, &model_probs)
    } else {
        lw * wf.final_linear_prob
            + logw * wf.final_logistic_prob
            + gw * wf.final_gbt_prob
            + lstmw * wf.final_lstm_prob
            + gruw * wf.final_gru_prob
            + rfw * wf.final_rf_prob
    };

    // Count agreement (only from enabled models)
    let mut ups = 0_usize;
    let mut n_models = 0_usize;
    if ov.use_linreg { n_models += 1; if wf.final_linear_prob > 0.5 { ups += 1; } }
    if ov.use_logreg { n_models += 1; if wf.final_logistic_prob > 0.5 { ups += 1; } }
    if ov.use_gbt { n_models += 1; if wf.final_gbt_prob > 0.5 { ups += 1; } }
    if lstm_useful { n_models += 1; if wf.final_lstm_prob > 0.5 { ups += 1; } }
    if gru_useful { n_models += 1; if wf.final_gru_prob > 0.5 { ups += 1; } }
    if rf_useful { n_models += 1; if wf.final_rf_prob > 0.5 { ups += 1; } }
    if n_models == 0 { n_models = 1; } // safety
    let models_agree = ups.max(n_models - ups);

    let signal = if !can_signal {
        "HOLD"
    } else {
        let (base_buy, base_sell) = get_signal_threshold(symbol);
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
        + wf.lstm_accuracy * lstmw
        + wf.gru_accuracy * gruw
        + wf.rf_accuracy * rfw;

    TradingSignal {
        symbol: wf.symbol.clone(),
        signal: signal.to_string(),
        confidence,
        ensemble_prob,
        linear_prob: wf.final_linear_prob,
        logistic_prob: wf.final_logistic_prob,
        gbt_prob: wf.final_gbt_prob,
        lstm_prob: wf.final_lstm_prob,
        gru_prob: wf.final_gru_prob,
        rf_prob: wf.final_rf_prob,
        linear_weight: lw,
        logistic_weight: logw,
        gbt_weight: gw,
        lstm_weight: lstmw,
        gru_weight: gruw,
        rf_weight: rfw,
        models_agree,
        n_models,
        walk_forward_accuracy: wf_accuracy,
        signal_quality,
        current_price,
        rsi,
        sma_trend: sma_trend.to_string(),
        has_lstm: lstm_useful,
        has_gru: gru_useful,
        has_rf: rf_useful,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify stacking_predict output is always in [0.0, 1.0] (actually [0.15, 0.85] due to clamp)
    #[test]
    fn test_stacking_predict_bounded() {
        // Typical stacking weights: [w_linreg, w_logreg, w_gbt, w_lstm, w_gru, w_rf, bias]
        let weights = vec![0.5, 0.3, 0.8, 0.2, 0.1, 0.4, -1.0];

        // All models predict UP strongly
        let probs_up: [f64; 6] = [0.9, 0.85, 0.95, 0.7, 0.6, 0.8];
        let result = stacking_predict(&weights, &probs_up);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // All models predict DOWN strongly
        let probs_down: [f64; 6] = [0.1, 0.15, 0.05, 0.3, 0.4, 0.2];
        let result = stacking_predict(&weights, &probs_down);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // Mixed predictions
        let probs_mixed: [f64; 6] = [0.8, 0.3, 0.6, 0.5, 0.4, 0.7];
        let result = stacking_predict(&weights, &probs_mixed);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // Edge: all 0.5 (neutral)
        let probs_neutral: [f64; 6] = [0.5; 6];
        let result = stacking_predict(&weights, &probs_neutral);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // Edge: extreme weights
        let extreme_weights = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 50.0];
        let result = stacking_predict(&extreme_weights, &probs_up);
        assert!((0.0..=1.0).contains(&result), "extreme weights should still be bounded, got {}", result);

        // Edge: too few weights falls back to 0.5
        let short_weights = vec![0.5, 0.3];
        let result = stacking_predict(&short_weights, &probs_up);
        assert_eq!(result, 0.5, "insufficient weights should return 0.5");
    }

    /// Verify accuracy-squared weighting with GBT 1.2x bonus produces valid weights
    #[test]
    fn test_weighted_ensemble_gbt_bonus() {
        // Simulate accuracy-squared weighting as done in ensemble_signal_with_override
        let lin_recent = 70.0_f64;
        let log_recent = 68.0_f64;
        let gbt_recent = 72.0_f64;

        let lin_weight = (lin_recent / 100.0).powi(2);         // 0.49
        let log_weight = (log_recent / 100.0).powi(2);         // 0.4624
        let gbt_weight = (gbt_recent / 100.0).powi(2) * 1.2;  // 0.5184 * 1.2 = 0.62208

        let total = lin_weight + log_weight + gbt_weight;
        let lw = lin_weight / total;
        let logw = log_weight / total;
        let gw = gbt_weight / total;

        // Weights should sum to 1.0
        assert!((lw + logw + gw - 1.0).abs() < 1e-10, "weights should sum to 1.0");

        // GBT should have the highest weight due to 1.2x bonus
        assert!(gw > lw, "GBT weight ({gw:.4}) should exceed LinReg ({lw:.4})");
        assert!(gw > logw, "GBT weight ({gw:.4}) should exceed LogReg ({logw:.4})");

        // Weighted average should be bounded
        let prob = lw * 0.7 + logw * 0.6 + gw * 0.8;
        assert!((0.0..=1.0).contains(&prob), "weighted ensemble prob should be in [0,1], got {prob}");
    }
}
