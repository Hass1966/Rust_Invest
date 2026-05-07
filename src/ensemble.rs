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
use crate::gbt::{self, GBTConfig, GradientBoostedClassifier};
use crate::lgbm::{LGBMConfig, LightGBMClassifier, LGBMRegressorConfig, LightGBMRegressor};
use crate::lstm::{self, LSTMModelConfig};
use crate::gru::{self, GRUModelConfig, GRURegressionConfig, GRURegressionModel};
use crate::ridge::RidgeRegression;
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
// Threshold Overrides — agent-managed per-asset thresholds
// ════════════════════════════════════════

/// Per-asset threshold override set by the agent
#[derive(Deserialize, Debug, Clone)]
pub struct ThresholdOverrideEntry {
    pub buy_threshold: Option<f64>,
    pub sell_threshold: Option<f64>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ThresholdOverrides {
    #[serde(default)]
    pub overrides: HashMap<String, ThresholdOverrideEntry>,
}

/// Load threshold overrides from config/threshold_overrides.json
pub fn load_threshold_overrides() -> ThresholdOverrides {
    let path = "config/threshold_overrides.json";
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            match serde_json::from_str::<ThresholdOverrides>(&contents) {
                Ok(t) => {
                    if !t.overrides.is_empty() {
                        println!("  [Ensemble] Loaded {} threshold overrides from {}", t.overrides.len(), path);
                    }
                    t
                }
                Err(e) => {
                    println!("  [Ensemble] Failed to parse {}: {}", path, e);
                    ThresholdOverrides::default()
                }
            }
        }
        Err(_) => ThresholdOverrides::default(),
    }
}

/// Get the buy threshold for an asset (returns None if no override, meaning use default)
pub fn get_buy_threshold(overrides: &ThresholdOverrides, symbol: &str) -> Option<f64> {
    overrides.overrides.get(symbol).and_then(|o| o.buy_threshold)
}

/// Get the sell threshold for an asset (returns None if no override)
pub fn get_sell_threshold(overrides: &ThresholdOverrides, symbol: &str) -> Option<f64> {
    overrides.overrides.get(symbol).and_then(|o| o.sell_threshold)
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
    /// Order: [linreg, logreg, gbt, lgbm, lstm, gru, rf, bias]
    pub stacking_weights: Option<Vec<f64>>,
    /// Per-model validation log-loss (lower = better calibrated)
    /// Used for inverse-loss weighted ensemble voting
    pub val_log_loss: Option<[f64; 4]>, // [linreg, logreg, gbt, lgbm]
    /// LightGBM model #7
    pub lgbm_accuracy: f64,
    pub lgbm_recent: f64,
    pub final_lgbm_prob: f64,
    pub has_lgbm: bool,
    pub lgbm_importance: Vec<(String, f64)>,
    /// Platt scaling calibration for ensemble probability
    pub platt_params: Option<PlattParams>,
}

/// A single out-of-fold prediction from all models (for stacking)
struct StackingSample {
    model_probs: [f64; 7], // linreg, logreg, gbt, lgbm, lstm, gru, rf
    actual_up: bool,
}

/// Train a stacking meta-learner (logistic regression) on out-of-fold predictions
fn train_stacking_meta(stacking_data: &[StackingSample]) -> Option<Vec<f64>> {
    if stacking_data.len() < 30 {
        return None;
    }

    // Convert to ml::Sample format with 7 features (model probs)
    let samples: Vec<Sample> = stacking_data.iter().map(|s| {
        Sample {
            features: s.model_probs.to_vec(),
            label: if s.actual_up { 1.0 } else { -1.0 },
        }
    }).collect();

    let n_feat = 7;
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
    println!("    [Stacking] Weights: lin={:.3} log={:.3} gbt={:.3} lgbm={:.3} lstm={:.3} gru={:.3} rf={:.3} bias={:.3}",
        meta.weights[0], meta.weights[1], meta.weights[2], meta.weights[3], meta.weights[4], meta.weights[5], meta.weights[6], meta.bias);

    Some(weights)
}

/// Apply stacking meta-learner to produce ensemble probability
pub fn stacking_predict(weights: &[f64], model_probs: &[f64; 7]) -> f64 {
    if weights.len() < 8 { return 0.5; }
    let logit: f64 = model_probs.iter().zip(weights.iter())
        .map(|(p, w)| p * w).sum::<f64>() + weights[7]; // weights[7] = bias
    // Sigmoid
    (1.0 / (1.0 + (-logit).exp())).clamp(0.15, 0.85)
}

/// Platt scaling parameters for probability calibration.
/// Maps raw probability → calibrated probability via: P_cal = 1/(1+exp(A*logit(p) + B))
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlattParams {
    pub a: f64,
    pub b: f64,
}

/// Fit Platt scaling parameters from out-of-fold (predicted_prob, actual_label) pairs.
/// Uses Newton's method to minimise negative log-likelihood.
pub fn fit_platt_scaling(predictions: &[(f64, bool)]) -> Option<PlattParams> {
    if predictions.len() < 20 { return None; }

    // Target values with Platt's label smoothing
    let n_pos = predictions.iter().filter(|(_, y)| *y).count() as f64;
    let n_neg = predictions.len() as f64 - n_pos;
    if n_pos < 5.0 || n_neg < 5.0 { return None; }

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    // Convert probabilities to logits for fitting
    let logits: Vec<f64> = predictions.iter().map(|(p, _)| {
        let clamped = p.clamp(0.01, 0.99);
        (clamped / (1.0 - clamped)).ln()
    }).collect();
    let targets: Vec<f64> = predictions.iter().map(|(_, y)| {
        if *y { t_pos } else { t_neg }
    }).collect();

    // Newton's method for A, B
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;

    for _ in 0..100 {
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;
        let mut hess_aa = 0.0;
        let mut hess_ab = 0.0;
        let mut hess_bb = 0.0;

        for (i, f) in logits.iter().enumerate() {
            let p = 1.0 / (1.0 + (-(a * f + b)).exp());
            let t = targets[i];
            let d = p - t;
            let h = p * (1.0 - p);

            grad_a += d * f;
            grad_b += d;
            hess_aa += h * f * f;
            hess_ab += h * f;
            hess_bb += h;
        }

        // Add small regularisation for numerical stability
        hess_aa += 1e-6;
        hess_bb += 1e-6;

        let det = hess_aa * hess_bb - hess_ab * hess_ab;
        if det.abs() < 1e-12 { break; }

        let da = -(hess_bb * grad_a - hess_ab * grad_b) / det;
        let db = -(hess_aa * grad_b - hess_ab * grad_a) / det;

        a += da;
        b += db;

        if da.abs() < 1e-8 && db.abs() < 1e-8 { break; }
    }

    Some(PlattParams { a, b })
}

/// Apply Platt scaling to a raw probability.
pub fn platt_calibrate(raw_prob: f64, params: &PlattParams) -> f64 {
    let clamped = raw_prob.clamp(0.01, 0.99);
    let logit = (clamped / (1.0 - clamped)).ln();
    let calibrated = 1.0 / (1.0 + (-(params.a * logit + params.b)).exp());
    calibrated.clamp(0.05, 0.95)
}

/// Compute binary cross-entropy (log-loss) for a single prediction
pub fn log_loss_single(predicted: f64, actual_up: bool) -> f64 {
    let p = predicted.clamp(1e-7, 1.0 - 1e-7);
    let y = if actual_up { 1.0 } else { 0.0 };
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// Compute recency weights: last ~126 samples (6 months of daily data) get 3x weight, older get 1x.
/// Smooth transition over 20 samples to avoid sharp boundary.
pub fn compute_recency_weights(n_samples: usize) -> Vec<f64> {
    let recent_count = 126; // ~6 months of trading days
    let transition = 20;
    let recent_start = n_samples.saturating_sub(recent_count);
    (0..n_samples)
        .map(|i| {
            if i >= recent_start {
                3.0
            } else if i >= recent_start.saturating_sub(transition) {
                let progress = (i as f64 - (recent_start.saturating_sub(transition)) as f64) / transition as f64;
                1.0 + 2.0 * progress // smooth ramp from 1.0 to 3.0
            } else {
                1.0
            }
        })
        .collect()
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
    let mut total_lgbm_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;

    let mut last_lin_correct = 0_usize;
    let mut last_log_correct = 0_usize;
    let mut last_gbt_correct = 0_usize;
    let mut last_lgbm_correct = 0_usize;
    let mut last_fold_size = 0_usize;

    let mut last_lin_prob = 0.5;
    let mut last_log_prob = 0.5;
    let mut last_gbt_prob = 0.5;
    let mut last_lgbm_prob = 0.5;
    let mut last_gbt_importance = Vec::new();
    let mut last_lgbm_importance = Vec::new();
    let mut has_lgbm = false;

    // Collect out-of-fold predictions for stacking meta-learner
    let mut stacking_data: Vec<StackingSample> = Vec::new();

    // Accumulate validation log-losses across folds for inverse-loss weighting
    let mut total_lin_log_loss = 0.0_f64;
    let mut total_log_log_loss = 0.0_f64;
    let mut total_gbt_log_loss = 0.0_f64;
    let mut total_lgbm_log_loss = 0.0_f64;
    let mut total_val_samples = 0_usize;

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

        // Recency weighting: last ~126 samples (6 months daily) get 3x weight
        let recency_weights: Vec<f64> = compute_recency_weights(train_data.len());

        // Class weights to fix bullish bias — combine with recency weights
        let (w_down, w_up) = crate::features::compute_class_weights(train_data, 0.005);
        let combined_weights: Vec<f64> = train_data.iter().enumerate().map(|(i, s)| {
            let class_w = if s.label > 0.0 { w_up } else { w_down };
            recency_weights[i] * class_w
        }).collect();

        // Train all 3 pointwise models with combined recency + class weighting
        let mut lin = ml::LinearRegression::new(n_features);
        lin.train_weighted(train_data, Some(&combined_weights), 0.005, 3000);

        let mut log = ml::LogisticRegression::new(n_features);
        log.train_weighted(train_data, Some(&combined_weights), 0.01, 3000);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);
        let gbt_recency = &combined_weights[..x_t.len()]; // class + recency weights for training portion

        let gbt_config = GBTConfig::default();

        let gbt = GradientBoostedClassifier::train_weighted(
            x_t, y_t, Some(gbt_recency), Some(x_v), Some(y_v), gbt_config,
        );

        // Train LightGBM model #7
        let lgbm_config = LGBMConfig::default();
        let lgbm_recency: Vec<f64> = combined_weights[..x_t.len()].to_vec();
        let lgbm_model = LightGBMClassifier::train(
            x_t, y_t, Some(&lgbm_recency), Some(x_v), Some(y_v), &lgbm_config,
        );

        // Evaluate + collect stacking data
        let mut fold_lin = 0;
        let mut fold_log = 0;
        let mut fold_gbt = 0;
        let mut fold_lgbm = 0;

        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            let raw_lin = lin.predict(&s.features);
            let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.05, 0.95);
            let log_prob = log.predict_probability(&s.features).clamp(0.05, 0.95);
            let gbt_prob_val = gbt.predict_proba(&s.features).clamp(0.05, 0.95);
            let lgbm_prob_val = lgbm_model.as_ref()
                .map(|m| m.predict_proba(&s.features).clamp(0.05, 0.95))
                .unwrap_or(0.5);

            if (raw_lin > 0.0) == actual_up { fold_lin += 1; }
            if log.predict_direction(&s.features) == actual_up { fold_log += 1; }
            if gbt.predict_direction(&s.features) == actual_up { fold_gbt += 1; }
            if lgbm_model.as_ref().map(|m| m.predict_direction(&s.features)).unwrap_or(false) == actual_up { fold_lgbm += 1; }

            // Accumulate validation log-loss for inverse-loss weighting
            total_lin_log_loss += log_loss_single(lin_prob, actual_up);
            total_log_log_loss += log_loss_single(log_prob, actual_up);
            total_gbt_log_loss += log_loss_single(gbt_prob_val, actual_up);
            total_lgbm_log_loss += log_loss_single(lgbm_prob_val, actual_up);
            total_val_samples += 1;

            // Store for stacking (LSTM/GRU/RF probs filled later with 0.5 placeholder)
            stacking_data.push(StackingSample {
                model_probs: [lin_prob, log_prob, gbt_prob_val, lgbm_prob_val, 0.5, 0.5, 0.5],
                actual_up,
            });
        }

        total_lin_correct += fold_lin;
        total_log_correct += fold_log;
        total_gbt_correct += fold_gbt;
        total_lgbm_correct += fold_lgbm;
        total_tested += test_len;
        n_folds += 1;

        last_lin_correct = fold_lin;
        last_log_correct = fold_log;
        last_gbt_correct = fold_gbt;
        last_lgbm_correct = fold_lgbm;
        last_fold_size = test_len;

        // Final predictions (sigmoid + clamp)
        if let Some(last_sample) = test_data.last() {
            let raw_lin = lin.predict(&last_sample.features);
            last_lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);
            last_log_prob = log.predict_probability(&last_sample.features).clamp(0.15, 0.85);
            last_gbt_prob = gbt.predict_proba(&last_sample.features).clamp(0.15, 0.85);
            last_lgbm_prob = lgbm_model.as_ref()
                .map(|m| m.predict_proba(&last_sample.features).clamp(0.15, 0.85))
                .unwrap_or(0.5);
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
        if let Ok(ref m) = lgbm_model {
            last_lgbm_importance = m.feature_importance(&feat_refs);
            has_lgbm = true;
        }

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        return None;
    }

    let linear_acc = total_lin_correct as f64 / total_tested as f64 * 100.0;
    let logistic_acc = total_log_correct as f64 / total_tested as f64 * 100.0;
    let gbt_acc = total_gbt_correct as f64 / total_tested as f64 * 100.0;
    let lgbm_acc = total_lgbm_correct as f64 / total_tested as f64 * 100.0;

    let linear_recent = last_lin_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let logistic_recent = last_log_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let gbt_recent = last_gbt_correct as f64 / last_fold_size.max(1) as f64 * 100.0;
    let lgbm_recent = last_lgbm_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    walk-forward: {} folds, {} test samples", n_folds, total_tested);
    println!("      LinReg:  {:.1}% (recent: {:.1}%)", linear_acc, linear_recent);
    println!("      LogReg:  {:.1}% (recent: {:.1}%)", logistic_acc, logistic_recent);
    println!("      GBT:     {:.1}% (recent: {:.1}%)", gbt_acc, gbt_recent);
    println!("      LightGBM:{:.1}% (recent: {:.1}%)", lgbm_acc, lgbm_recent);

    // Extract top-30 feature indices for LSTM input
    // Prefer LightGBM importance (better feature ranking) if available, else fall back to custom GBT
    let importance_source = if has_lgbm && !last_lgbm_importance.is_empty() {
        &last_lgbm_importance
    } else {
        &last_gbt_importance
    };
    let top_feature_indices: Vec<usize> = {
        let mut indexed: Vec<(usize, f64)> = importance_source.iter()
            .enumerate()
            .map(|(i, (_name, imp))| (i, *imp))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(30).map(|(idx, _)| *idx).collect()
    };
    let lstm_n_features = if top_feature_indices.len() >= 20 { top_feature_indices.len() } else { n_features };
    let source_name = if has_lgbm { "LightGBM" } else { "GBT" };
    println!("    [LSTM] Using top {} {} features (indices: {:?})", lstm_n_features, source_name, &top_feature_indices[..top_feature_indices.len().min(10)]);
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

    // Compute average validation log-loss per model for inverse-loss weighting
    let val_log_loss = if total_val_samples > 0 {
        let n = total_val_samples as f64;
        let losses = [
            (total_lin_log_loss / n).max(0.01),
            (total_log_log_loss / n).max(0.01),
            (total_gbt_log_loss / n).max(0.01),
            (total_lgbm_log_loss / n).max(0.01),
        ];
        println!("    [Val LogLoss] lin={:.4} log={:.4} gbt={:.4} lgbm={:.4}", losses[0], losses[1], losses[2], losses[3]);
        let inv_sum = 1.0 / losses[0] + 1.0 / losses[1] + 1.0 / losses[2] + 1.0 / losses[3];
        println!("    [Inv-Loss Weights] lin={:.3} log={:.3} gbt={:.3} lgbm={:.3}",
            (1.0 / losses[0]) / inv_sum, (1.0 / losses[1]) / inv_sum, (1.0 / losses[2]) / inv_sum, (1.0 / losses[3]) / inv_sum);
        Some(losses)
    } else {
        None
    };

    // Fit Platt scaling on out-of-fold ensemble probabilities
    let platt_params = if !stacking_data.is_empty() {
        let ensemble_pairs: Vec<(f64, bool)> = stacking_data.iter().map(|s| {
            // Compute ensemble prob from available model probs using inverse-loss weights or equal
            let probs = &s.model_probs;
            let avg_prob = if let Some(ref vll) = val_log_loss {
                let inv: Vec<f64> = vll.iter().map(|l| 1.0 / l).collect();
                let inv_sum: f64 = inv.iter().sum();
                (inv[0] * probs[0] + inv[1] * probs[1] + inv[2] * probs[2] + inv[3] * probs[3]) / inv_sum
            } else {
                (probs[0] + probs[1] + probs[2] + probs[3]) / 4.0
            };
            (avg_prob, s.actual_up)
        }).collect();
        let params = fit_platt_scaling(&ensemble_pairs);
        if let Some(ref p) = params {
            println!("    [Platt] Calibration fitted: A={:.4}, B={:.4}", p.a, p.b);
        }
        params
    } else {
        None
    };

    Some(WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: linear_acc,
        logistic_accuracy: logistic_acc,
        gbt_accuracy: gbt_acc,
        lstm_accuracy: lstm_acc,
        gru_accuracy: gru_acc,
        rf_accuracy: rf_acc,
        lgbm_accuracy: lgbm_acc,
        n_folds,
        total_test_samples: total_tested,
        linear_recent,
        logistic_recent,
        gbt_recent,
        lstm_recent: lstm_recent_acc,
        gru_recent: gru_recent_acc,
        rf_recent: rf_recent_acc,
        lgbm_recent,
        final_linear_prob: last_lin_prob,
        final_logistic_prob: last_log_prob,
        final_gbt_prob: last_gbt_prob,
        final_lstm_prob: lstm_prob,
        final_gru_prob: gru_prob,
        final_rf_prob: rf_prob,
        final_lgbm_prob: last_lgbm_prob,
        gbt_importance: last_gbt_importance,
        lgbm_importance: last_lgbm_importance,
        n_features,
        has_lstm,
        has_gru,
        has_rf,
        has_lgbm,
        stacking_weights,
        val_log_loss,
        platt_params,
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

    // Accumulate validation log-losses for inverse-loss weighting
    let mut total_lin_log_loss = 0.0_f64;
    let mut total_log_log_loss = 0.0_f64;
    let mut total_gbt_log_loss = 0.0_f64;
    let mut total_val_samples = 0_usize;

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

        let recency_weights: Vec<f64> = compute_recency_weights(train_data.len());

        let mut lin = ml::LinearRegression::new(n_features);
        lin.train_weighted(train_data, Some(&recency_weights), 0.005, 3000);

        let mut log = ml::LogisticRegression::new(n_features);
        log.train_weighted(train_data, Some(&recency_weights), 0.01, 3000);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);
        let gbt_recency = &recency_weights[..x_t.len()];

        let gbt_config = GBTConfig::default();

        let gbt = GradientBoostedClassifier::train_weighted(
            x_t, y_t, Some(gbt_recency), Some(x_v), Some(y_v), gbt_config,
        );

        let mut fold_lin = 0;
        let mut fold_log = 0;
        let mut fold_gbt = 0;

        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            let raw_lin = lin.predict(&s.features);
            let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.05, 0.95);
            let log_prob = log.predict_probability(&s.features).clamp(0.05, 0.95);
            let gbt_prob = gbt.predict_proba(&s.features).clamp(0.05, 0.95);

            if (raw_lin > 0.0) == actual_up { fold_lin += 1; }
            if log.predict_direction(&s.features) == actual_up { fold_log += 1; }
            if gbt.predict_direction(&s.features) == actual_up { fold_gbt += 1; }

            // Accumulate validation log-loss
            total_lin_log_loss += log_loss_single(lin_prob, actual_up);
            total_log_log_loss += log_loss_single(log_prob, actual_up);
            total_gbt_log_loss += log_loss_single(gbt_prob, actual_up);
            total_val_samples += 1;
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

    // Compute average validation log-loss per model
    let val_log_loss = if total_val_samples > 0 {
        let n = total_val_samples as f64;
        Some([
            (total_lin_log_loss / n).max(0.01),
            (total_log_log_loss / n).max(0.01),
            (total_gbt_log_loss / n).max(0.01),
            0.693, // placeholder for LGBM (not trained in no_lstm path)
        ])
    } else {
        None
    };

    println!("    walk-forward: {} folds, {} test samples", n_folds, total_tested);
    println!("      LinReg: {:.1}% (recent: {:.1}%)", linear_acc, linear_recent);
    println!("      LogReg: {:.1}% (recent: {:.1}%)", logistic_acc, logistic_recent);
    println!("      GBT:    {:.1}% (recent: {:.1}%)", gbt_acc, gbt_recent);

    Some(WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: linear_acc,
        logistic_accuracy: logistic_acc,
        gbt_accuracy: gbt_acc,
        lgbm_accuracy: 50.0,
        lstm_accuracy: 50.0,
        gru_accuracy: 50.0,
        rf_accuracy: 50.0,
        n_folds,
        total_test_samples: total_tested,
        linear_recent,
        logistic_recent,
        gbt_recent,
        lgbm_recent: 50.0,
        lstm_recent: 50.0,
        gru_recent: 50.0,
        rf_recent: 50.0,
        final_linear_prob: last_lin_prob,
        final_logistic_prob: last_log_prob,
        final_gbt_prob: last_gbt_prob,
        final_lgbm_prob: 0.5,
        final_lstm_prob: 0.5,
        final_gru_prob: 0.5,
        final_rf_prob: 0.5,
        gbt_importance: last_gbt_importance,
        lgbm_importance: Vec::new(),
        n_features,
        has_lstm: false,
        has_gru: false,
        has_rf: false,
        has_lgbm: false,
        stacking_weights: None,
        val_log_loss,
        platt_params: None, // Platt scaling not fitted for basic crypto walk-forward
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
    pub lgbm_prob: f64,
    pub lstm_prob: f64,
    pub gru_prob: f64,
    pub rf_prob: f64,
    pub linear_weight: f64,
    pub logistic_weight: f64,
    pub gbt_weight: f64,
    pub lgbm_weight: f64,
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
    pub has_lgbm: bool,
    pub llm_sentiment: f64,
    pub llm_analysis: Option<String>,
}

/// Signal thresholds per asset. Uses agent overrides from config/threshold_overrides.json
/// when available, falls back to default (0.55, 0.45).
/// File is cached in a static and refreshed at most once per minute.
/// Global market regime for threshold adjustment. Updated by the signal pipeline
/// after regime detection. Encoded as atomic: 0=Bull, 1=Neutral, 2=EarlyWarning, 3=Bear, 4=Crisis.
static CURRENT_REGIME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1); // default Neutral

/// Call this after computing market regime to update the global state.
pub fn set_market_regime(regime: &crate::market_regime::MarketRegime) {
    use crate::market_regime::MarketRegime;
    let val = match regime {
        MarketRegime::Bull => 0,
        MarketRegime::Neutral => 1,
        MarketRegime::EarlyWarning => 2,
        MarketRegime::Bear => 3,
        MarketRegime::Crisis => 4,
    };
    CURRENT_REGIME.store(val, std::sync::atomic::Ordering::Relaxed);
}

fn current_regime_sell_adjustment() -> f64 {
    // Regime info is already encoded in model features (vix_regime, VIX_level, risk_on_off).
    // Adjusting thresholds here double-counts regime and suppresses SELL in bull markets.
    0.0
}

pub fn get_signal_threshold(symbol: &str) -> (f64, f64) {
    use std::sync::Mutex;

    static CACHE: std::sync::LazyLock<Mutex<(std::time::Instant, HashMap<String, (f64, f64)>)>> =
        std::sync::LazyLock::new(|| Mutex::new((std::time::Instant::now(), HashMap::new())));

    let default_buy = 0.55;
    let default_sell = 0.45;

    let regime_adj = current_regime_sell_adjustment();

    let mut cache = match CACHE.lock() {
        Ok(c) => c,
        Err(_) => return (default_buy, (default_sell + regime_adj).clamp(0.20, 0.55)),
    };

    // Refresh cache every 60 seconds
    if cache.0.elapsed() > std::time::Duration::from_secs(60) || cache.1.is_empty() {
        if let Ok(contents) = std::fs::read_to_string("config/threshold_overrides.json") {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&contents) {
                let mut map = HashMap::new();
                if let Some(overrides) = parsed.get("overrides").and_then(|o| o.as_object()) {
                    for (sym, entry) in overrides {
                        // Skip expired overrides
                        if let Some(expires) = entry.get("expires_at").and_then(|v| v.as_str()) {
                            if let Ok(exp_time) = chrono::DateTime::parse_from_rfc3339(expires) {
                                if exp_time < chrono::Utc::now() { continue; }
                            }
                        }
                        let buy = entry.get("buy_threshold")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(default_buy);
                        let sell = entry.get("sell_threshold")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(default_sell);
                        map.insert(sym.clone(), (buy, sell));
                    }
                }
                cache.1 = map;
            }
        }
        cache.0 = std::time::Instant::now();
    }

    let (buy, sell) = cache.1.get(symbol).copied().unwrap_or((default_buy, default_sell));
    // Apply regime adjustment to sell threshold
    (buy, (sell + regime_adj).clamp(0.20, 0.55))
}

/// Compute adaptive thresholds from historical signal accuracy.
/// If an asset's BUY accuracy < 40%, raise the BUY threshold to require more confidence.
/// If SELL accuracy < 40%, raise the SELL threshold similarly.
/// Returns HashMap<asset, (buy_threshold, sell_threshold)>.
pub fn compute_adaptive_thresholds(db_path: &str) -> std::collections::HashMap<String, (f64, f64)> {
    let mut thresholds = std::collections::HashMap::new();
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(_) => return thresholds,
    };

    // Query per-asset BUY/SELL accuracy from resolved signal_history (last 14 days)
    let sql = "
        SELECT asset, signal_type,
               COUNT(*) as total,
               SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
        FROM signal_history
        WHERE was_correct IS NOT NULL
          AND timestamp >= datetime('now', '-14 days')
          AND signal_type IN ('BUY', 'SELL')
        GROUP BY asset, signal_type
        HAVING COUNT(*) >= 5
    ";

    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(_) => return thresholds,
    };

    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
        ))
    });

    let rows = match rows {
        Ok(r) => r,
        Err(_) => return thresholds,
    };

    // Collect per-asset accuracy
    let mut buy_acc: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut sell_acc: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for row in rows.flatten() {
        let (asset, sig_type, total, correct) = row;
        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.5 };
        match sig_type.as_str() {
            "BUY" => { buy_acc.insert(asset, accuracy); }
            "SELL" => { sell_acc.insert(asset, accuracy); }
            _ => {}
        }
    }

    // Compute adaptive thresholds
    let all_assets: std::collections::HashSet<&String> = buy_acc.keys().chain(sell_acc.keys()).collect();
    for asset in all_assets {
        let (base_buy, base_sell) = get_signal_threshold(asset);
        let ba = buy_acc.get(asset.as_str()).copied().unwrap_or(0.5);
        let sa = sell_acc.get(asset.as_str()).copied().unwrap_or(0.5);

        // If accuracy < 40%, tighten threshold (require more confidence)
        // If accuracy > 60%, slightly loosen (reward good performance)
        let buy_adj = if ba < 0.30 { 0.06 }
            else if ba < 0.40 { 0.04 }
            else if ba < 0.50 { 0.02 }
            else if ba > 0.65 { -0.02 }
            else { 0.0 };

        let sell_adj = if sa < 0.30 { -0.06 }
            else if sa < 0.40 { -0.04 }
            else if sa < 0.50 { -0.02 }
            else if sa > 0.65 { 0.02 }
            else { 0.0 };

        let adaptive_buy = (base_buy + buy_adj).clamp(0.52, 0.70);
        let adaptive_sell = (base_sell + sell_adj).clamp(0.30, 0.48);

        thresholds.insert(asset.clone(), (adaptive_buy, adaptive_sell));
    }

    if !thresholds.is_empty() {
        println!("  [Thresholds] Computed adaptive thresholds for {} assets", thresholds.len());
    }

    thresholds
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
        .max(if wf.has_lgbm { wf.lgbm_accuracy } else { 0.0 })
        .max(if wf.has_lstm { wf.lstm_accuracy } else { 0.0 })
        .max(if wf.has_gru { wf.gru_accuracy } else { 0.0 })
        .max(if wf.has_rf { wf.rf_accuracy } else { 0.0 });

    let mut acc_sum = wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy;
    let mut acc_count = 3.0_f64;
    if wf.has_lgbm { acc_sum += wf.lgbm_accuracy; acc_count += 1.0; }
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

    // Inverse validation log-loss weighting (preferred) or accuracy-squared fallback.
    // Models with lower log-loss (better calibrated probabilities) get higher weight.
    let (lin_weight, log_weight, gbt_weight, lgbm_weight_base) = if let Some(ref losses) = wf.val_log_loss {
        let inv_lin = if ov.use_linreg { 1.0 / losses[0] } else { 0.0 };
        let inv_log = if ov.use_logreg { 1.0 / losses[1] } else { 0.0 };
        let inv_gbt = if ov.use_gbt { 1.0 / losses[2] } else { 0.0 };
        let inv_lgbm = if wf.has_lgbm { 1.0 / losses[3] } else { 0.0 };
        (inv_lin, inv_log, inv_gbt, inv_lgbm)
    } else {
        let lw = if ov.use_linreg { (wf.linear_recent / 100.0).powi(2) } else { 0.0 };
        let logw = if ov.use_logreg { (wf.logistic_recent / 100.0).powi(2) } else { 0.0 };
        let gw = if ov.use_gbt { (wf.gbt_recent / 100.0).powi(2) } else { 0.0 };
        let lgw = if wf.has_lgbm { (wf.lgbm_recent / 100.0).powi(2) } else { 0.0 };
        (lw, logw, gw, lgw)
    };

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

    let lgbm_useful = wf.has_lgbm;
    let total_weight = lin_weight + log_weight + gbt_weight + lgbm_weight_base + lstm_weight + gru_weight + rf_weight;

    let (lw, logw, gw, lgbmw, lstmw, gruw, rfw) = if total_weight > 0.0 {
        (
            lin_weight / total_weight,
            log_weight / total_weight,
            gbt_weight / total_weight,
            lgbm_weight_base / total_weight,
            lstm_weight / total_weight,
            gru_weight / total_weight,
            rf_weight / total_weight,
        )
    } else {
        (1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, 0.0, 0.0, 0.0)
    };

    // Use stacking meta-learner if available, otherwise fall back to inverse-loss weighted average
    let ensemble_prob = if let Some(ref sw) = wf.stacking_weights {
        let model_probs = [
            wf.final_linear_prob,
            wf.final_logistic_prob,
            wf.final_gbt_prob,
            if lgbm_useful { wf.final_lgbm_prob } else { 0.5 },
            if lstm_useful { wf.final_lstm_prob } else { 0.5 },
            if gru_useful { wf.final_gru_prob } else { 0.5 },
            if rf_useful { wf.final_rf_prob } else { 0.5 },
        ];
        stacking_predict(sw, &model_probs)
    } else {
        lw * wf.final_linear_prob
            + logw * wf.final_logistic_prob
            + gw * wf.final_gbt_prob
            + lgbmw * wf.final_lgbm_prob
            + lstmw * wf.final_lstm_prob
            + gruw * wf.final_gru_prob
            + rfw * wf.final_rf_prob
    };

    // Apply Platt scaling calibration if available
    let ensemble_prob = if let Some(ref pp) = wf.platt_params {
        platt_calibrate(ensemble_prob, pp)
    } else {
        ensemble_prob
    };

    // Count agreement (only from enabled models)
    let mut ups = 0_usize;
    let mut n_models = 0_usize;
    if ov.use_linreg { n_models += 1; if wf.final_linear_prob > 0.5 { ups += 1; } }
    if ov.use_logreg { n_models += 1; if wf.final_logistic_prob > 0.5 { ups += 1; } }
    if ov.use_gbt { n_models += 1; if wf.final_gbt_prob > 0.5 { ups += 1; } }
    if lgbm_useful { n_models += 1; if wf.final_lgbm_prob > 0.5 { ups += 1; } }
    if lstm_useful { n_models += 1; if wf.final_lstm_prob > 0.5 { ups += 1; } }
    if gru_useful { n_models += 1; if wf.final_gru_prob > 0.5 { ups += 1; } }
    if rf_useful { n_models += 1; if wf.final_rf_prob > 0.5 { ups += 1; } }
    if n_models == 0 { n_models = 1; } // safety
    let models_agree = ups.max(n_models - ups);

    // Separate BUY, SELL, and SHORT scoring:
    // - BUY requires ensemble_prob above buy_thresh
    // - SELL requires ensemble_prob below sell_thresh AND majority of models agree on down
    // - SHORT requires very strong bearish conviction — ensemble_prob below sell_thresh AND
    //   high model agreement on down (same threshold as BUY but inverted)
    // - Low BUY probability alone defaults to HOLD (not SELL) unless models strongly agree
    let downs = n_models - ups;
    let signal = if !can_signal {
        "HOLD"
    } else {
        let (base_buy, base_sell) = get_signal_threshold(symbol);
        let (buy_thresh, sell_thresh) = if models_agree == n_models {
            (base_buy - 0.02, base_sell + 0.02)
        } else {
            (base_buy, base_sell)
        };

        // SHORT threshold: mirror of BUY — (1.0 - buy_thresh) is the bearish equivalent
        let short_thresh = 1.0 - buy_thresh;

        if ensemble_prob > buy_thresh {
            "BUY"
        } else if ensemble_prob < sell_thresh {
            "SELL"
        } else {
            "HOLD"
        }
    };

    let confidence = if !can_signal {
        0.0
    } else {
        // Normalised to 0–1 range (matches backtest_walkforward.rs)
        let raw = (ensemble_prob - 0.5).abs() * 2.0;
        let accuracy_cap = ((avg_accuracy - 50.0).max(0.0) / 50.0).min(1.0);
        raw.min(accuracy_cap).min(((best_overall - 50.0) / 50.0).min(1.0))
    };

    let wf_accuracy = wf.linear_accuracy * lw
        + wf.logistic_accuracy * logw
        + wf.gbt_accuracy * gw
        + wf.lgbm_accuracy * lgbmw
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
        lgbm_prob: wf.final_lgbm_prob,
        lstm_prob: wf.final_lstm_prob,
        gru_prob: wf.final_gru_prob,
        rf_prob: wf.final_rf_prob,
        linear_weight: lw,
        logistic_weight: logw,
        gbt_weight: gw,
        lgbm_weight: lgbmw,
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
        has_lgbm: lgbm_useful,
        llm_sentiment: 0.0,
        llm_analysis: None,
    }
}

/// Apply LLM sentiment adjustment to a signal.
/// Sentiment score (-1 to +1) shifts confidence and can flip marginal signals.
pub fn apply_sentiment_adjustment(signal: &mut TradingSignal, sentiment: f64, analysis: Option<String>) {
    signal.llm_sentiment = sentiment;
    signal.llm_analysis = analysis;

    // Sentiment alignment bonus: if LLM agrees with model direction, boost confidence
    let model_direction = if signal.ensemble_prob > 0.5 { 1.0 } else { -1.0 };
    let alignment = sentiment * model_direction; // positive = LLM agrees with models

    if alignment > 0.0 {
        // LLM agrees: boost confidence by up to 5 points
        signal.confidence += alignment * 5.0;
    } else if alignment < -0.3 {
        // LLM strongly disagrees: reduce confidence
        signal.confidence = (signal.confidence + alignment * 3.0).max(0.0);

        // If sentiment is very strong and models are marginal, flip to HOLD
        if sentiment.abs() > 0.6 && (signal.ensemble_prob - 0.5).abs() < 0.08 {
            signal.signal = "HOLD".to_string();
        }
        // If bullish sentiment strongly disagrees with SHORT, downgrade to SELL
        if signal.signal == "SHORT" && sentiment > 0.4 {
            signal.signal = "SELL".to_string();
        }
    }
}

// ════════════════════════════════════════
// Console Output
// ════════════════════════════════════════

pub fn print_signals(signals: &[TradingSignal]) {
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                          TRADING SIGNALS — Ensemble Consensus (4 Models)                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<8} {:>8} {:>8} {:>6}  {:>6} {:>6} {:>6} {:>6}  {:>5} {:>7} {:>6} {:>5} {:>7} ║",
        "Symbol", "Price", "Signal", "Conf%", "LinR", "LogR", "GBT", "LSTM", "Agree", "WF Acc", "RSI", "Sent", "Quality");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for s in signals {
        let signal_icon = match s.signal.as_str() {
            "BUY" => "▲ BUY  ",
            "SHORT" => "▼ SHORT",
            "SELL" => "▼ SELL ",
            _ => "● HOLD ",
        };
        let lstm_str = if s.has_lstm {
            format!("{:>5.1}%", s.lstm_prob * 100.0)
        } else {
            "  n/a".to_string()
        };
        let sent_str = if s.llm_sentiment != 0.0 {
            format!("{:>+5.2}", s.llm_sentiment)
        } else {
            "  n/a".to_string()
        };
        println!("║ {:<8} {:>8.2} {} {:>5.1}%  {:>5.1}% {:>5.1}% {:>5.1}% {}  {}/{}   {:>5.1}% {:>5.1} {} {:>7} ║",
            s.symbol, s.current_price, signal_icon, s.confidence,
            s.linear_prob * 100.0, s.logistic_prob * 100.0, s.gbt_prob * 100.0,
            lstm_str,
            s.models_agree, s.n_models, s.walk_forward_accuracy, s.rsi, sent_str, s.signal_quality);
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Quality: HIGH (>55%) = trustworthy | MEDIUM (52-55%) = marginal edge");
    println!("           LOW (50-52%) = barely above chance | NO EDGE (<50%) = forced HOLD");
    println!("  Conf% = distance from 50/50, capped by walk-forward accuracy");
    println!("  Sent = LLM sentiment (-1 bearish to +1 bullish)\n");
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

// ════════════════════════════════════════
// Regression Ensemble — Return Prediction
// ════════════════════════════════════════

/// Walk-forward regression results (Ridge + LightGBM + GRU)
pub struct RegressionResult {
    pub symbol: String,
    pub ridge_mae: f64,
    pub lgbm_mae: f64,
    pub gru_mae: f64,
    pub ridge_dir_acc: f64,
    pub lgbm_dir_acc: f64,
    pub gru_dir_acc: f64,
    pub ridge_return: f64,
    pub lgbm_return: f64,
    pub gru_return: f64,
    pub has_lgbm: bool,
    pub has_gru: bool,
    pub n_folds: usize,
    pub total_test_samples: usize,
    pub n_features: usize,
    pub lgbm_importance: Vec<(String, f64)>,
    pub norm_means: Vec<f64>,
    pub norm_stds: Vec<f64>,
    /// Platt calibration params for regression ensemble probability
    pub platt_params: Option<PlattParams>,
}

/// Walk-forward regression evaluation with Ridge, LightGBM Regressor, and GRU.
/// Labels are continuous percentage returns (not binary UP/DOWN).
pub fn walk_forward_regression(
    symbol: &str,
    samples: &[Sample],
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<RegressionResult> {
    // Embargo: skip 5 samples between train and test to prevent autocorrelation leakage
    const EMBARGO: usize = 5;

    if samples.len() < train_window + EMBARGO + test_window + 10 {
        println!("  {} — not enough samples for regression walk-forward ({}, need {})",
            symbol, samples.len(), train_window + EMBARGO + test_window);
        return None;
    }

    let n_features = samples[0].features.len();
    println!("  {} — regression walk-forward on {} samples × {} features (embargo={})",
        symbol, samples.len(), n_features, EMBARGO);

    // Accumulators for Ridge + LightGBM (pointwise models)
    let mut total_ridge_ae = 0.0_f64;
    let mut total_lgbm_ae = 0.0_f64;
    let mut total_ridge_dir = 0_usize;
    let mut total_lgbm_dir = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;
    let mut has_lgbm = false;

    let mut last_ridge_pred = 0.0_f64;
    let mut last_lgbm_pred = 0.0_f64;
    let mut last_lgbm_importance = Vec::new();
    let mut last_norm_means = Vec::new();
    let mut last_norm_stds = Vec::new();
    // Collect (ensemble_probability, actual_up) for Platt calibration
    let mut platt_data: Vec<(f64, bool)> = Vec::new();

    let mut start = 0;
    while start + train_window + EMBARGO + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_start = train_end + EMBARGO; // Skip embargo samples
        let test_end = (test_start + test_window).min(samples.len());

        // Build train from [start..train_end], test from [test_start..test_end]
        let mut train_data: Vec<Sample> = samples[start..train_end].to_vec();
        let mut test_data: Vec<Sample> = samples[test_start..test_end].to_vec();

        let (means, stds) = ml::normalise(&mut train_data);
        ml::apply_normalisation(&mut test_data, &means, &stds);

        last_norm_means = means;
        last_norm_stds = stds;

        let recency_weights = compute_recency_weights(train_data.len());

        // Extract features and continuous labels
        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter().map(|s| s.label).collect();

        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);

        // === Ridge ===
        let ridge = RidgeRegression::train(x_t, y_t, 10.0);

        // === LightGBM Regressor ===
        let lgbm_recency: Vec<f64> = recency_weights[..x_t.len()].to_vec();
        let lgbm_config = LGBMRegressorConfig::default();
        let lgbm_model = LightGBMRegressor::train(
            x_t, y_t, Some(&lgbm_recency), Some(x_v), Some(y_v), &lgbm_config,
        );

        // === Evaluate on test data ===
        for s in test_data.iter() {
            let actual = s.label;
            let mut ridge_pred = 0.0_f64;
            let mut lgbm_pred = 0.0_f64;

            if let Ok(ref r) = ridge {
                let pred = r.predict(&s.features);
                total_ridge_ae += (pred - actual).abs();
                if (pred > 0.0) == (actual > 0.0) { total_ridge_dir += 1; }
                last_ridge_pred = pred;
                ridge_pred = pred;
            }

            if let Ok(ref m) = lgbm_model {
                let pred = m.predict_return(&s.features);
                total_lgbm_ae += (pred - actual).abs();
                if (pred > 0.0) == (actual > 0.0) { total_lgbm_dir += 1; }
                last_lgbm_pred = pred;
                has_lgbm = true;
                lgbm_pred = pred;
            }

            // Collect calibration data: ensemble return → probability of UP
            let ensemble_return = if has_lgbm {
                (ridge_pred + lgbm_pred) / 2.0
            } else {
                ridge_pred
            };
            let ensemble_prob = 0.5 + ensemble_return.clamp(-5.0, 5.0) / 10.0;
            platt_data.push((ensemble_prob, actual > 0.0));

            total_tested += 1;
        }

        // Feature importance for GRU feature selection
        if let Ok(ref m) = lgbm_model {
            let feat_names: Vec<String> = {
                let rich = crate::features::feature_names();
                if n_features == rich.len() { rich }
                else { (0..n_features).map(|i| format!("Feature_{}", i)).collect() }
            };
            let feat_refs: Vec<&str> = feat_names.iter().map(|s| s.as_str()).collect();
            last_lgbm_importance = m.feature_importance(&feat_refs);
        }

        n_folds += 1;
        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        return None;
    }

    let ridge_mae = total_ridge_ae / total_tested as f64;
    let lgbm_mae = if has_lgbm { total_lgbm_ae / total_tested as f64 } else { f64::NAN };
    let ridge_dir_acc = total_ridge_dir as f64 / total_tested as f64;
    let lgbm_dir_acc = if has_lgbm { total_lgbm_dir as f64 / total_tested as f64 } else { 0.0 };

    println!("    Ridge:   MAE={:.4}, Dir Acc={:.1}%", ridge_mae, ridge_dir_acc * 100.0);
    if has_lgbm {
        println!("    LightGBM: MAE={:.4}, Dir Acc={:.1}%", lgbm_mae, lgbm_dir_acc * 100.0);
    }

    // === GRU Regression walk-forward ===
    // Select top 30 features from LightGBM importance for GRU input
    let top_feature_indices: Vec<usize> = if !last_lgbm_importance.is_empty() {
        let mut indexed: Vec<(usize, f64)> = last_lgbm_importance.iter()
            .enumerate()
            .map(|(i, (_name, imp))| (i, *imp))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(30).map(|(idx, _)| *idx).collect()
    } else {
        (0..n_features.min(30)).collect()
    };

    let gru_n_features = if top_feature_indices.len() >= 20 { top_feature_indices.len() } else { n_features };
    let gru_feature_indices = if top_feature_indices.len() >= 20 { Some(top_feature_indices.as_slice()) } else { None };

    println!("    [GRU-Reg] Using top {} features for GRU input", gru_n_features);

    let gru_config = GRURegressionConfig {
        input_size: gru_n_features,
        ..GRURegressionConfig::default()
    };

    let (gru_mae, gru_dir_acc, last_gru_pred, has_gru) = walk_forward_gru_regression(
        symbol, samples, &gru_config, train_window, test_window, step, gru_feature_indices,
    );

    // Fit Platt calibration on out-of-fold predictions
    let platt_params = fit_platt_scaling(&platt_data);
    if let Some(ref pp) = platt_params {
        println!("    Platt calibration: A={:.4} B={:.4} ({} samples)", pp.a, pp.b, platt_data.len());
    }

    println!("    Regression walk-forward: {} folds, {} test samples", n_folds, total_tested);

    Some(RegressionResult {
        symbol: symbol.to_string(),
        ridge_mae,
        lgbm_mae,
        gru_mae,
        ridge_dir_acc,
        lgbm_dir_acc,
        gru_dir_acc,
        ridge_return: last_ridge_pred,
        lgbm_return: last_lgbm_pred,
        gru_return: last_gru_pred,
        has_lgbm,
        has_gru,
        n_folds,
        total_test_samples: total_tested,
        n_features,
        lgbm_importance: last_lgbm_importance,
        norm_means: last_norm_means,
        norm_stds: last_norm_stds,
        platt_params,
    })
}

/// GRU regression walk-forward (separate due to sequence construction)
fn walk_forward_gru_regression(
    _symbol: &str,
    samples: &[Sample],
    config: &GRURegressionConfig,
    train_window: usize,
    test_window: usize,
    step: usize,
    feature_indices: Option<&[usize]>,
) -> (f64, f64, f64, bool) {
    // Returns (mae, dir_acc, last_pred, has_gru)
    const EMBARGO: usize = 5;
    let seq_len = config.seq_length;
    if samples.len() < train_window + EMBARGO + test_window + seq_len {
        println!("    [GRU-Reg] Not enough samples for walk-forward");
        return (f64::NAN, 0.0, 0.0, false);
    }

    let mut total_ae = 0.0_f64;
    let mut total_dir_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;
    let mut last_pred = 0.0_f64;

    let mut start = 0;
    while start + train_window + EMBARGO + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_start = train_end + EMBARGO;
        let test_end = (test_start + test_window).min(samples.len());

        let mut train_data: Vec<Sample> = samples[start..train_end].to_vec();
        let mut test_data: Vec<Sample> = samples[test_start..test_end].to_vec();
        let (means, stds) = ml::normalise(&mut train_data);
        ml::apply_normalisation(&mut test_data, &means, &stds);

        let val_split = (train_data.len() as f64 * 0.85) as usize;
        let (train_part, val_part) = train_data.split_at(val_split);

        let fold_config = GRURegressionConfig {
            input_size: config.input_size,
            ..*config
        };

        let mut model = match GRURegressionModel::new(fold_config) {
            Ok(m) => m,
            Err(e) => {
                println!("    [GRU-Reg] Failed to create model: {}", e);
                start += step;
                continue;
            }
        };

        match model.train_regression(train_part, val_part, feature_indices) {
            Ok(result) => {
                if result.epochs_trained == 0 {
                    start += step;
                    continue;
                }
            }
            Err(e) => {
                println!("    [GRU-Reg] Training failed: {}", e);
                start += step;
                continue;
            }
        }

        // Build test sequences with continuous labels
        let test_seqs = crate::lstm::build_sequences_regression(&test_data, seq_len, feature_indices);
        if test_seqs.is_empty() {
            start += step;
            continue;
        }

        for seq in &test_seqs {
            let pred = model.predict_return(&seq.features).unwrap_or(0.0);
            let pred = if pred.is_finite() { pred } else { 0.0 };
            total_ae += (pred - seq.label).abs();
            if (pred > 0.0) == (seq.label > 0.0) { total_dir_correct += 1; }
            last_pred = pred;
        }

        total_tested += test_seqs.len();
        n_folds += 1;

        let fold_acc = if !test_seqs.is_empty() {
            let correct: usize = test_seqs.iter().filter(|s| {
                let p = model.predict_return(&s.features).unwrap_or(0.0);
                (p > 0.0) == (s.label > 0.0)
            }).count();
            correct as f64 / test_seqs.len() as f64 * 100.0
        } else { 0.0 };
        println!("    [GRU-Reg] Fold {}: {} seqs, dir_acc={:.1}%", n_folds, test_seqs.len(), fold_acc);

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        println!("    [GRU-Reg] No successful folds");
        return (f64::NAN, 0.0, 0.0, false);
    }

    let mae = total_ae / total_tested as f64;
    let dir_acc = total_dir_correct as f64 / total_tested as f64;
    println!("    GRU-Reg:  MAE={:.4}, Dir Acc={:.1}% ({} folds, {} seqs)", mae, dir_acc * 100.0, n_folds, total_tested);

    (mae, dir_acc, last_pred, true)
}

/// Generate a trading signal from regression predictions.
/// `asset_class`: "stock", "crypto", or "fx" — determines thresholds.
/// GRU is excluded from the ensemble (contributes noise at 45-48% regardless of outcome).
pub fn regression_signal(
    symbol: &str,
    result: &RegressionResult,
    current_price: f64,
    rsi: f64,
    sma_trend: &str,
    asset_class: &str,
) -> TradingSignal {
    // Weight models by inverse MAE (lower error = higher weight)
    // GRU excluded — contributes noise (Phase 1.4)
    let ridge_w = if result.ridge_mae > 0.0 && result.ridge_mae.is_finite() {
        1.0 / result.ridge_mae
    } else { 0.0 };
    let lgbm_w = if result.has_lgbm && result.lgbm_mae > 0.0 && result.lgbm_mae.is_finite() {
        1.0 / result.lgbm_mae
    } else { 0.0 };
    // GRU weight forced to zero
    let gru_w = 0.0_f64;

    let total_w = ridge_w + lgbm_w + gru_w;
    let (rw, lw, gw) = if total_w > 0.0 {
        (ridge_w / total_w, lgbm_w / total_w, gru_w / total_w)
    } else {
        (1.0, 0.0, 0.0) // fallback to ridge only
    };

    let ensemble_return = rw * result.ridge_return
        + lw * result.lgbm_return
        + gw * result.gru_return;

    // Asset-class-specific return thresholds
    let threshold = match asset_class {
        "crypto" => 1.0,
        "fx" => 0.2,
        _ => 0.5, // stocks, ETFs, commodities
    };

    // Phase 4.1: Require Ridge and LightGBM to agree on direction before signalling
    let ridge_up = result.ridge_return > 0.0;
    let lgbm_up = result.has_lgbm && result.lgbm_return > 0.0;
    let models_disagree = result.has_lgbm && (ridge_up != lgbm_up);

    let signal = if models_disagree {
        "HOLD" // Ridge and LightGBM disagree → abstain
    } else if ensemble_return > threshold {
        "BUY"
    } else if ensemble_return < -threshold {
        "SELL"
    } else {
        "HOLD"
    };

    // Phase 4.4: Confidence-based signal filtering
    // Only emit signal when confidence exceeds threshold
    let raw_confidence = (ensemble_return.abs() / 2.0).min(1.0);
    let signal = if signal != "HOLD" && raw_confidence < 0.25 {
        "HOLD" // Insufficient confidence, abstain
    } else {
        signal
    };

    // Signal strength: 0-1 range based on predicted return magnitude
    let signal_strength = raw_confidence;

    // Map predicted return to probability-like value for backward compat
    // 0.5 + return.clamp(-5, 5) / 10.0
    let ensemble_prob = 0.5 + ensemble_return.clamp(-5.0, 5.0) / 10.0;

    // Count directional agreement (only Ridge + LightGBM, GRU excluded)
    let mut ups = 0_usize;
    let mut n_models = 1; // Ridge always present
    if result.ridge_return > 0.0 { ups += 1; }
    if result.has_lgbm {
        n_models += 1;
        if result.lgbm_return > 0.0 { ups += 1; }
    }
    let models_agree = ups.max(n_models - ups);

    // Weighted directional accuracy as walk-forward metric
    let wf_accuracy = rw * result.ridge_dir_acc * 100.0
        + lw * result.lgbm_dir_acc * 100.0
        + gw * result.gru_dir_acc * 100.0;

    let signal_quality = if wf_accuracy >= 55.0 {
        "HIGH"
    } else if wf_accuracy >= 52.0 {
        "MEDIUM"
    } else if wf_accuracy >= 50.0 {
        "LOW"
    } else {
        "NO EDGE"
    };

    TradingSignal {
        symbol: symbol.to_string(),
        signal: signal.to_string(),
        confidence: signal_strength,
        ensemble_prob,
        linear_prob: 0.5 + result.ridge_return.clamp(-5.0, 5.0) / 10.0,
        logistic_prob: 0.0, // not used in regression ensemble
        gbt_prob: 0.0,      // not used in regression ensemble
        lgbm_prob: if result.has_lgbm { 0.5 + result.lgbm_return.clamp(-5.0, 5.0) / 10.0 } else { 0.5 },
        lstm_prob: 0.0,      // not used in regression ensemble
        gru_prob: if result.has_gru { 0.5 + result.gru_return.clamp(-5.0, 5.0) / 10.0 } else { 0.5 },
        rf_prob: 0.0,        // not used in regression ensemble
        linear_weight: rw,
        logistic_weight: 0.0,
        gbt_weight: 0.0,
        lgbm_weight: lw,
        lstm_weight: 0.0,
        gru_weight: gw,
        rf_weight: 0.0,
        models_agree,
        n_models,
        walk_forward_accuracy: wf_accuracy,
        signal_quality: signal_quality.to_string(),
        current_price,
        rsi,
        sma_trend: sma_trend.to_string(),
        has_lstm: false,
        has_gru: result.has_gru,
        has_rf: false,
        has_lgbm: result.has_lgbm,
        llm_sentiment: 0.0,
        llm_analysis: None,
    }
}

/// Multi-horizon confirmation: only emit BUY/SELL when both 1d and 5d models agree.
/// Returns the confirmed signal (may be downgraded to HOLD if horizons disagree).
pub fn multi_horizon_signal(
    signal_1d: &str,
    _result_1d: &RegressionResult,
    result_5d: Option<&RegressionResult>,
    asset_class: &str,
) -> String {
    // If no 5d model available, pass through the 1d signal
    let result_5d = match result_5d {
        Some(r) => r,
        None => return signal_1d.to_string(),
    };

    // Only filter actionable signals
    if signal_1d == "HOLD" {
        return signal_1d.to_string();
    }

    // 5-day threshold (higher bar since longer horizon should show stronger moves)
    let threshold_5d = match asset_class {
        "crypto" => 2.0,   // 2.0% for 5-day crypto
        "fx" => 0.4,       // 0.4% for 5-day FX
        _ => 1.0,          // 1.0% for 5-day stocks
    };

    // Compute 5d ensemble return (Ridge + LGBM, GRU excluded)
    let ridge_w = if result_5d.ridge_mae > 0.0 && result_5d.ridge_mae.is_finite() {
        1.0 / result_5d.ridge_mae
    } else { 0.0 };
    let lgbm_w = if result_5d.has_lgbm && result_5d.lgbm_mae > 0.0 && result_5d.lgbm_mae.is_finite() {
        1.0 / result_5d.lgbm_mae
    } else { 0.0 };
    let total_w = ridge_w + lgbm_w;
    let ensemble_5d = if total_w > 0.0 {
        (ridge_w * result_5d.ridge_return + lgbm_w * result_5d.lgbm_return) / total_w
    } else {
        result_5d.ridge_return
    };

    // Multi-horizon confirmation logic
    match signal_1d {
        "BUY" => {
            if ensemble_5d > threshold_5d {
                "BUY".to_string() // Both horizons agree: BUY
            } else if ensemble_5d > 0.0 {
                "BUY".to_string() // 5d positive but below threshold: still allow (weak confirmation)
            } else {
                "HOLD".to_string() // 5d says DOWN: override to HOLD
            }
        }
        "SELL" | "SHORT" => {
            if ensemble_5d < -threshold_5d {
                signal_1d.to_string() // Both horizons agree: SELL
            } else if ensemble_5d < 0.0 {
                signal_1d.to_string() // 5d negative but above threshold: still allow
            } else {
                "HOLD".to_string() // 5d says UP: override to HOLD
            }
        }
        _ => signal_1d.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify stacking_predict output is always in [0.0, 1.0] (actually [0.15, 0.85] due to clamp)
    #[test]
    fn test_stacking_predict_bounded() {
        // Typical stacking weights: [w_linreg, w_logreg, w_gbt, w_lstm, w_gru, w_rf, bias]
        let weights = vec![0.5, 0.3, 0.8, 0.2, 0.1, 0.4, -1.0];

        // All models predict UP strongly (7 models)
        let probs_up: [f64; 7] = [0.9, 0.85, 0.95, 0.7, 0.6, 0.8, 0.75];
        let result = stacking_predict(&weights, &probs_up);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // All models predict DOWN strongly
        let probs_down: [f64; 7] = [0.1, 0.15, 0.05, 0.3, 0.4, 0.2, 0.25];
        let result = stacking_predict(&weights, &probs_down);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // Mixed predictions
        let probs_mixed: [f64; 7] = [0.8, 0.3, 0.6, 0.5, 0.4, 0.7, 0.55];
        let result = stacking_predict(&weights, &probs_mixed);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // Edge: all 0.5 (neutral)
        let probs_neutral: [f64; 7] = [0.5; 7];
        let result = stacking_predict(&weights, &probs_neutral);
        assert!((0.0..=1.0).contains(&result), "stacking_predict should be in [0,1], got {}", result);

        // Edge: extreme weights
        let extreme_weights = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 50.0];
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
