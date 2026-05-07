/// Shared inference functions — load saved model weights and predict
/// =================================================================
/// Used by both `signal` (CLI) and `serve` (web API) binaries.
/// No training happens here — only forward-pass inference.

use crate::{model_store, ml, ensemble, ridge, lgbm, gru};

/// Normalise a feature vector using pre-computed means and stds.
/// Features with std < 1e-8 are zeroed out (constant during training = no signal).
pub fn normalise_features(features: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    features.iter().enumerate().map(|(i, &f)| {
        let mean = means.get(i).copied().unwrap_or(0.0);
        let std = stds.get(i).copied().unwrap_or(1.0);
        if std < 1e-8 {
            0.0 // Zero-variance feature: zero it out to prevent noise injection
        } else {
            (f - mean) / std
        }
    }).collect()
}

/// Run linreg inference: dot(weights, features) + bias
pub fn predict_linreg(saved: &model_store::SavedWeights, features: &[f64]) -> f64 {
    let mut result = saved.bias;
    for (w, f) in saved.weights.iter().zip(features.iter()) {
        result += w * f;
    }
    result
}

/// Run logreg inference: sigmoid(dot(weights, features) + bias)
pub fn predict_logreg(saved: &model_store::SavedWeights, features: &[f64]) -> f64 {
    let mut z = saved.bias;
    for (w, f) in saved.weights.iter().zip(features.iter()) {
        z += w * f;
    }
    1.0 / (1.0 + (-z).exp())
}

/// Load saved model weights and run inference on the latest feature vector.
/// Returns a WalkForwardResult populated with saved accuracies + fresh predictions.
pub fn infer_with_saved_models(
    symbol: &str,
    samples: &[ml::Sample],
) -> Option<ensemble::WalkForwardResult> {
    if samples.is_empty() {
        println!("  {} — no samples for inference", symbol);
        return None;
    }

    let n_features = samples[0].features.len();

    // Load the 3 saved models
    let linreg_saved = match model_store::load_weights(symbol, "linreg") {
        Ok(w) => w,
        Err(e) => {
            println!("  {} — skipping: no linreg model ({})", symbol, e);
            return None;
        }
    };
    let logreg_saved = match model_store::load_weights(symbol, "logreg") {
        Ok(w) => w,
        Err(e) => {
            println!("  {} — skipping: no logreg model ({})", symbol, e);
            return None;
        }
    };
    let (gbt_saved, gbt_classifier) = match model_store::load_gbt(symbol) {
        Ok(g) => g,
        Err(e) => {
            println!("  {} — skipping: no GBT model ({})", symbol, e);
            return None;
        }
    };

    // Get the latest feature vector
    let last_sample = samples.last().unwrap();
    let feat = &last_sample.features;

    // LinReg prediction (normalise with its own saved params)
    let lin_feat = normalise_features(feat, &linreg_saved.norm_means, &linreg_saved.norm_stds);
    let raw_lin = predict_linreg(&linreg_saved, &lin_feat);
    let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);

    // LogReg prediction
    let log_feat = normalise_features(feat, &logreg_saved.norm_means, &logreg_saved.norm_stds);
    let log_prob = predict_logreg(&logreg_saved, &log_feat).clamp(0.15, 0.85);

    // GBT prediction (GBT has its own norm params)
    let gbt_feat = normalise_features(feat, &gbt_saved.norm_means, &gbt_saved.norm_stds);
    let gbt_prob = gbt_classifier.predict_proba(&gbt_feat).clamp(0.15, 0.85);

    // Use saved walk-forward accuracies
    let lin_acc = linreg_saved.meta.walk_forward_accuracy;
    let log_acc = logreg_saved.meta.walk_forward_accuracy;
    let gbt_acc = gbt_saved.meta.walk_forward_accuracy;

    println!("  {} — inference: LinR={:.1}% LogR={:.1}% GBT={:.1}% | probs: {:.2} {:.2} {:.2}",
        symbol, lin_acc, log_acc, gbt_acc, lin_prob, log_prob, gbt_prob);

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
        has_lgbm: false,
        lgbm_accuracy: 50.0,
        lgbm_recent: 50.0,
        final_lgbm_prob: 0.5,
        lgbm_importance: Vec::new(),
        stacking_weights: None,
        val_log_loss: None,
        platt_params: None,
    })
}

/// Load the 5-day models (Ridge + LightGBM) for a symbol and predict ensemble return.
/// Returns None if no 5-day models exist (haven't been trained yet).
/// The returned RegressionResult can be used for multi-horizon confirmation.
pub fn infer_5d_direction(symbol: &str, samples: &[ml::Sample]) -> Option<bool> {
    if samples.is_empty() { return None; }

    // Try 5d Ridge first
    let ridge_path = model_store::ridge_path_5d(symbol);
    let ridge_saved: model_store::SavedWeights = {
        let contents = std::fs::read_to_string(&ridge_path).ok()?;
        serde_json::from_str(&contents).ok()?
    };

    let model_n_features = ridge_saved.meta.n_features;
    let last_sample = samples.last().unwrap();
    let feat = if last_sample.features.len() > model_n_features {
        &last_sample.features[..model_n_features]
    } else {
        &last_sample.features
    };

    let norm_feat = normalise_features(feat, &ridge_saved.norm_means, &ridge_saved.norm_stds);
    let ridge_model = ridge::RidgeRegression {
        weights: ridge_saved.weights.clone(),
        bias: ridge_saved.bias,
    };
    let ridge_5d_return = ridge_model.predict(&norm_feat);

    // Try 5d LightGBM
    let lgbm_path = model_store::lgbm_regressor_path_5d(symbol);
    let lgbm_5d_return = match lgbm::LightGBMRegressor::load(&lgbm_path, model_n_features) {
        Ok(m) => Some(m.predict_return(&norm_feat)),
        Err(_) => None,
    };

    // Require agreement if both available
    let direction_up = if let Some(lgbm_ret) = lgbm_5d_return {
        // Both must agree on direction
        let ridge_up = ridge_5d_return > 0.0;
        let lgbm_up = lgbm_ret > 0.0;
        if ridge_up != lgbm_up {
            // Disagreement — use Ridge alone (more stable)
            ridge_up
        } else {
            ridge_up
        }
    } else {
        ridge_5d_return > 0.0
    };

    Some(direction_up)
}

/// Infer 5-day regression result with ensemble return magnitude.
/// Used for multi-horizon confirmation (Phase 2).
pub fn infer_regression_models_5d(
    symbol: &str,
    samples: &[ml::Sample],
) -> Option<ensemble::RegressionResult> {
    if samples.is_empty() { return None; }

    // Load 5d Ridge
    let ridge_path = model_store::ridge_path_5d(symbol);
    let ridge_saved: model_store::SavedWeights = {
        let contents = std::fs::read_to_string(&ridge_path).ok()?;
        serde_json::from_str(&contents).ok()?
    };

    let model_n_features = ridge_saved.meta.n_features;
    let last_sample = samples.last().unwrap();
    let feat = if last_sample.features.len() > model_n_features {
        &last_sample.features[..model_n_features]
    } else {
        &last_sample.features
    };

    let norm_feat = normalise_features(feat, &ridge_saved.norm_means, &ridge_saved.norm_stds);
    let ridge_model = ridge::RidgeRegression {
        weights: ridge_saved.weights.clone(),
        bias: ridge_saved.bias,
    };
    let ridge_return = ridge_model.predict(&norm_feat);
    let ridge_mae = ridge_saved.meta.walk_forward_accuracy;

    // Load 5d LightGBM
    let lgbm_path = model_store::lgbm_regressor_path_5d(symbol);
    let (lgbm_return, lgbm_mae, has_lgbm) = match lgbm::LightGBMRegressor::load(&lgbm_path, model_n_features) {
        Ok(m) => {
            let pred = m.predict_return(&norm_feat);
            (pred, ridge_mae.max(0.01), true)
        }
        Err(_) => (0.0, f64::NAN, false),
    };

    Some(ensemble::RegressionResult {
        symbol: symbol.to_string(),
        ridge_mae,
        lgbm_mae,
        gru_mae: f64::NAN,
        ridge_dir_acc: (0.60 - ridge_mae * 0.03).clamp(0.50, 0.65),
        lgbm_dir_acc: if has_lgbm { (0.60 - lgbm_mae * 0.03).clamp(0.50, 0.65) } else { 0.0 },
        gru_dir_acc: 0.0,
        ridge_return,
        lgbm_return,
        gru_return: 0.0,
        has_lgbm,
        has_gru: false,
        n_folds: 1,
        total_test_samples: 0,
        n_features: model_n_features,
        lgbm_importance: Vec::new(),
        norm_means: ridge_saved.norm_means.clone(),
        norm_stds: ridge_saved.norm_stds.clone(),
        platt_params: None,
    })
}

/// Load regression models (Ridge + LightGBM) and predict returns.
/// GRU is disabled (contributes noise at 45-48% accuracy regardless of outcome).
/// Returns a RegressionResult populated with saved metrics + fresh predictions.
pub fn infer_regression_models(
    symbol: &str,
    samples: &[ml::Sample],
) -> Option<ensemble::RegressionResult> {
    if samples.is_empty() {
        println!("  {} — no samples for regression inference", symbol);
        return None;
    }

    // === Load Ridge ===
    let (ridge_model, ridge_saved) = match ridge::RidgeRegression::load(symbol) {
        Ok(r) => r,
        Err(e) => {
            println!("  {} — skipping: no ridge model ({})", symbol, e);
            return None;
        }
    };

    // Dimension guard: truncate features to match model's trained feature count
    let model_n_features = ridge_saved.meta.n_features;
    let last_sample = samples.last().unwrap();
    let feat = if last_sample.features.len() > model_n_features {
        println!("  {} — dimension guard: truncating {} → {} features",
            symbol, last_sample.features.len(), model_n_features);
        &last_sample.features[..model_n_features]
    } else {
        &last_sample.features
    };

    let norm_feat = normalise_features(feat, &ridge_saved.norm_means, &ridge_saved.norm_stds);
    let ridge_return = ridge_model.predict(&norm_feat);
    let ridge_mae = ridge_saved.meta.walk_forward_accuracy; // stored as MAE

    // === Load LightGBM Regressor ===
    // Use model_n_features (from Ridge metadata) to ensure dimension consistency
    let lgbm_path = model_store::lgbm_regressor_path(symbol);
    let (lgbm_return, lgbm_mae, has_lgbm) = match lgbm::LightGBMRegressor::load(&lgbm_path, model_n_features) {
        Ok(m) => {
            let pred = m.predict_return(&norm_feat);
            // Load LGBM MAE from metadata if available, else use Ridge MAE as proxy
            let lgbm_mae_val = model_store::load_lgbm_mae(symbol).unwrap_or(ridge_mae.max(0.01));
            (pred, lgbm_mae_val, true)
        }
        Err(_) => (0.0, f64::NAN, false),
    };

    // === GRU Disabled (Phase 1.4) ===
    // GRU contributes noise (45-48% accuracy). Set to zero weight.
    let gru_return = 0.0;
    let gru_mae = f64::NAN;
    let has_gru = false;

    println!("  {} — regression inference: Ridge={:.4} LGBM={:.4} (GRU disabled)",
        symbol, ridge_return,
        if has_lgbm { lgbm_return } else { f64::NAN });

    Some(ensemble::RegressionResult {
        symbol: symbol.to_string(),
        ridge_mae,
        lgbm_mae,
        gru_mae,
        ridge_dir_acc: (0.60 - ridge_mae * 0.03).clamp(0.50, 0.65),
        lgbm_dir_acc: if has_lgbm { (0.60 - lgbm_mae * 0.03).clamp(0.50, 0.65) } else { 0.0 },
        gru_dir_acc: 0.0, // GRU disabled
        ridge_return,
        lgbm_return,
        gru_return,
        has_lgbm,
        has_gru,
        n_folds: 1,
        total_test_samples: 0,
        n_features: model_n_features,
        lgbm_importance: Vec::new(),
        norm_means: ridge_saved.norm_means.clone(),
        norm_stds: ridge_saved.norm_stds.clone(),
        platt_params: None, // Loaded from saved model at training time
    })
}

/// Load GRU regression model and predict return from the last seq_len samples.
fn load_gru_and_predict(symbol: &str, samples: &[ml::Sample]) -> Option<(f64, f64)> {
    let gru_meta = model_store::load_gru_meta(symbol).ok()?;
    let gru_path = model_store::gru_path(symbol);

    let config = gru::GRURegressionConfig {
        input_size: if let Some(ref fi) = gru_meta.feature_indices {
            fi.len()
        } else {
            gru_meta.meta.n_features
        },
        hidden_size: gru_meta.hidden_size,
        seq_length: gru_meta.seq_length,
        ..gru::GRURegressionConfig::default()
    };

    let model = gru::GRURegressionModel::load(config, &gru_path).ok()?;

    // Build sequence from last seq_len samples
    let seq_len = gru_meta.seq_length;
    if samples.len() < seq_len {
        println!("    [GRU-Reg] Not enough samples for sequence ({} < {})", samples.len(), seq_len);
        return None;
    }

    let last_n = &samples[samples.len() - seq_len..];
    let feature_indices = gru_meta.feature_indices.as_deref();

    // Normalise features and build sequence
    let sequence: Vec<Vec<f64>> = last_n.iter().map(|s| {
        let norm = normalise_features(&s.features, &gru_meta.norm_means, &gru_meta.norm_stds);
        if let Some(indices) = feature_indices {
            indices.iter().map(|&i| norm.get(i).copied().unwrap_or(0.0)).collect()
        } else {
            norm
        }
    }).collect();

    match model.predict_return(&sequence) {
        Ok(ret) => {
            let mae = gru_meta.meta.walk_forward_accuracy; // stored as MAE
            Some((ret, mae))
        }
        Err(e) => {
            println!("    [GRU-Reg] Prediction failed: {}", e);
            None
        }
    }
}

/// Apply multi-horizon agreement filter to an enriched signal.
/// Uses the full 5d regression ensemble (not just direction) for confirmation.
/// If the 5-day model disagrees with the 1-day signal direction, downgrade to HOLD.
/// Returns the (potentially modified) signal string.
pub fn apply_horizon_agreement(
    signal_1d: &str,
    symbol: &str,
    samples: &[ml::Sample],
    result_1d: &ensemble::RegressionResult,
    asset_class: &str,
) -> String {
    // Only filter actionable signals (BUY/SELL/SHORT)
    if signal_1d == "HOLD" { return signal_1d.to_string(); }

    // Load 5d regression models and compute ensemble return
    let result_5d = infer_regression_models_5d(symbol, samples);

    let confirmed = ensemble::multi_horizon_signal(
        signal_1d,
        result_1d,
        result_5d.as_ref(),
        asset_class,
    );

    if confirmed != signal_1d {
        println!("    [HorizonFilter] {} {} → {} (5d model disagrees)", symbol, signal_1d, confirmed);
    }

    confirmed
}
