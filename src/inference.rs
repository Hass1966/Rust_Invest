/// Shared inference functions — load saved model weights and predict
/// =================================================================
/// Used by both `signal` (CLI) and `serve` (web API) binaries.
/// No training happens here — only forward-pass inference.

use crate::{model_store, ml, ensemble};

/// Normalise a feature vector using pre-computed means and stds
pub fn normalise_features(features: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    features.iter().enumerate().map(|(i, &f)| {
        let mean = means.get(i).copied().unwrap_or(0.0);
        let std = stds.get(i).copied().unwrap_or(1.0);
        if std == 0.0 { f - mean } else { (f - mean) / std }
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
        stacking_weights: None,
        val_log_loss: None,
    })
}
