/// Ridge Regression — Closed-Form Linear Model
/// =============================================
/// Solves: w = (X'X + αI)^{-1} X'y via Cholesky decomposition.
///
/// No epochs, no learning rate — single exact matrix solve.
/// Strong regularisation (α = 10.0) appropriate for 84 features / ~1000 samples.
///
/// Save/load reuses the existing `SavedWeights` JSON format from model_store.

use nalgebra::{DMatrix, DVector};
use crate::ml::Sample;
use crate::model_store::{self, SavedWeights};

/// Trained Ridge regression model
pub struct RidgeRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl RidgeRegression {
    /// Train Ridge regression via closed-form solution.
    /// `x` is a slice of feature vectors, `y` is the target (continuous returns).
    /// `alpha` is the L2 regularisation strength.
    pub fn train(x: &[Vec<f64>], y: &[f64], alpha: f64) -> Result<Self, String> {
        let n = x.len();
        if n < 2 {
            return Err("Ridge: need at least 2 samples".to_string());
        }
        let p = x[0].len();
        if p == 0 {
            return Err("Ridge: zero features".to_string());
        }

        // Build design matrix with bias column: [features | 1]
        let cols = p + 1;
        let mut x_data = Vec::with_capacity(n * cols);
        for row in x {
            if row.len() != p {
                return Err(format!("Ridge: inconsistent feature count ({} vs {})", row.len(), p));
            }
            for &val in row {
                x_data.push(if val.is_finite() { val } else { 0.0 });
            }
            x_data.push(1.0); // bias column
        }

        let x_mat = DMatrix::from_row_slice(n, cols, &x_data);
        let y_vec = DVector::from_iterator(n, y.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));

        // X'X + αI (don't regularise the bias column)
        let xtx = x_mat.transpose() * &x_mat;
        let mut reg = DMatrix::identity(cols, cols) * alpha;
        reg[(cols - 1, cols - 1)] = 0.0; // no regularisation on bias
        let a = xtx + reg;

        // X'y
        let xty = x_mat.transpose() * &y_vec;

        // Solve via Cholesky decomposition
        let cholesky = a.cholesky().ok_or("Ridge: Cholesky decomposition failed (singular matrix)")?;
        let w = cholesky.solve(&xty);

        let weights: Vec<f64> = w.iter().take(p).copied().collect();
        let bias = w[p];

        println!("    [Ridge] Trained: {} features, α={}, bias={:.4}, w_norm={:.4}",
            p, alpha, bias, weights.iter().map(|w| w * w).sum::<f64>().sqrt());

        Ok(RidgeRegression { weights, bias })
    }

    /// Train from Sample slices (convenience wrapper)
    pub fn train_from_samples(samples: &[Sample], alpha: f64) -> Result<Self, String> {
        let x: Vec<Vec<f64>> = samples.iter().map(|s| s.features.clone()).collect();
        let y: Vec<f64> = samples.iter().map(|s| s.label).collect();
        Self::train(&x, &y, alpha)
    }

    /// Predict return for a single feature vector
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, f) in self.weights.iter().zip(features.iter()) {
            result += w * f;
        }
        result
    }

    /// Batch predict
    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Save model using existing SavedWeights format
    pub fn save(
        &self,
        symbol: &str,
        n_features: usize,
        train_samples: usize,
        mae: f64,
        norm_means: &[f64],
        norm_stds: &[f64],
    ) -> Result<(), String> {
        model_store::save_weights(
            symbol, "ridge",
            &self.weights, self.bias,
            n_features, train_samples, mae,
            norm_means, norm_stds,
        )
    }

    /// Save model with a custom name (used for 5d models: "5d_<symbol>")
    pub fn save_as(
        &self,
        name: &str,
        n_features: usize,
        train_samples: usize,
        mae: f64,
        norm_means: &[f64],
        norm_stds: &[f64],
    ) -> Result<(), String> {
        model_store::save_weights(
            name, "ridge",
            &self.weights, self.bias,
            n_features, train_samples, mae,
            norm_means, norm_stds,
        )
    }

    /// Load model from SavedWeights format
    pub fn load(symbol: &str) -> Result<(Self, SavedWeights), String> {
        let saved = model_store::load_weights(symbol, "ridge")?;
        let model = RidgeRegression {
            weights: saved.weights.clone(),
            bias: saved.bias,
        };
        Ok((model, saved))
    }
}

/// Evaluate Ridge on test data: returns (MAE, directional accuracy)
pub fn evaluate_ridge(
    model: &RidgeRegression,
    test_features: &[Vec<f64>],
    test_labels: &[f64],
) -> (f64, f64) {
    if test_features.is_empty() {
        return (f64::NAN, 0.0);
    }

    let mut total_ae = 0.0;
    let mut dir_correct = 0usize;

    for (feat, &actual) in test_features.iter().zip(test_labels.iter()) {
        let pred = model.predict(feat);
        total_ae += (pred - actual).abs();
        // Directional accuracy: predicted sign matches actual sign
        if (pred > 0.0) == (actual > 0.0) {
            dir_correct += 1;
        }
    }

    let mae = total_ae / test_features.len() as f64;
    let dir_acc = dir_correct as f64 / test_features.len() as f64;
    (mae, dir_acc)
}
