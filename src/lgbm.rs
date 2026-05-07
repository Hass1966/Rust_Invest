/// LightGBM Wrapper — Model #7 in the Ensemble
/// =============================================
/// Wraps the lightgbm3 crate to match the GBT interface used by ensemble.rs.
///
/// Key advantages over custom GBT:
///   - Leaf-wise (best-first) tree growth instead of level-wise
///   - Histogram-based splitting (O(features × bins) instead of O(features × samples))
///   - Native L1/L2 regularization (lambda_l1, lambda_l2)
///   - Column subsampling (feature_fraction) per tree
///   - Row subsampling (bagging_fraction)
///   - Early stopping on validation set

use lightgbm3::{Booster, Dataset, ImportanceType};
use serde_json::json;

/// Configuration for LightGBM training
#[derive(Debug, Clone)]
pub struct LGBMConfig {
    pub num_iterations: usize,
    pub max_depth: i32,
    pub learning_rate: f64,
    pub num_leaves: i32,
    pub min_data_in_leaf: i32,
    pub feature_fraction: f64,
    pub bagging_fraction: f64,
    pub bagging_freq: i32,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub early_stopping_rounds: i32,
    pub verbose: i32,
}

impl Default for LGBMConfig {
    fn default() -> Self {
        LGBMConfig {
            num_iterations: 500,
            max_depth: 6,
            learning_rate: 0.05,
            num_leaves: 63,
            min_data_in_leaf: 20,
            feature_fraction: 0.7,
            bagging_fraction: 0.8,
            bagging_freq: 1,
            lambda_l1: 0.1,
            lambda_l2: 1.0,
            early_stopping_rounds: 20,
            verbose: -1,
        }
    }
}

/// Trained LightGBM classifier
pub struct LightGBMClassifier {
    booster: Booster,
    n_features: usize,
}

impl LightGBMClassifier {
    /// Train a binary classifier on (x_train, y_train) with optional validation set.
    /// y_train values: 1.0 = positive (up), 0.0 = negative (down).
    pub fn train(
        x_train: &[Vec<f64>],
        y_train: &[f64],
        sample_weights: Option<&[f64]>,
        x_val: Option<&[Vec<f64>]>,
        y_val: Option<&[f64]>,
        config: &LGBMConfig,
    ) -> Result<Self, String> {
        let n_features = if x_train.is_empty() { 0 } else { x_train[0].len() };
        let n_train = x_train.len();

        if n_train < 10 {
            return Err("Too few training samples for LightGBM".to_string());
        }

        // Validate: all rows must have identical feature count and no NaN/Inf
        for (i, row) in x_train.iter().enumerate() {
            if row.len() != n_features {
                return Err(format!("LightGBM: train row {} has {} features, expected {}", i, row.len(), n_features));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("LightGBM: train row {} contains NaN/Inf", i));
            }
        }

        // Flatten features into row-major Vec<Vec<f64>> for lightgbm3
        let labels: Vec<f32> = y_train.iter().map(|&y| y as f32).collect();

        // Build training dataset
        let mut train_dataset = Dataset::from_vec_of_vec(
            x_train.to_vec(),
            labels.clone(),
            true,
        ).map_err(|e| format!("LightGBM Dataset error: {}", e))?;

        // Apply sample weights if provided
        if let Some(weights) = sample_weights {
            let w: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
            train_dataset.set_weights(&w)
                .map_err(|e| format!("LightGBM set_weights error: {}", e))?;
        }

        // Compute class weights for balanced training
        let n_positive = y_train.iter().filter(|&&y| y > 0.5).count() as f64;
        let n_negative = n_train as f64 - n_positive;
        let scale_pos_weight = if n_positive > 0.0 && n_negative > 0.0 {
            n_negative / n_positive
        } else {
            1.0
        };

        // Build parameters
        let params = json!({
            "objective": "binary",
            "metric": "binary_logloss",
            "num_iterations": config.num_iterations,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": config.bagging_fraction,
            "bagging_freq": config.bagging_freq,
            "lambda_l1": config.lambda_l1,
            "lambda_l2": config.lambda_l2,
            "scale_pos_weight": scale_pos_weight,
            "verbose": config.verbose,
            "num_threads": 4,
            "seed": 42,
            "force_col_wise": true,
        });

        // Train with or without validation
        let has_valid = matches!((x_val, y_val), (Some(xv), Some(yv)) if !xv.is_empty() && !yv.is_empty());

        let booster = if has_valid {
            let xv = x_val.unwrap();
            let yv = y_val.unwrap();

            // Validate feature counts match and no NaN/Inf — skip gracefully if mismatch
            if !xv.is_empty() && xv[0].len() != n_features {
                eprintln!("[LightGBM] SKIP: validation features ({}) != training features ({}), falling back to no-validation training",
                    xv[0].len(), n_features);
                return Booster::train(train_dataset, &params)
                    .map(|booster| LightGBMClassifier { booster, n_features })
                    .map_err(|e| format!("LightGBM train error: {}", e));
            }
            if xv.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
                return Err("LightGBM: validation data contains NaN/Inf values".to_string());
            }

            let val_labels: Vec<f32> = yv.iter().map(|&y| y as f32).collect();

            // Create validation dataset WITH reference to training dataset.
            // This ensures identical bin mappers, fixing the Fatal error:
            // "Cannot add validation data, since it has different bin mappers"
            let val_flat: Vec<f64> = xv.iter().flat_map(|row| row.iter().copied()).collect();
            let val_dataset = Dataset::from_slice_with_reference(
                &val_flat,
                &val_labels,
                n_features as i32,
                true,
                Some(&train_dataset),
            ).map_err(|e| format!("LightGBM val Dataset error: {}", e))?;

            // Train with early stopping via validation
            let mut params_with_es = params.clone();
            params_with_es["early_stopping_rounds"] = json!(config.early_stopping_rounds);

            Booster::train_with_valid(train_dataset, Some(val_dataset), &params_with_es)
                .map_err(|e| format!("LightGBM train error: {}", e))?
        } else {
            Booster::train(train_dataset, &params)
                .map_err(|e| format!("LightGBM train error: {}", e))?
        };

        Ok(LightGBMClassifier { booster, n_features })
    }

    /// Predict P(up) for a single feature vector
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        // predict_with_params expects a flat &[T] — flatten our single row
        match self.booster.predict_with_params(
            features,
            self.n_features as i32,
            true,
            "num_threads=1",
        ) {
            Ok(results) => {
                if !results.is_empty() {
                    results[0].clamp(0.01, 0.99)
                } else {
                    0.5
                }
            }
            Err(_) => 0.5,
        }
    }

    /// Predict class (true = up, false = down)
    pub fn predict_direction(&self, features: &[f64]) -> bool {
        self.predict_proba(features) > 0.5
    }

    /// Batch predict — more efficient than calling predict_proba in a loop
    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        // Flatten into a single contiguous array for predict_with_params
        let flat: Vec<f64> = features.iter().flat_map(|row| row.iter().copied()).collect();
        match self.booster.predict_with_params(
            &flat,
            self.n_features as i32,
            true,
            "num_threads=4",
        ) {
            Ok(results) => {
                results.iter().map(|&p| p.clamp(0.01, 0.99)).collect()
            }
            Err(_) => vec![0.5; features.len()],
        }
    }

    /// Feature importance — gain-based, normalised
    pub fn feature_importance(&self, feature_names: &[&str]) -> Vec<(String, f64)> {
        let n = self.n_features;
        let importances = match self.booster.feature_importance(ImportanceType::Gain) {
            Ok(imp) => imp,
            Err(_) => vec![0.0; n],
        };

        let total: f64 = importances.iter().sum();
        importances.iter().enumerate()
            .map(|(i, &imp)| {
                let name = if i < feature_names.len() {
                    feature_names[i].to_string()
                } else {
                    format!("Feature_{}", i)
                };
                let norm_imp = if total > 0.0 { imp / total } else { 0.0 };
                (name, norm_imp)
            })
            .collect()
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.booster.save_file(path)
            .map_err(|e| format!("LightGBM save error: {}", e))
    }

    /// Load model from file
    pub fn load(path: &str, n_features: usize) -> Result<Self, String> {
        let booster = Booster::from_file(path)
            .map_err(|e| format!("LightGBM load error: {}", e))?;
        Ok(LightGBMClassifier { booster, n_features })
    }

    /// Number of trees actually trained (may be less than num_iterations due to early stopping)
    pub fn num_trees(&self) -> i32 {
        self.booster.num_iterations()
    }
}

// ════════════════════════════════════════
// LightGBM Regressor — for return prediction
// ════════════════════════════════════════

/// Configuration for LightGBM regression training
#[derive(Debug, Clone)]
pub struct LGBMRegressorConfig {
    pub num_iterations: usize,
    pub max_depth: i32,
    pub learning_rate: f64,
    pub num_leaves: i32,
    pub min_data_in_leaf: i32,
    pub feature_fraction: f64,
    pub bagging_fraction: f64,
    pub bagging_freq: i32,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub early_stopping_rounds: i32,
    pub verbose: i32,
}

impl Default for LGBMRegressorConfig {
    fn default() -> Self {
        LGBMRegressorConfig {
            num_iterations: 300,
            max_depth: 4,
            learning_rate: 0.03,
            num_leaves: 31,
            min_data_in_leaf: 25,
            feature_fraction: 0.6,
            bagging_fraction: 0.75,
            bagging_freq: 1,
            lambda_l1: 0.5,
            lambda_l2: 2.0,
            early_stopping_rounds: 25,
            verbose: -1,
        }
    }
}

/// Trained LightGBM regressor for return prediction
pub struct LightGBMRegressor {
    booster: Booster,
    n_features: usize,
}

impl LightGBMRegressor {
    /// Train a regression model on (x_train, y_train) with optional validation set.
    /// y_train values: continuous returns (percentage).
    pub fn train(
        x_train: &[Vec<f64>],
        y_train: &[f64],
        sample_weights: Option<&[f64]>,
        x_val: Option<&[Vec<f64>]>,
        y_val: Option<&[f64]>,
        config: &LGBMRegressorConfig,
    ) -> Result<Self, String> {
        let n_features = if x_train.is_empty() { 0 } else { x_train[0].len() };
        let n_train = x_train.len();

        if n_train < 10 {
            return Err("Too few training samples for LightGBM regressor".to_string());
        }

        for (i, row) in x_train.iter().enumerate() {
            if row.len() != n_features {
                return Err(format!("LightGBM Reg: train row {} has {} features, expected {}", i, row.len(), n_features));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("LightGBM Reg: train row {} contains NaN/Inf", i));
            }
        }

        let labels: Vec<f32> = y_train.iter().map(|&y| y as f32).collect();

        let mut train_dataset = Dataset::from_vec_of_vec(
            x_train.to_vec(),
            labels.clone(),
            true,
        ).map_err(|e| format!("LightGBM Reg Dataset error: {}", e))?;

        if let Some(weights) = sample_weights {
            let w: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
            train_dataset.set_weights(&w)
                .map_err(|e| format!("LightGBM Reg set_weights error: {}", e))?;
        }

        let params = json!({
            "objective": "regression",
            "metric": "mse",
            "num_iterations": config.num_iterations,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": config.bagging_fraction,
            "bagging_freq": config.bagging_freq,
            "lambda_l1": config.lambda_l1,
            "lambda_l2": config.lambda_l2,
            "verbose": config.verbose,
            "num_threads": 4,
            "seed": 42,
            "force_col_wise": true,
        });

        let has_valid = matches!((x_val, y_val), (Some(xv), Some(yv)) if !xv.is_empty() && !yv.is_empty());

        let booster = if has_valid {
            let xv = x_val.unwrap();
            let yv = y_val.unwrap();

            if !xv.is_empty() && xv[0].len() != n_features {
                eprintln!("[LightGBM Reg] SKIP: validation features ({}) != training features ({})",
                    xv[0].len(), n_features);
                return Booster::train(train_dataset, &params)
                    .map(|booster| LightGBMRegressor { booster, n_features })
                    .map_err(|e| format!("LightGBM Reg train error: {}", e));
            }
            if xv.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
                return Err("LightGBM Reg: validation data contains NaN/Inf values".to_string());
            }

            let val_labels: Vec<f32> = yv.iter().map(|&y| y as f32).collect();
            let val_flat: Vec<f64> = xv.iter().flat_map(|row| row.iter().copied()).collect();
            let val_dataset = Dataset::from_slice_with_reference(
                &val_flat,
                &val_labels,
                n_features as i32,
                true,
                Some(&train_dataset),
            ).map_err(|e| format!("LightGBM Reg val Dataset error: {}", e))?;

            let mut params_with_es = params.clone();
            params_with_es["early_stopping_rounds"] = json!(config.early_stopping_rounds);

            Booster::train_with_valid(train_dataset, Some(val_dataset), &params_with_es)
                .map_err(|e| format!("LightGBM Reg train error: {}", e))?
        } else {
            Booster::train(train_dataset, &params)
                .map_err(|e| format!("LightGBM Reg train error: {}", e))?
        };

        Ok(LightGBMRegressor { booster, n_features })
    }

    /// Predict return for a single feature vector (raw, no sigmoid)
    pub fn predict_return(&self, features: &[f64]) -> f64 {
        match self.booster.predict_with_params(
            features,
            self.n_features as i32,
            true,
            "num_threads=1",
        ) {
            Ok(results) => {
                if !results.is_empty() { results[0] } else { 0.0 }
            }
            Err(_) => 0.0,
        }
    }

    /// Batch predict returns
    pub fn predict_batch_returns(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let flat: Vec<f64> = features.iter().flat_map(|row| row.iter().copied()).collect();
        match self.booster.predict_with_params(
            &flat,
            self.n_features as i32,
            true,
            "num_threads=4",
        ) {
            Ok(results) => results,
            Err(_) => vec![0.0; features.len()],
        }
    }

    /// Feature importance — gain-based, normalised
    pub fn feature_importance(&self, feature_names: &[&str]) -> Vec<(String, f64)> {
        let n = self.n_features;
        let importances = match self.booster.feature_importance(ImportanceType::Gain) {
            Ok(imp) => imp,
            Err(_) => vec![0.0; n],
        };

        let total: f64 = importances.iter().sum();
        importances.iter().enumerate()
            .map(|(i, &imp)| {
                let name = if i < feature_names.len() {
                    feature_names[i].to_string()
                } else {
                    format!("Feature_{}", i)
                };
                let norm_imp = if total > 0.0 { imp / total } else { 0.0 };
                (name, norm_imp)
            })
            .collect()
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.booster.save_file(path)
            .map_err(|e| format!("LightGBM Reg save error: {}", e))
    }

    /// Load model from file
    pub fn load(path: &str, n_features: usize) -> Result<Self, String> {
        let booster = Booster::from_file(path)
            .map_err(|e| format!("LightGBM Reg load error: {}", e))?;
        Ok(LightGBMRegressor { booster, n_features })
    }

    /// Number of trees actually trained
    pub fn num_trees(&self) -> i32 {
        self.booster.num_iterations()
    }
}

/// Evaluate LightGBM regressor: returns (MAE, directional accuracy)
pub fn evaluate_lgbm_regressor(
    model: &LightGBMRegressor,
    test_features: &[Vec<f64>],
    test_labels: &[f64],
) -> (f64, f64) {
    if test_features.is_empty() {
        return (f64::NAN, 0.0);
    }

    let mut total_ae = 0.0;
    let mut dir_correct = 0usize;

    for (features, &actual) in test_features.iter().zip(test_labels.iter()) {
        let pred = model.predict_return(features);
        total_ae += (pred - actual).abs();
        if (pred > 0.0) == (actual > 0.0) {
            dir_correct += 1;
        }
    }

    let mae = total_ae / test_features.len() as f64;
    let dir_acc = dir_correct as f64 / test_features.len() as f64;
    (mae, dir_acc)
}

/// Evaluate LightGBM on test samples
pub fn evaluate_lgbm(
    model: &LightGBMClassifier,
    test_features: &[Vec<f64>],
    test_labels: &[f64],
) -> (usize, f64) {
    let mut correct = 0_usize;
    let mut total_log_loss = 0.0_f64;

    for (features, &label) in test_features.iter().zip(test_labels.iter()) {
        let prob = model.predict_proba(features);
        let actual_up = label > 0.5;
        let predicted_up = prob > 0.5;
        if predicted_up == actual_up {
            correct += 1;
        }
        // Binary cross-entropy
        let p = prob.clamp(1e-7, 1.0 - 1e-7);
        let y = if actual_up { 1.0 } else { 0.0 };
        total_log_loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }

    let avg_log_loss = if test_labels.is_empty() { 0.693 } else { total_log_loss / test_labels.len() as f64 };
    (correct, avg_log_loss)
}
