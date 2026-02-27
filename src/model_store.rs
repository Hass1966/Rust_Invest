/// Model Serialisation — Save / Load Trained Models
/// ==================================================
/// Avoids retraining every run by persisting model weights to disk.
///
/// Saves to: models/<symbol>_<model_type>.json
///   - LinReg: weights + bias
///   - LogReg: weights + bias
///   - GBT: tree structure (splits, leaves, thresholds)
///   - LSTM: candle VarMap (separate .safetensors file)
///
/// Metadata tracks training timestamp + feature count so stale
/// models are automatically invalidated when features change.

use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

const MODEL_DIR: &str = "models";
const MODEL_VERSION: u32 = 2; // Bump when feature set changes

/// Metadata for a saved model
#[derive(Serialize, Deserialize, Debug)]
pub struct ModelMeta {
    pub version: u32,
    pub symbol: String,
    pub model_type: String,
    pub n_features: usize,
    pub trained_at: String,
    pub train_samples: usize,
    pub walk_forward_accuracy: f64,
}

/// Saved linear/logistic regression weights
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedWeights {
    pub meta: ModelMeta,
    pub weights: Vec<f64>,
    pub bias: f64,
}

/// Saved GBT model
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedGBT {
    pub meta: ModelMeta,
    pub trees_json: String,  // Serialised tree structure
    pub base_prediction: f64,
    pub learning_rate: f64,
}

/// Ensure model directory exists
pub fn ensure_model_dir() {
    let _ = fs::create_dir_all(MODEL_DIR);
}

/// Get path for a model file
pub fn model_path(symbol: &str, model_type: &str) -> String {
    format!("{}/{}_{}.json", MODEL_DIR, symbol.to_lowercase(), model_type)
}

/// Get path for LSTM safetensors file
pub fn lstm_path(symbol: &str) -> String {
    format!("{}/{}_lstm.safetensors", MODEL_DIR, symbol.to_lowercase())
}

/// Check if a saved model is still valid (correct version + feature count)
pub fn is_model_valid(path: &str, expected_features: usize) -> bool {
    if !Path::new(path).exists() {
        return false;
    }

    // Quick check: read just the meta
    if let Ok(contents) = fs::read_to_string(path) {
        if let Ok(saved) = serde_json::from_str::<SavedWeights>(&contents) {
            return saved.meta.version == MODEL_VERSION
                && saved.meta.n_features == expected_features;
        }
        if let Ok(saved) = serde_json::from_str::<SavedGBT>(&contents) {
            return saved.meta.version == MODEL_VERSION
                && saved.meta.n_features == expected_features;
        }
    }

    false
}

/// Save linear/logistic regression weights
pub fn save_weights(
    symbol: &str,
    model_type: &str,
    weights: &[f64],
    bias: f64,
    n_features: usize,
    train_samples: usize,
    accuracy: f64,
) -> Result<(), String> {
    ensure_model_dir();

    let meta = ModelMeta {
        version: MODEL_VERSION,
        symbol: symbol.to_string(),
        model_type: model_type.to_string(),
        n_features,
        trained_at: chrono::Utc::now().to_rfc3339(),
        train_samples,
        walk_forward_accuracy: accuracy,
    };

    let saved = SavedWeights { meta, weights: weights.to_vec(), bias };
    let json = serde_json::to_string_pretty(&saved)
        .map_err(|e| format!("JSON serialisation error: {}", e))?;

    let path = model_path(symbol, model_type);
    fs::write(&path, json).map_err(|e| format!("Write error: {}", e))?;

    Ok(())
}

/// Load linear/logistic regression weights
pub fn load_weights(symbol: &str, model_type: &str) -> Result<SavedWeights, String> {
    let path = model_path(symbol, model_type);
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Read error: {}", e))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("Deserialise error: {}", e))
}

/// Summary of all cached models
pub fn list_cached_models() -> Vec<String> {
    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir(MODEL_DIR) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") || name.ends_with(".safetensors") {
                    models.push(name.to_string());
                }
            }
        }
    }
    models.sort();
    models
}

/// Clear all cached models (force retrain)
pub fn clear_cache() -> usize {
    let mut count = 0;
    if let Ok(entries) = fs::read_dir(MODEL_DIR) {
        for entry in entries.flatten() {
            let _ = fs::remove_file(entry.path());
            count += 1;
        }
    }
    count
}
