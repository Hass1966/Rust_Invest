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
///
/// Retrain policy:
///   - Models older than 7 days → retrain
///   - Feature count mismatch → retrain
///   - Version bump → retrain
///   - Explicit `--retrain` flag → retrain all

use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;
use crate::gbt;

const MODEL_DIR: &str = "models";
pub const MODEL_VERSION: u32 = 6; // v6: regression ensemble (Ridge + LightGBM + GRU), percentage return labels
const RETRAIN_DAYS: i64 = 7;  // Retrain after 7 days

/// Metadata for a saved model
#[derive(Serialize, Deserialize, Debug, Clone)]
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
    /// Normalisation parameters (so we can apply the same transform at prediction time)
    pub norm_means: Vec<f64>,
    pub norm_stds: Vec<f64>,
}

// ════════════════════════════════════════
// GBT Serialisation — serde-friendly tree structure
// ════════════════════════════════════════

/// Serialisable representation of a tree node
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SerializableNode {
    Leaf {
        value: f64,
        n_samples: usize,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        gain: f64,
        left: Box<SerializableNode>,
        right: Box<SerializableNode>,
    },
}

impl SerializableNode {
    /// Convert from gbt::Node to serializable form
    pub fn from_node(node: &gbt::Node) -> Self {
        match node {
            gbt::Node::Leaf { value, n_samples } => {
                SerializableNode::Leaf { value: *value, n_samples: *n_samples }
            }
            gbt::Node::Split { feature_idx, threshold, gain, left, right } => {
                SerializableNode::Split {
                    feature_idx: *feature_idx,
                    threshold: *threshold,
                    gain: *gain,
                    left: Box::new(SerializableNode::from_node(left)),
                    right: Box::new(SerializableNode::from_node(right)),
                }
            }
        }
    }

    /// Convert back to gbt::Node
    pub fn to_node(&self) -> gbt::Node {
        match self {
            SerializableNode::Leaf { value, n_samples } => {
                gbt::Node::Leaf { value: *value, n_samples: *n_samples }
            }
            SerializableNode::Split { feature_idx, threshold, gain, left, right } => {
                gbt::Node::Split {
                    feature_idx: *feature_idx,
                    threshold: *threshold,
                    gain: *gain,
                    left: Box::new(left.to_node()),
                    right: Box::new(right.to_node()),
                }
            }
        }
    }
}

/// Saved GBT model
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedGBT {
    pub meta: ModelMeta,
    pub trees: Vec<SerializableNode>,
    pub initial_prediction: f64,
    pub learning_rate: f64,
    pub n_features: usize,
    /// Normalisation parameters
    pub norm_means: Vec<f64>,
    pub norm_stds: Vec<f64>,
}

/// Saved LSTM metadata (weights in .safetensors, this tracks meta + norm params)
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedLSTMMeta {
    pub meta: ModelMeta,
    pub hidden_size: usize,
    pub seq_length: usize,
    pub norm_means: Vec<f64>,
    pub norm_stds: Vec<f64>,
}

/// Saved GRU metadata (weights in .safetensors, this tracks meta + norm params)
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedGRUMeta {
    pub meta: ModelMeta,
    pub hidden_size: usize,
    pub seq_length: usize,
    pub norm_means: Vec<f64>,
    pub norm_stds: Vec<f64>,
    /// Feature indices used for GRU input (top-N from importance ranking).
    /// If None, all features are used.
    #[serde(default)]
    pub feature_indices: Option<Vec<usize>>,
}

// ════════════════════════════════════════
// File Paths
// ════════════════════════════════════════

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

/// Get path for LSTM metadata file
pub fn lstm_meta_path(symbol: &str) -> String {
    format!("{}/{}_lstm_meta.json", MODEL_DIR, symbol.to_lowercase())
}

/// Get path for GRU safetensors file
pub fn gru_path(symbol: &str) -> String {
    format!("{}/{}_gru.safetensors", MODEL_DIR, symbol.to_lowercase())
}

/// Get path for GRU metadata file
pub fn gru_meta_path(symbol: &str) -> String {
    format!("{}/{}_gru_meta.json", MODEL_DIR, symbol.to_lowercase())
}

/// Get path for Ridge regression model (uses SavedWeights format)
pub fn ridge_path(symbol: &str) -> String {
    model_path(symbol, "ridge")
}

/// Get path for LightGBM regressor model file
pub fn lgbm_regressor_path(symbol: &str) -> String {
    format!("{}/{}_lgbm_reg.txt", MODEL_DIR, symbol.to_lowercase())
}

/// Get path for LightGBM MAE metadata file
pub fn lgbm_mae_path(symbol: &str) -> String {
    format!("{}/{}_lgbm_mae.json", MODEL_DIR, symbol.to_lowercase())
}

/// Get path for 5-day Ridge model
pub fn ridge_path_5d(symbol: &str) -> String {
    format!("{}/5d_{}_ridge.json", MODEL_DIR, symbol.to_lowercase())
}

/// Get path for 5-day LightGBM regressor model
pub fn lgbm_regressor_path_5d(symbol: &str) -> String {
    format!("{}/5d_{}_lgbm_reg.txt", MODEL_DIR, symbol.to_lowercase())
}

/// Save LightGBM MAE metadata alongside the model
pub fn save_lgbm_mae(symbol: &str, mae: f64, dir_acc: f64) -> Result<(), String> {
    let path = lgbm_mae_path(symbol);
    let data = serde_json::json!({ "mae": mae, "dir_acc": dir_acc });
    fs::write(&path, serde_json::to_string(&data).unwrap())
        .map_err(|e| format!("Failed to write LGBM MAE: {}", e))
}

/// Load LightGBM MAE from saved metadata
pub fn load_lgbm_mae(symbol: &str) -> Option<f64> {
    let path = lgbm_mae_path(symbol);
    let contents = fs::read_to_string(&path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&contents).ok()?;
    val.get("mae").and_then(|v| v.as_f64())
}

// ════════════════════════════════════════
// Staleness & Validity Checks
// ════════════════════════════════════════

/// Check if a saved model is still valid (correct version, feature count, not stale)
pub fn is_model_valid(path: &str, expected_features: usize) -> bool {
    if !Path::new(path).exists() {
        return false;
    }

    if let Ok(contents) = fs::read_to_string(path) {
        // Try SavedWeights first
        if let Ok(saved) = serde_json::from_str::<SavedWeights>(&contents) {
            return check_meta(&saved.meta, expected_features);
        }
        // Try SavedGBT
        if let Ok(saved) = serde_json::from_str::<SavedGBT>(&contents) {
            return check_meta(&saved.meta, expected_features);
        }
        // Try SavedLSTMMeta
        if let Ok(saved) = serde_json::from_str::<SavedLSTMMeta>(&contents) {
            return check_meta(&saved.meta, expected_features);
        }
    }

    false
}

/// Check metadata validity: version, feature count, staleness
fn check_meta(meta: &ModelMeta, expected_features: usize) -> bool {
    // Version check
    if meta.version != MODEL_VERSION {
        return false;
    }

    // Feature count check
    if meta.n_features != expected_features {
        return false;
    }

    // Staleness check
    if is_stale(&meta.trained_at) {
        return false;
    }

    true
}

/// Check if a model trained at `trained_at` is older than RETRAIN_DAYS
fn is_stale(trained_at: &str) -> bool {
    if let Ok(trained) = chrono::DateTime::parse_from_rfc3339(trained_at) {
        let age = chrono::Utc::now().signed_duration_since(trained.with_timezone(&chrono::Utc));
        return age.num_days() >= RETRAIN_DAYS;
    }
    true // If we can't parse the date, assume stale
}

/// Check if all core models exist and are valid for a symbol (regression ensemble)
pub fn has_valid_models(symbol: &str, expected_features: usize) -> bool {
    // Ridge is the minimum required model; LightGBM and GRU are optional
    is_model_valid(&ridge_path(symbol), expected_features)
}

/// Get staleness info for a model
pub fn model_age_days(path: &str) -> Option<i64> {
    if !Path::new(path).exists() { return None; }

    if let Ok(contents) = fs::read_to_string(path) {
        if let Ok(saved) = serde_json::from_str::<SavedWeights>(&contents) {
            if let Ok(trained) = chrono::DateTime::parse_from_rfc3339(&saved.meta.trained_at) {
                let age = chrono::Utc::now().signed_duration_since(trained.with_timezone(&chrono::Utc));
                return Some(age.num_days());
            }
        }
        if let Ok(saved) = serde_json::from_str::<SavedGBT>(&contents) {
            if let Ok(trained) = chrono::DateTime::parse_from_rfc3339(&saved.meta.trained_at) {
                let age = chrono::Utc::now().signed_duration_since(trained.with_timezone(&chrono::Utc));
                return Some(age.num_days());
            }
        }
    }

    None
}

// ════════════════════════════════════════
// Save Operations
// ════════════════════════════════════════

fn make_meta(symbol: &str, model_type: &str, n_features: usize,
             train_samples: usize, accuracy: f64) -> ModelMeta {
    ModelMeta {
        version: MODEL_VERSION,
        symbol: symbol.to_string(),
        model_type: model_type.to_string(),
        n_features,
        trained_at: chrono::Utc::now().to_rfc3339(),
        train_samples,
        walk_forward_accuracy: accuracy,
    }
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
    norm_means: &[f64],
    norm_stds: &[f64],
) -> Result<(), String> {
    ensure_model_dir();

    let saved = SavedWeights {
        meta: make_meta(symbol, model_type, n_features, train_samples, accuracy),
        weights: weights.to_vec(),
        bias,
        norm_means: norm_means.to_vec(),
        norm_stds: norm_stds.to_vec(),
    };

    let json = serde_json::to_string_pretty(&saved)
        .map_err(|e| format!("JSON serialisation error: {}", e))?;

    let path = model_path(symbol, model_type);
    fs::write(&path, json).map_err(|e| format!("Write error: {}", e))?;
    println!("    [Store] Saved {} {} → {}", symbol, model_type, path);

    Ok(())
}

/// Load linear/logistic regression weights
pub fn load_weights(symbol: &str, model_type: &str) -> Result<SavedWeights, String> {
    let path = model_path(symbol, model_type);
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Read error: {}", e))?;
    let saved: SavedWeights = serde_json::from_str(&contents)
        .map_err(|e| format!("Deserialise error: {}", e))?;
    println!("    [Store] Loaded {} {} (trained: {}, acc: {:.1}%)",
        symbol, model_type, &saved.meta.trained_at[..10], saved.meta.walk_forward_accuracy);
    Ok(saved)
}

/// Save GBT model (trees + config)
pub fn save_gbt(
    symbol: &str,
    classifier: &gbt::GradientBoostedClassifier,
    train_samples: usize,
    accuracy: f64,
    norm_means: &[f64],
    norm_stds: &[f64],
) -> Result<(), String> {
    save_gbt_as(symbol, "gbt", classifier, train_samples, accuracy, norm_means, norm_stds)
}

/// Save GBT model with a custom model type name (e.g. "5d_gbt")
pub fn save_gbt_as(
    symbol: &str,
    model_type: &str,
    classifier: &gbt::GradientBoostedClassifier,
    train_samples: usize,
    accuracy: f64,
    norm_means: &[f64],
    norm_stds: &[f64],
) -> Result<(), String> {
    ensure_model_dir();

    let trees: Vec<SerializableNode> = classifier.trees.iter()
        .map(|t| SerializableNode::from_node(t))
        .collect();

    let saved = SavedGBT {
        meta: make_meta(symbol, model_type, classifier.n_features, train_samples, accuracy),
        trees,
        initial_prediction: classifier.initial_prediction,
        learning_rate: classifier.config.learning_rate,
        n_features: classifier.n_features,
        norm_means: norm_means.to_vec(),
        norm_stds: norm_stds.to_vec(),
    };

    let json = serde_json::to_string(&saved)
        .map_err(|e| format!("JSON serialisation error: {}", e))?;

    let path = model_path(symbol, model_type);
    fs::write(&path, json).map_err(|e| format!("Write error: {}", e))?;
    println!("    [Store] Saved {} {} ({} trees) → {}", symbol, model_type, classifier.trees.len(), path);

    Ok(())
}

/// Load GBT model and reconstruct classifier
pub fn load_gbt(symbol: &str) -> Result<(SavedGBT, gbt::GradientBoostedClassifier), String> {
    load_gbt_as(symbol, "gbt")
}

/// Load a GBT model with custom type name (e.g., "5d_gbt" for 5-day horizon)
pub fn load_gbt_as(symbol: &str, model_type: &str) -> Result<(SavedGBT, gbt::GradientBoostedClassifier), String> {
    let path = model_path(symbol, model_type);
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Read error: {}", e))?;
    let saved: SavedGBT = serde_json::from_str(&contents)
        .map_err(|e| format!("Deserialise error: {}", e))?;

    let trees: Vec<gbt::Node> = saved.trees.iter()
        .map(|t| t.to_node())
        .collect();

    let classifier = gbt::GradientBoostedClassifier {
        trees,
        config: gbt::GBTConfig {
            n_trees: saved.trees.len(),
            learning_rate: saved.learning_rate,
            ..Default::default()
        },
        initial_prediction: saved.initial_prediction,
        train_losses: Vec::new(),
        val_losses: Vec::new(),
        n_features: saved.n_features,
    };

    println!("    [Store] Loaded {} GBT ({} trees, trained: {}, acc: {:.1}%)",
        symbol, saved.trees.len(), &saved.meta.trained_at[..10], saved.meta.walk_forward_accuracy);

    Ok((saved, classifier))
}

/// Save LSTM metadata (weights saved separately via candle VarMap)
pub fn save_lstm_meta(
    symbol: &str,
    n_features: usize,
    hidden_size: usize,
    seq_length: usize,
    train_samples: usize,
    accuracy: f64,
    norm_means: &[f64],
    norm_stds: &[f64],
) -> Result<(), String> {
    ensure_model_dir();

    let saved = SavedLSTMMeta {
        meta: make_meta(symbol, "lstm", n_features, train_samples, accuracy),
        hidden_size,
        seq_length,
        norm_means: norm_means.to_vec(),
        norm_stds: norm_stds.to_vec(),
    };

    let json = serde_json::to_string_pretty(&saved)
        .map_err(|e| format!("JSON serialisation error: {}", e))?;

    let path = lstm_meta_path(symbol);
    fs::write(&path, json).map_err(|e| format!("Write error: {}", e))?;
    println!("    [Store] Saved {} LSTM meta → {}", symbol, path);

    Ok(())
}

/// Load LSTM metadata
pub fn load_lstm_meta(symbol: &str) -> Result<SavedLSTMMeta, String> {
    let path = lstm_meta_path(symbol);
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Read error: {}", e))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("Deserialise error: {}", e))
}

/// Save GRU metadata (weights saved separately via candle VarMap)
pub fn save_gru_meta(
    symbol: &str,
    n_features: usize,
    hidden_size: usize,
    seq_length: usize,
    train_samples: usize,
    mae: f64,
    norm_means: &[f64],
    norm_stds: &[f64],
    feature_indices: Option<&[usize]>,
) -> Result<(), String> {
    ensure_model_dir();

    let saved = SavedGRUMeta {
        meta: make_meta(symbol, "gru", n_features, train_samples, mae),
        hidden_size,
        seq_length,
        norm_means: norm_means.to_vec(),
        norm_stds: norm_stds.to_vec(),
        feature_indices: feature_indices.map(|fi| fi.to_vec()),
    };

    let json = serde_json::to_string_pretty(&saved)
        .map_err(|e| format!("JSON serialisation error: {}", e))?;

    let path = gru_meta_path(symbol);
    fs::write(&path, json).map_err(|e| format!("Write error: {}", e))?;
    println!("    [Store] Saved {} GRU meta → {}", symbol, path);

    Ok(())
}

/// Load GRU metadata
pub fn load_gru_meta(symbol: &str) -> Result<SavedGRUMeta, String> {
    let path = gru_meta_path(symbol);
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Read error: {}", e))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("Deserialise error: {}", e))
}

// ════════════════════════════════════════
// Cache Management
// ════════════════════════════════════════

/// Summary of all cached models
/// Load pre-retrain accuracy for an asset (average of available model accuracies)
/// For regression models, returns average directional accuracy (from walk_forward_accuracy field).
pub fn load_model_accuracy(symbol: &str) -> f64 {
    let mut acc_sum = 0.0;
    let mut count = 0;

    // Check regression models first (ridge stores MAE in walk_forward_accuracy)
    if let Ok(saved) = load_weights(symbol, "ridge") {
        // For ridge, walk_forward_accuracy stores MAE — invert for a rough "accuracy" proxy
        acc_sum += saved.meta.walk_forward_accuracy;
        count += 1;
    }

    // Fallback: check legacy classification models
    if count == 0 {
        for model_type in &["linreg", "logreg"] {
            if let Ok(saved) = load_weights(symbol, model_type) {
                acc_sum += saved.meta.walk_forward_accuracy;
                count += 1;
            }
        }
        if let Ok((saved, _)) = load_gbt(symbol) {
            acc_sum += saved.meta.walk_forward_accuracy;
            count += 1;
        }
    }

    if count > 0 { acc_sum / count as f64 } else { 0.0 }
}

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

// ════════════════════════════════════════
// Model Manifest — top-level summary of all models
// ════════════════════════════════════════

const MANIFEST_PATH: &str = "models/manifest.json";

/// Summary entry for one asset in the manifest
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ManifestAssetEntry {
    #[serde(default)]
    pub linreg_accuracy: Option<f64>,
    #[serde(default)]
    pub logreg_accuracy: Option<f64>,
    #[serde(default)]
    pub gbt_accuracy: Option<f64>,
    #[serde(default)]
    pub lstm_accuracy: Option<f64>,
    #[serde(default)]
    pub regime_accuracy: Option<f64>,
    #[serde(default)]
    pub ensemble_accuracy: Option<f64>,
    /// Regression model MAE values (v6+)
    #[serde(default)]
    pub ridge_mae: Option<f64>,
    #[serde(default)]
    pub lgbm_mae: Option<f64>,
    #[serde(default)]
    pub gru_mae: Option<f64>,
    /// Regression model directional accuracies (v6+)
    #[serde(default)]
    pub ridge_dir_acc: Option<f64>,
    #[serde(default)]
    pub lgbm_dir_acc: Option<f64>,
    #[serde(default)]
    pub gru_dir_acc: Option<f64>,
    pub last_trained: Option<String>,
    pub weights_present: bool,
}

/// Top-level model manifest
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelManifest {
    pub version: u32,
    pub generated_at: String,
    pub assets: std::collections::HashMap<String, ManifestAssetEntry>,
}

/// Generate model manifest by scanning the models/ directory
pub fn generate_manifest(symbols: &[&str]) -> ModelManifest {
    ensure_model_dir();
    let mut assets = std::collections::HashMap::new();

    for symbol in symbols {
        // Check regression models first (v6+)
        let ridge_path_str = ridge_path(symbol);
        let lgbm_reg_path_str = lgbm_regressor_path(symbol);
        let gru_meta_path_str = gru_meta_path(symbol);

        let mut ridge_mae_val = None;
        let lgbm_mae_val: Option<f64> = None;
        let mut gru_mae_val: Option<f64> = None;
        let ridge_dir_val: Option<f64> = None;
        let lgbm_dir_val: Option<f64> = None;
        let gru_dir_val: Option<f64> = None;
        let mut last_trained = None;
        let mut weights_present = false;

        // Ridge (SavedWeights format, walk_forward_accuracy stores MAE for regression)
        if let Ok(contents) = fs::read_to_string(&ridge_path_str) {
            if let Ok(saved) = serde_json::from_str::<SavedWeights>(&contents) {
                ridge_mae_val = Some(saved.meta.walk_forward_accuracy);
                last_trained = Some(saved.meta.trained_at.clone());
                weights_present = true;
            }
        }

        // LightGBM regressor (check file existence)
        if std::path::Path::new(&lgbm_reg_path_str).exists() {
            weights_present = true;
        }

        // GRU regression (meta JSON)
        if let Ok(contents) = fs::read_to_string(&gru_meta_path_str) {
            if let Ok(saved) = serde_json::from_str::<SavedGRUMeta>(&contents) {
                gru_mae_val = Some(saved.meta.walk_forward_accuracy);
                if last_trained.is_none() || saved.meta.trained_at > *last_trained.as_ref().unwrap_or(&String::new()) {
                    last_trained = Some(saved.meta.trained_at.clone());
                }
                weights_present = true;
            }
        }

        // Also check legacy classification models for backward compat
        let linreg_path = model_path(symbol, "linreg");
        let logreg_path = model_path(symbol, "logreg");
        let gbt_path_str = model_path(symbol, "gbt");

        let mut linreg_acc = None;
        let mut logreg_acc = None;
        let mut gbt_acc = None;

        if let Ok(contents) = fs::read_to_string(&linreg_path) {
            if let Ok(saved) = serde_json::from_str::<SavedWeights>(&contents) {
                linreg_acc = Some(saved.meta.walk_forward_accuracy);
                if last_trained.is_none() { last_trained = Some(saved.meta.trained_at.clone()); }
                weights_present = true;
            }
        }
        if let Ok(contents) = fs::read_to_string(&logreg_path) {
            if let Ok(saved) = serde_json::from_str::<SavedWeights>(&contents) {
                logreg_acc = Some(saved.meta.walk_forward_accuracy);
                weights_present = true;
            }
        }
        if let Ok(contents) = fs::read_to_string(&gbt_path_str) {
            if let Ok(saved) = serde_json::from_str::<SavedGBT>(&contents) {
                gbt_acc = Some(saved.meta.walk_forward_accuracy);
                weights_present = true;
            }
        }

        let ensemble_accuracy = match (linreg_acc, logreg_acc, gbt_acc) {
            (Some(a), Some(b), Some(c)) => Some((a + b + c) / 3.0),
            _ => None,
        };

        if weights_present {
            let (lstm_acc, regime_acc) = load_extended_accuracies(symbol);

            assets.insert(symbol.to_string(), ManifestAssetEntry {
                linreg_accuracy: linreg_acc,
                logreg_accuracy: logreg_acc,
                gbt_accuracy: gbt_acc,
                lstm_accuracy: lstm_acc,
                regime_accuracy: regime_acc,
                ensemble_accuracy,
                ridge_mae: ridge_mae_val,
                lgbm_mae: lgbm_mae_val,
                gru_mae: gru_mae_val,
                ridge_dir_acc: ridge_dir_val,
                lgbm_dir_acc: lgbm_dir_val,
                gru_dir_acc: gru_dir_val,
                last_trained,
                weights_present,
            });
        }
    }

    let manifest = ModelManifest {
        version: MODEL_VERSION,
        generated_at: chrono::Utc::now().to_rfc3339(),
        assets,
    };

    // Write to disk
    if let Ok(json) = serde_json::to_string_pretty(&manifest) {
        let _ = fs::write(MANIFEST_PATH, json);
        println!("  [Manifest] Generated {} with {} assets", MANIFEST_PATH, manifest.assets.len());
    }

    manifest
}

/// Load LSTM/Regime accuracies from reports/improved.json (generated by train binary)
fn load_extended_accuracies(symbol: &str) -> (Option<f64>, Option<f64>) {
    let path = "reports/improved.json";
    if let Ok(contents) = fs::read_to_string(path) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&contents) {
            if let Some(asset) = val.get("assets").and_then(|a| a.get(symbol)) {
                let lstm = asset.get("lstm").and_then(|x| x.as_f64());
                let regime = asset.get("regime").and_then(|x| x.as_f64());
                return (lstm, regime);
            }
        }
    }
    (None, None)
}

/// Load manifest from disk
pub fn load_manifest() -> Result<ModelManifest, String> {
    let contents = fs::read_to_string(MANIFEST_PATH)
        .map_err(|e| format!("Failed to read manifest: {}", e))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse manifest: {}", e))
}

// ════════════════════════════════════════
// Backup / Restore (for agent rollbacks)
// ════════════════════════════════════════

const BACKUP_DIR: &str = "models/backups";

/// Backup all model files for a symbol to models/backups/
pub fn backup_models(symbol: &str) -> Result<usize, String> {
    let _ = fs::create_dir_all(BACKUP_DIR);
    let sym_lower = symbol.to_lowercase();
    let mut count = 0;

    if let Ok(entries) = fs::read_dir(MODEL_DIR) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&sym_lower) && (name.ends_with(".json") || name.ends_with(".safetensors")) {
                    let dest = format!("{}/{}", BACKUP_DIR, name);
                    if fs::copy(entry.path(), &dest).is_ok() {
                        count += 1;
                    }
                }
            }
        }
    }

    if count > 0 {
        println!("    [Backup] Backed up {} model files for {}", count, symbol);
        Ok(count)
    } else {
        Err(format!("No model files found for {}", symbol))
    }
}

/// Restore model files for a symbol from models/backups/
pub fn restore_models(symbol: &str) -> Result<usize, String> {
    let sym_lower = symbol.to_lowercase();
    let mut count = 0;

    if !Path::new(BACKUP_DIR).exists() {
        return Err("No backup directory found".to_string());
    }

    if let Ok(entries) = fs::read_dir(BACKUP_DIR) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&sym_lower) && (name.ends_with(".json") || name.ends_with(".safetensors")) {
                    let dest = format!("{}/{}", MODEL_DIR, name);
                    if fs::copy(entry.path(), &dest).is_ok() {
                        count += 1;
                    }
                }
            }
        }
    }

    if count > 0 {
        println!("    [Restore] Restored {} model files for {}", count, symbol);
        Ok(count)
    } else {
        Err(format!("No backup files found for {}", symbol))
    }
}

/// Check if backups exist for a symbol
pub fn has_backup(symbol: &str) -> bool {
    let sym_lower = symbol.to_lowercase();
    if let Ok(entries) = fs::read_dir(BACKUP_DIR) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&sym_lower) {
                    return true;
                }
            }
        }
    }
    false
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
    println!("  [Store] Cleared {} cached model files", count);
    count
}

/// Print cache status summary
pub fn print_cache_status(symbols: &[&str], n_features: usize) {
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  MODEL CACHE STATUS (v6 Regression)                         │");
    println!("  ├──────────┬───────┬───────┬───────┬──────────────────────────┤");
    println!("  │ Symbol   │ Ridge │ LGBM  │  GRU  │ Status                   │");
    println!("  ├──────────┼───────┼───────┼───────┼──────────────────────────┤");

    for symbol in symbols {
        let ridge_ok = is_model_valid(&ridge_path(symbol), n_features);
        let lgbm_ok = Path::new(&lgbm_regressor_path(symbol)).exists();
        let gru_ok = Path::new(&gru_path(symbol)).exists()
            && Path::new(&gru_meta_path(symbol)).exists();

        let status = if ridge_ok && lgbm_ok && gru_ok {
            "✓ All cached"
        } else if ridge_ok {
            if lgbm_ok || gru_ok { "⚠ Partial (2/3)" } else { "⚠ Ridge only" }
        } else {
            "✗ Need train"
        };

        let check = |ok: bool| if ok { " ✓ " } else { " ✗ " };

        println!("  │ {:<8} │ {}   │ {}   │ {}   │ {:<24} │",
            symbol, check(ridge_ok), check(lgbm_ok), check(gru_ok), status);
    }

    println!("  └──────────┴───────┴───────┴───────┴──────────────────────────┘");
    println!("  Retrain policy: every {} days or on feature/version change", RETRAIN_DAYS);
}
