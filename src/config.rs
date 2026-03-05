/// Asset Configuration — Central JSON-driven asset registry
/// =========================================================
/// Loads config/assets.json at startup. The enabled flag controls
/// whether an asset is served in production (signal endpoints).
/// Training always processes ALL assets regardless of enabled flag.

use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

const CONFIG_PATH: &str = "config/assets.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetEntry {
    pub symbol: String,
    pub name: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetConfig {
    pub stocks: Vec<AssetEntry>,
    pub fx: Vec<AssetEntry>,
    pub crypto: Vec<AssetEntry>,
}

impl AssetConfig {
    /// Load asset config from config/assets.json
    pub fn load() -> Result<Self, String> {
        Self::load_from(CONFIG_PATH)
    }

    /// Load from a specific path
    pub fn load_from(path: &str) -> Result<Self, String> {
        if !Path::new(path).exists() {
            return Err(format!("Config file not found: {}", path));
        }
        let contents = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to parse {}: {}", path, e))
    }

    /// Get only enabled stocks
    pub fn enabled_stocks(&self) -> Vec<&AssetEntry> {
        self.stocks.iter().filter(|a| a.enabled).collect()
    }

    /// Get only enabled FX pairs
    pub fn enabled_fx(&self) -> Vec<&AssetEntry> {
        self.fx.iter().filter(|a| a.enabled).collect()
    }

    /// Get only enabled crypto assets
    pub fn enabled_crypto(&self) -> Vec<&AssetEntry> {
        self.crypto.iter().filter(|a| a.enabled).collect()
    }

    /// Get all enabled assets across all classes
    pub fn all_enabled(&self) -> Vec<(&AssetEntry, &str)> {
        let mut out = Vec::new();
        for a in self.enabled_stocks() { out.push((a, "stock")); }
        for a in self.enabled_fx() { out.push((a, "fx")); }
        for a in self.enabled_crypto() { out.push((a, "crypto")); }
        out
    }
}
