/// Agent Module — Shared types and logic for the autonomous agent
/// ==============================================================
/// Used by both src/bin/agent.rs and src/bin/serve.rs (API endpoints).

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

// ════════════════════════════════════════
// Configuration defaults
// ════════════════════════════════════════

pub const DEFAULT_ACCURACY_CRISIS_THRESHOLD: f64 = 35.0;
pub const DEFAULT_ACCURACY_DEGRADATION_THRESHOLD: f64 = 45.0;
pub const DEFAULT_DEGRADATION_DROP_PP: f64 = 10.0;
pub const DEFAULT_STALENESS_DAYS: i64 = 14;
pub const DEFAULT_RETRAIN_COOLDOWN_HOURS: i32 = 24;
pub const DEFAULT_MAX_CONCURRENT_RETRAINS: usize = 3;
pub const DEFAULT_MAX_DAILY_RETRAINS: usize = 20;
pub const DEFAULT_ROLLBACK_THRESHOLD_PP: f64 = 3.0;
pub const DEFAULT_EVAL_WAIT_HOURS: i32 = 24;
pub const DEFAULT_MIN_SIGNALS_FOR_ACTION: usize = 10;
pub const DEFAULT_BUY_THRESHOLD_TIGHTEN: f64 = 0.60;
pub const DEFAULT_ASYMMETRIC_BUY_THRESHOLD: f64 = 40.0;
pub const DEFAULT_ASYMMETRIC_SELL_THRESHOLD: f64 = 55.0;
pub const DEFAULT_MODEL_DOMINANCE_PP: f64 = 8.0;

// US equity market hours: 09:30 ET = 13:30 UTC (open) to 16:00 ET = 21:00 UTC (close)
// Block retrains during full trading day to avoid interfering with live price discovery.
pub const MARKET_BLOCK_START_HOUR: u32 = 13;
pub const MARKET_BLOCK_START_MIN: u32 = 30;
pub const MARKET_BLOCK_END_HOUR: u32 = 21;

// ════════════════════════════════════════
// Agent State
// ════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub enabled: bool,
    pub approval_required: bool,
    pub last_run: Option<String>,
    pub last_run_duration_ms: Option<u64>,
    pub total_runs: u64,
    pub actions_proposed: u64,
    pub actions_executed: u64,
    pub actions_rolled_back: u64,
    pub phase: String,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            enabled: true,
            approval_required: false,
            last_run: None,
            last_run_duration_ms: None,
            total_runs: 0,
            actions_proposed: 0,
            actions_executed: 0,
            actions_rolled_back: 0,
            phase: "startup".to_string(),
        }
    }
}

// ════════════════════════════════════════
// Agent Config (loaded from DB)
// ════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub enabled: bool,
    pub approval_required: bool,
    pub accuracy_crisis_threshold: f64,
    pub accuracy_degradation_threshold: f64,
    pub degradation_drop_pp: f64,
    pub staleness_days: i64,
    pub retrain_cooldown_hours: i32,
    pub max_concurrent_retrains: usize,
    pub max_daily_retrains: usize,
    pub rollback_threshold_pp: f64,
    pub eval_wait_hours: i32,
    pub min_signals_for_action: usize,
    pub asymmetric_buy_threshold: f64,
    pub asymmetric_sell_threshold: f64,
    pub model_dominance_pp: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            approval_required: false,
            accuracy_crisis_threshold: DEFAULT_ACCURACY_CRISIS_THRESHOLD,
            accuracy_degradation_threshold: DEFAULT_ACCURACY_DEGRADATION_THRESHOLD,
            degradation_drop_pp: DEFAULT_DEGRADATION_DROP_PP,
            staleness_days: DEFAULT_STALENESS_DAYS,
            retrain_cooldown_hours: DEFAULT_RETRAIN_COOLDOWN_HOURS,
            max_concurrent_retrains: DEFAULT_MAX_CONCURRENT_RETRAINS,
            max_daily_retrains: DEFAULT_MAX_DAILY_RETRAINS,
            rollback_threshold_pp: DEFAULT_ROLLBACK_THRESHOLD_PP,
            eval_wait_hours: DEFAULT_EVAL_WAIT_HOURS,
            min_signals_for_action: DEFAULT_MIN_SIGNALS_FOR_ACTION,
            asymmetric_buy_threshold: DEFAULT_ASYMMETRIC_BUY_THRESHOLD,
            asymmetric_sell_threshold: DEFAULT_ASYMMETRIC_SELL_THRESHOLD,
            model_dominance_pp: DEFAULT_MODEL_DOMINANCE_PP,
        }
    }
}

impl AgentConfig {
    /// Load config from agent_config table, falling back to defaults
    pub fn load_from_db(db: &crate::db::Database) -> Self {
        let mut config = Self::default();
        if let Ok(pairs) = db.get_all_agent_config() {
            for (key, value) in pairs {
                match key.as_str() {
                    "enabled" => config.enabled = value == "true",
                    "approval_required" => config.approval_required = value == "true",
                    "accuracy_crisis_threshold" => if let Ok(v) = value.parse() { config.accuracy_crisis_threshold = v; },
                    "accuracy_degradation_threshold" => if let Ok(v) = value.parse() { config.accuracy_degradation_threshold = v; },
                    "degradation_drop_pp" => if let Ok(v) = value.parse() { config.degradation_drop_pp = v; },
                    "staleness_days" => if let Ok(v) = value.parse() { config.staleness_days = v; },
                    "retrain_cooldown_hours" => if let Ok(v) = value.parse() { config.retrain_cooldown_hours = v; },
                    "max_concurrent_retrains" => if let Ok(v) = value.parse() { config.max_concurrent_retrains = v; },
                    "max_daily_retrains" => if let Ok(v) = value.parse() { config.max_daily_retrains = v; },
                    "rollback_threshold_pp" => if let Ok(v) = value.parse() { config.rollback_threshold_pp = v; },
                    "eval_wait_hours" => if let Ok(v) = value.parse() { config.eval_wait_hours = v; },
                    "min_signals_for_action" => if let Ok(v) = value.parse() { config.min_signals_for_action = v; },
                    _ => {}
                }
            }
        }
        config
    }

    /// Save current config to agent_config table
    pub fn save_to_db(&self, db: &crate::db::Database) {
        let _ = db.set_agent_config("enabled", &self.enabled.to_string());
        let _ = db.set_agent_config("approval_required", &self.approval_required.to_string());
        let _ = db.set_agent_config("accuracy_crisis_threshold", &self.accuracy_crisis_threshold.to_string());
        let _ = db.set_agent_config("accuracy_degradation_threshold", &self.accuracy_degradation_threshold.to_string());
        let _ = db.set_agent_config("degradation_drop_pp", &self.degradation_drop_pp.to_string());
        let _ = db.set_agent_config("staleness_days", &self.staleness_days.to_string());
        let _ = db.set_agent_config("retrain_cooldown_hours", &self.retrain_cooldown_hours.to_string());
        let _ = db.set_agent_config("max_concurrent_retrains", &self.max_concurrent_retrains.to_string());
        let _ = db.set_agent_config("max_daily_retrains", &self.max_daily_retrains.to_string());
        let _ = db.set_agent_config("rollback_threshold_pp", &self.rollback_threshold_pp.to_string());
        let _ = db.set_agent_config("eval_wait_hours", &self.eval_wait_hours.to_string());
        let _ = db.set_agent_config("min_signals_for_action", &self.min_signals_for_action.to_string());
    }

    /// Load config from PostgreSQL agent_config table, falling back to defaults
    pub async fn load_from_pg(pool: &crate::pg::PgPool) -> Self {
        let mut config = Self::default();
        if let Ok(pairs) = crate::pg::get_all_agent_config(pool).await {
            for (key, value) in pairs {
                match key.as_str() {
                    "enabled" => config.enabled = value == "true",
                    "approval_required" => config.approval_required = value == "true",
                    "accuracy_crisis_threshold" => if let Ok(v) = value.parse() { config.accuracy_crisis_threshold = v; },
                    "accuracy_degradation_threshold" => if let Ok(v) = value.parse() { config.accuracy_degradation_threshold = v; },
                    "degradation_drop_pp" => if let Ok(v) = value.parse() { config.degradation_drop_pp = v; },
                    "staleness_days" => if let Ok(v) = value.parse() { config.staleness_days = v; },
                    "retrain_cooldown_hours" => if let Ok(v) = value.parse() { config.retrain_cooldown_hours = v; },
                    "max_concurrent_retrains" => if let Ok(v) = value.parse() { config.max_concurrent_retrains = v; },
                    "max_daily_retrains" => if let Ok(v) = value.parse() { config.max_daily_retrains = v; },
                    "rollback_threshold_pp" => if let Ok(v) = value.parse() { config.rollback_threshold_pp = v; },
                    "eval_wait_hours" => if let Ok(v) = value.parse() { config.eval_wait_hours = v; },
                    "min_signals_for_action" => if let Ok(v) = value.parse() { config.min_signals_for_action = v; },
                    _ => {}
                }
            }
        }
        config
    }

    /// Save current config to PostgreSQL agent_config table
    pub async fn save_to_pg(&self, pool: &crate::pg::PgPool) {
        let _ = crate::pg::set_agent_config(pool, "enabled", &self.enabled.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "approval_required", &self.approval_required.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "accuracy_crisis_threshold", &self.accuracy_crisis_threshold.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "accuracy_degradation_threshold", &self.accuracy_degradation_threshold.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "degradation_drop_pp", &self.degradation_drop_pp.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "staleness_days", &self.staleness_days.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "retrain_cooldown_hours", &self.retrain_cooldown_hours.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "max_concurrent_retrains", &self.max_concurrent_retrains.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "max_daily_retrains", &self.max_daily_retrains.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "rollback_threshold_pp", &self.rollback_threshold_pp.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "eval_wait_hours", &self.eval_wait_hours.to_string()).await;
        let _ = crate::pg::set_agent_config(pool, "min_signals_for_action", &self.min_signals_for_action.to_string()).await;
    }
}

// ════════════════════════════════════════
// Proposed Actions
// ════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    pub action_type: String,
    pub asset: Option<String>,
    pub trigger_reason: String,
    pub accuracy_before: Option<f64>,
    pub details: HashMap<String, serde_json::Value>,
}

// ════════════════════════════════════════
// Threshold Overrides
// ════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThresholdOverrides {
    pub overrides: HashMap<String, AssetThresholdOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetThresholdOverride {
    pub buy_threshold: Option<f64>,
    pub sell_threshold: Option<f64>,
    pub reason: String,
    pub set_by: String,
    pub set_at: String,
    #[serde(default)]
    pub expires_at: Option<String>,
}

impl ThresholdOverrides {
    pub fn load() -> Self {
        let path = "config/threshold_overrides.json";
        match std::fs::read_to_string(path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let _ = std::fs::create_dir_all("config");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("JSON error: {}", e))?;
        std::fs::write("config/threshold_overrides.json", json)
            .map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }
}

// ════════════════════════════════════════
// Utility: market hours check
// ════════════════════════════════════════

/// Check if current time is during US market hours (14:30-21:00 UTC)
pub fn is_us_market_hours() -> bool {
    let now = chrono::Utc::now();
    let hour = now.hour();
    let min = now.minute();
    let time_mins = hour * 60 + min;
    let block_start = MARKET_BLOCK_START_HOUR * 60 + MARKET_BLOCK_START_MIN;
    let block_end = MARKET_BLOCK_END_HOUR * 60;
    time_mins >= block_start && time_mins < block_end
}

use chrono::Timelike;
