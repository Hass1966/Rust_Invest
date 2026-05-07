/// agents — Fleet of autonomous AI agents for Alpha Signal
/// ========================================================
/// 6 agents run on independent schedules, sharing a PgPool and SQLite handle.
/// All activity is logged to agent_actions / agent_metrics in PostgreSQL.

pub mod data_quality;
pub mod risk_guardian;
pub mod model_drift;
pub mod signal_arbitrator;
pub mod portfolio_rebalancer;
pub mod performance_analyst;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

use crate::pg;

// ════════════════════════════════════════
// Shared Fleet State
// ════════════════════════════════════════

pub struct FleetState {
    pub pg_pool: pg::PgPool,
    pub db_path: String,
    pub http_client: reqwest::Client,
    /// Per-asset signal overrides (e.g., risk veto forcing HOLD)
    pub signal_overrides: Arc<RwLock<HashMap<String, SignalOverride>>>,
    /// Global risk halt flag — when true, all BUYs are vetoed
    pub risk_halt: Arc<RwLock<bool>>,
    /// Per-agent enabled flags (live-toggleable via agent_config)
    pub agent_enabled: Arc<RwLock<HashMap<String, bool>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalOverride {
    pub asset: String,
    pub forced_signal: String,
    pub reason: String,
    pub source_agent: String,
    pub expires_at: DateTime<Utc>,
}

pub const AGENT_NAMES: [&str; 6] = [
    "data_quality_sentinel",
    "risk_guardian",
    "model_drift_detector",
    "signal_arbitrator",
    "portfolio_rebalancer",
    "performance_analyst",
];

pub const SERVE_URL: &str = "http://localhost:8081";

// ════════════════════════════════════════
// Heartbeat & Run Logging
// ════════════════════════════════════════

/// Write a heartbeat to agent_config so the dashboard can see agent status.
pub async fn log_agent_heartbeat(
    pool: &pg::PgPool,
    agent_name: &str,
    status: &str,
) {
    let key = format!("fleet_{}_heartbeat", agent_name);
    let value = serde_json::json!({
        "status": status,
        "timestamp": Utc::now().to_rfc3339(),
    }).to_string();
    let _ = pg::set_agent_config(pool, &key, &value).await;
}

/// Log that an agent run completed (duration in ms).
pub async fn log_agent_run_complete(
    pool: &pg::PgPool,
    agent_name: &str,
    duration_ms: u64,
) {
    let key = format!("fleet_{}_last_run", agent_name);
    let value = serde_json::json!({
        "completed_at": Utc::now().to_rfc3339(),
        "duration_ms": duration_ms,
    }).to_string();
    let _ = pg::set_agent_config(pool, &key, &value).await;
}

/// Check if a specific agent is enabled via agent_config.
pub async fn is_agent_enabled(state: &FleetState, name: &str) -> bool {
    let map = state.agent_enabled.read().await;
    *map.get(name).unwrap_or(&true)
}

/// Reload enabled flags from PG agent_config into the shared map.
pub async fn reload_enabled_flags(state: &FleetState) {
    let configs = pg::get_all_agent_config(&state.pg_pool).await.unwrap_or_default();
    let mut map = state.agent_enabled.write().await;
    for name in &AGENT_NAMES {
        let key = format!("fleet_{}_enabled", name);
        let enabled = configs.iter()
            .find(|(k, _)| k == &key)
            .map(|(_, v)| v != "false")
            .unwrap_or(true);
        map.insert(name.to_string(), enabled);
    }
}

/// Seed default enabled=true for each agent in agent_config (if not set).
pub async fn seed_fleet_config(pool: &pg::PgPool) {
    for name in &AGENT_NAMES {
        let key = format!("fleet_{}_enabled", name);
        if pg::get_agent_config(pool, &key).await.ok().flatten().is_none() {
            let _ = pg::set_agent_config(pool, &key, "true").await;
        }
    }
}
