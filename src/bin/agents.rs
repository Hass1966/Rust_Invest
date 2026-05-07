/// agents — Unified Fleet Scheduler for Alpha Signal
/// ==================================================
/// Runs 6 autonomous agents on independent schedules:
///   1. Data Quality Sentinel  (30 min)
///   2. Risk Guardian          (60 min)
///   3. Model Drift Detector   (6h)
///   4. Signal Arbitrator      (60 min)
///   5. Portfolio Rebalancer   (daily 21:00 UTC)
///   6. Performance Analyst    (daily 22:00 UTC)
///
/// All agents share a PgPool and write to agent_actions / agent_metrics.
///
/// Usage: cargo run --release --bin agents

use rust_invest::*;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     ALPHA SIGNAL — AGENT FLEET (6 autonomous agents)           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let db_path = "rust_invest.db".to_string();

    // Verify SQLite is accessible
    {
        let database = db::Database::new(&db_path)?;
        database.set_wal_mode();
        println!("  SQLite: OK ({})", db_path);
    }

    // PostgreSQL pool
    let pg_pool = pg::create_pool()?;
    {
        let client = pg_pool.get().await?;
        let row = client.query_one("SELECT 1", &[]).await?;
        let _: i32 = row.get(0);
        println!("  PostgreSQL: OK (alpha_signal)");
    }

    // Seed fleet config defaults
    agents::seed_fleet_config(&pg_pool).await;
    println!("  Fleet config seeded");

    // Build shared state
    let state = Arc::new(agents::FleetState {
        pg_pool,
        db_path,
        http_client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?,
        signal_overrides: Arc::new(RwLock::new(HashMap::new())),
        risk_halt: Arc::new(RwLock::new(false)),
        agent_enabled: Arc::new(RwLock::new(HashMap::new())),
    });

    // Load initial enabled flags
    agents::reload_enabled_flags(&state).await;

    println!("\n  Starting 6 agents...\n");
    for name in &agents::AGENT_NAMES {
        let enabled = agents::is_agent_enabled(&state, name).await;
        println!("    {} — {}", name, if enabled { "enabled" } else { "DISABLED" });
    }
    println!();

    // Spawn all agent loops
    let handles = vec![
        tokio::spawn(agents::data_quality::run_loop(state.clone())),
        tokio::spawn(agents::risk_guardian::run_loop(state.clone())),
        tokio::spawn(agents::model_drift::run_loop(state.clone())),
        tokio::spawn(agents::signal_arbitrator::run_loop(state.clone())),
        tokio::spawn(agents::portfolio_rebalancer::run_loop(state.clone())),
        tokio::spawn(agents::performance_analyst::run_loop(state.clone())),
    ];

    // Also spawn a config reload task (every 5 min)
    let reload_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        loop {
            interval.tick().await;
            agents::reload_enabled_flags(&reload_state).await;
        }
    });

    println!("  All agents running. Press Ctrl+C to stop.\n");

    // Wait for all (they run forever)
    futures::future::join_all(handles).await;

    Ok(())
}
