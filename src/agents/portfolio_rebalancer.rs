/// Agent 5: Portfolio Rebalancer — daily at 21:00 UTC
/// ===================================================
/// Half-Kelly position sizing, sector exposure checks,
/// generates rebalance proposals.

use std::sync::Arc;
use chrono::{Utc, Timelike};
use crate::{pg, agents::{FleetState, log_agent_heartbeat, log_agent_run_complete, is_agent_enabled, SERVE_URL}};

const AGENT_NAME: &str = "portfolio_rebalancer";
const CHECK_INTERVAL_SECS: u64 = 3600; // check hourly, act at 21:00
const TARGET_HOUR: u32 = 21;
const MAX_POSITION_PCT: f64 = 15.0;
const MAX_SECTOR_PCT: f64 = 40.0;

pub async fn run_loop(state: Arc<FleetState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(CHECK_INTERVAL_SECS));
    loop {
        interval.tick().await;
        if !is_agent_enabled(&state, AGENT_NAME).await {
            continue;
        }
        // Only run at target hour
        if Utc::now().hour() != TARGET_HOUR {
            continue;
        }
        log_agent_heartbeat(&state.pg_pool, AGENT_NAME, "running").await;
        let start = std::time::Instant::now();

        if let Err(e) = run_once(&state).await {
            eprintln!("[{}] Error: {}", AGENT_NAME, e);
            log_agent_heartbeat(&state.pg_pool, AGENT_NAME, "error").await;
        } else {
            let ms = start.elapsed().as_millis() as u64;
            log_agent_run_complete(&state.pg_pool, AGENT_NAME, ms).await;
            log_agent_heartbeat(&state.pg_pool, AGENT_NAME, "idle").await;
        }
    }
}

async fn run_once(state: &FleetState) -> Result<(), pg::PgError> {
    let timestamp = Utc::now().to_rfc3339();

    // Fetch current signals with accuracy/confidence data
    let signals = fetch_signal_data(&state.http_client).await;
    if signals.is_empty() {
        println!("[{}] No signals available", AGENT_NAME);
        return Ok(());
    }

    // Compute half-Kelly sizing for each BUY signal
    let mut proposals: Vec<serde_json::Value> = Vec::new();
    let mut sector_exposure: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for sig in &signals {
        let asset = sig.get("asset").and_then(|v| v.as_str()).unwrap_or("");
        let signal = sig.get("signal").and_then(|v| v.as_str()).unwrap_or("");
        let confidence = sig.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let sector = sig.get("sector").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();

        if signal != "BUY" {
            continue;
        }

        // Half-Kelly: f = (p*b - q)/b where p=win rate, b=avg win/avg loss, q=1-p
        // Use confidence as proxy for p
        let p = confidence.max(0.01).min(0.99);
        let b = 1.5; // assumed win/loss ratio
        let q = 1.0 - p;
        let kelly = (p * b - q) / b;
        let half_kelly = (kelly / 2.0).max(0.0).min(MAX_POSITION_PCT / 100.0);
        let position_pct = half_kelly * 100.0;

        // Track sector exposure
        *sector_exposure.entry(sector.clone()).or_insert(0.0) += position_pct;

        proposals.push(serde_json::json!({
            "asset": asset,
            "signal": signal,
            "confidence": confidence,
            "kelly_fraction": kelly,
            "half_kelly_pct": position_pct,
            "sector": sector,
        }));

        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, asset,
            "kelly_fraction", kelly,
            None, None,
        ).await;
    }

    // Check sector exposure warnings
    for (sector, exposure) in &sector_exposure {
        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, sector,
            "sector_exposure", *exposure,
            None, None,
        ).await;

        if *exposure > MAX_SECTOR_PCT {
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "sector_exposure_warning",
                None,
                &format!("{} sector at {:.1}% (limit {:.1}%)", sector, exposure, MAX_SECTOR_PCT),
                "executed",
                None,
                Some(&serde_json::json!({
                    "sector": sector,
                    "exposure_pct": exposure,
                    "limit_pct": MAX_SECTOR_PCT,
                }).to_string()),
            ).await;
        }
    }

    // Log rebalance proposal
    if !proposals.is_empty() {
        let _ = pg::insert_agent_action(
            &state.pg_pool,
            "rebalance_proposal",
            None,
            &format!("{} positions sized via half-Kelly", proposals.len()),
            "proposed",
            None,
            Some(&serde_json::json!({
                "positions": proposals,
                "sector_exposure": sector_exposure,
                "timestamp": timestamp,
            }).to_string()),
        ).await;
    }

    println!("[{}] Generated {} position proposals, {} sectors tracked",
        AGENT_NAME, proposals.len(), sector_exposure.len());

    Ok(())
}

async fn fetch_signal_data(client: &reqwest::Client) -> Vec<serde_json::Value> {
    let resp = client.get(format!("{}/api/v1/signals/current", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await;
    match resp {
        Ok(r) => r.json::<Vec<serde_json::Value>>().await.unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}
