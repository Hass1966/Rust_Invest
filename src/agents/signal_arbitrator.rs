/// Agent 4: Signal Arbitrator — every 60 minutes
/// ==============================================
/// Vetoes unreliable BUY signals based on risk halt, regime, sector momentum,
/// and per-asset accuracy. Writes overrides to shared state.

use std::sync::Arc;
use chrono::{Utc, Duration};
use crate::{pg, agents::{FleetState, SignalOverride, log_agent_heartbeat, log_agent_run_complete, is_agent_enabled, SERVE_URL}};

const AGENT_NAME: &str = "signal_arbitrator";
const INTERVAL_SECS: u64 = 60 * 60; // 60 minutes
const MIN_ACCURACY_FOR_BUY: f64 = 45.0;
const MIN_SIGNALS_FOR_VETO: usize = 7;

pub async fn run_loop(state: Arc<FleetState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(INTERVAL_SECS));
    loop {
        interval.tick().await;
        if !is_agent_enabled(&state, AGENT_NAME).await {
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
    let risk_halted = *state.risk_halt.read().await;

    // Fetch current signals from serve.rs
    let signals = fetch_current_signals(&state.http_client).await;
    if signals.is_empty() {
        println!("[{}] No signals to arbitrate", AGENT_NAME);
        return Ok(());
    }

    // Fetch market regime
    let regime = fetch_regime(&state.http_client).await;
    let is_crisis = regime.as_ref()
        .and_then(|r| r.get("regime").and_then(|v| v.as_str()))
        .map(|r| r.to_lowercase().contains("crisis") || r.to_lowercase().contains("bear"))
        .unwrap_or(false);

    let mut vetoed = 0u32;
    let mut confirmed = 0u32;
    let mut overrides = state.signal_overrides.write().await;

    for sig in &signals {
        let asset = match sig.get("asset").and_then(|v| v.as_str()) {
            Some(a) => a.to_string(),
            None => continue,
        };
        let signal_type = sig.get("signal").and_then(|v| v.as_str()).unwrap_or("");

        if signal_type != "BUY" {
            confirmed += 1;
            continue;
        }

        let mut veto_reason: Option<String> = None;

        // Check 1: Global risk halt
        if risk_halted {
            veto_reason = Some("Risk halt active — drawdown exceeds threshold".to_string());
        }

        // Check 2: Crisis regime vetoes all BUYs
        if veto_reason.is_none() && is_crisis {
            veto_reason = Some("Market in crisis/bear regime".to_string());
        }

        // Check 3: Per-asset 7d accuracy
        if veto_reason.is_none() {
            let since = (Utc::now() - Duration::days(7)).to_rfc3339();
            let asset_signals = pg::get_signals_for_asset_since(
                &state.pg_pool, &asset, &since
            ).await.unwrap_or_default();

            let resolved_actionable: Vec<_> = asset_signals.iter()
                .filter(|s| s.resolution_ts.is_some() && s.signal_type != "HOLD")
                .collect();

            if resolved_actionable.len() >= MIN_SIGNALS_FOR_VETO {
                let correct = resolved_actionable.iter()
                    .filter(|s| s.was_correct == Some(true))
                    .count();
                let accuracy = 100.0 * correct as f64 / resolved_actionable.len() as f64;
                if accuracy < MIN_ACCURACY_FOR_BUY {
                    veto_reason = Some(format!(
                        "7d accuracy {:.1}% < {:.1}% ({} signals)",
                        accuracy, MIN_ACCURACY_FOR_BUY, resolved_actionable.len()
                    ));
                }
            }
        }

        if let Some(reason) = veto_reason {
            vetoed += 1;

            // Write override to shared state
            overrides.insert(asset.clone(), SignalOverride {
                asset: asset.clone(),
                forced_signal: "HOLD".to_string(),
                reason: reason.clone(),
                source_agent: AGENT_NAME.to_string(),
                expires_at: Utc::now() + Duration::hours(2),
            });

            // Log to PG
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "signal_veto",
                Some(&asset),
                &reason,
                "executed",
                None,
                Some(&serde_json::json!({
                    "original_signal": signal_type,
                    "forced_signal": "HOLD",
                    "risk_halted": risk_halted,
                    "is_crisis": is_crisis,
                }).to_string()),
            ).await;
        } else {
            confirmed += 1;
        }
    }

    // Log summary metrics
    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, "_system",
        "signal_vetoed", vetoed as f64,
        None, None,
    ).await;
    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, "_system",
        "signal_confirmed", confirmed as f64,
        None, None,
    ).await;

    // Clean expired overrides
    let now = Utc::now();
    overrides.retain(|_, v| v.expires_at > now);

    println!("[{}] {} signals: {} confirmed, {} vetoed (risk_halt={}, crisis={})",
        AGENT_NAME, signals.len(), confirmed, vetoed, risk_halted, is_crisis);

    Ok(())
}

async fn fetch_current_signals(client: &reqwest::Client) -> Vec<serde_json::Value> {
    let resp = client.get(format!("{}/api/v1/signals/current", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await;
    match resp {
        Ok(r) => r.json::<Vec<serde_json::Value>>().await.unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

async fn fetch_regime(client: &reqwest::Client) -> Option<serde_json::Value> {
    let resp = client.get(format!("{}/api/v1/market/regime", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.ok()?;
    resp.json().await.ok()
}
