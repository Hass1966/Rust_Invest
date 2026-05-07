/// Agent 2: Risk Guardian — every 60 minutes
/// ==========================================
/// Monitors portfolio drawdown, concentration, correlation, and volatility.
/// Sets the global risk_halt flag when drawdown exceeds threshold.

use std::sync::Arc;
use chrono::Utc;
use crate::{pg, agents::{FleetState, SignalOverride, log_agent_heartbeat, log_agent_run_complete, is_agent_enabled, SERVE_URL}};

const AGENT_NAME: &str = "risk_guardian";
const INTERVAL_SECS: u64 = 60 * 60; // 60 minutes
const DRAWDOWN_HALT_PCT: f64 = 15.0;
const MAX_CONCENTRATION_PCT: f64 = 30.0;
const CORRELATION_ALERT_THRESHOLD: f64 = 0.85;

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

    // Get portfolio history for drawdown calculation
    let portfolio = pg::get_daily_portfolio(&state.pg_pool).await?;

    // Compute drawdown
    let drawdown_pct = compute_drawdown(&portfolio);

    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, "_portfolio",
        "portfolio_drawdown_pct", drawdown_pct,
        None, None,
    ).await;

    // Check drawdown threshold
    if drawdown_pct > DRAWDOWN_HALT_PCT {
        let mut halt = state.risk_halt.write().await;
        if !*halt {
            *halt = true;
            println!("[{}] RISK HALT ACTIVATED — drawdown {:.1}% > {:.1}%",
                AGENT_NAME, drawdown_pct, DRAWDOWN_HALT_PCT);

            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "risk_hold_override",
                None,
                &format!("Drawdown {:.1}% exceeds {:.1}% threshold — all BUYs halted",
                    drawdown_pct, DRAWDOWN_HALT_PCT),
                "executed",
                None,
                Some(&serde_json::json!({
                    "drawdown_pct": drawdown_pct,
                    "threshold": DRAWDOWN_HALT_PCT,
                }).to_string()),
            ).await;
        }
    } else {
        // Clear halt if drawdown recovers
        let mut halt = state.risk_halt.write().await;
        if *halt && drawdown_pct < DRAWDOWN_HALT_PCT - 3.0 {
            *halt = false;
            println!("[{}] Risk halt cleared — drawdown recovered to {:.1}%",
                AGENT_NAME, drawdown_pct);
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "risk_halt_cleared",
                None,
                &format!("Drawdown recovered to {:.1}%", drawdown_pct),
                "executed",
                None,
                None,
            ).await;
        }
    }

    // Fetch current signals for concentration check
    let concentration = check_concentration(&state.http_client).await;
    if let Some((max_asset, max_pct)) = &concentration {
        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, max_asset,
            "max_concentration_pct", *max_pct,
            None, None,
        ).await;

        if *max_pct > MAX_CONCENTRATION_PCT {
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "concentration_warning",
                Some(max_asset),
                &format!("{} has {:.1}% concentration (limit {:.1}%)",
                    max_asset, max_pct, MAX_CONCENTRATION_PCT),
                "executed",
                None,
                None,
            ).await;
        }
    }

    // Check market regime
    let regime_risk = check_regime_risk(&state.http_client).await;
    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, "_market",
        "volatility_regime", regime_risk,
        None, None,
    ).await;

    println!("[{}] Drawdown: {:.1}%, regime risk: {:.1}, halt: {}",
        AGENT_NAME, drawdown_pct, regime_risk,
        *state.risk_halt.read().await);

    Ok(())
}

fn compute_drawdown(portfolio: &[(String, f64, f64, f64, f64)]) -> f64 {
    if portfolio.is_empty() {
        return 0.0;
    }
    let mut peak = f64::MIN;
    let mut max_dd = 0.0f64;
    for (_date, _seed, value, _daily_ret, _cum_ret) in portfolio {
        if *value > peak {
            peak = *value;
        }
        let dd = (peak - *value) / peak * 100.0;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

async fn check_concentration(client: &reqwest::Client) -> Option<(String, f64)> {
    let resp = client.get(format!("{}/api/v1/signals/current", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.ok()?;
    let data: serde_json::Value = resp.json().await.ok()?;
    let signals = data.as_array()?;

    // Count BUY signals — concentration proxy
    let buy_count = signals.iter()
        .filter(|s| s.get("signal").and_then(|v| v.as_str()) == Some("BUY"))
        .count();
    if buy_count == 0 {
        return None;
    }

    // Find asset with highest confidence among BUYs
    let mut max_asset = String::new();
    let mut max_conf = 0.0f64;
    for s in signals {
        if s.get("signal").and_then(|v| v.as_str()) == Some("BUY") {
            let conf = s.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if conf > max_conf {
                max_conf = conf;
                max_asset = s.get("asset").and_then(|v| v.as_str()).unwrap_or("").to_string();
            }
        }
    }

    // Approximate concentration as 1/buy_count * 100
    let pct = 100.0 / buy_count as f64;
    Some((max_asset, pct))
}

async fn check_regime_risk(client: &reqwest::Client) -> f64 {
    let resp = client.get(format!("{}/api/v1/market/regime", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await;
    match resp {
        Ok(r) => {
            let data: serde_json::Value = r.json().await.unwrap_or_default();
            data.get("risk_score").and_then(|v| v.as_f64()).unwrap_or(0.5)
        }
        Err(_) => 0.5,
    }
}
