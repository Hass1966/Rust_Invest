/// Agent 1: Data Quality Sentinel — every 30 minutes
/// ==================================================
/// Monitors price freshness and API health. Pauses stale assets.

use std::sync::Arc;
use chrono::{Utc, Datelike, NaiveDate};
use crate::{db, pg, agents::{FleetState, log_agent_heartbeat, log_agent_run_complete, is_agent_enabled, SERVE_URL}};

const AGENT_NAME: &str = "data_quality_sentinel";
const INTERVAL_SECS: u64 = 30 * 60; // 30 minutes

/// Main loop — spawned as a tokio task.
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
    let database = db::Database::new(&state.db_path)?;
    database.set_wal_mode();

    let assets = database.get_active_assets(30)?;
    let now = Utc::now();
    let timestamp = now.to_rfc3339();
    let is_trading_day = is_us_trading_day(now.date_naive());

    let mut stale_count = 0u32;
    let mut ok_count = 0u32;

    for asset in &assets {
        let history = database.get_stock_history(asset)?;
        if history.is_empty() {
            continue;
        }

        // Find last price timestamp
        let last_ts = &history.last().unwrap().timestamp;
        let last_dt = chrono::DateTime::parse_from_rfc3339(last_ts)
            .or_else(|_| chrono::DateTime::parse_from_rfc3339(&format!("{}Z", last_ts)))
            .map(|dt| dt.with_timezone(&Utc));

        let staleness_hours = match last_dt {
            Ok(dt) => (now - dt).num_hours() as f64,
            Err(_) => 999.0,
        };

        // Allow 72h on weekends/holidays, 6h on trading days
        let threshold = if is_trading_day { 6.0 } else { 72.0 };
        let is_stale = staleness_hours > threshold;

        // Log metric
        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, asset,
            "price_staleness_hours", staleness_hours,
            None, None,
        ).await;

        if is_stale {
            stale_count += 1;
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "data_quality_pause",
                Some(asset),
                &format!("Price stale for {:.0}h (threshold {:.0}h)", staleness_hours, threshold),
                "executed",
                None,
                Some(&serde_json::json!({
                    "staleness_hours": staleness_hours,
                    "threshold_hours": threshold,
                    "is_trading_day": is_trading_day,
                }).to_string()),
            ).await;
        } else {
            ok_count += 1;
        }
    }

    // Check API health
    let api_ok = check_api_health(&state.http_client).await;
    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, "_system",
        "api_health", if api_ok { 1.0 } else { 0.0 },
        None, None,
    ).await;

    if !api_ok {
        let _ = pg::insert_agent_action(
            &state.pg_pool,
            "data_quality_api_failure",
            None,
            "API health check failed",
            "executed",
            None,
            None,
        ).await;
    }

    println!("[{}] Scanned {} assets: {} ok, {} stale, API {}",
        AGENT_NAME, assets.len(), ok_count, stale_count,
        if api_ok { "healthy" } else { "DOWN" });

    Ok(())
}

async fn check_api_health(client: &reqwest::Client) -> bool {
    match client.get(format!("{}/health", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

fn is_us_trading_day(date: NaiveDate) -> bool {
    let weekday = date.weekday();
    if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
        return false;
    }
    // Major US holidays (approximate — good enough for staleness checks)
    let month = date.month();
    let day = date.day();
    let is_holiday = matches!((month, day),
        (1, 1) | (1, 20) |  // New Year, MLK
        (2, 17) |            // Presidents Day
        (3, 29) |            // Good Friday (approximate)
        (5, 26) |            // Memorial Day (approximate)
        (6, 19) |            // Juneteenth
        (7, 4) |             // Independence Day
        (9, 1) |             // Labor Day (approximate)
        (11, 27) |           // Thanksgiving (approximate)
        (12, 25)             // Christmas
    );
    !is_holiday
}
