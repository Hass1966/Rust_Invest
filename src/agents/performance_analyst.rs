/// Agent 6: Performance Analyst — daily at 22:00 UTC
/// ==================================================
/// Sector attribution, regime accuracy, benchmark alpha, feature importance.

use std::sync::Arc;
use chrono::{Utc, Timelike, Duration};
use crate::{pg, agents::{FleetState, log_agent_heartbeat, log_agent_run_complete, is_agent_enabled, SERVE_URL}};

const AGENT_NAME: &str = "performance_analyst";
const CHECK_INTERVAL_SECS: u64 = 3600; // check hourly, act at 22:00
const TARGET_HOUR: u32 = 22;

pub async fn run_loop(state: Arc<FleetState>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(CHECK_INTERVAL_SECS));
    loop {
        interval.tick().await;
        if !is_agent_enabled(&state, AGENT_NAME).await {
            continue;
        }
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
    let since_30d = (Utc::now() - Duration::days(30)).to_rfc3339();

    // Get resolved signals from last 30 days
    let signals = pg::get_resolved_signals(&state.pg_pool, 5000).await?;
    let recent: Vec<_> = signals.iter()
        .filter(|s| s.timestamp > since_30d && s.signal_type != "HOLD")
        .collect();

    if recent.is_empty() {
        println!("[{}] No recent resolved signals to analyse", AGENT_NAME);
        return Ok(());
    }

    // ── Sector Attribution ──
    let mut sector_returns: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for sig in &recent {
        let sector = classify_sector(&sig.asset);
        let ret = match sig.signal_type.as_str() {
            "BUY" => sig.pct_change.unwrap_or(0.0),
            "SELL" | "SHORT" => -sig.pct_change.unwrap_or(0.0),
            _ => 0.0,
        };
        sector_returns.entry(sector).or_default().push(ret);
    }

    let mut sector_summary = Vec::new();
    for (sector, returns) in &sector_returns {
        let total: f64 = returns.iter().sum();
        let avg = total / returns.len() as f64;
        sector_summary.push(serde_json::json!({
            "sector": sector,
            "total_return_pct": total,
            "avg_return_pct": avg,
            "signal_count": returns.len(),
        }));

        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, sector,
            "attribution_score", total,
            Some(30), None,
        ).await;
    }

    // ── Regime Accuracy ──
    let correct = recent.iter().filter(|s| s.was_correct == Some(true)).count();
    let regime_accuracy = 100.0 * correct as f64 / recent.len() as f64;

    let regime = fetch_current_regime(&state.http_client).await;
    let regime_name = regime.as_ref()
        .and_then(|r| r.get("regime").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string();

    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, &regime_name,
        "regime_accuracy", regime_accuracy,
        Some(30), None,
    ).await;

    // ── Benchmark Alpha (vs SPY) ──
    let portfolio = pg::get_daily_portfolio(&state.pg_pool).await?;
    let alpha_bps = compute_alpha_vs_spy(&portfolio, &state.http_client).await;

    let _ = pg::insert_agent_metric(
        &state.pg_pool, &timestamp, "_portfolio",
        "weekly_alpha_bps", alpha_bps,
        Some(30), None,
    ).await;

    // ── Performance Report ──
    let report = serde_json::json!({
        "date": Utc::now().format("%Y-%m-%d").to_string(),
        "regime": regime_name,
        "regime_accuracy_pct": regime_accuracy,
        "actionable_signals_30d": recent.len(),
        "correct_30d": correct,
        "alpha_vs_spy_bps": alpha_bps,
        "sector_attribution": sector_summary,
    });

    let _ = pg::insert_agent_action(
        &state.pg_pool,
        "performance_report",
        None,
        &format!("30d accuracy {:.1}%, alpha {}bps, {} signals",
            regime_accuracy, alpha_bps as i64, recent.len()),
        "executed",
        Some(regime_accuracy),
        Some(&report.to_string()),
    ).await;

    println!("[{}] 30d accuracy: {:.1}%, alpha: {:.0}bps, {} signals across {} sectors",
        AGENT_NAME, regime_accuracy, alpha_bps, recent.len(), sector_returns.len());

    Ok(())
}

fn classify_sector(asset: &str) -> String {
    // Simple sector classification based on known assets
    match asset {
        // Tech
        "AAPL" | "MSFT" | "GOOGL" | "AMZN" | "META" | "NVDA" | "TSM" | "AVGO"
        | "CRM" | "ORCL" | "ADBE" | "AMD" | "INTC" | "QCOM" | "NOW" | "SHOP"
        | "SAP.DE" => "Technology",

        // Healthcare
        "JNJ" | "UNH" | "PFE" | "ABBV" | "LLY" | "MRK" | "BMY" | "CVS" | "CI"
        | "AMGN" | "GILD" | "ISRG" | "TMO" | "ABT" | "DHR" | "MDT" | "REGN"
        | "VRTX" | "SYK" | "BDX" => "Healthcare",

        // Financials
        "JPM" | "BAC" | "GS" | "MS" | "BRK-B" | "V" | "MA" | "AXP" | "BLK"
        | "C" | "WFC" | "SPGI" | "CME" | "ICE" | "AFL" | "TRV"
        | "LGEN.L" | "III.L" | "ALV.DE" => "Financials",

        // Energy
        "XOM" | "CVX" | "COP" | "SLB" | "EOG" | "MPC" | "VLO" | "PSX" | "OXY"
        | "BP.L" | "SHEL.L" => "Energy",

        // Consumer
        "KO" | "PEP" | "PG" | "WMT" | "COST" | "MCD" | "NKE" | "SBUX"
        | "GIS" | "CL" | "MO" | "DIS" | "NFLX" | "HD" | "LOW" | "TGT"
        | "TSCO.L" | "SBRY.L" | "IMB.L" | "BATS.L" => "Consumer",

        // Industrials
        "BA" | "CAT" | "HON" | "UPS" | "RTX" | "DE" | "GE" | "MMM" | "LMT"
        | "AIR.PA" | "SAF.PA" | "SIE.DE" => "Industrials",

        // UK
        a if a.ends_with(".L") => "UK Stocks",

        // European
        a if a.ends_with(".PA") || a.ends_with(".DE") => "European",

        // ETFs
        "SPY" | "QQQ" | "IWM" | "VTI" | "VOO" | "DIA" | "VDE" | "PDBC" | "DBC"
        | "VHT" | "IHI" | "VPU" | "IDU" | "VNQ" | "TLT" | "AGG" | "BND"
        | "VGLT" | "SHY" => "ETFs",

        _ => "Other",
    }.to_string()
}

async fn fetch_current_regime(client: &reqwest::Client) -> Option<serde_json::Value> {
    let resp = client.get(format!("{}/api/v1/market/regime", SERVE_URL))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.ok()?;
    resp.json().await.ok()
}

async fn compute_alpha_vs_spy(
    portfolio: &[(String, f64, f64, f64, f64)],
    client: &reqwest::Client,
) -> f64 {
    // Get last 30 days of portfolio returns
    if portfolio.len() < 2 {
        return 0.0;
    }

    let recent: Vec<_> = portfolio.iter().rev().take(30).collect();
    if recent.len() < 2 {
        return 0.0;
    }

    let portfolio_return = {
        let last = recent.first().map(|r| r.2).unwrap_or(0.0);
        let first = recent.last().map(|r| r.2).unwrap_or(0.0);
        if first > 0.0 { (last - first) / first * 100.0 } else { 0.0 }
    };

    // Get SPY return from simulator data
    let spy_return = fetch_spy_return(client).await;

    // Alpha = portfolio return - benchmark return, in bps
    (portfolio_return - spy_return) * 100.0
}

async fn fetch_spy_return(client: &reqwest::Client) -> f64 {
    let resp = client.get(format!("{}/api/v1/simulator/data", SERVE_URL))
        .timeout(std::time::Duration::from_secs(15))
        .send().await;
    match resp {
        Ok(r) => {
            let data: serde_json::Value = r.json().await.unwrap_or_default();
            // Try to extract SPY 30d return from simulator data
            data.get("spy_30d_return").and_then(|v| v.as_f64()).unwrap_or(0.0)
        }
        Err(_) => 0.0,
    }
}
