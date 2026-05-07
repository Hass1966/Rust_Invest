/// pg — PostgreSQL data layer for Alpha Signal
/// =============================================
/// Replaces SQLite for the signal pipeline. All signal writes, reads,
/// and portfolio operations go through this module.
///
/// Connection: alpha_signal database on the same Postgres instance
/// as agent_alpha. Max pool size: 15 connections.

use deadpool_postgres::{Config, Pool, Runtime, ManagerConfig, RecyclingMethod};
use tokio_postgres::NoTls;
use chrono::{DateTime, Utc};
use crate::db;

/// Canonical asset universe filter. Only these asset classes are
/// written to and read from the signals pipeline.
pub const ASSET_UNIVERSE: &[&str] = &["stock"];

pub type PgPool = Pool;
pub type PgError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone)]
pub struct SignalRow {
    pub id: i64,
    pub timestamp: String,
    pub asset: String,
    pub asset_class: String,
    pub signal_type: String,
    pub price_at_signal: f64,
    pub confidence: f64,
    pub linreg_prob: Option<f64>,
    pub logreg_prob: Option<f64>,
    pub gbt_prob: Option<f64>,
    pub outcome_price: Option<f64>,
    pub pct_change: Option<f64>,
    pub was_correct: Option<bool>,
    pub resolution_ts: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PredictionRow {
    pub id: i64,
    pub timestamp: String,
    pub asset: String,
    pub signal: String,
    pub confidence: f64,
    pub price_at_prediction: f64,
}

#[derive(Debug, Clone)]
pub struct SizingRejection {
    pub asset: String,
    pub signal_type: String,
    pub requested_amount: f64,
    pub available_capital: f64,
    pub total_committed: f64,
    pub max_allowed: f64,
    pub reason: String,
}

/// Create a connection pool to the alpha_signal database.
/// Max 15 connections.
pub fn create_pool() -> Result<Pool, PgError> {
    let mut cfg = Config::new();
    cfg.host = Some("localhost".to_string());
    cfg.port = Some(5434);
    cfg.dbname = Some("alpha_signal".to_string());
    cfg.user = Some("agent".to_string());
    cfg.password = Some("agent".to_string());
    cfg.manager = Some(ManagerConfig {
        recycling_method: RecyclingMethod::Fast,
    });

    // Override from env if set
    if let Ok(url) = std::env::var("DATABASE_URL_ALPHA") {
        // Parse postgresql:// URL
        if let Ok(parsed) = url.parse::<deadpool_postgres::tokio_postgres::config::Config>() {
            let _ = parsed; // just validate it parses
        }
    }

    let pool = cfg.create_pool(Some(Runtime::Tokio1), NoTls)?;
    Ok(pool)
}

// ── Signal writes ──

pub async fn insert_signal(
    pool: &Pool,
    timestamp: &str,
    asset: &str,
    asset_class: &str,
    signal_type: &str,
    price_at_signal: f64,
    confidence: f64,
    linreg_prob: f64,
    logreg_prob: f64,
    gbt_prob: f64,
) -> Result<(), PgError> {
    if !ASSET_UNIVERSE.contains(&asset_class) {
        return Ok(());
    }

    let ts: DateTime<Utc> = timestamp.parse()
        .unwrap_or_else(|_| Utc::now());
    let signal_date = ts.date_naive();
    let client = pool.get().await?;
    // One signal per asset per calendar day. First write wins; subsequent
    // hourly refreshes update signal/price/confidence but don't create duplicates.
    client.execute(
        "INSERT INTO signals (timestamp, asset, asset_class, signal_type, price_at_signal,
            confidence, linreg_prob, logreg_prob, gbt_prob, signal_date)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
         ON CONFLICT (asset, signal_date) DO UPDATE SET
            signal_type = EXCLUDED.signal_type,
            price_at_signal = EXCLUDED.price_at_signal,
            confidence = EXCLUDED.confidence,
            linreg_prob = EXCLUDED.linreg_prob,
            logreg_prob = EXCLUDED.logreg_prob,
            gbt_prob = EXCLUDED.gbt_prob
         WHERE signals.resolution_ts IS NULL",
        &[&ts, &asset, &asset_class, &signal_type,
          &price_at_signal, &confidence, &linreg_prob, &logreg_prob, &gbt_prob, &signal_date],
    ).await?;

    Ok(())
}

pub async fn insert_signal_snapshot(
    pool: &Pool,
    timestamp: &str,
    asset: &str,
    asset_class: &str,
    signal: &str,
    confidence: f64,
    probability_up: f64,
    model_agreement: &str,
    rsi: f64,
    trend: &str,
    price: f64,
    model_version: i32,
    quality: &str,
    reason: &str,
    suggested_action: &str,
) -> Result<(), PgError> {
    if !ASSET_UNIVERSE.contains(&asset_class) {
        return Ok(());
    }

    let ts: DateTime<Utc> = timestamp.parse()
        .unwrap_or_else(|_| Utc::now());
    let client = pool.get().await?;
    client.execute(
        "INSERT INTO signal_snapshots (timestamp, asset, asset_class, signal, confidence,
            probability_up, model_agreement, rsi, trend, price, model_version,
            quality, reason, suggested_action)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
        &[&ts, &asset, &asset_class, &signal, &confidence,
          &probability_up, &model_agreement, &rsi, &trend, &price,
          &model_version, &quality, &reason, &suggested_action],
    ).await?;

    Ok(())
}

// ── Prediction writes ──

pub async fn insert_prediction(
    pool: &Pool,
    timestamp: &str,
    asset: &str,
    signal: &str,
    confidence: f64,
    price_at_prediction: f64,
) -> Result<(), PgError> {
    let ts: DateTime<Utc> = timestamp.parse()
        .unwrap_or_else(|_| Utc::now());
    let client = pool.get().await?;
    client.execute(
        "INSERT INTO predictions (timestamp, asset, signal, confidence, price_at_prediction)
         VALUES ($1, $2, $3, $4, $5)",
        &[&ts, &asset, &signal, &confidence, &price_at_prediction],
    ).await?;

    Ok(())
}

pub async fn get_pending_predictions(pool: &Pool) -> Result<Vec<PredictionRow>, PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT id, timestamp::text, asset, signal, confidence, price_at_prediction
         FROM predictions WHERE outcome_timestamp IS NULL",
        &[],
    ).await?;

    Ok(rows.iter().map(|r| PredictionRow {
        id: r.get(0),
        timestamp: r.get(1),
        asset: r.get(2),
        signal: r.get(3),
        confidence: r.get(4),
        price_at_prediction: r.get(5),
    }).collect())
}

pub async fn update_prediction_outcome(
    pool: &Pool,
    id: i64,
    actual_direction: &str,
    was_correct: bool,
    price_at_outcome: f64,
    outcome_timestamp: &str,
) -> Result<(), PgError> {
    let ots: DateTime<Utc> = outcome_timestamp.parse()
        .unwrap_or_else(|_| Utc::now());
    let client = pool.get().await?;
    client.execute(
        "UPDATE predictions SET actual_direction = $1, was_correct = $2,
            price_at_outcome = $3, outcome_timestamp = $4
         WHERE id = $5",
        &[&actual_direction, &was_correct, &price_at_outcome, &ots, &id],
    ).await?;

    Ok(())
}

// ── Signal reads ──

fn row_to_signal(r: &tokio_postgres::Row) -> SignalRow {
    SignalRow {
        id: r.get(0),
        timestamp: r.get(1),
        asset: r.get(2),
        asset_class: r.get(3),
        signal_type: r.get(4),
        price_at_signal: r.get(5),
        confidence: r.get(6),
        linreg_prob: r.get(7),
        logreg_prob: r.get(8),
        gbt_prob: r.get(9),
        outcome_price: r.get(10),
        pct_change: r.get(11),
        was_correct: r.get(12),
        resolution_ts: r.get(13),
    }
}

const SIGNAL_COLS: &str =
    "id, timestamp::text, asset, asset_class, signal_type, price_at_signal, \
     confidence, linreg_prob, logreg_prob, gbt_prob, \
     outcome_price, pct_change, was_correct, resolution_ts::text";

pub async fn get_all_unresolved_signals(pool: &Pool) -> Result<Vec<SignalRow>, PgError> {
    let client = pool.get().await?;
    let q = format!(
        "SELECT {} FROM signals WHERE resolution_ts IS NULL AND asset_class = 'stock' ORDER BY timestamp",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

pub async fn get_last_unresolved_signal(pool: &Pool, asset: &str) -> Result<Option<SignalRow>, PgError> {
    let client = pool.get().await?;
    let q = format!(
        "SELECT {} FROM signals WHERE asset = $1 AND resolution_ts IS NULL ORDER BY timestamp DESC LIMIT 1",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&asset]).await?;
    Ok(rows.first().map(row_to_signal))
}

pub async fn resolve_signal(
    pool: &Pool,
    id: i64,
    outcome_price: f64,
    pct_change: f64,
    was_correct: bool,
    resolution_ts: &str,
) -> Result<(), PgError> {
    let rts: DateTime<Utc> = resolution_ts.parse()
        .unwrap_or_else(|_| Utc::now());
    let client = pool.get().await?;
    client.execute(
        "UPDATE signals SET outcome_price = $1, pct_change = $2,
            was_correct = $3, resolution_ts = $4
         WHERE id = $5",
        &[&outcome_price, &pct_change, &was_correct, &rts, &id],
    ).await?;

    Ok(())
}

pub async fn get_resolved_signals(pool: &Pool, limit: i64) -> Result<Vec<SignalRow>, PgError> {
    let client = pool.get().await?;
    let q = format!(
        "SELECT {} FROM signals WHERE resolution_ts IS NOT NULL AND asset_class = 'stock' ORDER BY timestamp DESC LIMIT $1",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&limit]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

pub async fn get_signals_for_asset_since(
    pool: &Pool,
    asset: &str,
    since: &str,
) -> Result<Vec<SignalRow>, PgError> {
    let client = pool.get().await?;
    let since_ts: DateTime<Utc> = since.parse()
        .unwrap_or_else(|_| Utc::now());
    let q = format!(
        "SELECT {} FROM signals WHERE asset = $1 AND timestamp >= $2 ORDER BY timestamp",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&asset, &since_ts]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

/// Get all signals on a given date (for portfolio reconstruction).
pub async fn get_signals_for_date(pool: &Pool, date: &str) -> Result<Vec<SignalRow>, PgError> {
    let parsed_date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| -> PgError { format!("Invalid date '{}': {}", date, e).into() })?;
    let client = pool.get().await?;
    let q = format!(
        "SELECT {} FROM signals WHERE DATE(timestamp) = $1 AND asset_class = 'stock' ORDER BY asset",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&parsed_date]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

/// Get distinct signal dates (for portfolio reconstruction).
pub async fn get_signal_dates(pool: &Pool) -> Result<Vec<String>, PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT DISTINCT DATE(timestamp)::text as dt FROM signals WHERE asset_class = 'stock' ORDER BY dt",
        &[],
    ).await?;
    Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
}

// ── Sizing rejections ──

pub async fn log_sizing_rejection(
    pool: &Pool,
    rejection: &SizingRejection,
) -> Result<(), PgError> {
    let client = pool.get().await?;
    client.execute(
        "INSERT INTO sizing_rejections (asset, signal_type, requested_amount,
            available_capital, total_committed, max_allowed, reason)
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
        &[&rejection.asset, &rejection.signal_type, &rejection.requested_amount,
          &rejection.available_capital, &rejection.total_committed,
          &rejection.max_allowed, &rejection.reason],
    ).await?;

    Ok(())
}

// ── Daily portfolio ──

pub async fn upsert_daily_portfolio(
    pool: &Pool,
    date: &str,
    seed_value: f64,
    portfolio_value: f64,
    daily_return: f64,
    cumulative_return: f64,
    signals_json: Option<&serde_json::Value>,
    model_version: i32,
) -> Result<(), PgError> {
    let parsed_date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| -> PgError { format!("Invalid date '{}': {}", date, e).into() })?;
    let client = pool.get().await?;
    client.execute(
        "INSERT INTO daily_portfolio (date, seed_value, portfolio_value, daily_return,
            cumulative_return, signals_json, model_version)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (date) DO UPDATE SET
            portfolio_value = EXCLUDED.portfolio_value,
            daily_return = EXCLUDED.daily_return,
            cumulative_return = EXCLUDED.cumulative_return,
            signals_json = EXCLUDED.signals_json",
        &[&parsed_date, &seed_value, &portfolio_value, &daily_return,
          &cumulative_return, &signals_json, &model_version],
    ).await?;

    Ok(())
}

pub async fn get_daily_portfolio(pool: &Pool) -> Result<Vec<(String, f64, f64, f64, f64)>, PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT date::text, seed_value, portfolio_value, daily_return, cumulative_return
         FROM daily_portfolio ORDER BY date",
        &[],
    ).await?;

    Ok(rows.iter().map(|r| (
        r.get::<_, String>(0),
        r.get::<_, f64>(1),
        r.get::<_, f64>(2),
        r.get::<_, f64>(3),
        r.get::<_, f64>(4),
    )).collect())
}

/// Get all signals (resolved + unresolved), ordered by timestamp DESC.
/// Used by the signal truth / track record endpoint.
pub async fn get_all_signals(pool: &Pool, limit: i64) -> Result<Vec<SignalRow>, PgError> {
    let client = pool.get().await?;
    let q = format!(
        "SELECT {} FROM signals WHERE asset_class = 'stock' ORDER BY timestamp DESC LIMIT $1",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&limit]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

/// Get all signals since a given date (for managed simulation / live portfolio).
pub async fn get_signals_since(pool: &Pool, since: &str) -> Result<Vec<SignalRow>, PgError> {
    let since_dt = chrono::DateTime::parse_from_rfc3339(since)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .or_else(|_| {
            // Try date-only format
            chrono::NaiveDate::parse_from_str(&since[..10], "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
        })
        .map_err(|e| -> PgError { format!("Invalid date '{}': {}", since, e).into() })?;
    let client = pool.get().await?;
    let q = format!(
        "SELECT {} FROM signals WHERE timestamp >= $1 AND asset_class = 'stock' ORDER BY timestamp",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&since_dt]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

/// Get total signal counts and accuracy stats (for API endpoints).
pub async fn get_signal_stats(pool: &Pool) -> Result<serde_json::Value, PgError> {
    let client = pool.get().await?;
    let row = client.query_one(
        "SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN resolution_ts IS NOT NULL THEN 1 END) as resolved,
            COUNT(CASE WHEN resolution_ts IS NULL THEN 1 END) as pending,
            COUNT(CASE WHEN was_correct AND signal_type IN ('BUY','SELL','SHORT') THEN 1 END) as correct_actionable,
            COUNT(CASE WHEN resolution_ts IS NOT NULL AND signal_type IN ('BUY','SELL','SHORT') THEN 1 END) as resolved_actionable,
            COUNT(CASE WHEN was_correct AND signal_type = 'BUY' THEN 1 END) as buy_correct,
            COUNT(CASE WHEN resolution_ts IS NOT NULL AND signal_type = 'BUY' THEN 1 END) as buy_resolved,
            COUNT(CASE WHEN was_correct AND signal_type IN ('SELL','SHORT') THEN 1 END) as sell_correct,
            COUNT(CASE WHEN resolution_ts IS NOT NULL AND signal_type IN ('SELL','SHORT') THEN 1 END) as sell_resolved,
            COALESCE(AVG(CASE WHEN resolution_ts IS NOT NULL AND signal_type IN ('BUY','SELL','SHORT') THEN
                CASE WHEN signal_type = 'BUY' THEN pct_change
                     WHEN signal_type IN ('SELL','SHORT') THEN -pct_change
                END
            END), 0) as expected_value_pct
         FROM signals WHERE asset_class = 'stock'",
        &[],
    ).await?;

    let total: i64 = row.get(0);
    let resolved: i64 = row.get(1);
    let pending: i64 = row.get(2);
    let correct_act: i64 = row.get(3);
    let resolved_act: i64 = row.get(4);
    let buy_correct: i64 = row.get(5);
    let buy_resolved: i64 = row.get(6);
    let sell_correct: i64 = row.get(7);
    let sell_resolved: i64 = row.get(8);
    let ev_pct: f64 = row.get(9);

    Ok(serde_json::json!({
        "total_signals": total,
        "resolved": resolved,
        "pending": pending,
        "actionable_accuracy": if resolved_act > 0 { 100.0 * correct_act as f64 / resolved_act as f64 } else { 0.0 },
        "buy_accuracy": if buy_resolved > 0 { 100.0 * buy_correct as f64 / buy_resolved as f64 } else { 0.0 },
        "sell_accuracy": if sell_resolved > 0 { 100.0 * sell_correct as f64 / sell_resolved as f64 } else { 0.0 },
        "buy_count": buy_resolved,
        "sell_count": sell_resolved,
        "expected_value_bps": ev_pct * 100.0, // pct to bps
    }))
}

// ── Agent CRUD ──

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentActionRow {
    pub id: i64,
    pub action_type: String,
    pub asset: Option<String>,
    pub trigger_reason: String,
    pub status: String,
    pub accuracy_before: Option<f64>,
    pub accuracy_after: Option<f64>,
    pub details_json: Option<String>,
    pub created_at: Option<String>,
    pub executed_at: Option<String>,
    pub evaluated_at: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentMetricRow {
    pub id: i64,
    pub timestamp: String,
    pub asset: String,
    pub metric_type: String,
    pub value: f64,
    pub window_days: Option<i32>,
    pub details_json: Option<String>,
}

pub async fn insert_agent_action(
    pool: &Pool,
    action_type: &str,
    asset: Option<&str>,
    trigger_reason: &str,
    status: &str,
    accuracy_before: Option<f64>,
    details_json: Option<&str>,
) -> Result<i64, PgError> {
    let client = pool.get().await?;
    let row = client.query_one(
        "INSERT INTO agent_actions
            (action_type, asset, trigger_reason, status, accuracy_before, details_json)
         VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        &[&action_type, &asset, &trigger_reason, &status, &accuracy_before, &details_json],
    ).await?;
    Ok(row.get(0))
}

pub async fn update_agent_action_status(
    pool: &Pool,
    id: i64,
    status: &str,
    accuracy_after: Option<f64>,
) -> Result<(), PgError> {
    let client = pool.get().await?;
    match status {
        "executed" => {
            client.execute(
                "UPDATE agent_actions SET status = $1, executed_at = NOW() WHERE id = $2",
                &[&status, &id],
            ).await?;
        }
        "evaluated" | "rolled_back" | "approved" | "rejected" => {
            client.execute(
                "UPDATE agent_actions SET status = $1, accuracy_after = $2, evaluated_at = NOW() WHERE id = $3",
                &[&status, &accuracy_after, &id],
            ).await?;
        }
        _ => {
            client.execute(
                "UPDATE agent_actions SET status = $1 WHERE id = $2",
                &[&status, &id],
            ).await?;
        }
    }
    Ok(())
}

pub async fn get_agent_actions(
    pool: &Pool,
    limit: i64,
    status_filter: Option<&str>,
) -> Result<Vec<AgentActionRow>, PgError> {
    let client = pool.get().await?;
    let rows = match status_filter {
        Some(s) => {
            client.query(
                "SELECT id, action_type, asset, trigger_reason, status, accuracy_before,
                        accuracy_after, details_json, created_at::text, executed_at::text, evaluated_at::text
                 FROM agent_actions WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
                &[&s, &limit],
            ).await?
        }
        None => {
            client.query(
                "SELECT id, action_type, asset, trigger_reason, status, accuracy_before,
                        accuracy_after, details_json, created_at::text, executed_at::text, evaluated_at::text
                 FROM agent_actions ORDER BY created_at DESC LIMIT $1",
                &[&limit],
            ).await?
        }
    };
    Ok(rows.iter().map(|r| AgentActionRow {
        id: r.get(0),
        action_type: r.get(1),
        asset: r.get(2),
        trigger_reason: r.get(3),
        status: r.get(4),
        accuracy_before: r.get(5),
        accuracy_after: r.get(6),
        details_json: r.get(7),
        created_at: r.get(8),
        executed_at: r.get(9),
        evaluated_at: r.get(10),
    }).collect())
}

pub async fn get_agent_metrics(
    pool: &Pool,
    asset: Option<&str>,
    metric_type: Option<&str>,
    limit: i64,
) -> Result<Vec<AgentMetricRow>, PgError> {
    let client = pool.get().await?;
    let rows = match (asset, metric_type) {
        (Some(a), Some(m)) => {
            client.query(
                "SELECT id, timestamp::text, asset, metric_type, value, window_days, details_json
                 FROM agent_metrics WHERE asset = $1 AND metric_type = $2
                 ORDER BY timestamp DESC LIMIT $3",
                &[&a, &m, &limit],
            ).await?
        }
        (Some(a), None) => {
            client.query(
                "SELECT id, timestamp::text, asset, metric_type, value, window_days, details_json
                 FROM agent_metrics WHERE asset = $1
                 ORDER BY timestamp DESC LIMIT $2",
                &[&a, &limit],
            ).await?
        }
        (None, Some(m)) => {
            client.query(
                "SELECT id, timestamp::text, asset, metric_type, value, window_days, details_json
                 FROM agent_metrics WHERE metric_type = $1
                 ORDER BY timestamp DESC LIMIT $2",
                &[&m, &limit],
            ).await?
        }
        (None, None) => {
            client.query(
                "SELECT id, timestamp::text, asset, metric_type, value, window_days, details_json
                 FROM agent_metrics ORDER BY timestamp DESC LIMIT $1",
                &[&limit],
            ).await?
        }
    };
    Ok(rows.iter().map(|r| AgentMetricRow {
        id: r.get(0),
        timestamp: r.get(1),
        asset: r.get(2),
        metric_type: r.get(3),
        value: r.get(4),
        window_days: r.get(5),
        details_json: r.get(6),
    }).collect())
}

pub async fn insert_agent_metric(
    pool: &Pool,
    timestamp: &str,
    asset: &str,
    metric_type: &str,
    value: f64,
    window_days: Option<i32>,
    details_json: Option<&str>,
) -> Result<(), PgError> {
    let ts: DateTime<Utc> = timestamp.parse().unwrap_or_else(|_| Utc::now());
    let client = pool.get().await?;
    client.execute(
        "INSERT INTO agent_metrics (timestamp, asset, metric_type, value, window_days, details_json)
         VALUES ($1, $2, $3, $4, $5, $6)",
        &[&ts, &asset, &metric_type, &value, &window_days, &details_json],
    ).await?;
    Ok(())
}

pub async fn get_agent_config(pool: &Pool, key: &str) -> Result<Option<String>, PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT value FROM agent_config WHERE key = $1",
        &[&key],
    ).await?;
    Ok(rows.first().map(|r| r.get(0)))
}

pub async fn set_agent_config(pool: &Pool, key: &str, value: &str) -> Result<(), PgError> {
    let client = pool.get().await?;
    client.execute(
        "INSERT INTO agent_config (key, value, updated_at) VALUES ($1, $2, NOW())
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
        &[&key, &value],
    ).await?;
    Ok(())
}

pub async fn get_all_agent_config(pool: &Pool) -> Result<Vec<(String, String)>, PgError> {
    let client = pool.get().await?;
    let rows = client.query("SELECT key, value FROM agent_config ORDER BY key", &[]).await?;
    Ok(rows.iter().map(|r| (r.get(0), r.get(1))).collect())
}

pub async fn get_daily_retrain_count(pool: &Pool) -> Result<usize, PgError> {
    let client = pool.get().await?;
    let row = client.query_one(
        "SELECT COUNT(*) FROM agent_actions
         WHERE action_type = 'retrain' AND status IN ('executed', 'approved')
         AND created_at >= CURRENT_DATE",
        &[],
    ).await?;
    let count: i64 = row.get(0);
    Ok(count as usize)
}

pub async fn get_recent_action_count(
    pool: &Pool,
    asset: &str,
    action_type: &str,
    hours: i32,
) -> Result<usize, PgError> {
    let client = pool.get().await?;
    let row = client.query_one(
        "SELECT COUNT(*) FROM agent_actions
         WHERE asset = $1 AND action_type = $2
         AND created_at >= NOW() - ($3 || ' hours')::interval
         AND status IN ('executed', 'approved')",
        &[&asset, &action_type, &hours.to_string()],
    ).await?;
    let count: i64 = row.get(0);
    Ok(count as usize)
}

pub async fn get_pending_evaluations(
    pool: &Pool,
    min_age_hours: i32,
) -> Result<Vec<AgentActionRow>, PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT id, action_type, asset, trigger_reason, status, accuracy_before,
                accuracy_after, details_json, created_at::text, executed_at::text, evaluated_at::text
         FROM agent_actions
         WHERE status = 'executed' AND executed_at <= NOW() - ($1 || ' hours')::interval
         ORDER BY executed_at ASC",
        &[&min_age_hours.to_string()],
    ).await?;
    Ok(rows.iter().map(|r| AgentActionRow {
        id: r.get(0),
        action_type: r.get(1),
        asset: r.get(2),
        trigger_reason: r.get(3),
        status: r.get(4),
        accuracy_before: r.get(5),
        accuracy_after: r.get(6),
        details_json: r.get(7),
        created_at: r.get(8),
        executed_at: r.get(9),
        evaluated_at: r.get(10),
    }).collect())
}

/// Aggregate agent summary: by type, by status, retrain outcomes, monthly success rate, recent 7d activity
pub async fn get_agent_summary(pool: &Pool) -> Result<serde_json::Value, PgError> {
    let client = pool.get().await?;

    // By action type
    let by_type = client.query(
        "SELECT action_type, COUNT(*) FROM agent_actions GROUP BY action_type ORDER BY COUNT(*) DESC",
        &[],
    ).await?;

    // By status
    let by_status = client.query(
        "SELECT status, COUNT(*) FROM agent_actions GROUP BY status ORDER BY COUNT(*) DESC",
        &[],
    ).await?;

    // Retrain outcomes (evaluated retrains)
    let retrain_outcomes = client.query_one(
        "SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN accuracy_after > accuracy_before THEN 1 END) as improved,
            COUNT(CASE WHEN accuracy_after <= accuracy_before AND accuracy_after IS NOT NULL THEN 1 END) as not_improved,
            COUNT(CASE WHEN status = 'rolled_back' THEN 1 END) as rolled_back,
            COALESCE(AVG(accuracy_after - accuracy_before) FILTER (WHERE accuracy_after IS NOT NULL), 0) as avg_delta
         FROM agent_actions WHERE action_type = 'retrain' AND status IN ('evaluated', 'rolled_back')",
        &[],
    ).await?;

    // Monthly success rate (last 6 months)
    let monthly = client.query(
        "SELECT TO_CHAR(created_at, 'YYYY-MM') as month,
                COUNT(*) as total,
                COUNT(CASE WHEN accuracy_after > accuracy_before THEN 1 END) as improved
         FROM agent_actions
         WHERE action_type = 'retrain' AND status IN ('evaluated', 'rolled_back')
         AND created_at >= NOW() - INTERVAL '6 months'
         GROUP BY TO_CHAR(created_at, 'YYYY-MM')
         ORDER BY month",
        &[],
    ).await?;

    // Recent 7d activity
    let recent = client.query(
        "SELECT id, action_type, asset, trigger_reason, status,
                accuracy_before, accuracy_after, created_at::text, executed_at::text
         FROM agent_actions
         WHERE created_at >= NOW() - INTERVAL '7 days'
         ORDER BY created_at DESC LIMIT 50",
        &[],
    ).await?;

    // Total counts
    let totals = client.query_one(
        "SELECT COUNT(*), COUNT(DISTINCT asset) FROM agent_actions", &[],
    ).await?;

    let by_type_json: Vec<serde_json::Value> = by_type.iter().map(|r| {
        let t: String = r.get(0);
        let c: i64 = r.get(1);
        serde_json::json!({"type": t, "count": c})
    }).collect();

    let by_status_json: Vec<serde_json::Value> = by_status.iter().map(|r| {
        let s: String = r.get(0);
        let c: i64 = r.get(1);
        serde_json::json!({"status": s, "count": c})
    }).collect();

    let monthly_json: Vec<serde_json::Value> = monthly.iter().map(|r| {
        let m: String = r.get(0);
        let total: i64 = r.get(1);
        let improved: i64 = r.get(2);
        serde_json::json!({
            "month": m,
            "total": total,
            "improved": improved,
            "success_rate": if total > 0 { 100.0 * improved as f64 / total as f64 } else { 0.0 }
        })
    }).collect();

    let recent_json: Vec<serde_json::Value> = recent.iter().map(|r| {
        serde_json::json!({
            "id": r.get::<_, i64>(0),
            "action_type": r.get::<_, String>(1),
            "asset": r.get::<_, Option<String>>(2),
            "trigger_reason": r.get::<_, String>(3),
            "status": r.get::<_, String>(4),
            "accuracy_before": r.get::<_, Option<f64>>(5),
            "accuracy_after": r.get::<_, Option<f64>>(6),
            "created_at": r.get::<_, Option<String>>(7),
            "executed_at": r.get::<_, Option<String>>(8),
        })
    }).collect();

    let retrain_total: i64 = retrain_outcomes.get(0);
    let retrain_improved: i64 = retrain_outcomes.get(1);
    let retrain_not_improved: i64 = retrain_outcomes.get(2);
    let retrain_rolled_back: i64 = retrain_outcomes.get(3);
    let retrain_avg_delta: f64 = retrain_outcomes.get(4);
    let total_actions: i64 = totals.get(0);
    let unique_assets: i64 = totals.get(1);

    Ok(serde_json::json!({
        "total_actions": total_actions,
        "unique_assets": unique_assets,
        "by_type": by_type_json,
        "by_status": by_status_json,
        "retrain_outcomes": {
            "total": retrain_total,
            "improved": retrain_improved,
            "not_improved": retrain_not_improved,
            "rolled_back": retrain_rolled_back,
            "avg_accuracy_delta_pp": retrain_avg_delta,
            "success_rate": if retrain_total > 0 { 100.0 * retrain_improved as f64 / retrain_total as f64 } else { 0.0 }
        },
        "monthly_trend": monthly_json,
        "recent_7d": recent_json,
    }))
}

/// Get resolved signals filtered by asset suffix (e.g., ".L" for FTSE)
pub async fn get_resolved_signals_by_suffix(
    pool: &Pool,
    suffix: &str,
) -> Result<Vec<SignalRow>, PgError> {
    let client = pool.get().await?;
    let pattern = format!("%{}", suffix);
    let q = format!(
        "SELECT {} FROM signals WHERE asset LIKE $1 AND resolution_ts IS NOT NULL ORDER BY timestamp",
        SIGNAL_COLS
    );
    let rows = client.query(&q, &[&pattern]).await?;
    Ok(rows.iter().map(row_to_signal).collect())
}

/// Parse an optional timestamp string to DateTime<Utc>
fn parse_optional_ts(s: &Option<String>) -> Option<DateTime<Utc>> {
    s.as_ref().and_then(|ts| {
        ts.parse::<DateTime<Utc>>().ok()
            .or_else(|| chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S")
                .ok().map(|dt| dt.and_utc()))
    })
}

/// Migrate agent data from SQLite to PostgreSQL (one-time)
pub async fn migrate_agent_data_to_pg(pool: &Pool, sqlite_db: &db::Database) -> Result<(), PgError> {
    let client = pool.get().await?;

    // Migrate agent_actions
    let actions = sqlite_db.get_agent_actions(100_000, None)
        .map_err(|e| -> PgError { format!("SQLite read error: {}", e).into() })?;
    let action_count = actions.len();
    for a in &actions {
        let created = parse_optional_ts(&a.created_at).unwrap_or_else(Utc::now);
        let executed = parse_optional_ts(&a.executed_at);
        let evaluated = parse_optional_ts(&a.evaluated_at);
        client.execute(
            "INSERT INTO agent_actions
                (action_type, asset, trigger_reason, status, accuracy_before, accuracy_after,
                 details_json, created_at, executed_at, evaluated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            &[&a.action_type, &a.asset, &a.trigger_reason, &a.status,
              &a.accuracy_before, &a.accuracy_after, &a.details_json,
              &created, &executed, &evaluated],
        ).await?;
    }
    eprintln!("[migrate] Migrated {} agent_actions", action_count);

    // Migrate agent_metrics
    let metrics = sqlite_db.get_agent_metrics(None, None, 200_000)
        .map_err(|e| -> PgError { format!("SQLite read error: {}", e).into() })?;
    let metric_count = metrics.len();
    for batch in metrics.chunks(500) {
        for m in batch {
            let ts: DateTime<Utc> = m.timestamp.parse().unwrap_or_else(|_| Utc::now());
            client.execute(
                "INSERT INTO agent_metrics (timestamp, asset, metric_type, value, window_days, details_json)
                 VALUES ($1, $2, $3, $4, $5, $6)",
                &[&ts, &m.asset, &m.metric_type, &m.value, &m.window_days, &m.details_json],
            ).await?;
        }
    }
    eprintln!("[migrate] Migrated {} agent_metrics", metric_count);

    // Migrate agent_config
    let configs = sqlite_db.get_all_agent_config()
        .map_err(|e| -> PgError { format!("SQLite read error: {}", e).into() })?;
    for (key, value) in &configs {
        set_agent_config(pool, key, value).await?;
    }
    eprintln!("[migrate] Migrated {} agent_config entries", configs.len());

    Ok(())
}

// ── Fleet Status Queries ──

/// Get fleet status for all 6 agents. Reads heartbeat + last_run from agent_config.
pub async fn get_fleet_status(pool: &Pool) -> Result<serde_json::Value, PgError> {
    let configs = get_all_agent_config(pool).await?;
    let config_map: std::collections::HashMap<&str, &str> = configs.iter()
        .map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let agent_names = [
        ("data_quality_sentinel", "Data Quality Sentinel", "30m"),
        ("risk_guardian", "Risk Guardian", "60m"),
        ("model_drift_detector", "Model Drift Detector", "6h"),
        ("signal_arbitrator", "Signal Arbitrator", "60m"),
        ("portfolio_rebalancer", "Portfolio Rebalancer", "daily 21:00"),
        ("performance_analyst", "Performance Analyst", "daily 22:00"),
    ];

    // Count today's actions per agent type
    let client = pool.get().await?;
    let today_counts_rows = client.query(
        "SELECT action_type, COUNT(*) FROM agent_actions
         WHERE created_at >= CURRENT_DATE
         GROUP BY action_type",
        &[],
    ).await?;
    let mut today_counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for row in &today_counts_rows {
        let at: String = row.get(0);
        let c: i64 = row.get(1);
        today_counts.insert(at, c);
    }

    let mut agents_json = Vec::new();

    for (name, display, schedule) in &agent_names {
        let hb_key = format!("fleet_{}_heartbeat", name);
        let run_key = format!("fleet_{}_last_run", name);
        let enabled_key = format!("fleet_{}_enabled", name);

        let heartbeat: Option<serde_json::Value> = config_map.get(hb_key.as_str())
            .and_then(|v| serde_json::from_str(v).ok());
        let last_run: Option<serde_json::Value> = config_map.get(run_key.as_str())
            .and_then(|v| serde_json::from_str(v).ok());
        let enabled = config_map.get(enabled_key.as_str())
            .map(|v| *v != "false")
            .unwrap_or(true);

        let status = heartbeat.as_ref()
            .and_then(|h| h.get("status").and_then(|v| v.as_str()))
            .unwrap_or("never_run")
            .to_string();

        let last_heartbeat = heartbeat.as_ref()
            .and_then(|h| h.get("timestamp").and_then(|v| v.as_str()))
            .map(|s| s.to_string());

        let last_run_at = last_run.as_ref()
            .and_then(|r| r.get("completed_at").and_then(|v| v.as_str()))
            .map(|s| s.to_string());

        let duration_ms = last_run.as_ref()
            .and_then(|r| r.get("duration_ms").and_then(|v| v.as_u64()));

        // Count actions for this agent's action types
        let action_prefixes: Vec<&str> = match *name {
            "data_quality_sentinel" => vec!["data_quality_pause", "data_quality_resume", "data_quality_api_failure"],
            "risk_guardian" => vec!["risk_hold_override", "risk_halt_cleared", "concentration_warning"],
            "model_drift_detector" => vec!["drift_retrain_flag", "drift_weight_adjust"],
            "signal_arbitrator" => vec!["signal_veto"],
            "portfolio_rebalancer" => vec!["rebalance_proposal", "sector_exposure_warning"],
            "performance_analyst" => vec!["performance_report"],
            _ => vec![],
        };
        let actions_today: i64 = action_prefixes.iter()
            .filter_map(|p| today_counts.get(*p))
            .sum();

        agents_json.push(serde_json::json!({
            "name": name,
            "display_name": display,
            "schedule": schedule,
            "enabled": enabled,
            "status": status,
            "last_heartbeat": last_heartbeat,
            "last_run_at": last_run_at,
            "duration_ms": duration_ms,
            "actions_today": actions_today,
        }));
    }

    Ok(serde_json::json!({ "agents": agents_json }))
}

/// Get recent fleet activity (actions from all fleet agents).
pub async fn get_fleet_activity(pool: &Pool, limit: i64) -> Result<Vec<AgentActionRow>, PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT id, action_type, asset, trigger_reason, status, accuracy_before,
                accuracy_after, details_json, created_at::text, executed_at::text, evaluated_at::text
         FROM agent_actions
         WHERE action_type IN (
             'data_quality_pause', 'data_quality_resume', 'data_quality_api_failure',
             'risk_hold_override', 'risk_halt_cleared', 'concentration_warning',
             'drift_retrain_flag', 'drift_weight_adjust',
             'signal_veto',
             'rebalance_proposal', 'sector_exposure_warning',
             'performance_report'
         )
         ORDER BY created_at DESC LIMIT $1",
        &[&limit],
    ).await?;
    Ok(rows.iter().map(|r| AgentActionRow {
        id: r.get(0),
        action_type: r.get(1),
        asset: r.get(2),
        trigger_reason: r.get(3),
        status: r.get(4),
        accuracy_before: r.get(5),
        accuracy_after: r.get(6),
        details_json: r.get(7),
        created_at: r.get(8),
        executed_at: r.get(9),
        evaluated_at: r.get(10),
    }).collect())
}
