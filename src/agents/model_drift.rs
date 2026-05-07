/// Agent 3: Model Drift Detector — every 6 hours
/// ===============================================
/// Detects feature drift and confidence decay. Flags assets for retraining.

use std::sync::Arc;
use chrono::Utc;
use crate::{db, pg, model_store, agents::{FleetState, log_agent_heartbeat, log_agent_run_complete, is_agent_enabled}};

const AGENT_NAME: &str = "model_drift_detector";
const INTERVAL_SECS: u64 = 6 * 3600; // 6 hours
const DRIFT_RETRAIN_THRESHOLD: f64 = 0.3;
const CONFIDENCE_DECAY_THRESHOLD_PP: f64 = 8.0;

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
    let timestamp = Utc::now().to_rfc3339();

    let mut drift_flags = 0u32;
    let mut decay_flags = 0u32;

    for asset in &assets {
        // Load saved model weights (linreg has norm_means/norm_stds)
        let weights = match model_store::load_weights(asset, "linreg") {
            Ok(w) => w,
            Err(_) => continue,
        };

        // Get recent price history for feature drift check
        let history = database.get_stock_history(asset)?;
        if history.len() < 20 {
            continue;
        }

        // Compute recent feature means from last 20 days of prices
        let recent = &history[history.len().saturating_sub(20)..];
        let recent_returns: Vec<f64> = recent.windows(2)
            .map(|w| (w[1].price - w[0].price) / w[0].price)
            .collect();

        if recent_returns.is_empty() {
            continue;
        }

        let recent_mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let recent_std = {
            let variance = recent_returns.iter()
                .map(|r| (r - recent_mean).powi(2))
                .sum::<f64>() / recent_returns.len() as f64;
            variance.sqrt()
        };

        // Compare with training norm stats (feature drift)
        let drift_score = if !weights.norm_means.is_empty() && !weights.norm_stds.is_empty() {
            // Average absolute z-score deviation across stored norm parameters
            let mut total_z = 0.0f64;
            let n = weights.norm_means.len().min(weights.norm_stds.len());
            if n > 0 {
                // Use first feature (price return) as representative
                let training_mean = weights.norm_means[0];
                let training_std = weights.norm_stds[0].max(1e-10);
                let z_mean = ((recent_mean - training_mean) / training_std).abs();
                let z_std = if training_std > 1e-10 {
                    ((recent_std - training_std) / training_std).abs()
                } else {
                    0.0
                };
                total_z = (z_mean + z_std) / 2.0;
            }
            total_z
        } else {
            0.0
        };

        // Log drift metric
        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, asset,
            "feature_drift_score", drift_score,
            None, None,
        ).await;

        // Check 7-day accuracy from resolved signals
        let since = (Utc::now() - chrono::Duration::days(7)).to_rfc3339();
        let signals = pg::get_signals_for_asset_since(&state.pg_pool, asset, &since).await?;
        let resolved_actionable: Vec<_> = signals.iter()
            .filter(|s| s.resolution_ts.is_some() && s.signal_type != "HOLD")
            .collect();
        let correct = resolved_actionable.iter()
            .filter(|s| s.was_correct == Some(true))
            .count();
        let rolling_accuracy = if resolved_actionable.is_empty() {
            50.0 // default
        } else {
            100.0 * correct as f64 / resolved_actionable.len() as f64
        };

        // Confidence decay: compare training accuracy to rolling accuracy
        let training_accuracy = weights.meta.walk_forward_accuracy;
        let confidence_decay = training_accuracy - rolling_accuracy;

        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, asset,
            "confidence_decay", confidence_decay,
            Some(7), None,
        ).await;

        let _ = pg::insert_agent_metric(
            &state.pg_pool, &timestamp, asset,
            "model_rolling_accuracy", rolling_accuracy,
            Some(7), None,
        ).await;

        // Flag for retraining if drift is high
        if drift_score > DRIFT_RETRAIN_THRESHOLD {
            drift_flags += 1;
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "drift_retrain_flag",
                Some(asset),
                &format!("Feature drift {:.2} > {:.2} threshold", drift_score, DRIFT_RETRAIN_THRESHOLD),
                "proposed",
                Some(rolling_accuracy),
                Some(&serde_json::json!({
                    "drift_score": drift_score,
                    "rolling_accuracy_7d": rolling_accuracy,
                    "training_accuracy": training_accuracy,
                }).to_string()),
            ).await;
        }

        // Flag if accuracy has decayed significantly
        if confidence_decay > CONFIDENCE_DECAY_THRESHOLD_PP && resolved_actionable.len() >= 7 {
            decay_flags += 1;
            let _ = pg::insert_agent_action(
                &state.pg_pool,
                "drift_weight_adjust",
                Some(asset),
                &format!("Accuracy dropped {:.1}pp (train {:.1}% → live {:.1}%)",
                    confidence_decay, training_accuracy, rolling_accuracy),
                "proposed",
                Some(rolling_accuracy),
                Some(&serde_json::json!({
                    "confidence_decay_pp": confidence_decay,
                    "training_accuracy": training_accuracy,
                    "rolling_accuracy": rolling_accuracy,
                    "signal_count": resolved_actionable.len(),
                }).to_string()),
            ).await;
        }
    }

    println!("[{}] Scanned {} assets: {} drift flags, {} decay flags",
        AGENT_NAME, assets.len(), drift_flags, decay_flags);

    Ok(())
}
