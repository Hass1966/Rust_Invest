/// agent — Autonomous monitoring and self-improvement agent
/// ========================================================
/// Runs as a long-lived process alongside serve.rs.
/// Agent data (actions, metrics, config) stored in PostgreSQL.
/// Signal accuracy reads from SQLite (signal_history table).
/// Implements a 4-phase Observe-Decide-Act-Evaluate loop every 6 hours.
///
/// Usage: cargo run --release --bin agent

use rust_invest::*;
use chrono::{Utc, Timelike};
use std::collections::HashMap;

const LOOP_INTERVAL_SECS: u64 = 6 * 3600; // 6 hours
const SERVE_URL: &str = "http://localhost:8081";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       ALPHA SIGNAL — AUTONOMOUS AGENT (Observe-Decide-Act)     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let db_path = "rust_invest.db";
    let http_client = reqwest::Client::new();

    // Initialise PostgreSQL pool for agent data
    let pg_pool = pg::create_pool()?;
    println!("  PostgreSQL pool created (agent data)");

    // Initialise SQLite for signal accuracy reads
    {
        let database = db::Database::new(db_path)?;
        database.set_wal_mode();
    }

    // Load and seed config from PG
    let config = agent::AgentConfig::load_from_pg(&pg_pool).await;
    config.save_to_pg(&pg_pool).await;
    println!("  Agent config loaded:");
    println!("    enabled: {}", config.enabled);
    println!("    approval_required: {}", config.approval_required);
    println!("    accuracy_crisis: <{:.0}%", config.accuracy_crisis_threshold);
    println!("    accuracy_degradation: <{:.0}%", config.accuracy_degradation_threshold);
    println!("    staleness: >{}d", config.staleness_days);
    println!("    retrain_cooldown: {}h", config.retrain_cooldown_hours);
    println!("    max_daily_retrains: {}", config.max_daily_retrains);

    let mut state = agent::AgentState::default();
    println!("\n  Agent starting main loop (every {} hours)...\n", LOOP_INTERVAL_SECS / 3600);

    // Run immediately on startup, then every 6 hours
    loop {
        let start = std::time::Instant::now();
        state.phase = "running".to_string();
        state.total_runs += 1;

        println!("\n{}", "=".repeat(70));
        println!("  [Agent] Run #{} at {}", state.total_runs, Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!("{}\n", "=".repeat(70));

        // SQLite for signal accuracy reads
        let database = match db::Database::new(db_path) {
            Ok(d) => { d.set_wal_mode(); d }
            Err(e) => {
                eprintln!("  [Agent] SQLite error: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                continue;
            }
        };
        // Reload config from PG each run (allows live config updates via API)
        let config = agent::AgentConfig::load_from_pg(&pg_pool).await;

        if !config.enabled {
            println!("  [Agent] Disabled via config. Sleeping...");
            state.phase = "disabled".to_string();
            tokio::time::sleep(tokio::time::Duration::from_secs(LOOP_INTERVAL_SECS)).await;
            continue;
        }

        // ══════════════════════════════
        // PHASE 1: OBSERVE
        // ══════════════════════════════
        println!("  ┌── PHASE 1: OBSERVE ──┐");
        let observations = observe(&database, &config);
        println!("  │  Scanned {} assets", observations.len());

        let timestamp = Utc::now().to_rfc3339();

        // Log metrics to PG agent_metrics
        for obs in &observations {
            let _ = pg::insert_agent_metric(
                &pg_pool, &timestamp, &obs.asset, "accuracy_7d", obs.accuracy_7d, Some(7), None,
            ).await;
            let _ = pg::insert_agent_metric(
                &pg_pool, &timestamp, &obs.asset, "accuracy_30d", obs.accuracy_30d, Some(30), None,
            ).await;
            if let Some(drift) = obs.drift_score {
                let _ = pg::insert_agent_metric(
                    &pg_pool, &timestamp, &obs.asset, "drift_score", drift, None, None,
                ).await;
            }
            if let Some(age) = obs.model_age_days {
                let _ = pg::insert_agent_metric(
                    &pg_pool, &timestamp, &obs.asset, "model_age_days", age as f64, None, None,
                ).await;
            }
        }
        println!("  │  Logged {} metric snapshots", observations.len() * 4);
        println!("  └────────────────────────┘\n");

        // ══════════════════════════════
        // PHASE 2: DECIDE
        // ══════════════════════════════
        println!("  ┌── PHASE 2: DECIDE ──┐");
        let proposed = decide(&database, &config, &observations);
        println!("  │  Proposed {} actions", proposed.len());

        // Log all proposals to PG agent_actions
        let mut action_ids: Vec<(i64, agent::ProposedAction)> = Vec::new();
        for action in &proposed {
            let details_json = serde_json::to_string(&action.details).ok();
            match pg::insert_agent_action(
                &pg_pool,
                &action.action_type,
                action.asset.as_deref(),
                &action.trigger_reason,
                "proposed",
                action.accuracy_before,
                details_json.as_deref(),
            ).await {
                Ok(id) => {
                    action_ids.push((id, action.clone()));
                    println!("  │  [{}] {} — {} (acc: {:.1}%)",
                        id, action.action_type,
                        action.asset.as_deref().unwrap_or("global"),
                        action.accuracy_before.unwrap_or(0.0));
                }
                Err(e) => eprintln!("  │  Failed to log action: {}", e),
            }
        }
        state.actions_proposed += proposed.len() as u64;
        println!("  └────────────────────────┘\n");

        // ══════════════════════════════
        // PHASE 3: ACT
        // ══════════════════════════════
        println!("  ┌── PHASE 3: ACT ──┐");
        let mut executed = 0;

        if config.approval_required {
            println!("  │  Approval mode ON — actions require manual approval");
        } else {
            for (action_id, action) in &action_ids {
                // Check daily limits via PG
                if let Ok(daily_count) = pg::get_daily_retrain_count(&pg_pool).await {
                    if daily_count >= config.max_daily_retrains && action.action_type == "retrain" {
                        println!("  │  Daily retrain limit reached ({}/{}), skipping",
                            daily_count, config.max_daily_retrains);
                        continue;
                    }
                }

                if action.action_type == "retrain" && agent::is_us_market_hours() {
                    println!("  │  US market open ({}) — deferring retrain for {} to overnight",
                        chrono::Utc::now().format("%H:%M UTC"),
                        action.asset.as_deref().unwrap_or("?"));
                    continue;
                }

                // Check cooldown via PG
                if let Some(ref asset) = action.asset {
                    if let Ok(recent) = pg::get_recent_action_count(
                        &pg_pool, asset, &action.action_type, config.retrain_cooldown_hours
                    ).await {
                        if recent > 0 {
                            println!("  │  Cooldown active for {} ({}h window)",
                                asset, config.retrain_cooldown_hours);
                            continue;
                        }
                    }
                }

                match action.action_type.as_str() {
                    "retrain" => {
                        if let Some(ref asset) = action.asset {
                            println!("  │  Executing retrain for {}...", asset);
                            match execute_retrain(asset, db_path, &http_client).await {
                                Ok(train_result) => {
                                    let _ = pg::update_agent_action_status(
                                        &pg_pool, *action_id, "executed", Some(train_result.post_accuracy),
                                    ).await;
                                    executed += 1;
                                    println!("  │  ✓ {} retrained: {:.1}% → {:.1}%",
                                        asset, train_result.pre_accuracy, train_result.post_accuracy);
                                }
                                Err(e) => {
                                    let _ = pg::update_agent_action_status(&pg_pool, *action_id, "failed", None).await;
                                    eprintln!("  │  ✗ {} retrain failed: {}", asset, e);
                                }
                            }
                        }
                    }
                    "threshold_adjust" => {
                        if let Some(ref asset) = action.asset {
                            println!("  │  Adjusting thresholds for {}...", asset);
                            if execute_threshold_adjust(asset, &action.details) {
                                let _ = pg::update_agent_action_status(&pg_pool, *action_id, "executed", None).await;
                                executed += 1;
                                println!("  │  ✓ {} thresholds updated", asset);
                            } else {
                                let _ = pg::update_agent_action_status(&pg_pool, *action_id, "failed", None).await;
                            }
                        }
                    }
                    "ensemble_override" => {
                        if let Some(ref asset) = action.asset {
                            println!("  │  Updating ensemble weights for {}...", asset);
                            if execute_ensemble_override(asset, &action.details) {
                                let _ = pg::update_agent_action_status(&pg_pool, *action_id, "executed", None).await;
                                executed += 1;
                                println!("  │  ✓ {} ensemble updated", asset);
                            } else {
                                let _ = pg::update_agent_action_status(&pg_pool, *action_id, "failed", None).await;
                            }
                        }
                    }
                    _ => {
                        println!("  │  Unknown action type: {}", action.action_type);
                    }
                }
            }
        }
        state.actions_executed += executed as u64;
        println!("  │  Executed {}/{} actions", executed, action_ids.len());
        println!("  └────────────────────────┘\n");

        // ══════════════════════════════
        // PHASE 4: EVALUATE
        // ══════════════════════════════
        println!("  ┌── PHASE 4: EVALUATE ──┐");
        let evaluated = evaluate(&database, &config, db_path, &pg_pool).await;
        state.actions_rolled_back += evaluated.rolled_back as u64;
        println!("  │  Evaluated: {} actions, {} rolled back", evaluated.total, evaluated.rolled_back);
        println!("  └────────────────────────┘\n");

        let elapsed = start.elapsed();
        state.last_run = Some(Utc::now().to_rfc3339());
        state.last_run_duration_ms = Some(elapsed.as_millis() as u64);
        state.phase = "idle".to_string();

        // Save state to PG
        let state_json = serde_json::to_string(&state).unwrap_or_default();
        let _ = pg::set_agent_config(&pg_pool, "agent_state", &state_json).await;

        println!("  [Agent] Run #{} complete in {:.1}s. Sleeping {}h...",
            state.total_runs, elapsed.as_secs_f64(), LOOP_INTERVAL_SECS / 3600);

        tokio::time::sleep(tokio::time::Duration::from_secs(LOOP_INTERVAL_SECS)).await;
    }
}

// ════════════════════════════════════════
// OBSERVE — Gather per-asset accuracy and model health
// ════════════════════════════════════════

struct AssetObservation {
    asset: String,
    accuracy_7d: f64,
    accuracy_30d: f64,
    signal_count_7d: usize,
    training_accuracy: f64,
    drift_score: Option<f64>,
    model_age_days: Option<i64>,
    buy_accuracy_7d: f64,
    sell_accuracy_7d: f64,
}

fn observe(database: &db::Database, _config: &agent::AgentConfig) -> Vec<AssetObservation> {
    let mut observations = Vec::new();

    // Get all assets with recent resolved signals
    let assets = match database.get_active_assets(30) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("  [Observe] Failed to get active assets: {}", e);
            return observations;
        }
    };

    // Filter out descoped FX and crypto assets
    let crypto_ids = ["bitcoin","ethereum","solana","ripple","dogecoin","cardano",
        "avalanche-2","chainlink","polkadot","near","sui","aptos","arbitrum",
        "the-open-network","uniswap","tron","litecoin","shiba-inu","stellar","matic-network"];
    let assets: Vec<String> = assets.into_iter().filter(|a| {
        !a.ends_with("=X") && !crypto_ids.contains(&a.as_str())
    }).collect();

    for asset in &assets {
        let acc_7d = database.get_asset_accuracy(asset, 7).unwrap_or_else(|_| db::AssetAccuracy {
            asset: asset.clone(), total_signals: 0, correct_signals: 0,
            accuracy: 0.0, buy_accuracy: 0.0, sell_accuracy: 0.0, buy_count: 0, sell_count: 0,
        });
        let acc_30d = database.get_asset_accuracy(asset, 30).unwrap_or_else(|_| db::AssetAccuracy {
            asset: asset.clone(), total_signals: 0, correct_signals: 0,
            accuracy: 0.0, buy_accuracy: 0.0, sell_accuracy: 0.0, buy_count: 0, sell_count: 0,
        });

        // Training accuracy from saved model
        let training_accuracy = model_store::load_model_accuracy(asset);

        // Drift score: (training_accuracy - recent_accuracy) weighted
        let drift_score = if training_accuracy > 0.0 && acc_7d.total_signals >= 5 {
            Some(training_accuracy - acc_7d.accuracy)
        } else {
            None
        };

        // Model age from the linreg file (representative)
        let model_age = model_store::model_age_days(&model_store::model_path(asset, "linreg"));

        observations.push(AssetObservation {
            asset: asset.clone(),
            accuracy_7d: acc_7d.accuracy,
            accuracy_30d: acc_30d.accuracy,
            signal_count_7d: acc_7d.total_signals,
            training_accuracy,
            drift_score,
            model_age_days: model_age,
            buy_accuracy_7d: acc_7d.buy_accuracy,
            sell_accuracy_7d: acc_7d.sell_accuracy,
        });
    }

    observations
}

// ════════════════════════════════════════
// DECIDE — Apply rules to identify underperformers
// ════════════════════════════════════════

fn decide(
    database: &db::Database,
    config: &agent::AgentConfig,
    observations: &[AssetObservation],
) -> Vec<agent::ProposedAction> {
    let mut actions = Vec::new();

    for obs in observations {
        // Rule 1: Accuracy crisis — 7d accuracy < 35% with 10+ signals → retrain all
        if obs.accuracy_7d < config.accuracy_crisis_threshold
            && obs.signal_count_7d >= config.min_signals_for_action
        {
            actions.push(agent::ProposedAction {
                action_type: "retrain".to_string(),
                asset: Some(obs.asset.clone()),
                trigger_reason: format!(
                    "Accuracy crisis: 7d accuracy {:.1}% < {:.0}% threshold ({} signals)",
                    obs.accuracy_7d, config.accuracy_crisis_threshold, obs.signal_count_7d
                ),
                accuracy_before: Some(obs.accuracy_7d),
                details: {
                    let mut d = HashMap::new();
                    d.insert("rule".into(), "accuracy_crisis".into());
                    d.insert("accuracy_7d".into(), serde_json::json!(obs.accuracy_7d));
                    d.insert("signal_count".into(), serde_json::json!(obs.signal_count_7d));
                    d.insert("scope".into(), "all_models".into());
                    d
                },
            });
            continue; // Crisis takes priority, skip other rules for this asset
        }

        // Rule 2: Accuracy degradation — 7d accuracy < 45% AND dropped 10pp from training
        if obs.accuracy_7d < config.accuracy_degradation_threshold
            && obs.signal_count_7d >= config.min_signals_for_action
        {
            if let Some(drift) = obs.drift_score {
                if drift >= config.degradation_drop_pp {
                    actions.push(agent::ProposedAction {
                        action_type: "retrain".to_string(),
                        asset: Some(obs.asset.clone()),
                        trigger_reason: format!(
                            "Accuracy degradation: 7d {:.1}% (training was {:.1}%, drift {:.1}pp)",
                            obs.accuracy_7d, obs.training_accuracy, drift
                        ),
                        accuracy_before: Some(obs.accuracy_7d),
                        details: {
                            let mut d = HashMap::new();
                            d.insert("rule".into(), "accuracy_degradation".into());
                            d.insert("accuracy_7d".into(), serde_json::json!(obs.accuracy_7d));
                            d.insert("training_accuracy".into(), serde_json::json!(obs.training_accuracy));
                            d.insert("drift_pp".into(), serde_json::json!(drift));
                            d.insert("scope".into(), "worst_model".into());
                            d
                        },
                    });
                    continue;
                }
            }
        }

        // Rule 3: Staleness — model older than staleness_days
        if let Some(age) = obs.model_age_days {
            if age > config.staleness_days {
                actions.push(agent::ProposedAction {
                    action_type: "retrain".to_string(),
                    asset: Some(obs.asset.clone()),
                    trigger_reason: format!(
                        "Model stale: {}d old (threshold: {}d)",
                        age, config.staleness_days
                    ),
                    accuracy_before: Some(obs.accuracy_7d),
                    details: {
                        let mut d = HashMap::new();
                        d.insert("rule".into(), "staleness".into());
                        d.insert("model_age_days".into(), serde_json::json!(age));
                        d.insert("scope".into(), "scheduled_retrain".into());
                        d
                    },
                });
                continue;
            }
        }

        // Rule 4: Asymmetric accuracy — BUY acc < 40% but SELL acc > 55% → tighten BUY threshold
        if obs.buy_accuracy_7d < config.asymmetric_buy_threshold
            && obs.sell_accuracy_7d > config.asymmetric_sell_threshold
            && obs.signal_count_7d >= config.min_signals_for_action
        {
            actions.push(agent::ProposedAction {
                action_type: "threshold_adjust".to_string(),
                asset: Some(obs.asset.clone()),
                trigger_reason: format!(
                    "Asymmetric accuracy: BUY {:.1}% < {:.0}%, SELL {:.1}% > {:.0}%",
                    obs.buy_accuracy_7d, config.asymmetric_buy_threshold,
                    obs.sell_accuracy_7d, config.asymmetric_sell_threshold
                ),
                accuracy_before: Some(obs.accuracy_7d),
                details: {
                    let mut d = HashMap::new();
                    d.insert("rule".into(), "asymmetric_accuracy".into());
                    d.insert("buy_accuracy".into(), serde_json::json!(obs.buy_accuracy_7d));
                    d.insert("sell_accuracy".into(), serde_json::json!(obs.sell_accuracy_7d));
                    d.insert("new_buy_threshold".into(), serde_json::json!(agent::DEFAULT_BUY_THRESHOLD_TIGHTEN));
                    d
                },
            });
        }
    }

    // Rule 5: Model dominance check (30d window, per-model accuracy)
    // This would require per-model accuracy tracking which we get from model_store
    // For now, check if any single model is dominant via training accuracies
    check_model_dominance(database, config, observations, &mut actions);

    actions
}

/// Check if any single model dominates the ensemble for 30d
fn check_model_dominance(
    _database: &db::Database,
    config: &agent::AgentConfig,
    observations: &[AssetObservation],
    actions: &mut Vec<agent::ProposedAction>,
) {
    for obs in observations {
        // Load individual model accuracies from saved files
        let linreg_acc = load_single_model_accuracy(&obs.asset, "linreg");
        let logreg_acc = load_single_model_accuracy(&obs.asset, "logreg");
        let gbt_acc = load_single_model_accuracy(&obs.asset, "gbt");

        let accs = [linreg_acc, logreg_acc, gbt_acc];
        let avg: f64 = accs.iter().sum::<f64>() / accs.len() as f64;

        for (i, &acc) in accs.iter().enumerate() {
            if acc > avg + config.model_dominance_pp && acc > 0.0 {
                let model_name = match i {
                    0 => "linreg",
                    1 => "logreg",
                    _ => "gbt",
                };
                actions.push(agent::ProposedAction {
                    action_type: "ensemble_override".to_string(),
                    asset: Some(obs.asset.clone()),
                    trigger_reason: format!(
                        "Model dominance: {} at {:.1}% vs avg {:.1}% (+{:.1}pp)",
                        model_name, acc, avg, acc - avg
                    ),
                    accuracy_before: Some(obs.accuracy_30d),
                    details: {
                        let mut d = HashMap::new();
                        d.insert("rule".into(), "model_dominance".into());
                        d.insert("dominant_model".into(), model_name.into());
                        d.insert("model_accuracy".into(), serde_json::json!(acc));
                        d.insert("ensemble_avg".into(), serde_json::json!(avg));
                        d
                    },
                });
                break; // One override per asset max
            }
        }
    }
}

fn load_single_model_accuracy(symbol: &str, model_type: &str) -> f64 {
    let path = model_store::model_path(symbol, model_type);
    if let Ok(contents) = std::fs::read_to_string(&path) {
        if let Ok(saved) = serde_json::from_str::<model_store::SavedWeights>(&contents) {
            return saved.meta.walk_forward_accuracy;
        }
        if let Ok(saved) = serde_json::from_str::<model_store::SavedGBT>(&contents) {
            return saved.meta.walk_forward_accuracy;
        }
    }
    0.0
}

// ════════════════════════════════════════
// ACT — Execute proposed actions
// ════════════════════════════════════════

async fn execute_retrain(
    symbol: &str,
    db_path: &str,
    http_client: &reqwest::Client,
) -> Result<targeted_train::TrainResult, String> {
    // 1. Backup existing models
    let _ = model_store::backup_models(symbol);

    // 2. Run targeted retrain (blocking — CPU-bound)
    let sym = symbol.to_string();
    let path = db_path.to_string();
    let result = tokio::task::spawn_blocking(move || {
        targeted_train::train_single_asset(&sym, &path)
    }).await.map_err(|e| format!("Spawn error: {}", e))?;

    if !result.success {
        // Restore backup on failure
        let _ = model_store::restore_models(symbol);
        return Err(result.error.unwrap_or_else(|| "Unknown training error".to_string()));
    }

    // 3. Notify serve.rs to reload models
    let reload_url = format!("{}/api/v1/models/reload", SERVE_URL);
    match http_client.post(&reload_url).send().await {
        Ok(resp) => println!("    [Agent] Model reload: {}", resp.status()),
        Err(e) => eprintln!("    [Agent] Model reload failed (serve.rs may be down): {}", e),
    }

    Ok(result)
}

fn execute_threshold_adjust(asset: &str, details: &HashMap<String, serde_json::Value>) -> bool {
    let new_threshold = details.get("new_buy_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(agent::DEFAULT_BUY_THRESHOLD_TIGHTEN);

    let mut overrides = agent::ThresholdOverrides::load();
    overrides.overrides.insert(asset.to_string(), agent::AssetThresholdOverride {
        buy_threshold: Some(new_threshold),
        sell_threshold: None,
        reason: format!("Agent: asymmetric accuracy adjustment"),
        set_by: "agent".to_string(),
        set_at: Utc::now().to_rfc3339(),
        expires_at: Some((Utc::now() + chrono::Duration::days(7)).to_rfc3339()),
    });

    match overrides.save() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("    [Agent] Threshold save failed: {}", e);
            false
        }
    }
}

fn execute_ensemble_override(asset: &str, details: &HashMap<String, serde_json::Value>) -> bool {
    let dominant_model = details.get("dominant_model")
        .and_then(|v| v.as_str())
        .unwrap_or("gbt");

    // Load current overrides
    let path = "config/ensemble_overrides.json";
    let mut overrides: HashMap<String, serde_json::Value> = match std::fs::read_to_string(path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => HashMap::new(),
    };

    // Set the dominant model as primary (others still active but dominant model preferred)
    let entry = serde_json::json!({
        "use_linreg": true,
        "use_logreg": true,
        "use_gbt": true,
        "dominant_model": dominant_model,
        "reason": format!("Agent: {} dominant (auto-detected)", dominant_model)
    });
    overrides.insert(asset.to_string(), entry);

    match serde_json::to_string_pretty(&overrides) {
        Ok(json) => {
            let _ = std::fs::create_dir_all("config");
            std::fs::write(path, json).is_ok()
        }
        Err(_) => false,
    }
}

// ════════════════════════════════════════
// EVALUATE — Check post-action accuracy, rollback if needed
// ════════════════════════════════════════

struct EvalResult {
    total: usize,
    rolled_back: usize,
}

async fn evaluate(
    database: &db::Database,
    config: &agent::AgentConfig,
    _db_path: &str,
    pg_pool: &pg::PgPool,
) -> EvalResult {
    let mut result = EvalResult { total: 0, rolled_back: 0 };

    // Find executed actions older than eval_wait_hours (from PG)
    let pending_evals = match pg::get_pending_evaluations(pg_pool, config.eval_wait_hours).await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  [Evaluate] Error fetching pending evals: {}", e);
            return result;
        }
    };

    for action in &pending_evals {
        result.total += 1;

        if let Some(ref asset) = action.asset {
            // Get current accuracy from SQLite (signal_history)
            let current_acc = database.get_asset_accuracy(asset, 7)
                .map(|a| a.accuracy)
                .unwrap_or(0.0);
            let pre_acc = action.accuracy_before.unwrap_or(0.0);
            let acc_change = current_acc - pre_acc;

            if acc_change < -config.rollback_threshold_pp {
                println!("  │  Rolling back {} (acc: {:.1}% → {:.1}%, delta: {:.1}pp)",
                    asset, pre_acc, current_acc, acc_change);

                if model_store::has_backup(asset) {
                    if model_store::restore_models(asset).is_ok() {
                        let _ = pg::update_agent_action_status(
                            pg_pool, action.id, "rolled_back", Some(current_acc),
                        ).await;
                        result.rolled_back += 1;

                        let reload_url = format!("{}/api/v1/models/reload", SERVE_URL);
                        let client = reqwest::Client::new();
                        let _ = client.post(&reload_url).send().await;
                    } else {
                        let _ = pg::update_agent_action_status(
                            pg_pool, action.id, "rollback_failed", Some(current_acc),
                        ).await;
                    }
                } else {
                    let _ = pg::update_agent_action_status(
                        pg_pool, action.id, "evaluated", Some(current_acc),
                    ).await;
                    println!("  │  No backup available for {} — cannot rollback", asset);
                }
            } else {
                let _ = pg::update_agent_action_status(
                    pg_pool, action.id, "evaluated", Some(current_acc),
                ).await;
                if acc_change > 0.0 {
                    println!("  │  ✓ {} improved: {:.1}% → {:.1}% (+{:.1}pp)",
                        asset, pre_acc, current_acc, acc_change);
                } else {
                    println!("  │  ~ {} stable: {:.1}% → {:.1}% ({:.1}pp)",
                        asset, pre_acc, current_acc, acc_change);
                }
            }
        } else {
            let _ = pg::update_agent_action_status(pg_pool, action.id, "evaluated", None).await;
        }
    }

    result
}
