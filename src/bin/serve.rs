/// serve — Web API server for Alpha Signal
/// ========================================
/// Loads pre-trained model weights, runs inference on current market data,
/// and serves enriched trading signals via an Axum web server on port 8080.
///
/// NO training. NO walk-forward. NO backtesting.
/// Just: load weights → compute features → predict → serve.
///
/// Usage: cargo run --release --bin serve

use rust_invest::*;
use axum::{
    Router,
    routing::{get, post, patch, put, delete},
    extract::{Path, Query, State},
    Json,
    http::StatusCode,
};
use tower_http::services::{ServeDir, ServeFile};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{Utc, Datelike, Timelike, Weekday};

// ════════════════════════════════════════
// Application State
// ════════════════════════════════════════

/// Tracks data quality metrics for monitoring and alerting
#[derive(Debug, Clone, Default)]
struct DataQualityState {
    pg_write_failures: u64,
    pg_write_successes: u64,
    last_pg_failure: Option<String>,
    last_pg_failure_error: Option<String>,
    last_successful_signal_write: Option<String>,
    last_signal_generation: Option<String>,
    stale_assets: Vec<String>,
}

#[derive(Clone)]
struct AppState {
    signals: Arc<RwLock<HashMap<String, enriched_signals::EnrichedSignal>>>,
    asset_config: Arc<RwLock<config::AssetConfig>>,
    regime: Arc<RwLock<Option<market_regime::MarketRegimeState>>>,
    data_quality: Arc<RwLock<DataQualityState>>,
    db_path: String,
    http_client: reqwest::Client,
    rate_limiter: auth::RateLimiter,
    oauth_config: Option<auth::OAuthConfig>,
    pg_pool: pg::PgPool,
}

// ════════════════════════════════════════
// Main
// ════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         ALPHA SIGNAL — SERVE MODE (Web API Server)             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Check that we have trained models
    let cached = model_store::list_cached_models();
    if cached.is_empty() {
        eprintln!("  No cached models found in models/");
        eprintln!("  Run `cargo run --release --bin train` first to train models.");
        return Err("No trained models available".into());
    }
    println!("  Found {} cached model files", cached.len());

    // Load asset config
    let asset_config = config::AssetConfig::load()
        .unwrap_or_else(|e| {
            eprintln!("  Warning: {}", e);
            eprintln!("  Using default asset config (built-in stock/FX lists)");
            default_asset_config()
        });
    println!("  Loaded asset config: {} stocks, {} FX, {} crypto",
        asset_config.stocks.len(), asset_config.fx.len(), asset_config.crypto.len());

    println!("  Chat: Rule-based summaries (no LLM required)");

    let oauth_config = auth::OAuthConfig::from_env();
    match &oauth_config {
        Some(_) => println!("  OAuth: Google + Microsoft configured"),
        None => println!("  OAuth: Not configured (set GOOGLE_CLIENT_ID etc.)"),
    }

    // PostgreSQL: single source of truth for signals + portfolio
    let pg_pool = pg::create_pool()?;
    {
        let _conn = pg_pool.get().await?;
        println!("  PostgreSQL: Connected to alpha_signal");
    }

    let state = AppState {
        signals: Arc::new(RwLock::new(HashMap::new())),
        asset_config: Arc::new(RwLock::new(asset_config)),
        regime: Arc::new(RwLock::new(None)),
        data_quality: Arc::new(RwLock::new(DataQualityState::default())),
        db_path: "rust_invest.db".to_string(),
        http_client: reqwest::Client::new(),
        rate_limiter: auth::RateLimiter::new(),
        oauth_config,
        pg_pool,
    };

    // ── Startup database migrations ──
    run_startup_migrations(&state.db_path);

    // Generate signals on startup (inference only — loads saved weights)
    println!("\n━━━ GENERATING INITIAL SIGNALS (inference only) ━━━\n");
    if let Err(e) = refresh_signals(&state).await {
        eprintln!("  Warning: Initial signal generation failed: {}", e);
        eprintln!("  Server will start with empty signals. Ensure models are trained and database has data.");
    }

    {
        let sigs = state.signals.read().await;
        println!("\n  Initial signals generated: {}", sigs.len());
    }

    // ── Batch-resolve all stale signal_history on startup ──
    {
        let sigs = state.signals.read().await;
        let signals_map = sigs.clone();
        let signals_vec: Vec<enriched_signals::EnrichedSignal> = sigs.values().cloned().collect();
        drop(sigs);
        let db_path = state.db_path.clone();
        tokio::task::spawn_blocking(move || {
            batch_resolve_signal_history(&db_path, &signals_map);
            resolve_pending_predictions(&db_path, &signals_map);
        }).await.ok();

        // Also resolve in Postgres
        batch_resolve_signals_pg(&state.pg_pool, &signals_vec).await;
    }

    // Start hourly scheduler
    let scheduler_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));

        // Skip the first tick (fires immediately) — startup refresh already ran above
        interval.tick().await;

        loop {
            interval.tick().await;
            let now = Utc::now();
            println!("\n  [Scheduler] Hourly refresh at {}", now.format("%H:%M:%S UTC"));

            if let Err(e) = refresh_signals_with_market_hours(&scheduler_state).await {
                eprintln!("  [Scheduler] Error: {}", e);
            }

            // Run portfolio tracker every hour so it's current throughout the day
            {
                let sigs = scheduler_state.signals.read().await;
                let signals_clone: std::collections::HashMap<String, enriched_signals::EnrichedSignal> =
                    sigs.clone();
                drop(sigs);
                let db_path = scheduler_state.db_path.clone();
                tokio::task::spawn_blocking(move || {
                    daily_tracker::run_daily_update(&signals_clone, &db_path);
                }).await.ok();
            }

            // Batch-resolve all pending signal_history + predictions
            {
                let sigs = scheduler_state.signals.read().await;
                let signals_map = sigs.clone();
                drop(sigs);
                let db_path = scheduler_state.db_path.clone();
                tokio::task::spawn_blocking(move || {
                    batch_resolve_signal_history(&db_path, &signals_map);
                    resolve_pending_predictions(&db_path, &signals_map);
                }).await.ok();
            }

            // Send daily email alerts at 7am UTC
            if now.hour() == 7 {
                let sigs = scheduler_state.signals.read().await;
                let signals_clone = sigs.clone();
                drop(sigs);
                let db_path_clone = scheduler_state.db_path.clone();
                if let Some(email_cfg) = email_alerts::EmailConfig::from_env() {
                    tokio::task::spawn_blocking(move || {
                        let database = match db::Database::new(&db_path_clone) {
                            Ok(d) => d,
                            Err(_) => return,
                        };
                        let rt = tokio::runtime::Handle::current();
                        rt.block_on(email_alerts::send_daily_alerts(&database, &signals_clone, &email_cfg));
                    }).await.ok();
                }
            }

            // Fetch fresh sentiment every 6 hours (at 0, 6, 12, 18 UTC)
            if now.hour() % 6 == 0 {
                let sigs = scheduler_state.signals.read().await;
                let asset_symbols: Vec<String> = sigs.keys().cloned().collect();
                drop(sigs);
                let db_path_sent = scheduler_state.db_path.clone();
                let client_sent = scheduler_state.http_client.clone();

                // Spawn sentiment fetch as a separate task
                tokio::task::spawn(async move {
                    let newsapi_key = std::env::var("NEWSAPI_KEY").unwrap_or_default();
                    let has_newsapi = !newsapi_key.is_empty() && newsapi_key != "REPLACE_WHEN_YOU_HAVE_IT";
                    if !has_newsapi && std::env::var("SERPER_API_KEY").is_err() { return; }

                    println!("  [Sentiment] Fetching fresh sentiment for {} assets...", asset_symbols.len());
                    let needs_fetch: Vec<String> = {
                        let conn = match rusqlite::Connection::open(&db_path_sent) {
                            Ok(c) => c,
                            Err(_) => { eprintln!("  [Sentiment] DB open failed"); return; }
                        };
                        asset_symbols.iter()
                            .filter(|s| !news_sentiment::has_today_sentiment(&conn, s))
                            .cloned()
                            .collect()
                    }; // conn dropped here

                    let mut count = asset_symbols.len() - needs_fetch.len();
                    for symbol in &needs_fetch {
                        let newsapi = if has_newsapi { newsapi_key.as_str() } else { "" };
                        // Use _by_path variant: opens its own connection, no Send issues
                        match news_sentiment::fetch_and_store_sentiment_by_path(
                            &client_sent, &db_path_sent, symbol, newsapi,
                        ).await {
                            Ok(_) => count += 1,
                            Err(e) => eprintln!("    [Sentiment] Failed for {}: {}", symbol, e),
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    }
                    println!("  [Sentiment] Done — {} of {} assets have sentiment data", count, asset_symbols.len());
                });
            }

            let sigs = scheduler_state.signals.read().await;
            println!("  [Scheduler] Refresh complete. {} signals cached.", sigs.len());
        }
    });

    // Build router
    let cors = tower_http::cors::CorsLayer::permissive();

    // Serve frontend static files from frontend/dist/
    // Falls back to index.html for SPA routing
    let frontend_dist = std::path::PathBuf::from("frontend/dist");
    let serve_frontend = if frontend_dist.exists() {
        println!("  Serving frontend from: frontend/dist/");
        true
    } else {
        println!("  Note: frontend/dist/ not found — run 'npm run build' in frontend/ to enable static serving");
        println!("  For development, run 'npm run dev' in frontend/ separately");
        false
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/config/assets", get(get_assets))
        .route("/api/v1/models/current", get(get_models))
        .route("/api/v1/models/reload", post(reload_models))
        .route("/api/v1/retrain/:asset", post(retrain_asset))
        .route("/api/v1/signals/current", get(get_all_signals))
        .route("/api/v1/signals/current/stocks", get(get_stock_signals))
        .route("/api/v1/signals/current/fx", get(get_fx_signals))
        .route("/api/v1/signals/current/crypto", get(get_crypto_signals))
        .route("/api/v1/signals/history/:asset", get(get_signal_history))
        .route("/api/v1/portfolio/simulate", get(simulate_portfolio))
        .route("/api/v1/portfolio/daily-tracker", get(get_daily_tracker))
        .route("/api/v1/portfolio/rebalance", post(compute_rebalance))
        .route("/api/v1/history/portfolio", get(get_portfolio_history))
        .route("/api/v1/portfolio/live", get(get_portfolio_live))
        .route("/api/v1/history/signals", get(get_signals_history))
        .route("/api/v1/hints", get(get_hints))
        .route("/api/v1/simulate", post(simulate_signals))
        .route("/api/v1/training/results", get(get_training_results))
        .route("/api/v1/chat", post(chat_handler))
        .route("/api/v1/admin/assets", post(add_asset))
        .route("/api/v1/admin/assets/:symbol", patch(toggle_asset))
        .route("/api/v1/predictions/history", get(get_predictions_history))
        .route("/api/v1/signals/truth", get(get_signal_truth))
        .route("/api/v1/signals/truth/historical", get(get_signal_truth_historical))
        .route("/api/v1/signals/resolve", post(force_resolve_signals))
        .route("/api/v1/auth/register", post(auth_register))
        .route("/api/v1/auth/login", post(auth_login))
        .route("/api/v1/auth/logout", post(auth_logout))
        .route("/api/v1/auth/me", get(auth_me))
        .route("/api/v1/auth/google", get(auth_google_redirect))
        .route("/api/v1/auth/google/callback", get(auth_google_callback))
        .route("/api/v1/auth/microsoft", get(auth_microsoft_redirect))
        .route("/api/v1/auth/microsoft/callback", get(auth_microsoft_callback))
        .route("/api/v1/user-portfolio", get(get_user_holdings).post(add_user_holding))
        .route("/api/v1/user-portfolio/compare", post(compare_portfolio))
        .route("/api/v1/user-portfolio/:id", put(update_user_holding).delete(delete_user_holding))
        .route("/api/v1/sentiment/:symbol", get(get_sentiment))
        .route("/api/v1/email/unsubscribe", get(email_unsubscribe))
        .route("/api/v1/feedback/signal", post(submit_signal_feedback))
        .route("/api/v1/feedback/survey", post(submit_survey_feedback))
        .route("/api/v1/feedback", get(get_feedback))
        .route("/api/v1/simulator/data", get(get_simulator_data))
        .route("/api/v1/simulator/walkforward", get(get_walkforward_data))
        .route("/api/v1/portfolio/managed-simulation", get(get_managed_simulation))
        .route("/api/v1/deep-dive/:asset", get(get_deep_dive))
        .route("/api/v1/market/regime", get(get_market_regime))
        .route("/api/v1/sectors", get(get_sector_overview))
        .route("/api/v1/sectors/backtest", get(get_sector_backtest))
        // Agent endpoints (PostgreSQL)
        .route("/api/v1/agent/status", get(get_agent_status))
        .route("/api/v1/agent/actions", get(get_agent_actions))
        .route("/api/v1/agent/metrics", get(get_agent_metrics))
        .route("/api/v1/agent/approve/:action_id", post(approve_agent_action))
        .route("/api/v1/agent/reject/:action_id", post(reject_agent_action))
        .route("/api/v1/agent/config", get(get_agent_config).patch(update_agent_config))
        .route("/api/v1/agent/summary", get(get_agent_summary))
        // Fleet endpoints (6-agent fleet)
        .route("/api/v1/agents/fleet", get(get_fleet_status_handler))
        .route("/api/v1/agents/activity", get(get_fleet_activity_handler))
        .route("/api/v1/portfolio/ftse", get(get_portfolio_ftse))
        .route("/api/v1/system/health", get(get_system_health))
        .layer(cors.clone())
        .with_state(state);

    // Add static file serving — API routes take priority, SPA fallback for client-side routing
    let app = if serve_frontend {
        let index = frontend_dist.join("index.html");
        app
            .nest_service("/assets", ServeDir::new(frontend_dist.join("assets")))
            .nest_service("/app/assets", ServeDir::new(frontend_dist.join("assets")))
            .nest_service("/app/vite.svg", ServeFile::new(frontend_dist.join("vite.svg")))
            .fallback_service(ServeFile::new(&index))
    } else {
        app
    };

    let port = std::env::var("PORT").unwrap_or_else(|_| "8081".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("\n  Server listening on http://0.0.0.0:{}", port);
    println!("  Endpoints:");
    println!("    GET  /health");
    println!("    GET  /api/v1/config/assets");
    println!("    GET  /api/v1/models/current");
    println!("    POST /api/v1/models/reload");
    println!("    GET  /api/v1/signals/current");
    println!("    GET  /api/v1/signals/current/stocks");
    println!("    GET  /api/v1/signals/current/fx");
    println!("    GET  /api/v1/signals/current/crypto");
    println!("    GET  /api/v1/signals/history/:asset");
    println!("    GET  /api/v1/portfolio/simulate");
    println!("    GET  /api/v1/portfolio/daily-tracker");
    println!("    GET  /api/v1/hints");
    println!("    POST /api/v1/simulate");
    println!("    POST /api/v1/chat");
    println!("    POST /api/v1/admin/assets");
    println!("    PATCH /api/v1/admin/assets/:symbol");
    println!("    GET  /api/v1/predictions/history");
    println!("    GET  /api/v1/market/regime");
    println!("    GET  /api/v1/agent/status");
    println!("    GET  /api/v1/agent/actions");
    println!("    GET  /api/v1/agent/metrics");
    println!("    POST /api/v1/agent/approve/:action_id");
    println!("    POST /api/v1/agent/reject/:action_id");
    println!("    GET/PATCH /api/v1/agent/config\n");

    axum::serve(listener, app).await?;
    Ok(())
}

// ════════════════════════════════════════
// API Handlers
// ════════════════════════════════════════

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let regime = state.regime.read().await;
    let regime_info = match &*regime {
        Some(r) => serde_json::json!({
            "regime": r.regime.to_string(),
            "spy_return_20d_pct": r.spy_return_20d_pct,
            "spy_return_10d_pct": r.spy_return_10d_pct,
            "spy_return_5d_pct": r.spy_return_5d_pct,
            "risk_score": r.risk_score,
            "regime_strength": r.regime_strength,
        }),
        None => serde_json::json!(null),
    };
    Json(serde_json::json!({
        "status": "ok",
        "timestamp": Utc::now().to_rfc3339(),
        "market_regime": regime_info,
    }))
}

async fn get_market_regime(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let regime = state.regime.read().await;
    match &*regime {
        Some(r) => Json(serde_json::json!({
            "regime": r.regime.to_string(),
            "spy_return_20d_pct": r.spy_return_20d_pct,
            "spy_return_10d_pct": r.spy_return_10d_pct,
            "spy_return_5d_pct": r.spy_return_5d_pct,
            "risk_score": r.risk_score,
            "regime_strength": r.regime_strength,
            "spy_price_current": r.spy_price_current,
            "spy_price_20d_ago": r.spy_price_20d_ago,
            "defensive_assets": r.defensive_assets,
            "thresholds": {
                "bear_below_pct": -3.0,
                "bull_above_pct": 3.0,
                "fast_drop_warning": -2.0,
                "fast_drop_crisis": -5.0,
            },
            "timestamp": r.timestamp,
        })),
        None => Json(serde_json::json!({
            "regime": "UNKNOWN",
            "error": "Insufficient SPY data for regime calculation",
        })),
    }
}

async fn get_sector_overview(
    State(state): State<AppState>,
) -> Json<sector::SectorOverview> {
    let sigs = state.signals.read().await;
    let inputs: Vec<sector::SignalInput> = sigs.values().map(|s| {
        sector::SignalInput {
            asset: s.asset.clone(),
            asset_class: s.asset_class.clone(),
            signal: s.signal.clone(),
            probability_up: s.technical.probability_up,
            confidence: s.technical.confidence,
        }
    }).collect();
    Json(sector::build_sector_overview(&inputs))
}

async fn get_sector_backtest() -> Result<Json<serde_json::Value>, StatusCode> {
    let path = "reports/sector_backtest.json";
    match std::fs::read_to_string(path) {
        Ok(raw) => {
            let data: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Json(data))
        }
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_assets(
    State(state): State<AppState>,
) -> Json<config::AssetConfig> {
    let cfg = state.asset_config.read().await;
    Json(cfg.clone())
}

async fn get_models() -> Result<Json<serde_json::Value>, StatusCode> {
    match model_store::load_manifest() {
        Ok(manifest) => {
            let val = serde_json::to_value(manifest).unwrap_or_default();
            Ok(Json(val))
        }
        Err(_) => {
            // Try to generate it from what's on disk
            let all_symbols: Vec<&str> = stocks::STOCK_LIST.iter().map(|s| s.symbol)
                .chain(stocks::FX_LIST.iter().map(|s| s.symbol))
                .collect();
            let manifest = model_store::generate_manifest(&all_symbols);
            let val = serde_json::to_value(manifest).unwrap_or_default();
            Ok(Json(val))
        }
    }
}

async fn reload_models() -> Json<serde_json::Value> {
    let all_symbols: Vec<&str> = stocks::STOCK_LIST.iter().map(|s| s.symbol)
        .chain(stocks::FX_LIST.iter().map(|s| s.symbol))
        .collect();
    let manifest = model_store::generate_manifest(&all_symbols);
    Json(serde_json::json!({
        "status": "reloaded",
        "assets_found": manifest.assets.len(),
        "generated_at": manifest.generated_at,
    }))
}

/// Retrain a single asset's models. Called by agent_alpha for auto-retraining.
async fn retrain_asset(
    Path(asset): Path<String>,
) -> Json<serde_json::Value> {
    let symbol = asset.to_uppercase();
    println!("[retrain] Triggered for {}", symbol);

    // Backup existing models first
    let _ = model_store::backup_models(&symbol);

    // Run training in a blocking task (CPU-bound)
    let sym = symbol.clone();
    let result = tokio::task::spawn_blocking(move || {
        targeted_train::train_single_asset(&sym, "rust_invest.db")
    }).await;

    match result {
        Ok(train_result) if train_result.success => {
            // Regenerate model manifest so serve uses new models
            let all_symbols: Vec<&str> = stocks::STOCK_LIST.iter().map(|s| s.symbol)
                .chain(stocks::FX_LIST.iter().map(|s| s.symbol))
                .collect();
            let _ = model_store::generate_manifest(&all_symbols);

            println!("[retrain] {} complete: {:.1}% -> {:.1}%",
                symbol, train_result.pre_accuracy, train_result.post_accuracy);

            Json(serde_json::json!({
                "status": "success",
                "asset": symbol,
                "pre_accuracy": train_result.pre_accuracy,
                "post_accuracy": train_result.post_accuracy,
                "linreg_accuracy": train_result.linreg_accuracy,
                "logreg_accuracy": train_result.logreg_accuracy,
                "gbt_accuracy": train_result.gbt_accuracy,
            }))
        }
        Ok(train_result) => {
            // Training failed — restore backup
            let _ = model_store::restore_models(&symbol);
            let err = train_result.error.unwrap_or_else(|| "Unknown error".to_string());
            println!("[retrain] {} failed: {}", symbol, err);
            Json(serde_json::json!({
                "status": "failed",
                "asset": symbol,
                "error": err,
            }))
        }
        Err(e) => {
            let _ = model_store::restore_models(&symbol);
            println!("[retrain] {} spawn error: {}", symbol, e);
            Json(serde_json::json!({
                "status": "failed",
                "asset": symbol,
                "error": format!("Spawn error: {}", e),
            }))
        }
    }
}

/// Training results endpoint — serves reports/improved.json with all 6-model accuracies
async fn get_training_results() -> Result<Json<serde_json::Value>, StatusCode> {
    match std::fs::read_to_string("reports/improved.json") {
        Ok(contents) => {
            match serde_json::from_str::<serde_json::Value>(&contents) {
                Ok(val) => Ok(Json(val)),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
        Err(_) => Ok(Json(serde_json::json!({
            "version": "no_data",
            "note": "No training results found. Run `cargo run --release --bin train` first.",
            "assets": {}
        }))),
    }
}

/// GET /api/v1/predictions/history — prediction tracker data
async fn get_predictions_history(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit: usize = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(500);

    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let records = database.get_predictions_history(limit).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Compute stats
        let resolved: Vec<_> = records.iter().filter(|r| r.was_correct.is_some()).collect();
        let correct_count = resolved.iter().filter(|r| r.was_correct == Some(true)).count();
        let total_resolved = resolved.len();

        let now = Utc::now();
        let stats_24h = compute_accuracy_stats(&records, now - chrono::Duration::hours(24), now);
        let stats_7d = compute_accuracy_stats(&records, now - chrono::Duration::days(7), now);
        let stats_30d = compute_accuracy_stats(&records, now - chrono::Duration::days(30), now);

        // Per-asset breakdown
        let mut asset_stats: HashMap<String, (usize, usize)> = HashMap::new();
        for r in &resolved {
            let entry = asset_stats.entry(r.asset.clone()).or_insert((0, 0));
            entry.1 += 1; // total
            if r.was_correct == Some(true) { entry.0 += 1; } // correct
        }
        let per_asset: Vec<serde_json::Value> = {
            let mut sorted: Vec<_> = asset_stats.iter().collect();
            sorted.sort_by_key(|(k, _)| k.clone());
            sorted.iter().map(|(asset, (correct, total))| {
                serde_json::json!({
                    "asset": asset,
                    "correct": correct,
                    "total": total,
                    "accuracy": if *total > 0 { *correct as f64 / *total as f64 * 100.0 } else { 0.0 },
                })
            }).collect()
        };

        // Confidence calibration bands
        let bands = compute_confidence_bands(&resolved);

        Ok::<_, StatusCode>(Json(serde_json::json!({
            "predictions": records,
            "stats": {
                "total_predictions": records.len(),
                "total_resolved": total_resolved,
                "total_correct": correct_count,
                "overall_accuracy": if total_resolved > 0 { correct_count as f64 / total_resolved as f64 * 100.0 } else { 0.0 },
                "last_24h": stats_24h,
                "last_7d": stats_7d,
                "last_30d": stats_30d,
            },
            "per_asset": per_asset,
            "confidence_bands": bands,
        })))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(result)
}

fn compute_accuracy_stats(records: &[db::PredictionRecord], from: chrono::DateTime<Utc>, to: chrono::DateTime<Utc>) -> serde_json::Value {
    let in_range: Vec<_> = records.iter().filter(|r| {
        if let Ok(t) = chrono::DateTime::parse_from_rfc3339(&r.timestamp) {
            let t_utc = t.with_timezone(&Utc);
            t_utc >= from && t_utc <= to
        } else { false }
    }).collect();

    let resolved: Vec<_> = in_range.iter().filter(|r| r.was_correct.is_some()).collect();
    let correct = resolved.iter().filter(|r| r.was_correct == Some(true)).count();
    let total = resolved.len();

    serde_json::json!({
        "predictions": in_range.len(),
        "resolved": total,
        "correct": correct,
        "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
    })
}

fn compute_confidence_bands(resolved: &[&db::PredictionRecord]) -> Vec<serde_json::Value> {
    let bands = [(0.0, 10.0, "0-10%"), (10.0, 20.0, "10-20%"), (20.0, 30.0, "20-30%"), (30.0, 50.0, "30-50%"), (50.0, 100.0, "50%+")];
    bands.iter().map(|(lo, hi, label)| {
        let in_band: Vec<_> = resolved.iter().filter(|r| r.confidence >= *lo && r.confidence < *hi).collect();
        let correct = in_band.iter().filter(|r| r.was_correct == Some(true)).count();
        let total = in_band.len();
        serde_json::json!({
            "band": label,
            "predictions": total,
            "correct": correct,
            "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
        })
    }).collect()
}

/// GET /api/v1/signals/truth — signal truth / track record data
/// Source: PostgreSQL alpha_signal.signals (single source of truth)
async fn get_signal_truth(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit: i64 = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(5000);
    let resolved_only = params.get("resolved").map(|v| v == "true").unwrap_or(false);

    let records = if resolved_only {
        pg::get_resolved_signals(&state.pg_pool, limit).await
    } else {
        pg::get_all_signals(&state.pg_pool, limit).await
    }.map_err(|e| { eprintln!("  [signal_truth] Postgres error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    // ── Overall stats ──
    let total = records.len();
    let resolved: Vec<_> = records.iter().filter(|r| r.was_correct.is_some()).collect();
    let pending_count = records.iter().filter(|r| r.was_correct.is_none()).count();
    let correct_count = resolved.iter().filter(|r| r.was_correct == Some(true)).count();
    let total_resolved = resolved.len();
    let overall_accuracy = if total_resolved > 0 { correct_count as f64 / total_resolved as f64 * 100.0 } else { 0.0 };

    // ── Accuracy by signal type ──
    let mut by_signal: HashMap<String, (usize, usize)> = HashMap::new();
    for r in &resolved {
        let entry = by_signal.entry(r.signal_type.clone()).or_insert((0, 0));
        entry.1 += 1;
        if r.was_correct == Some(true) { entry.0 += 1; }
    }
    let signal_type_accuracy: Vec<serde_json::Value> = ["BUY", "SHORT", "SELL", "HOLD"].iter().map(|&st| {
        let (correct, total) = by_signal.get(st).copied().unwrap_or((0, 0));
        serde_json::json!({
            "signal_type": st,
            "correct": correct,
            "total": total,
            "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
        })
    }).collect();

    // ── Accuracy by asset class (stocks only now) ──
    let asset_class_accuracy: Vec<serde_json::Value> = ["stock"].iter().map(|&ac| {
        let (correct, total) = (correct_count, total_resolved);
        serde_json::json!({
            "asset_class": ac,
            "correct": correct,
            "total": total,
            "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
        })
    }).collect();

    // ── Rolling accuracy (today, this week, all time) ──
    let now = Utc::now();
    let today_start = now.format("%Y-%m-%d").to_string();
    let week_ago = (now - chrono::Duration::days(7)).format("%Y-%m-%d").to_string();

    let today_resolved: Vec<_> = resolved.iter().filter(|r| r.timestamp.starts_with(&today_start)).collect();
    let today_correct = today_resolved.iter().filter(|r| r.was_correct == Some(true)).count();

    let week_resolved: Vec<_> = resolved.iter().filter(|r| &r.timestamp[..10] >= week_ago.as_str()).collect();
    let week_correct = week_resolved.iter().filter(|r| r.was_correct == Some(true)).count();

    let rolling = serde_json::json!({
        "today": {
            "resolved": today_resolved.len(),
            "correct": today_correct,
            "accuracy": if !today_resolved.is_empty() { today_correct as f64 / today_resolved.len() as f64 * 100.0 } else { 0.0 },
        },
        "this_week": {
            "resolved": week_resolved.len(),
            "correct": week_correct,
            "accuracy": if !week_resolved.is_empty() { week_correct as f64 / week_resolved.len() as f64 * 100.0 } else { 0.0 },
        },
        "all_time": {
            "resolved": total_resolved,
            "correct": correct_count,
            "accuracy": overall_accuracy,
        },
    });

    // ── Per-asset accuracy ──
    let mut per_asset: HashMap<String, (usize, usize)> = HashMap::new();
    for r in &resolved {
        let entry = per_asset.entry(r.asset.clone()).or_insert((0, 0));
        entry.1 += 1;
        if r.was_correct == Some(true) { entry.0 += 1; }
    }
    let mut per_asset_vec: Vec<serde_json::Value> = per_asset.iter().map(|(asset, (correct, total))| {
        serde_json::json!({
            "asset": asset,
            "correct": correct,
            "total": total,
            "accuracy": if *total > 0 { *correct as f64 / *total as f64 * 100.0 } else { 0.0 },
        })
    }).collect();
    per_asset_vec.sort_by(|a, b| a["asset"].as_str().cmp(&b["asset"].as_str()));

    // ── Full signal history ──
    let signals: Vec<serde_json::Value> = records.iter().map(|r| {
        serde_json::json!({
            "id": r.id,
            "timestamp": r.timestamp,
            "asset": r.asset,
            "asset_class": r.asset_class,
            "signal_type": r.signal_type,
            "price_at_signal": r.price_at_signal,
            "confidence": r.confidence,
            "linreg_prob": r.linreg_prob,
            "logreg_prob": r.logreg_prob,
            "gbt_prob": r.gbt_prob,
            "outcome_price": r.outcome_price,
            "pct_change": r.pct_change,
            "was_correct": r.was_correct,
            "resolution_ts": r.resolution_ts,
        })
    }).collect();

    // ── Actionable metrics (BUY + SELL + SHORT, excluding HOLD) ──
    let actionable: Vec<_> = resolved.iter()
        .filter(|r| r.signal_type != "HOLD")
        .collect();
    let actionable_correct = actionable.iter().filter(|r| r.was_correct == Some(true)).count();
    let actionable_count = actionable.len();
    let actionable_accuracy = if actionable_count > 0 { actionable_correct as f64 / actionable_count as f64 * 100.0 } else { 0.0 };

    let buy_sigs: Vec<_> = resolved.iter().filter(|r| r.signal_type == "BUY").collect();
    let buy_correct = buy_sigs.iter().filter(|r| r.was_correct == Some(true)).count();
    let buy_accuracy = if !buy_sigs.is_empty() { buy_correct as f64 / buy_sigs.len() as f64 * 100.0 } else { 0.0 };

    let sell_sigs: Vec<_> = resolved.iter().filter(|r| r.signal_type == "SELL" || r.signal_type == "SHORT").collect();
    let sell_correct = sell_sigs.iter().filter(|r| r.was_correct == Some(true)).count();
    let sell_accuracy = if !sell_sigs.is_empty() { sell_correct as f64 / sell_sigs.len() as f64 * 100.0 } else { 0.0 };

    let hold_sigs: Vec<_> = resolved.iter().filter(|r| r.signal_type == "HOLD").collect();
    let hold_correct = hold_sigs.iter().filter(|r| r.was_correct == Some(true)).count();
    let hold_accuracy = if !hold_sigs.is_empty() { hold_correct as f64 / hold_sigs.len() as f64 * 100.0 } else { 0.0 };

    // Expected value per actionable signal (in basis points)
    let returns: Vec<f64> = actionable.iter().filter_map(|r| {
        r.pct_change.map(|pct| match r.signal_type.as_str() {
            "BUY" => pct,
            "SELL" | "SHORT" => -pct,
            _ => 0.0,
        })
    }).collect();
    let expected_value_bps = if !returns.is_empty() {
        returns.iter().sum::<f64>() / returns.len() as f64 * 100.0
    } else { 0.0 };

    // Profit factor
    let winners: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losers: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let profit_factor = if losers > 0.0 { winners / losers } else if winners > 0.0 { 99.99 } else { 0.0 };

    let mut response = serde_json::json!({
        "total_signals": total,
        "total_resolved": total_resolved,
        "total_pending": pending_count,
        "total_correct": correct_count,
        "overall_accuracy": overall_accuracy,
        "actionable_accuracy": actionable_accuracy,
        "actionable_signals": actionable_count,
        "actionable_correct": actionable_correct,
        "buy_accuracy": buy_accuracy,
        "buy_signals": buy_sigs.len(),
        "buy_correct": buy_correct,
        "sell_accuracy": sell_accuracy,
        "sell_signals": sell_sigs.len(),
        "sell_correct": sell_correct,
        "hold_accuracy": hold_accuracy,
        "hold_signals": hold_sigs.len(),
        "hold_correct": hold_correct,
        "expected_value_bps": expected_value_bps,
        "profit_factor": profit_factor,
        "by_signal_type": signal_type_accuracy,
        "by_asset_class": asset_class_accuracy,
        "rolling": rolling,
        "per_asset": per_asset_vec,
        "signals": signals,
    });

    // Attach canonical Sharpe + max drawdown from daily_portfolio (Postgres, single source of truth)
    if let Ok(portfolio) = pg::get_daily_portfolio(&state.pg_pool).await {
        if portfolio.len() >= 2 {
            let rets: Vec<f64> = portfolio.iter().map(|(_, _, _, dr, _)| *dr / 100.0).collect();
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
            let std = var.sqrt();
            let rf = 0.045 / 252.0;
            let sharpe = if std > 1e-10 { (mean - rf) / std * 252.0_f64.sqrt() } else { 0.0 };
            let values: Vec<f64> = portfolio.iter().map(|(_, _, v, _, _)| *v).collect();
            let mut peak = 0.0_f64;
            let mut max_dd = 0.0_f64;
            for v in &values { if *v > peak { peak = *v; } let dd = if peak > 0.0 { (peak - v) / peak * 100.0 } else { 0.0 }; if dd > max_dd { max_dd = dd; } }
            response["sharpe"] = serde_json::json!((sharpe * 100.0).round() / 100.0);
            response["max_drawdown"] = serde_json::json!((max_dd * 10.0).round() / 10.0);
        }
    }

    Ok(Json(response))
}

/// GET /api/v1/signals/truth/historical — backtest signal accuracy over portfolio holdings
async fn get_signal_truth_historical(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let frequency = params.get("frequency").cloned().unwrap_or_else(|| "weekly".to_string());

    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let holdings = database.get_user_holdings().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        if holdings.is_empty() {
            return Ok::<_, StatusCode>(Json(serde_json::json!({
                "has_data": false,
                "note": "No holdings in portfolio. Add holdings to see historical signal accuracy."
            })));
        }

        let result = backtest_signal_accuracy(&database, &holdings, &frequency);
        Ok(Json(result))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(result)
}

/// Backtest signal accuracy across all holdings from their start dates to today
fn backtest_signal_accuracy(
    database: &db::Database,
    holdings: &[db::UserHolding],
    frequency: &str,
) -> serde_json::Value {
    let market_context = simulator::build_market_context_from_db(database);
    let today = chrono::Utc::now().date_naive().format("%Y-%m-%d").to_string();

    // Per-signal record for aggregation
    struct SignalRecord {
        date: String,
        asset: String,
        asset_class: String,
        signal_type: String,
        price_at_signal: f64,
        outcome_price: Option<f64>,
        was_correct: Option<bool>, // None = pending (no outcome yet)
    }

    let mut all_records: Vec<SignalRecord> = Vec::new();

    for holding in holdings {
        let all_prices = simulator::load_asset_prices(database, &holding.symbol, &holding.asset_class);
        if all_prices.is_empty() { continue; }

        // Find actual start date
        let holding_start = &holding.start_date[..10.min(holding.start_date.len())];
        let start_idx = match all_prices.iter().position(|(d, _)| d.as_str() >= holding_start) {
            Some(i) => i,
            None => continue,
        };
        let actual_start_date = &all_prices[start_idx].0;

        // Build trading dates from start to end of available data (up to today)
        let trading_dates: Vec<String> = all_prices.iter()
            .filter(|(d, _)| d.as_str() >= actual_start_date.as_str() && d.as_str() <= today.as_str())
            .map(|(d, _)| d.clone())
            .collect();

        if trading_dates.is_empty() { continue; }

        // Determine signal dates
        let signal_dates: std::collections::HashSet<String> = if frequency == "weekly" {
            weekly_signal_dates(&trading_dates)
        } else {
            trading_dates.iter().cloned().collect()
        };

        // Load models and generate signals in bulk
        let models = simulator::load_models_for_symbol(&holding.symbol);
        if models.is_none() { continue; }
        let signal_map = simulator::generate_signals_bulk(
            &holding.symbol, &holding.asset_class,
            &all_prices, &market_context, models.as_ref().unwrap(),
        );

        // Build price map for lookup
        let price_map: std::collections::HashMap<&str, f64> = all_prices.iter()
            .map(|(d, p)| (d.as_str(), *p))
            .collect();

        // Collect signal dates in order
        let mut ordered_signal_dates: Vec<&String> = trading_dates.iter()
            .filter(|d| signal_dates.contains(*d))
            .collect();
        ordered_signal_dates.sort();

        // For each signal date, look ahead to the next signal date for outcome
        for (i, &sig_date) in ordered_signal_dates.iter().enumerate() {
            let signal = match signal_map.get(sig_date.as_str()) {
                Some(s) => s.clone(),
                None => continue, // generate_signal_cached returned None (< 100 points)
            };

            let price_at_signal = match price_map.get(sig_date.as_str()) {
                Some(&p) if p > 0.0 => p,
                _ => continue,
            };

            // Look ahead: next signal date's price is the outcome
            let (outcome_price, was_correct) = if i + 1 < ordered_signal_dates.len() {
                let next_date = ordered_signal_dates[i + 1];
                match price_map.get(next_date.as_str()) {
                    Some(&next_p) if next_p > 0.0 => {
                        let price_up = next_p > price_at_signal;
                        let price_down = next_p < price_at_signal;
                        let correct = match signal.as_str() {
                            "BUY" => price_up,
                            "SELL" | "SHORT" => price_down,
                            _ => false, // HOLD: tracked but considered incorrect for accuracy
                        };
                        (Some(next_p), Some(correct))
                    }
                    _ => (None, None),
                }
            } else {
                // Last signal period — no outcome yet, mark as pending
                (None, None)
            };

            all_records.push(SignalRecord {
                date: sig_date.clone(),
                asset: holding.symbol.clone(),
                asset_class: holding.asset_class.clone(),
                signal_type: signal,
                price_at_signal,
                outcome_price,
                was_correct,
            });
        }
    }

    // ── Aggregation ──

    // Exclude HOLD from main accuracy calculation
    let buy_sell_resolved: Vec<&SignalRecord> = all_records.iter()
        .filter(|r| r.was_correct.is_some() && r.signal_type != "HOLD")
        .collect();
    let total_buy_sell = buy_sell_resolved.len();
    let correct_buy_sell = buy_sell_resolved.iter().filter(|r| r.was_correct == Some(true)).count();
    let overall_accuracy = if total_buy_sell > 0 {
        correct_buy_sell as f64 / total_buy_sell as f64 * 100.0
    } else { 0.0 };

    let total_signals = all_records.len();
    let total_pending = all_records.iter().filter(|r| r.was_correct.is_none()).count();
    let total_resolved = all_records.iter().filter(|r| r.was_correct.is_some()).count();

    // ── By signal type ──
    let mut by_signal: HashMap<String, (usize, usize)> = HashMap::new(); // (correct, total_resolved)
    for r in all_records.iter().filter(|r| r.was_correct.is_some()) {
        let entry = by_signal.entry(r.signal_type.clone()).or_insert((0, 0));
        entry.1 += 1;
        if r.was_correct == Some(true) { entry.0 += 1; }
    }
    let signal_type_accuracy: Vec<serde_json::Value> = ["BUY", "SHORT", "SELL", "HOLD"].iter().map(|&st| {
        let (correct, total) = by_signal.get(st).copied().unwrap_or((0, 0));
        let total_incl_pending = all_records.iter().filter(|r| r.signal_type == st).count();
        serde_json::json!({
            "signal_type": st,
            "correct": correct,
            "total": total,
            "total_including_pending": total_incl_pending,
            "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
        })
    }).collect();

    // ── By asset class ──
    let mut by_class: HashMap<String, (usize, usize)> = HashMap::new();
    for r in buy_sell_resolved.iter() {
        let entry = by_class.entry(r.asset_class.clone()).or_insert((0, 0));
        entry.1 += 1;
        if r.was_correct == Some(true) { entry.0 += 1; }
    }
    let asset_class_accuracy: Vec<serde_json::Value> = ["stock", "fx", "crypto"].iter().map(|&ac| {
        let (correct, total) = by_class.get(ac).copied().unwrap_or((0, 0));
        serde_json::json!({
            "asset_class": ac,
            "correct": correct,
            "total": total,
            "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
        })
    }).collect();

    // ── Per-asset ──
    let mut per_asset_map: HashMap<String, (String, usize, usize, String, String)> = HashMap::new();
    // (asset_class, correct, total_resolved_buy_sell, earliest_date, latest_date)
    for r in &all_records {
        let entry = per_asset_map.entry(r.asset.clone()).or_insert_with(|| {
            (r.asset_class.clone(), 0, 0, r.date.clone(), r.date.clone())
        });
        if r.signal_type != "HOLD" && r.was_correct.is_some() {
            entry.2 += 1; // total
            if r.was_correct == Some(true) { entry.1 += 1; } // correct
        }
        if r.date < entry.3 { entry.3 = r.date.clone(); }
        if r.date > entry.4 { entry.4 = r.date.clone(); }
    }
    let mut per_asset_vec: Vec<serde_json::Value> = per_asset_map.iter().map(|(asset, (ac, correct, total, from, to))| {
        let total_signals_for_asset = all_records.iter().filter(|r| r.asset == *asset).count();
        serde_json::json!({
            "asset": asset,
            "asset_class": ac,
            "correct": correct,
            "total": total,
            "total_signals": total_signals_for_asset,
            "accuracy": if *total > 0 { *correct as f64 / *total as f64 * 100.0 } else { 0.0 },
            "date_from": from,
            "date_to": to,
        })
    }).collect();
    per_asset_vec.sort_by(|a, b| a["asset"].as_str().cmp(&b["asset"].as_str()));

    // ── Monthly breakdown ──
    let mut monthly: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new();
    for r in buy_sell_resolved.iter() {
        let month_key = if r.date.len() >= 7 { &r.date[..7] } else { &r.date };
        let entry = monthly.entry(month_key.to_string()).or_insert((0, 0));
        entry.1 += 1;
        if r.was_correct == Some(true) { entry.0 += 1; }
    }
    let monthly_accuracy: Vec<serde_json::Value> = monthly.iter().map(|(month, (correct, total))| {
        serde_json::json!({
            "month": month,
            "correct": correct,
            "total": total,
            "accuracy": if *total > 0 { *correct as f64 / *total as f64 * 100.0 } else { 0.0 },
        })
    }).collect();

    let timestamp = chrono::Utc::now().to_rfc3339();

    serde_json::json!({
        "has_data": true,
        "frequency": frequency,
        "total_signals": total_signals,
        "total_resolved": total_resolved,
        "total_pending": total_pending,
        "total_correct": correct_buy_sell,
        "overall_accuracy": round2(overall_accuracy),
        "by_signal_type": signal_type_accuracy,
        "by_asset_class": asset_class_accuracy,
        "per_asset": per_asset_vec,
        "monthly_accuracy": monthly_accuracy,
        "generated_at": timestamp,
    })
}

/// POST /api/v1/signals/resolve — Force-resolve all pending signal_history + predictions
async fn force_resolve_signals(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let sigs = state.signals.read().await;
    let signals_map = sigs.clone();
    let signals_vec: Vec<enriched_signals::EnrichedSignal> = sigs.values().cloned().collect();
    drop(sigs);
    let db_path = state.db_path.clone();

    // Resolve SQLite signals
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let before_unresolved = database.get_all_unresolved_signals().unwrap_or_default().len();
        let before_pending = database.get_pending_predictions().unwrap_or_default().len();

        batch_resolve_signal_history(&db_path, &signals_map);
        resolve_pending_predictions(&db_path, &signals_map);

        let after_unresolved = database.get_all_unresolved_signals().unwrap_or_default().len();
        let after_pending = database.get_pending_predictions().unwrap_or_default().len();

        Ok::<_, StatusCode>((
            before_unresolved.saturating_sub(after_unresolved),
            before_pending.saturating_sub(after_pending),
            after_unresolved,
            after_pending,
        ))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    // Resolve Postgres signals
    let pg_before = pg::get_all_unresolved_signals(&state.pg_pool).await
        .map(|u| u.len()).unwrap_or(0);
    batch_resolve_signals_pg(&state.pg_pool, &signals_vec).await;
    let pg_after = pg::get_all_unresolved_signals(&state.pg_pool).await
        .map(|u| u.len()).unwrap_or(0);

    Ok(Json(serde_json::json!({
        "signals_resolved": result.0,
        "predictions_resolved": result.1,
        "signals_still_pending": result.2,
        "predictions_still_pending": result.3,
        "pg_signals_resolved": pg_before.saturating_sub(pg_after),
        "pg_signals_still_pending": pg_after,
    })))
}

async fn get_all_signals(
    State(state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    // Descoped: only return stocks (ASSET_UNIVERSE = ["stock"])
    let sigs = state.signals.read().await;
    let mut signals: Vec<_> = sigs.values()
        .filter(|s| pg::ASSET_UNIVERSE.contains(&s.asset_class.as_str()))
        .cloned()
        .collect();
    signals.sort_by(|a, b| a.asset.cmp(&b.asset));
    Json(signals)
}

async fn get_stock_signals(
    State(state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    let sigs = state.signals.read().await;
    let mut signals: Vec<_> = sigs.values()
        .filter(|s| s.asset_class == "stock")
        .cloned()
        .collect();
    signals.sort_by(|a, b| a.asset.cmp(&b.asset));
    Json(signals)
}

/// FX signals — descoped (returns empty array)
async fn get_fx_signals(
    State(_state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    Json(vec![])
}

/// Crypto signals — descoped (returns empty array)
async fn get_crypto_signals(
    State(_state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    Json(vec![])
}

async fn get_signal_history(
    State(state): State<AppState>,
    Path(asset): Path<String>,
) -> Result<Json<Vec<db::SignalSnapshotRow>>, StatusCode> {
    let database = db::Database::new(&state.db_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let history = database.get_signal_history(&asset, 100)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(history))
}

#[derive(serde::Deserialize)]
struct PortfolioParams {
    start_date: Option<String>,
    capital: Option<f64>,
    strategy: Option<String>,
}

async fn simulate_portfolio(
    State(state): State<AppState>,
    Query(_params): Query<PortfolioParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let database = db::Database::new(&state.db_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let model_version = model_store::MODEL_VERSION;

    let has_data = database.has_backtest_data(model_version)
        .unwrap_or(false);

    if !has_data {
        return Ok(Json(serde_json::json!({
            "starting_capital": 100000,
            "has_data": false,
            "note": "Portfolio simulation requires a completed training run. Backtest data will appear after the next training cycle."
        })));
    }

    // Build strategies map
    let portfolio_rows = database.get_portfolio_results(model_version)
        .unwrap_or_default();
    let backtest_rows = database.get_backtest_results(model_version)
        .unwrap_or_default();

    // Get current signals for each asset
    let sigs = state.signals.read().await;

    let mut strategies = serde_json::Map::new();
    for pr in &portfolio_rows {
        let allocs = database.get_portfolio_allocations(model_version, &pr.strategy)
            .unwrap_or_default();
        let alloc_json: Vec<serde_json::Value> = allocs.iter().map(|a| {
            let signal = sigs.get(&a.asset).map(|s| s.signal.as_str()).unwrap_or("N/A");
            let asset_class = backtest_rows.iter()
                .find(|b| b.asset == a.asset)
                .map(|b| b.asset_class.as_str())
                .unwrap_or("unknown");
            serde_json::json!({
                "asset": a.asset,
                "asset_class": asset_class,
                "weight": (a.weight * 1000.0).round() / 10.0,
                "allocated": (a.allocated_amount * 100.0).round() / 100.0,
                "return": (a.asset_return * 100.0).round() / 100.0,
                "contribution": (a.contribution * 100.0).round() / 100.0,
                "sharpe": (a.sharpe * 100.0).round() / 100.0,
                "signal": signal,
            })
        }).collect();

        strategies.insert(pr.strategy.clone(), serde_json::json!({
            "final_value": (pr.final_value * 100.0).round() / 100.0,
            "total_return": (pr.total_return * 100.0).round() / 100.0,
            "annualised_return": (pr.annualised_return * 100.0).round() / 100.0,
            "benchmark_return": (pr.benchmark_return * 100.0).round() / 100.0,
            "excess_return": (pr.excess_return * 100.0).round() / 100.0,
            "sharpe_ratio": (pr.sharpe_ratio * 100.0).round() / 100.0,
            "max_drawdown": (pr.max_drawdown * 100.0).round() / 100.0,
            "volatility": (pr.volatility * 100.0).round() / 100.0,
            "n_assets": pr.n_assets,
            "allocations": alloc_json,
        }));
    }

    // Build per-asset backtest array
    let per_asset: Vec<serde_json::Value> = backtest_rows.iter().map(|b| {
        let verdict = if b.sharpe_ratio > 1.0 && b.excess_return > 0.0 {
            "EDGE"
        } else if b.sharpe_ratio > 0.5 {
            "MARGINAL"
        } else {
            "NO EDGE"
        };
        serde_json::json!({
            "asset": b.asset,
            "asset_class": b.asset_class,
            "total_return": (b.total_return * 100.0).round() / 100.0,
            "buy_hold_return": (b.buy_hold_return * 100.0).round() / 100.0,
            "excess_return": (b.excess_return * 100.0).round() / 100.0,
            "annualised_return": (b.annualised_return * 100.0).round() / 100.0,
            "sharpe_ratio": (b.sharpe_ratio * 100.0).round() / 100.0,
            "max_drawdown": (b.max_drawdown * 100.0).round() / 100.0,
            "win_rate": (b.win_rate * 100.0).round() / 100.0,
            "profit_factor": (b.profit_factor * 100.0).round() / 100.0,
            "expectancy": (b.expectancy * 1000.0).round() / 1000.0,
            "days_in_market": b.days_in_market,
            "total_days": b.total_days,
            "verdict": verdict,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "starting_capital": 100000,
        "has_data": true,
        "strategies": strategies,
        "per_asset_backtest": per_asset,
    })))
}

async fn get_daily_tracker(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        daily_tracker::build_api_response(&db_path)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(result))
}

/// POST /api/v1/portfolio/rebalance — compute optimal sector-weighted allocation
async fn compute_rebalance(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let signals = state.signals.read().await.clone();
    let regime = state.regime.read().await.clone();
    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        daily_tracker::compute_rebalance(&signals, regime.as_ref(), &db_path)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(result))
}

/// GET /api/v1/history/portfolio — portfolio equity curve
/// Source: PostgreSQL alpha_signal.daily_portfolio (single source of truth)
async fn get_portfolio_history(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let rows = pg::get_daily_portfolio(&state.pg_pool).await
        .map_err(|e| { eprintln!("  [portfolio_history] Postgres error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    if rows.is_empty() {
        return Ok(Json(serde_json::json!({
            "has_data": false,
            "note": "No portfolio history yet. Run rebuild_portfolio to populate."
        })));
    }

    // rows are already in chronological order from Postgres
    let (seed_date, seed_value, _, _, _) = &rows[0];
    let (_, _, latest_value, _, latest_cum) = &rows[rows.len() - 1];
    let _ = seed_date;

    let points: Vec<serde_json::Value> = rows.iter().map(|(date, _seed, value, daily_ret, cum_ret)| {
        serde_json::json!({
            "date": date,
            "value": value,
            "daily_return": daily_ret,
            "cumulative_return": cum_ret,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "has_data": true,
        "seed_value": seed_value,
        "current_value": latest_value,
        "cumulative_return": latest_cum,
        "days": rows.len(),
        "points": points,
    })))
}

/// GET /api/v1/portfolio/live — live portfolio state from Postgres
/// Returns current portfolio value, equity curve, signal stats, and SPY benchmark.
async fn get_portfolio_live(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Portfolio equity curve from Postgres
    let portfolio = pg::get_daily_portfolio(&state.pg_pool).await
        .map_err(|e| { eprintln!("  [portfolio/live] Postgres error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    // Signal stats from Postgres
    let stats = pg::get_signal_stats(&state.pg_pool).await
        .map_err(|e| { eprintln!("  [portfolio/live] Stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    if portfolio.is_empty() {
        return Ok(Json(serde_json::json!({
            "has_data": false,
            "stats": stats,
        })));
    }

    let (_, seed_value, _, _, _) = &portfolio[0];
    let (latest_date, _, latest_value, _, latest_cum) = &portfolio[portfolio.len() - 1];
    let seed = *seed_value;

    // Build portfolio date→value map for SPY alignment
    let portfolio_dates: Vec<String> = portfolio.iter().map(|(d, _, _, _, _)| d.clone()).collect();
    let first_date = portfolio_dates[0].clone();

    let points: Vec<serde_json::Value> = portfolio.iter().map(|(date, _seed, value, daily_ret, cum_ret)| {
        serde_json::json!({
            "date": date,
            "value": value,
            "daily_return": daily_ret,
            "cumulative_return": cum_ret,
        })
    }).collect();

    // Fetch SPY benchmark from SQLite for the same period
    let db_path = state.db_path.clone();
    let first_date_clone = first_date.clone();
    let portfolio_dates_clone = portfolio_dates.clone();
    let spy_benchmark: Vec<serde_json::Value> = tokio::task::spawn_blocking(move || {
        let database = match db::Database::new(&db_path) {
            Ok(db) => db,
            Err(_) => return Vec::new(),
        };
        let spy_history = match database.get_stock_history("SPY") {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };
        // Build date→price map from SPY history
        let spy_map: std::collections::HashMap<String, f64> = spy_history.iter()
            .map(|p| (p.timestamp[..10].to_string(), p.price))
            .collect();
        // Find SPY price on the portfolio start date (or nearest prior)
        let spy_start_price = spy_map.get(&first_date_clone[..10])
            .copied()
            .or_else(|| {
                // Find nearest available price before start date
                let mut prices: Vec<(&str, f64)> = spy_history.iter()
                    .map(|p| (p.timestamp.as_str(), p.price))
                    .collect();
                prices.sort_by_key(|(d, _)| d.to_string());
                prices.iter()
                    .filter(|(d, _)| *d <= first_date_clone.as_str())
                    .last()
                    .map(|(_, p)| *p)
            });
        let spy_start = match spy_start_price {
            Some(p) => p,
            None => return Vec::new(),
        };
        // For each portfolio date, compute SPY value normalised to seed capital
        portfolio_dates_clone.iter().filter_map(|date| {
            let date_key = &date[..10];
            spy_map.get(date_key).map(|spy_price| {
                let spy_value = seed * (spy_price / spy_start);
                serde_json::json!({ "date": date, "value": spy_value })
            })
        }).collect()
    }).await.unwrap_or_default();

    let spy_return = if spy_benchmark.len() >= 2 {
        let last_val = spy_benchmark.last().and_then(|v| v["value"].as_f64()).unwrap_or(seed);
        (last_val / seed - 1.0) * 100.0
    } else { 0.0 };

    // Canonical Sharpe ratio and max drawdown from daily_portfolio (server-side, single source of truth)
    let daily_returns: Vec<f64> = portfolio.iter().map(|(_, _, _, dr, _)| *dr / 100.0).collect();
    let (sharpe, max_drawdown) = if daily_returns.len() >= 2 {
        let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let var = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (daily_returns.len() - 1) as f64;
        let std = var.sqrt();
        let rf_daily = 0.045 / 252.0;
        let s = if std > 1e-10 { (mean - rf_daily) / std * 252.0_f64.sqrt() } else { 0.0 };
        // Max drawdown from equity curve
        let values: Vec<f64> = portfolio.iter().map(|(_, _, v, _, _)| *v).collect();
        let mut peak = 0.0_f64;
        let mut max_dd = 0.0_f64;
        for v in &values { if *v > peak { peak = *v; } let dd = if peak > 0.0 { (peak - v) / peak * 100.0 } else { 0.0 }; if dd > max_dd { max_dd = dd; } }
        (s, max_dd)
    } else { (0.0, 0.0) };

    Ok(Json(serde_json::json!({
        "has_data": true,
        "seed_value": seed_value,
        "current_value": latest_value,
        "cumulative_return": latest_cum,
        "latest_date": latest_date,
        "days": portfolio.len(),
        "points": points,
        "spy_benchmark": spy_benchmark,
        "spy_return": spy_return,
        "sharpe": (sharpe * 100.0).round() / 100.0,
        "max_drawdown": (max_drawdown * 10.0).round() / 10.0,
        "stats": stats,
    })))
}

#[derive(serde::Deserialize)]
struct HistoryParams {
    days: Option<usize>,
}

async fn get_signals_history(
    State(state): State<AppState>,
    Query(params): Query<HistoryParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let database = db::Database::new(&state.db_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let days = params.days.unwrap_or(14);
    let rows = database.get_recent_signals_all_assets(days).unwrap_or_default();

    if rows.is_empty() {
        return Ok(Json(serde_json::json!({
            "has_data": false,
            "note": "No signal history yet."
        })));
    }

    // Group by asset, then by date (take one signal per asset per day — most recent)
    let mut by_asset: std::collections::HashMap<String, Vec<serde_json::Value>> =
        std::collections::HashMap::new();

    for row in &rows {
        let date = &row.timestamp[..10]; // YYYY-MM-DD
        let entry = serde_json::json!({
            "date": date,
            "signal": row.signal,
            "confidence": row.confidence,
            "probability_up": row.probability_up,
            "price": row.price,
            "rsi": row.rsi,
            "asset_class": row.asset_class,
        });
        by_asset.entry(row.asset.clone()).or_default().push(entry);
    }

    // Deduplicate — keep one per asset per date (first = most recent due to ORDER BY timestamp DESC)
    let mut deduped: std::collections::HashMap<String, Vec<serde_json::Value>> =
        std::collections::HashMap::new();
    for (asset, entries) in &by_asset {
        let mut seen_dates = std::collections::HashSet::new();
        let unique: Vec<serde_json::Value> = entries.iter().filter(|e| {
            let date = e["date"].as_str().unwrap_or("").to_string();
            seen_dates.insert(date)
        }).cloned().collect();
        deduped.insert(asset.clone(), unique);
    }

    // Compute per-asset accuracy
    let accuracy: std::collections::HashMap<String, serde_json::Value> = deduped.iter()
        .map(|(asset, entries)| {
            let total = entries.len();
            // We don't have actual returns here, but we can show signal distribution
            let buys = entries.iter().filter(|e| e["signal"] == "BUY").count();
            let sells = entries.iter().filter(|e| e["signal"] == "SELL").count();
            let holds = entries.iter().filter(|e| e["signal"] == "HOLD").count();
            (asset.clone(), serde_json::json!({
                "total": total,
                "buys": buys,
                "sells": sells,
                "holds": holds,
            }))
        }).collect();

    Ok(Json(serde_json::json!({
        "has_data": true,
        "days": days,
        "signals": deduped,
        "accuracy": accuracy,
    })))
}

// ════════════════════════════════════════
// Hints Handler
// ════════════════════════════════════════

async fn get_hints(
    State(state): State<AppState>,
) -> Json<Vec<hints::Hint>> {
    let sigs = state.signals.read().await;
    let hints = hints::generate_hints(&sigs);
    Json(hints)
}

// ════════════════════════════════════════
// Simulate Handler
// ════════════════════════════════════════

#[derive(serde::Deserialize)]
struct SimulateRequest {
    days: Option<usize>,
    capital: Option<f64>,
}

async fn simulate_signals(
    State(state): State<AppState>,
    Json(req): Json<SimulateRequest>,
) -> Result<Json<simulator::SimResult>, StatusCode> {
    let days = req.days.unwrap_or(14);
    let capital = req.capital.unwrap_or(10_000.0);
    let db_path = state.db_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        simulator::run_simulation(days, capital, &db_path)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    match result {
        Ok(sim) => Ok(Json(sim)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// ════════════════════════════════════════
// Chat Handler
// ════════════════════════════════════════

#[derive(serde::Deserialize)]
struct ChatRequest {
    message: String,
    tab_context: Option<String>,
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let tab_context = req.tab_context.unwrap_or_else(|| "overview".to_string());
    let is_morning_briefing = req.message.trim() == "morning_briefing";

    // Build portfolio context from daily tracker
    let portfolio_context = {
        let db_path = state.db_path.clone();
        tokio::task::spawn_blocking(move || {
            daily_tracker::build_api_response(&db_path)
        }).await.unwrap_or_else(|_| serde_json::json!({"has_data": false}))
    };

    // Collect signals filtered by tab
    let sigs = state.signals.read().await;
    let mut relevant: Vec<_> = match tab_context.as_str() {
        "stocks" => sigs.values().filter(|s| s.asset_class == "stock").cloned().collect(),
        "fx" => sigs.values().filter(|s| s.asset_class == "fx").cloned().collect(),
        "crypto" => sigs.values().filter(|s| s.asset_class == "crypto").cloned().collect(),
        _ => sigs.values().cloned().collect(),
    };
    drop(sigs);

    let response = if is_morning_briefing {
        generate_morning_briefing(&relevant, &portfolio_context)
    } else {
        generate_chat_response(&req.message, &relevant, &portfolio_context)
    };

    Ok(Json(serde_json::json!({ "response": response })))
}

/// Rule-based morning briefing generator — no LLM required.
fn generate_morning_briefing(
    signals: &[enriched_signals::EnrichedSignal],
    portfolio_context: &serde_json::Value,
) -> String {
    let total = signals.len();
    let buys: Vec<_> = signals.iter().filter(|s| s.signal == "BUY").collect();
    let sells: Vec<_> = signals.iter().filter(|s| s.signal == "SELL" || s.signal == "SHORT").collect();
    let holds: Vec<_> = signals.iter().filter(|s| s.signal == "HOLD").collect();

    // Determine mood
    let buy_pct = if total > 0 { buys.len() as f64 / total as f64 * 100.0 } else { 0.0 };
    let sell_pct = if total > 0 { sells.len() as f64 / total as f64 * 100.0 } else { 0.0 };
    let mood = if buy_pct > 50.0 {
        "optimistic"
    } else if sell_pct > 40.0 {
        "cautious"
    } else {
        "mixed"
    };

    // Portfolio summary
    let portfolio_line = if portfolio_context["has_data"].as_bool().unwrap_or(false) {
        let value = portfolio_context["current_value"].as_f64().unwrap_or(0.0);
        let daily_ret = portfolio_context["daily_return"].as_f64().unwrap_or(0.0);
        let cum_ret = portfolio_context["cumulative_return"].as_f64().unwrap_or(0.0);
        format!(
            "The portfolio stands at {:.0} ({:+.2}% today, {:+.2}% cumulative).",
            value, daily_ret, cum_ret
        )
    } else {
        String::new()
    };

    // Para 1: Market mood
    let para1 = format!(
        "Alpha Signal is tracking {} assets today with {} BUY, {} SELL/SHORT, and {} HOLD signals — the overall mood is {}. {}",
        total, buys.len(), sells.len(), holds.len(), mood, portfolio_line
    );

    // Para 2: Strongest signals (top 3 by confidence)
    let mut strong: Vec<_> = signals.iter()
        .filter(|s| s.signal != "HOLD")
        .collect();
    strong.sort_by(|a, b| b.technical.confidence.partial_cmp(&a.technical.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let top_signals: Vec<String> = strong.iter().take(3).map(|s| {
        format!("{} {} (confidence {:.1}/10, prob {:.0}%)",
            s.signal, s.asset, s.technical.confidence, s.technical.probability_up)
    }).collect();
    let para2 = if top_signals.is_empty() {
        "No strong actionable signals today.".to_string()
    } else {
        format!("Strongest signals: {}.", top_signals.join("; "))
    };

    // Para 3: Risk watch — highest-confidence SELL or oversold RSI
    let risk_signal = signals.iter()
        .filter(|s| s.signal == "SELL" || s.signal == "SHORT" || s.technical.rsi > 70.0 || s.technical.rsi < 30.0)
        .max_by(|a, b| a.technical.confidence.partial_cmp(&b.technical.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let para3 = match risk_signal {
        Some(s) => {
            let reason = if s.technical.rsi > 70.0 { "overbought RSI" }
                else if s.technical.rsi < 30.0 { "oversold RSI" }
                else { "bearish signal" };
            format!("Watch out: {} shows {} (RSI {:.0}, signal {}).", s.asset, reason, s.technical.rsi, s.signal)
        }
        None => "No major risk flags in today's signals.".to_string(),
    };

    format!("{}\n\n{}\n\n{}", para1, para2, para3)
}

/// Rule-based chat response — answers common queries from signal/portfolio data.
fn generate_chat_response(
    question: &str,
    signals: &[enriched_signals::EnrichedSignal],
    portfolio_context: &serde_json::Value,
) -> String {
    let q = question.to_lowercase();

    // Portfolio status questions
    if q.contains("portfolio") || q.contains("value") || q.contains("return") || q.contains("performance") {
        if portfolio_context["has_data"].as_bool().unwrap_or(false) {
            let value = portfolio_context["current_value"].as_f64().unwrap_or(0.0);
            let daily_ret = portfolio_context["daily_return"].as_f64().unwrap_or(0.0);
            let cum_ret = portfolio_context["cumulative_return"].as_f64().unwrap_or(0.0);
            let accuracy = portfolio_context["model_accuracy_pct"].as_f64().unwrap_or(0.0);
            return format!(
                "Portfolio value: {:.2}, daily return: {:+.2}%, cumulative return: {:+.2}%, model accuracy: {:.1}%.\n\
                 Note: past performance does not guarantee future results.",
                value, daily_ret, cum_ret, accuracy
            );
        } else {
            return "Portfolio tracking has not started yet — no historical data available.".to_string();
        }
    }

    // Signal-specific questions
    if q.contains("buy") || q.contains("sell") || q.contains("short") || q.contains("signal") {
        let buys: Vec<_> = signals.iter().filter(|s| s.signal == "BUY").collect();
        let sells: Vec<_> = signals.iter().filter(|s| s.signal == "SELL" || s.signal == "SHORT").collect();
        let mut lines = vec![format!("Currently {} BUY and {} SELL/SHORT signals active.", buys.len(), sells.len())];
        let mut top: Vec<_> = signals.iter().filter(|s| s.signal != "HOLD").collect();
        top.sort_by(|a, b| b.technical.confidence.partial_cmp(&a.technical.confidence).unwrap_or(std::cmp::Ordering::Equal));
        for s in top.iter().take(5) {
            lines.push(format!("  {} {} — confidence {:.1}/10, prob {:.0}%, RSI {:.0}",
                s.signal, s.asset, s.technical.confidence, s.technical.probability_up, s.technical.rsi));
        }
        lines.push("Note: past performance does not guarantee future results.".to_string());
        return lines.join("\n");
    }

    // Asset-specific questions
    for sig in signals {
        if q.contains(&sig.asset.to_lowercase()) {
            return format!(
                "{}: signal={}, confidence={:.1}/10, probability_up={:.1}%, RSI={:.0}, price={:.2}, quality={}, accuracy={:.1}%.\n\
                 Note: past performance does not guarantee future results.",
                sig.asset, sig.signal, sig.technical.confidence,
                sig.technical.probability_up, sig.technical.rsi,
                sig.price, sig.technical.quality, sig.technical.walk_forward_accuracy
            );
        }
    }

    // Default
    format!(
        "I can answer questions about your portfolio, current signals, and specific assets. \
         Try asking about portfolio performance, today's BUY/SELL signals, or a specific asset like AAPL.\n\
         Currently tracking {} signals ({} BUY, {} SELL/SHORT).\n\
         Note: past performance does not guarantee future results.",
        signals.len(),
        signals.iter().filter(|s| s.signal == "BUY").count(),
        signals.iter().filter(|s| s.signal == "SELL" || s.signal == "SHORT").count(),
    )
}

// ════════════════════════════════════════
// Admin Asset Management Handlers
// ════════════════════════════════════════

#[derive(serde::Deserialize)]
struct AddAssetRequest {
    symbol: String,
    name: String,
    class: String,
    enabled: Option<bool>,
}

async fn add_asset(
    State(state): State<AppState>,
    Json(req): Json<AddAssetRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let symbol = req.symbol.trim().to_uppercase();
    let name = req.name.trim().to_string();
    let class = req.class.trim().to_lowercase();
    let enabled = req.enabled.unwrap_or(true);

    if symbol.is_empty() || name.is_empty() {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "symbol and name required"}))));
    }
    if !["stock", "fx", "crypto", "etf"].contains(&class.as_str()) {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "class must be stock, fx, crypto, or etf"}))));
    }

    let entry = config::AssetEntry {
        symbol: symbol.clone(),
        name: name.clone(),
        enabled,
        tags: Vec::new(),
    };

    // Update in-memory config
    {
        let mut cfg = state.asset_config.write().await;
        let list = match class.as_str() {
            "fx" => &mut cfg.fx,
            "crypto" => &mut cfg.crypto,
            _ => &mut cfg.stocks, // stock and etf both go in stocks
        };
        if list.iter().any(|a| a.symbol == symbol) {
            return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("{} already exists", symbol)}))));
        }
        list.push(entry);

        // Write to disk atomically
        if let Err(e) = save_asset_config(&cfg) {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to save: {}", e)}))));
        }
    }

    Ok(Json(serde_json::json!({"status": "ok", "symbol": symbol, "class": class})))
}

#[derive(serde::Deserialize)]
struct ToggleAssetRequest {
    enabled: bool,
}

async fn toggle_asset(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Json(req): Json<ToggleAssetRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut cfg = state.asset_config.write().await;

    // Find which list contains this symbol, then update
    let target: Option<&mut config::AssetEntry> =
        if let Some(pos) = cfg.stocks.iter().position(|a| a.symbol == symbol) {
            Some(&mut cfg.stocks[pos])
        } else if let Some(pos) = cfg.fx.iter().position(|a| a.symbol == symbol) {
            Some(&mut cfg.fx[pos])
        } else if let Some(pos) = cfg.crypto.iter().position(|a| a.symbol == symbol) {
            Some(&mut cfg.crypto[pos])
        } else {
            None
        };

    match target {
        Some(entry) => entry.enabled = req.enabled,
        None => return Err((StatusCode::NOT_FOUND, Json(serde_json::json!({"error": format!("{} not found", symbol)})))),
    }

    if let Err(e) = save_asset_config(&cfg) {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("Failed to save: {}", e)}))));
    }

    Ok(Json(serde_json::json!({"status": "ok", "symbol": symbol, "enabled": req.enabled})))
}

/// Save AssetConfig to config/assets.json atomically (write to temp, then rename)
fn save_asset_config(cfg: &config::AssetConfig) -> Result<(), String> {
    let json = serde_json::to_string_pretty(cfg)
        .map_err(|e| format!("Serialize error: {}", e))?;
    let tmp_path = "config/assets.json.tmp";
    std::fs::write(tmp_path, &json)
        .map_err(|e| format!("Write error: {}", e))?;
    std::fs::rename(tmp_path, "config/assets.json")
        .map_err(|e| format!("Rename error: {}", e))?;
    Ok(())
}

// ════════════════════════════════════════
// Live Price Fetching
// ════════════════════════════════════════

/// Fetch live prices from Yahoo Finance (stocks/FX/market indicators) and CoinGecko (crypto),
/// then insert into the database so inference always uses the latest price.
async fn fetch_and_store_live_prices(
    state: &AppState,
    include_stocks: bool,
    include_fx: bool,
    include_crypto: bool,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let asset_config = state.asset_config.read().await.clone();
    let today = Utc::now().format("%Y-%m-%d").to_string();
    let client = &state.http_client;

    // Collect all (class, symbol, price, volume) tuples
    let mut updates: Vec<(&str, String, f64, Option<f64>)> = Vec::new();

    // ── Stocks ──
    if include_stocks {
        for entry in asset_config.enabled_stocks() {
            match stocks::fetch_quote(client, &entry.symbol).await {
                Ok(q) if q.price > 0.0 => {
                    updates.push(("stock", entry.symbol.clone(), q.price, Some(q.volume as f64)));
                }
                Err(e) => eprintln!("  [LivePrice] {} error: {}", entry.symbol, e),
                _ => {}
            }
        }
    }

    // ── FX ──
    if include_fx {
        for entry in asset_config.enabled_fx() {
            match stocks::fetch_quote(client, &entry.symbol).await {
                Ok(q) if q.price > 0.0 => {
                    updates.push(("fx", entry.symbol.clone(), q.price, None));
                }
                Err(e) => eprintln!("  [LivePrice] {} error: {}", entry.symbol, e),
                _ => {}
            }
        }
    }

    // ── Crypto (CoinGecko bulk — single API call for all coins) ──
    if include_crypto {
        match crypto::fetch_top_coins(client).await {
            Ok(coins) => {
                let enabled_ids: std::collections::HashSet<String> = asset_config
                    .enabled_crypto().iter().map(|e| e.symbol.clone()).collect();
                for coin in &coins {
                    if coin.current_price > 0.0 && (enabled_ids.contains(&coin.id) || enabled_ids.is_empty()) {
                        updates.push(("crypto", coin.id.clone(), coin.current_price, coin.total_volume));
                    }
                }
            }
            Err(e) => eprintln!("  [LivePrice] CoinGecko error: {}", e),
        }
    }

    // ── Market indicators (VIX, treasuries, sector ETFs, gold, dollar) ──
    // Always fetch these — they feed into feature engineering for all asset classes
    for ticker in features::MARKET_TICKERS {
        match stocks::fetch_quote(client, ticker).await {
            Ok(q) if q.price > 0.0 => {
                updates.push(("market", ticker.to_string(), q.price, None));
            }
            Err(e) => eprintln!("  [LivePrice] {} error: {}", ticker, e),
            _ => {}
        }
    }
    // SPY is used for market context too
    if !include_stocks {
        // SPY might not have been fetched if stocks were skipped
        if let Ok(q) = stocks::fetch_quote(client, "SPY").await {
            if q.price > 0.0 {
                updates.push(("stock", "SPY".to_string(), q.price, Some(q.volume as f64)));
            }
        }
    }

    let count = updates.len();
    let db_path = state.db_path.clone();
    let today_clone = today.clone();

    tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        for (class, symbol, price, volume) in &updates {
            let _ = match *class {
                "stock" => database.upsert_stock_price(symbol, *price, *volume, &today_clone),
                "fx" => database.upsert_fx_price(symbol, *price, *volume, &today_clone),
                "crypto" => database.upsert_crypto_price(symbol, *price, *volume, &today_clone),
                "market" => database.upsert_market_price(symbol, *price, &today_clone),
                _ => Ok(()),
            };
        }
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
    }).await??;

    println!("  [LivePrice] Updated {} live prices for {}", count, today);
    Ok(count)
}

// ════════════════════════════════════════
// Signal Generation Pipeline (INFERENCE ONLY)
// ════════════════════════════════════════

/// Refresh all signals (loads saved model weights, runs inference)
async fn refresh_signals(state: &AppState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Fetch live prices first so inference uses current data
    if let Err(e) = fetch_and_store_live_prices(state, true, true, true).await {
        eprintln!("  [LivePrice] Warning: {}", e);
    }

    let db_path = state.db_path.clone();
    let asset_config = state.asset_config.read().await.clone();

    let (new_signals, regime_state) = tokio::task::spawn_blocking(move || {
        generate_all_signals(&db_path, &asset_config)
    }).await??;

    // Update regime state
    {
        let mut regime = state.regime.write().await;
        *regime = regime_state;
    }

    // Store snapshots + signal history in SQLite (backward compat)
    {
        let database = db::Database::new(&state.db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        let ts = Utc::now().to_rfc3339();
        for sig in &new_signals {
            let _ = database.insert_signal_snapshot(sig, 3);
            record_signal_history(&database, sig, &ts);
        }
    }

    // Dual-write: also write signals to Postgres (single source of truth)
    {
        let ts = Utc::now().to_rfc3339();
        let mut pg_ok = 0u64;
        let mut pg_fail = 0u64;
        let mut last_err = String::new();
        for sig in &new_signals {
            if !pg::ASSET_UNIVERSE.contains(&sig.asset_class.as_str()) { continue; }
            let linreg = sig.models.get("linreg").or_else(|| sig.models.get("ridge")).map(|m| m.probability_up).unwrap_or(0.0);
            let logreg = sig.models.get("logreg").or_else(|| sig.models.get("lgbm")).map(|m| m.probability_up).unwrap_or(0.0);
            let gbt = sig.models.get("gbt").or_else(|| sig.models.get("gru")).map(|m| m.probability_up).unwrap_or(0.0);
            if let Err(e) = pg::insert_signal(
                &state.pg_pool, &ts, &sig.asset, &sig.asset_class, &sig.signal,
                sig.price, sig.technical.confidence, linreg, logreg, gbt,
            ).await {
                pg_fail += 1;
                last_err = format!("{}: {}", sig.asset, e);
                eprintln!("  [PG] Signal write failed for {}: {}", sig.asset, e);
            } else {
                pg_ok += 1;
            }
        }
        // Update data quality tracking
        {
            let mut dq = state.data_quality.write().await;
            dq.pg_write_successes += pg_ok;
            dq.pg_write_failures += pg_fail;
            dq.last_signal_generation = Some(ts.clone());
            if pg_fail > 0 {
                dq.last_pg_failure = Some(ts.clone());
                dq.last_pg_failure_error = Some(last_err);
            }
            if pg_ok > 0 {
                dq.last_successful_signal_write = Some(ts);
            }
        }
        if pg_fail > 0 {
            eprintln!("  [DataQuality] PG write: {}/{} succeeded, {} failed",
                pg_ok, pg_ok + pg_fail, pg_fail);
        }
    }

    // Resolve stale Postgres signals
    batch_resolve_signals_pg(&state.pg_pool, &new_signals).await;

    // Update the cache
    let mut sigs = state.signals.write().await;
    sigs.clear();
    for sig in new_signals {
        sigs.insert(sig.asset.clone(), sig);
    }

    Ok(())
}

/// Refresh signals with market hours awareness
async fn refresh_signals_with_market_hours(state: &AppState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let now = Utc::now();
    let weekday = now.weekday();
    let hour = now.hour();

    let is_weekend = weekday == Weekday::Sat || weekday == Weekday::Sun;

    // US stocks: Mon-Fri, 14:30-21:00 UTC (for price fetching)
    let us_market_open = !is_weekend && hour >= 14 && hour <= 21;

    // FX: Sun 22:00 UTC to Fri 22:00 UTC (nearly 24/5) (for price fetching)
    let fx_open = match weekday {
        Weekday::Sat => false,
        Weekday::Sun => hour >= 22,
        Weekday::Fri => hour < 22,
        _ => true,
    };

    // Crypto: 24/7
    let crypto_open = true;

    // Fetch live prices only when markets are open
    if let Err(e) = fetch_and_store_live_prices(state, us_market_open, fx_open, crypto_open).await {
        eprintln!("  [LivePrice] Warning: {}", e);
    }

    let db_path = state.db_path.clone();
    let asset_config = state.asset_config.read().await.clone();

    // Generate signals for ALL asset classes regardless of market hours.
    // On weekends/off-hours, models use last known prices from DB (e.g. Friday close).
    // This ensures accuracy data flows continuously so the agent can trigger retrains.
    // Resolution logic (can_resolve_signal) already skips weekends for stocks.
    let (new_signals, regime_state) = tokio::task::spawn_blocking(move || {
        generate_signals_filtered(&db_path, &asset_config, true, true, true)
    }).await??;

    // Update regime state
    {
        let mut regime = state.regime.write().await;
        *regime = regime_state;
    }

    // Store snapshots + signal history in SQLite (backward compat)
    {
        let database = db::Database::new(&state.db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        let ts = Utc::now().to_rfc3339();
        for sig in &new_signals {
            let _ = database.insert_signal_snapshot(sig, 3);
            record_signal_history(&database, sig, &ts);
        }
    }

    // Dual-write: also write signals to Postgres (single source of truth)
    {
        let ts = Utc::now().to_rfc3339();
        let mut pg_ok = 0u64;
        let mut pg_fail = 0u64;
        let mut last_err = String::new();
        for sig in &new_signals {
            if !pg::ASSET_UNIVERSE.contains(&sig.asset_class.as_str()) { continue; }
            let linreg = sig.models.get("linreg").or_else(|| sig.models.get("ridge")).map(|m| m.probability_up).unwrap_or(0.0);
            let logreg = sig.models.get("logreg").or_else(|| sig.models.get("lgbm")).map(|m| m.probability_up).unwrap_or(0.0);
            let gbt = sig.models.get("gbt").or_else(|| sig.models.get("gru")).map(|m| m.probability_up).unwrap_or(0.0);
            if let Err(e) = pg::insert_signal(
                &state.pg_pool, &ts, &sig.asset, &sig.asset_class, &sig.signal,
                sig.price, sig.technical.confidence, linreg, logreg, gbt,
            ).await {
                pg_fail += 1;
                last_err = format!("{}: {}", sig.asset, e);
                eprintln!("  [PG] Signal write failed for {}: {}", sig.asset, e);
            } else {
                pg_ok += 1;
            }
        }
        {
            let mut dq = state.data_quality.write().await;
            dq.pg_write_successes += pg_ok;
            dq.pg_write_failures += pg_fail;
            dq.last_signal_generation = Some(ts.clone());
            if pg_fail > 0 {
                dq.last_pg_failure = Some(ts.clone());
                dq.last_pg_failure_error = Some(last_err);
            }
            if pg_ok > 0 {
                dq.last_successful_signal_write = Some(ts);
            }
        }
        if pg_fail > 0 {
            eprintln!("  [DataQuality] PG write: {}/{} succeeded, {} failed",
                pg_ok, pg_ok + pg_fail, pg_fail);
        }
    }

    // Resolve stale Postgres signals
    batch_resolve_signals_pg(&state.pg_pool, &new_signals).await;

    // Merge into cache (only replace refreshed assets)
    let mut sigs = state.signals.write().await;
    for sig in new_signals {
        sigs.insert(sig.asset.clone(), sig);
    }

    if !us_market_open { println!("  [Scheduler] US stock market closed — skipped stocks"); }
    if !fx_open { println!("  [Scheduler] FX market closed — skipped FX"); }

    Ok(())
}

/// Generate signals for all enabled assets
fn generate_all_signals(
    db_path: &str,
    asset_config: &config::AssetConfig,
) -> Result<(Vec<enriched_signals::EnrichedSignal>, Option<market_regime::MarketRegimeState>), Box<dyn std::error::Error + Send + Sync>> {
    generate_signals_filtered(db_path, asset_config, true, true, true)
}

/// Generate signals with market hours filtering — inference only, no training
fn generate_signals_filtered(
    db_path: &str,
    asset_config: &config::AssetConfig,
    include_stocks: bool,
    include_fx: bool,
    include_crypto: bool,
) -> Result<(Vec<enriched_signals::EnrichedSignal>, Option<market_regime::MarketRegimeState>), Box<dyn std::error::Error + Send + Sync>> {
    let database = db::Database::new(db_path)
        .map_err(|e| format!("DB error: {}", e))?;

    // Build market context from stored data
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")
        .unwrap_or_default().iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker).unwrap_or_default();
        market_histories.insert(ticker.to_string(), prices);
    }
    market_histories.insert("HY_SPREAD".to_string(), database.get_market_prices("HY_SPREAD").unwrap_or_default());
    market_histories.insert("BREAKEVEN_5Y".to_string(), database.get_market_prices("BREAKEVEN_5Y").unwrap_or_default());
    let market_context = features::build_market_context(&market_histories);

    // Load Fear & Greed history once — used by stocks, FX, and crypto
    let fear_greed_history = database.get_fear_greed_history().unwrap_or_default();
    let fg_ref: Option<&[(String, f64)]> = if fear_greed_history.is_empty() {
        None
    } else {
        Some(&fear_greed_history)
    };

    // ── Compute regime BEFORE signal generation so sell thresholds are regime-aware ──
    let spy_prices_for_regime: Vec<f64> = market_histories.get("SPY")
        .cloned()
        .unwrap_or_default();
    let regime_computed = market_regime::compute_regime(&spy_prices_for_regime);
    if let Some(ref regime_state) = regime_computed {
        ensemble::set_market_regime(&regime_state.regime);
    }

    let mut enriched_signals = Vec::new();

    // ── Stock signals (inference only) ──
    if include_stocks {
        let enabled_stocks = asset_config.enabled_stocks();
        for asset_entry in &enabled_stocks {
            let points = match database.get_stock_history(&asset_entry.symbol) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if points.len() < 300 { continue; }

            if let Some(sig) = infer_stock_signal(&asset_entry.symbol, &points, &market_context, fg_ref) {
                enriched_signals.push(sig);
            }
        }
    }

    // ── FX signals disabled (descoped — separate project) ──
    // ── Crypto signals disabled (descoped — separate project) ──

    // ── Populate LLM sentiment (display only, no signal adjustment) ──
    if let Ok(conn) = rusqlite::Connection::open(db_path) {
        for sig in enriched_signals.iter_mut() {
            let data = news_sentiment::get_recent_sentiment(&conn, &sig.asset, 1);
            if let Some(latest) = data.first() {
                sig.llm_sentiment = latest.combined_score;
                sig.llm_analysis = latest.llm_analysis.clone();
            }
        }
    }

    // ── Apply market regime defensive rotation ──
    // (regime_computed was already computed above before signal generation)
    if let Some(ref regime_state) = regime_computed {
        let defensive = asset_config.defensive_symbols();
        let modified = market_regime::apply_regime_overlay(regime_state, &mut enriched_signals, &defensive);
        println!("  [Regime] {} (SPY 20d: {:.2}%), {} signals modified",
            regime_state.regime, regime_state.spy_return_20d_pct, modified);

        // Stamp each signal with the current regime
        let regime_str = regime_state.regime.to_string();
        for sig in enriched_signals.iter_mut() {
            sig.market_regime = Some(regime_str.clone());
        }
    } else {
        println!("  [Regime] Insufficient SPY data for regime calculation");
    }

    println!("  Generated {} enriched signals (inference only)", enriched_signals.len());
    Ok((enriched_signals, regime_computed))
}

// ════════════════════════════════════════
// Inference-Only Signal Generation
// ════════════════════════════════════════

// Inference functions are in the shared inference module (src/inference.rs)
// Used via: inference::infer_with_saved_models(), inference::normalise_features(), etc.

/// Generate a single stock signal via inference
fn infer_stock_signal(
    symbol: &str,
    points: &[analysis::PricePoint],
    market_context: &features::MarketContext,
    fear_greed: Option<&[(String, f64)]>,
) -> Option<enriched_signals::EnrichedSignal> {
    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
    let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

    let samples = features::build_rich_features_ext(
        &prices, &volumes, &timestamps,
        Some(market_context), "stock",
        features::sector_etf_for(symbol),
        None, fear_greed, None, None,
    );
    if samples.is_empty() { return None; }

    // Try regression models first (v6), fall back to classification (v5)
    let reg = inference::infer_regression_models(symbol, &samples);

    let result = analysis::analyse_coin(symbol, points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let mut signal = if let Some(ref reg_result) = reg {
        let mut sig = ensemble::regression_signal(symbol, reg_result, result.current_price, result.rsi_14.unwrap_or(50.0), trend, "stock");
        // Multi-horizon confirmation: filter through 5d model agreement
        let filtered = inference::apply_horizon_agreement(&sig.signal, symbol, &samples, reg_result, "stock");
        if filtered != sig.signal {
            sig.signal = filtered;
        }
        sig
    } else {
        let wf = inference::infer_with_saved_models(symbol, &samples)?;
        ensemble::ensemble_signal(symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend)
    };

    let vol_5d = if prices.len() >= 5 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-5..])))
    } else { None };
    let vol_20d = if prices.len() >= 20 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-20..])))
    } else { None };

    let bb_pos = extract_bb_position(&samples);

    Some(enriched_signals::enrich_signal(&signal, "stock", bb_pos, vol_5d, vol_20d))
}

/// Generate a single FX signal via inference
fn infer_fx_signal(
    symbol: &str,
    points: &[analysis::PricePoint],
    market_context: &features::MarketContext,
    fear_greed: Option<&[(String, f64)]>,
) -> Option<enriched_signals::EnrichedSignal> {
    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
    let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

    let samples = features::build_rich_features_ext(
        &prices, &volumes, &timestamps,
        Some(market_context), "fx",
        Some(symbol), None, fear_greed, None, None,
    );
    if samples.is_empty() { return None; }

    let reg = inference::infer_regression_models(symbol, &samples);

    let result = analysis::analyse_coin(symbol, points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = if let Some(ref reg_result) = reg {
        let mut sig = ensemble::regression_signal(symbol, reg_result, result.current_price, result.rsi_14.unwrap_or(50.0), trend, "fx");
        let filtered = inference::apply_horizon_agreement(&sig.signal, symbol, &samples, reg_result, "fx");
        if filtered != sig.signal {
            sig.signal = filtered;
        }
        sig
    } else {
        let wf = inference::infer_with_saved_models(symbol, &samples)?;
        ensemble::ensemble_signal(symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend)
    };

    let vol_5d = if prices.len() >= 5 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-5..])))
    } else { None };
    let vol_20d = if prices.len() >= 20 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-20..])))
    } else { None };

    let bb_pos = extract_bb_position(&samples);

    Some(enriched_signals::enrich_signal(&signal, "fx", bb_pos, vol_5d, vol_20d))
}

/// Generate a single crypto signal via inference
fn infer_crypto_signal(
    coin_id: &str,
    database: &db::Database,
    crypto_enrichment: &HashMap<String, Vec<crypto_features::CryptoFeatureRow>>,
) -> Option<enriched_signals::EnrichedSignal> {
    let points = database.get_coin_history(coin_id).ok()?;
    if points.len() < 200 { return None; }

    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();

    let base_samples = gbt::build_extended_features(&prices, &volumes);
    if base_samples.is_empty() { return None; }

    let enriched_samples: Vec<ml::Sample> = if let Some(crypto_rows) = crypto_enrichment.get(coin_id) {
        let base_start = 33_usize;
        base_samples.iter().enumerate().map(|(i, sample)| {
            let mut features = sample.features.clone();
            let date_idx = base_start + i;
            if date_idx < crypto_rows.len() {
                for (_name, val) in crypto_rows[date_idx].to_feature_vec() { features.push(val); }
            } else {
                for _ in 0..crypto_features::CryptoFeatureRow::feature_count() { features.push(0.0); }
            }
            ml::Sample { features, label: sample.label }
        }).collect()
    } else {
        base_samples.iter().map(|sample| {
            let mut features = sample.features.clone();
            for _ in 0..crypto_features::CryptoFeatureRow::feature_count() { features.push(0.0); }
            ml::Sample { features, label: sample.label }
        }).collect()
    };

    if enriched_samples.is_empty() { return None; }

    let reg = inference::infer_regression_models(coin_id, &enriched_samples);

    let result = analysis::analyse_coin(coin_id, &points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = if let Some(ref reg_result) = reg {
        let mut sig = ensemble::regression_signal(coin_id, reg_result, result.current_price, result.rsi_14.unwrap_or(50.0), trend, "crypto");
        let filtered = inference::apply_horizon_agreement(&sig.signal, coin_id, &enriched_samples, reg_result, "crypto");
        if filtered != sig.signal {
            sig.signal = filtered;
        }
        sig
    } else {
        let wf = inference::infer_with_saved_models(coin_id, &enriched_samples)?;
        ensemble::ensemble_signal(coin_id, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend)
    };

    let vol_5d = if prices.len() >= 5 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-5..])))
    } else { None };
    let vol_20d = if prices.len() >= 20 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-20..])))
    } else { None };

    Some(enriched_signals::enrich_signal(&signal, "crypto", None, vol_5d, vol_20d))
}

// ════════════════════════════════════════
// Utility Functions
// ════════════════════════════════════════

fn daily_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
}

/// Extract BB position from the last sample's features (feature index 3 is BB Position)
fn extract_bb_position(samples: &[ml::Sample]) -> Option<f64> {
    samples.last().map(|s| {
        if s.features.len() > 3 {
            s.features[3].clamp(0.0, 1.0)
        } else {
            0.5
        }
    })
}

// ════════════════════════════════════════
// Feedback Handlers
// ════════════════════════════════════════

#[derive(serde::Deserialize)]
struct SignalFeedbackRequest {
    asset: String,
    signal_type: String,
    reaction: String,  // "up" or "down"
}

async fn email_unsubscribe(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> axum::response::Html<String> {
    let email = params.get("email").cloned().unwrap_or_default();
    if !email.is_empty() {
        let db_path = state.db_path.clone();
        let email_clone = email.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Ok(database) = db::Database::new(&db_path) {
                let _ = database.disable_email_alerts_by_email(&email_clone);
            }
        }).await;
    }
    axum::response::Html(format!(
        r#"<!DOCTYPE html><html><head><title>Unsubscribed</title></head>
        <body style="background:#0a0e17;color:#e5e7eb;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh">
        <div style="text-align:center"><h1>Unsubscribed</h1><p>You will no longer receive email alerts from Alpha Signal.</p></div>
        </body></html>"#
    ))
}

async fn submit_signal_feedback(
    State(state): State<AppState>,
    Json(req): Json<SignalFeedbackRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.execute_raw(&format!(
            "INSERT INTO signal_feedback (asset, signal_type, reaction) VALUES ('{}', '{}', '{}')",
            req.asset.replace('\'', "''"),
            req.signal_type.replace('\'', "''"),
            req.reaction.replace('\'', "''"),
        )).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok::<_, StatusCode>(Json(serde_json::json!({"status": "ok"})))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
}

#[derive(serde::Deserialize)]
struct SurveyFeedbackRequest {
    q_understand: Option<String>,
    q_check_daily: Option<String>,
    q_trust_more: Option<String>,
    q_missing: Option<String>,
    q_would_pay: Option<String>,
}

async fn submit_survey_feedback(
    State(state): State<AppState>,
    Json(req): Json<SurveyFeedbackRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.execute_raw(&format!(
            "INSERT INTO survey_feedback (q_understand, q_check_daily, q_trust_more, q_missing, q_would_pay) \
             VALUES ({}, {}, {}, {}, {})",
            opt_sql(&req.q_understand), opt_sql(&req.q_check_daily),
            opt_sql(&req.q_trust_more), opt_sql(&req.q_missing),
            opt_sql(&req.q_would_pay),
        )).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok::<_, StatusCode>(Json(serde_json::json!({"status": "ok"})))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
}

fn opt_sql(v: &Option<String>) -> String {
    match v {
        Some(s) => format!("'{}'", s.replace('\'', "''")),
        None => "NULL".to_string(),
    }
}

async fn get_feedback(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Count signal feedback
        let signal_up: i64 = database.execute_raw(
            "SELECT 0 FROM signal_feedback WHERE reaction = 'up'"
        ).unwrap_or(0) as i64;
        let signal_count: i64 = database.execute_raw(
            "SELECT 0 FROM signal_feedback"
        ).unwrap_or(0) as i64;

        // Count survey submissions
        let survey_count: i64 = database.execute_raw(
            "SELECT 0 FROM survey_feedback"
        ).unwrap_or(0) as i64;

        Ok::<_, StatusCode>(Json(serde_json::json!({
            "signal_feedback_count": signal_count,
            "signal_thumbs_up": signal_up,
            "survey_count": survey_count,
        })))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
}

// ════════════════════════════════════════
// Sentiment Handler
// ════════════════════════════════════════

async fn get_sentiment(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let data = news_sentiment::get_sentiment_history(&conn, &symbol, 30);
        Ok::<_, StatusCode>(Json(serde_json::json!({
            "symbol": symbol,
            "data": data,
        })))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
}

// ════════════════════════════════════════
// Startup Database Migrations
// ════════════════════════════════════════

/// Batch-resolve ALL unresolved signal_history entries using current prices from the cache.
/// Unlike record_signal_history() which resolves one per asset per cycle, this resolves everything.
/// Only requires signals to be 4+ hours old (drops the end-of-day market hours restriction).
/// Compute the minimum resolution time for a signal based on asset class.
/// - Stocks/ETFs: next trading day close (20:00 UTC) — skip weekends
/// - Crypto: 24 hours after signal timestamp
/// - FX: next trading day close (17:00 UTC) — skip weekends
fn resolution_ready(signal_dt: chrono::DateTime<Utc>, asset_class: &str, now: chrono::DateTime<Utc>) -> bool {
    match asset_class {
        "crypto" => {
            // Resolve after 24 hours
            (now - signal_dt).num_hours() >= 24
        }
        "fx" => {
            // Next trading day close at 17:00 UTC, skip weekends
            let mut resolve_date = signal_dt.date_naive() + chrono::Duration::days(1);
            // Skip Saturday and Sunday
            while resolve_date.weekday() == chrono::Weekday::Sat || resolve_date.weekday() == chrono::Weekday::Sun {
                resolve_date += chrono::Duration::days(1);
            }
            let resolve_dt = resolve_date.and_hms_opt(17, 0, 0)
                .map(|ndt| ndt.and_utc())
                .unwrap_or(signal_dt);
            now >= resolve_dt
        }
        _ => {
            // Stocks/ETFs/commodities: next trading day close at 20:00 UTC, skip weekends
            let mut resolve_date = signal_dt.date_naive() + chrono::Duration::days(1);
            while resolve_date.weekday() == chrono::Weekday::Sat || resolve_date.weekday() == chrono::Weekday::Sun {
                resolve_date += chrono::Duration::days(1);
            }
            let resolve_dt = resolve_date.and_hms_opt(20, 0, 0)
                .map(|ndt| ndt.and_utc())
                .unwrap_or(signal_dt);
            now >= resolve_dt
        }
    }
}

/// Parse a timestamp string from either RFC3339 (`2026-04-17T01:02:36+01:00`)
/// or PostgreSQL `::text` format (`2026-04-17 01:02:36.948369+01`).
fn parse_pg_timestamp(ts: &str) -> Option<chrono::DateTime<Utc>> {
    // Try RFC3339 first
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts) {
        return Some(dt.with_timezone(&Utc));
    }
    // Normalize Postgres ::text format: replace space with T, pad short timezone offset
    let mut s = ts.replace(' ', "T");
    // Postgres outputs +01 or -05, RFC3339 needs +01:00 or -05:00
    if let Some(pos) = s.rfind('+').or_else(|| s.rfind('-').filter(|&p| p > 10)) {
        let tz_part = &s[pos..];
        if tz_part.len() == 3 {
            // +01 -> +01:00
            s.push_str(":00");
        } else if tz_part.len() == 5 && !tz_part.contains(':') {
            // +0100 -> +01:00
            s.insert(pos + 3, ':');
        }
    }
    chrono::DateTime::parse_from_rfc3339(&s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Minimum price move threshold for a signal to be considered "correct", by asset class.
fn min_threshold_for_class(asset_class: &str) -> f64 {
    match asset_class {
        "crypto" => 1.0,   // 1.0% minimum move
        "fx"     => 0.2,   // 0.2% minimum move
        _        => 0.5,   // 0.5% for stocks, ETFs, commodities
    }
}

fn batch_resolve_signal_history(db_path: &str, signals: &HashMap<String, enriched_signals::EnrichedSignal>) {
    let database = match db::Database::new(db_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("  [BatchResolve] DB error: {}", e); return; }
    };

    let unresolved = match database.get_all_unresolved_signals() {
        Ok(u) => u,
        Err(e) => { eprintln!("  [BatchResolve] Error fetching unresolved: {}", e); return; }
    };

    if unresolved.is_empty() { return; }

    let now = Utc::now();
    let resolve_ts = now.to_rfc3339();
    let mut resolved = 0;

    for sig in &unresolved {
        let signal_dt = match chrono::DateTime::parse_from_rfc3339(&sig.timestamp) {
            Ok(t) => t.with_timezone(&Utc),
            Err(_) => continue,
        };

        // Fixed holding period resolution based on asset class
        if !resolution_ready(signal_dt, &sig.asset_class, now) { continue; }

        // Get current price from signals cache
        let current_price = if let Some(cached) = signals.get(&sig.asset) {
            cached.price
        } else {
            continue; // No current price available
        };

        if sig.price_at_signal <= 0.0 || current_price <= 0.0 { continue; }

        let pct_change = (current_price - sig.price_at_signal) / sig.price_at_signal * 100.0;
        let threshold = min_threshold_for_class(&sig.asset_class);
        let was_correct = match sig.signal_type.as_str() {
            "BUY"  => pct_change > threshold,
            "SELL" | "SHORT" => pct_change < -threshold,
            "HOLD" => pct_change.abs() < threshold,
            _ => false,
        };

        if let Err(e) = database.resolve_signal_history(sig.id, current_price, pct_change, was_correct, &resolve_ts) {
            eprintln!("  [BatchResolve] Error resolving signal {}: {}", sig.id, e);
        } else {
            resolved += 1;
        }
    }

    if resolved > 0 {
        println!("  [BatchResolve] Resolved {} of {} unresolved signal_history entries", resolved, unresolved.len());
    }
}

/// Resolve pending predictions by comparing prediction price to current price.
/// Uses simplified 4-hour minimum rule (no end-of-day gating for scorecard accuracy).
fn resolve_pending_predictions(db_path: &str, signals: &HashMap<String, enriched_signals::EnrichedSignal>) {
    let database = match db::Database::new(db_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("  [Predictions] DB error: {}", e); return; }
    };

    let pending = match database.get_pending_predictions() {
        Ok(p) => p,
        Err(e) => { eprintln!("  [Predictions] Error fetching pending: {}", e); return; }
    };

    if pending.is_empty() { return; }

    let now = Utc::now();
    let resolve_ts = now.to_rfc3339();
    let mut resolved = 0;

    for pred in &pending {
        // Must be at least 4 hours old
        let pred_dt = match chrono::DateTime::parse_from_rfc3339(&pred.timestamp) {
            Ok(t) => t.with_timezone(&Utc),
            Err(_) => continue,
        };
        if (now - pred_dt).num_hours() < 4 { continue; }

        // Get current price from signals cache
        let current_price = match signals.get(&pred.asset) {
            Some(s) => s.price,
            None => continue,
        };

        let price_change = current_price - pred.price_at_prediction;
        let actual_direction = if price_change.abs() < pred.price_at_prediction * 0.001 {
            "FLAT"
        } else if price_change > 0.0 {
            "UP"
        } else {
            "DOWN"
        };

        let was_correct = match (pred.signal.as_str(), actual_direction) {
            ("BUY", "UP") | ("SELL", "DOWN") | ("SHORT", "DOWN") => true,
            ("BUY", "DOWN") | ("SELL", "UP") | ("SHORT", "UP") => false,
            _ => true,
        };

        let _ = database.update_prediction_outcome(pred.id, actual_direction, was_correct, current_price, &resolve_ts);
        resolved += 1;
    }

    if resolved > 0 {
        println!("  [Predictions] Resolved {} of {} pending predictions", resolved, pending.len());
    }
}

/// Batch-resolve unresolved signals in PostgreSQL using current cached prices.
/// Mirrors the SQLite batch_resolve logic but runs async against Postgres.
async fn batch_resolve_signals_pg(
    pool: &pg::PgPool,
    current_signals: &[enriched_signals::EnrichedSignal],
) {
    let unresolved = match pg::get_all_unresolved_signals(pool).await {
        Ok(u) => u,
        Err(e) => { eprintln!("  [PG-Resolve] Error: {}", e); return; }
    };
    if unresolved.is_empty() { return; }

    let now = Utc::now();
    let resolve_ts = now.to_rfc3339();
    let mut resolved = 0;

    // Build price lookup from current signals
    let prices: HashMap<String, f64> = current_signals.iter()
        .map(|s| (s.asset.clone(), s.price))
        .collect();

    for sig in &unresolved {
        let signal_dt = match parse_pg_timestamp(&sig.timestamp) {
            Some(t) => t,
            None => continue,
        };

        if !resolution_ready(signal_dt, &sig.asset_class, now) { continue; }

        let current_price = match prices.get(&sig.asset) {
            Some(&p) if p > 0.0 => p,
            _ => continue,
        };

        if sig.price_at_signal <= 0.0 { continue; }

        let pct_change = (current_price - sig.price_at_signal) / sig.price_at_signal * 100.0;
        let threshold = min_threshold_for_class(&sig.asset_class);
        let was_correct = match sig.signal_type.as_str() {
            "BUY" => pct_change > threshold,
            "SELL" | "SHORT" => pct_change < -threshold,
            "HOLD" => pct_change.abs() < threshold,
            _ => false,
        };

        if let Err(e) = pg::resolve_signal(pool, sig.id, current_price, pct_change, was_correct, &resolve_ts).await {
            eprintln!("  [PG-Resolve] Error resolving {}: {}", sig.id, e);
        } else {
            resolved += 1;
        }
    }

    if resolved > 0 {
        println!("  [PG-Resolve] Resolved {} of {} unresolved signals in Postgres", resolved, unresolved.len());
    }
}

// ════════════════════════════════════════
// User Portfolio Tracker
// ════════════════════════════════════════

async fn get_user_holdings(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Vec<db::UserHolding>>, StatusCode> {
    let user_id = extract_user_id(&headers);
    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.get_user_holdings_for(user_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;
    Ok(Json(result))
}

#[derive(serde::Deserialize)]
struct AddHoldingRequest {
    symbol: String,
    quantity: f64,
    start_date: String,
}

async fn add_user_holding(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<AddHoldingRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let user_id = extract_user_id(&headers);
    let symbol = req.symbol.trim().to_string();
    let asset_class = detect_asset_class(&symbol);
    let db_path = state.db_path.clone();
    let start_date = req.start_date.clone();
    let quantity = req.quantity;

    let id = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.insert_user_holding_for(user_id, &symbol, quantity, &start_date, &asset_class)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(Json(serde_json::json!({"status": "ok", "id": id})))
}

#[derive(serde::Deserialize)]
struct UpdateHoldingRequest {
    quantity: f64,
    start_date: String,
}

async fn update_user_holding(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateHoldingRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.update_user_holding(id, req.quantity, &req.start_date)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;
    Ok(Json(serde_json::json!({"status": "ok"})))
}

async fn delete_user_holding(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.delete_user_holding(id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;
    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// Detect asset class from a symbol string
fn detect_asset_class(symbol: &str) -> String {
    if symbol.ends_with("=X") {
        "fx".to_string()
    } else if is_crypto_id(symbol) {
        "crypto".to_string()
    } else {
        "stock".to_string()
    }
}

/// Check if a symbol is a known CoinGecko crypto ID or short ticker
fn is_crypto_id(s: &str) -> bool {
    matches!(s, "bitcoin" | "ethereum" | "solana" | "ripple" | "dogecoin"
        | "cardano" | "avalanche-2" | "chainlink" | "polkadot" | "near"
        | "sui" | "aptos" | "arbitrum" | "the-open-network" | "uniswap"
        | "tron" | "litecoin" | "shiba-inu" | "stellar" | "matic-network")
    || matches!(s, "BTC" | "ETH" | "DOGE" | "ADA" | "XRP" | "SOL" | "TRX"
        | "AVAX" | "LINK" | "DOT" | "NEAR" | "SUI" | "APT" | "ARB"
        | "TON" | "UNI" | "LTC" | "SHIB" | "XLM" | "MATIC")
}

/// POST /api/v1/user-portfolio/compare — run Follow Signals vs Buy & Hold comparison
// ── Portfolio backtest ────────────────────────────────────────

#[derive(serde::Deserialize)]
struct CompareRequest {
    frequency: Option<String>, // "daily" | "weekly", default "weekly"
}

async fn compare_portfolio(
    State(state): State<AppState>,
    Json(req): Json<CompareRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    let frequency = req.frequency.unwrap_or_else(|| "weekly".to_string());

    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let holdings = database.get_user_holdings().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        if holdings.is_empty() {
            return Ok::<_, StatusCode>(serde_json::json!({
                "has_data": false,
                "note": "No holdings added yet. Add your assets to see the comparison."
            }));
        }

        // Build market context ONCE for all holdings
        let market_context = simulator::build_market_context_from_db(&database);

        let mut per_asset = Vec::new();
        let mut total_buy_hold = 0.0_f64;
        let mut total_follow_signals = 0.0_f64;
        let mut total_cost = 0.0_f64;
        let mut total_trades = 0_usize;
        let mut total_wins = 0_usize;
        let mut total_trade_signals = 0_usize;

        for holding in &holdings {
            let asset_result = backtest_holding(
                &database, holding, &frequency, &market_context,
            );
            if let Some(ar) = asset_result {
                total_cost += ar.cost_basis;
                total_buy_hold += ar.buy_hold_value;
                total_follow_signals += ar.signal_value;
                total_trades += ar.total_trades;
                total_wins += ar.wins;
                total_trade_signals += ar.trade_signals;
                per_asset.push(ar);
            }
        }

        let buy_hold_return = if total_cost > 0.0 {
            (total_buy_hold - total_cost) / total_cost * 100.0
        } else { 0.0 };
        let signal_return = if total_cost > 0.0 {
            (total_follow_signals - total_cost) / total_cost * 100.0
        } else { 0.0 };

        let verdict = if (signal_return - buy_hold_return).abs() < 1.0 {
            "roughly_equal"
        } else if signal_return > buy_hold_return {
            "signals_win"
        } else {
            "buy_hold_wins"
        };

        let overall_win_rate = if total_trade_signals > 0 {
            total_wins as f64 / total_trade_signals as f64 * 100.0
        } else { 0.0 };

        // Build aggregate equity curve (sum across assets by date)
        let agg_curve = build_aggregate_equity_curve(&per_asset);

        // Compute portfolio-level Sharpe from aggregate curve
        let annualise = if frequency == "weekly" { 52.0_f64.sqrt() } else { 252.0_f64.sqrt() };
        let (sharpe_sig, sharpe_bh) = compute_sharpe_from_curve(&agg_curve, annualise);

        let per_asset_json: Vec<serde_json::Value> = per_asset.iter().map(|a| {
            // Downsample per-asset curve for JSON
            let curve = downsample_curve(&a.equity_curve, 300);
            serde_json::json!({
                "symbol": a.symbol,
                "asset_class": a.asset_class,
                "quantity": a.quantity,
                "start_date": a.start_date,
                "actual_start_date": a.actual_start_date,
                "start_price": a.start_price,
                "current_price": a.current_price,
                "cost_basis": round2(a.cost_basis),
                "buy_hold_value": round2(a.buy_hold_value),
                "buy_hold_return_pct": round2(a.buy_hold_return_pct),
                "signal_value": round2(a.signal_value),
                "signal_return_pct": round2(a.signal_return_pct),
                "signals_used": a.signals_used,
                "total_trades": a.total_trades,
                "win_rate_pct": round2(a.win_rate_pct),
                "sharpe_signals": round2(a.sharpe_signals),
                "sharpe_buy_hold": round2(a.sharpe_buy_hold),
                "equity_curve": curve,
                "note": a.note,
            })
        }).collect();

        let agg_curve_json = downsample_curve(&agg_curve, 500);

        Ok(serde_json::json!({
            "has_data": true,
            "frequency": frequency,
            "total_cost": round2(total_cost),
            "buy_hold_value": round2(total_buy_hold),
            "buy_hold_return_pct": round2(buy_hold_return),
            "signal_value": round2(total_follow_signals),
            "signal_return_pct": round2(signal_return),
            "verdict": verdict,
            "sharpe_signals": round2(sharpe_sig),
            "sharpe_buy_hold": round2(sharpe_bh),
            "overall_win_rate_pct": round2(overall_win_rate),
            "total_trades": total_trades,
            "equity_curve": agg_curve_json,
            "per_asset": per_asset_json,
        }))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(Json(result))
}

fn round2(v: f64) -> f64 { (v * 100.0).round() / 100.0 }

struct BacktestResult {
    symbol: String,
    asset_class: String,
    quantity: f64,
    start_date: String,
    actual_start_date: String,
    start_price: f64,
    current_price: f64,
    cost_basis: f64,
    buy_hold_value: f64,
    buy_hold_return_pct: f64,
    signal_value: f64,
    signal_return_pct: f64,
    signals_used: usize,
    total_trades: usize,
    wins: usize,
    trade_signals: usize,
    win_rate_pct: f64,
    sharpe_signals: f64,
    sharpe_buy_hold: f64,
    equity_curve: Vec<(String, f64, f64)>, // (date, signal_value, buy_hold_value)
    note: Option<String>,
}

fn backtest_holding(
    database: &db::Database,
    holding: &db::UserHolding,
    frequency: &str,
    market_context: &features::MarketContext,
) -> Option<BacktestResult> {
    // 1. Get full price history (dates are 10-char "YYYY-MM-DD")
    let all_prices = simulator::load_asset_prices(database, &holding.symbol, &holding.asset_class);
    if all_prices.is_empty() { return None; }

    // 2. Find actual start date from price array directly (avoids DB timestamp format issues)
    let holding_start = &holding.start_date[..10.min(holding.start_date.len())];
    let start_idx = all_prices.iter()
        .position(|(d, _)| d.as_str() >= holding_start)
        .unwrap_or(0);
    let actual_start_date = all_prices[start_idx].0.clone();
    let start_price = all_prices[start_idx].1;
    let current_price = all_prices.last()?.1;

    if start_price <= 0.0 { return None; }

    let cost_basis = holding.quantity * start_price;
    let buy_hold_value = holding.quantity * current_price;
    let buy_hold_return_pct = (current_price - start_price) / start_price * 100.0;

    // 3. Snap to nearest market trading day; only warn if gap > 7 calendar days
    let note = if actual_start_date.as_str() != holding_start {
        let requested = chrono::NaiveDate::parse_from_str(holding_start, "%Y-%m-%d").ok();
        let actual = chrono::NaiveDate::parse_from_str(&actual_start_date, "%Y-%m-%d").ok();
        match (requested, actual) {
            (Some(req), Some(act)) if (act - req).num_days().abs() > 7 => {
                Some(format!("Snapped to market open: {} (requested {} was {} days away)",
                    actual_start_date, holding_start, (act - req).num_days().abs()))
            }
            _ => None, // Silently snap to nearest trading day for weekends/holidays
        }
    } else {
        None
    };

    // 4. Build trading dates from actual_start_date onwards
    let price_map: std::collections::HashMap<String, f64> = all_prices.iter().cloned().collect();
    let actual_start_short = &actual_start_date[..10.min(actual_start_date.len())];
    let trading_dates: Vec<String> = all_prices.iter()
        .filter(|(d, _)| d.as_str() >= actual_start_short)
        .map(|(d, _)| d.clone())
        .collect();

    if trading_dates.is_empty() {
        return Some(BacktestResult {
            symbol: holding.symbol.clone(),
            asset_class: holding.asset_class.clone(),
            quantity: holding.quantity,
            start_date: holding.start_date.clone(),
            actual_start_date,
            start_price, current_price, cost_basis,
            buy_hold_value, buy_hold_return_pct,
            signal_value: buy_hold_value,
            signal_return_pct: buy_hold_return_pct,
            signals_used: 0, total_trades: 0, wins: 0, trade_signals: 0,
            win_rate_pct: 0.0, sharpe_signals: 0.0, sharpe_buy_hold: 0.0,
            equity_curve: Vec::new(), note,
        });
    }

    // 5. Determine signal dates based on frequency
    let signal_dates: std::collections::HashSet<String> = if frequency == "weekly" {
        weekly_signal_dates(&trading_dates)
    } else {
        trading_dates.iter().cloned().collect()
    };

    // 6. Generate signals: try model-based bulk, then fall back to signal_history
    let mut signal_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut note_suffix: Option<String> = None;

    let models = simulator::load_models_for_symbol(&holding.symbol);
    if let Some(ref m) = models {
        signal_map = simulator::generate_signals_bulk(
            &holding.symbol, &holding.asset_class,
            &all_prices, market_context, m,
        );
    }

    // Fallback: if model signals are empty, use actual signal_history from the database
    if signal_map.is_empty() {
        if let Ok(history) = database.get_signal_history_for_asset_from(
            &holding.symbol, actual_start_short,
        ) {
            for rec in &history {
                let date = rec.timestamp[..10.min(rec.timestamp.len())].to_string();
                signal_map.insert(date, rec.signal_type.clone());
            }
            if !signal_map.is_empty() {
                note_suffix = Some("Using recorded signal history".to_string());
            }
        }
    }

    if signal_map.is_empty() {
        // No signals at all — return buy-and-hold only
        let equity_curve: Vec<(String, f64, f64)> = trading_dates.iter()
            .filter_map(|d| price_map.get(d.as_str()).filter(|&&p| p > 0.0).map(|&p| {
                let v = round2(holding.quantity * p);
                (d.clone(), v, v)
            }))
            .collect();
        let annualise = if frequency == "weekly" { 52.0_f64.sqrt() } else { 252.0_f64.sqrt() };
        let bh_values: Vec<f64> = equity_curve.iter().map(|(_, _, b)| *b).collect();
        let sharpe_bh = compute_sharpe(&bh_values, annualise);
        return Some(BacktestResult {
            symbol: holding.symbol.clone(),
            asset_class: holding.asset_class.clone(),
            quantity: holding.quantity,
            start_date: holding.start_date.clone(),
            actual_start_date,
            start_price, current_price, cost_basis,
            buy_hold_value, buy_hold_return_pct,
            signal_value: buy_hold_value,
            signal_return_pct: buy_hold_return_pct,
            signals_used: 0, total_trades: 0, wins: 0, trade_signals: 0,
            win_rate_pct: 0.0, sharpe_signals: 0.0, sharpe_buy_hold: sharpe_bh,
            equity_curve,
            note: Some("No trained models or signal history found — showing buy & hold only".to_string()),
        });
    }

    // 7. Walk through all trading dates, executing trades on signal dates
    let mut in_position = true;
    let mut shares = holding.quantity;
    let mut cash = 0.0_f64;
    let mut total_trades = 0_usize;
    let mut signals_used = 0_usize;
    let mut wins = 0_usize;
    let mut trade_signals = 0_usize;
    let mut equity_curve: Vec<(String, f64, f64)> = Vec::new();
    let mut last_trade_price: Option<(String, f64)> = None;

    for date in &trading_dates {
        let price = match price_map.get(date.as_str()) {
            Some(&p) if p > 0.0 => p,
            _ => continue,
        };

        if signal_dates.contains(date) {
            let signal = signal_map.get(date.as_str())
                .cloned()
                .unwrap_or_else(|| "HOLD".to_string());

            signals_used += 1;

            match signal.as_str() {
                "SELL" if in_position => {
                    if let Some((ref sig_type, prev_price)) = last_trade_price {
                        if sig_type == "BUY" && price > prev_price { wins += 1; }
                        trade_signals += 1;
                    }
                    cash = shares * price * (1.0 - 0.001); // 0.1% transaction cost
                    shares = 0.0;
                    in_position = false;
                    total_trades += 1;
                    last_trade_price = Some(("SELL".to_string(), price));
                }
                "BUY" if !in_position => {
                    if let Some((ref sig_type, prev_price)) = last_trade_price {
                        if sig_type == "SELL" && price < prev_price { wins += 1; }
                        trade_signals += 1;
                    }
                    if price > 0.0 {
                        let cost_adjusted_cash = cash * (1.0 - 0.001); // 0.1% transaction cost
                        shares = cost_adjusted_cash / price;
                    }
                    cash = 0.0;
                    in_position = true;
                    total_trades += 1;
                    last_trade_price = Some(("BUY".to_string(), price));
                }
                _ => {}
            }
        }

        let signal_val = if in_position { shares * price } else { cash };
        let bh_val = holding.quantity * price;
        equity_curve.push((date.clone(), round2(signal_val), round2(bh_val)));
    }

    let signal_value = if in_position { shares * current_price } else { cash };
    let signal_return_pct = if cost_basis > 0.0 {
        (signal_value - cost_basis) / cost_basis * 100.0
    } else { 0.0 };

    let win_rate_pct = if trade_signals > 0 {
        wins as f64 / trade_signals as f64 * 100.0
    } else { 0.0 };

    let annualise = if frequency == "weekly" { 52.0_f64.sqrt() } else { 252.0_f64.sqrt() };
    let sig_values: Vec<f64> = equity_curve.iter().map(|(_, s, _)| *s).collect();
    let bh_values: Vec<f64> = equity_curve.iter().map(|(_, _, b)| *b).collect();
    let sharpe_signals = compute_sharpe(&sig_values, annualise);
    let sharpe_buy_hold = compute_sharpe(&bh_values, annualise);

    // Combine note and note_suffix
    let final_note = match (note, note_suffix) {
        (Some(n), Some(s)) => Some(format!("{} · {}", n, s)),
        (Some(n), None) => Some(n),
        (None, Some(s)) => Some(s),
        (None, None) => None,
    };

    Some(BacktestResult {
        symbol: holding.symbol.clone(),
        asset_class: holding.asset_class.clone(),
        quantity: holding.quantity,
        start_date: holding.start_date.clone(),
        actual_start_date,
        start_price, current_price, cost_basis,
        buy_hold_value, buy_hold_return_pct,
        signal_value, signal_return_pct,
        signals_used, total_trades, wins, trade_signals,
        win_rate_pct, sharpe_signals, sharpe_buy_hold,
        equity_curve, note: final_note,
    })
}

/// Pick the last trading day of each ISO week as signal dates
fn weekly_signal_dates(dates: &[String]) -> std::collections::HashSet<String> {
    use chrono::Datelike;
    let mut weeks: std::collections::HashMap<(i32, u32), String> = std::collections::HashMap::new();
    for d in dates {
        if let Ok(nd) = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d") {
            let key = (nd.iso_week().year(), nd.iso_week().week());
            weeks.entry(key)
                .and_modify(|existing| { if d.as_str() > existing.as_str() { *existing = d.clone(); } })
                .or_insert_with(|| d.clone());
        }
    }
    weeks.into_values().collect()
}

/// Compute annualised Sharpe ratio from a value series
fn compute_sharpe(values: &[f64], annualise: f64) -> f64 {
    if values.len() < 2 { return 0.0; }
    let returns: Vec<f64> = values.windows(2)
        .filter_map(|w| {
            if w[0] > 0.0 { Some((w[1] - w[0]) / w[0]) } else { None }
        })
        .collect();
    if returns.is_empty() { return 0.0; }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();
    if std == 0.0 { return 0.0; }
    (mean / std) * annualise
}

/// Compute Sharpe from aggregate equity curve tuples
fn compute_sharpe_from_curve(curve: &[(String, f64, f64)], annualise: f64) -> (f64, f64) {
    let sig: Vec<f64> = curve.iter().map(|(_, s, _)| *s).collect();
    let bh: Vec<f64> = curve.iter().map(|(_, _, b)| *b).collect();
    (compute_sharpe(&sig, annualise), compute_sharpe(&bh, annualise))
}

/// Build aggregate portfolio equity curve by summing per-asset curves aligned by date
fn build_aggregate_equity_curve(assets: &[BacktestResult]) -> Vec<(String, f64, f64)> {
    let mut by_date: std::collections::BTreeMap<String, (f64, f64)> = std::collections::BTreeMap::new();
    for a in assets {
        for (date, sig, bh) in &a.equity_curve {
            let entry = by_date.entry(date.clone()).or_insert((0.0, 0.0));
            entry.0 += sig;
            entry.1 += bh;
        }
    }
    by_date.into_iter().map(|(d, (s, b))| (d, round2(s), round2(b))).collect()
}

/// Downsample equity curve to at most `max_points` entries
fn downsample_curve(curve: &[(String, f64, f64)], max_points: usize) -> Vec<serde_json::Value> {
    if curve.len() <= max_points {
        return curve.iter().map(|(d, s, b)| {
            serde_json::json!({"date": d, "signal_value": s, "buy_hold_value": b})
        }).collect();
    }
    let step = curve.len() as f64 / max_points as f64;
    let mut result = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = (i as f64 * step) as usize;
        let (ref d, s, b) = curve[idx.min(curve.len() - 1)];
        result.push(serde_json::json!({"date": d, "signal_value": s, "buy_hold_value": b}));
    }
    // Always include the last point
    if let Some((ref d, s, b)) = curve.last() {
        let last = serde_json::json!({"date": d, "signal_value": s, "buy_hold_value": b});
        if result.last() != Some(&last) { result.push(last); }
    }
    result
}

/// Check whether a signal should be resolved now, based on asset class, signal type,
/// market hours, and minimum age rules.
///
/// Rules:
///   - No signal resolves in less than 4 hours
///   - Stocks:  only resolve Mon-Fri 14:30-21:00 UTC; BUY/SELL at end of day (hour >= 20)
///   - FX:      only resolve Mon 00:00 – Fri 22:00 UTC; BUY/SELL at end of day (hour >= 21)
///   - Crypto:  resolves 24/7; BUY/SELL at end of day (hour >= 23 or age >= 24h)
///   - HOLD:    resolves any time during open market hours (respecting 4h minimum)
fn can_resolve_signal(
    asset_class: &str,
    signal_type: &str,
    signal_ts: &str,
    now: chrono::DateTime<Utc>,
) -> bool {
    let signal_dt = match chrono::DateTime::parse_from_rfc3339(signal_ts) {
        Ok(t) => t.with_timezone(&Utc),
        Err(_) => return false,
    };

    let age_hours = (now - signal_dt).num_hours();

    // ── Global: no signal resolves in less than 4 hours ──
    if age_hours < 4 {
        return false;
    }

    let weekday = now.weekday();
    let hour = now.hour();

    match asset_class {
        "stock" => {
            // Stocks only resolve Mon-Fri 14:30-21:00 UTC
            if matches!(weekday, Weekday::Sat | Weekday::Sun) { return false; }
            if hour < 14 || hour > 21 { return false; }

            match signal_type {
                "BUY" | "SELL" => hour >= 20,  // end of trading day
                "HOLD" => true,                 // any time during open hours
                _ => false,
            }
        }
        "fx" => {
            // FX resolves Mon 00:00 through Fri 22:00 UTC
            let fx_open = match weekday {
                Weekday::Sat | Weekday::Sun => false,
                Weekday::Fri => hour < 22,
                _ => true,
            };
            if !fx_open { return false; }

            match signal_type {
                "BUY" | "SELL" => hour >= 21,  // end of NY session
                "HOLD" => true,
                _ => false,
            }
        }
        "crypto" => {
            // Crypto resolves 24/7 (4h minimum already enforced)
            match signal_type {
                "BUY" | "SELL" => hour >= 23 || age_hours >= 24,  // end of day
                "HOLD" => true,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Record a signal to signal_history table, resolving the previous unresolved signal for the same asset
fn record_signal_history(database: &db::Database, sig: &enriched_signals::EnrichedSignal, timestamp: &str) {
    // Extract model probabilities from the enriched signal
    let linreg_prob = sig.models.get("linreg").or_else(|| sig.models.get("ridge")).map(|m| m.probability_up);
    let logreg_prob = sig.models.get("logreg").or_else(|| sig.models.get("lgbm")).map(|m| m.probability_up);
    let gbt_prob = sig.models.get("gbt").or_else(|| sig.models.get("gru")).map(|m| m.probability_up);

    // Resolve previous unresolved signal for this asset (market-hours aware)
    if let Ok(Some(prev)) = database.get_last_unresolved_signal(&sig.asset) {
        let now = Utc::now();
        if can_resolve_signal(&prev.asset_class, &prev.signal_type, &prev.timestamp, now) {
            let current_price = sig.price;
            let prev_price = prev.price_at_signal;
            if prev_price > 0.0 {
                let pct_change = (current_price - prev_price) / prev_price * 100.0;
                let threshold = min_threshold_for_class(&prev.asset_class);
                let was_correct = match prev.signal_type.as_str() {
                    "BUY" => pct_change > threshold,
                    "SELL" | "SHORT" => pct_change < -threshold,
                    "HOLD" => pct_change.abs() < threshold,
                    _ => false,
                };
                let _ = database.resolve_signal_history(
                    prev.id, current_price, pct_change, was_correct, timestamp,
                );
            }
        } else {
            // Time stop: force-resolve signals older than 5 days
            // This prevents stale signals accumulating and caps holding period
            let signal_date = &prev.timestamp[..10]; // "2026-04-11"
            let today = &timestamp[..10];
            if signal_date < today {
                // Parse dates to check 5-day time stop
                if let (Some(sd), Some(td)) = (
                    chrono::NaiveDate::parse_from_str(signal_date, "%Y-%m-%d").ok(),
                    chrono::NaiveDate::parse_from_str(today, "%Y-%m-%d").ok(),
                ) {
                    let days_held = (td - sd).num_days();
                    if days_held >= 5 {
                        let current_price = sig.price;
                        let prev_price = prev.price_at_signal;
                        if prev_price > 0.0 {
                            let pct_change = (current_price - prev_price) / prev_price * 100.0;
                            let threshold = min_threshold_for_class(&prev.asset_class);
                            let was_correct = match prev.signal_type.as_str() {
                                "BUY" => pct_change > threshold,
                                "SELL" | "SHORT" => pct_change < -threshold,
                                "HOLD" => pct_change.abs() < threshold,
                                _ => false,
                            };
                            let _ = database.resolve_signal_history(
                                prev.id, current_price, pct_change, was_correct, timestamp,
                            );
                            println!("  [TimeStop] Force-resolved {} {} after {} days (pct={:.2}%)",
                                prev.asset, prev.signal_type, days_held, pct_change);
                        }
                    }
                }
            }
        }
    }

    // Dedup: skip if we already have an unresolved signal for this asset today
    if let Ok(Some(existing)) = database.get_last_unresolved_signal(&sig.asset) {
        let today = &timestamp[..10]; // "2026-04-11"
        if existing.timestamp.starts_with(today) {
            return; // Already have today's signal for this asset, skip duplicate
        }
    }

    // Insert the new signal
    let _ = database.insert_signal_history(
        timestamp,
        &sig.asset,
        &sig.asset_class,
        &sig.signal,
        sig.price,
        sig.technical.confidence,
        linreg_prob.unwrap_or(0.0),
        logreg_prob.unwrap_or(0.0),
        gbt_prob.unwrap_or(0.0),
    );
}

// ════════════════════════════════════════
// Auth Handlers
// ════════════════════════════════════════

async fn auth_register(
    State(state): State<AppState>,
    Json(req): Json<auth::RegisterRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let key = req.email.clone();
    if !state.rate_limiter.check(&key).await {
        return Err((StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({"error": "Too many attempts. Try again in 1 minute."}))));
    }

    let db_path = state.db_path.clone();
    let email = req.email.clone();
    let password = req.password.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
        auth::register(&conn, &email, &password)
    }).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("{}", e)}))))?;

    match result {
        Ok(resp) => Ok(Json(serde_json::json!({"token": resp.token, "user": resp.user}))),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e})))),
    }
}

async fn auth_login(
    State(state): State<AppState>,
    Json(req): Json<auth::LoginRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let key = req.email.clone();
    if !state.rate_limiter.check(&key).await {
        return Err((StatusCode::TOO_MANY_REQUESTS, Json(serde_json::json!({"error": "Too many attempts. Try again in 1 minute."}))));
    }

    let db_path = state.db_path.clone();
    let email = req.email.clone();
    let password = req.password.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
        auth::login(&conn, &email, &password)
    }).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("{}", e)}))))?;

    match result {
        Ok(resp) => Ok(Json(serde_json::json!({"token": resp.token, "user": resp.user}))),
        Err(e) => Err((StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e})))),
    }
}

async fn auth_logout() -> Json<serde_json::Value> {
    // JWT is stateless — client just discards the token
    Json(serde_json::json!({"status": "ok"}))
}

async fn auth_me(
    headers: axum::http::HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
    let token = auth::extract_token(auth_header)
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;
    let claims = auth::verify_token(&token)
        .map_err(|e| (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": e}))))?;

    Ok(Json(serde_json::json!({
        "id": claims.sub,
        "email": claims.email,
    })))
}

// ════════════════════════════════════════
// OAuth Handlers
// ════════════════════════════════════════

#[derive(serde::Deserialize)]
struct OAuthCallbackQuery {
    code: Option<String>,
    error: Option<String>,
}

async fn auth_google_redirect(
    State(state): State<AppState>,
) -> Result<axum::response::Redirect, (StatusCode, Json<serde_json::Value>)> {
    let config = state.oauth_config.as_ref()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": "OAuth not configured"}))))?;
    let url = auth::google_auth_url(config);
    Ok(axum::response::Redirect::temporary(&url))
}

async fn auth_google_callback(
    State(state): State<AppState>,
    Query(params): Query<OAuthCallbackQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    if let Some(err) = params.error {
        return axum::response::Redirect::temporary(&format!("/login?error={}", err)).into_response();
    }

    let code = match params.code {
        Some(c) => c,
        None => return axum::response::Redirect::temporary("/login?error=no_code").into_response(),
    };

    let config = match state.oauth_config.as_ref() {
        Some(c) => c,
        None => return axum::response::Redirect::temporary("/login?error=oauth_not_configured").into_response(),
    };

    match auth::exchange_google_code(&state.http_client, config, &code).await {
        Ok((email, oauth_id)) => {
            let db_path = state.db_path.clone();
            let email_c = email.clone();
            let oauth_id_c = oauth_id.clone();
            let result = tokio::task::spawn_blocking(move || {
                let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
                auth::find_or_create_oauth_user(&conn, &email_c, "google", &oauth_id_c)
            }).await;

            match result {
                Ok(Ok(resp)) => {
                    axum::response::Redirect::temporary(&format!("/auth/callback?token={}", resp.token)).into_response()
                }
                Ok(Err(e)) => {
                    axum::response::Redirect::temporary(&format!("/login?error={}", urlencoding::encode(&e))).into_response()
                }
                Err(e) => {
                    axum::response::Redirect::temporary(&format!("/login?error={}", urlencoding::encode(&e.to_string()))).into_response()
                }
            }
        }
        Err(e) => {
            axum::response::Redirect::temporary(&format!("/login?error={}", urlencoding::encode(&e))).into_response()
        }
    }
}

async fn auth_microsoft_redirect(
    State(state): State<AppState>,
) -> Result<axum::response::Redirect, (StatusCode, Json<serde_json::Value>)> {
    let config = state.oauth_config.as_ref()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": "OAuth not configured"}))))?;
    let url = auth::microsoft_auth_url(config);
    Ok(axum::response::Redirect::temporary(&url))
}

async fn auth_microsoft_callback(
    State(state): State<AppState>,
    Query(params): Query<OAuthCallbackQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    if let Some(err) = params.error {
        return axum::response::Redirect::temporary(&format!("/login?error={}", err)).into_response();
    }

    let code = match params.code {
        Some(c) => c,
        None => return axum::response::Redirect::temporary("/login?error=no_code").into_response(),
    };

    let config = match state.oauth_config.as_ref() {
        Some(c) => c,
        None => return axum::response::Redirect::temporary("/login?error=oauth_not_configured").into_response(),
    };

    match auth::exchange_microsoft_code(&state.http_client, config, &code).await {
        Ok((email, oauth_id)) => {
            let db_path = state.db_path.clone();
            let email_c = email.clone();
            let oauth_id_c = oauth_id.clone();
            let result = tokio::task::spawn_blocking(move || {
                let conn = rusqlite::Connection::open(&db_path).map_err(|e| format!("DB: {}", e))?;
                auth::find_or_create_oauth_user(&conn, &email_c, "microsoft", &oauth_id_c)
            }).await;

            match result {
                Ok(Ok(resp)) => {
                    axum::response::Redirect::temporary(&format!("/auth/callback?token={}", resp.token)).into_response()
                }
                Ok(Err(e)) => {
                    axum::response::Redirect::temporary(&format!("/login?error={}", urlencoding::encode(&e))).into_response()
                }
                Err(e) => {
                    axum::response::Redirect::temporary(&format!("/login?error={}", urlencoding::encode(&e.to_string()))).into_response()
                }
            }
        }
        Err(e) => {
            axum::response::Redirect::temporary(&format!("/login?error={}", urlencoding::encode(&e))).into_response()
        }
    }
}

/// Extract user_id from Authorization header, returns 0 if not authenticated (backwards compat)
fn extract_user_id(headers: &axum::http::HeaderMap) -> i64 {
    headers.get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| auth::extract_token(Some(h)).ok())
        .and_then(|t| auth::verify_token(&t).ok())
        .map(|c| c.sub)
        .unwrap_or(0)
}

fn run_startup_migrations(db_path: &str) {
    let database = match db::Database::new(db_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  [Migration] Cannot open DB: {}", e);
            return;
        }
    };

    // Auth tables
    if let Ok(conn) = rusqlite::Connection::open(db_path) {
        if let Err(e) = auth::create_auth_tables(&conn) {
            eprintln!("  [Migration] Auth table error: {}", e);
        }
        match auth::ensure_admin_user(&conn, "hassan@hassanshuman.co.uk", "changeme123") {
            Ok(id) => println!("  [Migration] Admin user ready (id={})", id),
            Err(e) => eprintln!("  [Migration] Admin user error: {}", e),
        }
        // Sentiment table
        if let Err(e) = news_sentiment::create_sentiment_table(&conn) {
            eprintln!("  [Migration] Sentiment table error: {}", e);
        }
    }

    // Ensure user_holdings table exists
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS user_holdings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            quantity    REAL NOT NULL,
            start_date  TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )"
    );

    // Ensure signal_history table exists
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS signal_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            asset           TEXT NOT NULL,
            asset_class     TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            price_at_signal REAL NOT NULL,
            confidence      REAL NOT NULL,
            linreg_prob     REAL,
            logreg_prob     REAL,
            gbt_prob        REAL,
            outcome_price   REAL,
            pct_change      REAL,
            was_correct     INTEGER,
            resolution_ts   TEXT
        )"
    );
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_signal_history_asset ON signal_history(asset, timestamp)");
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_signal_history_pending ON signal_history(resolution_ts)");
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_signal_history_ts ON signal_history(timestamp)");

    // Ensure predictions table exists
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            asset           TEXT NOT NULL,
            signal          TEXT NOT NULL,
            confidence      REAL NOT NULL,
            price_at_prediction REAL NOT NULL,
            actual_direction TEXT,
            was_correct     INTEGER,
            price_at_outcome REAL,
            outcome_timestamp TEXT
        )"
    );
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_predictions_asset_time ON predictions(asset, timestamp)");
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_predictions_pending ON predictions(outcome_timestamp)");

    // Ensure signal_feedback table exists (thumbs up/down on individual signals)
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS signal_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset       TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            reaction    TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )"
    );

    // Ensure survey_feedback table exists (Feedback page survey responses)
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS survey_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            q_understand    TEXT,
            q_check_daily   TEXT,
            q_trust_more    TEXT,
            q_missing       TEXT,
            q_would_pay     TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )"
    );

    // Agent tables
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS agent_actions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type     TEXT NOT NULL,
            asset           TEXT,
            trigger_reason  TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'proposed',
            accuracy_before REAL,
            accuracy_after  REAL,
            details_json    TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            executed_at     TEXT,
            evaluated_at    TEXT
        )"
    );
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_agent_actions_status ON agent_actions(status, created_at)");
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_agent_actions_asset ON agent_actions(asset, created_at)");
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS agent_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            asset           TEXT NOT NULL,
            metric_type     TEXT NOT NULL,
            value           REAL NOT NULL,
            window_days     INTEGER,
            details_json    TEXT
        )"
    );
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_agent_metrics_asset ON agent_metrics(asset, timestamp)");
    let _ = database.execute_raw("CREATE INDEX IF NOT EXISTS idx_agent_metrics_type ON agent_metrics(metric_type, timestamp)");
    let _ = database.execute_raw(
        "CREATE TABLE IF NOT EXISTS agent_config (
            key             TEXT PRIMARY KEY,
            value           TEXT NOT NULL,
            updated_at      TEXT DEFAULT (datetime('now'))
        )"
    );

    // FIX 1: Restore correct seed value (£133,993.00)
    match database.execute_raw(
        "UPDATE daily_portfolio
         SET seed_value = 133993.00,
             cumulative_return = ROUND((portfolio_value - 133993.00) / 133993.00 * 100, 2)
         WHERE seed_value != 133993.00"
    ) {
        Ok(n) if n > 0 => println!("  [Migration] Fixed seed value in {} rows", n),
        Ok(_) => {},
        Err(e) => eprintln!("  [Migration] Seed fix error: {}", e),
    }

    // FIX 2: Restore historical portfolio entries (March 6-9)
    let historical_rows = [
        ("2026-03-06", 133993.00, 153797.00, 1.09, 14.78),
        ("2026-03-07", 133993.00, 175241.00, 14.63, 30.78),
        ("2026-03-08", 133993.00, 175241.00, 0.00, 30.78),
        ("2026-03-09", 133993.00, 175241.00, 0.00, 30.78),
    ];
    for (date, seed, value, daily, cumulative) in &historical_rows {
        match database.execute_raw(&format!(
            "INSERT OR IGNORE INTO daily_portfolio \
             (date, seed_value, portfolio_value, daily_return, cumulative_return, signals_json, model_version) \
             VALUES ('{}', {}, {}, {}, {}, '[]', 1)",
            date, seed, value, daily, cumulative
        )) {
            Ok(1) => println!("  [Migration] Inserted historical row: {}", date),
            Ok(_) => {},
            Err(e) => eprintln!("  [Migration] Insert {} error: {}", date, e),
        }
    }

    // Email alert columns
    let _ = database.execute_raw("ALTER TABLE users ADD COLUMN email_alerts INTEGER DEFAULT 1");
    let _ = database.execute_raw("ALTER TABLE users ADD COLUMN last_signal_hash TEXT");
    // user_id column on user_holdings if missing
    let _ = database.execute_raw("ALTER TABLE user_holdings ADD COLUMN user_id INTEGER DEFAULT 0");
}

/// GET /api/v1/simulator/data — historical prices + signals for 10 simulator assets + SPY
/// GET /api/v1/simulator/walkforward — serve pre-computed walk-forward backtest data
async fn get_walkforward_data() -> Result<Json<serde_json::Value>, StatusCode> {
    let path = std::path::Path::new("reports/walkforward_backtest.json");
    if !path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }
    let data = std::fs::read_to_string(path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let json: serde_json::Value = serde_json::from_str(&data).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(json))
}

/// GET /api/v1/simulator/data — historical prices + signals for simulator assets + SPY
/// Descoped: stocks/ETFs only (no FX, no crypto)
async fn get_simulator_data(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Simulator assets — stocks/ETFs only (descoped FX/crypto)
        let stock_symbols = [
            "AAPL", "MSFT", "GOOGL", "JPM", "HSBA.L", "AZN.L", "XOM", "GLD", "SPY",
            "TLT", "AGG", "BND", "USO", "CPER", "BRK-B",
        ];

        let mut price_history = serde_json::Map::new();

        for sym in &stock_symbols {
            let points = database.get_stock_history(sym).unwrap_or_default();
            let arr: Vec<serde_json::Value> = points.iter().map(|p| {
                serde_json::json!({"date": &p.timestamp[..10], "price": p.price})
            }).collect();
            price_history.insert(sym.to_string(), serde_json::Value::Array(arr));
        }

        // Signal history for simulator assets (not SPY)
        let sim_assets = [
            "AAPL", "MSFT", "GOOGL", "JPM", "HSBA.L", "AZN.L", "XOM", "GLD",
            "TLT", "AGG", "BND", "USO", "CPER", "BRK-B",
        ];
        let all_signals = database.get_signal_history_all(10000).unwrap_or_default();

        let mut signal_history = serde_json::Map::new();
        for asset in &sim_assets {
            let asset_signals: Vec<serde_json::Value> = all_signals.iter()
                .filter(|s| s.asset == *asset)
                .map(|s| serde_json::json!({
                    "date": &s.timestamp[..10],
                    "signal": s.signal_type,
                    "price": s.price_at_signal,
                    "was_correct": s.was_correct,
                    "outcome_price": s.outcome_price,
                }))
                .collect();
            signal_history.insert(asset.to_string(), serde_json::Value::Array(asset_signals));
        }

        Ok::<_, StatusCode>(Json(serde_json::json!({
            "price_history": price_history,
            "signal_history": signal_history,
        })))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(result)
}

/// GET /api/v1/portfolio/managed-simulation — REMOVED (410 Gone)
/// Dynamic Universe mode has been retired. Use /api/v1/portfolio/live instead.
async fn get_managed_simulation() -> StatusCode {
    StatusCode::GONE
}

// ════════════════════════════════════════
// Deep Dive — Macro Dashboard
// ════════════════════════════════════════

#[derive(serde::Deserialize)]
struct FredResponse { observations: Vec<FredObservation> }
#[derive(serde::Deserialize)]
struct FredObservation { date: String, value: String }

async fn fetch_fred_series(client: &reqwest::Client, api_key: &str, series_id: &str) -> Option<(String, f64)> {
    let url = format!(
        "https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json&sort_order=desc&limit=1",
        series_id, api_key
    );
    let resp = client.get(&url).send().await.ok()?;
    let fred: FredResponse = resp.json().await.ok()?;
    let obs = fred.observations.first()?;
    let val = obs.value.parse::<f64>().ok()?;
    Some((obs.date.clone(), val))
}

async fn get_deep_dive(
    State(state): State<AppState>,
    Path(asset): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Determine asset class and symbols
    let (asset_class, price_symbol, signal_key) = match asset.as_str() {
        "SPY" => ("stock", "SPY", "SPY"),
        "GLD" => ("commodity", "GLD", "GLD"),
        "CL=F" | "CLF" | "USO" => ("commodity", "USO", "CL=F"),
        "bitcoin" => ("crypto", "bitcoin", "bitcoin"),
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Read current signal
    let signal_data = {
        let signals = state.signals.read().await;
        signals.get(signal_key).map(|s| serde_json::json!({
            "type": s.signal,
            "confidence": s.technical.confidence,
            "probability_up": s.technical.probability_up,
            "timestamp": s.timestamp,
            "price": s.price,
        }))
    };

    // DB queries in spawn_blocking
    let db_path = state.db_path.clone();
    let asset_str = asset.clone();
    let price_sym = price_symbol.to_string();
    let asset_cl = asset_class.to_string();

    let db_result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // 7-day accuracy
        let from_date = (chrono::Utc::now() - chrono::Duration::days(7)).format("%Y-%m-%d").to_string();
        let history = database.get_signal_history_for_asset_from(&asset_str, &from_date).unwrap_or_default();
        let resolved: Vec<_> = history.iter().filter(|s| s.was_correct.is_some()).collect();
        let correct = resolved.iter().filter(|s| s.was_correct == Some(true)).count();
        let total = resolved.len();
        let accuracy = if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 };

        // Macro indicators
        let vix = database.get_market_history("^VIX").unwrap_or_default();
        let tnx = database.get_market_history("^TNX").unwrap_or_default();
        let irx = database.get_market_history("^IRX").unwrap_or_default();
        let uup = database.get_market_history("UUP").unwrap_or_default();

        let last_val = |data: &[(String, f64)]| -> serde_json::Value {
            if let Some((ts, val)) = data.last() {
                serde_json::json!({"value": val, "timestamp": ts})
            } else {
                serde_json::Value::Null
            }
        };

        // Sector ETFs with change_pct
        let sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLP", "XLU", "XLC", "XLY", "XLI"];
        let mut sectors = serde_json::Map::new();
        for etf in &sector_etfs {
            let data = database.get_market_history(etf).unwrap_or_default();
            if data.len() >= 2 {
                let curr = data[data.len() - 1].1;
                let prev = data[data.len() - 2].1;
                let change_pct = if prev > 0.0 { (curr - prev) / prev * 100.0 } else { 0.0 };
                sectors.insert(etf.to_string(), serde_json::json!({
                    "value": curr, "timestamp": data.last().unwrap().0, "change_pct": (change_pct * 10.0).round() / 10.0
                }));
            }
        }

        // Fear & Greed
        let fg = database.get_fear_greed_history().unwrap_or_default();
        let fear_greed = fg.last().map(|(d, v)| serde_json::json!({"value": v, "date": d}));

        // Sentiment
        let conn = rusqlite::Connection::open("rust_invest.db").ok();
        let sentiment = conn.as_ref().and_then(|c| {
            let data = news_sentiment::get_sentiment_history(c, &asset_str, 7);
            data.last().map(|s| serde_json::json!({
                "news_score": s.news_score,
                "reddit_score": s.reddit_score,
                "combined_score": s.combined_score,
                "llm_analysis": s.llm_analysis,
                "date": s.date,
            }))
        });

        // 90-day price history
        let price_history: Vec<serde_json::Value> = if asset_cl == "crypto" {
            let points = database.get_coin_history(&price_sym).unwrap_or_default();
            let n = points.len();
            let skip = if n > 90 { n - 90 } else { 0 };
            points[skip..].iter().map(|p| serde_json::json!({"date": &p.timestamp[..10], "price": p.price})).collect()
        } else {
            // stock and commodity (GLD) both use stock_history
            let points = database.get_stock_history(&price_sym).unwrap_or_default();
            let n = points.len();
            let skip = if n > 90 { n - 90 } else { 0 };
            points[skip..].iter().map(|p| serde_json::json!({"date": &p.timestamp[..10], "price": p.price})).collect()
        };

        Ok::<_, StatusCode>(serde_json::json!({
            "accuracy_7d": {"correct": correct, "total": total, "accuracy": (accuracy * 10.0).round() / 10.0},
            "macro": {
                "vix": last_val(&vix),
                "tnx": last_val(&tnx),
                "irx": last_val(&irx),
                "uup": last_val(&uup),
                "sectors": sectors,
            },
            "fear_greed": fear_greed,
            "sentiment": sentiment,
            "price_history": price_history,
        }))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    // FRED API calls (async, outside spawn_blocking)
    let fred_api_key = std::env::var("FRED_API_KEY").unwrap_or_default();
    let (fed_funds, yield_curve) = if !fred_api_key.is_empty() {
        tokio::join!(
            fetch_fred_series(&state.http_client, &fred_api_key, "FEDFUNDS"),
            fetch_fred_series(&state.http_client, &fred_api_key, "T10Y2Y")
        )
    } else {
        (None, None)
    };

    let fred = serde_json::json!({
        "fed_funds_rate": fed_funds.as_ref().map(|(d, v)| serde_json::json!({"value": v, "date": d})),
        "yield_curve": yield_curve.as_ref().map(|(d, v)| serde_json::json!({"value": v, "date": d})),
    });

    // Merge everything
    let mut result = db_result.as_object().unwrap().clone();
    result.insert("asset".to_string(), serde_json::json!(asset));
    result.insert("asset_class".to_string(), serde_json::json!(asset_class));
    result.insert("signal".to_string(), signal_data.unwrap_or(serde_json::Value::Null));
    result.insert("fred".to_string(), fred);

    Ok(Json(serde_json::Value::Object(result)))
}

// ════════════════════════════════════════
// Agent API Handlers (PostgreSQL)
// ════════════════════════════════════════

/// GET /api/v1/agent/status — Agent state + last run info
async fn get_agent_status(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let pool = &state.pg_pool;
    let agent_state: agent::AgentState = pg::get_agent_config(pool, "agent_state").await
        .ok()
        .flatten()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let config = agent::AgentConfig::load_from_pg(pool).await;
    Json(serde_json::json!({
        "state": agent_state,
        "config": config,
    }))
}

/// GET /api/v1/agent/actions — Action log with filtering and outcomes
async fn get_agent_actions(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit: i64 = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(500);
    let status_filter = params.get("status").cloned();
    let show_historical = params.get("historical").map(|v| v == "true").unwrap_or(false);

    let all_actions = pg::get_agent_actions(&state.pg_pool, limit, status_filter.as_deref()).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let now = chrono::Utc::now();
    let cutoff = now - chrono::Duration::hours(48);
    let cutoff_str = cutoff.to_rfc3339();

    let actions: Vec<serde_json::Value> = all_actions.into_iter().filter(|a| {
        if let Some(s) = &a.asset {
            if s.contains('=') || ["bitcoin","ethereum","solana","dogecoin","cardano","ripple",
                "polkadot","avalanche","chainlink","litecoin","uniswap","stellar","monero","aave","cosmos","tron"].contains(&s.as_str()) {
                return false;
            }
        }
        if show_historical { return true; }
        let is_resolved = matches!(a.status.as_str(), "evaluated" | "rolled_back" | "approved" | "rejected");
        if is_resolved { return true; }
        match &a.created_at {
            Some(ts) => ts.as_str() >= cutoff_str.as_str(),
            None => false,
        }
    }).map(|a| {
        let outcome = match a.status.as_str() {
            "evaluated" => {
                match (a.accuracy_before, a.accuracy_after) {
                    (Some(before), Some(after)) => {
                        let delta = after - before;
                        if delta > 1.0 { format!("improved +{:.1}pp", delta) }
                        else if delta < -1.0 { format!("degraded {:.1}pp", delta) }
                        else { "no significant change".to_string() }
                    }
                    _ => "evaluated (no accuracy data)".to_string(),
                }
            }
            "rolled_back" => "rolled back — accuracy degraded".to_string(),
            "approved" => "manually approved".to_string(),
            "rejected" => "manually rejected".to_string(),
            "proposed" => "pending approval".to_string(),
            "executed" => "awaiting evaluation".to_string(),
            s => s.to_string(),
        };
        let mut v = serde_json::to_value(&a).unwrap_or_default();
        v["outcome"] = serde_json::json!(outcome);
        v
    }).collect();

    Ok(Json(serde_json::json!({
        "actions": actions,
        "total": actions.len(),
    })))
}

/// GET /api/v1/agent/metrics — Accuracy time-series for dashboard
async fn get_agent_metrics(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit: i64 = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(2000);
    let asset = params.get("asset").cloned();
    let metric_type = params.get("type").cloned();

    let all_metrics = pg::get_agent_metrics(&state.pg_pool, asset.as_deref(), metric_type.as_deref(), limit).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let metrics: Vec<_> = all_metrics.into_iter().filter(|m| {
        let a = &m.asset;
        !a.contains('=') && !["bitcoin","ethereum","solana","dogecoin","cardano","ripple",
         "polkadot","avalanche","chainlink","litecoin","uniswap","stellar","monero","aave","cosmos","tron"].contains(&a.as_str())
    }).collect();

    Ok(Json(serde_json::json!({
        "metrics": metrics,
        "total": metrics.len(),
    })))
}

/// POST /api/v1/agent/approve/:action_id
async fn approve_agent_action(
    State(state): State<AppState>,
    Path(action_id): Path<i64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    pg::update_agent_action_status(&state.pg_pool, action_id, "approved", None).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({"status": "approved", "action_id": action_id})))
}

/// POST /api/v1/agent/reject/:action_id
async fn reject_agent_action(
    State(state): State<AppState>,
    Path(action_id): Path<i64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    pg::update_agent_action_status(&state.pg_pool, action_id, "rejected", None).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({"status": "rejected", "action_id": action_id})))
}

/// GET /api/v1/agent/config
async fn get_agent_config(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let config = agent::AgentConfig::load_from_pg(&state.pg_pool).await;
    let val = serde_json::to_value(config).unwrap_or_default();
    Ok(Json(val))
}

/// PATCH /api/v1/agent/config
async fn update_agent_config(
    State(state): State<AppState>,
    Json(updates): Json<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    for (key, value) in &updates {
        pg::set_agent_config(&state.pg_pool, key, value).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    let config = agent::AgentConfig::load_from_pg(&state.pg_pool).await;
    let val = serde_json::to_value(config).unwrap_or_default();
    Ok(Json(val))
}

/// GET /api/v1/agent/summary — Aggregated agent stats for dashboard
async fn get_agent_summary(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let summary = pg::get_agent_summary(&state.pg_pool).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(summary))
}

/// GET /api/v1/agents/fleet — Fleet status for all 6 agents
async fn get_fleet_status_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let fleet = pg::get_fleet_status(&state.pg_pool).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(fleet))
}

/// GET /api/v1/agents/activity — Recent fleet activity feed
async fn get_fleet_activity_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit: i64 = params.get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let activities = pg::get_fleet_activity(&state.pg_pool, limit).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({ "activities": activities })))
}

/// GET /api/v1/portfolio/ftse — FTSE equity curve from PG signals + ISF.L benchmark
async fn get_portfolio_ftse(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Get all resolved .L signals from PG
    let signals = pg::get_resolved_signals_by_suffix(&state.pg_pool, ".L").await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Get ISF.L benchmark from SQLite
    let db_path = state.db_path.clone();
    let isf_prices = tokio::task::spawn_blocking(move || -> Vec<(String, f64)> {
        let database = match db::Database::new(&db_path) {
            Ok(d) => d,
            Err(_) => return vec![],
        };
        database.get_stock_history("ISF.L")
            .unwrap_or_default()
            .into_iter()
            .map(|p| (p.timestamp, p.price))
            .collect()
    }).await.unwrap_or_default();

    if signals.is_empty() {
        return Ok(Json(serde_json::json!({
            "error": "No FTSE signals found",
            "signals": [],
            "equity_curve": [],
            "benchmark": [],
        })));
    }

    // Build daily equity curve from signals
    let starting_capital = 100_000.0;
    let mut capital = starting_capital;
    let mut equity_curve: Vec<serde_json::Value> = Vec::new();

    // Group signals by date
    let mut daily_signals: std::collections::BTreeMap<String, Vec<&pg::SignalRow>> = std::collections::BTreeMap::new();
    for sig in &signals {
        let date = sig.timestamp.split('T').next().unwrap_or(&sig.timestamp).to_string();
        daily_signals.entry(date).or_default().push(sig);
    }

    for (date, day_sigs) in &daily_signals {
        let n = day_sigs.len() as f64;
        if n == 0.0 { continue; }
        let daily_return: f64 = day_sigs.iter().filter_map(|s| {
            let pct = s.pct_change?;
            match s.signal_type.as_str() {
                "BUY" => Some(pct / 100.0),
                "SELL" | "SHORT" => Some(-pct / 100.0),
                _ => None,
            }
        }).sum::<f64>() / n;
        capital *= 1.0 + daily_return;
        equity_curve.push(serde_json::json!({
            "date": date,
            "value": (capital * 100.0).round() / 100.0,
        }));
    }

    // Normalize ISF.L benchmark to same starting capital, aligned to signal period
    let first_signal_date = daily_signals.keys().next().cloned().unwrap_or_default();
    let benchmark: Vec<serde_json::Value> = if !isf_prices.is_empty() && !first_signal_date.is_empty() {
        // Find the ISF.L price on or just before the first signal date
        let anchor_price = isf_prices.iter()
            .filter(|(d, _)| d.as_str() <= first_signal_date.as_str())
            .last()
            .map(|(_, p)| *p)
            .unwrap_or(isf_prices[0].1);
        // Only include dates from the first signal date onward
        isf_prices.iter()
            .filter(|(d, _)| d.as_str() >= first_signal_date.as_str())
            .map(|(d, p)| {
                serde_json::json!({
                    "date": d,
                    "value": (starting_capital * p / anchor_price * 100.0).round() / 100.0,
                })
            }).collect()
    } else {
        vec![]
    };

    // Summary stats
    let total_signals = signals.len();
    let ftse_assets: std::collections::HashSet<&str> = signals.iter().map(|s| s.asset.as_str()).collect();
    let correct = signals.iter().filter(|s| s.was_correct == Some(true)).count();
    let accuracy = if total_signals > 0 { 100.0 * correct as f64 / total_signals as f64 } else { 0.0 };
    let total_return = if starting_capital > 0.0 { (capital / starting_capital - 1.0) * 100.0 } else { 0.0 };

    Ok(Json(serde_json::json!({
        "starting_capital": starting_capital,
        "current_value": (capital * 100.0).round() / 100.0,
        "total_return_pct": (total_return * 100.0).round() / 100.0,
        "total_signals": total_signals,
        "accuracy": (accuracy * 100.0).round() / 100.0,
        "ftse_assets": ftse_assets.len(),
        "equity_curve": equity_curve,
        "benchmark": benchmark,
    })))
}

// ════════════════════════════════════════
// System Health / Data Quality endpoint
// ════════════════════════════════════════

async fn get_system_health(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let dq = state.data_quality.read().await;
    let signal_count = state.signals.read().await.len();

    // Check PG pool health
    let pg_healthy = match state.pg_pool.get().await {
        Ok(_) => true,
        Err(_) => false,
    };

    // Determine overall status
    let total_writes = dq.pg_write_successes + dq.pg_write_failures;
    let failure_rate = if total_writes > 0 {
        (dq.pg_write_failures as f64 / total_writes as f64) * 100.0
    } else { 0.0 };

    let status = if !pg_healthy {
        "critical"
    } else if failure_rate > 50.0 {
        "critical"
    } else if failure_rate > 10.0 {
        "warning"
    } else {
        "healthy"
    };

    Json(serde_json::json!({
        "status": status,
        "timestamp": Utc::now().to_rfc3339(),
        "postgres": {
            "connected": pg_healthy,
            "write_successes": dq.pg_write_successes,
            "write_failures": dq.pg_write_failures,
            "failure_rate_pct": format!("{:.1}", failure_rate),
            "last_failure": dq.last_pg_failure,
            "last_failure_error": dq.last_pg_failure_error,
            "last_successful_write": dq.last_successful_signal_write,
        },
        "signals": {
            "cached_count": signal_count,
            "last_generation": dq.last_signal_generation,
        },
        "uptime": {
            "server": "running",
        }
    }))
}

/// Build a default asset config from the existing STOCK_LIST/FX_LIST
fn default_asset_config() -> config::AssetConfig {
    config::AssetConfig {
        stocks: stocks::STOCK_LIST.iter().map(|s| config::AssetEntry {
            symbol: s.symbol.to_string(),
            name: s.name.to_string(),
            enabled: true,
            tags: Vec::new(),
        }).collect(),
        fx: stocks::FX_LIST.iter().map(|s| config::AssetEntry {
            symbol: s.symbol.to_string(),
            name: s.name.to_string(),
            enabled: true,
            tags: Vec::new(),
        }).collect(),
        crypto: Vec::new(),
    }
}
