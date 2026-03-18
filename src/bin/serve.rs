/// serve — Web API server for Rust Invest
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
    routing::{get, post, patch, put},
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

#[derive(Clone)]
struct AppState {
    signals: Arc<RwLock<HashMap<String, enriched_signals::EnrichedSignal>>>,
    asset_config: Arc<RwLock<config::AssetConfig>>,
    db_path: String,
    llm_provider: Option<llm::LlmProvider>,
    http_client: reqwest::Client,
}

// ════════════════════════════════════════
// Main
// ════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         RUST INVEST — SERVE MODE (Web API Server)              ║");
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

    // Load LLM provider
    let llm_provider = llm::load_provider();
    match &llm_provider {
        Some(llm::LlmProvider::Ollama { base_url, model }) =>
            println!("  LLM: Ollama ({}) at {}", model, base_url),
        Some(llm::LlmProvider::Anthropic { model, .. }) =>
            println!("  LLM: Anthropic ({})", model),
        None =>
            println!("  LLM: Not configured (set LLM_PROVIDER in .env)"),
    }

    let state = AppState {
        signals: Arc::new(RwLock::new(HashMap::new())),
        asset_config: Arc::new(RwLock::new(asset_config)),
        db_path: "rust_invest.db".to_string(),
        llm_provider,
        http_client: reqwest::Client::new(),
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

            // Resolve pending predictions with current prices
            {
                let sigs = scheduler_state.signals.read().await;
                let signals_map = sigs.clone();
                drop(sigs);
                let db_path = scheduler_state.db_path.clone();
                tokio::task::spawn_blocking(move || {
                    resolve_pending_predictions(&db_path, &signals_map);
                }).await.ok();
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
        .route("/api/v1/signals/current", get(get_all_signals))
        .route("/api/v1/signals/current/stocks", get(get_stock_signals))
        .route("/api/v1/signals/current/fx", get(get_fx_signals))
        .route("/api/v1/signals/current/crypto", get(get_crypto_signals))
        .route("/api/v1/signals/history/{asset}", get(get_signal_history))
        .route("/api/v1/portfolio/simulate", get(simulate_portfolio))
        .route("/api/v1/portfolio/daily-tracker", get(get_daily_tracker))
        .route("/api/v1/history/portfolio", get(get_portfolio_history))
        .route("/api/v1/history/signals", get(get_signals_history))
        .route("/api/v1/hints", get(get_hints))
        .route("/api/v1/simulate", post(simulate_signals))
        .route("/api/v1/training/results", get(get_training_results))
        .route("/api/v1/chat", post(chat_handler))
        .route("/api/v1/admin/assets", post(add_asset))
        .route("/api/v1/admin/assets/{symbol}", patch(toggle_asset))
        .route("/api/v1/predictions/history", get(get_predictions_history))
        .route("/api/v1/signals/truth", get(get_signal_truth))
        .route("/api/v1/signals/truth/historical", get(get_signal_truth_historical))
        .route("/api/v1/user-portfolio", get(get_user_holdings))
        .route("/api/v1/user-portfolio", post(add_user_holding))
        .route("/api/v1/user-portfolio/compare", post(compare_portfolio))
        .route("/api/v1/user-portfolio/{id}", put(update_user_holding).delete(delete_user_holding))
        .route("/api/v1/feedback/signal", post(submit_signal_feedback))
        .route("/api/v1/feedback/survey", post(submit_survey_feedback))
        .route("/api/v1/feedback", get(get_feedback))
        .layer(cors.clone())
        .with_state(state);

    // Nest static file serving outside the API router so CORS doesn't interfere
    let app = if serve_frontend {
        let index = frontend_dist.join("index.html");
        Router::new()
            .nest_service("/assets", ServeDir::new(frontend_dist.join("assets")))
            .route_service("/", ServeFile::new(&index))
            .fallback_service(ServeFile::new(&index))
            .merge(app)
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
    println!("    GET  /api/v1/predictions/history\n");

    axum::serve(listener, app).await?;
    Ok(())
}

// ════════════════════════════════════════
// API Handlers
// ════════════════════════════════════════

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "timestamp": Utc::now().to_rfc3339()}))
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
async fn get_signal_truth(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let limit: usize = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(1000);

    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let records = database.get_signal_history_all(limit).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
        let signal_type_accuracy: Vec<serde_json::Value> = ["BUY", "SELL", "HOLD"].iter().map(|&st| {
            let (correct, total) = by_signal.get(st).copied().unwrap_or((0, 0));
            serde_json::json!({
                "signal_type": st,
                "correct": correct,
                "total": total,
                "accuracy": if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 },
            })
        }).collect();

        // ── Accuracy by asset class ──
        let mut by_class: HashMap<String, (usize, usize)> = HashMap::new();
        for r in &resolved {
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

        // ── Rolling accuracy (today, this week, all time) ──
        let now = Utc::now();
        let today_start = now.format("%Y-%m-%d").to_string();
        let week_ago = (now - chrono::Duration::days(7)).to_rfc3339();

        let today_resolved: Vec<_> = resolved.iter().filter(|r| r.timestamp.starts_with(&today_start)).collect();
        let today_correct = today_resolved.iter().filter(|r| r.was_correct == Some(true)).count();

        let week_resolved: Vec<_> = resolved.iter().filter(|r| r.timestamp.as_str() >= week_ago.as_str()).collect();
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

        Ok::<_, StatusCode>(Json(serde_json::json!({
            "total_signals": total,
            "total_resolved": total_resolved,
            "total_pending": pending_count,
            "total_correct": correct_count,
            "overall_accuracy": overall_accuracy,
            "by_signal_type": signal_type_accuracy,
            "by_asset_class": asset_class_accuracy,
            "rolling": rolling,
            "per_asset": per_asset_vec,
            "signals": signals,
        })))
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(result)
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
                            "SELL" => price_down,
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
    let signal_type_accuracy: Vec<serde_json::Value> = ["BUY", "SELL", "HOLD"].iter().map(|&st| {
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

async fn get_all_signals(
    State(state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    let sigs = state.signals.read().await;
    let mut signals: Vec<_> = sigs.values().cloned().collect();
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

async fn get_fx_signals(
    State(state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    let sigs = state.signals.read().await;
    let mut signals: Vec<_> = sigs.values()
        .filter(|s| s.asset_class == "fx")
        .cloned()
        .collect();
    signals.sort_by(|a, b| a.asset.cmp(&b.asset));
    Json(signals)
}

async fn get_crypto_signals(
    State(state): State<AppState>,
) -> Json<Vec<enriched_signals::EnrichedSignal>> {
    let sigs = state.signals.read().await;
    let mut signals: Vec<_> = sigs.values()
        .filter(|s| s.asset_class == "crypto")
        .cloned()
        .collect();
    signals.sort_by(|a, b| a.asset.cmp(&b.asset));
    Json(signals)
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

async fn get_portfolio_history(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let database = db::Database::new(&state.db_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let rows = database.get_daily_portfolio(90).unwrap_or_default();

    if rows.is_empty() {
        return Ok(Json(serde_json::json!({
            "has_data": false,
            "note": "No portfolio history yet. Data accumulates hourly once serve is running."
        })));
    }

    // Reverse to chronological order for charting
    let mut rows = rows;
    rows.reverse();

    let seed = rows.first().map(|r| r.seed_value).unwrap_or(0.0);
    let latest = rows.last().unwrap();

    let points: Vec<serde_json::Value> = rows.iter().map(|r| {
        serde_json::json!({
            "date": r.date,
            "value": r.portfolio_value,
            "daily_return": r.daily_return,
            "cumulative_return": r.cumulative_return,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "has_data": true,
        "seed_value": seed,
        "current_value": latest.portfolio_value,
        "cumulative_return": latest.cumulative_return,
        "days": rows.len(),
        "points": points,
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
    let provider = match &state.llm_provider {
        Some(p) => p,
        None => {
            return Ok(Json(serde_json::json!({
                "response": "LLM not configured. Install Ollama or set LLM_PROVIDER in .env"
            })));
        }
    };

    let tab_context = req.tab_context.unwrap_or_else(|| "overview".to_string());
    let is_morning_briefing = req.message.trim() == "morning_briefing";

    // Build portfolio context from daily tracker
    let portfolio_context = {
        let db_path = state.db_path.clone();
        tokio::task::spawn_blocking(move || {
            daily_tracker::build_api_response(&db_path)
        }).await.unwrap_or_else(|_| serde_json::json!({"has_data": false}))
    };

    // Build signals context (shared between briefing and general chat)
    let (signals_table, portfolio_summary, accuracy_summary) = {
        let sigs = state.signals.read().await;
        let relevant: Vec<_> = match tab_context.as_str() {
            "stocks" => sigs.values().filter(|s| s.asset_class == "stock").cloned().collect(),
            "fx" => sigs.values().filter(|s| s.asset_class == "fx").cloned().collect(),
            "crypto" => sigs.values().filter(|s| s.asset_class == "crypto").cloned().collect(),
            _ => sigs.values().cloned().collect(),
        };

        // Format signals table
        let mut signals_table = String::from("Asset | Signal | Confidence | Prob Up | RSI | Price | Quality\n");
        for s in &relevant {
            signals_table.push_str(&format!(
                "{} | {} | {:.1}/10 | {:.1}% | {:.0} | {:.2} | {}\n",
                s.asset, s.signal, s.technical.confidence,
                s.technical.probability_up, s.technical.rsi,
                s.price, s.technical.quality,
            ));
        }

        // Format portfolio summary
        let mut portfolio_summary = String::new();
        if portfolio_context["has_data"].as_bool().unwrap_or(false) {
            let value = portfolio_context["current_value"].as_f64().unwrap_or(0.0);
            let daily_ret = portfolio_context["daily_return"].as_f64().unwrap_or(0.0);
            let cum_ret = portfolio_context["cumulative_return"].as_f64().unwrap_or(0.0);
            let accuracy = portfolio_context["model_accuracy_pct"].as_f64().unwrap_or(0.0);
            let seed = portfolio_context["seed_value"].as_f64().unwrap_or(100_000.0);

            portfolio_summary.push_str(&format!(
                "Portfolio value: £{:.2} (seed: £{:.0})\n\
                 Daily return: {:.2}%\n\
                 Cumulative return: {:.2}%\n\
                 Model accuracy: {:.1}%\n",
                value, seed, daily_ret, cum_ret, accuracy,
            ));

            // Asset allocations
            if let Some(today_sigs) = portfolio_context["today_signals"].as_array() {
                portfolio_summary.push_str("\nAsset allocations:\n");
                portfolio_summary.push_str("Asset | Weight | Signal | Daily Return | Contribution\n");
                for ts in today_sigs {
                    portfolio_summary.push_str(&format!(
                        "{} | {:.1}% | {} | {:.2}% | {:.3}%\n",
                        ts["asset"].as_str().unwrap_or("?"),
                        ts["weight"].as_f64().unwrap_or(0.0),
                        ts["signal"].as_str().unwrap_or("?"),
                        ts["price_return"].as_f64().unwrap_or(0.0),
                        ts["contribution"].as_f64().unwrap_or(0.0),
                    ));
                }
            }
        } else {
            portfolio_summary.push_str("Portfolio tracking has not started yet — no historical data available.\n");
        }

        // Per-asset model accuracy from signals
        let mut accuracy_summary = String::new();
        for s in &relevant {
            accuracy_summary.push_str(&format!(
                "{}: walk-forward accuracy {:.1}%, agreement {}\n",
                s.asset, s.technical.walk_forward_accuracy, s.technical.model_agreement,
            ));
        }

        (signals_table, portfolio_summary, accuracy_summary)
    };

    // Build system prompt — specialised for morning briefing vs generic chat
    let system_prompt = if is_morning_briefing {
        format!(
            "You are a concise financial morning briefing generator for Rust Invest. \
             Given today's signals, generate a 3-paragraph briefing in plain English \
             (no jargon without explanation):\n\
             Para 1: What markets are doing today in one sentence, then the overall mood \
             (calm/cautious/fearful) based on signal distribution and any VIX data available.\n\
             Para 2: The 2-3 strongest signals today and what they mean for someone with money to invest.\n\
             Para 3: One 'watch out' — the biggest risk or caution signal visible in today's data.\n\
             Keep total response under 150 words. No bullet points. \
             Write as if speaking to someone new to investing.\n\n\
             === TODAY'S SIGNALS ===\n{}\n\
             === PORTFOLIO ===\n{}\n\
             === MODEL ACCURACY ===\n{}",
            signals_table, portfolio_summary, accuracy_summary,
        )
    } else {
        format!(
            "You are an AI analyst for a quantitative investment system (Rust_Invest).\n\
             Here is the current portfolio state:\n\n\
             === PORTFOLIO ===\n{}\n\
             === TODAY'S SIGNALS ===\n{}\n\
             === MODEL ACCURACY ===\n{}\n\
             Answer questions about this specific portfolio only.\n\
             Be concise and specific with numbers.\n\
             Never give financial advice. Always note past performance doesn't guarantee future results.\n\
             Explain technical concepts in plain language when you use them.",
            portfolio_summary, signals_table, accuracy_summary,
        )
    };

    let user_message = if is_morning_briefing {
        "Generate today's morning briefing."
    } else {
        &req.message
    };

    match llm::chat(&state.http_client, provider, &system_prompt, user_message).await {
        Ok(response) => Ok(Json(serde_json::json!({ "response": response }))),
        Err(e) => Ok(Json(serde_json::json!({ "response": format!("Error: {}", e) }))),
    }
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

    let new_signals = tokio::task::spawn_blocking(move || {
        generate_all_signals(&db_path, &asset_config)
    }).await??;

    // Store snapshots + signal history in DB
    {
        let database = db::Database::new(&state.db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        let ts = Utc::now().to_rfc3339();
        for sig in &new_signals {
            let _ = database.insert_signal_snapshot(sig, 3);
            record_signal_history(&database, sig, &ts);
        }
    }

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

    // US stocks: Mon-Fri, 14:30-21:00 UTC
    let us_market_open = !is_weekend && hour >= 14 && hour <= 21;

    // FX: Sun 22:00 UTC to Fri 22:00 UTC (nearly 24/5)
    let fx_open = match weekday {
        Weekday::Sat => false,
        Weekday::Sun => hour >= 22,
        Weekday::Fri => hour < 22,
        _ => true,
    };

    // Crypto: 24/7
    let crypto_open = true;

    // Fetch live prices first so inference uses current data
    if let Err(e) = fetch_and_store_live_prices(state, us_market_open, fx_open, crypto_open).await {
        eprintln!("  [LivePrice] Warning: {}", e);
    }

    let db_path = state.db_path.clone();
    let asset_config = state.asset_config.read().await.clone();

    let new_signals = tokio::task::spawn_blocking(move || {
        generate_signals_filtered(&db_path, &asset_config, us_market_open, fx_open, crypto_open)
    }).await??;

    // Store snapshots + signal history
    {
        let database = db::Database::new(&state.db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        let ts = Utc::now().to_rfc3339();
        for sig in &new_signals {
            let _ = database.insert_signal_snapshot(sig, 3);
            record_signal_history(&database, sig, &ts);
        }
    }

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
) -> Result<Vec<enriched_signals::EnrichedSignal>, Box<dyn std::error::Error + Send + Sync>> {
    generate_signals_filtered(db_path, asset_config, true, true, true)
}

/// Generate signals with market hours filtering — inference only, no training
fn generate_signals_filtered(
    db_path: &str,
    asset_config: &config::AssetConfig,
    include_stocks: bool,
    include_fx: bool,
    include_crypto: bool,
) -> Result<Vec<enriched_signals::EnrichedSignal>, Box<dyn std::error::Error + Send + Sync>> {
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
    let market_context = features::build_market_context(&market_histories);

    // Load Fear & Greed history once — used by stocks, FX, and crypto
    let fear_greed_history = database.get_fear_greed_history().unwrap_or_default();
    let fg_ref: Option<&[(String, f64)]> = if fear_greed_history.is_empty() {
        None
    } else {
        Some(&fear_greed_history)
    };

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

    // ── FX signals (inference only) ──
    if include_fx {
        let enabled_fx = asset_config.enabled_fx();
        for asset_entry in &enabled_fx {
            let points = match database.get_fx_history(&asset_entry.symbol) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if points.len() < 300 { continue; }

            if let Some(sig) = infer_fx_signal(&asset_entry.symbol, &points, &market_context, fg_ref) {
                enriched_signals.push(sig);
            }
        }
    }

    // ── Crypto signals (inference only) ──
    if include_crypto {
        let enabled_crypto = asset_config.enabled_crypto();
        if !enabled_crypto.is_empty() {
            let coin_ids: Vec<String> = database.get_all_coin_ids()
                .unwrap_or_default().into_iter().filter(|id| id != "tether").collect();

            let mut crypto_prices_map: HashMap<String, Vec<f64>> = HashMap::new();
            let mut crypto_returns_map: HashMap<String, Vec<f64>> = HashMap::new();
            let mut crypto_dates_map: HashMap<String, Vec<String>> = HashMap::new();
            for coin_id in &coin_ids {
                let points = match database.get_coin_history(coin_id) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if points.len() < 60 { continue; }
                let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
                let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                let dates: Vec<String> = points.iter().map(|p| p.timestamp[..10].to_string()).collect();
                crypto_prices_map.insert(coin_id.clone(), prices);
                crypto_returns_map.insert(coin_id.clone(), returns);
                crypto_dates_map.insert(coin_id.clone(), dates);
            }
            let crypto_syms: Vec<&str> = coin_ids.iter().map(|s| s.as_str()).collect();
            let crypto_enrichment = crypto_features::enrich_crypto_features(
                &crypto_syms, &crypto_prices_map, &crypto_returns_map, &crypto_dates_map,
            );

            for asset_entry in &enabled_crypto {
                if let Some(sig) = infer_crypto_signal(
                    &asset_entry.symbol, &database, &crypto_enrichment,
                ) {
                    enriched_signals.push(sig);
                }
            }
        }
    }

    println!("  Generated {} enriched signals (inference only)", enriched_signals.len());
    Ok(enriched_signals)
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

    let samples = features::build_rich_features(
        &prices, &volumes, &timestamps,
        Some(market_context), "stock",
        features::sector_etf_for(symbol),
        None, fear_greed,
    );
    if samples.is_empty() { return None; }

    let wf = inference::infer_with_saved_models(symbol, &samples)?;

    let result = analysis::analyse_coin(symbol, points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend);

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

    let samples = features::build_rich_features(
        &prices, &volumes, &timestamps,
        Some(market_context), "fx",
        Some(symbol), None, fear_greed,
    );
    if samples.is_empty() { return None; }

    let wf = inference::infer_with_saved_models(symbol, &samples)?;

    let result = analysis::analyse_coin(symbol, points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend);

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

    let wf = inference::infer_with_saved_models(coin_id, &enriched_samples)?;

    let result = analysis::analyse_coin(coin_id, &points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(coin_id, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend);

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
// Startup Database Migrations
// ════════════════════════════════════════

/// Resolve pending predictions by comparing prediction price to current price.
/// Uses market-hours-aware resolution logic (same rules as signal_history).
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
        // Determine asset class from the current signals cache
        let sig = match signals.get(&pred.asset) {
            Some(s) => s,
            None => continue,
        };
        let asset_class = sig.asset_class.as_str();

        // Apply market-hours-aware resolution rules
        if !can_resolve_signal(asset_class, &pred.signal, &pred.timestamp, now) {
            continue;
        }

        let current_price = sig.price;
        let price_change = current_price - pred.price_at_prediction;
        let actual_direction = if price_change.abs() < pred.price_at_prediction * 0.001 {
            "FLAT"
        } else if price_change > 0.0 {
            "UP"
        } else {
            "DOWN"
        };

        let was_correct = match (pred.signal.as_str(), actual_direction) {
            ("BUY", "UP") | ("SELL", "DOWN") => true,
            ("BUY", "DOWN") | ("SELL", "UP") => false,
            _ => true,
        };

        let _ = database.update_prediction_outcome(pred.id, actual_direction, was_correct, current_price, &resolve_ts);
        resolved += 1;
    }

    if resolved > 0 {
        println!("  [Predictions] Resolved {} of {} pending predictions", resolved, pending.len());
    }
}

// ════════════════════════════════════════
// User Portfolio Tracker
// ════════════════════════════════════════

async fn get_user_holdings(
    State(state): State<AppState>,
) -> Result<Json<Vec<db::UserHolding>>, StatusCode> {
    let db_path = state.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.get_user_holdings().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
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
    Json(req): Json<AddHoldingRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let symbol = req.symbol.trim().to_string();
    let asset_class = detect_asset_class(&symbol);
    let db_path = state.db_path.clone();
    let start_date = req.start_date.clone();
    let quantity = req.quantity;

    let id = tokio::task::spawn_blocking(move || {
        let database = db::Database::new(&db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        database.insert_user_holding(&symbol, quantity, &start_date, &asset_class)
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

/// Check if a symbol is a known CoinGecko crypto ID
fn is_crypto_id(s: &str) -> bool {
    matches!(s, "bitcoin" | "ethereum" | "solana" | "ripple" | "dogecoin"
        | "cardano" | "avalanche-2" | "chainlink" | "polkadot" | "near"
        | "sui" | "aptos" | "arbitrum" | "the-open-network" | "uniswap"
        | "tron" | "litecoin" | "shiba-inu" | "stellar" | "matic-network")
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

    // 3. Only show note if actual date differs by more than 7 calendar days
    let note = if actual_start_date.as_str() != holding_start {
        let requested = chrono::NaiveDate::parse_from_str(holding_start, "%Y-%m-%d").ok();
        let actual = chrono::NaiveDate::parse_from_str(&actual_start_date, "%Y-%m-%d").ok();
        match (requested, actual) {
            (Some(req), Some(act)) if (act - req).num_days().abs() > 7 => {
                Some(format!("Nearest trading date: {} ({} days from {})",
                    actual_start_date, (act - req).num_days().abs(), holding_start))
            }
            _ => None, // Silently use nearest trading day for small gaps (weekends/holidays)
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

    // 6. Generate ALL signals at once using bulk method (builds features once, O(n) not O(n²))
    let models = simulator::load_models_for_symbol(&holding.symbol);
    if models.is_none() {
        // No trained models — return buy-and-hold only (no signal loop)
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
            note: Some("No trained models found — showing buy & hold only".to_string()),
        });
    }
    let signal_map = simulator::generate_signals_bulk(
        &holding.symbol, &holding.asset_class,
        &all_prices, market_context, models.as_ref().unwrap(),
    );

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
                    cash = shares * price;
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
                        shares = cash / price;
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
        equity_curve, note,
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
    let linreg_prob = sig.models.get("linreg").map(|m| m.probability_up);
    let logreg_prob = sig.models.get("logreg").map(|m| m.probability_up);
    let gbt_prob = sig.models.get("gbt").map(|m| m.probability_up);

    // Resolve previous unresolved signal for this asset (market-hours aware)
    if let Ok(Some(prev)) = database.get_last_unresolved_signal(&sig.asset) {
        let now = Utc::now();
        if can_resolve_signal(&prev.asset_class, &prev.signal_type, &prev.timestamp, now) {
            let current_price = sig.price;
            let prev_price = prev.price_at_signal;
            if prev_price > 0.0 {
                let pct_change = (current_price - prev_price) / prev_price * 100.0;
                let was_correct = match prev.signal_type.as_str() {
                    "BUY" => current_price > prev_price,
                    "SELL" => current_price < prev_price,
                    "HOLD" => pct_change.abs() < 1.0,
                    _ => false,
                };
                let _ = database.resolve_signal_history(
                    prev.id, current_price, pct_change, was_correct, timestamp,
                );
            }
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

fn run_startup_migrations(db_path: &str) {
    let database = match db::Database::new(db_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  [Migration] Cannot open DB: {}", e);
            return;
        }
    };

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
}

/// Build a default asset config from the existing STOCK_LIST/FX_LIST
fn default_asset_config() -> config::AssetConfig {
    config::AssetConfig {
        stocks: stocks::STOCK_LIST.iter().map(|s| config::AssetEntry {
            symbol: s.symbol.to_string(),
            name: s.name.to_string(),
            enabled: true,
        }).collect(),
        fx: stocks::FX_LIST.iter().map(|s| config::AssetEntry {
            symbol: s.symbol.to_string(),
            name: s.name.to_string(),
            enabled: true,
        }).collect(),
        crypto: Vec::new(),
    }
}
