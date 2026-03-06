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
    routing::{get, post},
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
        .route("/api/v1/chat", post(chat_handler))
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

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("\n  Server listening on http://0.0.0.0:8080");
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
    println!("    POST /api/v1/chat\n");

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

    // Build system prompt with current signal data as context
    let signals_context = {
        let sigs = state.signals.read().await;
        let relevant: Vec<_> = match tab_context.as_str() {
            "stocks" => sigs.values().filter(|s| s.asset_class == "stock").cloned().collect(),
            "fx" => sigs.values().filter(|s| s.asset_class == "fx").cloned().collect(),
            _ => sigs.values().cloned().collect(),
        };
        serde_json::to_string_pretty(&relevant).unwrap_or_else(|_| "[]".to_string())
    };

    let system_prompt = format!(
        "You are the AI analyst for Rust_Invest, an AI investment copilot.\n\
        Your role is decision support — helping users understand risk and make informed decisions.\n\
        You are NOT a prediction engine. You explain what the models show, assess risk, and guide decisions.\n\
        Never give financial advice. Always note past performance doesn't guarantee future results.\n\
        Speak in plain language. Explain technical concepts when you use them.\n\n\
        Current signals data (JSON):\n{}", signals_context
    );

    match llm::chat(&state.http_client, provider, &system_prompt, &req.message).await {
        Ok(response) => Ok(Json(serde_json::json!({ "response": response }))),
        Err(e) => Ok(Json(serde_json::json!({ "response": format!("Error: {}", e) }))),
    }
}

// ════════════════════════════════════════
// Signal Generation Pipeline (INFERENCE ONLY)
// ════════════════════════════════════════

/// Refresh all signals (loads saved model weights, runs inference)
async fn refresh_signals(state: &AppState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db_path = state.db_path.clone();
    let asset_config = state.asset_config.read().await.clone();

    let new_signals = tokio::task::spawn_blocking(move || {
        generate_all_signals(&db_path, &asset_config)
    }).await??;

    // Store snapshots in DB
    {
        let database = db::Database::new(&state.db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        for sig in &new_signals {
            let _ = database.insert_signal_snapshot(sig, 3);
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

    let db_path = state.db_path.clone();
    let asset_config = state.asset_config.read().await.clone();

    let new_signals = tokio::task::spawn_blocking(move || {
        generate_signals_filtered(&db_path, &asset_config, us_market_open, fx_open, crypto_open)
    }).await??;

    // Store snapshots
    {
        let database = db::Database::new(&state.db_path)
            .map_err(|e| format!("DB error: {}", e))?;
        for sig in &new_signals {
            let _ = database.insert_signal_snapshot(sig, 3);
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

            if let Some(sig) = infer_stock_signal(&asset_entry.symbol, &points, &market_context) {
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

            if let Some(sig) = infer_fx_signal(&asset_entry.symbol, &points, &market_context) {
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

/// Load saved model weights and run inference on the latest feature vector.
/// Returns a WalkForwardResult populated with saved accuracies + fresh predictions.
fn infer_with_saved_models(
    symbol: &str,
    samples: &[ml::Sample],
) -> Option<ensemble::WalkForwardResult> {
    if samples.is_empty() {
        println!("  {} — no samples for inference", symbol);
        return None;
    }

    let n_features = samples[0].features.len();

    // Load the 3 saved models
    let linreg_saved = match model_store::load_weights(symbol, "linreg") {
        Ok(w) => w,
        Err(e) => {
            println!("  {} — skipping: no linreg model ({})", symbol, e);
            return None;
        }
    };
    let logreg_saved = match model_store::load_weights(symbol, "logreg") {
        Ok(w) => w,
        Err(e) => {
            println!("  {} — skipping: no logreg model ({})", symbol, e);
            return None;
        }
    };
    let (gbt_saved, gbt_classifier) = match model_store::load_gbt(symbol) {
        Ok(g) => g,
        Err(e) => {
            println!("  {} — skipping: no GBT model ({})", symbol, e);
            return None;
        }
    };

    // Get the latest feature vector
    let last_sample = samples.last().unwrap();
    let feat = &last_sample.features;

    // LinReg prediction (normalise with its own saved params)
    let lin_feat = normalise_features(feat, &linreg_saved.norm_means, &linreg_saved.norm_stds);
    let raw_lin = predict_linreg(&linreg_saved, &lin_feat);
    let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);

    // LogReg prediction
    let log_feat = normalise_features(feat, &logreg_saved.norm_means, &logreg_saved.norm_stds);
    let log_prob = predict_logreg(&logreg_saved, &log_feat).clamp(0.15, 0.85);

    // GBT prediction (GBT has its own norm params)
    let gbt_feat = normalise_features(feat, &gbt_saved.norm_means, &gbt_saved.norm_stds);
    let gbt_prob = gbt_classifier.predict_proba(&gbt_feat).clamp(0.15, 0.85);

    // Use saved walk-forward accuracies
    let lin_acc = linreg_saved.meta.walk_forward_accuracy;
    let log_acc = logreg_saved.meta.walk_forward_accuracy;
    let gbt_acc = gbt_saved.meta.walk_forward_accuracy;

    println!("  {} — inference: LinR={:.1}% LogR={:.1}% GBT={:.1}% | probs: {:.2} {:.2} {:.2}",
        symbol, lin_acc, log_acc, gbt_acc, lin_prob, log_prob, gbt_prob);

    Some(ensemble::WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: lin_acc,
        logistic_accuracy: log_acc,
        gbt_accuracy: gbt_acc,
        lstm_accuracy: 50.0,
        n_folds: 1,
        total_test_samples: 0,
        linear_recent: lin_acc,
        logistic_recent: log_acc,
        gbt_recent: gbt_acc,
        lstm_recent: 50.0,
        final_linear_prob: lin_prob,
        final_logistic_prob: log_prob,
        final_gbt_prob: gbt_prob,
        final_lstm_prob: 0.5,
        gbt_importance: Vec::new(),
        n_features,
        has_lstm: false,
    })
}

/// Normalise a feature vector using pre-computed means and stds
fn normalise_features(features: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    features.iter().enumerate().map(|(i, &f)| {
        let mean = means.get(i).copied().unwrap_or(0.0);
        let std = stds.get(i).copied().unwrap_or(1.0);
        if std == 0.0 { f - mean } else { (f - mean) / std }
    }).collect()
}

/// Run linreg inference: dot(weights, features) + bias
fn predict_linreg(saved: &model_store::SavedWeights, features: &[f64]) -> f64 {
    let mut result = saved.bias;
    for (w, f) in saved.weights.iter().zip(features.iter()) {
        result += w * f;
    }
    result
}

/// Run logreg inference: sigmoid(dot(weights, features) + bias)
fn predict_logreg(saved: &model_store::SavedWeights, features: &[f64]) -> f64 {
    let mut z = saved.bias;
    for (w, f) in saved.weights.iter().zip(features.iter()) {
        z += w * f;
    }
    1.0 / (1.0 + (-z).exp())
}

/// Generate a single stock signal via inference
fn infer_stock_signal(
    symbol: &str,
    points: &[analysis::PricePoint],
    market_context: &features::MarketContext,
) -> Option<enriched_signals::EnrichedSignal> {
    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
    let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

    let samples = features::build_rich_features(
        &prices, &volumes, &timestamps,
        Some(market_context), "stock",
        features::sector_etf_for(symbol),
        None, None,
    );
    if samples.is_empty() { return None; }

    let wf = infer_with_saved_models(symbol, &samples)?;

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
) -> Option<enriched_signals::EnrichedSignal> {
    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
    let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

    let samples = features::build_rich_features(
        &prices, &volumes, &timestamps,
        Some(market_context), "fx",
        None, None, None,
    );
    if samples.is_empty() { return None; }

    let wf = infer_with_saved_models(symbol, &samples)?;

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

    let wf = infer_with_saved_models(coin_id, &enriched_samples)?;

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
