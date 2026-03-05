/// serve — Web API server for Rust Invest
/// ========================================
/// Starts an Axum web server on port 8080 serving enriched trading signals.
/// Usage: cargo run --release --bin serve

use rust_invest::*;
use axum::{
    Router,
    routing::{get, post},
    extract::{Path, Query, State},
    Json,
    http::StatusCode,
};
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
}

// ════════════════════════════════════════
// Main
// ════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         RUST INVEST — SERVE MODE (Web API Server)              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Load asset config
    let asset_config = config::AssetConfig::load()
        .unwrap_or_else(|e| {
            eprintln!("  Warning: {}", e);
            eprintln!("  Using default asset config (built-in stock/FX lists)");
            default_asset_config()
        });
    println!("  Loaded asset config: {} stocks, {} FX, {} crypto",
        asset_config.stocks.len(), asset_config.fx.len(), asset_config.crypto.len());

    let state = AppState {
        signals: Arc::new(RwLock::new(HashMap::new())),
        asset_config: Arc::new(RwLock::new(asset_config)),
        db_path: "rust_invest.db".to_string(),
    };

    // Generate signals on startup
    println!("\n━━━ GENERATING INITIAL SIGNALS ━━━\n");
    if let Err(e) = refresh_signals(&state).await {
        eprintln!("  Warning: Initial signal generation failed: {}", e);
        eprintln!("  Server will start with empty signals. Ensure database has data.");
    }

    {
        let sigs = state.signals.read().await;
        println!("\n  Initial signals generated: {}", sigs.len());
    }

    // Start hourly scheduler
    let scheduler_state = state.clone();
    tokio::spawn(async move {
        // Wait 1 hour before first scheduled refresh (we just did initial)
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600));
        interval.tick().await; // skip immediate tick

        loop {
            interval.tick().await;
            let now = Utc::now();
            println!("\n  [Scheduler] Hourly refresh at {}", now.format("%H:%M:%S UTC"));

            if let Err(e) = refresh_signals_with_market_hours(&scheduler_state).await {
                eprintln!("  [Scheduler] Error: {}", e);
            }

            let sigs = scheduler_state.signals.read().await;
            println!("  [Scheduler] Refresh complete. {} signals cached.", sigs.len());
        }
    });

    // Build router
    let cors = tower_http::cors::CorsLayer::permissive();

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
        .layer(cors)
        .with_state(state);

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
    println!("    GET  /api/v1/portfolio/simulate\n");

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
    Query(params): Query<PortfolioParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let capital = params.capital.unwrap_or(100_000.0);
    let strategy = params.strategy.unwrap_or_else(|| "sharpe".to_string());

    let sigs = state.signals.read().await;
    let signal_count = sigs.len();
    let buy_count = sigs.values().filter(|s| s.signal == "BUY").count();
    let sell_count = sigs.values().filter(|s| s.signal == "SELL").count();
    let hold_count = signal_count - buy_count - sell_count;

    Ok(Json(serde_json::json!({
        "capital": capital,
        "strategy": strategy,
        "start_date": params.start_date,
        "signal_summary": {
            "total": signal_count,
            "buy": buy_count,
            "sell": sell_count,
            "hold": hold_count,
        },
        "note": "Full portfolio simulation requires historical backtest data. Use cargo run --bin train for full backtest."
    })))
}

// ════════════════════════════════════════
// Signal Generation Pipeline
// ════════════════════════════════════════

/// Refresh all signals (runs the ML pipeline on stored data)
async fn refresh_signals(state: &AppState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db_path = state.db_path.clone();
    let asset_config = state.asset_config.read().await.clone();

    // Run the CPU-heavy signal generation on a blocking thread
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

/// Generate signals with market hours filtering
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

    // ── Stock signals ──
    if include_stocks {
        let enabled_stocks = asset_config.enabled_stocks();
        for asset_entry in &enabled_stocks {
            // Find matching stock in STOCK_LIST or use the symbol directly
            let points = match database.get_stock_history(&asset_entry.symbol) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if points.len() < 300 { continue; }

            if let Some(sig) = generate_stock_signal(&asset_entry.symbol, &points, &market_context) {
                enriched_signals.push(sig);
            }
        }
    }

    // ── FX signals ──
    if include_fx {
        let enabled_fx = asset_config.enabled_fx();
        for asset_entry in &enabled_fx {
            let points = match database.get_fx_history(&asset_entry.symbol) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if points.len() < 300 { continue; }

            if let Some(sig) = generate_fx_signal(&asset_entry.symbol, &points, &market_context) {
                enriched_signals.push(sig);
            }
        }
    }

    // ── Crypto signals ──
    if include_crypto {
        let enabled_crypto = asset_config.enabled_crypto();
        if !enabled_crypto.is_empty() {
            // Build crypto enrichment
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
                if let Some(sig) = generate_crypto_signal(
                    &asset_entry.symbol, &database, &crypto_enrichment,
                ) {
                    enriched_signals.push(sig);
                }
            }
        }
    }

    println!("  Generated {} enriched signals", enriched_signals.len());
    Ok(enriched_signals)
}

/// Generate a single stock signal
fn generate_stock_signal(
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
    );
    if samples.len() < 100 { return None; }

    let train_window = (samples.len() as f64 * 0.6) as usize;
    let test_window = 30.min(samples.len() / 10);
    let step = test_window;

    let wf = ensemble::walk_forward_samples(symbol, &samples, train_window, test_window, step)?;

    let result = analysis::analyse_coin(symbol, points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(&wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend);

    // Extract volatility for enrichment
    let vol_5d = if prices.len() >= 5 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-5..])))
    } else { None };
    let vol_20d = if prices.len() >= 20 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-20..])))
    } else { None };

    // BB position from recent features
    let bb_pos = extract_bb_position(&samples);

    Some(enriched_signals::enrich_signal(&signal, "stock", bb_pos, vol_5d, vol_20d))
}

/// Generate a single FX signal
fn generate_fx_signal(
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
    );
    if samples.len() < 100 { return None; }

    let train_window = (samples.len() as f64 * 0.6) as usize;
    let test_window = 30.min(samples.len() / 10);
    let step = test_window;

    let wf = ensemble::walk_forward_samples(symbol, &samples, train_window, test_window, step)?;

    let result = analysis::analyse_coin(symbol, points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(&wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend);

    let vol_5d = if prices.len() >= 5 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-5..])))
    } else { None };
    let vol_20d = if prices.len() >= 20 {
        Some(analysis::std_dev(&daily_returns(&prices[prices.len()-20..])))
    } else { None };

    let bb_pos = extract_bb_position(&samples);

    Some(enriched_signals::enrich_signal(&signal, "fx", bb_pos, vol_5d, vol_20d))
}

/// Generate a single crypto signal
fn generate_crypto_signal(
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

    if enriched_samples.len() < 100 { return None; }

    let train_window = (enriched_samples.len() as f64 * 0.6) as usize;
    let test_window = 20.min(enriched_samples.len() / 10);
    let step = test_window;

    let wf = ensemble::walk_forward_samples(coin_id, &enriched_samples, train_window, test_window, step)?;

    let result = analysis::analyse_coin(coin_id, &points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(&wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend);

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
    // In the rich features, BB Position is at index 3 (see ml::FEATURE_NAMES)
    // But after build_rich_features, the ordering may differ
    // The 83 rich features have BB position typically at index 3
    samples.last().map(|s| {
        if s.features.len() > 3 {
            s.features[3].clamp(0.0, 1.0) // BB position is normalised 0-1
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
