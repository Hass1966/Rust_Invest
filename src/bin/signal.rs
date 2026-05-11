/// signal — Fast daily/twice-daily signal generation (INFERENCE ONLY)
/// ==================================================================
/// Loads saved models, fetches latest prices from DB, generates trading signals.
/// Includes word-based sentiment analysis via Serper + NewsAPI + Reddit.
/// NO training happens here — only forward-pass inference on saved weights.
///
/// Writes to: PostgreSQL alpha_signal database (signals + predictions tables)
/// Reads from: SQLite rust_invest.db (price history, market data, sentiment cache)
///
/// Usage: cargo run --release --bin signal

use rust_invest::*;
use std::collections::HashMap;
use chrono::{Datelike, Timelike};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    // SQLite: read-only for price history + market data
    let database = db::Database::new("rust_invest.db")?;

    // PostgreSQL: write target for signals + predictions
    let pg_pool = pg::create_pool()?;
    // Verify connection
    {
        let _conn = pg_pool.get().await?;
        println!("  Connected to PostgreSQL (alpha_signal)");
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       ALPHA SIGNAL — SIGNAL MODE (Inference Only)              ║");
    println!("║       Writes: PostgreSQL alpha_signal                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Load .env for API keys (NEWSAPI_KEY, SERPER_API_KEY, etc.)
    llm::load_provider();

    // Check that we have trained models
    let cached = model_store::list_cached_models();
    if cached.is_empty() {
        println!("  No cached models found in models/");
        println!("  Run `cargo run --release --bin train` first to train models.\n");
        return Ok(());
    }
    println!("  Found {} cached model files\n", cached.len());

    // ── Initialise sentiment table (SQLite — sentiment cache stays here) ──
    {
        let conn = rusqlite::Connection::open("rust_invest.db")?;
        news_sentiment::create_sentiment_table(&conn)?;
    }

    // ── Build market context from stored data (SQLite reads) ──
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?.iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        market_histories.insert(ticker.to_string(), prices);
    }
    market_histories.insert("HY_SPREAD".to_string(), database.get_market_prices("HY_SPREAD").unwrap_or_default());
    market_histories.insert("BREAKEVEN_5Y".to_string(), database.get_market_prices("BREAKEVEN_5Y").unwrap_or_default());
    let market_context = features::build_market_context(&market_histories);

    // Load Fear & Greed history
    let fear_greed_history = database.get_fear_greed_history().unwrap_or_default();
    let fg_ref: Option<&[(String, f64)]> = if fear_greed_history.is_empty() {
        None
    } else {
        Some(&fear_greed_history)
    };

    let mut signals: Vec<ensemble::TradingSignal> = Vec::new();

    // ── Stock signals (inference only) ──
    println!("━━━ GENERATING STOCK SIGNALS (inference only) ━━━\n");
    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "stock", features::sector_etf_for(stock.symbol), None, fg_ref);
        if samples.is_empty() { continue; }

        // Try regression models first (v6: Ridge + LightGBM), fall back to classification (v5)
        let reg = inference::infer_regression_models(stock.symbol, &samples);

        let result = analysis::analyse_coin(stock.symbol, &points);
        let sma_7 = analysis::sma(&prices, 7);
        let sma_30 = analysis::sma(&prices, 30);
        let trend = match (sma_7.last(), sma_30.last()) {
            (Some(s), Some(l)) if s > l => "BULLISH",
            _ => "BEARISH",
        };

        let signal = if let Some(ref reg_result) = reg {
            let mut sig = ensemble::regression_signal(stock.symbol, reg_result, result.current_price, result.rsi_14.unwrap_or(50.0), trend, "stock");
            // Multi-horizon confirmation: filter through 5d model agreement
            let filtered = inference::apply_horizon_agreement(&sig.signal, stock.symbol, &samples, reg_result, "stock");
            if filtered != sig.signal {
                sig.signal = filtered;
            }
            sig
        } else if let Some(wf) = inference::infer_with_saved_models(stock.symbol, &samples) {
            ensemble::ensemble_signal(stock.symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend)
        } else {
            continue;
        };
        signals.push(signal);
    }

    // ── FX signals disabled (descoped — separate project) ──
    println!("━━━ FX SIGNALS SKIPPED (descoped) ━━━\n");
    // ── Crypto signals disabled (descoped — separate project) ──
    println!("━━━ CRYPTO SIGNALS SKIPPED (descoped) ━━━\n");

    // ── News Sentiment Analysis (word-based) ──
    // Fetch news + Reddit for each signal, score with word-based method, adjust signals
    println!("\n━━━ FETCHING NEWS & SENTIMENT ANALYSIS ━━━\n");
    let newsapi_key = std::env::var("NEWSAPI_KEY").unwrap_or_default();
    let has_newsapi = !newsapi_key.is_empty() && newsapi_key != "REPLACE_WHEN_YOU_HAVE_IT";

    if has_newsapi || std::env::var("SERPER_API_KEY").is_ok() {
        let conn = rusqlite::Connection::open("rust_invest.db")?;
        let mut sentiment_count = 0;

        for signal in signals.iter_mut() {
            // Skip if we already have today's sentiment cached
            if news_sentiment::has_today_sentiment(&conn, &signal.symbol) {
                // Load existing sentiment
                let data = news_sentiment::get_recent_sentiment(&conn, &signal.symbol, 1);
                if let Some(latest) = data.first() {
                    ensemble::apply_sentiment_adjustment(
                        signal,
                        latest.combined_score,
                        latest.llm_analysis.clone(),
                    );
                    sentiment_count += 1;
                }
                continue;
            }

            // Fetch fresh sentiment (Serper + NewsAPI + Reddit → word-based scoring)
            let newsapi = if has_newsapi { &newsapi_key } else { "" };
            match news_sentiment::fetch_and_store_sentiment(&client, &conn, &signal.symbol, newsapi).await {
                Ok(data) => {
                    ensemble::apply_sentiment_adjustment(
                        signal,
                        data.combined_score,
                        data.llm_analysis,
                    );
                    sentiment_count += 1;
                }
                Err(e) => {
                    eprintln!("  Sentiment fetch failed for {}: {}", signal.symbol, e);
                }
            }

            // Rate limit: small delay between API calls
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        println!("  Analysed sentiment for {} assets\n", sentiment_count);
    } else {
        println!("  Skipping (no NEWSAPI_KEY or SERPER_API_KEY configured)\n");
    }

    // ── Suppress BUY→HOLD for chronically poor-performing assets ──
    const SUPPRESSED_ASSETS: &[&str] = &[
        "NZDUSD=X", "AUDUSD=X", "USDIDR=X", "CRM", "LMT", "XLI", "USDMXN=X",
    ];
    for signal in signals.iter_mut() {
        if signal.signal == "BUY" && SUPPRESSED_ASSETS.contains(&signal.symbol.as_str()) {
            signal.signal = "HOLD".to_string();
        }
    }

    // ── Print signals ──
    if !signals.is_empty() {
        println!();
        ensemble::print_signals(&signals);
    } else {
        println!("  No signals generated. Ensure database has data (run train first).\n");
    }

    println!("\n  Total signals: {}", signals.len());
    println!("  Assets: {} stocks (FX/crypto descoped)\n",
        stocks::STOCK_LIST.len());

    // ── Log predictions to PostgreSQL ──
    println!("\n━━━ LOGGING PREDICTIONS (PostgreSQL) ━━━\n");
    let now_ts = chrono::Utc::now().to_rfc3339();
    let mut logged = 0;
    for s in &signals {
        if s.signal == "HOLD" { continue; } // Only log actionable signals
        if let Err(e) = pg::insert_prediction(&pg_pool, &now_ts, &s.symbol, &s.signal, s.confidence, s.current_price).await {
            eprintln!("  Failed to log prediction for {}: {}", s.symbol, e);
        } else {
            logged += 1;
        }
    }
    println!("  Logged {} predictions to PostgreSQL", logged);

    // ── Resolve pending predictions (market-hours aware) ──
    println!("\n━━━ RESOLVING PENDING PREDICTIONS ━━━\n");
    let pending = pg::get_pending_predictions(&pg_pool).await?;
    let now = chrono::Utc::now();
    let resolve_ts = now.to_rfc3339();
    let mut resolved = 0;
    for pred in &pending {
        let pred_time = match chrono::DateTime::parse_from_rfc3339(&pred.timestamp) {
            Ok(t) => t.with_timezone(&chrono::Utc),
            Err(_) => continue,
        };
        let age_hours = (now - pred_time).num_hours();

        // No signal resolves in less than 4 hours
        if age_hours < 4 { continue; }

        // Get current price for this asset (stocks only now)
        let current_price = if let Some(sig) = signals.iter().find(|s| s.symbol == pred.asset) {
            sig.current_price
        } else {
            continue;
        };

        // Apply market-hours resolution rules (stocks only)
        let weekday = now.weekday();
        let hour = now.hour();
        let can_resolve = {
            if matches!(weekday, chrono::Weekday::Sat | chrono::Weekday::Sun) { false }
            else if hour < 14 || hour > 21 { false }
            else { match pred.signal.as_str() { "BUY"|"SHORT"|"SELL" => hour >= 20, _ => true } }
        };
        if !can_resolve { continue; }

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
            _ => true, // FLAT counts as correct for any signal
        };

        if let Err(e) = pg::update_prediction_outcome(&pg_pool, pred.id, actual_direction, was_correct, current_price, &resolve_ts).await {
            eprintln!("  Failed to resolve prediction {}: {}", pred.id, e);
        } else {
            let icon = if was_correct { "✓" } else { "✗" };
            println!("  {} {} {} @ {:.2} → {:.2} ({})", icon, pred.asset, pred.signal, pred.price_at_prediction, current_price, actual_direction);
            resolved += 1;
        }
    }
    println!("  Resolved {} of {} pending predictions", resolved, pending.len());

    Ok(())
}
