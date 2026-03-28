/// signal — Fast daily/twice-daily signal generation (INFERENCE ONLY)
/// ==================================================================
/// Loads saved models, fetches latest prices from DB, generates trading signals.
/// Now includes Claude LLM sentiment analysis via Serper + NewsAPI + Reddit.
/// NO training happens here — only forward-pass inference on saved weights.
/// Usage: cargo run --release --bin signal

use rust_invest::*;
use std::collections::HashMap;
use chrono::{Datelike, Timelike};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let database = db::Database::new("rust_invest.db")?;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       ALPHA SIGNAL — SIGNAL MODE (Inference Only)              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Load .env for API keys
    llm::load_provider(); // triggers dotenv loading

    // Check that we have trained models
    let cached = model_store::list_cached_models();
    if cached.is_empty() {
        println!("  No cached models found in models/");
        println!("  Run `cargo run --release --bin train` first to train models.\n");
        return Ok(());
    }
    println!("  Found {} cached model files\n", cached.len());

    // ── Initialise sentiment table ──
    {
        let conn = rusqlite::Connection::open("rust_invest.db")?;
        news_sentiment::create_sentiment_table(&conn)?;
    }

    // ── Build market context from stored data ──
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?.iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        market_histories.insert(ticker.to_string(), prices);
    }
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

        if let Some(wf) = inference::infer_with_saved_models(stock.symbol, &samples) {
            let result = analysis::analyse_coin(stock.symbol, &points);
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);
            let trend = match (sma_7.last(), sma_30.last()) {
                (Some(s), Some(l)) if s > l => "BULLISH",
                _ => "BEARISH",
            };
            signals.push(ensemble::ensemble_signal(stock.symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend));
        }
    }

    // ── FX signals (inference only) ──
    println!("━━━ GENERATING FX SIGNALS (inference only) ━━━\n");
    for fx in stocks::FX_LIST {
        let points = database.get_fx_history(fx.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "fx", Some(fx.symbol), None, fg_ref);
        if samples.is_empty() { continue; }

        if let Some(wf) = inference::infer_with_saved_models(fx.symbol, &samples) {
            let result = analysis::analyse_coin(fx.symbol, &points);
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);
            let trend = match (sma_7.last(), sma_30.last()) {
                (Some(s), Some(l)) if s > l => "BULLISH",
                _ => "BEARISH",
            };
            signals.push(ensemble::ensemble_signal(fx.symbol, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend));
        }
    }

    // ── Crypto signals (inference only) ──
    println!("━━━ GENERATING CRYPTO SIGNALS (inference only) ━━━\n");
    let coin_ids: Vec<String> = database.get_all_coin_ids()?.into_iter().filter(|id| id != "tether").collect();

    // Build crypto enrichment data
    let mut crypto_prices_map: HashMap<String, Vec<f64>> = HashMap::new();
    let mut crypto_returns_map: HashMap<String, Vec<f64>> = HashMap::new();
    let mut crypto_dates_map: HashMap<String, Vec<String>> = HashMap::new();
    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 60 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
        let dates: Vec<String> = points.iter().map(|p| p.timestamp[..10].to_string()).collect();
        crypto_prices_map.insert(coin_id.clone(), prices);
        crypto_returns_map.insert(coin_id.clone(), returns);
        crypto_dates_map.insert(coin_id.clone(), dates);
    }
    let crypto_syms: Vec<&str> = coin_ids.iter().map(|s| s.as_str()).collect();
    let crypto_enrichment = crypto_features::enrich_crypto_features(&crypto_syms, &crypto_prices_map, &crypto_returns_map, &crypto_dates_map);

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 200 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let base_samples = gbt::build_extended_features(&prices, &volumes);
        if base_samples.is_empty() { continue; }

        let enriched_samples: Vec<ml::Sample> = if let Some(crypto_rows) = crypto_enrichment.get(coin_id.as_str()) {
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

        if enriched_samples.is_empty() { continue; }

        if let Some(wf) = inference::infer_with_saved_models(coin_id, &enriched_samples) {
            let result = analysis::analyse_coin(coin_id, &points);
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);
            let trend = match (sma_7.last(), sma_30.last()) {
                (Some(s), Some(l)) if s > l => "BULLISH",
                _ => "BEARISH",
            };
            signals.push(ensemble::ensemble_signal(coin_id, &wf, result.current_price, result.rsi_14.unwrap_or(50.0), trend));
        }
    }

    // ── LLM Sentiment Analysis (Claude) ──
    // Fetch news + Reddit for each signal, analyse with Claude, adjust signals
    println!("\n━━━ FETCHING NEWS & LLM SENTIMENT ANALYSIS ━━━\n");
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

            // Fetch fresh sentiment (Serper + NewsAPI + Reddit → Claude analysis)
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
    println!("  Assets: {} stocks, {} FX, {} crypto\n",
        stocks::STOCK_LIST.len(), stocks::FX_LIST.len(), coin_ids.len());

    // ── Log predictions to database ──
    println!("\n━━━ LOGGING PREDICTIONS ━━━\n");
    let now_ts = chrono::Utc::now().to_rfc3339();
    let mut logged = 0;
    for s in &signals {
        if s.signal == "HOLD" { continue; } // Only log actionable signals
        if let Err(e) = database.insert_prediction(&now_ts, &s.symbol, &s.signal, s.confidence, s.current_price) {
            eprintln!("  Failed to log prediction for {}: {}", s.symbol, e);
        } else {
            logged += 1;
        }
    }
    println!("  Logged {} predictions to database", logged);

    // ── Resolve pending predictions (market-hours aware) ──
    println!("\n━━━ RESOLVING PENDING PREDICTIONS ━━━\n");
    let pending = database.get_pending_predictions()?;
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

        // Get current price and asset class for this asset
        let (current_price, asset_class) = if let Some(sig) = signals.iter().find(|s| s.symbol == pred.asset) {
            let ac = if sig.symbol.ends_with("=X") { "fx" }
                else if matches!(sig.symbol.as_str(), "bitcoin"|"ethereum"|"solana"|"ripple"|"dogecoin"|"cardano"|"avalanche-2"|"chainlink"|"polkadot"|"near"|"sui"|"aptos"|"arbitrum"|"the-open-network"|"uniswap"|"tron"|"litecoin"|"shiba-inu"|"stellar"|"matic-network") { "crypto" }
                else { "stock" };
            (sig.current_price, ac)
        } else {
            continue;
        };

        // Apply market-hours resolution rules
        let weekday = now.weekday();
        let hour = now.hour();
        let can_resolve = match asset_class {
            "stock" => {
                if matches!(weekday, chrono::Weekday::Sat | chrono::Weekday::Sun) { false }
                else if hour < 14 || hour > 21 { false }
                else { match pred.signal.as_str() { "BUY"|"SHORT"|"SELL" => hour >= 20, _ => true } }
            }
            "fx" => {
                let fx_open = match weekday {
                    chrono::Weekday::Sat | chrono::Weekday::Sun => false,
                    chrono::Weekday::Fri => hour < 22,
                    _ => true,
                };
                if !fx_open { false }
                else { match pred.signal.as_str() { "BUY"|"SHORT"|"SELL" => hour >= 21, _ => true } }
            }
            "crypto" => match pred.signal.as_str() {
                "BUY"|"SHORT"|"SELL" => hour >= 23 || age_hours >= 24,
                _ => true,
            },
            _ => false,
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

        if let Err(e) = database.update_prediction_outcome(pred.id, actual_direction, was_correct, current_price, &resolve_ts) {
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
