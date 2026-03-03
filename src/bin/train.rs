/// train — Weekly heavy job
/// ========================
/// Fetches all data, trains models, runs backtester, saves to models/ directory.
/// Usage: cargo run --release --bin train

use rust_invest::*;
use chrono::Utc;
use tokio::time::{sleep, Duration};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let database = db::Database::new("rust_invest.db")?;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         RUST INVEST — TRAIN MODE (Weekly Heavy Job)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let now = Utc::now().to_rfc3339();

    // ── Fetch crypto ──
    println!("━━━ FETCHING CRYPTO DATA ━━━\n");
    let coins = crypto::fetch_top_coins(&client).await?;
    for coin in &coins {
        database.insert_crypto(coin, &now)?;
    }
    println!("  ✓ Stored {} crypto prices\n", coins.len());

    let top_coins: Vec<&models::CoinData> = coins.iter()
        .filter(|c| c.id != "tether" && c.symbol.to_lowercase() != "usdt")
        .take(5)
        .collect();

    for coin in &top_coins {
        let existing = database.count_crypto_history(&coin.id)?;
        if existing > 0 {
            println!("  {} — already have {} records, skipping", coin.name, existing);
            continue;
        }
        println!("  Fetching {} history...", coin.name);
        sleep(Duration::from_secs(12)).await;
        match crypto::fetch_history(&client, &coin.id, 365).await {
            Ok(chart) => {
                let mut count = 0;
                for (i, price_point) in chart.prices.iter().enumerate() {
                    let timestamp_ms = price_point[0] as i64;
                    let price = price_point[1];
                    let volume = chart.total_volumes.get(i).map(|v| v[1]);
                    let timestamp = chrono::DateTime::from_timestamp_millis(timestamp_ms)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();
                    database.insert_history(&coin.id, price, volume, &timestamp)?;
                    count += 1;
                }
                println!("    ✓ {} data points for {}", count, coin.name);
            }
            Err(e) => println!("    ✗ {}: {}", coin.name, e),
        }
    }

    // ── Fetch stocks ──
    println!("\n━━━ FETCHING STOCK DATA (5 years) ━━━\n");
    for stock in stocks::STOCK_LIST {
        let existing = database.count_stock_history(stock.symbol)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", stock.symbol, existing);
            continue;
        }
        println!("  Fetching {} 5-year history...", stock.symbol);
        match stocks::fetch_history(&client, stock.symbol, "5y").await {
            Ok(points) => {
                let mut count = 0;
                for (ts, price, volume) in &points {
                    let timestamp = chrono::DateTime::from_timestamp(*ts, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();
                    database.insert_stock_history(stock.symbol, *price, volume.map(|v| v as f64), &timestamp)?;
                    count += 1;
                }
                println!("    ✓ {} data points for {}", count, stock.symbol);
            }
            Err(e) => println!("    ✗ {}: {}", stock.symbol, e),
        }
    }

    // ── Fetch FX ──
    println!("\n━━━ FETCHING FX DATA (5 years) ━━━\n");
    for fx in stocks::FX_LIST {
        let existing = database.count_fx_history(fx.symbol)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", fx.symbol, existing);
            continue;
        }
        println!("  Fetching {} 5-year history...", fx.symbol);
        match stocks::fetch_history(&client, fx.symbol, "5y").await {
            Ok(points) => {
                let mut count = 0;
                for (ts, price, volume) in &points {
                    let timestamp = chrono::DateTime::from_timestamp(*ts, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();
                    database.insert_fx_history(fx.symbol, *price, volume.map(|v| v as f64), &timestamp)?;
                    count += 1;
                }
                println!("    ✓ {} data points for {}", count, fx.symbol);
            }
            Err(e) => println!("    ✗ {}: {}", fx.symbol, e),
        }
    }

    // ── Fetch market indicators ──
    println!("\n━━━ FETCHING MARKET INDICATORS ━━━\n");
    for ticker in features::MARKET_TICKERS {
        let existing = database.count_market_history(ticker)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", ticker, existing);
            continue;
        }
        println!("  Fetching {} 5-year history...", ticker);
        match stocks::fetch_history(&client, ticker, "5y").await {
            Ok(points) => {
                let mut count = 0;
                for (ts, price, volume) in &points {
                    let timestamp = chrono::DateTime::from_timestamp(*ts, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();
                    database.insert_market_history(ticker, *price, volume.map(|v| v as f64), &timestamp)?;
                    count += 1;
                }
                println!("    ✓ {} data points for {}", count, ticker);
            }
            Err(e) => println!("    ✗ {}: {}", ticker, e),
        }
    }

    // ── Build market context ──
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?.iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        market_histories.insert(ticker.to_string(), prices);
    }
    let market_context = features::build_market_context(&market_histories);

    // ── Train & save models for stocks ──
    println!("\n━━━ TRAINING MODELS (Walk-Forward) ━━━\n");

    let mut backtest_results: Vec<backtester::BacktestResult> = Vec::new();
    let bt_config = backtester::BacktestConfig::default();

    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "stock");
        if samples.len() < 100 { continue; }

        let n_feat = samples[0].features.len();
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        if let Some(wf) = ensemble::walk_forward_samples(stock.symbol, &samples, train_window, test_window, step) {
            // Save final-fold models
            let last_train_end = {
                let mut s = 0; let mut last = 0;
                while s + train_window + test_window <= samples.len() { last = s + train_window; s += step; }
                last
            };
            let mut last_fold: Vec<ml::Sample> = samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
            let (means, stds) = ml::normalise(&mut last_fold);

            let mut lin = ml::LinearRegression::new(n_feat);
            lin.train(&last_fold, 0.005, 3000);
            let _ = model_store::save_weights(stock.symbol, "linreg", &lin.weights, lin.bias, n_feat, last_fold.len(), wf.linear_accuracy, &means, &stds);

            let mut log = ml::LogisticRegression::new(n_feat);
            log.train(&last_fold, 0.01, 3000);
            let _ = model_store::save_weights(stock.symbol, "logreg", &log.weights, log.bias, n_feat, last_fold.len(), wf.logistic_accuracy, &means, &stds);

            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);
            let gbt_config = gbt::GBTConfig { n_trees: 80, learning_rate: 0.08, tree_config: gbt::TreeConfig { max_depth: 4, min_samples_leaf: 8, min_samples_split: 16 }, subsample_ratio: 0.8, early_stopping_rounds: Some(8) };
            let gbt_model = gbt::GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), gbt_config);
            let _ = model_store::save_gbt(stock.symbol, &gbt_model, last_fold.len(), wf.gbt_accuracy, &means, &stds);

            println!("  ✓ Models saved for {}", stock.symbol);
        }

        if let Some(bt) = backtester::run_backtest(stock.symbol, &samples, &prices, train_window, test_window, step, &bt_config) {
            backtest_results.push(bt);
        }
    }

    // ── Train FX models ──
    for fx in stocks::FX_LIST {
        let points = database.get_fx_history(fx.symbol)?;
        if points.len() < 300 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();
        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "fx");
        if samples.len() < 100 { continue; }
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;
        let _ = ensemble::walk_forward_samples(fx.symbol, &samples, train_window, test_window, step);
        if let Some(bt) = backtester::run_backtest(fx.symbol, &samples, &prices, train_window, test_window, step, &bt_config) {
            backtest_results.push(bt);
        }
    }

    // ── Train crypto models ──
    let coin_ids: Vec<String> = database.get_all_coin_ids()?.into_iter().filter(|id| id != "tether").collect();

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

        if enriched_samples.len() < 100 { continue; }
        let train_window = (enriched_samples.len() as f64 * 0.6) as usize;
        let test_window = 20.min(enriched_samples.len() / 10);
        let step = test_window;
        let _ = ensemble::walk_forward_samples(coin_id, &enriched_samples, train_window, test_window, step);
        if let Some(bt) = backtester::run_backtest(coin_id, &enriched_samples, &prices, train_window, test_window, step, &bt_config) {
            backtest_results.push(bt);
        }
    }

    if !backtest_results.is_empty() {
        backtester::print_backtest_summary(&backtest_results);
    }

    println!("\n━━━ TRAINING COMPLETE ━━━");
    println!("  Models saved to: models/");
    println!("  Database: rust_invest.db");
    let cached = model_store::list_cached_models();
    println!("  Cached models: {} files\n", cached.len());

    Ok(())
}
