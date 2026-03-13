/// train — Weekly heavy job
/// ========================
/// Fetches all data, trains models, runs backtester, saves to models/ directory.
/// Usage: cargo run --release --bin train

use rust_invest::*;
use chrono::Utc;
use tokio::time::{sleep, Duration};
use std::collections::HashMap;

/// Accuracy results for one asset (used in comparison report)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct AssetAccuracy {
    linreg: f64,
    logreg: f64,
    gbt: f64,
    lstm: f64,
    regime: f64,
    ensemble: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let test_lstm = args.iter().any(|a| a == "--test-lstm");

    let client = reqwest::Client::new();
    let database = db::Database::new("rust_invest.db")?;

    if test_lstm {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║    RUST INVEST — LSTM TEST MODE (SPY + MSFT only)              ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
        return run_lstm_test(&database).await;
    }

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
        .take(15)
        .collect();

    for coin in &top_coins {
        let existing = database.count_crypto_history(&coin.id)?;
        if existing > 0 {
            println!("  {} — already have {} records, skipping", coin.name, existing);
            continue;
        }
        println!("  Fetching {} history...", coin.name);
        sleep(Duration::from_secs(12)).await;
        match crypto::fetch_history(&client, &coin.id, 1000).await {
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
    println!("\n━━━ FETCHING STOCK DATA (7 years) ━━━\n");
    for stock in stocks::STOCK_LIST {
        let existing = database.count_stock_history(stock.symbol)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", stock.symbol, existing);
            continue;
        }
        println!("  Fetching {} 7-year history...", stock.symbol);
        match stocks::fetch_history(&client, stock.symbol, "7y").await {
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
    println!("\n━━━ FETCHING FX DATA (7 years) ━━━\n");
    for fx in stocks::FX_LIST {
        let existing = database.count_fx_history(fx.symbol)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", fx.symbol, existing);
            continue;
        }
        println!("  Fetching {} 7-year history...", fx.symbol);
        match stocks::fetch_history(&client, fx.symbol, "7y").await {
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
        println!("  Fetching {} 7-year history...", ticker);
        match stocks::fetch_history(&client, ticker, "7y").await {
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

    // ── Train & save all 6 models for stocks ──
    println!("\n━━━ TRAINING ALL 6 MODELS (Walk-Forward) ━━━\n");
    let total_features = features::feature_names().len();
    println!("  Total features: {} (active: {}, pruned: {})",
        total_features, features::active_feature_count(), total_features - features::active_feature_count());

    let ensemble_overrides = ensemble::load_ensemble_overrides();
    let mut accuracy_results: HashMap<String, AssetAccuracy> = HashMap::new();

    let mut backtest_results: Vec<backtester::BacktestResult> = Vec::new();
    let bt_config = backtester::BacktestConfig::default();
    // let tft_config = tft::TFTConfig::default();

    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "stock", features::sector_etf_for(stock.symbol), None, None);
        if samples.len() < 100 { continue; }

        let n_feat = samples[0].features.len();
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        // Model 1-4: Core ensemble (LinReg, LogReg, GBT, LSTM)
        let mut lstm_acc = 0.0;
        let mut regime_acc = 0.0;

        if let Some(wf) = ensemble::walk_forward_samples(stock.symbol, &samples, train_window, test_window, step) {
            lstm_acc = if wf.has_lstm { wf.lstm_accuracy } else { 0.0 };

            // Model 5: Regime-Aware Ensemble
            if let Some(rw) = regime::walk_forward_regime(stock.symbol, &samples, train_window, test_window, step) {
                regime_acc = rw.overall_accuracy;
            }

            // Track accuracy for comparison report
            let ov = ensemble::get_override(&ensemble_overrides, stock.symbol);
            let ens_acc = compute_ensemble_accuracy(&wf, &ov);
            accuracy_results.insert(stock.symbol.to_string(), AssetAccuracy {
                linreg: wf.linear_accuracy,
                logreg: wf.logistic_accuracy,
                gbt: wf.gbt_accuracy,
                lstm: lstm_acc,
                regime: regime_acc,
                ensemble: ens_acc,
            });

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

            println!("  ✓ All 5 models trained for {} [LinReg:{:.1}% LogReg:{:.1}% GBT:{:.1}% LSTM:{:.1}% Regime:{:.1}%]",
                stock.symbol, wf.linear_accuracy, wf.logistic_accuracy, wf.gbt_accuracy,
                lstm_acc, regime_acc);
        }

        if let Some(bt) = backtester::run_backtest(stock.symbol, &samples, &prices, train_window, test_window, step, &bt_config) {
            backtest_results.push(bt);
        }
    }

    // ── Train & save all 6 models for FX ──
    println!("\n━━━ TRAINING FX MODELS ━━━\n");
    for fx in stocks::FX_LIST {
        let points = database.get_fx_history(fx.symbol)?;
        if points.len() < 300 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();
        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "fx", Some(fx.symbol), None, None);
        if samples.len() < 100 { continue; }

        let n_feat = samples[0].features.len();
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        let mut lstm_acc = 0.0;
        let mut regime_acc = 0.0;

        if let Some(wf) = ensemble::walk_forward_samples(fx.symbol, &samples, train_window, test_window, step) {
            lstm_acc = if wf.has_lstm { wf.lstm_accuracy } else { 0.0 };

            if let Some(rw) = regime::walk_forward_regime(fx.symbol, &samples, train_window, test_window, step) {
                regime_acc = rw.overall_accuracy;
            }

            let ov = ensemble::get_override(&ensemble_overrides, fx.symbol);
            let ens_acc = compute_ensemble_accuracy(&wf, &ov);
            accuracy_results.insert(fx.symbol.to_string(), AssetAccuracy {
                linreg: wf.linear_accuracy,
                logreg: wf.logistic_accuracy,
                gbt: wf.gbt_accuracy,
                lstm: lstm_acc,
                regime: regime_acc,
                ensemble: ens_acc,
            });

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
            let _ = model_store::save_weights(fx.symbol, "linreg", &lin.weights, lin.bias, n_feat, last_fold.len(), wf.linear_accuracy, &means, &stds);

            let mut log = ml::LogisticRegression::new(n_feat);
            log.train(&last_fold, 0.01, 3000);
            let _ = model_store::save_weights(fx.symbol, "logreg", &log.weights, log.bias, n_feat, last_fold.len(), wf.logistic_accuracy, &means, &stds);

            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);
            let gbt_config = gbt::GBTConfig { n_trees: 80, learning_rate: 0.08, tree_config: gbt::TreeConfig { max_depth: 4, min_samples_leaf: 8, min_samples_split: 16 }, subsample_ratio: 0.8, early_stopping_rounds: Some(8) };
            let gbt_model = gbt::GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), gbt_config);
            let _ = model_store::save_gbt(fx.symbol, &gbt_model, last_fold.len(), wf.gbt_accuracy, &means, &stds);

            println!("  ✓ All 5 models trained for {} [LinReg:{:.1}% LogReg:{:.1}% GBT:{:.1}% LSTM:{:.1}% Regime:{:.1}%]",
                fx.symbol, wf.linear_accuracy, wf.logistic_accuracy, wf.gbt_accuracy,
                lstm_acc, regime_acc);
        }

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
        let n_feat = enriched_samples[0].features.len();
        let train_window = (enriched_samples.len() as f64 * 0.6) as usize;
        let test_window = 20.min(enriched_samples.len() / 10);
        let step = test_window;

        let mut lstm_acc = 0.0;
        let mut regime_acc = 0.0;

        if let Some(wf) = ensemble::walk_forward_samples(coin_id, &enriched_samples, train_window, test_window, step) {
            lstm_acc = if wf.has_lstm { wf.lstm_accuracy } else { 0.0 };

            if let Some(rw) = regime::walk_forward_regime(coin_id, &enriched_samples, train_window, test_window, step) {
                regime_acc = rw.overall_accuracy;
            }

            accuracy_results.insert(coin_id.to_string(), AssetAccuracy {
                linreg: wf.linear_accuracy,
                logreg: wf.logistic_accuracy,
                gbt: wf.gbt_accuracy,
                lstm: lstm_acc,
                regime: regime_acc,
                ensemble: (wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy) / 3.0,
            });

            // Save final-fold models
            let last_train_end = {
                let mut s = 0; let mut last = 0;
                while s + train_window + test_window <= enriched_samples.len() { last = s + train_window; s += step; }
                last
            };
            let mut last_fold: Vec<ml::Sample> = enriched_samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
            let (means, stds) = ml::normalise(&mut last_fold);

            let mut lin = ml::LinearRegression::new(n_feat);
            lin.train(&last_fold, 0.005, 3000);
            let _ = model_store::save_weights(coin_id, "linreg", &lin.weights, lin.bias, n_feat, last_fold.len(), wf.linear_accuracy, &means, &stds);

            let mut log = ml::LogisticRegression::new(n_feat);
            log.train(&last_fold, 0.01, 3000);
            let _ = model_store::save_weights(coin_id, "logreg", &log.weights, log.bias, n_feat, last_fold.len(), wf.logistic_accuracy, &means, &stds);

            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);
            let gbt_config = gbt::GBTConfig { n_trees: 80, learning_rate: 0.08, tree_config: gbt::TreeConfig { max_depth: 4, min_samples_leaf: 8, min_samples_split: 16 }, subsample_ratio: 0.8, early_stopping_rounds: Some(8) };
            let gbt_model = gbt::GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), gbt_config);
            let _ = model_store::save_gbt(coin_id, &gbt_model, last_fold.len(), wf.gbt_accuracy, &means, &stds);

            println!("  ✓ All 5 models trained for {} [LinReg:{:.1}% LogReg:{:.1}% GBT:{:.1}% LSTM:{:.1}% Regime:{:.1}%]",
                coin_id, wf.linear_accuracy, wf.logistic_accuracy, wf.gbt_accuracy,
                lstm_acc, regime_acc);
        }

        if let Some(bt) = backtester::run_backtest(coin_id, &enriched_samples, &prices, train_window, test_window, step, &bt_config) {
            backtest_results.push(bt);
        }
    }

    if !backtest_results.is_empty() {
        backtester::print_backtest_summary(&backtest_results);

        // Persist backtest results to DB
        println!("\n━━━ PERSISTING BACKTEST & PORTFOLIO DATA ━━━\n");
        let model_version = model_store::MODEL_VERSION;
        for bt in &backtest_results {
            let asset_class = if stocks::STOCK_LIST.iter().any(|s| s.symbol == bt.symbol) {
                "stock"
            } else if stocks::FX_LIST.iter().any(|s| s.symbol == bt.symbol) {
                "fx"
            } else {
                "crypto"
            };
            let _ = database.insert_backtest_result(model_version, &bt.symbol, asset_class, bt);
        }
        println!("  Saved {} backtest results to DB", backtest_results.len());

        // Build and persist portfolio results for all 3 strategies
        let strategies = [
            ("sharpe", portfolio::WeightingScheme::SharpeWeighted),
            ("equal", portfolio::WeightingScheme::EqualWeight),
            ("inverse_volatility", portfolio::WeightingScheme::InverseVolatility),
        ];
        for (name, scheme) in &strategies {
            let cfg = portfolio::PortfolioConfig {
                initial_capital: 100_000.0,
                weighting: scheme.clone(),
                ..portfolio::PortfolioConfig::default()
            };
            if let Some(result) = portfolio::build_portfolio(&backtest_results, &cfg) {
                let _ = database.insert_portfolio_result(model_version, name, 100_000.0, &result);
                println!("  Saved portfolio result: {}", name);
            }
        }
    }

    // ── Save improved results and generate comparison report ──
    if !accuracy_results.is_empty() {
        let improved = serde_json::json!({
            "version": "v10_5model",
            "date": Utc::now().format("%Y-%m-%d").to_string(),
            "features": features::active_feature_count(),
            "models": ["LinReg", "LogReg", "GBT", "LSTM", "RegimeEnsemble"],
            "assets": accuracy_results.iter().map(|(k, v)| {
                (k.clone(), serde_json::json!({
                    "linreg": (v.linreg * 10.0).round() / 10.0,
                    "logreg": (v.logreg * 10.0).round() / 10.0,
                    "gbt": (v.gbt * 10.0).round() / 10.0,
                    "lstm": (v.lstm * 10.0).round() / 10.0,
                    "regime": (v.regime * 10.0).round() / 10.0,
                    "ensemble": (v.ensemble * 10.0).round() / 10.0,
                }))
            }).collect::<serde_json::Map<String, serde_json::Value>>()
        });

        let _ = std::fs::create_dir_all("reports");
        let _ = std::fs::write("reports/improved.json", serde_json::to_string_pretty(&improved).unwrap_or_default());

        generate_comparison_report(&accuracy_results, &ensemble_overrides);
        println!("  Comparison report: reports/improvement_report.html");
    }

    // ── Print 6-model summary table ──
    if !accuracy_results.is_empty() {
        println!("\n━━━ 5-MODEL ACCURACY SUMMARY ━━━\n");
        println!("  {:<16} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "Asset", "LinReg", "LogReg", "GBT", "LSTM", "Regime");
        println!("  {}", "-".repeat(56));
        let mut sorted_assets: Vec<&String> = accuracy_results.keys().collect();
        sorted_assets.sort();
        for asset in sorted_assets {
            let a = &accuracy_results[asset];
            println!("  {:<16} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}%",
                asset, a.linreg, a.logreg, a.gbt, a.lstm, a.regime);
        }
    }

    println!("\n━━━ TRAINING COMPLETE ━━━");
    println!("  Models: 5 (LinReg, LogReg, GBT, LSTM, RegimeEnsemble)");
    println!("  Features: {} active ({} total, {} pruned)",
        features::active_feature_count(), features::feature_names().len(),
        features::feature_names().len() - features::active_feature_count());
    println!("  Models saved to: models/");
    println!("  Database: rust_invest.db");
    let cached = model_store::list_cached_models();
    println!("  Cached models: {} files\n", cached.len());

    Ok(())
}

/// LSTM test mode — runs only SPY + MSFT with full diagnostic output
async fn run_lstm_test(database: &db::Database) -> Result<(), Box<dyn std::error::Error>> {
    let test_symbols = ["SPY", "MSFT"];

    // Build market context
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?.iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        market_histories.insert(ticker.to_string(), prices);
    }
    let market_context = features::build_market_context(&market_histories);

    for symbol in &test_symbols {
        println!("\n{}", "=".repeat(70));
        println!("  LSTM DIAGNOSTIC TEST: {}", symbol);
        println!("{}\n", "=".repeat(70));

        let points = database.get_stock_history(symbol)?;
        if points.is_empty() {
            println!("  ERROR: No data for {} in database. Run full training first to fetch data.", symbol);
            continue;
        }
        println!("  Data points: {}", points.len());

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(
            &prices, &volumes, &timestamps,
            Some(&market_context), "stock",
            features::sector_etf_for(symbol), None, None,
        );

        println!("  Samples built: {}", samples.len());
        if samples.is_empty() {
            println!("  ERROR: No samples produced for {}", symbol);
            continue;
        }

        let n_feat = samples[0].features.len();
        println!("  Features per sample: {}", n_feat);

        // Check for NaN/Inf in features
        let bad_count: usize = samples.iter()
            .map(|s| s.features.iter().filter(|x| !x.is_finite()).count())
            .sum();
        println!("  NaN/Inf values in features: {}", bad_count);

        // Label stats
        let n_up = samples.iter().filter(|s| s.label > 0.0).count();
        println!("  Labels: {:.1}% up, {:.1}% down",
            n_up as f64 / samples.len() as f64 * 100.0,
            (samples.len() - n_up) as f64 / samples.len() as f64 * 100.0);

        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        println!("  train_window={}, test_window={}, step={}", train_window, test_window, step);

        // Run full ensemble walk-forward (includes LSTM)
        println!("\n  --- Running walk-forward with LSTM ---\n");

        if let Some(wf) = ensemble::walk_forward_samples(symbol, &samples, train_window, test_window, step) {
            println!("\n  --- Results for {} ---", symbol);
            println!("  LinReg:  {:.1}% (recent: {:.1}%)", wf.linear_accuracy, wf.linear_recent);
            println!("  LogReg:  {:.1}% (recent: {:.1}%)", wf.logistic_accuracy, wf.logistic_recent);
            println!("  GBT:     {:.1}% (recent: {:.1}%)", wf.gbt_accuracy, wf.gbt_recent);
            println!("  LSTM:    {:.1}% (recent: {:.1}%) [has_lstm={}]",
                wf.lstm_accuracy, wf.lstm_recent, wf.has_lstm);
            println!("  Final probs: lin={:.3} log={:.3} gbt={:.3} lstm={:.3}",
                wf.final_linear_prob, wf.final_logistic_prob, wf.final_gbt_prob, wf.final_lstm_prob);
        } else {
            println!("  FAILED: walk_forward_samples returned None");
        }
    }

    println!("\n━━━ LSTM TEST COMPLETE ━━━\n");
    Ok(())
}

/// Compute ensemble accuracy for a walk-forward result using overrides
fn compute_ensemble_accuracy(wf: &ensemble::WalkForwardResult, ov: &ensemble::EnsembleOverride) -> f64 {
    let mut accs = Vec::new();
    if ov.use_linreg { accs.push(wf.linear_accuracy); }
    if ov.use_logreg { accs.push(wf.logistic_accuracy); }
    if ov.use_gbt { accs.push(wf.gbt_accuracy); }
    if wf.has_lstm { accs.push(wf.lstm_accuracy); }
    if accs.is_empty() {
        (wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy) / 3.0
    } else {
        accs.iter().sum::<f64>() / accs.len() as f64
    }
}

/// Generate comparison HTML report
fn generate_comparison_report(
    new_results: &HashMap<String, AssetAccuracy>,
    overrides: &HashMap<String, ensemble::EnsembleOverride>,
) {
    // Load baseline
    let baseline: HashMap<String, AssetAccuracy> = match std::fs::read_to_string("reports/baseline.json") {
        Ok(contents) => {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&contents) {
                if let Some(assets) = val.get("assets").and_then(|a| a.as_object()) {
                    assets.iter().map(|(k, v)| {
                        (k.clone(), AssetAccuracy {
                            linreg: v.get("linreg").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            logreg: v.get("logreg").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            gbt: v.get("gbt").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            lstm: v.get("lstm").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            regime: v.get("regime").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            ensemble: v.get("ensemble").and_then(|x| x.as_f64()).unwrap_or(0.0),
                        })
                    }).collect()
                } else { HashMap::new() }
            } else { HashMap::new() }
        }
        Err(_) => {
            println!("  [Report] No baseline found, skipping comparison");
            return;
        }
    };

    let n_features = features::active_feature_count();

    // Compute stats
    let mut total_before = 0.0_f64;
    let mut total_after = 0.0_f64;
    let mut improved = 0_usize;
    let mut degraded = 0_usize;
    let mut unchanged = 0_usize;
    let mut count = 0_usize;

    // All assets that appear in both
    let mut all_assets: Vec<String> = baseline.keys()
        .filter(|k| new_results.contains_key(*k))
        .cloned().collect();
    all_assets.sort();

    for asset in &all_assets {
        let before = baseline[asset].ensemble;
        let after = new_results[asset].ensemble;
        total_before += before;
        total_after += after;
        count += 1;
        let diff = after - before;
        if diff > 0.5 { improved += 1; }
        else if diff < -0.5 { degraded += 1; }
        else { unchanged += 1; }
    }

    let avg_before = if count > 0 { total_before / count as f64 } else { 0.0 };
    let avg_after = if count > 0 { total_after / count as f64 } else { 0.0 };

    let mut html = String::new();
    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Model Improvement Report</title>
<style>
body {{ background:#0a0e17; color:#e0e0e0; font-family:'Courier New',monospace; padding:32px; }}
h1 {{ color:#00d4aa; }} h2 {{ color:#00bcd4; margin-top:32px; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
th {{ background:#111827; color:#00d4aa; padding:10px 12px; text-align:right; border-bottom:1px solid #1f2937; }}
th:first-child {{ text-align:left; }}
td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #1f2937; font-size:13px; }}
td:first-child {{ text-align:left; font-weight:bold; }}
.up {{ color:#00e676; }} .down {{ color:#ff5252; }} .flat {{ color:#888; }}
.summary {{ background:#111827; border:1px solid #1f2937; border-radius:8px; padding:20px; margin:16px 0; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold; }}
.tag-up {{ background:rgba(0,230,118,0.15); color:#00e676; }}
.tag-down {{ background:rgba(255,82,82,0.15); color:#ff5252; }}
.tag-flat {{ background:rgba(136,136,136,0.15); color:#888; }}
</style></head><body>
<h1>Model Improvement Report</h1>
<p>Baseline: v8 (pruned features, 4-model ensemble)</p>
<p>Improved: v10 ({} features, 5-model ensemble — LinReg, LogReg, GBT, LSTM, RegimeEnsemble)</p>
<p>Generated: {}</p>
"#, n_features, Utc::now().format("%Y-%m-%d %H:%M UTC")));

    // Summary table
    html.push_str(r#"<h2>Overall Summary</h2><div class="summary"><table>
<tr><th style="text-align:left">Metric</th><th>Before</th><th>After</th><th>Change</th></tr>"#);
    html.push_str(&format!(
        "<tr><td>Average Ensemble Accuracy</td><td>{:.1}%</td><td>{:.1}%</td><td class='{}'>{:+.1}pp</td></tr>",
        avg_before, avg_after,
        if avg_after > avg_before { "up" } else { "down" },
        avg_after - avg_before
    ));
    html.push_str(&format!(
        "<tr><td>Feature Count</td><td>83</td><td>{}</td><td>{}</td></tr>",
        n_features, n_features as i32 - 83
    ));
    html.push_str(&format!(
        "<tr><td>Assets Improved</td><td>-</td><td>{}/{}</td><td>-</td></tr>",
        improved, count
    ));
    html.push_str(&format!(
        "<tr><td>Assets Degraded</td><td>-</td><td>{}/{}</td><td>-</td></tr>",
        degraded, count
    ));
    html.push_str("</table></div>");

    // Per-asset comparison table
    html.push_str(r#"<h2>Per-Asset Comparison</h2><table>
<tr><th style="text-align:left">Asset</th><th>Before (Ens)</th><th>After (Ens)</th><th>Change</th>
<th>LinReg</th><th>LogReg</th><th>GBT</th><th>LSTM</th><th>Regime</th><th>Status</th></tr>"#);

    for asset in &all_assets {
        let before = &baseline[asset];
        let after = &new_results[asset];
        let diff = after.ensemble - before.ensemble;
        let (cls, status) = if diff > 0.5 { ("up", "IMPROVED") }
            else if diff < -0.5 { ("down", "DEGRADED") }
            else { ("flat", "UNCHANGED") };
        let tag_cls = if diff > 0.5 { "tag-up" } else if diff < -0.5 { "tag-down" } else { "tag-flat" };

        html.push_str(&format!(
            "<tr><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td class='{}'>{:+.1}pp</td>\
             <td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td>\
             <td>{:.1}%</td><td>{:.1}%</td>\
             <td><span class='tag {}'>{}</span></td></tr>\n",
            asset, before.ensemble, after.ensemble, cls, diff,
            after.linreg, after.logreg, after.gbt,
            after.lstm, after.regime,
            tag_cls, status,
        ));
    }
    html.push_str("</table>");

    // Recommendations
    html.push_str(r#"<h2>Recommendations for Next Improvement Cycle</h2><ul>"#);
    for asset in &all_assets {
        let after = &new_results[asset];
        if after.ensemble < 60.0 {
            html.push_str(&format!("<li>{} still below 60% ({:.1}%) — consider different approach</li>", asset, after.ensemble));
        }
    }
    for asset in &all_assets {
        let before = &baseline[asset];
        let after = &new_results[asset];
        if after.ensemble < before.ensemble - 0.5 {
            html.push_str(&format!("<li>{} degraded by {:.1}pp — consider reverting to default ensemble</li>", asset, before.ensemble - after.ensemble));
        }
    }
    html.push_str("<li>LSTM status: fixed tensor striding bug, check training output above</li>");
    html.push_str("<li>Run diagnostics to find more near-zero importance features to prune</li>");
    html.push_str("</ul>");

    html.push_str("</body></html>");

    let _ = std::fs::write("reports/improvement_report.html", &html);
}
