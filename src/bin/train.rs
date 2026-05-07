/// train — Weekly heavy job
/// ========================
/// Fetches all data, trains models, runs backtester, saves to models/ directory.
/// Usage: cargo run --release --bin train

use rust_invest::*;
use chrono::Utc;
use tokio::time::{sleep, Duration};
use std::collections::HashMap;
use rayon::prelude::*;

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

/// Regression accuracy results for one asset (v6+)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct RegressionAccuracy {
    ridge_mae: f64,
    lgbm_mae: f64,
    gru_mae: f64,
    ridge_dir_acc: f64,
    lgbm_dir_acc: f64,
    gru_dir_acc: f64,
}

/// Save training loss curves to JSON for chart generation
fn save_training_curves(symbol: &str, model_name: &str, train_losses: &[f64], val_losses: &[f64]) {
    let _ = std::fs::create_dir_all("reports/training_curves");
    let data: Vec<serde_json::Value> = train_losses.iter().zip(val_losses.iter()).enumerate()
        .map(|(epoch, (&tl, &vl))| serde_json::json!({
            "epoch": epoch + 1,
            "train_loss": (tl * 10000.0).round() / 10000.0,
            "val_loss": (vl * 10000.0).round() / 10000.0,
        }))
        .collect();
    let path = format!("reports/training_curves/{}_{}.json", symbol.to_lowercase(), model_name);
    if let Ok(json) = serde_json::to_string_pretty(&data) {
        let _ = std::fs::write(&path, json);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Limit Rayon to 6 cores — leaves 2 free for OS and running services
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()
        .unwrap();

    let args: Vec<String> = std::env::args().collect();
    let test_lstm = args.iter().any(|a| a == "--test-lstm");
    let horizon: usize = args.iter().position(|a| a == "--horizon")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let horizon_prefix = if horizon > 1 { format!("{}d_", horizon) } else { String::new() };

    let client = reqwest::Client::new();
    let database = db::Database::new("rust_invest.db")?;
    // Enable WAL mode for safe concurrent reads during parallel training
    database.set_wal_mode();

    // Load Polygon API key for primary price data
    let polygon_key = std::env::var("POLYGON_API_KEY").ok();

    if test_lstm {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║    ALPHA SIGNAL — LSTM TEST MODE (SPY + MSFT only)             ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
        return run_lstm_test(&database).await;
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         ALPHA SIGNAL — TRAIN MODE (Weekly Heavy Job)           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let now = Utc::now().to_rfc3339();

    // ── Crypto data fetch disabled (descoped) ──
    println!("━━━ CRYPTO DATA FETCH SKIPPED (descoped) ━━━\n");
    if false {
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
    } // end if false (crypto fetch disabled)

    // ── Fetch stocks (Polygon primary, Yahoo fallback) ──
    println!("\n━━━ FETCHING STOCK DATA (7 years) — Polygon primary, Yahoo fallback ━━━\n");
    for stock in stocks::STOCK_LIST {
        let existing = database.count_stock_history(stock.symbol)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", stock.symbol, existing);
            continue;
        }
        println!("  Fetching {} 7-year history...", stock.symbol);
        match polygon::fetch_history_with_fallback(&client, stock.symbol, polygon_key.as_deref(), "7y").await {
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

    // ── FX data fetch disabled (descoped) ──
    println!("\n━━━ FX DATA FETCH SKIPPED (descoped) ━━━\n");
    if false {
    for fx in stocks::FX_LIST {
        let existing = database.count_fx_history(fx.symbol)?;
        if existing > 1000 {
            println!("  {} — already have {} records, skipping", fx.symbol, existing);
            continue;
        }
        println!("  Fetching {} 7-year history...", fx.symbol);
        match polygon::fetch_history_with_fallback(&client, fx.symbol, polygon_key.as_deref(), "7y").await {
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
    } // end if false (FX fetch disabled)

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

    // ── Fetch BOE/ECB macro data (free APIs, no key needed) ──
    println!("\n━━━ FETCHING BOE/ECB MACRO DATA ━━━\n");
    let (boe_rate_res, gilt_res, ecb_rate_res, eu_infl_res) = tokio::join!(
        macro_data::fetch_boe_base_rate(&client),
        macro_data::fetch_uk_gilt_yield(&client),
        macro_data::fetch_ecb_refi_rate(&client),
        macro_data::fetch_eu_inflation(&client),
    );

    let boe_rates = boe_rate_res.unwrap_or_else(|e| { println!("  BOE base rate failed: {}", e); vec![] });
    let gilt_yields = gilt_res.unwrap_or_else(|e| { println!("  UK gilt yield failed: {}", e); vec![] });
    let ecb_rates = ecb_rate_res.unwrap_or_else(|e| { println!("  ECB refi rate failed: {}", e); vec![] });
    let eu_inflation = eu_infl_res.unwrap_or_else(|e| { println!("  EU inflation failed: {}", e); vec![] });

    // Store BOE/ECB data in market_history
    for (date, val) in &boe_rates {
        let _ = database.insert_market_history("BOE_RATE", *val, None, date);
    }
    for (date, val) in &gilt_yields {
        let _ = database.insert_market_history("UK_10Y_GILT", *val, None, date);
    }
    for (date, val) in &ecb_rates {
        let _ = database.insert_market_history("ECB_RATE", *val, None, date);
    }
    for (date, val) in &eu_inflation {
        let _ = database.insert_market_history("EU_INFLATION", *val, None, date);
    }
    println!("  Stored: BOE={}, Gilt={}, ECB={}, EU_HICP={} data points",
        boe_rates.len(), gilt_yields.len(), ecb_rates.len(), eu_inflation.len());

    // ── Fetch FRED series (HY credit spread, breakeven inflation) ──
    let fred_api_key = std::env::var("FRED_API_KEY").unwrap_or_default();
    if !fred_api_key.is_empty() {
        println!("\n━━━ FETCHING FRED SERIES ━━━\n");
        let (hy_res, be_res) = tokio::join!(
            macro_data::fetch_hy_spread(&client, &fred_api_key),
            macro_data::fetch_breakeven_inflation(&client, &fred_api_key),
        );
        let hy_spread = hy_res.unwrap_or_else(|e| { println!("  HY spread failed: {}", e); vec![] });
        let breakeven = be_res.unwrap_or_else(|e| { println!("  Breakeven inflation failed: {}", e); vec![] });
        for (date, val) in &hy_spread {
            let _ = database.insert_market_history("HY_SPREAD", *val, None, date);
        }
        for (date, val) in &breakeven {
            let _ = database.insert_market_history("BREAKEVEN_5Y", *val, None, date);
        }
        println!("  Stored: HY_SPREAD={}, BREAKEVEN_5Y={} data points", hy_spread.len(), breakeven.len());
    } else {
        println!("\n  Skipping FRED series (no FRED_API_KEY set)\n");
    }

    // ── Build market context ──
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?.iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        market_histories.insert(ticker.to_string(), prices);
    }
    // FRED series (stored in market_history by train)
    market_histories.insert("HY_SPREAD".to_string(), database.get_market_prices("HY_SPREAD").unwrap_or_default());
    market_histories.insert("BREAKEVEN_5Y".to_string(), database.get_market_prices("BREAKEVEN_5Y").unwrap_or_default());
    let market_context = features::build_market_context(&market_histories);

    // Build ExtendedMacro with BOE/ECB data
    let ext_macro = features::ExtendedMacro {
        dxy: database.get_market_prices("UUP").unwrap_or_default(),
        yield_spread: vec![], // populated from FRED if available
        fed_funds: vec![],
        boe_rate: boe_rates.iter().map(|(_, v)| *v).collect(),
        uk_10y_gilt: gilt_yields.iter().map(|(_, v)| *v).collect(),
        ecb_rate: ecb_rates.iter().map(|(_, v)| *v).collect(),
        eu_inflation: eu_inflation.iter().map(|(_, v)| *v).collect(),
        insider_score: 0.0,
        short_interest_ratio: 0.0,
    };

    // ── Train regression models for stocks (Ridge + LightGBM + GRU) ──
    println!("\n━━━ TRAINING REGRESSION MODELS (v6: Ridge + LightGBM + GRU) ━━━\n");
    let total_features = features::feature_names().len();
    println!("  Total features: {} (active: {}, pruned: {})",
        total_features, features::active_feature_count(), total_features - features::active_feature_count());

    let ensemble_overrides = ensemble::load_ensemble_overrides();
    let mut accuracy_results: HashMap<String, AssetAccuracy> = HashMap::new();
    let mut regression_results: HashMap<String, RegressionAccuracy> = HashMap::new();

    let mut backtest_results: Vec<backtester::BacktestResult> = Vec::new();
    let bt_config = backtester::BacktestConfig::default();

    // Pre-fetch all stock data from database (sequential reads)
    let stock_data: Vec<_> = stocks::STOCK_LIST.iter().filter_map(|stock| {
        database.get_stock_history(stock.symbol).ok()
            .filter(|points| points.len() >= 300)
            .map(|points| (stock, points))
    }).collect();
    println!("  Loaded {} stocks with sufficient data — training in parallel\n", stock_data.len());

    // Train all stocks in parallel using all CPU cores
    type RetrainLog = (String, f64, f64, i64, i64, i64, i64);
    let stock_results: Vec<(Option<(String, RegressionAccuracy, RetrainLog)>, Option<backtester::BacktestResult>)> =
        stock_data.par_iter().filter_map(|(stock, points)| {
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features_horizon(&prices, &volumes, &timestamps, Some(&market_context), "stock", features::sector_etf_for(stock.symbol), None, None, Some(&ext_macro), None, horizon);
        if samples.len() < 100 { return None; }

        let pre_acc = model_store::load_model_accuracy(stock.symbol);

        let n_feat = samples[0].features.len();
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        let mut accuracy_data = None;

        if let Some(reg) = ensemble::walk_forward_regression(stock.symbol, &samples, train_window, test_window, step) {
            // Save final-fold models
            let last_train_end = {
                let mut s = 0; let mut last = 0;
                while s + train_window + test_window <= samples.len() { last = s + train_window; s += step; }
                last
            };
            let mut last_fold: Vec<ml::Sample> = samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
            let (means, stds) = ml::normalise(&mut last_fold);
            let recency_weights = ensemble::compute_recency_weights(last_fold.len());

            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter().map(|s| s.label).collect(); // continuous % returns

            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);

            // === Save Ridge ===
            if let Ok(ridge) = ridge::RidgeRegression::train(x_t, y_t, 10.0) {
                if horizon_prefix.is_empty() {
                    let _ = ridge.save(stock.symbol, n_feat, last_fold.len(), reg.ridge_mae, &means, &stds);
                } else {
                    // Save 5d Ridge model with prefix
                    let _ = ridge.save_as(&format!("{}{}",horizon_prefix, stock.symbol.to_lowercase()), n_feat, last_fold.len(), reg.ridge_mae, &means, &stds);
                }
            }

            // === Save LightGBM Regressor ===
            let lgbm_recency: Vec<f64> = recency_weights[..x_t.len()].to_vec();
            let lgbm_config = lgbm::LGBMRegressorConfig::default();
            if let Ok(lgbm) = lgbm::LightGBMRegressor::train(x_t, y_t, Some(&lgbm_recency), Some(x_v), Some(y_v), &lgbm_config) {
                let save_path = if horizon_prefix.is_empty() {
                    model_store::lgbm_regressor_path(stock.symbol)
                } else {
                    model_store::lgbm_regressor_path_5d(stock.symbol)
                };
                let _ = lgbm.save(&save_path);
                // Save LGBM MAE separately for proper weighting at inference
                let _ = model_store::save_lgbm_mae(stock.symbol, reg.lgbm_mae, reg.lgbm_dir_acc);
            }

            // === Save GRU Regression ===
            let top_feature_indices: Vec<usize> = if !reg.lgbm_importance.is_empty() {
                let mut indexed: Vec<(usize, f64)> = reg.lgbm_importance.iter()
                    .enumerate().map(|(i, (_, imp))| (i, *imp)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.iter().take(30).map(|(idx, _)| *idx).collect()
            } else {
                (0..n_feat.min(30)).collect()
            };

            let gru_n_feat = if top_feature_indices.len() >= 20 { top_feature_indices.len() } else { n_feat };
            let gru_fi = if top_feature_indices.len() >= 20 { Some(top_feature_indices.as_slice()) } else { None };

            let gru_config = gru::GRURegressionConfig {
                input_size: gru_n_feat,
                ..gru::GRURegressionConfig::default()
            };

            if let Ok(mut gru_model) = gru::GRURegressionModel::new(gru_config) {
                let val_split = (last_fold.len() as f64 * 0.85) as usize;
                let (train_part, val_part) = last_fold.split_at(val_split);
                match gru_model.train_regression(train_part, val_part, gru_fi) {
                    Ok(result) => {
                        let gru_path = model_store::gru_path(stock.symbol);
                        let _ = gru_model.save(&gru_path);
                        let _ = model_store::save_gru_meta(
                            stock.symbol, gru_n_feat, gru_config.hidden_size,
                            gru_config.seq_length, train_part.len(), result.val_mae,
                            &means, &stds, gru_fi,
                        );
                        save_training_curves(stock.symbol, "gru", &result.train_losses, &result.val_losses);
                    }
                    Err(e) => println!("    [GRU-Reg] {} training failed: {}", stock.symbol, e),
                }
            }

            let post_acc = reg.ridge_dir_acc * 100.0;
            println!("  ✓ Regression models trained for {} [Ridge MAE:{:.4} Dir:{:.1}% | LGBM MAE:{:.4} Dir:{:.1}% | GRU MAE:{:.4} Dir:{:.1}%]",
                stock.symbol, reg.ridge_mae, reg.ridge_dir_acc * 100.0,
                reg.lgbm_mae, reg.lgbm_dir_acc * 100.0,
                reg.gru_mae, reg.gru_dir_acc * 100.0);

            accuracy_data = Some((stock.symbol.to_string(), RegressionAccuracy {
                ridge_mae: reg.ridge_mae,
                lgbm_mae: reg.lgbm_mae,
                gru_mae: reg.gru_mae,
                ridge_dir_acc: reg.ridge_dir_acc * 100.0,
                lgbm_dir_acc: reg.lgbm_dir_acc * 100.0,
                gru_dir_acc: reg.gru_dir_acc * 100.0,
            }, (stock.symbol.to_string(), pre_acc, post_acc,
                0, 0, 0, 0)));
        }

        let backtest = backtester::run_backtest(stock.symbol, &samples, &prices, train_window, test_window, step, &bt_config);
        Some((accuracy_data, backtest))
    }).collect();

    // Merge results and write retrain logs (sequential DB writes)
    for (accuracy_opt, backtest_opt) in stock_results {
        if let Some((symbol, acc, (sym, pre_acc, post_acc, buy, sell, short, hold))) = accuracy_opt {
            regression_results.insert(symbol, acc);
            let _ = database.insert_retrain_log(&sym, "regression", pre_acc, post_acc, buy, sell, short, hold);
        }
        if let Some(bt) = backtest_opt {
            backtest_results.push(bt);
        }
    }

    // ── FX training disabled (descoped — separate project) ──
    println!("\n━━━ FX TRAINING SKIPPED (descoped) ━━━\n");
    if false {
    println!("\n━━━ TRAINING FX REGRESSION MODELS ━━━\n");
    let fx_data: Vec<_> = stocks::FX_LIST.iter().filter_map(|fx| {
        database.get_fx_history(fx.symbol).ok()
            .filter(|points| points.len() >= 300)
            .map(|points| (fx, points))
    }).collect();
    println!("  Loaded {} FX pairs — training in parallel\n", fx_data.len());

    let fx_results: Vec<(Option<(String, RegressionAccuracy, RetrainLog)>, Option<backtester::BacktestResult>)> =
        fx_data.par_iter().filter_map(|(fx, points)| {
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();
        let samples = features::build_rich_features_horizon(&prices, &volumes, &timestamps, Some(&market_context), "fx", Some(fx.symbol), None, None, Some(&ext_macro), None, horizon);
        if samples.len() < 100 { return None; }

        let pre_acc = model_store::load_model_accuracy(fx.symbol);

        let n_feat = samples[0].features.len();
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        let mut accuracy_data = None;

        if let Some(reg) = ensemble::walk_forward_regression(fx.symbol, &samples, train_window, test_window, step) {
            let last_train_end = {
                let mut s = 0; let mut last = 0;
                while s + train_window + test_window <= samples.len() { last = s + train_window; s += step; }
                last
            };
            let mut last_fold: Vec<ml::Sample> = samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
            let (means, stds) = ml::normalise(&mut last_fold);
            let recency_weights = ensemble::compute_recency_weights(last_fold.len());

            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter().map(|s| s.label).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);

            if let Ok(ridge) = ridge::RidgeRegression::train(x_t, y_t, 10.0) {
                let _ = ridge.save(fx.symbol, n_feat, last_fold.len(), reg.ridge_mae, &means, &stds);
            }

            let lgbm_recency: Vec<f64> = recency_weights[..x_t.len()].to_vec();
            let lgbm_config = lgbm::LGBMRegressorConfig::default();
            if let Ok(lgbm) = lgbm::LightGBMRegressor::train(x_t, y_t, Some(&lgbm_recency), Some(x_v), Some(y_v), &lgbm_config) {
                let _ = lgbm.save(&model_store::lgbm_regressor_path(fx.symbol));
            }

            let top_fi: Vec<usize> = if !reg.lgbm_importance.is_empty() {
                let mut indexed: Vec<(usize, f64)> = reg.lgbm_importance.iter()
                    .enumerate().map(|(i, (_, imp))| (i, *imp)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.iter().take(30).map(|(idx, _)| *idx).collect()
            } else { (0..n_feat.min(30)).collect() };
            let gru_n = if top_fi.len() >= 20 { top_fi.len() } else { n_feat };
            let gru_fi = if top_fi.len() >= 20 { Some(top_fi.as_slice()) } else { None };
            let gru_config = gru::GRURegressionConfig { input_size: gru_n, ..gru::GRURegressionConfig::default() };

            if let Ok(mut gru_model) = gru::GRURegressionModel::new(gru_config) {
                let val_split = (last_fold.len() as f64 * 0.85) as usize;
                let (tp, vp) = last_fold.split_at(val_split);
                if let Ok(result) = gru_model.train_regression(tp, vp, gru_fi) {
                    let _ = gru_model.save(&model_store::gru_path(fx.symbol));
                    let _ = model_store::save_gru_meta(fx.symbol, gru_n, gru_config.hidden_size, gru_config.seq_length, tp.len(), result.val_mae, &means, &stds, gru_fi);
                    save_training_curves(fx.symbol, "gru", &result.train_losses, &result.val_losses);
                }
            }

            let post_acc = reg.ridge_dir_acc * 100.0;
            println!("  ✓ FX regression trained for {} [Ridge MAE:{:.4} | LGBM MAE:{:.4} | GRU MAE:{:.4}]",
                fx.symbol, reg.ridge_mae, reg.lgbm_mae, reg.gru_mae);

            accuracy_data = Some((fx.symbol.to_string(), RegressionAccuracy {
                ridge_mae: reg.ridge_mae, lgbm_mae: reg.lgbm_mae, gru_mae: reg.gru_mae,
                ridge_dir_acc: reg.ridge_dir_acc * 100.0, lgbm_dir_acc: reg.lgbm_dir_acc * 100.0, gru_dir_acc: reg.gru_dir_acc * 100.0,
            }, (fx.symbol.to_string(), pre_acc, post_acc, 0, 0, 0, 0)));
        }

        let backtest = backtester::run_backtest(fx.symbol, &samples, &prices, train_window, test_window, step, &bt_config);
        Some((accuracy_data, backtest))
    }).collect();

    for (accuracy_opt, backtest_opt) in fx_results {
        if let Some((symbol, acc, (sym, pre_acc, post_acc, buy, sell, short, hold))) = accuracy_opt {
            regression_results.insert(symbol, acc);
            let _ = database.insert_retrain_log(&sym, "regression", pre_acc, post_acc, buy, sell, short, hold);
        }
        if let Some(bt) = backtest_opt {
            backtest_results.push(bt);
        }
    }
    } // end if false (FX disabled)

    // ── Crypto training disabled (descoped — separate project) ──
    println!("━━━ CRYPTO TRAINING SKIPPED (descoped) ━━━\n");
    if false {
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

    // Pre-fetch and build enriched crypto samples (sequential DB reads)
    let crypto_prepared: Vec<_> = coin_ids.iter().filter_map(|coin_id| {
        let points = database.get_coin_history(coin_id).ok()?;
        if points.len() < 200 { return None; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let base_samples = gbt::build_extended_features_horizon(&prices, &volumes, horizon);
        if base_samples.is_empty() { return None; }

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

        if enriched_samples.len() < 100 { return None; }
        Some((coin_id.clone(), enriched_samples, prices))
    }).collect();
    println!("  Prepared {} crypto assets — training in parallel\n", crypto_prepared.len());

    let crypto_results: Vec<(Option<(String, RegressionAccuracy, RetrainLog)>, Option<backtester::BacktestResult>)> =
        crypto_prepared.par_iter().filter_map(|(coin_id, enriched_samples, prices)| {
        let pre_acc = model_store::load_model_accuracy(coin_id);

        let n_feat = enriched_samples[0].features.len();
        let train_window = (enriched_samples.len() as f64 * 0.6) as usize;
        let test_window = 20.min(enriched_samples.len() / 10);
        let step = test_window;

        let mut accuracy_data = None;

        if let Some(reg) = ensemble::walk_forward_regression(coin_id, enriched_samples, train_window, test_window, step) {
            let last_train_end = {
                let mut s = 0; let mut last = 0;
                while s + train_window + test_window <= enriched_samples.len() { last = s + train_window; s += step; }
                last
            };
            let mut last_fold: Vec<ml::Sample> = enriched_samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
            let (means, stds) = ml::normalise(&mut last_fold);
            let recency_weights = ensemble::compute_recency_weights(last_fold.len());

            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter().map(|s| s.label).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);

            if let Ok(ridge) = ridge::RidgeRegression::train(x_t, y_t, 10.0) {
                let _ = ridge.save(coin_id, n_feat, last_fold.len(), reg.ridge_mae, &means, &stds);
            }

            let lgbm_recency: Vec<f64> = recency_weights[..x_t.len()].to_vec();
            let lgbm_config = lgbm::LGBMRegressorConfig::default();
            if let Ok(lgbm) = lgbm::LightGBMRegressor::train(x_t, y_t, Some(&lgbm_recency), Some(x_v), Some(y_v), &lgbm_config) {
                let _ = lgbm.save(&model_store::lgbm_regressor_path(coin_id));
            }

            let top_fi: Vec<usize> = if !reg.lgbm_importance.is_empty() {
                let mut indexed: Vec<(usize, f64)> = reg.lgbm_importance.iter()
                    .enumerate().map(|(i, (_, imp))| (i, *imp)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.iter().take(30).map(|(idx, _)| *idx).collect()
            } else { (0..n_feat.min(30)).collect() };
            let gru_n = if top_fi.len() >= 20 { top_fi.len() } else { n_feat };
            let gru_fi = if top_fi.len() >= 20 { Some(top_fi.as_slice()) } else { None };
            let gru_config = gru::GRURegressionConfig { input_size: gru_n, ..gru::GRURegressionConfig::default() };

            if let Ok(mut gru_model) = gru::GRURegressionModel::new(gru_config) {
                let val_split = (last_fold.len() as f64 * 0.85) as usize;
                let (tp, vp) = last_fold.split_at(val_split);
                if let Ok(result) = gru_model.train_regression(tp, vp, gru_fi) {
                    let _ = gru_model.save(&model_store::gru_path(coin_id));
                    let _ = model_store::save_gru_meta(coin_id, gru_n, gru_config.hidden_size, gru_config.seq_length, tp.len(), result.val_mae, &means, &stds, gru_fi);
                    save_training_curves(coin_id, "gru", &result.train_losses, &result.val_losses);
                }
            }

            let post_acc = reg.ridge_dir_acc * 100.0;
            println!("  ✓ Crypto regression trained for {} [Ridge MAE:{:.4} | LGBM MAE:{:.4} | GRU MAE:{:.4}]",
                coin_id, reg.ridge_mae, reg.lgbm_mae, reg.gru_mae);

            accuracy_data = Some((coin_id.to_string(), RegressionAccuracy {
                ridge_mae: reg.ridge_mae, lgbm_mae: reg.lgbm_mae, gru_mae: reg.gru_mae,
                ridge_dir_acc: reg.ridge_dir_acc * 100.0, lgbm_dir_acc: reg.lgbm_dir_acc * 100.0, gru_dir_acc: reg.gru_dir_acc * 100.0,
            }, (coin_id.to_string(), pre_acc, post_acc, 0, 0, 0, 0)));
        }

        let backtest = backtester::run_backtest(coin_id, enriched_samples, prices, train_window, test_window, step, &bt_config);
        Some((accuracy_data, backtest))
    }).collect();

    for (accuracy_opt, backtest_opt) in crypto_results {
        if let Some((symbol, acc, (sym, pre_acc, post_acc, buy, sell, short, hold))) = accuracy_opt {
            regression_results.insert(symbol, acc);
            let _ = database.insert_retrain_log(&sym, "regression", pre_acc, post_acc, buy, sell, short, hold);
        }
        if let Some(bt) = backtest_opt {
            backtest_results.push(bt);
        }
    }
    } // end if false (crypto disabled)

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

    // ── Save regression results ──
    if !regression_results.is_empty() {
        let improved = serde_json::json!({
            "version": "v6_regression",
            "date": Utc::now().format("%Y-%m-%d").to_string(),
            "features": features::active_feature_count(),
            "models": ["Ridge", "LightGBM", "GRU"],
            "assets": regression_results.iter().map(|(k, v)| {
                (k.clone(), serde_json::json!({
                    "ridge_mae": (v.ridge_mae * 10000.0).round() / 10000.0,
                    "lgbm_mae": (v.lgbm_mae * 10000.0).round() / 10000.0,
                    "gru_mae": (v.gru_mae * 10000.0).round() / 10000.0,
                    "ridge_dir_acc": (v.ridge_dir_acc * 10.0).round() / 10.0,
                    "lgbm_dir_acc": (v.lgbm_dir_acc * 10.0).round() / 10.0,
                    "gru_dir_acc": (v.gru_dir_acc * 10.0).round() / 10.0,
                }))
            }).collect::<serde_json::Map<String, serde_json::Value>>()
        });

        let _ = std::fs::create_dir_all("reports");
        let _ = std::fs::write("reports/improved.json", serde_json::to_string_pretty(&improved).unwrap_or_default());
    }

    // ── Print regression summary table ──
    if !regression_results.is_empty() {
        println!("\n━━━ REGRESSION MODEL SUMMARY (v6) ━━━\n");
        println!("  {:<16} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9}",
            "Asset", "Ridge MAE", "LGBM MAE", "GRU MAE", "Ridge%", "LGBM%", "GRU%");
        println!("  {}", "-".repeat(80));
        let mut sorted_assets: Vec<&String> = regression_results.keys().collect();
        sorted_assets.sort();
        for asset in sorted_assets {
            let a = &regression_results[asset];
            println!("  {:<16} {:>10.4} {:>10.4} {:>10.4} {:>8.1}% {:>8.1}% {:>8.1}%",
                asset, a.ridge_mae, a.lgbm_mae, a.gru_mae,
                a.ridge_dir_acc, a.lgbm_dir_acc, a.gru_dir_acc);
        }
    }

    // ── Aggregate feature importance across all assets ──
    {
        let mut global_importance: HashMap<String, Vec<f64>> = HashMap::new();
        // Load importance from saved LightGBM models
        let feature_names = features::active_feature_names();
        for symbol in regression_results.keys() {
            let lgbm_path = model_store::lgbm_regressor_path(symbol);
            if let Ok(m) = lgbm::LightGBMRegressor::load(&lgbm_path, feature_names.len()) {
                let feat_refs: Vec<&str> = feature_names.iter().map(|s| s.as_str()).collect();
                let importance = m.feature_importance(&feat_refs);
                for (name, imp) in &importance {
                    global_importance.entry(name.clone()).or_default().push(*imp);
                }
            }
        }

        if !global_importance.is_empty() {
            // Rank by mean importance
            let mut ranked: Vec<(String, f64)> = global_importance.iter()
                .map(|(name, imps)| (name.clone(), imps.iter().sum::<f64>() / imps.len() as f64))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Print top 30
            println!("\n━━━ FEATURE IMPORTANCE (Top 30) ━━━\n");
            for (i, (name, imp)) in ranked.iter().take(30).enumerate() {
                println!("  {:>2}. {:<30} {:.6}", i + 1, name, imp);
            }

            // Save full report
            let report = serde_json::json!({
                "generated_at": chrono::Utc::now().to_rfc3339(),
                "n_assets": regression_results.len(),
                "n_features": feature_names.len(),
                "ranked_features": ranked.iter().enumerate().map(|(i, (name, imp))| {
                    serde_json::json!({
                        "rank": i + 1,
                        "name": name,
                        "mean_importance": (imp * 1_000_000.0).round() / 1_000_000.0,
                        "n_assets_present": global_importance.get(name).map(|v| v.len()).unwrap_or(0),
                    })
                }).collect::<Vec<_>>(),
                "recommended_top_30": ranked.iter().take(30)
                    .filter(|(name, _)| {
                        // Remove auto-correlated features from recommendations
                        !["autocorr_1d", "consecutive_up_days", "consecutive_down_days"].contains(&name.as_str())
                    })
                    .map(|(name, _)| name.clone())
                    .collect::<Vec<_>>(),
            });

            let _ = std::fs::create_dir_all("reports");
            let _ = std::fs::write("reports/feature_importance.json",
                serde_json::to_string_pretty(&report).unwrap_or_default());
            println!("\n  Feature importance report saved to reports/feature_importance.json");
        }
    }

    println!("\n━━━ TRAINING COMPLETE ━━━");
    println!("  Models: 2 (Ridge, LightGBM Regressor) — GRU disabled");
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
    market_histories.insert("HY_SPREAD".to_string(), database.get_market_prices("HY_SPREAD").unwrap_or_default());
    market_histories.insert("BREAKEVEN_5Y".to_string(), database.get_market_prices("BREAKEVEN_5Y").unwrap_or_default());
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

        let samples = features::build_rich_features_ext(
            &prices, &volumes, &timestamps,
            Some(&market_context), "stock",
            features::sector_etf_for(symbol), None, None,
            None, None,
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
    _overrides: &HashMap<String, ensemble::EnsembleOverride>,
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
    let mut _unchanged = 0_usize;
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
        else { _unchanged += 1; }
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
