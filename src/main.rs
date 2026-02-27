mod models;
mod crypto;
mod stocks;
mod db;
mod analysis;
mod report;
mod charts;
mod ml;
mod gbt;
mod ensemble;
mod features;
mod lstm;
mod model_store;
mod backtester;
mod portfolio;

use chrono::Utc;
use tokio::time::{sleep, Duration};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // ── Open database ──
    let database = db::Database::new("rust_invest.db")?;
    println!("Database opened successfully.\n");

    // ════════════════════════════════════════
    // PART 1: Fetch and store live crypto
    // ════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║            RUST INVEST - Market Dashboard                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("━━━ TOP 20 CRYPTOCURRENCIES ━━━\n");

    let coins = crypto::fetch_top_coins(&client).await?;
    let now = Utc::now().to_rfc3339();

    println!(
        "{:<5} {:<15} {:<6} {:>12} {:>10} {:>14} {:>14}",
        "Rank", "Name", "Sym", "Price", "24h %", "24h High", "24h Low"
    );
    println!("{}", "─".repeat(80));

    for coin in &coins {
        let rank = coin.market_cap_rank.unwrap_or(0);
        let change = coin.price_change_percentage_24h.unwrap_or(0.0);
        let high = coin.high_24h.unwrap_or(0.0);
        let low = coin.low_24h.unwrap_or(0.0);
        let arrow = if change >= 0.0 { "▲" } else { "▼" };

        println!(
            "{:<5} {:<15} {:<6} {:>12.2} {:>8.2}% {} {:>12.2} {:>12.2}",
            rank, coin.name, coin.symbol.to_uppercase(),
            coin.current_price, change, arrow, high, low
        );

        database.insert_crypto(&coin, &now)?;
    }

    println!("\n  ✓ Stored {} crypto prices in database\n", coins.len());

    // ════════════════════════════════════════
    // PART 2: Backfill historical data (crypto)
    // ════════════════════════════════════════
    println!("━━━ LOADING HISTORICAL DATA (365 days) ━━━\n");

    let top_coins = &coins[..5];

    for coin in top_coins {
        let existing = database.count_crypto_history(&coin.id)?;

        if existing > 0 {
            println!("  {} — already have {} records, skipping",
                     coin.name, existing);
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
                println!("    ✓ Stored {} data points for {}", count, coin.name);
            }
            Err(e) => {
                println!("    ✗ Failed to fetch {}: {}", coin.name, e);
            }
        }
    }

    let total = database.count_all_history()?;
    println!("\n  Total historical records in database: {}\n", total);

    // ════════════════════════════════════════
    // PART 3: Analyse historical data (crypto)
    // ════════════════════════════════════════
    println!("━━━ TECHNICAL ANALYSIS ━━━\n");

    let coin_ids = database.get_all_coin_ids()?;

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;

        if points.len() < 30 {
            println!("  {} — not enough data ({} points)\n", coin_id, points.len());
            continue;
        }

        let (from, to) = database.get_price_range(coin_id)?;
        println!("  Data range: {} to {}", &from[..10], &to[..10]);

        let result = analysis::analyse_coin(coin_id, &points);
        analysis::print_report(&result);

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let sma_7 = analysis::sma(&prices, 7);
        let sma_30 = analysis::sma(&prices, 30);

        if let (Some(&short), Some(&long)) = (sma_7.last(), sma_30.last()) {
            let trend = if short > long {
                "BULLISH (7-day SMA above 30-day SMA)"
            } else {
                "BEARISH (7-day SMA below 30-day SMA)"
            };
            println!("  Trend signal: {}\n", trend);
        }
    }

    // ════════════════════════════════════════
    // PART 4: Cross-coin comparison
    // ════════════════════════════════════════
    println!("━━━ CROSS-COIN COMPARISON ━━━\n");

    println!(
        "{:<12} {:>10} {:>10} {:>12} {:>10} {:>8}",
        "Coin", "Price", "Volatility", "Avg Return", "RSI", "Signal"
    );
    println!("{}", "─".repeat(65));

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 30 { continue; }

        let result = analysis::analyse_coin(coin_id, &points);
        let rsi_val = result.rsi_14.unwrap_or(0.0);
        let signal = if rsi_val > 70.0 { "OVER" }
        else if rsi_val < 30.0 { "UNDER" }
        else { "OK" };

        println!(
            "{:<12} {:>10.2} {:>10.2} {:>11.4}% {:>10.2} {:>8}",
            coin_id, result.current_price, result.std_dev,
            result.daily_returns_mean, rsi_val, signal
        );
    }

    // Report data for crypto
    let mut report_data: Vec<(String, Vec<analysis::PricePoint>, analysis::AnalysisResult)> = Vec::new();

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 30 { continue; }
        let result = analysis::analyse_coin(coin_id, &points);
        report_data.push((coin_id.clone(), points, result));
    }

    // ════════════════════════════════════════
    // PART 5: Stock quotes & history
    // ════════════════════════════════════════
    println!("\n━━━ KEY STOCKS & INDICES (Yahoo Finance) ━━━\n");

    println!(
        "{:<6} {:<16} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Sym", "Name", "Price", "Change", "Change%", "High", "Low"
    );
    println!("{}", "─".repeat(80));

    for stock in stocks::STOCK_LIST {
        match stocks::fetch_quote(&client, stock.symbol).await {
            Ok(q) => {
                let arrow = if q.change >= 0.0 { "▲" } else { "▼" };
                println!(
                    "{:<6} {:<16} {:>10.2} {:>8.2} {} {:>8.2}% {:>10.2} {:>10.2}",
                    stock.symbol, stock.name, q.price, q.change, arrow,
                    q.change_percent, q.high, q.low
                );

                database.insert_stock(
                    stock.symbol, stock.name, q.price, q.change,
                    &format!("{:.4}%", q.change_percent),
                    q.high, q.low,
                    &q.volume.to_string(), &now
                )?;
            }
            Err(_) => {
                println!(
                    "{:<6} {:<16} -- error fetching --",
                    stock.symbol, stock.name
                );
            }
        }
    }

    // ── Load stock history (5 years) ──
    println!("\n━━━ LOADING STOCK HISTORY (5 years) ━━━\n");

    for stock in stocks::STOCK_LIST {
        let existing = database.count_stock_history(stock.symbol)?;

        if existing > 1000 {
            println!("  {} — already have {} records, skipping",
                     stock.symbol, existing);
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

                    database.insert_stock_history(
                        stock.symbol, *price,
                        volume.map(|v| v as f64),
                        &timestamp,
                    )?;
                    count += 1;
                }
                println!("    ✓ Stored {} data points for {}", count, stock.symbol);
            }
            Err(e) => {
                println!("    ✗ Failed to fetch {}: {}", stock.symbol, e);
            }
        }
    }

    // ════════════════════════════════════════
    // PART 5b: Fetch MARKET INDICATORS (VIX, treasuries, sectors, gold, dollar)
    // ════════════════════════════════════════
    println!("\n━━━ LOADING MARKET INDICATORS (5 years) ━━━\n");

    for ticker in features::MARKET_TICKERS {
        let existing = database.count_market_history(ticker)?;

        if existing > 1000 {
            println!("  {} — already have {} records, skipping",
                     ticker, existing);
            continue;
        }

        // Yahoo Finance uses the ticker as-is for indices (^VIX, ^TNX, etc.)
        println!("  Fetching {} 5-year history...", ticker);

        match stocks::fetch_history(&client, ticker, "5y").await {
            Ok(points) => {
                let mut count = 0;
                for (ts, price, volume) in &points {
                    let timestamp = chrono::DateTime::from_timestamp(*ts, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();

                    database.insert_market_history(
                        ticker, *price,
                        volume.map(|v| v as f64),
                        &timestamp,
                    )?;
                    count += 1;
                }
                println!("    ✓ Stored {} data points for {}", count, ticker);
            }
            Err(e) => {
                println!("    ✗ Failed to fetch {}: {}", ticker, e);
            }
        }
    }

    // ── Build MarketContext from stored data ──
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();

    // Load SPY into market context too
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?
        .iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);

    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        if !prices.is_empty() {
            println!("  {} — loaded {} prices for market context",
                     ticker, prices.len());
        }
        market_histories.insert(ticker.to_string(), prices);
    }

    let market_context = features::build_market_context(&market_histories);
    println!("  ✓ Market context built ({} indicators)\n",
             features::MARKET_TICKERS.len());

    // ── Analyse stocks ──
    println!("\n━━━ STOCK ANALYSIS ━━━\n");

    let mut stock_report_data: Vec<(String, Vec<analysis::PricePoint>, analysis::AnalysisResult)> = Vec::new();

    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 30 { continue; }
        let result = analysis::analyse_coin(stock.symbol, &points);
        analysis::print_report(&result);

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let sma_7 = analysis::sma(&prices, 7);
        let sma_30 = analysis::sma(&prices, 30);

        if let (Some(&short), Some(&long)) = (sma_7.last(), sma_30.last()) {
            let trend = if short > long {
                "BULLISH (7-day SMA above 30-day SMA)"
            } else {
                "BEARISH (7-day SMA below 30-day SMA)"
            };
            println!("  Trend signal: {}\n", trend);
        }

        stock_report_data.push((stock.symbol.to_string(), points, result));
    }

    // ════════════════════════════════════════
    // PART 5c: Machine Learning (original pipelines for backward compat)
    // ════════════════════════════════════════
    println!("\n━━━ MACHINE LEARNING ━━━");
    println!("  Models: Linear Regression + Logistic Regression + Gradient Boosted Trees");
    println!("  Rich features: {} per sample", features::feature_names().len());
    println!("  Models: LinReg + LogReg + GBT + LSTM (candle-nn)");
    println!("  Evaluation: Walk-forward with rolling retraining");

    // Show model cache status
    let n_features = features::feature_names().len();
    let all_symbols: Vec<&str> = stocks::STOCK_LIST.iter().map(|s| s.symbol).collect();
    model_store::print_cache_status(&all_symbols, n_features);

    let cached = model_store::list_cached_models();
    if !cached.is_empty() {
        println!("  Cached models: {} files in models/", cached.len());
    }
    println!();

    let mut ml_report_data: Vec<ml::PipelineResult> = Vec::new();
    let mut gbt_report_data: Vec<gbt::ExtendedPipelineResult> = Vec::new();

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 60 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();

        if let Some(result) = ml::run_pipeline(coin_id, &prices, &volumes, 0.8) {
            ml_report_data.push(result);
        }
        if let Some(result) = gbt::run_extended_pipeline(coin_id, &prices, &volumes, 0.6) {
            gbt_report_data.push(result);
        }
    }

    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 60 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();

        if let Some(result) = ml::run_pipeline(stock.symbol, &prices, &volumes, 0.8) {
            ml_report_data.push(result);
        }
        if let Some(result) = gbt::run_extended_pipeline(stock.symbol, &prices, &volumes, 0.6) {
            gbt_report_data.push(result);
        }
    }

    if !gbt_report_data.is_empty() {
        println!("━━━ ML RESULTS SUMMARY (3 Models) ━━━\n");
        println!("{:<14} {:>10} {:>10} {:>10} {:>10} {:>12}",
                 "Symbol", "LinReg %", "LogReg %", "GBT %", "Best", "Verdict");
        println!("{}", "─".repeat(72));

        for r in &gbt_report_data {
            let verdict = if r.best_direction_accuracy > 55.0 { "PROMISING" }
            else if r.best_direction_accuracy > 50.0 { "MARGINAL" }
            else { "NO EDGE" };
            let short_name = if r.best_model_name.contains("Gradient") { "GBT" }
            else if r.best_model_name.contains("Logistic") { "LogReg" }
            else { "LinReg" };
            println!("{:<14} {:>9.1}% {:>9.1}% {:>9.1}% {:>10} {:>12}",
                     r.linear_metrics.symbol,
                     r.linear_metrics.direction_accuracy,
                     r.logistic_metrics.direction_accuracy,
                     r.gbt_metrics.direction_accuracy,
                     short_name, verdict);
        }
        println!();
    }

    // ════════════════════════════════════════
    // PART 6: Ensemble — Walk-Forward + Rich Features + Buy/Hold/Sell
    // ════════════════════════════════════════
    println!("\n━━━ ENSEMBLE WALK-FORWARD (RICH FEATURES: {}) ━━━\n",
             features::feature_names().len());

    let mut signals: Vec<ensemble::TradingSignal> = Vec::new();

    // Stocks — use market context
    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        // Build rich features with market context
        let samples = features::build_rich_features(
            &prices, &volumes, &timestamps,
            Some(&market_context), "stock",
        );

        if samples.len() < 100 {
            println!("  {} — only {} rich samples, skipping ensemble", stock.symbol, samples.len());
            continue;
        }

        // Check if we have valid cached models
        let n_feat = samples[0].features.len();
        let use_cache = model_store::has_valid_models(stock.symbol, n_feat);

        if use_cache {
            println!("  {} — using cached models (valid, < 7 days old)", stock.symbol);
        }

        // Walk-forward on rich features (trains fresh models)
        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        if let Some(wf) = ensemble::walk_forward_samples(
            stock.symbol, &samples, train_window, test_window, step,
        ) {
            // Save final-fold models to disk for next run
            // (The last fold's trained models are the most recent)
            let last_train_end = {
                let mut s = 0;
                let mut last = 0;
                while s + train_window + test_window <= samples.len() {
                    last = s + train_window;
                    s += step;
                }
                last
            };

            // Normalise the last fold's training data to get norm params
            let mut last_fold: Vec<ml::Sample> = samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
            let (means, stds) = ml::normalise(&mut last_fold);

            // Train final models on the last fold for saving
            let mut lin = ml::LinearRegression::new(n_feat);
            lin.train(&last_fold, 0.005, 3000);
            let _ = model_store::save_weights(
                stock.symbol, "linreg", &lin.weights, lin.bias,
                n_feat, last_fold.len(), wf.linear_accuracy, &means, &stds,
            );

            let mut log = ml::LogisticRegression::new(n_feat);
            log.train(&last_fold, 0.01, 3000);
            let _ = model_store::save_weights(
                stock.symbol, "logreg", &log.weights, log.bias,
                n_feat, last_fold.len(), wf.logistic_accuracy, &means, &stds,
            );

            // Train and save GBT
            let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = last_fold.iter()
                .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);

            let gbt_config = gbt::GBTConfig {
                n_trees: 80,
                learning_rate: 0.08,
                tree_config: gbt::TreeConfig { max_depth: 4, min_samples_leaf: 8, min_samples_split: 16 },
                subsample_ratio: 0.8,
                early_stopping_rounds: Some(8),
            };
            let gbt_model = gbt::GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), gbt_config);
            let _ = model_store::save_gbt(
                stock.symbol, &gbt_model, last_fold.len(), wf.gbt_accuracy, &means, &stds,
            );

            let result = analysis::analyse_coin(stock.symbol, &points);
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);
            let trend = match (sma_7.last(), sma_30.last()) {
                (Some(s), Some(l)) if s > l => "BULLISH",
                _ => "BEARISH",
            };

            let signal = ensemble::ensemble_signal(
                &wf,
                result.current_price,
                result.rsi_14.unwrap_or(50.0),
                trend,
            );
            signals.push(signal);
        }
    }

    // Crypto — no market context (only 1 year data, can't build 260-day lookback)
    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 200 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();

        // Use basic features for crypto (not enough history for rich)
        let train_window = (prices.len() as f64 * 0.6) as usize;
        let test_window = 20.min(prices.len() / 10);
        let step = test_window;

        if let Some(wf) = ensemble::walk_forward(
            coin_id, &prices, &volumes, None,
            train_window, test_window, step,
        ) {
            let result = analysis::analyse_coin(coin_id, &points);
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);
            let trend = match (sma_7.last(), sma_30.last()) {
                (Some(s), Some(l)) if s > l => "BULLISH",
                _ => "BEARISH",
            };

            let signal = ensemble::ensemble_signal(
                &wf,
                result.current_price,
                result.rsi_14.unwrap_or(50.0),
                trend,
            );
            signals.push(signal);
        }
    }

    // Print the trading signals
    if !signals.is_empty() {
        println!();
        ensemble::print_signals(&signals);
    }

    // ════════════════════════════════════════
    // PART 6b: Backtester — Walk-Forward Replay
    // ════════════════════════════════════════
    println!("\n━━━ BACKTESTING — Walk-Forward Replay ━━━\n");

    let bt_config = backtester::BacktestConfig::default();
    let mut backtest_results: Vec<backtester::BacktestResult> = Vec::new();

    // Backtest stocks (rich features)
    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(
            &prices, &volumes, &timestamps,
            Some(&market_context), "stock",
        );

        if samples.len() < 100 { continue; }

        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        if let Some(bt) = backtester::run_backtest(
            stock.symbol, &samples, &prices,
            train_window, test_window, step, &bt_config,
        ) {
            backtest_results.push(bt);
        }
    }

    // Backtest crypto (basic features)
    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 200 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();

        let samples = gbt::build_extended_features(&prices, &volumes);
        if samples.len() < 100 { continue; }

        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 20.min(samples.len() / 10);
        let step = test_window;

        if let Some(bt) = backtester::run_backtest(
            coin_id, &samples, &prices,
            train_window, test_window, step, &bt_config,
        ) {
            backtest_results.push(bt);
        }
    }

    if !backtest_results.is_empty() {
        println!();
        backtester::print_backtest_summary(&backtest_results);
    }

    // ════════════════════════════════════════
    // PART 6c: Portfolio Allocation — $100K
    // ════════════════════════════════════════
    println!("\n━━━ PORTFOLIO ALLOCATION ━━━\n");

    let mut portfolio_results: Vec<portfolio::PortfolioResult> = Vec::new();

    // Run all three weighting schemes
    let schemes = vec![
        portfolio::PortfolioConfig {
            weighting: portfolio::WeightingScheme::SharpeWeighted,
            ..portfolio::PortfolioConfig::default()
        },
        portfolio::PortfolioConfig {
            weighting: portfolio::WeightingScheme::EqualWeight,
            ..portfolio::PortfolioConfig::default()
        },
        portfolio::PortfolioConfig {
            weighting: portfolio::WeightingScheme::InverseVolatility,
            ..portfolio::PortfolioConfig::default()
        },
    ];

    for config in &schemes {
        if let Some(pr) = portfolio::build_portfolio(&backtest_results, config) {
            portfolio_results.push(pr);
        }
    }

    // ════════════════════════════════════════
    // PART 7: Generate report
    // ════════════════════════════════════════
    println!("\n━━━ GENERATING REPORT ━━━\n");

    report::generate_html_report(
        &report_data, &stock_report_data,
        &ml_report_data, &gbt_report_data,
        &signals,
        &backtest_results,
        &portfolio_results,
        "report.html"
    )?;
    println!("  ✓ Report saved to: report.html\n");

    // ════════════════════════════════════════
    // PART 8: Summary
    // ════════════════════════════════════════
    println!("\n━━━ SUMMARY ━━━\n");

    let best = coins.iter()
        .max_by(|a, b| {
            a.price_change_percentage_24h.unwrap_or(0.0)
                .partial_cmp(&b.price_change_percentage_24h.unwrap_or(0.0))
                .unwrap()
        })
        .unwrap();

    let worst = coins.iter()
        .min_by(|a, b| {
            a.price_change_percentage_24h.unwrap_or(0.0)
                .partial_cmp(&b.price_change_percentage_24h.unwrap_or(0.0))
                .unwrap()
        })
        .unwrap();

    println!("  Best 24h crypto:  {} ({:+.2}%)",
             best.name, best.price_change_percentage_24h.unwrap_or(0.0));
    println!("  Worst 24h crypto: {} ({:+.2}%)",
             worst.name, worst.price_change_percentage_24h.unwrap_or(0.0));
    println!("  Historical data:  {} total records", total);
    println!("  Coins analysed:   {}", coin_ids.len());
    println!("  Stocks analysed:  {}", stock_report_data.len());
    println!("  Trading signals:  {}", signals.len());
    println!("  Backtest results: {}", backtest_results.len());
    println!("  Portfolios:       {}", portfolio_results.len());
    println!("  Feature count:    {} (rich) / 14 (basic)", features::feature_names().len());
    println!("  ML models:        LinReg + LogReg + GBT + LSTM");
    let cached = model_store::list_cached_models();
    if !cached.is_empty() {
        println!("  Cached models:    {} files", cached.len());
    }
    println!("\n  Database saved to: rust_invest.db");
    println!("  Report saved to:   report.html\n");

    Ok(())
}
