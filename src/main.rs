mod models;
mod crypto;
mod stocks;
mod db;
mod analysis;
mod report;
mod charts;
mod ml;

use chrono::Utc;
use tokio::time::{sleep, Duration};

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
    // PART 2: Backfill historical data
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
    // PART 3: Analyse historical data
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

        // Show recent trend
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
        if points.len() < 30 {
            continue;
        }

        let result = analysis::analyse_coin(coin_id, &points);

        let rsi_val = result.rsi_14.unwrap_or(0.0);
        let signal = if rsi_val > 70.0 {
            "OVER"
        } else if rsi_val < 30.0 {
            "UNDER"
        } else {
            "OK"
        };

        println!(
            "{:<12} {:>10.2} {:>10.2} {:>11.4}% {:>10.2} {:>8}",
            coin_id,
            result.current_price,
            result.std_dev,
            result.daily_returns_mean,
            rsi_val,
            signal
        );
    }

    // ════════════════════════════════════════
    // PART 4b: Generate HTML report
    // ════════════════════════════════════════
    println!("\n━━━ GENERATING REPORT ━━━\n");

    let mut report_data: Vec<(String, Vec<analysis::PricePoint>, analysis::AnalysisResult)> = Vec::new();

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 30 {
            continue;
        }
        let result = analysis::analyse_coin(coin_id, &points);
        report_data.push((coin_id.clone(), points, result));
    }


    // ════════════════════════════════════════
    // PART 5: Stock quotes & history (Yahoo Finance)
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

    // ── Load stock history into database ──
    println!("\n━━━ LOADING STOCK HISTORY (1 year) ━━━\n");

    for stock in stocks::STOCK_LIST {
        let existing = database.count_stock_history(stock.symbol)?;
        if existing > 0 {
            println!("  {} — already have {} records, skipping",
                     stock.symbol, existing);
            continue;
        }

        println!("  Fetching {} history...", stock.symbol);

        match stocks::fetch_history(&client, stock.symbol, "1y").await {
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

    // ── Analyse stocks ──
    println!("\n━━━ STOCK ANALYSIS ━━━\n");

    let mut stock_report_data: Vec<(String, Vec<analysis::PricePoint>, analysis::AnalysisResult)> = Vec::new();

    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 30 {
            continue;
        }
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

    // ── Generate combined report ──
    println!("\n━━━ GENERATING REPORT ━━━\n");

    // ════════════════════════════════════════
    // PART 5b: Machine Learning
    // ════════════════════════════════════════
    println!("\n━━━ MACHINE LEARNING ━━━");
    println!("  Models: Linear Regression + Logistic Regression");
    println!("  Features: {} enhanced indicators (normalised)", ml::FEATURE_NAMES.len());
    println!("  Split: 80% train / 20% test (chronological)\n");

    let mut ml_report_data: Vec<ml::PipelineResult> = Vec::new();

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 60 { continue; }
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        if let Some(result) = ml::run_pipeline(coin_id, &prices, &volumes, 0.8) {
            ml_report_data.push(result);
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
    }

    if !ml_report_data.is_empty() {
        println!("━━━ ML RESULTS SUMMARY ━━━\n");
        println!("{:<14} {:>10} {:>10} {:>10} {:>10} {:>12}",
                 "Symbol", "LinReg %", "LogReg %", "Best %", "Model", "Verdict");
        println!("{}", "─".repeat(70));
        for r in &ml_report_data {
            let verdict = if r.best_direction_accuracy > 55.0 { "PROMISING" }
            else if r.best_direction_accuracy > 50.0 { "MARGINAL" }
            else { "NO EDGE" };
            let short_name = if r.best_model_name.contains("Logistic") { "LogReg" } else { "LinReg" };
            println!("{:<14} {:>9.1}% {:>9.1}% {:>9.1}% {:>10} {:>12}",
                     r.linear_metrics.symbol,
                     r.linear_metrics.direction_accuracy,
                     r.logistic_metrics.direction_accuracy,
                     r.best_direction_accuracy,
                     short_name, verdict);
        }
        println!();
    }

    report::generate_html_report(&report_data, &stock_report_data, &ml_report_data, "report.html")?;
    println!("  ✓ Report saved to: report.html\n");

    // ════════════════════════════════════════
    // PART 6: Summary
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
    println!("\n  Database saved to: rust_invest.db");
    println!("  Report saved to:   report.html\n");
    println!("  ML models trained: {}", ml_report_data.len());

    Ok(())
}