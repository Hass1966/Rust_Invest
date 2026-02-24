mod models;
mod crypto;
mod stocks;
mod db;

use chrono::Utc;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let api_key = "GK86UK0Z0ECNUDBN";

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

        // Store in database
        database.insert_crypto(&coin, &now)?;
    }

    println!("\n  ✓ Stored {} crypto prices in database\n", coins.len());

    // ════════════════════════════════════════
    // PART 2: Backfill historical crypto data
    // ════════════════════════════════════════
    println!("━━━ LOADING HISTORICAL DATA (365 days) ━━━\n");

    // Top 5 coins by market cap
    let top_coins = &coins[..5];

    for coin in top_coins {
        let existing = database.count_crypto_history(&coin.id)?;

        if existing > 0 {
            println!("  {} — already have {} records, skipping",
                     coin.name, existing);
            continue;
        }

        println!("  Fetching {} history...", coin.name);

        // CoinGecko rate limit: ~10 calls/min for free tier
        sleep(Duration::from_secs(12)).await;

        match crypto::fetch_history(&client, &coin.id, 365).await {
            Ok(chart) => {
                let mut count = 0;

                for (i, price_point) in chart.prices.iter().enumerate() {
                    let timestamp_ms = price_point[0] as i64;
                    let price = price_point[1];

                    // Get matching volume if available
                    let volume = chart.total_volumes
                        .get(i)
                        .map(|v| v[1]);

                    let timestamp = chrono::DateTime::from_timestamp_millis(timestamp_ms)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();

                    database.insert_history(
                        &coin.id, price, volume, &timestamp
                    )?;
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
    // PART 3: Fetch and store stock quotes
    // ════════════════════════════════════════
    println!("━━━ KEY STOCKS & INDICES ━━━\n");

    println!(
        "{:<6} {:<16} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Sym", "Name", "Price", "Change", "Change%", "High", "Low"
    );
    println!("{}", "─".repeat(80));

    for stock in stocks::STOCK_LIST {
        sleep(Duration::from_secs(15)).await;

        match stocks::fetch_quote(&client, stock.symbol, api_key).await {
            Ok(q) => {
                let price: f64 = q.price.parse().unwrap_or(0.0);
                let change: f64 = q.change.parse().unwrap_or(0.0);
                let high: f64 = q.high.parse().unwrap_or(0.0);
                let low: f64 = q.low.parse().unwrap_or(0.0);
                let arrow = if change >= 0.0 { "▲" } else { "▼" };

                println!(
                    "{:<6} {:<16} {:>10.2} {:>8.2} {} {:>10} {:>10.2} {:>10.2}",
                    stock.symbol, stock.name, price, change, arrow,
                    q.change_percent, high, low
                );

                database.insert_stock(
                    stock.symbol, stock.name, price, change,
                    &q.change_percent, high, low, &q.volume, &now
                )?;
            }
            Err(_) => {
                println!(
                    "{:<6} {:<16} -- API limit or error --",
                    stock.symbol, stock.name
                );
            }
        }
    }

    // ════════════════════════════════════════
    // PART 4: Summary
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
    println!("\n  Database saved to: rust_invest.db");
    println!("  Run again to add more snapshots!\n");

    Ok(())
}