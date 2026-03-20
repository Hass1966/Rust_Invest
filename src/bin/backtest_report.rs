/// backtest_report — Run multi-frequency backtest comparison across all assets
/// ===========================================================================
/// Outputs: exports/backtest_comparison.csv, exports/backtest_report.md

use rust_invest::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     MULTI-FREQUENCY BACKTEST COMPARISON ENGINE                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let database = db::Database::new("rust_invest.db")?;

    // Build market context
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")?.iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker)?;
        market_histories.insert(ticker.to_string(), prices);
    }
    let market_context = features::build_market_context(&market_histories);

    let modes = [
        backtest_compare::BacktestMode::Academic,
        backtest_compare::BacktestMode::Realistic,
        backtest_compare::BacktestMode::SignalOnly,
    ];
    let freqs = [
        backtest_compare::Frequency::Daily,
        backtest_compare::Frequency::Weekly,
        backtest_compare::Frequency::Monthly,
    ];

    let mut all_rows: Vec<backtest_compare::ComparisonRow> = Vec::new();

    // ── Stocks ──
    println!("━━━ BACKTESTING STOCKS ━━━\n");
    for stock in stocks::STOCK_LIST {
        let points = database.get_stock_history(stock.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "stock", features::sector_etf_for(stock.symbol), None, None);
        if samples.len() < 100 { continue; }

        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        for mode in &modes {
            for freq in &freqs {
                if let Some(row) = backtest_compare::run_comparison(
                    stock.symbol, "stock", &samples, &prices, &timestamps,
                    train_window, test_window, step, *mode, *freq,
                ) {
                    all_rows.push(row);
                }
            }
        }
        print!("  {} ", stock.symbol);
    }
    println!("\n");

    // ── FX ──
    println!("━━━ BACKTESTING FX ━━━\n");
    for fx in stocks::FX_LIST {
        let points = database.get_fx_history(fx.symbol)?;
        if points.len() < 300 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let samples = features::build_rich_features(&prices, &volumes, &timestamps, Some(&market_context), "fx", Some(fx.symbol), None, None);
        if samples.len() < 100 { continue; }

        let train_window = (samples.len() as f64 * 0.6) as usize;
        let test_window = 30.min(samples.len() / 10);
        let step = test_window;

        for mode in &modes {
            for freq in &freqs {
                if let Some(row) = backtest_compare::run_comparison(
                    fx.symbol, "fx", &samples, &prices, &timestamps,
                    train_window, test_window, step, *mode, *freq,
                ) {
                    all_rows.push(row);
                }
            }
        }
        print!("  {} ", fx.symbol);
    }
    println!("\n");

    // ── Crypto ──
    println!("━━━ BACKTESTING CRYPTO ━━━\n");
    let coin_ids: Vec<String> = database.get_all_coin_ids()?.into_iter().filter(|id| id != "tether").collect();

    for coin_id in &coin_ids {
        let points = database.get_coin_history(coin_id)?;
        if points.len() < 200 { continue; }

        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
        let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

        let base_samples = gbt::build_extended_features(&prices, &volumes);
        if base_samples.is_empty() || base_samples.len() < 100 { continue; }

        let train_window = (base_samples.len() as f64 * 0.6) as usize;
        let test_window = 20.min(base_samples.len() / 10);
        let step = test_window;

        for mode in &modes {
            for freq in &freqs {
                if let Some(row) = backtest_compare::run_comparison(
                    coin_id, "crypto", &base_samples, &prices, &timestamps,
                    train_window, test_window, step, *mode, *freq,
                ) {
                    all_rows.push(row);
                }
            }
        }
        print!("  {} ", coin_id);
    }
    println!("\n");

    // ── Output ──
    let _ = std::fs::create_dir_all("exports");
    backtest_compare::write_csv(&all_rows, "exports/backtest_comparison.csv")?;
    backtest_compare::write_report(&all_rows, "exports/backtest_report.md")?;

    println!("━━━ BACKTEST COMPARISON COMPLETE ━━━");
    println!("  Total rows: {}", all_rows.len());
    println!("  CSV: exports/backtest_comparison.csv");
    println!("  Report: exports/backtest_report.md\n");

    // Quick summary
    let academic_daily: Vec<&backtest_compare::ComparisonRow> = all_rows.iter()
        .filter(|r| matches!(r.mode, backtest_compare::BacktestMode::Academic)
            && matches!(r.frequency, backtest_compare::Frequency::Daily))
        .collect();
    if !academic_daily.is_empty() {
        let beating = academic_daily.iter().filter(|r| r.vs_bh_pct > 0.0).count();
        let avg_sharpe = academic_daily.iter().map(|r| r.sharpe).sum::<f64>() / academic_daily.len() as f64;
        println!("  Academic Daily: {}/{} assets beat B&H, avg Sharpe {:.2}",
            beating, academic_daily.len(), avg_sharpe);
    }

    Ok(())
}
