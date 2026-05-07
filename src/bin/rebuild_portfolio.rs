/// rebuild_portfolio — Reconstruct paper portfolio from clean signal history
/// ==========================================================================
/// Reads deduped, stocks-only signals from PostgreSQL alpha_signal database.
/// Reads daily close prices from SQLite stock_history.
/// Builds a day-by-day portfolio simulation with:
///   - $100,000 starting capital
///   - Inverse-volatility weighting (60-day trailing)
///   - 15% per-position cap
///   - 5% cash buffer (min 5% always in cash)
///   - 5% rebalance threshold
///   - 10bps transaction cost per trade
///   - Hard assertion: reject trades where total_committed > capital × 0.95
///   - Log rejected trades to sizing_rejections table
///
/// Usage: cargo run --release --bin rebuild_portfolio

use rust_invest::{db, pg};
use std::collections::HashMap;

const STARTING_CAPITAL: f64 = 100_000.0;
const MAX_POSITION_PCT: f64 = 0.15;    // 15% per position
const CASH_BUFFER_PCT: f64 = 0.05;     // 5% always in cash
const REBALANCE_THRESHOLD: f64 = 0.05; // 5% weight drift before rebalance
fn tx_cost_bps(symbol: &str) -> f64 {
    if symbol.ends_with(".L") || symbol.ends_with(".DE") || symbol.ends_with(".PA") {
        30.0  // 50bps stamp duty on buy + 10bps sell, averaged
    } else {
        10.0  // US stocks/ETFs
    }
}
const VOL_LOOKBACK: usize = 60;        // 60 trading days for volatility calc

#[derive(Debug, Clone)]
struct Position {
    shares: f64,
    avg_cost: f64,
}

#[derive(Debug)]
struct PortfolioState {
    cash: f64,
    positions: HashMap<String, Position>,
}

impl PortfolioState {
    fn new(capital: f64) -> Self {
        Self {
            cash: capital,
            positions: HashMap::new(),
        }
    }

    fn total_committed(&self, prices: &HashMap<String, f64>) -> f64 {
        self.positions.iter().map(|(sym, pos)| {
            let price = prices.get(sym).copied().unwrap_or(pos.avg_cost);
            pos.shares * price
        }).sum::<f64>()
    }

    fn total_value(&self, prices: &HashMap<String, f64>) -> f64 {
        self.cash + self.total_committed(prices)
    }

    fn position_value(&self, symbol: &str, prices: &HashMap<String, f64>) -> f64 {
        if let Some(pos) = self.positions.get(symbol) {
            let price = prices.get(symbol).copied().unwrap_or(pos.avg_cost);
            pos.shares * price
        } else {
            0.0
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   ALPHA SIGNAL — PORTFOLIO REBUILD                             ║");
    println!("║   From: PostgreSQL alpha_signal (clean signals)                ║");
    println!("║   Capital: ${:.0}  |  Cash buffer: {}%                  ║", STARTING_CAPITAL, (CASH_BUFFER_PCT * 100.0) as i32);
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Connect to PostgreSQL
    let pg_pool = pg::create_pool()?;
    {
        let _conn = pg_pool.get().await?;
        println!("  Connected to PostgreSQL (alpha_signal)");
    }

    // Connect to SQLite for price history
    let sqlite_db = db::Database::new("rust_invest.db")?;
    println!("  Connected to SQLite (rust_invest.db)\n");

    // Get all signal dates
    let signal_dates = pg::get_signal_dates(&pg_pool).await?;
    println!("  Signal dates: {} days ({} to {})",
        signal_dates.len(),
        signal_dates.first().unwrap_or(&"?".to_string()),
        signal_dates.last().unwrap_or(&"?".to_string()));

    // Load all stock price histories from SQLite
    // Build a map: symbol → [(date, price)]
    println!("  Loading price histories...");
    let mut price_histories: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    let all_assets = get_all_signal_assets(&pg_pool).await?;
    for symbol in &all_assets {
        if let Ok(points) = sqlite_db.get_stock_history(symbol) {
            let daily: Vec<(String, f64)> = points.iter()
                .filter(|p| p.timestamp.len() == 10) // only bare dates (daily close)
                .map(|p| (p.timestamp.clone(), p.price))
                .collect();
            if !daily.is_empty() {
                price_histories.insert(symbol.clone(), daily);
            }
        }
    }
    println!("  Loaded price histories for {} assets\n", price_histories.len());

    // Build date→price lookup for fast access
    let mut date_prices: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (symbol, history) in &price_histories {
        for (date, price) in history {
            date_prices.entry(date.clone())
                .or_default()
                .insert(symbol.clone(), *price);
        }
    }

    // Pre-compute trailing volatilities per asset per date
    let volatilities = compute_volatilities(&price_histories, VOL_LOOKBACK);

    // Initialize portfolio
    let mut portfolio = PortfolioState::new(STARTING_CAPITAL);
    let mut equity_curve: Vec<(String, f64)> = Vec::new();
    let mut total_trades = 0;
    let mut total_rejections = 0;
    let mut total_tx_costs = 0.0;
    let max_investable = STARTING_CAPITAL * (1.0 - CASH_BUFFER_PCT);

    println!("━━━ SIMULATING DAY BY DAY ━━━\n");

    for date in &signal_dates {
        // Get today's prices
        let today_prices = match date_prices.get(date) {
            Some(p) => p.clone(),
            None => {
                // Try to use previous day's prices
                let prev = find_previous_price_date(&date_prices, date);
                match prev {
                    Some(p) => p,
                    None => {
                        eprintln!("  [{}] No prices available, skipping", date);
                        continue;
                    }
                }
            }
        };

        // Get today's signals
        let signals = pg::get_signals_for_date(&pg_pool, date).await?;
        if signals.is_empty() { continue; }

        let _portfolio_value = portfolio.total_value(&today_prices);

        // Step 1: Process SELL/SHORT signals — exit positions
        for sig in &signals {
            if sig.signal_type != "SELL" && sig.signal_type != "SHORT" { continue; }
            if let Some(pos) = portfolio.positions.remove(&sig.asset) {
                let sell_price = today_prices.get(&sig.asset).copied().unwrap_or(pos.avg_cost);
                let proceeds = pos.shares * sell_price;
                let tx_cost = proceeds * tx_cost_bps(&sig.asset) / 10_000.0;
                portfolio.cash += proceeds - tx_cost;
                total_tx_costs += tx_cost;
                total_trades += 1;
            }
        }

        // Step 2: Determine BUY candidates and their target weights
        let buy_signals: Vec<&pg::SignalRow> = signals.iter()
            .filter(|s| s.signal_type == "BUY" && today_prices.contains_key(&s.asset))
            .collect();

        if !buy_signals.is_empty() {
            // Compute inverse-volatility weights for BUY candidates
            let mut raw_weights: Vec<(String, f64)> = Vec::new();
            for sig in &buy_signals {
                let vol = volatilities.get(&sig.asset)
                    .and_then(|v| v.get(date))
                    .copied()
                    .unwrap_or(0.02); // default 2% daily vol
                let inv_vol = if vol > 0.001 { 1.0 / vol } else { 1.0 / 0.02 };
                raw_weights.push((sig.asset.clone(), inv_vol));
            }

            // Normalize weights
            let weight_sum: f64 = raw_weights.iter().map(|(_, w)| w).sum();
            let mut target_weights: Vec<(String, f64)> = raw_weights.iter()
                .map(|(s, w)| (s.clone(), w / weight_sum))
                .collect();

            // Apply 15% per-position cap and renormalize
            for (_, w) in &mut target_weights {
                if *w > MAX_POSITION_PCT {
                    *w = MAX_POSITION_PCT;
                }
            }
            let capped_sum: f64 = target_weights.iter().map(|(_, w)| w).sum();
            if capped_sum > 0.0 {
                for (_, w) in &mut target_weights {
                    *w /= capped_sum;
                    if *w > MAX_POSITION_PCT {
                        *w = MAX_POSITION_PCT;
                    }
                }
            }

            // Step 3: Allocate capital with hard assertion
            let updated_portfolio_value = portfolio.total_value(&today_prices);
            let investable = updated_portfolio_value * (1.0 - CASH_BUFFER_PCT);

            for (symbol, target_weight) in &target_weights {
                let target_value = investable * target_weight;
                let current_value = portfolio.position_value(symbol, &today_prices);
                let diff = target_value - current_value;

                // Rebalance threshold: skip if weight drift is small
                if current_value > 0.0 {
                    let current_weight = current_value / updated_portfolio_value;
                    if (current_weight - target_weight).abs() < REBALANCE_THRESHOLD {
                        continue;
                    }
                }

                if diff > 50.0 {
                    // BUY more
                    let price = match today_prices.get(symbol) {
                        Some(&p) if p > 0.0 => p,
                        _ => continue,
                    };

                    // HARD ASSERTION: check total committed won't exceed limit
                    let current_committed = portfolio.total_committed(&today_prices);
                    if current_committed + diff > max_investable {
                        // Reject this trade
                        let rejection = pg::SizingRejection {
                            asset: symbol.clone(),
                            signal_type: "BUY".to_string(),
                            requested_amount: diff,
                            available_capital: portfolio.cash,
                            total_committed: current_committed,
                            max_allowed: max_investable,
                            reason: format!(
                                "Would exceed capital limit: committed {:.0} + request {:.0} > max {:.0}",
                                current_committed, diff, max_investable
                            ),
                        };
                        let _ = pg::log_sizing_rejection(&pg_pool, &rejection).await;
                        total_rejections += 1;
                        continue;
                    }

                    // Also check we have enough cash
                    let cost_bps = tx_cost_bps(symbol);
                    let tx_cost = diff * cost_bps / 10_000.0;
                    let total_cost = diff + tx_cost;
                    if total_cost > portfolio.cash {
                        // Reduce to what we can afford
                        let affordable = (portfolio.cash - tx_cost).max(0.0);
                        if affordable < 50.0 { continue; }
                        let shares = affordable / price;
                        let actual_cost = shares * price;
                        let actual_tx = actual_cost * cost_bps / 10_000.0;

                        portfolio.cash -= actual_cost + actual_tx;
                        total_tx_costs += actual_tx;

                        let pos = portfolio.positions.entry(symbol.clone()).or_insert(Position {
                            shares: 0.0,
                            avg_cost: price,
                        });
                        let old_value = pos.shares * pos.avg_cost;
                        pos.shares += shares;
                        pos.avg_cost = (old_value + actual_cost) / pos.shares;
                    } else {
                        let shares = diff / price;
                        portfolio.cash -= total_cost;
                        total_tx_costs += tx_cost;

                        let pos = portfolio.positions.entry(symbol.clone()).or_insert(Position {
                            shares: 0.0,
                            avg_cost: price,
                        });
                        let old_value = pos.shares * pos.avg_cost;
                        pos.shares += shares;
                        pos.avg_cost = (old_value + diff) / pos.shares;
                    }
                    total_trades += 1;
                } else if diff < -50.0 {
                    // Trim position
                    let price = match today_prices.get(symbol) {
                        Some(&p) if p > 0.0 => p,
                        _ => continue,
                    };
                    let shares_to_sell = (-diff) / price;
                    if let Some(pos) = portfolio.positions.get_mut(symbol) {
                        let actual_shares = shares_to_sell.min(pos.shares);
                        let proceeds = actual_shares * price;
                        let tx_cost = proceeds * tx_cost_bps(symbol) / 10_000.0;
                        pos.shares -= actual_shares;
                        portfolio.cash += proceeds - tx_cost;
                        total_tx_costs += tx_cost;
                        total_trades += 1;

                        if pos.shares < 0.01 {
                            portfolio.positions.remove(symbol);
                        }
                    }
                }
            }
        }

        // Record daily portfolio value
        let final_value = portfolio.total_value(&today_prices);
        equity_curve.push((date.clone(), final_value));

        if equity_curve.len() <= 5 || equity_curve.len() % 5 == 0 {
            let pct_return = (final_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0;
            println!("  [{}] Value: ${:.0}  ({:+.2}%)  Positions: {}  Cash: ${:.0}",
                date, final_value, pct_return, portfolio.positions.len(), portfolio.cash);
        }
    }

    // Write daily portfolio to Postgres
    println!("\n━━━ WRITING PORTFOLIO TO POSTGRES ━━━\n");
    let mut prev_value = STARTING_CAPITAL;
    for (i, (date, value)) in equity_curve.iter().enumerate() {
        let daily_return = if i == 0 { 0.0 } else { (value - prev_value) / prev_value * 100.0 };
        let cum_return = (value - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0;

        pg::upsert_daily_portfolio(
            &pg_pool,
            date,
            STARTING_CAPITAL,
            *value,
            daily_return,
            cum_return,
            None,
            7, // model_version
        ).await?;

        prev_value = *value;
    }
    println!("  Written {} daily portfolio entries", equity_curve.len());

    // Summary
    let final_value = equity_curve.last().map(|(_, v)| *v).unwrap_or(STARTING_CAPITAL);
    let total_return = (final_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0;

    // Compute Sharpe
    let daily_returns: Vec<f64> = equity_curve.windows(2)
        .map(|w| (w[1].1 - w[0].1) / w[0].1)
        .collect();
    let mean_daily = daily_returns.iter().sum::<f64>() / daily_returns.len().max(1) as f64;
    let std_daily = if daily_returns.len() > 1 {
        let var = daily_returns.iter()
            .map(|r| (r - mean_daily).powi(2))
            .sum::<f64>() / (daily_returns.len() - 1) as f64;
        var.sqrt()
    } else { 0.0 };
    let rf_daily = 0.045 / 252.0;
    let sharpe = if std_daily > 0.0 { (mean_daily - rf_daily) / std_daily * 252.0_f64.sqrt() } else { 0.0 };

    // Max drawdown
    let mut peak = STARTING_CAPITAL;
    let mut max_dd = 0.0_f64;
    for (_, v) in &equity_curve {
        if *v > peak { peak = *v; }
        let dd = (peak - v) / peak * 100.0;
        if dd > max_dd { max_dd = dd; }
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║   PORTFOLIO REBUILD COMPLETE                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Starting Capital:  ${:>10.0}                                ║", STARTING_CAPITAL);
    println!("║  Final Value:       ${:>10.0}                                ║", final_value);
    println!("║  Total Return:      {:>9.2}%                                ║", total_return);
    println!("║  Sharpe Ratio:      {:>9.2}                                 ║", sharpe);
    println!("║  Max Drawdown:      {:>9.2}%                                ║", max_dd);
    println!("║  Trading Days:      {:>9}                                   ║", equity_curve.len());
    println!("║  Total Trades:      {:>9}                                   ║", total_trades);
    println!("║  Total Tx Costs:    ${:>9.2}                                ║", total_tx_costs);
    println!("║  Sizing Rejections: {:>9}                                   ║", total_rejections);
    println!("║  Final Positions:   {:>9}                                   ║", portfolio.positions.len());
    println!("║  Final Cash:        ${:>9.0}                                ║", portfolio.cash);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Print top holdings
    if !portfolio.positions.is_empty() {
        let final_prices = date_prices.get(signal_dates.last().unwrap())
            .cloned().unwrap_or_default();
        let mut holdings: Vec<(String, f64)> = portfolio.positions.iter()
            .map(|(sym, pos)| {
                let price = final_prices.get(sym).copied().unwrap_or(pos.avg_cost);
                (sym.clone(), pos.shares * price)
            })
            .collect();
        holdings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n  Top 20 Holdings:");
        for (i, (sym, val)) in holdings.iter().take(20).enumerate() {
            let weight = val / final_value * 100.0;
            println!("  {:>3}. {:>8}: ${:>8.0}  ({:.1}%)", i + 1, sym, val, weight);
        }
        println!("  ... {} total positions", holdings.len());

        // Position size stats
        let values: Vec<f64> = holdings.iter().map(|(_, v)| *v).collect();
        let min_pos = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_pos = values.iter().cloned().fold(0.0_f64, f64::max);
        let mean_pos = values.iter().sum::<f64>() / values.len() as f64;
        let median_pos = {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        println!("\n  Position Size Stats:");
        println!("    Min:    ${:.0}", min_pos);
        println!("    Max:    ${:.0}", max_pos);
        println!("    Mean:   ${:.0}", mean_pos);
        println!("    Median: ${:.0}", median_pos);
    }

    Ok(())
}

/// Get all distinct assets that have signals in the database.
async fn get_all_signal_assets(pool: &pg::PgPool) -> Result<Vec<String>, pg::PgError> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT DISTINCT asset FROM signals WHERE asset_class = 'stock' ORDER BY asset",
        &[],
    ).await?;
    Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
}

/// Compute trailing volatility (annualized std of daily log returns) for each asset and date.
fn compute_volatilities(
    price_histories: &HashMap<String, Vec<(String, f64)>>,
    lookback: usize,
) -> HashMap<String, HashMap<String, f64>> {
    let mut result: HashMap<String, HashMap<String, f64>> = HashMap::new();

    for (symbol, history) in price_histories {
        let mut vol_map: HashMap<String, f64> = HashMap::new();

        for i in lookback..history.len() {
            let window = &history[i - lookback..i];
            let log_returns: Vec<f64> = window.windows(2)
                .filter_map(|w| {
                    if w[0].1 > 0.0 && w[1].1 > 0.0 {
                        Some((w[1].1 / w[0].1).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if log_returns.len() >= 20 {
                let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
                let var = log_returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (log_returns.len() - 1) as f64;
                let daily_vol = var.sqrt();
                vol_map.insert(history[i].0.clone(), daily_vol);
            }
        }

        result.insert(symbol.clone(), vol_map);
    }

    result
}

/// Find prices from the most recent available date before the given date.
fn find_previous_price_date(
    date_prices: &HashMap<String, HashMap<String, f64>>,
    target: &str,
) -> Option<HashMap<String, f64>> {
    // Look back up to 5 days
    if let Ok(target_date) = chrono::NaiveDate::parse_from_str(target, "%Y-%m-%d") {
        for days_back in 1..=5 {
            let prev = target_date - chrono::Duration::days(days_back);
            let prev_str = prev.format("%Y-%m-%d").to_string();
            if let Some(prices) = date_prices.get(&prev_str) {
                return Some(prices.clone());
            }
        }
    }
    None
}
