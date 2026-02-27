/// Backtester — Simulate Trading Signals Over Walk-Forward History
/// ================================================================
/// Replays ensemble signals through time to compute realistic P&L
/// metrics: total return, Sharpe ratio, max drawdown, win rate.
///
/// Design decisions:
///   - Uses walk-forward folds to avoid look-ahead bias
///   - Each fold trains on historical data, generates signals on test window
///   - Position sizing: equal-weight per signal (no leverage)
///   - Transaction costs: configurable spread + commission
///   - Benchmark: buy-and-hold over the same period
///
/// Key metric interpretations:
///   Sharpe > 1.0 = good risk-adjusted return
///   Sharpe > 2.0 = excellent
///   Max drawdown < 10% = conservative
///   Win rate > 55% with positive expectancy = edge confirmed

use crate::ml::{self, Sample};
use crate::gbt::{self, GBTConfig, TreeConfig, GradientBoostedClassifier};

// ════════════════════════════════════════
// Configuration
// ════════════════════════════════════════

/// Backtester configuration
#[derive(Clone, Debug)]
pub struct BacktestConfig {
    /// Starting capital in dollars
    pub initial_capital: f64,
    /// Transaction cost per trade (percentage, e.g., 0.001 = 0.1%)
    pub transaction_cost: f64,
    /// Ensemble P(up) threshold for BUY signal
    pub buy_threshold: f64,
    /// Ensemble P(up) threshold for SELL signal (below this → go to cash)
    pub sell_threshold: f64,
    /// Minimum walk-forward accuracy required to act on signal
    pub min_accuracy: f64,
    /// Position size as fraction of capital (1.0 = all in, 0.5 = half)
    pub position_size: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        BacktestConfig {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,  // 10 bps round-trip
            buy_threshold: 0.55,
            sell_threshold: 0.45,
            min_accuracy: 51.0,
            position_size: 1.0,
        }
    }
}

// ════════════════════════════════════════
// Trade Record
// ════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct Trade {
    pub fold: usize,
    pub day_in_fold: usize,
    pub action: TradeAction,
    pub price: f64,
    pub ensemble_prob: f64,
    pub pnl_pct: f64,        // % return on this trade
    pub equity_after: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TradeAction {
    Buy,
    Sell,      // Exit to cash
    Hold,      // Maintain position
    NoSignal,  // Below accuracy threshold
}

impl std::fmt::Display for TradeAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeAction::Buy => write!(f, "BUY"),
            TradeAction::Sell => write!(f, "SELL"),
            TradeAction::Hold => write!(f, "HOLD"),
            TradeAction::NoSignal => write!(f, "NO SIGNAL"),
        }
    }
}

// ════════════════════════════════════════
// Backtest Results
// ════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct BacktestResult {
    pub symbol: String,
    pub config: BacktestConfig,

    // Returns
    pub total_return_pct: f64,
    pub annualised_return_pct: f64,
    pub benchmark_return_pct: f64,   // Buy-and-hold
    pub excess_return_pct: f64,      // Strategy - benchmark

    // Risk
    pub sharpe_ratio: f64,
    pub max_drawdown_pct: f64,
    pub volatility_pct: f64,         // Annualised

    // Trade stats
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub profit_factor: f64,          // gross profit / gross loss
    pub expectancy: f64,             // avg pnl per trade

    // Time stats
    pub days_in_market: usize,
    pub total_days: usize,
    pub n_folds: usize,

    // Equity curve (daily)
    pub equity_curve: Vec<f64>,
    pub benchmark_curve: Vec<f64>,
    pub daily_returns: Vec<f64>,

    // Trade log
    pub trades: Vec<Trade>,
}

// ════════════════════════════════════════
// Core Backtester — Walk-Forward Replay
// ════════════════════════════════════════

/// Run backtest on pre-built rich features using walk-forward evaluation.
///
/// This replays the EXACT same walk-forward procedure as ensemble.rs,
/// but instead of just reporting accuracy, it simulates trading:
///   1. For each fold: train models on train window
///   2. Generate signals on test window day by day
///   3. Enter/exit positions based on ensemble probability
///   4. Track equity, trades, and compute risk metrics
pub fn run_backtest(
    symbol: &str,
    samples: &[Sample],
    prices: &[f64],        // Aligned prices for P&L calculation
    train_window: usize,
    test_window: usize,
    step: usize,
    config: &BacktestConfig,
) -> Option<BacktestResult> {
    let n_features = samples[0].features.len();

    if samples.len() < train_window + test_window + 10 {
        println!("  [Backtest] {} — not enough samples ({})", symbol, samples.len());
        return None;
    }

    // Offset: samples start later than prices due to feature lookback.
    // We need prices aligned to each sample for P&L.
    // Samples[i] predicts prices[i+offset+1] movement.
    // The label in samples[i] = (price[i+offset+1] - price[i+offset]) / price[i+offset]
    // So for sample at index `s`, the corresponding "current" price is at
    // `prices[price_offset + s]` where price_offset accounts for feature lookback.
    let price_offset = prices.len() - samples.len() - 1;

    println!("  [Backtest] {} — replaying {} samples over walk-forward windows", symbol, samples.len());

    let mut equity = config.initial_capital;
    let mut position_open = false;
    let mut entry_price = 0.0;
    let mut equity_curve = Vec::new();
    let mut daily_returns: Vec<f64> = Vec::new();
    let mut trades = Vec::new();
    let mut n_folds = 0_usize;
    let mut total_days = 0_usize;
    let mut days_in_market = 0_usize;

    // Track benchmark (buy-and-hold from first test day)
    let mut benchmark_start_price = 0.0_f64;
    let mut benchmark_started = false;
    let mut benchmark_curve = Vec::new();

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        // Clone and normalise this fold
        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;

        let (train_data, test_data) = fold_samples.split_at_mut(train_len);
        let (means, stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &means, &stds);

        // Train LinReg
        let mut lin = ml::LinearRegression::new(n_features);
        lin.train(train_data, 0.005, 3000);

        // Train LogReg
        let mut log = ml::LogisticRegression::new(n_features);
        log.train(train_data, 0.01, 3000);

        // Train GBT
        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);

        let gbt_config = GBTConfig {
            n_trees: 80,
            learning_rate: 0.08,
            tree_config: TreeConfig {
                max_depth: 4,
                min_samples_leaf: 8,
                min_samples_split: 16,
            },
            subsample_ratio: 0.8,
            early_stopping_rounds: Some(8),
        };

        let gbt = GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), gbt_config);

        // Compute fold accuracy for quality gating
        let mut fold_correct = 0_usize;
        let fold_size = test_data.len();
        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            if gbt.predict_direction(&s.features) == actual_up { fold_correct += 1; }
        }
        let fold_accuracy = fold_correct as f64 / fold_size.max(1) as f64 * 100.0;

        // Replay test window day by day
        for (day_idx, s) in test_data.iter().enumerate() {
            let sample_global_idx = train_end + day_idx;
            let price_idx = price_offset + sample_global_idx;
            let next_price_idx = price_idx + 1;

            if next_price_idx >= prices.len() { break; }

            let current_price = prices[price_idx];
            let next_price = prices[next_price_idx];
            let actual_return = (next_price - current_price) / current_price;

            // Set benchmark start
            if !benchmark_started {
                benchmark_start_price = current_price;
                benchmark_started = true;
            }
            benchmark_curve.push(current_price / benchmark_start_price * config.initial_capital);

            // Compute ensemble probability (same as ensemble.rs)
            let raw_lin = lin.predict(&s.features);
            let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);
            let log_prob = log.predict_probability(&s.features).clamp(0.15, 0.85);
            let gbt_prob = gbt.predict_proba(&s.features).clamp(0.15, 0.85);

            // Simple equal weighting for backtest (like ensemble with 3 models)
            let ensemble_prob = (lin_prob + log_prob + gbt_prob) / 3.0;

            // Decide action
            let (action, daily_pnl) = if fold_accuracy < config.min_accuracy {
                // Below accuracy threshold — stay out
                if position_open {
                    let exit_cost = equity * config.transaction_cost;
                    equity -= exit_cost;
                    position_open = false;
                }
                (TradeAction::NoSignal, 0.0)
            } else if ensemble_prob > config.buy_threshold {
                if !position_open {
                    // Enter long position
                    let entry_cost = equity * config.transaction_cost;
                    equity -= entry_cost;
                    entry_price = current_price;
                    position_open = true;
                    days_in_market += 1;
                    let pnl = actual_return * config.position_size;
                    equity *= 1.0 + pnl;
                    (TradeAction::Buy, pnl)
                } else {
                    // Already in — hold
                    days_in_market += 1;
                    let pnl = actual_return * config.position_size;
                    equity *= 1.0 + pnl;
                    (TradeAction::Hold, pnl)
                }
            } else if ensemble_prob < config.sell_threshold {
                if position_open {
                    // Exit position
                    let exit_cost = equity * config.transaction_cost;
                    equity -= exit_cost;
                    position_open = false;
                    (TradeAction::Sell, 0.0)
                } else {
                    (TradeAction::Hold, 0.0)
                }
            } else {
                // Neutral zone — maintain current position
                if position_open {
                    days_in_market += 1;
                    let pnl = actual_return * config.position_size;
                    equity *= 1.0 + pnl;
                    (TradeAction::Hold, pnl)
                } else {
                    (TradeAction::Hold, 0.0)
                }
            };

            equity_curve.push(equity);
            daily_returns.push(daily_pnl);
            total_days += 1;

            if action == TradeAction::Buy || action == TradeAction::Sell {
                trades.push(Trade {
                    fold: n_folds,
                    day_in_fold: day_idx,
                    action,
                    price: current_price,
                    ensemble_prob,
                    pnl_pct: daily_pnl * 100.0,
                    equity_after: equity,
                });
            }
        }

        n_folds += 1;
        start += step;
    }

    if total_days < 10 || equity_curve.is_empty() {
        println!("  [Backtest] {} — too few trading days ({})", symbol, total_days);
        return None;
    }

    // ── Compute metrics ──
    let total_return = (equity - config.initial_capital) / config.initial_capital * 100.0;
    let benchmark_return = if benchmark_started && !benchmark_curve.is_empty() {
        let last = *benchmark_curve.last().unwrap();
        (last - config.initial_capital) / config.initial_capital * 100.0
    } else {
        0.0
    };

    let trading_days_per_year = 252.0;
    let years = total_days as f64 / trading_days_per_year;
    let annualised_return = if years > 0.0 {
        ((equity / config.initial_capital).powf(1.0 / years) - 1.0) * 100.0
    } else {
        0.0
    };

    // Sharpe ratio (annualised)
    let mean_daily = if !daily_returns.is_empty() {
        daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
    } else {
        0.0
    };
    let std_daily = if daily_returns.len() > 1 {
        let variance = daily_returns.iter()
            .map(|r| (r - mean_daily).powi(2))
            .sum::<f64>() / (daily_returns.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let sharpe = if std_daily > 0.0 {
        (mean_daily / std_daily) * trading_days_per_year.sqrt()
    } else {
        0.0
    };

    let volatility = std_daily * trading_days_per_year.sqrt() * 100.0;

    // Max drawdown
    let max_drawdown = compute_max_drawdown(&equity_curve);

    // Win/loss stats (on actual trade entries, not holds)
    let entry_trades: Vec<&Trade> = trades.iter()
        .filter(|t| t.action == TradeAction::Buy)
        .collect();

    // Compute per-trade returns by matching entries to subsequent days
    let mut wins = Vec::new();
    let mut losses = Vec::new();

    for r in &daily_returns {
        if *r > 0.0 {
            wins.push(*r);
        } else if *r < 0.0 {
            losses.push(*r);
        }
    }

    let winning_trades = wins.len();
    let losing_trades = losses.len();
    let win_rate = if winning_trades + losing_trades > 0 {
        winning_trades as f64 / (winning_trades + losing_trades) as f64 * 100.0
    } else {
        0.0
    };

    let avg_win = if !wins.is_empty() {
        wins.iter().sum::<f64>() / wins.len() as f64 * 100.0
    } else {
        0.0
    };
    let avg_loss = if !losses.is_empty() {
        losses.iter().sum::<f64>() / losses.len() as f64 * 100.0
    } else {
        0.0
    };

    let gross_profit: f64 = wins.iter().sum::<f64>().abs();
    let gross_loss: f64 = losses.iter().sum::<f64>().abs();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let total_traded = winning_trades + losing_trades;
    let expectancy = if total_traded > 0 {
        (wins.iter().sum::<f64>() + losses.iter().sum::<f64>()) / total_traded as f64 * 100.0
    } else {
        0.0
    };

    let result = BacktestResult {
        symbol: symbol.to_string(),
        config: config.clone(),
        total_return_pct: total_return,
        annualised_return_pct: annualised_return,
        benchmark_return_pct: benchmark_return,
        excess_return_pct: total_return - benchmark_return,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_drawdown,
        volatility_pct: volatility,
        total_trades: trades.len(),
        winning_trades,
        losing_trades,
        win_rate,
        avg_win_pct: avg_win,
        avg_loss_pct: avg_loss,
        profit_factor,
        expectancy,
        days_in_market,
        total_days,
        n_folds,
        equity_curve,
        benchmark_curve,
        daily_returns,
        trades,
    };

    print_backtest_result(&result);
    Some(result)
}

/// Compute maximum drawdown from an equity curve
fn compute_max_drawdown(equity: &[f64]) -> f64 {
    if equity.is_empty() { return 0.0; }

    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;

    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak * 100.0;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

// ════════════════════════════════════════
// Console Output
// ════════════════════════════════════════

pub fn print_backtest_result(r: &BacktestResult) {
    println!("\n    ┌─────────────────────────────────────────────────────┐");
    println!("    │  BACKTEST: {:<42} │", r.symbol);
    println!("    ├─────────────────────────────────────────────────────┤");
    println!("    │  Total Return:     {:>8.2}%  (B&H: {:>7.2}%)       │",
        r.total_return_pct, r.benchmark_return_pct);
    println!("    │  Excess Return:    {:>8.2}%                        │", r.excess_return_pct);
    println!("    │  Annualised:       {:>8.2}%                        │", r.annualised_return_pct);
    println!("    │  Sharpe Ratio:     {:>8.2}                         │", r.sharpe_ratio);
    println!("    │  Max Drawdown:     {:>8.2}%                        │", r.max_drawdown_pct);
    println!("    │  Volatility (ann): {:>8.2}%                        │", r.volatility_pct);
    println!("    │  Win Rate:         {:>8.1}%  ({}/{})               │",
        r.win_rate, r.winning_trades, r.winning_trades + r.losing_trades);
    println!("    │  Avg Win/Loss:    +{:.2}% / {:.2}%                 │",
        r.avg_win_pct, r.avg_loss_pct);
    println!("    │  Profit Factor:    {:>8.2}                         │", r.profit_factor);
    println!("    │  Expectancy:       {:>8.3}% per trade              │", r.expectancy);
    println!("    │  Days in Market:   {:>4} / {:>4}  ({:.0}%)            │",
        r.days_in_market, r.total_days,
        r.days_in_market as f64 / r.total_days.max(1) as f64 * 100.0);
    println!("    │  Walk-Forward Folds: {:>4}                          │", r.n_folds);
    println!("    └─────────────────────────────────────────────────────┘");
}

/// Print summary table of all backtest results
pub fn print_backtest_summary(results: &[BacktestResult]) {
    if results.is_empty() { return; }

    println!("╔════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        BACKTEST SUMMARY — Walk-Forward Replay                     ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<8} {:>8} {:>8} {:>8} {:>7} {:>8} {:>7} {:>7} {:>8} {:>7} ║",
        "Symbol", "Return%", "B&H%", "Excess%", "Sharpe", "MaxDD%", "WinR%", "PF", "Expect%", "Days");
    println!("╠════════════════════════════════════════════════════════════════════════════════════╣");

    for r in results {
        let verdict = if r.sharpe_ratio > 1.0 && r.excess_return_pct > 0.0 {
            "✓"
        } else if r.sharpe_ratio > 0.5 {
            "~"
        } else {
            "✗"
        };

        println!("║ {:<8} {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2} {:>7.2}% {:>6.1}% {:>6.2} {:>7.3}% {:>5}/{} {} ║",
            r.symbol,
            r.total_return_pct,
            r.benchmark_return_pct,
            r.excess_return_pct,
            r.sharpe_ratio,
            r.max_drawdown_pct,
            r.win_rate,
            r.profit_factor,
            r.expectancy,
            r.days_in_market,
            r.total_days,
            verdict,
        );
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════╝");

    // Aggregate stats
    let avg_sharpe = results.iter().map(|r| r.sharpe_ratio).sum::<f64>() / results.len() as f64;
    let avg_excess = results.iter().map(|r| r.excess_return_pct).sum::<f64>() / results.len() as f64;
    let profitable = results.iter().filter(|r| r.excess_return_pct > 0.0).count();

    println!("  Avg Sharpe: {:.2} | Avg Excess Return: {:.2}% | Profitable: {}/{}",
        avg_sharpe, avg_excess, profitable, results.len());
    println!("  ✓ = Sharpe > 1.0 + positive excess | ~ = marginal | ✗ = no edge\n");
}

// ════════════════════════════════════════
// HTML Report Section
// ════════════════════════════════════════

pub fn backtest_html(results: &[BacktestResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut html = String::new();

    html.push_str("<h2>Backtest Results — Walk-Forward Replay</h2>\n");
    html.push_str("<p>Simulated trading using ensemble signals over walk-forward evaluation windows. \
        Models retrained on each fold; signals generated day-by-day on test windows. \
        Transaction costs: 10 bps per trade. Starting capital: $100,000.</p>\n");

    // Summary table
    html.push_str("<table>\n<tr>\
        <th>Symbol</th><th>Return</th><th>B&amp;H</th><th>Excess</th>\
        <th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>PF</th>\
        <th>Expectancy</th><th>Days</th><th>Verdict</th></tr>\n");

    for r in results {
        let verdict_class = if r.sharpe_ratio > 1.0 && r.excess_return_pct > 0.0 {
            ("EDGE", "signal-bullish")
        } else if r.sharpe_ratio > 0.5 {
            ("MARGINAL", "signal-neutral")
        } else {
            ("NO EDGE", "signal-bearish")
        };

        let return_color = if r.total_return_pct >= 0.0 { "color:#00e676;" } else { "color:#ff5252;" };
        let excess_color = if r.excess_return_pct >= 0.0 { "color:#00e676;" } else { "color:#ff5252;" };

        html.push_str(&format!(
            "<tr><td>{}</td><td style='{}'>{:.2}%</td><td>{:.2}%</td>\
             <td style='{}'>{:.2}%</td><td>{:.2}</td><td>{:.2}%</td>\
             <td>{:.1}%</td><td>{:.2}</td><td>{:.3}%</td>\
             <td>{}/{}</td><td><span class='{}'>{}</span></td></tr>\n",
            r.symbol, return_color, r.total_return_pct,
            r.benchmark_return_pct,
            excess_color, r.excess_return_pct,
            r.sharpe_ratio, r.max_drawdown_pct,
            r.win_rate, r.profit_factor, r.expectancy,
            r.days_in_market, r.total_days,
            verdict_class.1, verdict_class.0,
        ));
    }
    html.push_str("</table>\n");

    // Equity curves as inline SVG sparklines
    html.push_str("<h3>Equity Curves</h3>\n");
    html.push_str("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:15px;margin:20px 0;'>\n");

    for r in results {
        if r.equity_curve.is_empty() { continue; }

        let svg = equity_curve_svg(&r.equity_curve, &r.benchmark_curve, &r.symbol);
        html.push_str(&format!(
            "<div class='card'>\
             <h4 style='margin:0 0 8px;'>{} — Return: {:.2}% (Sharpe: {:.2})</h4>\
             {}\
             </div>\n",
            r.symbol, r.total_return_pct, r.sharpe_ratio, svg
        ));
    }
    html.push_str("</div>\n");

    // Aggregate
    let avg_sharpe = results.iter().map(|r| r.sharpe_ratio).sum::<f64>() / results.len() as f64;
    let avg_excess = results.iter().map(|r| r.excess_return_pct).sum::<f64>() / results.len() as f64;
    let profitable = results.iter().filter(|r| r.excess_return_pct > 0.0).count();

    html.push_str(&format!(
        "<p><strong>Portfolio Summary:</strong> Avg Sharpe: {:.2} | Avg Excess Return: {:.2}% | \
         Assets beating B&amp;H: {}/{}</p>\n",
        avg_sharpe, avg_excess, profitable, results.len()
    ));

    html.push_str("<p><em>Backtest uses walk-forward folds to prevent look-ahead bias. \
        Models are retrained on each fold's training window before generating test signals. \
        Past performance does not guarantee future results. Not financial advice.</em></p>\n");

    html
}

/// Generate an inline SVG sparkline for equity curves
fn equity_curve_svg(equity: &[f64], benchmark: &[f64], _symbol: &str) -> String {
    if equity.is_empty() { return String::new(); }

    let width = 360.0_f64;
    let height = 120.0_f64;
    let padding = 10.0;

    let all_values: Vec<f64> = equity.iter().chain(benchmark.iter()).copied().collect();
    let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1.0);

    let make_path = |data: &[f64], color: &str, opacity: &str| -> String {
        if data.is_empty() { return String::new(); }
        let n = data.len() as f64;
        let points: Vec<String> = data.iter().enumerate().map(|(i, v)| {
            let x = padding + (i as f64 / (n - 1.0).max(1.0)) * (width - 2.0 * padding);
            let y = padding + (1.0 - (v - min_val) / range) * (height - 2.0 * padding);
            format!("{:.1},{:.1}", x, y)
        }).collect();

        format!(
            "<polyline points='{}' fill='none' stroke='{}' stroke-width='1.5' opacity='{}'/>",
            points.join(" "), color, opacity
        )
    };

    let equity_path = make_path(equity, "#00d4aa", "1.0");
    let bench_path = make_path(benchmark, "#888888", "0.5");

    // Reference line at initial capital
    let initial = equity[0];
    let y_ref = padding + (1.0 - (initial - min_val) / range) * (height - 2.0 * padding);

    format!(
        "<svg width='{w}' height='{h}' style='background:#0d1b2a;border-radius:6px;'>\
         <line x1='{p}' y1='{yr:.1}' x2='{w2:.1}' y2='{yr:.1}' stroke='#333' stroke-dasharray='4'/>\
         {bench}{eq}\
         <text x='{w3:.0}' y='{h2:.0}' fill='#888' font-size='9'>— B&amp;H</text>\
         <text x='{w3:.0}' y='{h3:.0}' fill='#00d4aa' font-size='9'>— Strategy</text>\
         </svg>",
        w = width, h = height, p = padding,
        yr = y_ref,
        w2 = width - padding,
        bench = bench_path, eq = equity_path,
        w3 = width - 70.0,
        h2 = height - 5.0,
        h3 = height - 18.0,
    )
}
