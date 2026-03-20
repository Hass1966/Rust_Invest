/// Multi-Frequency Backtest Comparison Engine
/// ============================================
/// Proves whether following signals beats buy & hold across:
///   - Three modes: Academic (100% in/out), Realistic (25% position), Signal-only (directional)
///   - Four frequencies: Daily, Weekly, Monthly, Hourly
///   - Per asset class transaction costs: Stocks 10bps, FX 5bps, Crypto 25bps
///
/// Outputs: exports/backtest_comparison.csv, exports/backtest_report.md

use crate::ml::{self, Sample};
use crate::gbt::{GBTConfig, TreeConfig, GradientBoostedClassifier};
use chrono::Datelike;

// ════════════════════════════════════════
// Types
// ════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub enum BacktestMode {
    Academic,   // Full position in/out on every signal
    Realistic,  // 25% of position per signal
    SignalOnly, // Directional accuracy only (no P&L)
}

impl std::fmt::Display for BacktestMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BacktestMode::Academic => write!(f, "Academic"),
            BacktestMode::Realistic => write!(f, "Realistic"),
            BacktestMode::SignalOnly => write!(f, "Signal-Only"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Frequency {
    Daily,
    Weekly,
    Monthly,
    Hourly,
}

impl std::fmt::Display for Frequency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frequency::Daily => write!(f, "Daily"),
            Frequency::Weekly => write!(f, "Weekly"),
            Frequency::Monthly => write!(f, "Monthly"),
            Frequency::Hourly => write!(f, "Hourly"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ComparisonRow {
    pub asset: String,
    pub asset_class: String,
    pub mode: BacktestMode,
    pub frequency: Frequency,
    pub sharpe: f64,
    pub win_rate: f64,
    pub vs_bh_pct: f64,
    pub trades_per_year: f64,
    pub max_drawdown: f64,
    pub total_return_pct: f64,
    pub bh_return_pct: f64,
}

// ════════════════════════════════════════
// Transaction costs by asset class
// ════════════════════════════════════════

pub fn tx_cost(asset_class: &str) -> f64 {
    match asset_class {
        "stock" => 0.0010,  // 10 bps
        "fx" => 0.0005,     // 5 bps
        "crypto" => 0.0025, // 25 bps
        _ => 0.0010,
    }
}

// ════════════════════════════════════════
// Core: run backtest for a single asset at a given mode + frequency
// ════════════════════════════════════════

pub fn run_comparison(
    symbol: &str,
    asset_class: &str,
    samples: &[Sample],
    prices: &[f64],
    timestamps: &[String],
    train_window: usize,
    test_window: usize,
    step: usize,
    mode: BacktestMode,
    frequency: Frequency,
) -> Option<ComparisonRow> {
    let n_features = samples[0].features.len();

    if samples.len() < train_window + test_window + 10 {
        return None;
    }

    let price_offset = prices.len().saturating_sub(samples.len() + 1);
    let cost = tx_cost(asset_class);

    let mut equity = 100_000.0_f64;
    let initial = equity;
    let mut position_pct = 0.0_f64; // fraction of capital in market
    let mut daily_returns: Vec<f64> = Vec::new();
    let mut equity_curve: Vec<f64> = Vec::new();
    let mut total_trades = 0_usize;
    let mut correct_signals = 0_usize;
    let mut total_signals = 0_usize;
    let mut total_days = 0_usize;

    // Benchmark
    let mut bh_start_price = 0.0_f64;
    let mut bh_started = false;

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        // Clone and normalise
        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let (train_data, test_data) = fold_samples.split_at_mut(train_window);
        let (means, stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &means, &stds);

        // Train models
        let mut lin = ml::LinearRegression::new(n_features);
        lin.train(train_data, 0.005, 3000);

        let mut log = ml::LogisticRegression::new(n_features);
        log.train(train_data, 0.01, 3000);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);
        let gbt_config = GBTConfig {
            n_trees: 80, learning_rate: 0.08,
            tree_config: TreeConfig { max_depth: 4, min_samples_leaf: 8, min_samples_split: 16 },
            subsample_ratio: 0.8, early_stopping_rounds: Some(8),
        };
        let gbt = GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), gbt_config);

        // Replay test window
        for (day_idx, s) in test_data.iter().enumerate() {
            let sample_global_idx = train_end + day_idx;
            let price_idx = price_offset + sample_global_idx;
            let next_price_idx = price_idx + 1;
            if next_price_idx >= prices.len() { break; }

            // Frequency filter: skip days that aren't on the right frequency
            let ts_idx = price_idx.min(timestamps.len().saturating_sub(1));
            if !should_trade_on_day(&timestamps, ts_idx, &frequency) {
                // Still track equity if in position
                if position_pct > 0.0 {
                    let actual_return = (prices[next_price_idx] - prices[price_idx]) / prices[price_idx];
                    let pnl = actual_return * position_pct;
                    equity *= 1.0 + pnl;
                    daily_returns.push(pnl);
                } else {
                    daily_returns.push(0.0);
                }
                equity_curve.push(equity);
                total_days += 1;
                if !bh_started { bh_start_price = prices[price_idx]; bh_started = true; }
                continue;
            }

            let current_price = prices[price_idx];
            let next_price = prices[next_price_idx];
            let actual_return = (next_price - current_price) / current_price;
            let actual_up = actual_return > 0.0;

            if !bh_started { bh_start_price = current_price; bh_started = true; }

            // Ensemble probability
            let raw_lin = lin.predict(&s.features);
            let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);
            let log_prob = log.predict_probability(&s.features).clamp(0.15, 0.85);
            let gbt_prob = gbt.predict_proba(&s.features).clamp(0.15, 0.85);
            let ensemble_prob = (lin_prob + log_prob + gbt_prob) / 3.0;

            let predicted_up = ensemble_prob > 0.55;
            let predicted_down = ensemble_prob < 0.45;

            // Track signal accuracy
            if predicted_up || predicted_down {
                total_signals += 1;
                if (predicted_up && actual_up) || (predicted_down && !actual_up) {
                    correct_signals += 1;
                }
            }

            match mode {
                BacktestMode::SignalOnly => {
                    // No position tracking, just accuracy
                    daily_returns.push(0.0);
                }
                BacktestMode::Academic => {
                    // Full in/out on every signal
                    if predicted_up && position_pct < 1.0 {
                        // Buy: go full position
                        let new_position = 1.0;
                        let delta = new_position - position_pct;
                        equity -= equity * delta * cost;
                        position_pct = new_position;
                        total_trades += 1;
                    } else if predicted_down && position_pct > 0.0 {
                        // Sell: exit completely
                        equity -= equity * position_pct * cost;
                        position_pct = 0.0;
                        total_trades += 1;
                    }
                    let pnl = actual_return * position_pct;
                    equity *= 1.0 + pnl;
                    daily_returns.push(pnl);
                }
                BacktestMode::Realistic => {
                    // Trade 25% of position per signal
                    let trade_size: f64 = 0.25;
                    if predicted_up && position_pct < 1.0 {
                        let delta = trade_size.min(1.0 - position_pct);
                        equity -= equity * delta * cost;
                        position_pct += delta;
                        total_trades += 1;
                    } else if predicted_down && position_pct > 0.0 {
                        let delta = trade_size.min(position_pct);
                        equity -= equity * delta * cost;
                        position_pct -= delta;
                        total_trades += 1;
                    }
                    let pnl = actual_return * position_pct;
                    equity *= 1.0 + pnl;
                    daily_returns.push(pnl);
                }
            }

            equity_curve.push(equity);
            total_days += 1;
        }

        start += step;
    }

    if total_days < 10 { return None; }

    let total_return = (equity - initial) / initial * 100.0;
    let bh_return = if bh_started && !equity_curve.is_empty() {
        let last_price_idx = price_offset + samples.len().min(prices.len() - 1);
        let last_price = prices[last_price_idx.min(prices.len() - 1)];
        if bh_start_price > 0.0 {
            (last_price - bh_start_price) / bh_start_price * 100.0
        } else { 0.0 }
    } else { 0.0 };

    // Sharpe
    let mean_daily = if !daily_returns.is_empty() {
        daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
    } else { 0.0 };
    let std_daily = if daily_returns.len() > 1 {
        let var = daily_returns.iter().map(|r| (r - mean_daily).powi(2)).sum::<f64>() / (daily_returns.len() - 1) as f64;
        var.sqrt()
    } else { 0.0 };
    let sharpe = if std_daily > 0.0 {
        (mean_daily / std_daily) * 252.0_f64.sqrt()
    } else { 0.0 };

    // Win rate
    let win_rate = if total_signals > 0 {
        correct_signals as f64 / total_signals as f64 * 100.0
    } else { 0.0 };

    // Max drawdown
    let max_dd = compute_max_drawdown(&equity_curve);

    // Trades per year
    let years = total_days as f64 / 252.0;
    let trades_yr = if years > 0.0 { total_trades as f64 / years } else { 0.0 };

    Some(ComparisonRow {
        asset: symbol.to_string(),
        asset_class: asset_class.to_string(),
        mode,
        frequency,
        sharpe,
        win_rate,
        vs_bh_pct: total_return - bh_return,
        trades_per_year: trades_yr,
        max_drawdown: max_dd,
        total_return_pct: total_return,
        bh_return_pct: bh_return,
    })
}

fn should_trade_on_day(timestamps: &[String], idx: usize, frequency: &Frequency) -> bool {
    match frequency {
        Frequency::Daily => true,
        Frequency::Hourly => true, // treat same as daily for daily data
        Frequency::Weekly => {
            // Only trade on Fridays (weekday 4)
            if idx >= timestamps.len() { return false; }
            if let Ok(dt) = chrono::NaiveDate::parse_from_str(&timestamps[idx].get(..10).unwrap_or(""), "%Y-%m-%d") {
                dt.weekday() == chrono::Weekday::Fri
            } else if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&timestamps[idx]) {
                dt.weekday() == chrono::Weekday::Fri
            } else {
                // Fallback: every 5th day
                idx % 5 == 4
            }
        }
        Frequency::Monthly => {
            // Last trading day of month: next day is different month or end of data
            if idx >= timestamps.len() { return false; }
            let current_month = extract_month(&timestamps[idx]);
            if idx + 1 < timestamps.len() {
                let next_month = extract_month(&timestamps[idx + 1]);
                current_month != next_month
            } else {
                true // last data point
            }
        }
    }
}

fn extract_month(ts: &str) -> String {
    ts.get(..7).unwrap_or("").to_string()
}

fn compute_max_drawdown(equity: &[f64]) -> f64 {
    if equity.is_empty() { return 0.0; }
    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > peak { peak = e; }
        let dd = (peak - e) / peak * 100.0;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

// ════════════════════════════════════════
// Output: CSV + Markdown report
// ════════════════════════════════════════

pub fn write_csv(rows: &[ComparisonRow], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "Asset,Asset Class,Mode,Frequency,Sharpe,Win Rate %,vs B&H %,Trades/yr,Max Drawdown %,Total Return %,B&H Return %")?;
    for r in rows {
        writeln!(f, "{},{},{},{},{:.2},{:.1},{:.2},{:.0},{:.2},{:.2},{:.2}",
            r.asset, r.asset_class, r.mode, r.frequency,
            r.sharpe, r.win_rate, r.vs_bh_pct, r.trades_per_year,
            r.max_drawdown, r.total_return_pct, r.bh_return_pct)?;
    }
    Ok(())
}

pub fn write_report(rows: &[ComparisonRow], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    writeln!(f, "# Multi-Frequency Backtest Report")?;
    writeln!(f, "")?;
    writeln!(f, "Generated: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M UTC"))?;
    writeln!(f, "")?;
    writeln!(f, "## Transaction Costs")?;
    writeln!(f, "- Stocks: 10 bps")?;
    writeln!(f, "- FX: 5 bps")?;
    writeln!(f, "- Crypto: 25 bps")?;
    writeln!(f, "")?;

    // Group by asset class
    for class in &["stock", "fx", "crypto"] {
        let class_rows: Vec<&ComparisonRow> = rows.iter().filter(|r| r.asset_class == *class).collect();
        if class_rows.is_empty() { continue; }

        let class_name = match *class {
            "stock" => "Stocks & ETFs",
            "fx" => "FX Pairs",
            "crypto" => "Crypto",
            _ => class,
        };

        writeln!(f, "## {}", class_name)?;
        writeln!(f, "")?;
        writeln!(f, "| Mode | Frequency | Sharpe | Win Rate | vs B&H % | Trades/yr | Max Drawdown |")?;
        writeln!(f, "|------|-----------|--------|----------|----------|-----------|--------------|")?;

        // Aggregate by mode + frequency
        let modes = [BacktestMode::Academic, BacktestMode::Realistic, BacktestMode::SignalOnly];
        let freqs = [Frequency::Daily, Frequency::Weekly, Frequency::Monthly];

        for mode in &modes {
            for freq in &freqs {
                let matching: Vec<&&ComparisonRow> = class_rows.iter()
                    .filter(|r| std::mem::discriminant(&r.mode) == std::mem::discriminant(mode)
                        && std::mem::discriminant(&r.frequency) == std::mem::discriminant(freq))
                    .collect();
                if matching.is_empty() { continue; }

                let avg_sharpe = matching.iter().map(|r| r.sharpe).sum::<f64>() / matching.len() as f64;
                let avg_wr = matching.iter().map(|r| r.win_rate).sum::<f64>() / matching.len() as f64;
                let avg_vs = matching.iter().map(|r| r.vs_bh_pct).sum::<f64>() / matching.len() as f64;
                let avg_tpy = matching.iter().map(|r| r.trades_per_year).sum::<f64>() / matching.len() as f64;
                let avg_dd = matching.iter().map(|r| r.max_drawdown).sum::<f64>() / matching.len() as f64;

                writeln!(f, "| {} | {} | {:.2} | {:.1}% | {:+.2}% | {:.0} | {:.2}% |",
                    mode, freq, avg_sharpe, avg_wr, avg_vs, avg_tpy, avg_dd)?;
            }
        }
        writeln!(f, "")?;

        // Per-asset detail for Daily Academic
        writeln!(f, "### Per-Asset Detail (Daily, Academic)")?;
        writeln!(f, "")?;
        writeln!(f, "| Asset | Sharpe | Win Rate | vs B&H % | Trades/yr | Max DD % | Return % | B&H % |")?;
        writeln!(f, "|-------|--------|----------|----------|-----------|----------|----------|-------|")?;

        let daily_academic: Vec<&&ComparisonRow> = class_rows.iter()
            .filter(|r| matches!(r.mode, BacktestMode::Academic) && matches!(r.frequency, Frequency::Daily))
            .collect();

        for r in &daily_academic {
            let verdict = if r.vs_bh_pct > 0.0 { "+" } else { "-" };
            writeln!(f, "| {} | {:.2} | {:.1}% | {:+.2}% | {:.0} | {:.2}% | {:.2}% | {:.2}% | {}",
                r.asset, r.sharpe, r.win_rate, r.vs_bh_pct, r.trades_per_year,
                r.max_drawdown, r.total_return_pct, r.bh_return_pct, verdict)?;
        }
        writeln!(f, "")?;
    }

    // Summary verdict
    writeln!(f, "## Overall Verdict")?;
    writeln!(f, "")?;

    let academic_daily: Vec<&ComparisonRow> = rows.iter()
        .filter(|r| matches!(r.mode, BacktestMode::Academic) && matches!(r.frequency, Frequency::Daily))
        .collect();

    if !academic_daily.is_empty() {
        let beating_bh = academic_daily.iter().filter(|r| r.vs_bh_pct > 0.0).count();
        let avg_excess = academic_daily.iter().map(|r| r.vs_bh_pct).sum::<f64>() / academic_daily.len() as f64;
        let avg_sharpe = academic_daily.iter().map(|r| r.sharpe).sum::<f64>() / academic_daily.len() as f64;

        writeln!(f, "- **Assets beating B&H (Academic, Daily):** {}/{} ({:.0}%)",
            beating_bh, academic_daily.len(),
            beating_bh as f64 / academic_daily.len() as f64 * 100.0)?;
        writeln!(f, "- **Average excess return:** {:+.2}%", avg_excess)?;
        writeln!(f, "- **Average Sharpe ratio:** {:.2}", avg_sharpe)?;
        writeln!(f, "")?;

        if avg_excess > 0.0 && avg_sharpe > 0.5 {
            writeln!(f, "**Verdict: Signals provide positive alpha on average.**")?;
        } else if avg_excess > 0.0 {
            writeln!(f, "**Verdict: Marginal edge, risk-adjusted returns mixed.**")?;
        } else {
            writeln!(f, "**Verdict: Buy & hold outperforms on average. Signal refinement needed.**")?;
        }
    }

    writeln!(f, "")?;
    writeln!(f, "_Not financial advice. Past performance does not guarantee future results._")?;

    Ok(())
}
