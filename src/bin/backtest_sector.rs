//! Sector Rotation Backtest
//!
//! Reads walk-forward signals and compares portfolio performance
//! WITH vs WITHOUT sector rotation weighting.
//!
//! Methodology matches the Simulator:
//!   - Long-only: SELL/SHORT signals exit to cash, never open short positions
//!   - Top 8 BUY signals by confidence per day
//!   - 15% max single position cap
//!   - 0.2% transaction cost per trade
//!   - Remaining capital held as cash
//!
//! Output: reports/sector_backtest.json

use rust_invest::sector;
use std::collections::HashMap;
use chrono::{NaiveDate, Datelike};

const MAX_POSITIONS: usize = 8;
const MAX_POSITION_PCT: f64 = 0.15; // 15% cap per position
const TX_COST: f64 = 0.002;         // 0.2% per trade
const MIN_CONFIDENCE: f64 = 0.10;   // 10% minimum confidence to enter

#[derive(serde::Deserialize)]
struct WalkForwardData {
    signals: Vec<WFSignal>,
}

#[derive(serde::Deserialize, Clone)]
struct WFSignal {
    date: String,
    asset: String,
    asset_class: String,
    signal: String,
    #[allow(dead_code)]
    entry_price: f64,
    exit_price: Option<f64>,
    pct_return: Option<f64>,
    confidence: f64,
}

#[derive(serde::Serialize)]
struct SectorBacktestOutput {
    generated_at: String,
    methodology: String,
    baseline: StrategyResult,
    sector_weighted: StrategyResult,
    improvement: Improvement,
    quarterly: Vec<QuarterlyComparison>,
}

#[derive(serde::Serialize, Clone)]
struct StrategyResult {
    name: String,
    total_return_pct: f64,
    cagr_pct: f64,
    sharpe_ratio: f64,
    max_drawdown_pct: f64,
    total_trades: usize,
    win_rate_pct: f64,
    profit_factor: f64,
    final_equity: f64,
    avg_positions_per_day: f64,
    days_fully_in_cash: usize,
}

#[derive(serde::Serialize)]
struct Improvement {
    cagr_delta_pct: f64,
    sharpe_delta: f64,
    drawdown_delta_pct: f64,
    win_rate_delta_pct: f64,
    verdict: String,
}

#[derive(serde::Serialize)]
struct QuarterlyComparison {
    quarter: String,
    baseline_return_pct: f64,
    sector_return_pct: f64,
    delta_pct: f64,
    strongest_sector: String,
    weakest_sector: String,
}

fn main() {
    println!("\n══════════════════════════════════════");
    println!("  Sector Rotation Backtest");
    println!("  Long-only | Top 8 | 15% cap | 0.2% tx cost");
    println!("══════════════════════════════════════\n");

    // Load walk-forward signals
    let wf_path = "reports/walkforward_backtest.json";
    let raw = match std::fs::read_to_string(wf_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {}", wf_path, e);
            eprintln!("Run `cargo run --release --bin backtest_walkforward` first.");
            std::process::exit(1);
        }
    };
    let wf_data: WalkForwardData = serde_json::from_str(&raw).expect("Failed to parse walk-forward data");

    // Keep ALL signals (including HOLD, SELL, SHORT) — we need them for exit logic
    let resolved: Vec<&WFSignal> = wf_data.signals.iter()
        .filter(|s| s.pct_return.is_some() && s.exit_price.is_some())
        .collect();

    println!("  Loaded {} resolved signals from walk-forward data", resolved.len());

    // Group ALL signals by date (need SELL/SHORT to trigger exits)
    let mut by_date: HashMap<String, Vec<&WFSignal>> = HashMap::new();
    for sig in &resolved {
        by_date.entry(sig.date.clone()).or_default().push(sig);
    }
    let mut dates: Vec<String> = by_date.keys().cloned().collect();
    dates.sort();

    println!("  Date range: {} to {} ({} trading days)", dates.first().unwrap_or(&String::new()),
        dates.last().unwrap_or(&String::new()), dates.len());

    // Count BUY signals for context
    let buy_count = resolved.iter().filter(|s| s.signal == "BUY").count();
    let sell_count = resolved.iter().filter(|s| s.signal == "SELL" || s.signal == "SHORT").count();
    let hold_count = resolved.iter().filter(|s| s.signal == "HOLD").count();
    println!("  Signals: {} BUY, {} SELL/SHORT, {} HOLD", buy_count, sell_count, hold_count);
    println!("  Rules: top {} BUY by confidence, {:.0}% max position, {:.1}% tx cost, {:.0}% min confidence\n",
        MAX_POSITIONS, MAX_POSITION_PCT * 100.0, TX_COST * 100.0, MIN_CONFIDENCE * 100.0);

    // Run both strategies
    let baseline = run_portfolio(&dates, &by_date, false);
    let sector_weighted = run_portfolio(&dates, &by_date, true);

    // Quarterly breakdowns
    let quarterly = compute_quarterly(&dates, &by_date);

    let cagr_delta = sector_weighted.cagr_pct - baseline.cagr_pct;
    let sharpe_delta = sector_weighted.sharpe_ratio - baseline.sharpe_ratio;
    let dd_delta = sector_weighted.max_drawdown_pct - baseline.max_drawdown_pct;
    let wr_delta = sector_weighted.win_rate_pct - baseline.win_rate_pct;

    let verdict = if cagr_delta > 0.5 && sharpe_delta > 0.05 {
        "Sector rotation improves both returns and risk-adjusted performance."
    } else if cagr_delta > 0.0 {
        "Sector rotation provides marginal improvement in returns."
    } else if sharpe_delta > 0.0 {
        "Sector rotation improves risk-adjusted returns despite lower absolute returns."
    } else {
        "Sector rotation does not improve performance in this period."
    };

    let output = SectorBacktestOutput {
        generated_at: chrono::Utc::now().to_rfc3339(),
        methodology: format!(
            "Long-only, top {} BUY signals by confidence, {}% max position, {}% tx cost, {}% min confidence. \
             SELL/SHORT signals exit to cash. Matches Simulator methodology.",
            MAX_POSITIONS, (MAX_POSITION_PCT * 100.0) as u32, (TX_COST * 100.0) as u32, (MIN_CONFIDENCE * 100.0) as u32
        ),
        baseline: baseline.clone(),
        sector_weighted: sector_weighted.clone(),
        improvement: Improvement {
            cagr_delta_pct: round2(cagr_delta),
            sharpe_delta: round2(sharpe_delta),
            drawdown_delta_pct: round2(dd_delta),
            win_rate_delta_pct: round2(wr_delta),
            verdict: verdict.to_string(),
        },
        quarterly,
    };

    // Write output
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize");
    std::fs::create_dir_all("reports").ok();
    std::fs::write("reports/sector_backtest.json", &json).expect("Failed to write report");

    // Print summary
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│  {:^60}  │", "RESULTS");
    println!("├──────────────────────┬──────────────────┬─────────────────────┤");
    println!("│  {:20} │  {:16} │  {:19} │", "Metric", "Baseline", "Sector-Weighted");
    println!("├──────────────────────┼──────────────────┼─────────────────────┤");
    println!("│  {:20} │  {:>15.2}% │  {:>18.2}% │", "CAGR", baseline.cagr_pct, sector_weighted.cagr_pct);
    println!("│  {:20} │  {:>16.2} │  {:>19.2} │", "Sharpe Ratio", baseline.sharpe_ratio, sector_weighted.sharpe_ratio);
    println!("│  {:20} │  {:>15.2}% │  {:>18.2}% │", "Max Drawdown", baseline.max_drawdown_pct, sector_weighted.max_drawdown_pct);
    println!("│  {:20} │  {:>15.1}% │  {:>18.1}% │", "Win Rate", baseline.win_rate_pct, sector_weighted.win_rate_pct);
    println!("│  {:20} │  {:>16.2} │  {:>19.2} │", "Profit Factor", baseline.profit_factor, sector_weighted.profit_factor);
    println!("│  {:20} │  {:>15.2}% │  {:>18.2}% │", "Total Return", baseline.total_return_pct, sector_weighted.total_return_pct);
    println!("│  {:20} │  {:>16.1} │  {:>19.1} │", "Avg Pos/Day", baseline.avg_positions_per_day, sector_weighted.avg_positions_per_day);
    println!("│  {:20} │  {:>16} │  {:>19} │", "Days in Cash", baseline.days_fully_in_cash, sector_weighted.days_fully_in_cash);
    println!("├──────────────────────┴──────────────────┴─────────────────────┤");
    println!("│  CAGR delta: {:+.2}%  |  Sharpe delta: {:+.2}  |  DD delta: {:+.2}%  │",
        cagr_delta, sharpe_delta, dd_delta);
    println!("│  {:60}  │", verdict);
    println!("└────────────────────────────────────────────────────────────────┘");
    println!("\n  Output: reports/sector_backtest.json\n");
}

/// Long-only portfolio simulation matching the Simulator methodology.
///
/// Each day:
/// 1. SELL phase: any held asset with a SELL/SHORT signal is sold to cash (with tx cost)
/// 2. BUY phase: top N BUY signals by confidence are bought (with tx cost)
/// 3. Position cap: 15% max per position
/// 4. Remaining capital stays in cash
///
/// If `use_sectors` is true, confidence is boosted by sector weight_multiplier
/// before ranking, so assets in strong sectors rank higher.
fn run_portfolio(
    dates: &[String],
    by_date: &HashMap<String, Vec<&WFSignal>>,
    use_sectors: bool,
) -> StrategyResult {
    let initial_equity = 100_000.0;
    let mut cash = initial_equity;
    let mut peak = initial_equity;
    let mut max_dd = 0.0_f64;
    let mut daily_returns: Vec<f64> = Vec::new();
    let mut total_trades = 0usize;
    let mut wins = 0usize;
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;
    let mut total_positions_sum = 0usize;
    let mut days_in_cash = 0usize;

    // Track current positions: asset → fraction of equity invested
    // We track the actual dollar amount invested in each position
    let mut positions: HashMap<String, f64> = HashMap::new(); // asset → dollar value

    for date in dates {
        let signals = match by_date.get(date) {
            Some(s) => s,
            None => {
                // No signals today — positions stay, record return of 0
                let equity = cash + positions.values().sum::<f64>();
                daily_returns.push(0.0);
                if equity > peak { peak = equity; }
                let dd = (peak - equity) / peak * 100.0;
                if dd > max_dd { max_dd = dd; }
                if positions.is_empty() { days_in_cash += 1; }
                total_positions_sum += positions.len();
                continue;
            }
        };

        let equity_before = cash + positions.values().sum::<f64>();

        // Build a lookup: asset → signal for this day
        let signal_map: HashMap<&str, &WFSignal> = signals.iter()
            .map(|s| (s.asset.as_str(), *s))
            .collect();

        // ── Phase 1: Mark-to-market existing positions using today's pct_return ──
        // Each position's value changes by the asset's pct_return for the day
        let held_assets: Vec<String> = positions.keys().cloned().collect();
        for asset in &held_assets {
            if let Some(sig) = signal_map.get(asset.as_str()) {
                if let Some(pct) = sig.pct_return {
                    let pos_value = positions.get_mut(asset).unwrap();
                    let daily_pnl = *pos_value * (pct / 100.0);
                    *pos_value += daily_pnl;

                    if daily_pnl > 0.0 {
                        gross_profit += daily_pnl;
                    } else {
                        gross_loss += daily_pnl.abs();
                    }
                }
            }
        }

        // ── Phase 2: SELL — exit positions with SELL/SHORT signals ──
        let mut assets_to_sell: Vec<String> = Vec::new();
        for asset in positions.keys() {
            if let Some(sig) = signal_map.get(asset.as_str()) {
                if sig.signal == "SELL" || sig.signal == "SHORT" {
                    assets_to_sell.push(asset.clone());
                }
            }
        }
        for asset in &assets_to_sell {
            if let Some(value) = positions.remove(asset) {
                let proceeds = value * (1.0 - TX_COST); // deduct tx cost
                cash += proceeds;
                total_trades += 1;
            }
        }

        // ── Phase 3: BUY — select top N BUY signals by (adjusted) confidence ──
        // Compute sector weights for this day if using sectors
        let sector_multipliers: HashMap<String, f64> = if use_sectors {
            let inputs: Vec<sector::SignalInput> = signals.iter().map(|s| {
                sector::SignalInput {
                    asset: s.asset.clone(),
                    asset_class: s.asset_class.clone(),
                    signal: s.signal.clone(),
                    probability_up: 50.0 + (s.confidence - 0.5) * 100.0,
                    confidence: s.confidence * 10.0,
                }
            }).collect();
            let scores = sector::calculate_sector_scores(&inputs);
            scores.into_iter()
                .map(|sc| (sc.label, sc.weight_multiplier))
                .collect()
        } else {
            HashMap::new()
        };

        // Filter to BUY signals with sufficient confidence, not already held
        let mut buy_candidates: Vec<(&WFSignal, f64)> = signals.iter()
            .filter(|s| {
                s.signal == "BUY"
                    && s.confidence >= MIN_CONFIDENCE
                    && !positions.contains_key(&s.asset)
            })
            .map(|s| {
                let sector_label = sector::classify_sector_with_class(&s.asset, &s.asset_class)
                    .label().to_string();
                let mult = if use_sectors {
                    *sector_multipliers.get(&sector_label).unwrap_or(&1.0)
                } else {
                    1.0
                };
                // Adjusted confidence: raw confidence * sector multiplier
                let adj_confidence = s.confidence * mult;
                (*s, adj_confidence)
            })
            .collect();

        // Sort by adjusted confidence descending, take top MAX_POSITIONS
        buy_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let slots_available = MAX_POSITIONS.saturating_sub(positions.len());
        buy_candidates.truncate(slots_available);

        // Allocate capital: equal weight among selected, capped at MAX_POSITION_PCT
        let current_equity = cash + positions.values().sum::<f64>();
        if !buy_candidates.is_empty() && cash > 0.0 {
            let target_weight = (1.0 / buy_candidates.len() as f64).min(MAX_POSITION_PCT);
            for (sig, _adj_conf) in &buy_candidates {
                let target_amount = current_equity * target_weight;
                let invest = target_amount.min(cash); // can't invest more than available cash
                if invest < 100.0 { continue; } // skip tiny positions
                let cost = invest * TX_COST;
                let net_invested = invest - cost;
                cash -= invest;
                positions.insert(sig.asset.clone(), net_invested);
                total_trades += 1;
            }
        }

        // ── Compute daily return ──
        let equity_after = cash + positions.values().sum::<f64>();
        let day_return = if equity_before > 0.0 {
            (equity_after - equity_before) / equity_before
        } else { 0.0 };
        daily_returns.push(day_return);

        // Track wins (days with positive return on invested capital)
        if day_return > 0.0 { wins += 1; }

        if equity_after > peak { peak = equity_after; }
        let dd = (peak - equity_after) / peak * 100.0;
        if dd > max_dd { max_dd = dd; }

        total_positions_sum += positions.len();
        if positions.is_empty() { days_in_cash += 1; }
    }

    let final_equity = cash + positions.values().sum::<f64>();
    let total_return = (final_equity - initial_equity) / initial_equity * 100.0;

    // CAGR
    let first_date = dates.first().and_then(|d| NaiveDate::parse_from_str(&d[..10], "%Y-%m-%d").ok());
    let last_date = dates.last().and_then(|d| NaiveDate::parse_from_str(&d[..10], "%Y-%m-%d").ok());
    let years = match (first_date, last_date) {
        (Some(f), Some(l)) => (l - f).num_days() as f64 / 365.25,
        _ => dates.len() as f64 / 252.0,
    };
    let cagr = if years > 0.0 && final_equity > 0.0 {
        ((final_equity / initial_equity).powf(1.0 / years) - 1.0) * 100.0
    } else { 0.0 };

    // Sharpe from daily returns
    let mean = if !daily_returns.is_empty() {
        daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
    } else { 0.0 };
    let std = if daily_returns.len() > 1 {
        let var = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (daily_returns.len() - 1) as f64;
        var.sqrt()
    } else { 0.0 };
    let rf_daily = 0.045 / 252.0;
    let sharpe = if std > 1e-10 { ((mean - rf_daily) / std) * 252.0_f64.sqrt() } else { 0.0 };

    // Win rate = fraction of trading days with positive return
    let trading_days = daily_returns.len();
    let win_rate = if trading_days > 0 { wins as f64 / trading_days as f64 * 100.0 } else { 0.0 };
    let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else if gross_profit > 0.0 { 99.99 } else { 0.0 };
    let avg_pos = if trading_days > 0 { total_positions_sum as f64 / trading_days as f64 } else { 0.0 };

    StrategyResult {
        name: if use_sectors { "Sector-Weighted".to_string() } else { "Baseline".to_string() },
        total_return_pct: round2(total_return),
        cagr_pct: round2(cagr),
        sharpe_ratio: round2(sharpe),
        max_drawdown_pct: round2(max_dd),
        total_trades,
        win_rate_pct: round1(win_rate),
        profit_factor: round2(profit_factor),
        final_equity: round2(final_equity),
        avg_positions_per_day: round1(avg_pos),
        days_fully_in_cash: days_in_cash,
    }
}

/// Quarterly breakdown comparing both strategies.
/// Note: quarterly results are independent (positions don't carry across quarters)
/// so they won't sum exactly to the full-period result.
fn compute_quarterly(
    dates: &[String],
    by_date: &HashMap<String, Vec<&WFSignal>>,
) -> Vec<QuarterlyComparison> {
    let mut results = Vec::new();

    // Group dates by quarter
    let mut quarters: HashMap<String, Vec<String>> = HashMap::new();
    for date in dates {
        if let Ok(d) = NaiveDate::parse_from_str(&date[..10], "%Y-%m-%d") {
            let q = match d.month() {
                1..=3 => format!("{}-Q1", d.year()),
                4..=6 => format!("{}-Q2", d.year()),
                7..=9 => format!("{}-Q3", d.year()),
                _ => format!("{}-Q4", d.year()),
            };
            quarters.entry(q).or_default().push(date.clone());
        }
    }

    let mut quarter_keys: Vec<String> = quarters.keys().cloned().collect();
    quarter_keys.sort();

    for qk in &quarter_keys {
        let q_dates = &quarters[qk];

        // Build sub-map for this quarter
        let q_by_date: HashMap<String, Vec<&WFSignal>> = q_dates.iter()
            .filter_map(|d| by_date.get(d).map(|sigs| (d.clone(), sigs.clone())))
            .collect();

        let baseline = run_portfolio(q_dates, &q_by_date, false);
        let sector = run_portfolio(q_dates, &q_by_date, true);

        // Find strongest/weakest sector for this quarter (from BUY signals only)
        let all_q_sigs: Vec<sector::SignalInput> = q_dates.iter()
            .filter_map(|d| by_date.get(d))
            .flat_map(|sigs| sigs.iter())
            .filter(|s| s.signal != "HOLD")
            .map(|s| sector::SignalInput {
                asset: s.asset.clone(),
                asset_class: s.asset_class.clone(),
                signal: s.signal.clone(),
                probability_up: 50.0 + (s.confidence - 0.5) * 100.0,
                confidence: s.confidence * 10.0,
            })
            .collect();
        let overview = sector::build_sector_overview(&all_q_sigs);

        results.push(QuarterlyComparison {
            quarter: qk.clone(),
            baseline_return_pct: baseline.total_return_pct,
            sector_return_pct: sector.total_return_pct,
            delta_pct: round2(sector.total_return_pct - baseline.total_return_pct),
            strongest_sector: overview.strongest_sector,
            weakest_sector: overview.weakest_sector,
        });
    }

    results
}

fn round2(v: f64) -> f64 { (v * 100.0).round() / 100.0 }
fn round1(v: f64) -> f64 { (v * 10.0).round() / 10.0 }
