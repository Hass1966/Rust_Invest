/// Portfolio Allocator — Distribute Capital Across Multiple Assets
/// ================================================================
/// Takes individual per-asset backtest results and simulates a
/// combined portfolio with intelligent capital allocation.
///
/// Weighting schemes:
///   1. Equal Weight — same $ in each asset
///   2. Sharpe-Weighted — more capital to better risk-adjusted assets
///   3. Inverse-Volatility — more capital to lower-risk assets
///
/// The portfolio replays daily returns with proper rebalancing,
/// giving a realistic "if I actually traded this" result.

use crate::backtester::BacktestResult;

// ════════════════════════════════════════
// Portfolio Configuration
// ════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct PortfolioConfig {
    pub initial_capital: f64,
    pub weighting: WeightingScheme,
    /// Only include assets with Sharpe > this threshold
    pub min_sharpe: f64,
    /// Only include assets with positive excess return
    pub require_positive_excess: bool,
    /// Transaction cost for rebalancing (per trade, fractional)
    pub rebalance_cost: f64,
}

#[derive(Clone, Debug)]
pub enum WeightingScheme {
    EqualWeight,
    SharpeWeighted,
    InverseVolatility,
}

impl std::fmt::Display for WeightingScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightingScheme::EqualWeight => write!(f, "Equal Weight"),
            WeightingScheme::SharpeWeighted => write!(f, "Sharpe-Weighted"),
            WeightingScheme::InverseVolatility => write!(f, "Inverse-Volatility"),
        }
    }
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        PortfolioConfig {
            initial_capital: 100_000.0,
            weighting: WeightingScheme::SharpeWeighted,
            min_sharpe: 0.5,
            require_positive_excess: true,
            rebalance_cost: 0.001,
        }
    }
}

// ════════════════════════════════════════
// Portfolio Result
// ════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct AssetAllocation {
    pub symbol: String,
    pub weight: f64,
    pub capital: f64,
    pub sharpe: f64,
    pub asset_return: f64,
    pub contribution: f64,  // weight × return
}

#[derive(Clone, Debug)]
pub struct PortfolioResult {
    pub config: PortfolioConfig,
    pub allocations: Vec<AssetAllocation>,
    pub total_return_pct: f64,
    pub annualised_return_pct: f64,
    pub benchmark_return_pct: f64,
    pub excess_return_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown_pct: f64,
    pub volatility_pct: f64,
    pub equity_curve: Vec<f64>,
    pub benchmark_curve: Vec<f64>,
    pub daily_returns: Vec<f64>,
    pub total_days: usize,
    pub n_assets: usize,
}

// ════════════════════════════════════════
// Portfolio Construction
// ════════════════════════════════════════

/// Build a combined portfolio from individual backtest results.
///
/// The approach:
///   1. Filter assets by quality criteria (min Sharpe, positive excess)
///   2. Compute weights based on the chosen scheme
///   3. Combine daily returns: portfolio_return = Σ(weight_i × return_i)
///   4. Track combined equity curve, compute portfolio-level metrics
pub fn build_portfolio(
    results: &[BacktestResult],
    config: &PortfolioConfig,
) -> Option<PortfolioResult> {
    // Filter to qualifying assets
    let qualifying: Vec<&BacktestResult> = results.iter()
        .filter(|r| {
            let passes_sharpe = r.sharpe_ratio >= config.min_sharpe;
            let passes_excess = !config.require_positive_excess || r.excess_return_pct > 0.0;
            passes_sharpe && passes_excess
        })
        .collect();

    if qualifying.is_empty() {
        println!("  [Portfolio] No assets meet quality criteria");
        return None;
    }

    println!("  [Portfolio] {} assets qualify (of {} tested)", qualifying.len(), results.len());

    // Compute weights
    let raw_weights: Vec<f64> = match config.weighting {
        WeightingScheme::EqualWeight => {
            vec![1.0; qualifying.len()]
        }
        WeightingScheme::SharpeWeighted => {
            qualifying.iter()
                .map(|r| r.sharpe_ratio.max(0.0))
                .collect()
        }
        WeightingScheme::InverseVolatility => {
            qualifying.iter()
                .map(|r| {
                    if r.volatility_pct > 0.0 { 1.0 / r.volatility_pct }
                    else { 1.0 }
                })
                .collect()
        }
    };

    // Normalise weights to sum to 1.0
    let weight_sum: f64 = raw_weights.iter().sum();
    let weights: Vec<f64> = if weight_sum > 0.0 {
        raw_weights.iter().map(|w| w / weight_sum).collect()
    } else {
        vec![1.0 / qualifying.len() as f64; qualifying.len()]
    };

    // Build allocations
    let allocations: Vec<AssetAllocation> = qualifying.iter().zip(weights.iter())
        .map(|(r, &w)| AssetAllocation {
            symbol: r.symbol.clone(),
            weight: w,
            capital: config.initial_capital * w,
            sharpe: r.sharpe_ratio,
            asset_return: r.total_return_pct,
            contribution: w * r.total_return_pct,
        })
        .collect();

    // Print allocations
    println!("  [Portfolio] Allocation ({}):", config.weighting);
    for a in &allocations {
        println!("    {:>8}: {:>5.1}%  (${:>9.0})  Sharpe: {:.2}  Return: {:>7.1}%",
            a.symbol, a.weight * 100.0, a.capital, a.sharpe, a.asset_return);
    }

    // Combine daily returns
    // Find the common trading day count (use the minimum across all assets)
    let min_days = qualifying.iter()
        .map(|r| r.daily_returns.len())
        .min()
        .unwrap_or(0);

    if min_days < 10 {
        println!("  [Portfolio] Not enough common trading days ({})", min_days);
        return None;
    }

    let mut portfolio_daily = Vec::with_capacity(min_days);
    let mut benchmark_daily = Vec::with_capacity(min_days);

    for day in 0..min_days {
        let port_return: f64 = qualifying.iter().zip(weights.iter())
            .map(|(r, &w)| w * r.daily_returns[day])
            .sum();
        portfolio_daily.push(port_return);

        // Benchmark: weighted buy-and-hold daily returns
        // (approximate from equity curves)
        let bench_return: f64 = qualifying.iter().zip(weights.iter())
            .map(|(r, &w)| {
                if r.benchmark_curve.len() > day + 1 && r.benchmark_curve[day] > 0.0 {
                    let b_ret = (r.benchmark_curve[day + 1] - r.benchmark_curve[day])
                        / r.benchmark_curve[day];
                    w * b_ret
                } else {
                    0.0
                }
            })
            .sum();
        benchmark_daily.push(bench_return);
    }

    // Build equity curves
    let mut equity = config.initial_capital;
    let mut equity_curve = Vec::with_capacity(min_days + 1);
    equity_curve.push(equity);

    for &r in &portfolio_daily {
        equity *= 1.0 + r;
        equity_curve.push(equity);
    }

    let mut bench_equity = config.initial_capital;
    let mut bench_curve = Vec::with_capacity(min_days + 1);
    bench_curve.push(bench_equity);

    for &r in &benchmark_daily {
        bench_equity *= 1.0 + r;
        bench_curve.push(bench_equity);
    }

    // Compute metrics
    let total_return = (equity - config.initial_capital) / config.initial_capital * 100.0;
    let bench_return = (bench_equity - config.initial_capital) / config.initial_capital * 100.0;

    let trading_days_per_year = 252.0;
    let years = min_days as f64 / trading_days_per_year;
    let ann_return = if years > 0.0 {
        ((equity / config.initial_capital).powf(1.0 / years) - 1.0) * 100.0
    } else {
        0.0
    };

    // Sharpe
    let mean_daily = portfolio_daily.iter().sum::<f64>() / portfolio_daily.len() as f64;
    let std_daily = if portfolio_daily.len() > 1 {
        let var = portfolio_daily.iter()
            .map(|r| (r - mean_daily).powi(2))
            .sum::<f64>() / (portfolio_daily.len() - 1) as f64;
        var.sqrt()
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
    let max_dd = compute_max_drawdown(&equity_curve);

    let result = PortfolioResult {
        config: config.clone(),
        allocations,
        total_return_pct: total_return,
        annualised_return_pct: ann_return,
        benchmark_return_pct: bench_return,
        excess_return_pct: total_return - bench_return,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_dd,
        volatility_pct: volatility,
        equity_curve,
        benchmark_curve: bench_curve,
        daily_returns: portfolio_daily,
        total_days: min_days,
        n_assets: qualifying.len(),
    };

    print_portfolio_result(&result);
    Some(result)
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
// Console Output
// ════════════════════════════════════════

fn print_portfolio_result(r: &PortfolioResult) {
    println!("\n  ╔═══════════════════════════════════════════════════════════════╗");
    println!("  ║  PORTFOLIO RESULT — {} ({} assets)          ║", r.config.weighting, r.n_assets);
    println!("  ╠═══════════════════════════════════════════════════════════════╣");
    println!("  ║  Starting Capital:   ${:>10.0}                            ║", r.config.initial_capital);
    println!("  ║  Final Value:        ${:>10.0}                            ║",
        r.config.initial_capital * (1.0 + r.total_return_pct / 100.0));
    println!("  ║  Total Return:       {:>8.2}%                              ║", r.total_return_pct);
    println!("  ║  Benchmark (B&H):    {:>8.2}%                              ║", r.benchmark_return_pct);
    println!("  ║  Excess Return:      {:>8.2}%                              ║", r.excess_return_pct);
    println!("  ║  Annualised:         {:>8.2}%                              ║", r.annualised_return_pct);
    println!("  ║  Sharpe Ratio:       {:>8.2}                               ║", r.sharpe_ratio);
    println!("  ║  Max Drawdown:       {:>8.2}%                              ║", r.max_drawdown_pct);
    println!("  ║  Volatility (ann):   {:>8.2}%                              ║", r.volatility_pct);
    println!("  ║  Trading Days:       {:>8}                                 ║", r.total_days);
    println!("  ╠═══════════════════════════════════════════════════════════════╣");
    println!("  ║  ALLOCATIONS:                                               ║");
    for a in &r.allocations {
        println!("  ║    {:>8}: {:>5.1}% (${:>8.0}) → contributed {:>6.1}%        ║",
            a.symbol, a.weight * 100.0, a.capital, a.contribution);
    }
    println!("  ╚═══════════════════════════════════════════════════════════════╝");
}

// ════════════════════════════════════════
// HTML for Report
// ════════════════════════════════════════

pub fn portfolio_html(result: &PortfolioResult) -> String {
    let mut html = String::new();

    let final_value = result.config.initial_capital * (1.0 + result.total_return_pct / 100.0);

    // Allocation donut data (for CSS donut chart)
    let mut donut_segments = String::new();
    let mut offset = 0.0_f64;
    let colors = ["#00d4aa","#4fc3f7","#ffd740","#ff8a65","#ce93d8",
                  "#80cbc4","#ef5350","#66bb6a","#42a5f5","#ffca28"];

    for (i, a) in result.allocations.iter().enumerate() {
        let pct = a.weight * 100.0;
        let color = colors[i % colors.len()];
        donut_segments.push_str(&format!(
            "<div class='alloc-row'>\
             <span class='alloc-dot' style='background:{}'></span>\
             <span class='alloc-sym'>{}</span>\
             <span class='alloc-pct'>{:.1}%</span>\
             <span class='alloc-amt'>${:.0}</span>\
             <span class='alloc-ret' style='color:{}'>{:+.1}%</span>\
             </div>",
            color, a.symbol, pct, a.capital,
            if a.contribution >= 0.0 { "#00e676" } else { "#ff5252" },
            a.contribution,
        ));
        offset += pct;
    }

    // Equity curve SVG
    let eq_svg = portfolio_equity_svg(&result.equity_curve, &result.benchmark_curve, 720.0, 200.0);

    html.push_str(&format!(r#"
<div class="portfolio-section">
  <div class="port-kpis">
    <div class="port-kpi port-kpi-hero">
      <div class="port-kpi-label">Final Portfolio Value</div>
      <div class="port-kpi-value" style="color:#00d4aa;font-size:32px;">${:.0}</div>
      <div class="port-kpi-sub">from $100,000 starting capital</div>
    </div>
    <div class="port-kpi">
      <div class="port-kpi-value" style="color:{}">{:+.1}%</div>
      <div class="port-kpi-label">Total Return</div>
    </div>
    <div class="port-kpi">
      <div class="port-kpi-value" style="color:#4fc3f7">{:.2}</div>
      <div class="port-kpi-label">Sharpe Ratio</div>
    </div>
    <div class="port-kpi">
      <div class="port-kpi-value">{:.1}%</div>
      <div class="port-kpi-label">Max Drawdown</div>
    </div>
    <div class="port-kpi">
      <div class="port-kpi-value" style="color:{}">{:+.1}%</div>
      <div class="port-kpi-label">vs Buy &amp; Hold</div>
    </div>
    <div class="port-kpi">
      <div class="port-kpi-value">{}</div>
      <div class="port-kpi-label">Assets</div>
    </div>
  </div>

  <div class="port-chart-container">
    <h3 style="margin-bottom:12px;">Portfolio Equity Curve</h3>
    <div class="port-legend">
      <span><span style="color:#00d4aa;">━</span> AI Strategy</span>
      <span><span style="color:#555;">━</span> Buy &amp; Hold</span>
      <span style="color:var(--text-dim);">--- $100K start</span>
    </div>
    {}
  </div>

  <div class="port-alloc">
    <h3>Capital Allocation — {}</h3>
    <div class="alloc-list">
      {}
    </div>
  </div>
</div>
"#,
        final_value,
        if result.total_return_pct >= 0.0 { "#00e676" } else { "#ff5252" },
        result.total_return_pct,
        result.sharpe_ratio,
        result.max_drawdown_pct,
        if result.excess_return_pct >= 0.0 { "#00e676" } else { "#ff5252" },
        result.excess_return_pct,
        result.n_assets,
        eq_svg,
        result.config.weighting,
        donut_segments,
    ));

    html
}

fn portfolio_equity_svg(equity: &[f64], benchmark: &[f64], width: f64, height: f64) -> String {
    if equity.is_empty() { return String::new(); }

    let padding = 12.0;
    let all_values: Vec<f64> = equity.iter().chain(benchmark.iter()).copied().collect();
    let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min) * 0.98;
    let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.02;
    let range = (max_val - min_val).max(1.0);

    let make_path = |data: &[f64]| -> String {
        let n = data.len() as f64;
        data.iter().enumerate().map(|(i, v)| {
            let x = padding + (i as f64 / (n - 1.0).max(1.0)) * (width - 2.0 * padding);
            let y = padding + (1.0 - (v - min_val) / range) * (height - 2.0 * padding);
            format!("{:.1},{:.1}", x, y)
        }).collect::<Vec<_>>().join(" ")
    };

    let eq_points = make_path(equity);
    let bm_points = make_path(benchmark);

    // Gradient fill under equity line
    let initial = equity[0];
    let y_ref = padding + (1.0 - (initial - min_val) / range) * (height - 2.0 * padding);

    format!(
        "<svg width='100%' viewBox='0 0 {w} {h}' style='background:#0a1018;border-radius:8px;'>\
         <defs>\
           <linearGradient id='eqGrad' x1='0' y1='0' x2='0' y2='1'>\
             <stop offset='0%' stop-color='#00d4aa' stop-opacity='0.25'/>\
             <stop offset='100%' stop-color='#00d4aa' stop-opacity='0.02'/>\
           </linearGradient>\
         </defs>\
         <line x1='{p}' y1='{yr:.1}' x2='{w2:.1}' y2='{yr:.1}' stroke='#1a2840' stroke-dasharray='6,4'/>\
         <polyline points='{bm}' fill='none' stroke='#3a3a3a' stroke-width='1.5'/>\
         <polygon points='{p},{h2:.0} {eq} {w2:.1},{h2:.0}' fill='url(#eqGrad)'/>\
         <polyline points='{eq}' fill='none' stroke='#00d4aa' stroke-width='2'/>\
         </svg>",
        w = width, h = height, p = padding,
        yr = y_ref, w2 = width - padding,
        h2 = height - padding,
        bm = bm_points, eq = eq_points,
    )
}
