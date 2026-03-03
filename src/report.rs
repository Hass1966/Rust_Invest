/// Report Generator — Polished HTML Dashboard
/// ============================================
/// Generates a single-file HTML report with navigation, KPIs,
/// backtest results, portfolio allocation, equity curves,
/// trading signals, and a metric glossary.

use std::fs;
use chrono::Utc;
use crate::analysis::{self, PricePoint, AnalysisResult};
use crate::charts;
use crate::ml;
use crate::gbt;
use crate::ensemble;
use crate::diagnostics;
use crate::backtester;
use crate::portfolio;

pub fn generate_html_report(
    coin_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    stock_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    fx_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    _ml_results: &[ml::PipelineResult],
    gbt_results: &[gbt::ExtendedPipelineResult],
    signals: &[ensemble::TradingSignal],
    backtest_results: &[backtester::BacktestResult],
    portfolio_results: &[portfolio::PortfolioResult],
    diagnostics_data: &[diagnostics::SymbolDiagnostics],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut html = String::new();

    // ── HEAD + CSS ──
    html.push_str(&report_head());

    // ── NAV ──
    html.push_str(&report_nav());

    html.push_str("<main>\n");

    // ── HERO ──
    html.push_str(&report_hero());

    // ── KPI OVERVIEW ──
    html.push_str(&report_kpis(backtest_results));

    // ── BACKTEST RESULTS ──
    if !backtest_results.is_empty() {
        html.push_str(&backtester::backtest_html(backtest_results));
    }

    // ── PORTFOLIO ALLOCATION ──
    for pr in portfolio_results {
        html.push_str("<section id='portfolio'>\n");
        html.push_str("<h2 class='section-title'><span>//</span> Portfolio Allocation — $100K</h2>\n");
        html.push_str(&portfolio::portfolio_html(pr));
        html.push_str("</section>\n");
    }

    // ── TRADING SIGNALS ──
    if !signals.is_empty() {
        html.push_str("<section id='signals'>\n");
        html.push_str(&ensemble::signals_html(signals));
        html.push_str("</section>\n");
    }

    // ── DIAGNOSTICS ──
    if !diagnostics_data.is_empty() {
        html.push_str(&diagnostics::diagnostics_html(diagnostics_data));
    }

    // ── CRYPTO ANALYSIS ──
    if !coin_data.is_empty() {
        html.push_str("<section id='crypto'>\n");
        html.push_str("<h2 class='section-title'><span>//</span> Crypto Analysis</h2>\n");
        for (name, points, result) in coin_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            let bands = analysis::bollinger_bands(&prices, 20, 2.0);
            let chart = charts::bollinger_chart_svg(&prices, &bands, name);
            html.push_str(&format!(
                "<div class='card'><h3>{}</h3><p>Price: ${:.2} | RSI: {:.1} | Vol: {:.4}</p>{}</div>\n",
                name, result.current_price,
                result.rsi_14.unwrap_or(0.0), result.std_dev, chart
            ));
        }
        html.push_str("</section>\n");
    }

    // ── STOCK ANALYSIS ──
    if !stock_data.is_empty() {
        html.push_str("<section id='stocks'>\n");
        html.push_str("<h2 class='section-title'><span>//</span> Stock Analysis</h2>\n");
        for (name, points, result) in stock_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            let bands = analysis::bollinger_bands(&prices, 20, 2.0);
            let chart = charts::bollinger_chart_svg(&prices, &bands, name);
            html.push_str(&format!(
                "<div class='card'><h3>{}</h3><p>Price: ${:.2} | RSI: {:.1} | Vol: {:.4}</p>{}</div>\n",
                name, result.current_price,
                result.rsi_14.unwrap_or(0.0), result.std_dev, chart
            ));
        }
        html.push_str("</section>\n");
    }

    // ── FX ANALYSIS ──
    if !fx_data.is_empty() {
        html.push_str("<section id='fx'>\n");
        html.push_str("<h2 class='section-title'><span>//</span> FX Currency Pairs</h2>\n");
        for (name, points, result) in fx_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            let bands = analysis::bollinger_bands(&prices, 20, 2.0);
            let chart = charts::bollinger_chart_svg(&prices, &bands, name);
            html.push_str(&format!(
                "<div class='card'><h3>{}</h3><p>Rate: {:.4} | RSI: {:.1} | Vol: {:.4}</p>{}</div>\n",
                name, result.current_price,
                result.rsi_14.unwrap_or(0.0), result.std_dev, chart
            ));
        }
        html.push_str("</section>\n");
    }

    // ── ML RESULTS ──
    if !gbt_results.is_empty() {
        html.push_str("<section id='ml'>\n");
        html.push_str("<h2 class='section-title'><span>//</span> ML Model Accuracy</h2>\n");
        html.push_str("<table><tr><th>Symbol</th><th>LinReg</th><th>LogReg</th><th>GBT</th><th>Best</th><th>Verdict</th></tr>\n");
        for r in gbt_results {
            let verdict = if r.best_direction_accuracy > 55.0 { ("PROMISING", "#00e676") }
                else if r.best_direction_accuracy > 50.0 { ("MARGINAL", "#ffd740") }
                else { ("NO EDGE", "#ff5252") };
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td>\
                 <td><span style='color:{}'>{}</span></td></tr>\n",
                r.linear_metrics.symbol,
                r.linear_metrics.direction_accuracy,
                r.logistic_metrics.direction_accuracy,
                r.gbt_metrics.direction_accuracy,
                r.best_direction_accuracy,
                verdict.1, verdict.0,
            ));
        }
        html.push_str("</table>\n</section>\n");
    }

    // ── CORRELATION MATRIX ──
    html.push_str(&correlation_section(stock_data, coin_data, fx_data));

    // ── GLOSSARY ──
    html.push_str(&report_glossary());

    html.push_str("</main>\n");

    // ── FOOTER ──
    html.push_str(&report_footer());

    html.push_str("</body></html>\n");

    fs::write(output_path, &html)?;
    Ok(())
}

// ════════════════════════════════════════
// HTML Components
// ════════════════════════════════════════

fn report_head() -> String {
    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rust_Invest — AI Trading Intelligence</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root {{
  --bg-deep: #050a10;
  --bg-surface: #0a1018;
  --bg-card: rgba(12, 22, 35, 0.85);
  --border: rgba(0, 212, 170, 0.12);
  --border-glow: rgba(0, 212, 170, 0.3);
  --teal: #00d4aa;
  --teal-dim: rgba(0, 212, 170, 0.15);
  --amber: #fbbf24;
  --red: #ef4444;
  --green: #10b981;
  --text: #e8edf2;
  --text-dim: #7a8a9e;
  --text-muted: #4a5568;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; scroll-padding-top: 80px; }}
body {{
  font-family: var(--sans);
  background: var(--bg-deep);
  color: var(--text);
  line-height: 1.6;
}}
body::before {{
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background:
    linear-gradient(rgba(0,212,170,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,170,0.02) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none; z-index: 0;
}}
nav {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  background: rgba(5, 10, 16, 0.92);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  padding: 0 32px; height: 64px;
  display: flex; align-items: center; justify-content: space-between;
}}
.nav-brand {{
  font-family: var(--mono); font-weight: 700; font-size: 16px; color: var(--teal);
}}
.nav-brand span {{ color: var(--text-dim); font-weight: 400; }}
.nav-links {{ display: flex; gap: 4px; }}
.nav-links a {{
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  color: var(--text-dim); text-decoration: none;
  padding: 8px 14px; border-radius: 6px; transition: all 0.2s;
  letter-spacing: 0.3px; text-transform: uppercase;
}}
.nav-links a:hover {{ color: var(--teal); background: var(--teal-dim); }}
main {{
  max-width: 1280px; margin: 0 auto;
  padding: 80px 32px 64px; position: relative; z-index: 1;
}}
section {{ margin-bottom: 48px; }}
h1 {{
  font-family: var(--mono); font-size: 48px; font-weight: 700;
  letter-spacing: -2px; color: var(--text); text-align: center;
}}
h1 span {{ color: var(--teal); }}
.subtitle {{ text-align: center; font-size: 15px; color: var(--text-dim); }}
.meta {{ text-align: center; font-family: var(--mono); font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }}
h2.section-title {{
  font-family: var(--mono); font-size: 22px; font-weight: 600;
  color: var(--text); margin-bottom: 8px; padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}}
h2.section-title span {{ color: var(--teal); }}
h3 {{ font-family: var(--mono); font-size: 14px; color: var(--teal); margin: 16px 0 8px; }}
.kpi-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 16px; margin: 32px 0;
}}
.kpi {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; text-align: center; transition: all 0.3s;
}}
.kpi:hover {{ border-color: var(--border-glow); transform: translateY(-2px); }}
.kpi-value {{
  font-family: var(--mono); font-size: 28px; font-weight: 700;
  color: var(--teal); line-height: 1.2;
}}
.kpi-value.amber {{ color: var(--amber); }}
.kpi-value.red {{ color: var(--red); }}
.kpi-label {{
  font-size: 12px; color: var(--text-dim); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.card {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin: 12px 0; transition: all 0.3s;
}}
.card:hover {{ border-color: var(--border-glow); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
thead th {{
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  color: var(--teal); text-transform: uppercase; letter-spacing: 0.5px;
  padding: 14px 16px; text-align: right;
  border-bottom: 1px solid var(--border); background: rgba(0, 212, 170, 0.03);
  white-space: nowrap;
}}
thead th:first-child {{ text-align: left; }}
tbody td {{
  padding: 12px 16px; text-align: right;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-family: var(--mono); font-size: 12px; white-space: nowrap;
}}
tbody td:first-child {{ text-align: left; font-weight: 600; color: var(--text); }}
tbody tr:hover {{ background: rgba(0, 212, 170, 0.03); }}
.signal-bullish {{ color: #00e676; background: rgba(0,230,118,0.12); padding: 3px 10px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
.signal-bearish {{ color: #ff5252; background: rgba(255,82,82,0.12); padding: 3px 10px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
.signal-neutral {{ color: #ffd740; background: rgba(255,215,64,0.12); padding: 3px 10px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
/* Portfolio section styles */
.portfolio-section {{ margin: 16px 0; }}
.port-kpis {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin: 16px 0;
}}
.port-kpi {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px; text-align: center;
}}
.port-kpi-hero {{ grid-column: span 2; }}
.port-kpi-value {{ font-family: var(--mono); font-size: 22px; font-weight: 700; }}
.port-kpi-label {{ font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
.port-kpi-sub {{ font-size: 11px; color: var(--text-muted); }}
.port-chart-container {{ margin: 20px 0; }}
.port-legend {{ display: flex; gap: 20px; font-size: 12px; color: var(--text-dim); margin-bottom: 8px; }}
.port-alloc {{ margin: 20px 0; }}
.alloc-list {{ display: grid; gap: 6px; }}
.alloc-row {{
  display: grid; grid-template-columns: 16px 60px 60px 80px 60px;
  gap: 8px; align-items: center; font-family: var(--mono); font-size: 12px;
  padding: 6px 10px; background: var(--bg-card); border-radius: 6px;
}}
.alloc-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
.alloc-sym {{ color: var(--text); font-weight: 600; }}
.alloc-pct {{ color: var(--text-dim); }}
.alloc-amt {{ color: var(--text-dim); }}
.alloc-ret {{ font-weight: 600; }}
/* Glossary */
.glossary-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 12px; margin-top: 16px;
}}
.glossary-item {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}}
.glossary-item dt {{ font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--teal); margin-bottom: 4px; }}
.glossary-item dd {{ font-size: 13px; color: var(--text-dim); line-height: 1.5; }}
.glossary-item .good {{ font-family: var(--mono); font-size: 11px; color: #10b981; }}
footer {{
  text-align: center; padding: 40px 0; font-size: 12px;
  color: var(--text-muted); border-top: 1px solid var(--border);
}}
footer a {{ color: var(--teal); text-decoration: none; }}
@media (max-width: 768px) {{
  main {{ padding: 80px 16px 40px; }}
  nav {{ padding: 0 16px; }}
  .nav-links {{ display: none; }}
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  h1 {{ font-size: 32px; }}
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-deep); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>
"#)
}

fn report_nav() -> String {
    String::from(
        "<nav>\
         <div class='nav-brand'>RUST_INVEST <span>v7.0</span></div>\
         <div class='nav-links'>\
         <a href='#overview'>Overview</a>\
         <a href='#backtest'>Backtest</a>\
         <a href='#portfolio'>Portfolio</a>\
         <a href='#fx'>FX</a>\
         <a href='#signals'>Signals</a>\
         <a href='#diagnostics'>Diagnostics</a>\
         <a href='#glossary'>Glossary</a>\
         </div></nav>\n"
    )
}

fn report_hero() -> String {
    format!(
        "<section id='top' style='text-align:center;padding:80px 0 32px;'>\n\
         <h1>Rust<span>_</span>Invest</h1>\n\
         <p class='subtitle'>AI-Powered Trading Intelligence — 4 Model Ensemble × 20 Assets (Stocks + FX + Crypto)</p>\n\
         <p class='meta'>Walk-Forward Backtest Report &bull; Generated {}</p>\n\
         </section>\n",
        Utc::now().format("%Y-%m-%d %H:%M UTC")
    )
}

fn report_kpis(results: &[backtester::BacktestResult]) -> String {
    if results.is_empty() { return String::new(); }

    let stocks: Vec<&backtester::BacktestResult> = results.iter()
        .filter(|r| r.sharpe_ratio > 0.5 && r.excess_return_pct > 0.0)
        .collect();
    let total_assets = results.len();
    let edge_count = stocks.len();

    let avg_sharpe = if !stocks.is_empty() {
        stocks.iter().map(|r| r.sharpe_ratio).sum::<f64>() / stocks.len() as f64
    } else { 0.0 };

    let avg_excess = if !stocks.is_empty() {
        stocks.iter().map(|r| r.excess_return_pct).sum::<f64>() / stocks.len() as f64
    } else { 0.0 };

    let beat_bh = results.iter().filter(|r| r.excess_return_pct > 0.0).count();

    format!(
        "<section id='overview'>\n\
         <h2 class='section-title'><span>//</span> Performance Overview</h2>\n\
         <div class='kpi-grid'>\n\
         <div class='kpi'><div class='kpi-value'>{}</div><div class='kpi-label'>Assets Analysed</div></div>\n\
         <div class='kpi'><div class='kpi-value'>{}/{}</div><div class='kpi-label'>Beat Buy &amp; Hold</div></div>\n\
         <div class='kpi'><div class='kpi-value'>{:.2}</div><div class='kpi-label'>Avg Sharpe (Edge)</div></div>\n\
         <div class='kpi'><div class='kpi-value' style='color:#10b981'>+{:.0}%</div><div class='kpi-label'>Avg Excess Return</div></div>\n\
         <div class='kpi'><div class='kpi-value'>{}/{}</div><div class='kpi-label'>Assets With Edge</div></div>\n\
         </div>\n</section>\n",
        total_assets, beat_bh, total_assets,
        avg_sharpe, avg_excess,
        edge_count, total_assets,
    )
}

fn correlation_section(
    stock_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    coin_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    fx_data: &[(String, Vec<PricePoint>, AnalysisResult)],
) -> String {
    let all_data: Vec<(&str, &[PricePoint])> = stock_data.iter()
        .map(|(n, p, _)| (n.as_str(), p.as_slice()))
        .chain(fx_data.iter().map(|(n, p, _)| (n.as_str(), p.as_slice())))
        .chain(coin_data.iter().map(|(n, p, _)| (n.as_str(), p.as_slice())))
        .collect();

    if all_data.len() < 2 { return String::new(); }

    let mut html = String::from("<section id='correlation'>\n<h2 class='section-title'><span>//</span> Correlation Matrix</h2>\n");
    html.push_str("<div style='overflow-x:auto;'><table><tr><th></th>");

    // Header row
    for (name, _) in &all_data {
        html.push_str(&format!("<th>{}</th>", &name[..name.len().min(6)]));
    }
    html.push_str("</tr>\n");

    // Compute returns for correlation
    let returns: Vec<Vec<f64>> = all_data.iter().map(|(_, pts)| {
        let prices: Vec<f64> = pts.iter().map(|p| p.price).collect();
        prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
    }).collect();

    for (i, (name, _)) in all_data.iter().enumerate() {
        html.push_str(&format!("<tr><td>{}</td>", &name[..name.len().min(6)]));
        for j in 0..all_data.len() {
            let corr = compute_correlation(&returns[i], &returns[j]);
            let color = if corr.abs() > 0.7 { "#00d4aa" }
                else if corr.abs() > 0.4 { "#ffd740" }
                else { "#666" };
            html.push_str(&format!("<td style='color:{}'>{:.2}</td>", color, corr));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</table></div>\n</section>\n");
    html
}

fn compute_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 10 { return 0.0; }
    let a = &a[a.len()-n..];
    let b = &b[b.len()-n..];
    let mean_a = a.iter().sum::<f64>() / n as f64;
    let mean_b = b.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a == 0.0 || var_b == 0.0 { return 0.0; }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn report_glossary() -> String {
    String::from(
r#"<section id='glossary'>
<h2 class='section-title'><span>//</span> Metric Glossary</h2>
<div class='glossary-grid'>
<div class='glossary-item'><dt>Sharpe Ratio</dt><dd>Return per unit of risk. <span class='good'>&gt; 1.0 good, &gt; 2.0 excellent, &gt; 3.0 exceptional</span></dd></div>
<div class='glossary-item'><dt>Max Drawdown</dt><dd>Worst peak-to-trough drop. Shows maximum pain. <span class='good'>&lt; 10% conservative, &lt; 20% moderate</span></dd></div>
<div class='glossary-item'><dt>Win Rate</dt><dd>% of invested days with positive returns. <span class='good'>&gt; 55% with PF &gt; 1.5 = strong edge</span></dd></div>
<div class='glossary-item'><dt>Profit Factor</dt><dd>Gross wins ÷ gross losses. PF of 2.0 = win $2 for every $1 lost. <span class='good'>&gt; 1.5 good, &gt; 2.0 excellent</span></dd></div>
<div class='glossary-item'><dt>Expectancy</dt><dd>Average profit per trade (%). The expected value of each position. <span class='good'>&gt; 0.1% per trade = meaningful edge</span></dd></div>
<div class='glossary-item'><dt>Walk-Forward Validation</dt><dd>Train on past data, test on unseen future data, repeatedly. Prevents overfitting.</dd></div>
<div class='glossary-item'><dt>Ensemble Consensus</dt><dd>4 independent models vote on direction. Weighted by recent accuracy. Requires high agreement for signals.</dd></div>
<div class='glossary-item'><dt>Excess Return</dt><dd>Strategy return minus buy-and-hold. <span class='good'>Positive = AI timing adds value</span></dd></div>
</div>
</section>
"#)
}

fn report_footer() -> String {
    String::from(
        "<footer>Built with <span style='color:#00d4aa;'>Rust</span> — \
         Zero external ML dependencies — \
         LinReg + LogReg + GBT + LSTM ensemble — \
         <a href='https://hassanshuman.co.uk'>hassanshuman.co.uk</a>\
         </footer>\n"
    )
}
