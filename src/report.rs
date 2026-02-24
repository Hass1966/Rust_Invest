use std::fs;
use crate::analysis::{self, PricePoint, AnalysisResult};
use crate::charts;
use crate::ml;

pub fn generate_html_report(
    coin_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    stock_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    ml_results: &[ml::PipelineResult],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut html = String::new();

    html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rust Invest - Market Analysis Report</title>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0f1923; color: #e0e0e0;
        max-width: 1200px; margin: 0 auto; padding: 20px;
    }
    h1 { color: #00d4aa; text-align: center; }
    h2 { color: #00d4aa; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; }
    h3 { color: #4fc3f7; }
    .timestamp { text-align: center; color: #888; margin-bottom: 30px; }
    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th {
        background: #1a2332; color: #00d4aa; padding: 10px;
        text-align: right; border-bottom: 2px solid #1e3a5f;
    }
    th:first-child { text-align: left; }
    td { padding: 8px 10px; border-bottom: 1px solid #1e3a5f; text-align: right; }
    td:first-child { text-align: left; font-weight: bold; }
    tr:hover { background: #1a2332; }
    .positive { color: #00e676; }
    .negative { color: #ff5252; }
    .neutral { color: #ffd740; }
    .card {
        background: #1a2332; border-radius: 8px; padding: 20px;
        margin: 20px 0; border: 1px solid #1e3a5f;
    }
    .signal-bullish {
        background: #1b3329; color: #00e676;
        padding: 4px 12px; border-radius: 4px; font-weight: bold;
    }
    .signal-bearish {
        background: #3d1f1f; color: #ff5252;
        padding: 4px 12px; border-radius: 4px; font-weight: bold;
    }
    .signal-neutral {
        background: #3d3520; color: #ffd740;
        padding: 4px 12px; border-radius: 4px; font-weight: bold;
    }
    .metric-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px; margin: 15px 0;
    }
    .metric-box {
        background: #0f1923; padding: 15px; border-radius: 6px; text-align: center;
    }
    .metric-box .value { font-size: 24px; font-weight: bold; color: #4fc3f7; }
    .metric-box .label { font-size: 12px; color: #888; margin-top: 5px; }
    .correlation-table td { font-size: 14px; padding: 6px 10px; }
    .weight-bar {
        display: inline-block; height: 12px; border-radius: 2px; vertical-align: middle;
    }
    .weight-positive { background: #00e676; }
    .weight-negative { background: #ff5252; }
    .model-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 768px) { .model-compare { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>RUST INVEST — Market Analysis Report</h1>
"#);

    html.push_str(&format!(
        "<p class='timestamp'>Generated: {}</p>\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    ));

    // ── Crypto Overview table ──
    html.push_str("<h2>Crypto Portfolio Overview</h2>\n<table>\n");
    html.push_str(
        "<tr><th>Coin</th><th>Price</th><th>Mean (365d)</th>\
         <th>Min</th><th>Max</th><th>Volatility</th>\
         <th>Avg Daily Return</th><th>RSI</th><th>Trend</th></tr>\n"
    );
    for (coin_id, points, result) in coin_data {
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        let sma_7 = analysis::sma(&prices, 7);
        let sma_30 = analysis::sma(&prices, 30);
        let trend = match (sma_7.last(), sma_30.last()) {
            (Some(s), Some(l)) if s > l => "<span class='signal-bullish'>BULLISH</span>",
            (Some(_), Some(_)) => "<span class='signal-bearish'>BEARISH</span>",
            _ => "<span class='signal-neutral'>N/A</span>",
        };
        let rc = if result.daily_returns_mean >= 0.0 { "positive" } else { "negative" };
        let rv = result.rsi_14.unwrap_or(0.0);
        let rclass = if rv > 70.0 { "negative" } else if rv < 30.0 { "positive" } else { "neutral" };
        html.push_str(&format!(
            "<tr><td>{}</td><td>${:.2}</td><td>${:.2}</td><td>${:.2}</td>\
             <td>${:.2}</td><td>${:.2}</td><td class='{}'>{:.4}%</td>\
             <td class='{}'>{:.1}</td><td>{}</td></tr>\n",
            coin_id, result.current_price, result.mean_price,
            result.min_price, result.max_price, result.std_dev,
            rc, result.daily_returns_mean, rclass, rv, trend
        ));
    }
    html.push_str("</table>\n");

    // ── Individual crypto cards ──
    for (coin_id, points, result) in coin_data {
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
        html.push_str(&format!("<div class='card'>\n<h3>{}</h3>\n", coin_id.to_uppercase()));
        html.push_str("<div class='metric-grid'>\n");
        for (label, value) in &[
            ("Current Price", format!("${:.2}", result.current_price)),
            ("365d Mean", format!("${:.2}", result.mean_price)),
            ("Std Deviation", format!("${:.2}", result.std_dev)),
            ("RSI (14)", format!("{:.1}", result.rsi_14.unwrap_or(0.0))),
            ("SMA 7-day", format!("${:.2}", result.sma_7.unwrap_or(0.0))),
            ("SMA 30-day", format!("${:.2}", result.sma_30.unwrap_or(0.0))),
        ] {
            html.push_str(&format!(
                "<div class='metric-box'><div class='value'>{}</div>\
                 <div class='label'>{}</div></div>\n", value, label));
        }
        html.push_str("</div>\n");

        let sma7 = analysis::sma(&prices, 7);
        let sma30 = analysis::sma(&prices, 30);
        html.push_str(&charts::price_chart_svg(&prices, &sma7, &sma30,
                                               &format!("{} — Price with Moving Averages", coin_id.to_uppercase())));
        let bc = analysis::bollinger_bands(&prices, 20, 2.0);
        html.push_str(&charts::bollinger_chart_svg(&prices, &bc,
                                                   &format!("{} — Bollinger Bands (20, 2σ)", coin_id.to_uppercase())));

        let (ml, sl, hist) = analysis::macd(&prices);
        if let (Some(&mv), Some(&sv), Some(&hv)) = (ml.last(), sl.last(), hist.last()) {
            let sig = if hv > 0.0 { "<span class='positive'>BULLISH (MACD above signal)</span>" }
            else { "<span class='negative'>BEARISH (MACD below signal)</span>" };
            html.push_str(&format!(
                "<p><strong>MACD:</strong> {:.4} | <strong>Signal:</strong> {:.4} | \
                 <strong>Histogram:</strong> {:.4} — {}</p>\n", mv, sv, hv, sig));
        }

        let bands = analysis::bollinger_bands(&prices, 20, 2.0);
        if let Some(&(u, m, l)) = bands.last() {
            let bp = if result.current_price > u { "<span class='negative'>ABOVE upper band — overbought</span>" }
            else if result.current_price < l { "<span class='positive'>BELOW lower band — oversold</span>" }
            else { "<span class='neutral'>Within bands — normal range</span>" };
            html.push_str(&format!(
                "<p><strong>Bollinger Bands (20,2):</strong> Upper ${:.2} | Middle ${:.2} | Lower ${:.2} — {}</p>\n",
                u, m, l, bp));
        }
        html.push_str("</div>\n");
    }

    // ── Stock analysis ──
    if !stock_data.is_empty() {
        html.push_str("<h2>Stock Analysis</h2>\n<table>\n");
        html.push_str(
            "<tr><th>Symbol</th><th>Price</th><th>Mean (1yr)</th>\
             <th>Min</th><th>Max</th><th>Volatility</th>\
             <th>Avg Daily Return</th><th>RSI</th><th>Trend</th></tr>\n");
        for (symbol, points, result) in stock_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            let sma7 = analysis::sma(&prices, 7);
            let sma30 = analysis::sma(&prices, 30);
            let trend = match (sma7.last(), sma30.last()) {
                (Some(s), Some(l)) if s > l => "<span class='signal-bullish'>BULLISH</span>",
                (Some(_), Some(_)) => "<span class='signal-bearish'>BEARISH</span>",
                _ => "<span class='signal-neutral'>N/A</span>",
            };
            let rc = if result.daily_returns_mean >= 0.0 { "positive" } else { "negative" };
            let rv = result.rsi_14.unwrap_or(0.0);
            let rclass = if rv > 70.0 { "negative" } else if rv < 30.0 { "positive" } else { "neutral" };
            html.push_str(&format!(
                "<tr><td>{}</td><td>${:.2}</td><td>${:.2}</td><td>${:.2}</td>\
                 <td>${:.2}</td><td>${:.2}</td><td class='{}'>{:.4}%</td>\
                 <td class='{}'>{:.1}</td><td>{}</td></tr>\n",
                symbol, result.current_price, result.mean_price,
                result.min_price, result.max_price, result.std_dev,
                rc, result.daily_returns_mean, rclass, rv, trend));
        }
        html.push_str("</table>\n");

        for (symbol, points, result) in stock_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            html.push_str(&format!("<div class='card'>\n<h3>{}</h3>\n", symbol));
            html.push_str("<div class='metric-grid'>\n");
            for (label, value) in &[
                ("Current Price", format!("${:.2}", result.current_price)),
                ("1yr Mean", format!("${:.2}", result.mean_price)),
                ("Std Deviation", format!("${:.2}", result.std_dev)),
                ("RSI (14)", format!("{:.1}", result.rsi_14.unwrap_or(0.0))),
                ("SMA 7-day", format!("${:.2}", result.sma_7.unwrap_or(0.0))),
                ("SMA 30-day", format!("${:.2}", result.sma_30.unwrap_or(0.0))),
            ] {
                html.push_str(&format!(
                    "<div class='metric-box'><div class='value'>{}</div>\
                     <div class='label'>{}</div></div>\n", value, label));
            }
            html.push_str("</div>\n");

            let sma7 = analysis::sma(&prices, 7);
            let sma30 = analysis::sma(&prices, 30);
            html.push_str(&charts::price_chart_svg(&prices, &sma7, &sma30,
                                                   &format!("{} — Price with Moving Averages", symbol)));
            let bc = analysis::bollinger_bands(&prices, 20, 2.0);
            html.push_str(&charts::bollinger_chart_svg(&prices, &bc,
                                                       &format!("{} — Bollinger Bands (20, 2σ)", symbol)));

            let (ml_line, sl, hist) = analysis::macd(&prices);
            if let (Some(&mv), Some(&sv), Some(&hv)) = (ml_line.last(), sl.last(), hist.last()) {
                let sig = if hv > 0.0 { "<span class='positive'>BULLISH (MACD above signal)</span>" }
                else { "<span class='negative'>BEARISH (MACD below signal)</span>" };
                html.push_str(&format!(
                    "<p><strong>MACD:</strong> {:.4} | <strong>Signal:</strong> {:.4} | \
                     <strong>Histogram:</strong> {:.4} — {}</p>\n", mv, sv, hv, sig));
            }

            let bands = analysis::bollinger_bands(&prices, 20, 2.0);
            if let Some(&(u, m, l)) = bands.last() {
                let bp = if result.current_price > u { "<span class='negative'>ABOVE upper band — overbought</span>" }
                else if result.current_price < l { "<span class='positive'>BELOW lower band — oversold</span>" }
                else { "<span class='neutral'>Within bands — normal range</span>" };
                html.push_str(&format!(
                    "<p><strong>Bollinger Bands (20,2):</strong> Upper ${:.2} | Middle ${:.2} | Lower ${:.2} — {}</p>\n",
                    u, m, l, bp));
            }
            html.push_str("</div>\n");
        }
    }

    // ── ML Results ──
    if !ml_results.is_empty() {
        html.push_str("<h2>Machine Learning Results</h2>\n");
        html.push_str(&format!(
            "<p>Two models trained per asset using {} features. \
             80/20 chronological split. Features normalised to zero mean, unit variance.</p>\n",
            ml::FEATURE_NAMES.len()));

        // Summary table
        html.push_str("<table>\n");
        html.push_str(
            "<tr><th>Symbol</th><th>Linear Reg %</th><th>Logistic Reg %</th>\
             <th>Best Model</th><th>Best %</th><th>Verdict</th></tr>\n");

        for r in ml_results {
            let (badge, text) = if r.best_direction_accuracy > 55.0 { ("signal-bullish", "PROMISING") }
            else if r.best_direction_accuracy > 50.0 { ("signal-neutral", "MARGINAL") }
            else { ("signal-bearish", "NO EDGE") };

            let lin_class = if r.linear_metrics.direction_accuracy > 55.0 { "positive" }
            else if r.linear_metrics.direction_accuracy > 50.0 { "neutral" }
            else { "negative" };
            let log_class = if r.logistic_metrics.direction_accuracy > 55.0 { "positive" }
            else if r.logistic_metrics.direction_accuracy > 50.0 { "neutral" }
            else { "negative" };
            let best_class = if r.best_direction_accuracy > 55.0 { "positive" }
            else if r.best_direction_accuracy > 50.0 { "neutral" }
            else { "negative" };

            html.push_str(&format!(
                "<tr><td>{}</td><td class='{}'>{:.1}%</td><td class='{}'>{:.1}%</td>\
                 <td>{}</td><td class='{}'>{:.1}%</td>\
                 <td><span class='{}'>{}</span></td></tr>\n",
                r.linear_metrics.symbol,
                lin_class, r.linear_metrics.direction_accuracy,
                log_class, r.logistic_metrics.direction_accuracy,
                r.best_model_name,
                best_class, r.best_direction_accuracy,
                badge, text
            ));
        }
        html.push_str("</table>\n");

        // Feature importance for promising models
        html.push_str("<h3>Feature Importance — Models Above 50%</h3>\n");

        for r in ml_results {
            if r.best_direction_accuracy < 50.0 { continue; }

            html.push_str(&format!(
                "<div class='card'>\n<h3>{} — {} ({:.1}%)</h3>\n<div class='model-compare'>\n",
                r.linear_metrics.symbol, r.best_model_name, r.best_direction_accuracy));

            // Linear weights
            html.push_str("<div>\n<h3 style='font-size:14px;'>Linear Regression</h3>\n");
            render_weights(&mut html, &r.linear_weights);
            html.push_str("</div>\n");

            // Logistic weights
            html.push_str("<div>\n<h3 style='font-size:14px;'>Logistic Regression</h3>\n");
            render_weights(&mut html, &r.logistic_weights);
            html.push_str("</div>\n");

            html.push_str("</div>\n</div>\n");
        }

        html.push_str(&format!(
            "<p><em>Features: {}. Green bars = predicts UP, Red = predicts DOWN. \
             Bar length shows relative importance.</em></p>\n",
            ml::FEATURE_NAMES.join(", ")));
    }

    // ── Correlation matrix ──
    html.push_str("<h2>Correlation Matrix (Daily Returns)</h2>\n");
    html.push_str("<table class='correlation-table'>\n<tr><th></th>");
    let coin_ids: Vec<&String> = coin_data.iter().map(|(id, _, _)| id).collect();
    for id in &coin_ids { html.push_str(&format!("<th>{}</th>", id)); }
    html.push_str("</tr>\n");

    let all_returns: Vec<Vec<f64>> = coin_data.iter()
        .map(|(_, points, _)| {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            analysis::daily_returns(&prices)
        }).collect();

    for (i, id) in coin_ids.iter().enumerate() {
        html.push_str(&format!("<tr><td>{}</td>", id));
        for j in 0..coin_ids.len() {
            let corr = analysis::correlation(&all_returns[i], &all_returns[j]);
            let class = if i == j { "" }
            else if corr > 0.7 { "negative" }
            else if corr < 0.3 { "positive" }
            else { "neutral" };
            html.push_str(&format!("<td class='{}'>{:.3}</td>", class, corr));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</table>\n");
    html.push_str("<p><em>High correlation (&gt;0.7) = assets move together. \
        Low correlation (&lt;0.3) = independent movement.</em></p>\n");

    // ── Footer ──
    html.push_str(
        "<hr style='border-color: #1e3a5f; margin-top: 40px;'>\n\
         <p style='text-align: center; color: #555;'>\
         Rust Invest — Built in Rust | Data: CoinGecko &amp; Yahoo Finance</p>\n\
         </body></html>");

    fs::write(output_path, &html)?;
    println!("  ✓ Report saved to: {}", output_path);
    Ok(())
}

fn render_weights(html: &mut String, weights: &[(String, f64)]) {
    let max_w = weights.iter().map(|(_, w)| w.abs()).reduce(f64::max).unwrap_or(1.0);
    for (name, weight) in weights {
        let bar_width = (weight.abs() / max_w * 150.0) as u32;
        let (color, sign) = if *weight >= 0.0 { ("weight-positive", "+") }
        else { ("weight-negative", "-") };
        html.push_str(&format!(
            "<p style='margin:3px 0;font-size:13px;'>\
             <span style='display:inline-block;width:100px;'>{}</span>\
             <span style='display:inline-block;width:65px;text-align:right;\
                    font-family:monospace;font-size:12px;'>{}{:.4}</span> \
             <span class='weight-bar {}' style='width:{}px;'></span></p>\n",
            name, sign, weight.abs(), color, bar_width));
    }
}