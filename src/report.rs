use std::fs;
use crate::analysis::{self, PricePoint, AnalysisResult};
use crate::charts;

pub fn generate_html_report(
    coin_data: &[(String, Vec<PricePoint>, AnalysisResult)],
    stock_data: &[(String, Vec<PricePoint>, AnalysisResult)],
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
        background: #0f1923;
        color: #e0e0e0;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 { color: #00d4aa; text-align: center; }
    h2 { color: #00d4aa; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; }
    h3 { color: #4fc3f7; }
    .timestamp { text-align: center; color: #888; margin-bottom: 30px; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th {
        background: #1a2332;
        color: #00d4aa;
        padding: 10px;
        text-align: right;
        border-bottom: 2px solid #1e3a5f;
    }
    th:first-child { text-align: left; }
    td {
        padding: 8px 10px;
        border-bottom: 1px solid #1e3a5f;
        text-align: right;
    }
    td:first-child { text-align: left; font-weight: bold; }
    tr:hover { background: #1a2332; }
    .positive { color: #00e676; }
    .negative { color: #ff5252; }
    .neutral { color: #ffd740; }
    .card {
        background: #1a2332;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #1e3a5f;
    }
    .signal-bullish {
        background: #1b3329;
        color: #00e676;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-bearish {
        background: #3d1f1f;
        color: #ff5252;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-neutral {
        background: #3d3520;
        color: #ffd740;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    .metric-box {
        background: #0f1923;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
    }
    .metric-box .value {
        font-size: 24px;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-box .label {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
    .correlation-table td {
        font-size: 14px;
        padding: 6px 10px;
    }
</style>
</head>
<body>
<h1>RUST INVEST — Market Analysis Report</h1>
"#);

    html.push_str(&format!(
        "<p class='timestamp'>Generated: {}</p>\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    ));

    // ── Overview table ──
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
            (Some(short), Some(long)) if short > long => {
                "<span class='signal-bullish'>BULLISH</span>"
            }
            (Some(_), Some(_)) => {
                "<span class='signal-bearish'>BEARISH</span>"
            }
            _ => "<span class='signal-neutral'>N/A</span>",
        };

        let return_class = if result.daily_returns_mean >= 0.0 { "positive" } else { "negative" };
        let rsi_val = result.rsi_14.unwrap_or(0.0);
        let rsi_class = if rsi_val > 70.0 { "negative" } else if rsi_val < 30.0 { "positive" } else { "neutral" };

        html.push_str(&format!(
            "<tr><td>{}</td><td>${:.2}</td><td>${:.2}</td><td>${:.2}</td>\
             <td>${:.2}</td><td>${:.2}</td><td class='{}'>{:.4}%</td>\
             <td class='{}'>{:.1}</td><td>{}</td></tr>\n",
            coin_id, result.current_price, result.mean_price,
            result.min_price, result.max_price, result.std_dev,
            return_class, result.daily_returns_mean,
            rsi_class, rsi_val, trend
        ));
    }
    html.push_str("</table>\n");

    // ── Individual crypto cards ──
    for (coin_id, points, result) in coin_data {
        let prices: Vec<f64> = points.iter().map(|p| p.price).collect();

        html.push_str(&format!("<div class='card'>\n<h3>{}</h3>\n", coin_id.to_uppercase()));

        // Metric boxes
        html.push_str("<div class='metric-grid'>\n");
        let metrics = vec![
            ("Current Price", format!("${:.2}", result.current_price)),
            ("365d Mean", format!("${:.2}", result.mean_price)),
            ("Std Deviation", format!("${:.2}", result.std_dev)),
            ("RSI (14)", format!("{:.1}", result.rsi_14.unwrap_or(0.0))),
            ("SMA 7-day", format!("${:.2}", result.sma_7.unwrap_or(0.0))),
            ("SMA 30-day", format!("${:.2}", result.sma_30.unwrap_or(0.0))),
        ];
        for (label, value) in &metrics {
            html.push_str(&format!(
                "<div class='metric-box'><div class='value'>{}</div>\
                 <div class='label'>{}</div></div>\n",
                value, label
            ));
        }
        html.push_str("</div>\n");

        // Price chart with moving averages
        let sma_7 = analysis::sma(&prices, 7);
        let sma_30 = analysis::sma(&prices, 30);
        html.push_str(&charts::price_chart_svg(
            &prices, &sma_7, &sma_30,
            &format!("{} — Price with Moving Averages", coin_id.to_uppercase())
        ));

        // Bollinger Bands chart
        let bands_chart = analysis::bollinger_bands(&prices, 20, 2.0);
        html.push_str(&charts::bollinger_chart_svg(
            &prices, &bands_chart,
            &format!("{} — Bollinger Bands (20, 2σ)", coin_id.to_uppercase())
        ));

        // MACD
        let (macd_line, signal_line, histogram) = analysis::macd(&prices);
        if let (Some(&macd_val), Some(&signal_val), Some(&hist_val)) =
            (macd_line.last(), signal_line.last(), histogram.last())
        {
            let macd_signal = if hist_val > 0.0 {
                "<span class='positive'>BULLISH (MACD above signal)</span>"
            } else {
                "<span class='negative'>BEARISH (MACD below signal)</span>"
            };
            html.push_str(&format!(
                "<p><strong>MACD:</strong> {:.4} | <strong>Signal:</strong> {:.4} | \
                 <strong>Histogram:</strong> {:.4} — {}</p>\n",
                macd_val, signal_val, hist_val, macd_signal
            ));
        }

        // Bollinger Bands text
        let bands = analysis::bollinger_bands(&prices, 20, 2.0);
        if let Some(&(upper, middle, lower)) = bands.last() {
            let bb_position = if result.current_price > upper {
                "<span class='negative'>ABOVE upper band — overbought</span>"
            } else if result.current_price < lower {
                "<span class='positive'>BELOW lower band — oversold</span>"
            } else {
                "<span class='neutral'>Within bands — normal range</span>"
            };
            html.push_str(&format!(
                "<p><strong>Bollinger Bands (20,2):</strong> \
                 Upper ${:.2} | Middle ${:.2} | Lower ${:.2} — {}</p>\n",
                upper, middle, lower, bb_position
            ));
        }

        html.push_str("</div>\n");
    }

    // ── Stock analysis ──
    if !stock_data.is_empty() {
        html.push_str("<h2>Stock Analysis</h2>\n<table>\n");
        html.push_str(
            "<tr><th>Symbol</th><th>Price</th><th>Mean (1yr)</th>\
             <th>Min</th><th>Max</th><th>Volatility</th>\
             <th>Avg Daily Return</th><th>RSI</th><th>Trend</th></tr>\n"
        );

        for (symbol, points, result) in stock_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);

            let trend = match (sma_7.last(), sma_30.last()) {
                (Some(short), Some(long)) if short > long => "<span class='signal-bullish'>BULLISH</span>",
                (Some(_), Some(_)) => "<span class='signal-bearish'>BEARISH</span>",
                _ => "<span class='signal-neutral'>N/A</span>",
            };

            let return_class = if result.daily_returns_mean >= 0.0 { "positive" } else { "negative" };
            let rsi_val = result.rsi_14.unwrap_or(0.0);
            let rsi_class = if rsi_val > 70.0 { "negative" } else if rsi_val < 30.0 { "positive" } else { "neutral" };

            html.push_str(&format!(
                "<tr><td>{}</td><td>${:.2}</td><td>${:.2}</td><td>${:.2}</td>\
                 <td>${:.2}</td><td>${:.2}</td><td class='{}'>{:.4}%</td>\
                 <td class='{}'>{:.1}</td><td>{}</td></tr>\n",
                symbol, result.current_price, result.mean_price,
                result.min_price, result.max_price, result.std_dev,
                return_class, result.daily_returns_mean,
                rsi_class, rsi_val, trend
            ));
        }
        html.push_str("</table>\n");

        // Individual stock cards
        for (symbol, points, result) in stock_data {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();

            html.push_str(&format!("<div class='card'>\n<h3>{}</h3>\n", symbol));
            html.push_str("<div class='metric-grid'>\n");

            let metrics = vec![
                ("Current Price", format!("${:.2}", result.current_price)),
                ("1yr Mean", format!("${:.2}", result.mean_price)),
                ("Std Deviation", format!("${:.2}", result.std_dev)),
                ("RSI (14)", format!("{:.1}", result.rsi_14.unwrap_or(0.0))),
                ("SMA 7-day", format!("${:.2}", result.sma_7.unwrap_or(0.0))),
                ("SMA 30-day", format!("${:.2}", result.sma_30.unwrap_or(0.0))),
            ];
            for (label, value) in &metrics {
                html.push_str(&format!(
                    "<div class='metric-box'><div class='value'>{}</div>\
                     <div class='label'>{}</div></div>\n",
                    value, label
                ));
            }
            html.push_str("</div>\n");

            // Price chart with moving averages
            let sma_7 = analysis::sma(&prices, 7);
            let sma_30 = analysis::sma(&prices, 30);
            html.push_str(&charts::price_chart_svg(
                &prices, &sma_7, &sma_30,
                &format!("{} — Price with Moving Averages", symbol)
            ));

            // Bollinger Bands chart
            let bands_chart = analysis::bollinger_bands(&prices, 20, 2.0);
            html.push_str(&charts::bollinger_chart_svg(
                &prices, &bands_chart,
                &format!("{} — Bollinger Bands (20, 2σ)", symbol)
            ));

            // MACD
            let (macd_line, signal_line, histogram) = analysis::macd(&prices);
            if let (Some(&macd_val), Some(&signal_val), Some(&hist_val)) =
                (macd_line.last(), signal_line.last(), histogram.last())
            {
                let macd_signal = if hist_val > 0.0 {
                    "<span class='positive'>BULLISH (MACD above signal)</span>"
                } else {
                    "<span class='negative'>BEARISH (MACD below signal)</span>"
                };
                html.push_str(&format!(
                    "<p><strong>MACD:</strong> {:.4} | <strong>Signal:</strong> {:.4} | \
                     <strong>Histogram:</strong> {:.4} — {}</p>\n",
                    macd_val, signal_val, hist_val, macd_signal
                ));
            }

            // Bollinger Bands text
            let bands = analysis::bollinger_bands(&prices, 20, 2.0);
            if let Some(&(upper, middle, lower)) = bands.last() {
                let bb_position = if result.current_price > upper {
                    "<span class='negative'>ABOVE upper band — overbought</span>"
                } else if result.current_price < lower {
                    "<span class='positive'>BELOW lower band — oversold</span>"
                } else {
                    "<span class='neutral'>Within bands — normal range</span>"
                };
                html.push_str(&format!(
                    "<p><strong>Bollinger Bands (20,2):</strong> \
                     Upper ${:.2} | Middle ${:.2} | Lower ${:.2} — {}</p>\n",
                    upper, middle, lower, bb_position
                ));
            }

            html.push_str("</div>\n");
        }
    }

    // ── Correlation matrix ──
    html.push_str("<h2>Correlation Matrix (Daily Returns)</h2>\n");
    html.push_str("<table class='correlation-table'>\n<tr><th></th>");

    let coin_ids: Vec<&String> = coin_data.iter().map(|(id, _, _)| id).collect();
    for id in &coin_ids {
        html.push_str(&format!("<th>{}</th>", id));
    }
    html.push_str("</tr>\n");

    let all_returns: Vec<Vec<f64>> = coin_data.iter()
        .map(|(_, points, _)| {
            let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
            analysis::daily_returns(&prices)
        })
        .collect();

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
    html.push_str("<p><em>High correlation (&gt;0.7) = coins move together. \
        Low correlation (&lt;0.3) = independent movement.</em></p>\n");

    // ── Footer ──
    html.push_str(
        "<hr style='border-color: #1e3a5f; margin-top: 40px;'>\n\
         <p style='text-align: center; color: #555;'>\
         Rust Invest — Built in Rust | Data: CoinGecko &amp; Yahoo Finance</p>\n\
         </body></html>"
    );

    fs::write(output_path, &html)?;
    println!("  ✓ Report saved to: {}", output_path);

    Ok(())
}