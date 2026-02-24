/// SVG chart generation for the HTML report

pub struct ChartConfig {
    pub width: f64,
    pub height: f64,
    pub padding: f64,
}

impl Default for ChartConfig {
    fn default() -> Self {
        ChartConfig {
            width: 800.0,
            height: 300.0,
            padding: 50.0,
        }
    }
}

pub fn price_chart_svg(
    prices: &[f64],
    sma_7: &[f64],
    sma_30: &[f64],
    title: &str,
) -> String {
    let config = ChartConfig::default();
    let w = config.width;
    let h = config.height;
    let pad = config.padding;

    if prices.is_empty() {
        return String::new();
    }

    let min_price = prices.iter().cloned().reduce(f64::min).unwrap_or(0.0);
    let max_price = prices.iter().cloned().reduce(f64::max).unwrap_or(1.0);
    let price_range = max_price - min_price;
    let price_range = if price_range == 0.0 { 1.0 } else { price_range };

    let chart_w = w - 2.0 * pad;
    let chart_h = h - 2.0 * pad;

    let to_x = |i: usize, len: usize| -> f64 {
        pad + (i as f64 / (len.max(1) - 1).max(1) as f64) * chart_w
    };

    let to_y = |price: f64| -> f64 {
        pad + chart_h - ((price - min_price) / price_range) * chart_h
    };

    let mut svg = format!(
        "<svg width='100%' viewBox='0 0 {} {}' xmlns='http://www.w3.org/2000/svg' \
         style='background: #0f1923; border-radius: 8px; margin: 10px 0;'>\n",
        w, h
    );

    // Grid lines
    let num_grid = 5;
    for i in 0..=num_grid {
        let y = pad + (i as f64 / num_grid as f64) * chart_h;
        let price_val = max_price - (i as f64 / num_grid as f64) * price_range;
        svg.push_str(&format!(
            "<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#1e3a5f' stroke-width='0.5'/>\n",
            pad, y, w - pad, y
        ));
        svg.push_str(&format!(
            "<text x='{}' y='{}' fill='#666' font-size='10' text-anchor='end'>${:.0}</text>\n",
            pad - 5.0, y + 4.0, price_val
        ));
    }

    // Price line
    svg.push_str(&make_line(prices, &to_x, &to_y, "#4fc3f7", "1.5", prices.len()));

    // SMA 7 line
    if sma_7.len() > 1 {
        let offset = prices.len() - sma_7.len();
        svg.push_str(&make_line_offset(sma_7, offset, &to_x, &to_y, "#00e676", "1", prices.len()));
    }

    // SMA 30 line
    if sma_30.len() > 1 {
        let offset = prices.len() - sma_30.len();
        svg.push_str(&make_line_offset(sma_30, offset, &to_x, &to_y, "#ff9800", "1", prices.len()));
    }

    // Title
    svg.push_str(&format!(
        "<text x='{}' y='25' fill='#e0e0e0' font-size='14' font-weight='bold'>{}</text>\n",
        pad, title
    ));

    // Legend
    let legend_y = h - 10.0;
    svg.push_str(&format!(
        "<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#4fc3f7' stroke-width='2'/>\n\
         <text x='{}' y='{}' fill='#888' font-size='10'>Price</text>\n",
        pad, legend_y, pad + 20.0, legend_y, pad + 25.0, legend_y + 4.0
    ));
    svg.push_str(&format!(
        "<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#00e676' stroke-width='2'/>\n\
         <text x='{}' y='{}' fill='#888' font-size='10'>SMA 7</text>\n",
        pad + 70.0, legend_y, pad + 90.0, legend_y, pad + 95.0, legend_y + 4.0
    ));
    svg.push_str(&format!(
        "<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#ff9800' stroke-width='2'/>\n\
         <text x='{}' y='{}' fill='#888' font-size='10'>SMA 30</text>\n",
        pad + 145.0, legend_y, pad + 165.0, legend_y, pad + 170.0, legend_y + 4.0
    ));

    svg.push_str("</svg>\n");
    svg
}

pub fn bollinger_chart_svg(
    prices: &[f64],
    bands: &[(f64, f64, f64)], // (upper, middle, lower)
    title: &str,
) -> String {
    let config = ChartConfig::default();
    let w = config.width;
    let h = config.height;
    let pad = config.padding;

    if prices.is_empty() || bands.is_empty() {
        return String::new();
    }

    // Find min/max across prices and bands
    let all_values: Vec<f64> = prices.iter().copied()
        .chain(bands.iter().map(|(u, _, _)| *u))
        .chain(bands.iter().map(|(_, _, l)| *l))
        .collect();

    let min_val = all_values.iter().cloned().reduce(f64::min).unwrap_or(0.0);
    let max_val = all_values.iter().cloned().reduce(f64::max).unwrap_or(1.0);
    let val_range = if max_val - min_val == 0.0 { 1.0 } else { max_val - min_val };

    let chart_w = w - 2.0 * pad;
    let chart_h = h - 2.0 * pad;

    let to_x = |i: usize, len: usize| -> f64 {
        pad + (i as f64 / (len.max(1) - 1).max(1) as f64) * chart_w
    };

    let to_y = |val: f64| -> f64 {
        pad + chart_h - ((val - min_val) / val_range) * chart_h
    };

    let mut svg = format!(
        "<svg width='100%' viewBox='0 0 {} {}' xmlns='http://www.w3.org/2000/svg' \
         style='background: #0f1923; border-radius: 8px; margin: 10px 0;'>\n",
        w, h
    );

    // Bollinger band fill
    let offset = prices.len() - bands.len();
    let mut band_path = String::from("M");

    // Upper line (forward)
    for (i, (upper, _, _)) in bands.iter().enumerate() {
        let x = to_x(i + offset, prices.len());
        let y = to_y(*upper);
        if i == 0 {
            band_path.push_str(&format!("{},{}", x, y));
        } else {
            band_path.push_str(&format!(" L{},{}", x, y));
        }
    }

    // Lower line (backward)
    for (i, (_, _, lower)) in bands.iter().enumerate().rev() {
        let x = to_x(i + offset, prices.len());
        let y = to_y(*lower);
        band_path.push_str(&format!(" L{},{}", x, y));
    }
    band_path.push('Z');

    svg.push_str(&format!(
        "<path d='{}' fill='#1e3a5f' fill-opacity='0.3'/>\n",
        band_path
    ));

    // Price line
    svg.push_str(&make_line(prices, &to_x, &to_y, "#4fc3f7", "1.5", prices.len()));

    // Middle band line
    let middles: Vec<f64> = bands.iter().map(|(_, m, _)| *m).collect();
    svg.push_str(&make_line_offset(&middles, offset, &to_x, &to_y, "#ffd740", "1", prices.len()));

    // Title
    svg.push_str(&format!(
        "<text x='{}' y='25' fill='#e0e0e0' font-size='14' font-weight='bold'>{}</text>\n",
        pad, title
    ));

    // Legend
    let legend_y = h - 10.0;
    svg.push_str(&format!(
        "<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#4fc3f7' stroke-width='2'/>\n\
         <text x='{}' y='{}' fill='#888' font-size='10'>Price</text>\n",
        pad, legend_y, pad + 20.0, legend_y, pad + 25.0, legend_y + 4.0
    ));
    svg.push_str(&format!(
        "<rect x='{}' y='{}' width='20' height='10' fill='#1e3a5f' fill-opacity='0.5'/>\n\
         <text x='{}' y='{}' fill='#888' font-size='10'>Bollinger Bands</text>\n",
        pad + 70.0, legend_y - 5.0, pad + 95.0, legend_y + 4.0
    ));

    svg.push_str("</svg>\n");
    svg
}

// ── Helper functions ──

fn make_line(
    data: &[f64],
    to_x: &dyn Fn(usize, usize) -> f64,
    to_y: &dyn Fn(f64) -> f64,
    color: &str,
    width: &str,
    total_len: usize,
) -> String {
    if data.len() < 2 {
        return String::new();
    }

    let mut path = String::from("<polyline points='");
    for (i, val) in data.iter().enumerate() {
        let x = to_x(i, total_len);
        let y = to_y(*val);
        if i > 0 {
            path.push(' ');
        }
        path.push_str(&format!("{:.1},{:.1}", x, y));
    }
    path.push_str(&format!(
        "' fill='none' stroke='{}' stroke-width='{}'/>\n",
        color, width
    ));
    path
}

fn make_line_offset(
    data: &[f64],
    offset: usize,
    to_x: &dyn Fn(usize, usize) -> f64,
    to_y: &dyn Fn(f64) -> f64,
    color: &str,
    width: &str,
    total_len: usize,
) -> String {
    if data.len() < 2 {
        return String::new();
    }

    let mut path = String::from("<polyline points='");
    for (i, val) in data.iter().enumerate() {
        let x = to_x(i + offset, total_len);
        let y = to_y(*val);
        if i > 0 {
            path.push(' ');
        }
        path.push_str(&format!("{:.1},{:.1}", x, y));
    }
    path.push_str(&format!(
        "' fill='none' stroke='{}' stroke-width='{}'/>\n",
        color, width
    ));
    path
}