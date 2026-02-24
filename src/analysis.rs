/// Statistical analysis functions for market data

pub struct PricePoint {
    pub timestamp: String,
    pub price: f64,
    pub volume: Option<f64>,
}

pub struct AnalysisResult {
    pub coin_id: String,
    pub current_price: f64,
    pub mean_price: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub std_dev: f64,
    pub daily_returns_mean: f64,
    pub daily_returns_std: f64,
    pub sma_7: Option<f64>,
    pub sma_30: Option<f64>,
    pub rsi_14: Option<f64>,
}

// ── Basic statistics ──

pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let avg = mean(data);
    let variance = data.iter()
        .map(|x| (x - avg).powi(2))
        .sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

pub fn min(data: &[f64]) -> f64 {
    data.iter().cloned().reduce(f64::min).unwrap_or(0.0)
}

pub fn max(data: &[f64]) -> f64 {
    data.iter().cloned().reduce(f64::max).unwrap_or(0.0)
}

// ── Financial calculations ──

/// Calculate daily returns as percentage changes
pub fn daily_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0] * 100.0)
        .collect()
}

/// Simple Moving Average over `period` days
pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![];
    }
    prices.windows(period)
        .map(|w| w.iter().sum::<f64>() / period as f64)
        .collect()
}

/// Exponential Moving Average
pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() || period == 0 {
        return vec![];
    }
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = vec![prices[0]]; // start with first price

    for i in 1..prices.len() {
        let prev = result[i - 1];
        let current = (prices[i] - prev) * multiplier + prev;
        result.push(current);
    }
    result
}

/// Relative Strength Index (14-day default)
pub fn rsi(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 {
        return None;
    }

    let changes: Vec<f64> = prices.windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    let recent = &changes[changes.len() - period..];

    let gains: f64 = recent.iter()
        .filter(|c| **c > 0.0)
        .sum();
    let losses: f64 = recent.iter()
        .filter(|c| **c < 0.0)
        .map(|c| c.abs())
        .sum();

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - (100.0 / (1.0 + rs)))
}

/// Bollinger Bands (returns upper, middle, lower)
pub fn bollinger_bands(prices: &[f64], period: usize, num_std: f64) -> Vec<(f64, f64, f64)> {
    if prices.len() < period {
        return vec![];
    }

    prices.windows(period)
        .map(|w| {
            let middle = mean(w);
            let sd = std_dev(w);
            let upper = middle + num_std * sd;
            let lower = middle - num_std * sd;
            (upper, middle, lower)
        })
        .collect()
}

/// Analyse a full coin's price history
pub fn analyse_coin(coin_id: &str, points: &[PricePoint]) -> AnalysisResult {
    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let returns = daily_returns(&prices);

    let sma_7_values = sma(&prices, 7);
    let sma_30_values = sma(&prices, 30);

    AnalysisResult {
        coin_id: coin_id.to_string(),
        current_price: *prices.last().unwrap_or(&0.0),
        mean_price: mean(&prices),
        min_price: min(&prices),
        max_price: max(&prices),
        std_dev: std_dev(&prices),
        daily_returns_mean: mean(&returns),
        daily_returns_std: std_dev(&returns),
        sma_7: sma_7_values.last().copied(),
        sma_30: sma_30_values.last().copied(),
        rsi_14: rsi(&prices, 14),
    }
}

/// Print a formatted analysis report
pub fn print_report(result: &AnalysisResult) {
    println!("┌─────────────────────────────────────────┐");
    println!("│  {} ", result.coin_id.to_uppercase());
    println!("├─────────────────────────────────────────┤");
    println!("│  Current Price:    ${:>12.2}          ", result.current_price);
    println!("│  Mean Price:       ${:>12.2}          ", result.mean_price);
    println!("│  Min Price:        ${:>12.2}          ", result.min_price);
    println!("│  Max Price:        ${:>12.2}          ", result.max_price);
    println!("│  Std Deviation:    ${:>12.2}          ", result.std_dev);
    println!("├─────────────────────────────────────────┤");
    println!("│  Daily Returns:                         ");
    println!("│    Mean:           {:>11.4}%          ", result.daily_returns_mean);
    println!("│    Std Dev:        {:>11.4}%          ", result.daily_returns_std);
    println!("├─────────────────────────────────────────┤");

    if let Some(sma7) = result.sma_7 {
        println!("│  SMA (7-day):      ${:>12.2}          ", sma7);
    }
    if let Some(sma30) = result.sma_30 {
        println!("│  SMA (30-day):     ${:>12.2}          ", sma30);
    }
    if let Some(rsi_val) = result.rsi_14 {
        let signal = if rsi_val > 70.0 {
            "OVERBOUGHT"
        } else if rsi_val < 30.0 {
            "OVERSOLD"
        } else {
            "NEUTRAL"
        };
        println!("│  RSI (14-day):     {:>12.2} ({})  ", rsi_val, signal);
    }

    println!("└─────────────────────────────────────────┘");
}


    /// MACD - Moving Average Convergence Divergence
    /// Returns (macd_line, signal_line, histogram)
    pub fn macd(prices: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_12 = ema(prices, 12);
        let ema_26 = ema(prices, 26);

        if ema_12.len() != ema_26.len() {
            return (vec![], vec![], vec![]);
        }

        // MACD line = EMA12 - EMA26
        let macd_line: Vec<f64> = ema_12.iter()
            .zip(ema_26.iter())
            .map(|(a, b)| a - b)
            .collect();

        // Signal line = 9-day EMA of MACD line
        let signal_line = ema(&macd_line, 9);

        // Histogram = MACD - Signal
        let histogram: Vec<f64> = macd_line.iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Correlation between two price series
    pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        if len < 2 {
            return 0.0;
        }

        let a = &a[..len];
        let b = &b[..len];

        let mean_a = mean(a);
        let mean_b = mean(b);

        let numerator: f64 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum();

        let denom_a: f64 = a.iter()
            .map(|x| (x - mean_a).powi(2))
            .sum::<f64>()
            .sqrt();

        let denom_b: f64 = b.iter()
            .map(|y| (y - mean_b).powi(2))
            .sum::<f64>()
            .sqrt();

        if denom_a == 0.0 || denom_b == 0.0 {
            return 0.0;
        }

        numerator / (denom_a * denom_b)
    
}