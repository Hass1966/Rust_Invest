/// Machine Learning models for price prediction

use crate::analysis;

// ── Feature engineering ──

pub struct Sample {
    pub features: Vec<f64>,
    pub label: f64,        // next day's return
}

pub struct PredictionResult {
    pub symbol: String,
    pub actual: f64,
    pub predicted: f64,
    pub direction_correct: bool,
}

pub struct ModelMetrics {
    pub symbol: String,
    pub mse: f64,
    pub mae: f64,
    pub direction_accuracy: f64,
    pub total_samples: usize,
    pub train_samples: usize,
    pub test_samples: usize,
}

/// Build feature vectors from price history
/// Each row = one day, columns = [rsi, macd_hist, bb_position, sma_ratio, volatility, return]
pub fn build_features(prices: &[f64], lookback: usize) -> Vec<Sample> {
    if prices.len() < lookback + 30 {
        return vec![];
    }

    let mut samples = Vec::new();

    let start = 30.max(lookback);

    for i in start..prices.len() - 1 {
        let window = &prices[..=i];

        let rsi = analysis::rsi(window, 14).unwrap_or(50.0) / 100.0;

        let (_, _, histogram) = analysis::macd(window);
        let macd_hist = histogram.last()
            .map(|h| h / prices[i])
            .unwrap_or(0.0);

        let bands = analysis::bollinger_bands(window, 20, 2.0);
        let bb_pos = bands.last()
            .map(|(upper, _, lower)| {
                let range = upper - lower;
                if range == 0.0 { 0.5 } else { (prices[i] - lower) / range }
            })
            .unwrap_or(0.5);

        let sma_7 = analysis::sma(window, 7);
        let sma_30 = analysis::sma(window, 30);
        let sma_ratio = match (sma_7.last(), sma_30.last()) {
            (Some(s), Some(l)) if *l != 0.0 => s / l - 1.0,
            _ => 0.0,
        };

        let recent_returns = analysis::daily_returns(
            &window[window.len().saturating_sub(15)..]
        );
        let volatility = analysis::std_dev(&recent_returns) / 100.0;

        let today_return = (prices[i] - prices[i - 1]) / prices[i - 1];

        let tomorrow_return = (prices[i + 1] - prices[i]) / prices[i] * 100.0;

        samples.push(Sample {
            features: vec![rsi, macd_hist, bb_pos, sma_ratio, volatility, today_return],
            label: tomorrow_return,
        });
    }

    samples
}

// ── Linear Regression ──

pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub feature_names: Vec<String>,
}

impl LinearRegression {
    pub fn new(num_features: usize) -> Self {
        LinearRegression {
            weights: vec![0.0; num_features],
            bias: 0.0,
            feature_names: vec![
                "RSI".into(),
                "MACD Hist".into(),
                "BB Position".into(),
                "SMA Ratio".into(),
                "Volatility".into(),
                "Today Return".into(),
            ],
        }
    }

    /// Return weights as name-value pairs for the report
    pub fn get_weights(&self) -> Vec<(String, f64)> {
        self.feature_names.iter()
            .zip(self.weights.iter())
            .map(|(name, weight)| (name.clone(), *weight))
            .collect()
    }

    /// Train using gradient descent
    pub fn train(&mut self, samples: &[Sample], learning_rate: f64, epochs: usize) {
        let n = samples.len() as f64;

        for epoch in 0..epochs {
            let mut weight_grads = vec![0.0; self.weights.len()];
            let mut bias_grad = 0.0;

            for sample in samples {
                let prediction = self.predict(&sample.features);
                let error = prediction - sample.label;

                for (j, feature) in sample.features.iter().enumerate() {
                    weight_grads[j] += error * feature;
                }
                bias_grad += error;
            }

            for (j, grad) in weight_grads.iter().enumerate() {
                self.weights[j] -= learning_rate * grad / n;
            }
            self.bias -= learning_rate * bias_grad / n;

            if (epoch + 1) % 1000 == 0 {
                let mse = self.calculate_mse(samples);
                println!("    Epoch {}: MSE = {:.6}", epoch + 1, mse);
            }
        }
    }

    /// Predict next day's return
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, f) in self.weights.iter().zip(features.iter()) {
            result += w * f;
        }
        result
    }

    fn calculate_mse(&self, samples: &[Sample]) -> f64 {
        let sum: f64 = samples.iter()
            .map(|s| {
                let error = self.predict(&s.features) - s.label;
                error * error
            })
            .sum();
        sum / samples.len() as f64
    }

    /// Evaluate on test data
    pub fn evaluate(&self, test_data: &[Sample], symbol: &str, train_size: usize) -> ModelMetrics {
        let mut sum_squared_error = 0.0;
        let mut sum_abs_error = 0.0;
        let mut correct_direction = 0;

        for sample in test_data {
            let predicted = self.predict(&sample.features);
            let error = predicted - sample.label;

            sum_squared_error += error * error;
            sum_abs_error += error.abs();

            if (predicted > 0.0 && sample.label > 0.0)
                || (predicted <= 0.0 && sample.label <= 0.0)
            {
                correct_direction += 1;
            }
        }

        let n = test_data.len() as f64;

        ModelMetrics {
            symbol: symbol.to_string(),
            mse: sum_squared_error / n,
            mae: sum_abs_error / n,
            direction_accuracy: correct_direction as f64 / n * 100.0,
            total_samples: train_size + test_data.len(),
            train_samples: train_size,
            test_samples: test_data.len(),
        }
    }

    /// Print learned weights (feature importance)
    pub fn print_weights(&self) {
        println!("    Learned weights:");
        for (name, weight) in self.feature_names.iter().zip(self.weights.iter()) {
            let bar_len = (weight.abs() * 100.0) as usize;
            let bar: String = "█".repeat(bar_len.min(30));
            let sign = if *weight >= 0.0 { "+" } else { "-" };
            println!("      {:<14} {:>8.4}  {} {}",
                     name, weight, sign, bar);
        }
        println!("      {:<14} {:>8.4}", "Bias", self.bias);
    }
}

/// Run the full ML pipeline for a given asset
pub fn run_pipeline(
    symbol: &str,
    prices: &[f64],
    train_ratio: f64,
) -> Option<(ModelMetrics, LinearRegression)> {
    let samples = build_features(prices, 30);

    if samples.len() < 50 {
        println!("  {} — not enough samples ({}) for training", symbol, samples.len());
        return None;
    }

    let split = (samples.len() as f64 * train_ratio) as usize;
    let (train, test) = samples.split_at(split);

    println!("\n  ┌─── {} ───", symbol);
    println!("  │ Samples: {} total ({} train / {} test)",
             samples.len(), train.len(), test.len());

    let mut model = LinearRegression::new(6);
    model.train(train, 0.01, 5000);

    let metrics = model.evaluate(test, symbol, train.len());

    println!("  │");
    model.print_weights();
    println!("  │");
    println!("  │ Test Results:");
    println!("  │   MSE:                {:.6}", metrics.mse);
    println!("  │   MAE:                {:.4}%", metrics.mae);
    println!("  │   Direction Accuracy:  {:.1}%", metrics.direction_accuracy);

    let verdict = if metrics.direction_accuracy > 55.0 {
        "PROMISING — beats random"
    } else if metrics.direction_accuracy > 50.0 {
        "MARGINAL — slightly better than coin flip"
    } else {
        "NO EDGE — model not predictive"
    };
    println!("  │   Verdict:            {}", verdict);
    println!("  └───────────────────────\n");

    Some((metrics, model))
}