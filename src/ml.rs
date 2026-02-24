/// Machine Learning models for price prediction
/// Model 1: Linear Regression (baseline) — predicts return magnitude
/// Model 2: Logistic Regression — predicts direction (up/down)
/// Enhanced feature set with lagged indicators and normalisation

use crate::analysis;

// ── Data structures ──

#[derive(Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: f64,
}

pub struct ModelMetrics {
    pub symbol: String,
    pub model_name: String,
    pub mse: f64,
    pub mae: f64,
    pub direction_accuracy: f64,
    pub total_samples: usize,
    pub train_samples: usize,
    pub test_samples: usize,
}

pub const FEATURE_NAMES: &[&str] = &[
    "RSI",
    "RSI Δ3d",
    "MACD Hist",
    "BB Position",
    "SMA Ratio",
    "Volatility",
    "Today Return",
    "Return 3d Avg",
    "Volume Ratio",
    "Momentum 5d",
];

// ── Feature engineering ──

pub fn build_features(prices: &[f64], volumes: &[Option<f64>]) -> Vec<Sample> {
    if prices.len() < 35 {
        return vec![];
    }

    let mut samples = Vec::new();
    let start = 33; // need lookback for lagged features

    for i in start..prices.len() - 1 {
        let window = &prices[..=i];

        // Feature 1: RSI (14-day), normalised 0-1
        let rsi = analysis::rsi(window, 14).unwrap_or(50.0) / 100.0;

        // Feature 2: RSI rate of change (RSI today - RSI 3 days ago)
        let rsi_3d_ago = analysis::rsi(&prices[..=i.saturating_sub(3)], 14)
            .unwrap_or(50.0) / 100.0;
        let rsi_delta = rsi - rsi_3d_ago;

        // Feature 3: MACD histogram normalised by price
        let (_, _, histogram) = analysis::macd(window);
        let macd_hist = histogram.last()
            .map(|h| h / prices[i])
            .unwrap_or(0.0);

        // Feature 4: Bollinger Band position (0 = lower, 1 = upper)
        let bands = analysis::bollinger_bands(window, 20, 2.0);
        let bb_pos = bands.last()
            .map(|(upper, _, lower)| {
                let range = upper - lower;
                if range == 0.0 { 0.5 } else { (prices[i] - lower) / range }
            })
            .unwrap_or(0.5);

        // Feature 5: SMA ratio (7/30 - 1)
        let sma_7 = analysis::sma(window, 7);
        let sma_30 = analysis::sma(window, 30);
        let sma_ratio = match (sma_7.last(), sma_30.last()) {
            (Some(s), Some(l)) if *l != 0.0 => s / l - 1.0,
            _ => 0.0,
        };

        // Feature 6: Rolling volatility (14-day)
        let recent_returns = analysis::daily_returns(
            &window[window.len().saturating_sub(15)..]
        );
        let volatility = analysis::std_dev(&recent_returns) / 100.0;

        // Feature 7: Today's return
        let today_return = (prices[i] - prices[i - 1]) / prices[i - 1];

        // Feature 8: Average return over last 3 days (smoothed momentum)
        let ret_3d = if i >= 3 {
            (prices[i] - prices[i - 3]) / prices[i - 3] / 3.0
        } else {
            0.0
        };

        // Feature 9: Volume ratio (today / 10-day average)
        let vol_ratio = if i >= 10 {
            let today_vol = volumes.get(i).and_then(|v| *v).unwrap_or(1.0);
            let avg_vol: f64 = (i.saturating_sub(10)..i)
                .filter_map(|j| volumes.get(j).and_then(|v| *v))
                .sum::<f64>() / 10.0;
            if avg_vol > 0.0 { today_vol / avg_vol } else { 1.0 }
        } else {
            1.0
        };

        // Feature 10: 5-day momentum (price now vs 5 days ago, normalised)
        let momentum_5d = if i >= 5 {
            (prices[i] - prices[i - 5]) / prices[i - 5]
        } else {
            0.0
        };

        // Label: tomorrow's return
        let tomorrow_return = (prices[i + 1] - prices[i]) / prices[i] * 100.0;

        samples.push(Sample {
            features: vec![
                rsi, rsi_delta, macd_hist, bb_pos, sma_ratio,
                volatility, today_return, ret_3d, vol_ratio, momentum_5d,
            ],
            label: tomorrow_return,
        });
    }

    samples
}

/// Normalise features to zero mean, unit variance
pub fn normalise(samples: &mut [Sample]) -> (Vec<f64>, Vec<f64>) {
    if samples.is_empty() {
        return (vec![], vec![]);
    }

    let num_features = samples[0].features.len();
    let n = samples.len() as f64;

    // Calculate means
    let mut means = vec![0.0; num_features];
    for sample in samples.iter() {
        for (j, f) in sample.features.iter().enumerate() {
            means[j] += f;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }

    // Calculate std devs
    let mut stds = vec![0.0; num_features];
    for sample in samples.iter() {
        for (j, f) in sample.features.iter().enumerate() {
            stds[j] += (f - means[j]).powi(2);
        }
    }
    for s in stds.iter_mut() {
        *s = (*s / n).sqrt();
        if *s == 0.0 { *s = 1.0; } // avoid division by zero
    }

    // Apply normalisation
    for sample in samples.iter_mut() {
        for (j, f) in sample.features.iter_mut().enumerate() {
            *f = (*f - means[j]) / stds[j];
        }
    }

    (means, stds)
}

/// Apply pre-computed normalisation to new data
pub fn apply_normalisation(samples: &mut [Sample], means: &[f64], stds: &[f64]) {
    for sample in samples.iter_mut() {
        for (j, f) in sample.features.iter_mut().enumerate() {
            *f = (*f - means[j]) / stds[j];
        }
    }
}

// ── Linear Regression ──

pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearRegression {
    pub fn new(num_features: usize) -> Self {
        LinearRegression {
            weights: vec![0.0; num_features],
            bias: 0.0,
        }
    }

    pub fn train(&mut self, samples: &[Sample], learning_rate: f64, epochs: usize) {
        let n = samples.len() as f64;

        for epoch in 0..epochs {
            let mut weight_grads = vec![0.0; self.weights.len()];
            let mut bias_grad = 0.0;

            for sample in samples {
                let pred = self.predict(&sample.features);
                let error = pred - sample.label;
                for (j, feature) in sample.features.iter().enumerate() {
                    weight_grads[j] += error * feature;
                }
                bias_grad += error;
            }

            for (j, grad) in weight_grads.iter().enumerate() {
                self.weights[j] -= learning_rate * grad / n;
            }
            self.bias -= learning_rate * bias_grad / n;

            if (epoch + 1) % 2000 == 0 {
                let mse = self.mse(samples);
                println!("    [LinReg] Epoch {}: MSE = {:.6}", epoch + 1, mse);
            }
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, f) in self.weights.iter().zip(features.iter()) {
            result += w * f;
        }
        result
    }

    fn mse(&self, samples: &[Sample]) -> f64 {
        let sum: f64 = samples.iter()
            .map(|s| (self.predict(&s.features) - s.label).powi(2))
            .sum();
        sum / samples.len() as f64
    }

    pub fn get_weights(&self) -> Vec<(String, f64)> {
        FEATURE_NAMES.iter()
            .zip(self.weights.iter())
            .map(|(name, weight)| (name.to_string(), *weight))
            .collect()
    }

    pub fn evaluate(&self, test: &[Sample], symbol: &str, train_size: usize) -> ModelMetrics {
        let mut sse = 0.0;
        let mut sae = 0.0;
        let mut correct = 0;

        for s in test {
            let pred = self.predict(&s.features);
            let err = pred - s.label;
            sse += err * err;
            sae += err.abs();
            if (pred > 0.0) == (s.label > 0.0) { correct += 1; }
        }

        let n = test.len() as f64;
        ModelMetrics {
            symbol: symbol.to_string(),
            model_name: "Linear Regression".to_string(),
            mse: sse / n,
            mae: sae / n,
            direction_accuracy: correct as f64 / n * 100.0,
            total_samples: train_size + test.len(),
            train_samples: train_size,
            test_samples: test.len(),
        }
    }
}

// ── Logistic Regression (direction classifier) ──

pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LogisticRegression {
    pub fn new(num_features: usize) -> Self {
        LogisticRegression {
            weights: vec![0.0; num_features],
            bias: 0.0,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Train: label > 0 = class 1 (up), label <= 0 = class 0 (down)
    pub fn train(&mut self, samples: &[Sample], learning_rate: f64, epochs: usize) {
        let n = samples.len() as f64;

        for epoch in 0..epochs {
            let mut weight_grads = vec![0.0; self.weights.len()];
            let mut bias_grad = 0.0;

            for sample in samples {
                let target = if sample.label > 0.0 { 1.0 } else { 0.0 };
                let pred = self.predict_probability(&sample.features);
                let error = pred - target;

                for (j, feature) in sample.features.iter().enumerate() {
                    weight_grads[j] += error * feature;
                }
                bias_grad += error;
            }

            for (j, grad) in weight_grads.iter().enumerate() {
                self.weights[j] -= learning_rate * grad / n;
            }
            self.bias -= learning_rate * bias_grad / n;

            if (epoch + 1) % 2000 == 0 {
                let acc = self.accuracy(samples);
                println!("    [LogReg] Epoch {}: Train Accuracy = {:.1}%", epoch + 1, acc);
            }
        }
    }

    pub fn predict_probability(&self, features: &[f64]) -> f64 {
        let mut z = self.bias;
        for (w, f) in self.weights.iter().zip(features.iter()) {
            z += w * f;
        }
        Self::sigmoid(z)
    }

    pub fn predict_direction(&self, features: &[f64]) -> bool {
        self.predict_probability(features) > 0.5
    }

    fn accuracy(&self, samples: &[Sample]) -> f64 {
        let correct = samples.iter()
            .filter(|s| {
                let predicted_up = self.predict_direction(&s.features);
                let actual_up = s.label > 0.0;
                predicted_up == actual_up
            })
            .count();
        correct as f64 / samples.len() as f64 * 100.0
    }

    pub fn get_weights(&self) -> Vec<(String, f64)> {
        FEATURE_NAMES.iter()
            .zip(self.weights.iter())
            .map(|(name, weight)| (name.to_string(), *weight))
            .collect()
    }

    pub fn evaluate(&self, test: &[Sample], symbol: &str, train_size: usize) -> ModelMetrics {
        let mut correct = 0;
        let mut sse = 0.0;
        let mut sae = 0.0;

        for s in test {
            let predicted_up = self.predict_direction(&s.features);
            let actual_up = s.label > 0.0;
            if predicted_up == actual_up { correct += 1; }

            // Use probability distance as error metric
            let prob = self.predict_probability(&s.features);
            let target = if actual_up { 1.0 } else { 0.0 };
            let err = prob - target;
            sse += err * err;
            sae += err.abs();
        }

        let n = test.len() as f64;
        ModelMetrics {
            symbol: symbol.to_string(),
            model_name: "Logistic Regression".to_string(),
            mse: sse / n,
            mae: sae / n,
            direction_accuracy: correct as f64 / n * 100.0,
            total_samples: train_size + test.len(),
            train_samples: train_size,
            test_samples: test.len(),
        }
    }
}

// ── Combined pipeline ──

pub struct PipelineResult {
    pub linear_metrics: ModelMetrics,
    pub logistic_metrics: ModelMetrics,
    pub linear_weights: Vec<(String, f64)>,
    pub logistic_weights: Vec<(String, f64)>,
    pub best_direction_accuracy: f64,
    pub best_model_name: String,
}

pub fn run_pipeline(
    symbol: &str,
    prices: &[f64],
    volumes: &[Option<f64>],
    train_ratio: f64,
) -> Option<PipelineResult> {
    let mut samples = build_features(prices, volumes);

    if samples.len() < 50 {
        println!("  {} — not enough samples ({})", symbol, samples.len());
        return None;
    }

    // Chronological split
    let split = (samples.len() as f64 * train_ratio) as usize;

    // Normalise on training data, apply same transform to test
    let (mut train, mut test) = {
        let (t, te) = samples.split_at_mut(split);
        (t.to_vec(), te.to_vec())
    };

    // We need to rebuild because split_at_mut borrows
    let mut all_samples = build_features(prices, volumes);
    let split = (all_samples.len() as f64 * train_ratio) as usize;

    let (train_slice, _) = all_samples.split_at_mut(split);
    let (means, stds) = normalise(train_slice);

    // Rebuild with normalisation applied consistently
    let mut all_samples = build_features(prices, volumes);
    let (train_data, test_data) = all_samples.split_at_mut(split);
    normalise(train_data);
    apply_normalisation(test_data, &means, &stds);

    let num_features = train_data[0].features.len();

    println!("\n  ┌─── {} ───", symbol);
    println!("  │ Samples: {} ({} train / {} test) × {} features",
             train_data.len() + test_data.len(), train_data.len(), test_data.len(), num_features);

    // ── Model 1: Linear Regression ──
    println!("  │");
    println!("  │ Model 1: Linear Regression (return prediction)");

    let mut lin_model = LinearRegression::new(num_features);
    lin_model.train(train_data, 0.005, 6000);
    let lin_metrics = lin_model.evaluate(test_data, symbol, train_data.len());
    let lin_weights = lin_model.get_weights();

    println!("  │   → Direction: {:.1}% | MAE: {:.4}%",
             lin_metrics.direction_accuracy, lin_metrics.mae);

    // ── Model 2: Logistic Regression ──
    println!("  │");
    println!("  │ Model 2: Logistic Regression (direction prediction)");

    let mut log_model = LogisticRegression::new(num_features);
    log_model.train(train_data, 0.01, 6000);
    let log_metrics = log_model.evaluate(test_data, symbol, train_data.len());
    let log_weights = log_model.get_weights();

    println!("  │   → Direction: {:.1}% | Probability MSE: {:.4}",
             log_metrics.direction_accuracy, log_metrics.mse);

    // ── Best model ──
    let (best_acc, best_name) = if log_metrics.direction_accuracy >= lin_metrics.direction_accuracy {
        (log_metrics.direction_accuracy, "Logistic Regression")
    } else {
        (lin_metrics.direction_accuracy, "Linear Regression")
    };

    let verdict = if best_acc > 55.0 { "PROMISING" }
    else if best_acc > 50.0 { "MARGINAL" }
    else { "NO EDGE" };

    println!("  │");
    println!("  │ Best: {} ({:.1}%) — {}",
             best_name, best_acc, verdict);
    println!("  └───────────────────────\n");

    Some(PipelineResult {
        linear_metrics: lin_metrics,
        logistic_metrics: log_metrics,
        linear_weights: lin_weights,
        logistic_weights: log_weights,
        best_direction_accuracy: best_acc,
        best_model_name: best_name.to_string(),
    })
}