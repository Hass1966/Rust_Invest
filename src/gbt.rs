/// Gradient Boosted Trees — From Scratch in Rust
/// ==============================================
/// Model 3 for Rust_Invest: non-linear classification
///
/// Fits alongside existing Linear + Logistic Regression in ml.rs
/// Uses the same Sample struct and feature pipeline.
///
/// Architecture:
///   - CART regression trees fitted to log-loss pseudo-residuals
///   - Additive boosting: F(x) = F₀ + η·Σ tree_m(x)
///   - Binary classification via sigmoid(F(x))
///   - SMA crossover features (golden/death cross) as new signals
///
/// Key Rust concepts exercised:
///   - Recursive enum + Box<Node> for tree ownership
///   - Trait implementations (Display, Default)
///   - Iterators and slice manipulation
///   - Zero unsafe code, zero external ML dependencies

use crate::analysis;
use crate::ml::{self, Sample, ModelMetrics, FEATURE_NAMES};
use std::fmt;
use rayon::prelude::*;

// ════════════════════════════════════════
// SECTION 1: Decision Tree Node
// ════════════════════════════════════════
// Rust's enum + Box is the idiomatic way to build trees.
// Each parent owns its children — no GC, no raw pointers.

/// A node in a CART decision tree.
#[derive(Debug, Clone)]
pub enum Node {
    /// Terminal node — predicts a constant value (mean of residuals)
    Leaf {
        value: f64,
        n_samples: usize,
    },
    /// Split node — routes left/right based on threshold
    Split {
        feature_idx: usize,
        threshold: f64,
        gain: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    /// Traverse the tree to predict for one sample
    pub fn predict(&self, features: &[f64]) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::Split { feature_idx, threshold, left, right, .. } => {
                if features[*feature_idx] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }

    /// Accumulate split gains per feature (for importance)
    pub fn accumulate_importance(&self, importances: &mut [f64]) {
        match self {
            Node::Leaf { .. } => {}
            Node::Split { feature_idx, gain, left, right, .. } => {
                importances[*feature_idx] += gain;
                left.accumulate_importance(importances);
                right.accumulate_importance(importances);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            Node::Leaf { .. } => 1,
            Node::Split { left, right, .. } => 1 + left.node_count() + right.node_count(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Node::Leaf { .. } => 0,
            Node::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_node(self, f, 0)
    }
}

fn fmt_node(node: &Node, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let pad = "  ".repeat(indent);
    match node {
        Node::Leaf { value, n_samples } => {
            writeln!(f, "{}Leaf(val={:.4}, n={})", pad, value, n_samples)
        }
        Node::Split { feature_idx, threshold, gain, left, right } => {
            writeln!(f, "{}Split(feat={}, thr={:.4}, gain={:.4})",
                pad, feature_idx, threshold, gain)?;
            fmt_node(left, f, indent + 1)?;
            fmt_node(right, f, indent + 1)
        }
    }
}

// ════════════════════════════════════════
// SECTION 2: Tree Builder (CART)
// ════════════════════════════════════════
// Regression trees that fit residuals. The standard approach in
// gradient boosting — trees predict continuous values even though
// the overall task is classification.

/// Config for individual trees in the ensemble
#[derive(Debug, Clone)]
pub struct TreeConfig {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            max_depth: 4,
            min_samples_leaf: 5,
            min_samples_split: 10,
        }
    }
}

/// Build a regression tree on (features, targets) indexed by `indices`.
///
/// Uses variance reduction to find the best split at each node.
/// The indices pattern lets us partition without copying data.
pub fn build_tree(
    x: &[Vec<f64>],
    y: &[f64],
    indices: &[usize],
    config: &TreeConfig,
    depth: usize,
) -> Node {
    let n = indices.len();
    let mean_y: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;

    // Base cases → leaf
    if depth >= config.max_depth || n < config.min_samples_split {
        return Node::Leaf { value: mean_y, n_samples: n };
    }

    let n_features = x[0].len();

    // Pre-compute totals for O(n) split evaluation
    let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let total_sq_sum: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();

    // ── Parallel feature scan using rayon ──
    // Each feature is evaluated independently, so we scan all 83 in parallel.
    // Each thread returns its best (gain, feature_idx, threshold, left, right).
    let best_split = (0..n_features).into_par_iter().filter_map(|feat| {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            x[a][feat].partial_cmp(&x[b][feat]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut local_best_gain = 0.0_f64;
        let mut local_best_threshold = 0.0_f64;
        let mut local_best_split_idx = 0_usize;

        let mut left_sum = 0.0_f64;
        let mut left_sq_sum = 0.0_f64;
        let mut left_count = 0_usize;

        for i in 0..sorted.len() - 1 {
            let idx = sorted[i];
            let val = y[idx];
            left_sum += val;
            left_sq_sum += val * val;
            left_count += 1;

            let right_count = n - left_count;

            if left_count < config.min_samples_leaf || right_count < config.min_samples_leaf {
                continue;
            }

            let next_idx = sorted[i + 1];
            if (x[idx][feat] - x[next_idx][feat]).abs() < 1e-12 {
                continue;
            }

            let right_sum = total_sum - left_sum;
            let right_sq_sum = total_sq_sum - left_sq_sum;

            let parent_loss = total_sq_sum - (total_sum * total_sum) / n as f64;
            let left_loss = left_sq_sum - (left_sum * left_sum) / left_count as f64;
            let right_loss = right_sq_sum - (right_sum * right_sum) / right_count as f64;

            let gain = parent_loss - left_loss - right_loss;

            if gain > local_best_gain {
                local_best_gain = gain;
                local_best_threshold = (x[idx][feat] + x[next_idx][feat]) / 2.0;
                local_best_split_idx = i;
            }
        }

        if local_best_gain > 0.0 {
            // Rebuild the left/right partition from the sorted order
            let left_indices: Vec<usize> = sorted[..=local_best_split_idx].to_vec();
            let right_indices: Vec<usize> = sorted[local_best_split_idx + 1..].to_vec();
            Some((local_best_gain, feat, local_best_threshold, left_indices, right_indices))
        } else {
            None
        }
    }).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let (best_gain, best_feature, best_threshold, best_left, best_right) = match best_split {
        Some((g, f, t, l, r)) if !l.is_empty() && !r.is_empty() => (g, f, t, l, r),
        _ => return Node::Leaf { value: mean_y, n_samples: n },
    };

    let left_node = build_tree(x, y, &best_left, config, depth + 1);
    let right_node = build_tree(x, y, &best_right, config, depth + 1);

    Node::Split {
        feature_idx: best_feature,
        threshold: best_threshold,
        gain: best_gain,
        left: Box::new(left_node),
        right: Box::new(right_node),
    }
}

// ════════════════════════════════════════
// SECTION 3: Gradient Boosted Classifier
// ════════════════════════════════════════
// Log-loss gradient boosting for binary classification.
//
// The connection to logistic regression in ml.rs:
//   - LogisticRegression fits a single linear decision boundary
//   - GBT fits a sum of step functions = piecewise-constant surface
//   - Both use sigmoid + log-loss, but GBT captures interactions
//     (e.g., "RSI < 30 AND volatility > 2%") that linear models cannot
//
// The gradient of log-loss w.r.t. F is (y - sigmoid(F)),
// identical to the logistic regression gradient — we're just
// doing gradient descent in function space rather than weight space.

/// Hyperparameters for gradient boosting
#[derive(Debug, Clone)]
pub struct GBTConfig {
    pub n_trees: usize,
    pub learning_rate: f64,
    pub tree_config: TreeConfig,
    pub subsample_ratio: f64,
    pub early_stopping_rounds: Option<usize>,
}

impl Default for GBTConfig {
    fn default() -> Self {
        GBTConfig {
            n_trees: 100,
            learning_rate: 0.1,
            tree_config: TreeConfig::default(),
            subsample_ratio: 0.8,
            early_stopping_rounds: Some(10),
        }
    }
}

/// Trained GBT model
pub struct GradientBoostedClassifier {
    pub trees: Vec<Node>,
    pub config: GBTConfig,
    pub initial_prediction: f64,  // F₀ = log(p/(1-p))
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub n_features: usize,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}

#[inline]
fn log_loss(y: f64, p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

fn mean_log_loss(y_true: &[f64], f_raw: &[f64]) -> f64 {
    y_true.iter().zip(f_raw.iter())
        .map(|(&y, &f)| log_loss(y, sigmoid(f)))
        .sum::<f64>() / y_true.len() as f64
}

impl GradientBoostedClassifier {
    /// Train on samples. Labels: 1.0 = up, 0.0 = down.
    pub fn train(
        x_train: &[Vec<f64>],
        y_train: &[f64],
        x_val: Option<&[Vec<f64>]>,
        y_val: Option<&[f64]>,
        config: GBTConfig,
    ) -> Self {
        Self::train_weighted(x_train, y_train, None, x_val, y_val, config)
    }

    pub fn train_weighted(
        x_train: &[Vec<f64>],
        y_train: &[f64],
        recency_weights: Option<&[f64]>,
        x_val: Option<&[Vec<f64>]>,
        y_val: Option<&[f64]>,
        config: GBTConfig,
    ) -> Self {
        let n_train = x_train.len();
        let n_features = x_train[0].len();

        // F₀ = log-odds of positive class base rate
        let pos_rate = y_train.iter().sum::<f64>() / n_train as f64;
        let initial_prediction = (pos_rate / (1.0 - pos_rate + 1e-15)).ln();

        let mut f_train = vec![initial_prediction; n_train];
        let mut f_val = match x_val {
            Some(xv) => vec![initial_prediction; xv.len()],
            None => Vec::new(),
        };

        let mut trees = Vec::with_capacity(config.n_trees);
        let mut train_losses = Vec::with_capacity(config.n_trees);
        let mut val_losses = Vec::with_capacity(config.n_trees);
        let mut best_val_loss = f64::MAX;
        let mut rounds_no_improve = 0_usize;

        let subsample_n = (n_train as f64 * config.subsample_ratio) as usize;

        // Class weighting to fix bullish bias
        let n_positive = y_train.iter().filter(|&&y| y > 0.5).count() as f64;
        let n_negative = n_train as f64 - n_positive;
        let weight_positive = if n_positive > 0.0 { n_train as f64 / (2.0 * n_positive) } else { 1.0 };
        let weight_negative = if n_negative > 0.0 { n_train as f64 / (2.0 * n_negative) } else { 1.0 };

        for round in 0..config.n_trees {
            // 1. Weighted pseudo-residuals: rᵢ = wᵢ * recency_wᵢ * (yᵢ - σ(Fᵢ))
            let residuals: Vec<f64> = (0..n_train)
                .map(|i| {
                    let w_class = if y_train[i] > 0.5 { weight_positive } else { weight_negative };
                    let w_recency = recency_weights.map(|ws| ws[i]).unwrap_or(1.0);
                    w_class * w_recency * (y_train[i] - sigmoid(f_train[i]))
                })
                .collect();

            // 2. Subsample (deterministic rotation for reproducibility)
            let indices: Vec<usize> = if subsample_n < n_train {
                let start = (round * 7) % n_train;
                (0..subsample_n).map(|i| (start + i) % n_train).collect()
            } else {
                (0..n_train).collect()
            };

            // 3. Fit tree to residuals
            let tree = build_tree(x_train, &residuals, &indices, &config.tree_config, 0);

            // 4. Update predictions
            for i in 0..n_train {
                f_train[i] += config.learning_rate * tree.predict(&x_train[i]);
            }
            if let Some(xv) = x_val {
                for i in 0..xv.len() {
                    f_val[i] += config.learning_rate * tree.predict(&xv[i]);
                }
            }

            // 5. Track losses
            let t_loss = mean_log_loss(y_train, &f_train);
            train_losses.push(t_loss);

            if let Some(yv) = y_val {
                let v_loss = mean_log_loss(yv, &f_val);
                val_losses.push(v_loss);

                if v_loss < best_val_loss - 1e-6 {
                    best_val_loss = v_loss;
                    rounds_no_improve = 0;
                } else {
                    rounds_no_improve += 1;
                }

                if let Some(patience) = config.early_stopping_rounds {
                    if rounds_no_improve >= patience {
                        println!("    [GBT] Early stopping at round {} (val_loss={:.4})",
                            round + 1, v_loss);
                        trees.push(tree);
                        break;
                    }
                }
            }

            trees.push(tree);

            if (round + 1) % 20 == 0 || round == 0 {
                let val_str = val_losses.last()
                    .map(|v| format!(", val={:.4}", v))
                    .unwrap_or_default();
                println!("    [GBT] Round {:3}/{}: train_loss={:.4}{}",
                    round + 1, config.n_trees, t_loss, val_str);
            }
        }

        GradientBoostedClassifier {
            trees,
            config,
            initial_prediction,
            train_losses,
            val_losses,
            n_features,
        }
    }

    /// Predict P(up) for a single feature vector
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let mut f = self.initial_prediction;
        for tree in &self.trees {
            f += self.config.learning_rate * tree.predict(features);
        }
        sigmoid(f)
    }

    /// Predict class (true = up, false = down)
    pub fn predict_direction(&self, features: &[f64]) -> bool {
        self.predict_proba(features) > 0.5
    }

    /// Feature importance — normalised gain across all trees
    pub fn feature_importance(&self, feature_names: &[&str]) -> Vec<(String, f64)> {
        let mut importances = vec![0.0_f64; self.n_features];
        for tree in &self.trees {
            tree.accumulate_importance(&mut importances);
        }
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }

        importances.iter().enumerate()
            .map(|(i, &imp)| {
                let name = if i < feature_names.len() {
                    feature_names[i].to_string()
                } else {
                    format!("Feature_{}", i)
                };
                (name, imp)
            })
            .collect()
    }

    /// Evaluate on test samples — returns ModelMetrics matching ml.rs format
    pub fn evaluate(&self, test: &[Sample], symbol: &str, train_size: usize) -> ModelMetrics {
        let mut correct = 0;
        let mut sse = 0.0;
        let mut sae = 0.0;

        for s in test {
            let predicted_up = self.predict_direction(&s.features);
            let actual_up = s.label > 0.0;
            if predicted_up == actual_up { correct += 1; }

            let prob = self.predict_proba(&s.features);
            let target = if actual_up { 1.0 } else { 0.0 };
            let err = prob - target;
            sse += err * err;
            sae += err.abs();
        }

        let n = test.len() as f64;
        ModelMetrics {
            symbol: symbol.to_string(),
            model_name: "Gradient Boosted Trees".to_string(),
            mse: sse / n,
            mae: sae / n,
            direction_accuracy: correct as f64 / n * 100.0,
            total_samples: train_size + test.len(),
            train_samples: train_size,
            test_samples: test.len(),
        }
    }

    /// Summary string
    pub fn summary(&self) -> String {
        format!(
            "{} trees, lr={}, {} total nodes, train_loss={:.4}, val_loss={}",
            self.trees.len(),
            self.config.learning_rate,
            self.trees.iter().map(|t| t.node_count()).sum::<usize>(),
            self.train_losses.last().unwrap_or(&f64::NAN),
            self.val_losses.last()
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string()),
        )
    }
}

// ════════════════════════════════════════
// SECTION 4: SMA Crossover Signals
// ════════════════════════════════════════
// Golden Cross: SMA50 crosses above SMA200 → bullish
// Death Cross:  SMA50 crosses below SMA200 → bearish

/// SMA crossover data for one time step
#[derive(Debug, Clone)]
pub struct SmaCrossoverSignal {
    pub sma_50: f64,
    pub sma_200: f64,
    pub golden_cross: bool,
    pub death_cross: bool,
    pub sma_50_above_200: bool,
    pub price_dev_sma50: f64,
    pub price_dev_sma200: f64,
    pub sma_spread: f64,
}

/// Compute SMA crossover signals using analysis::sma
pub fn compute_sma_crossover_signals(prices: &[f64]) -> Vec<(usize, SmaCrossoverSignal)> {
    let sma_50_vec = analysis::sma(prices, 50);
    let sma_200_vec = analysis::sma(prices, 200);

    let mut signals = Vec::new();

    if sma_200_vec.len() < 2 {
        return signals;
    }

    // sma_50_vec[j] corresponds to prices[j + 49]
    // sma_200_vec[j] corresponds to prices[j + 199]
    for j in 1..sma_200_vec.len() {
        let price_idx = j + 199;
        let s200 = sma_200_vec[j];
        let s200_prev = sma_200_vec[j - 1];

        let s50_idx = price_idx - 49;
        let s50_prev_idx = price_idx - 1 - 49;

        if s50_idx >= sma_50_vec.len() || s50_prev_idx >= sma_50_vec.len() {
            continue;
        }

        let s50 = sma_50_vec[s50_idx];
        let s50_prev = sma_50_vec[s50_prev_idx];

        let prev_above = s50_prev > s200_prev;
        let curr_above = s50 > s200;

        let price = prices[price_idx];

        signals.push((price_idx, SmaCrossoverSignal {
            sma_50: s50,
            sma_200: s200,
            golden_cross: !prev_above && curr_above,
            death_cross: prev_above && !curr_above,
            sma_50_above_200: curr_above,
            price_dev_sma50: (price - s50) / s50,
            price_dev_sma200: (price - s200) / s200,
            sma_spread: (s50 - s200) / s200,
        }));
    }
    signals
}

/// Convert crossover signal into 4 ML features
pub fn crossover_features(sig: &SmaCrossoverSignal) -> [f64; 4] {
    [
        if sig.sma_50_above_200 { 1.0 } else { -1.0 },
        sig.price_dev_sma50,
        sig.price_dev_sma200,
        sig.sma_spread,
    ]
}

// ════════════════════════════════════════
// SECTION 5: Extended Feature Names
// ════════════════════════════════════════

pub const GBT_FEATURE_NAMES: &[&str] = &[
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
    "SMA50>200",
    "Dev SMA50",
    "Dev SMA200",
    "SMA Spread",
];

// ════════════════════════════════════════
// SECTION 6: Extended Feature Builder
// ════════════════════════════════════════

/// Build extended features: original 10 + 4 SMA crossover = 14
/// Falls back to 10 if not enough data for SMA200
pub fn build_extended_features(
    prices: &[f64],
    volumes: &[Option<f64>],
) -> Vec<Sample> {
    let base_samples = ml::build_features(prices, volumes);

    if prices.len() < 201 {
        return base_samples;
    }

    let crossover_signals = compute_sma_crossover_signals(prices);

    let mut signal_map: std::collections::HashMap<usize, &SmaCrossoverSignal> =
        std::collections::HashMap::new();
    for (idx, sig) in &crossover_signals {
        signal_map.insert(*idx, sig);
    }

    let base_start = 33; // from ml::build_features

    let mut extended = Vec::with_capacity(base_samples.len());
    for (i, sample) in base_samples.iter().enumerate() {
        let price_idx = base_start + i;
        let mut features = sample.features.clone();

        if let Some(sig) = signal_map.get(&price_idx) {
            let cf = crossover_features(sig);
            features.extend_from_slice(&cf);
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }

        extended.push(Sample {
            features,
            label: sample.label,
        });
    }

    extended
}

// ════════════════════════════════════════
// SECTION 7: Extended Pipeline Result
// ════════════════════════════════════════

pub struct ExtendedPipelineResult {
    pub linear_metrics: ModelMetrics,
    pub logistic_metrics: ModelMetrics,
    pub gbt_metrics: ModelMetrics,
    pub linear_weights: Vec<(String, f64)>,
    pub logistic_weights: Vec<(String, f64)>,
    pub gbt_importance: Vec<(String, f64)>,
    pub best_direction_accuracy: f64,
    pub best_model_name: String,
    pub gbt_train_losses: Vec<f64>,
    pub gbt_val_losses: Vec<f64>,
}

/// Run the full 3-model pipeline for an asset
pub fn run_extended_pipeline(
    symbol: &str,
    prices: &[f64],
    volumes: &[Option<f64>],
    train_ratio: f64,
) -> Option<ExtendedPipelineResult> {
    let samples = build_extended_features(prices, volumes);

    if samples.len() < 50 {
        println!("  {} — not enough samples ({})", symbol, samples.len());
        return None;
    }

    let n_features = samples[0].features.len();
    let has_crossover = n_features > 10;

    let feature_names: &[&str] = if has_crossover {
        GBT_FEATURE_NAMES
    } else {
        FEATURE_NAMES
    };

    // Chronological split: train_ratio train, 10% val, rest test
    let split = (samples.len() as f64 * train_ratio) as usize;
    let val_split = (samples.len() as f64 * (train_ratio + 0.10)) as usize;

    // Normalise on training data, apply same transform to val+test
    let (train_slice, _) = {
        let mut temp = build_extended_features(prices, volumes);
        let (t, _) = temp.split_at_mut(split);
        ml::normalise(t)
    };
    let (_means, _stds) = (train_slice.clone(), train_slice.clone());

    // Rebuild cleanly so borrow checker is happy
    let mut temp_means = build_extended_features(prices, volumes);
    let (t_for_norm, _) = temp_means.split_at_mut(split);
    let (means, stds) = ml::normalise(t_for_norm);

    let mut all_samples = build_extended_features(prices, volumes);
    let (train_data, rest) = all_samples.split_at_mut(split);
    ml::normalise(train_data);
    ml::apply_normalisation(rest, &means, &stds);

    let val_offset = val_split.saturating_sub(split);
    let val_offset = val_offset.min(rest.len());
    let (val_data, test_data) = rest.split_at(val_offset);

    if test_data.is_empty() {
        println!("  {} — not enough test data", symbol);
        return None;
    }

    println!("\n  ┌─── {} ───", symbol);
    println!("  │ Samples: {} ({} train / {} val / {} test) × {} features{}",
        train_data.len() + val_data.len() + test_data.len(),
        train_data.len(), val_data.len(), test_data.len(),
        n_features,
        if has_crossover { " (incl. SMA crossover)" } else { "" });

    // ── Model 1: Linear Regression ──
    println!("  │");
    println!("  │ Model 1: Linear Regression");

    let mut lin_model = ml::LinearRegression::new(n_features);
    lin_model.train(train_data, 0.005, 6000);
    let lin_metrics = lin_model.evaluate(test_data, symbol, train_data.len());
    let lin_weights = lin_model.get_weights();

    println!("  │   → Direction: {:.1}% | MAE: {:.4}%",
        lin_metrics.direction_accuracy, lin_metrics.mae);

    // ── Model 2: Logistic Regression ──
    println!("  │");
    println!("  │ Model 2: Logistic Regression");

    let mut log_model = ml::LogisticRegression::new(n_features);
    log_model.train(train_data, 0.01, 6000);
    let log_metrics = log_model.evaluate(test_data, symbol, train_data.len());
    let log_weights = log_model.get_weights();

    println!("  │   → Direction: {:.1}%",
        log_metrics.direction_accuracy);

    // ── Model 3: Gradient Boosted Trees ──
    println!("  │");
    println!("  │ Model 3: Gradient Boosted Trees");

    let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
    let y_train: Vec<f64> = train_data.iter()
        .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let x_val: Vec<Vec<f64>> = val_data.iter().map(|s| s.features.clone()).collect();
    let y_val: Vec<f64> = val_data.iter()
        .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let gbt_config = GBTConfig {
        n_trees: 100,
        learning_rate: 0.08,
        tree_config: TreeConfig {
            max_depth: 4,
            min_samples_leaf: 8,
            min_samples_split: 16,
        },
        subsample_ratio: 0.8,
        early_stopping_rounds: Some(10),
    };

    let gbt_model = GradientBoostedClassifier::train(
        &x_train, &y_train,
        if !x_val.is_empty() { Some(&x_val) } else { None },
        if !y_val.is_empty() { Some(&y_val) } else { None },
        gbt_config,
    );

    let gbt_metrics = gbt_model.evaluate(test_data, symbol, train_data.len());
    let gbt_importance = gbt_model.feature_importance(feature_names);

    println!("  │   → Direction: {:.1}% | {}",
        gbt_metrics.direction_accuracy, gbt_model.summary());

    // ── Best model ──
    let accuracies = [
        (lin_metrics.direction_accuracy, "Linear Regression"),
        (log_metrics.direction_accuracy, "Logistic Regression"),
        (gbt_metrics.direction_accuracy, "Gradient Boosted Trees"),
    ];
    let (best_acc, best_name) = accuracies.iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let verdict = if *best_acc > 55.0 { "PROMISING" }
    else if *best_acc > 50.0 { "MARGINAL" }
    else { "NO EDGE" };

    println!("  │");
    println!("  │ Best: {} ({:.1}%) — {}", best_name, best_acc, verdict);
    println!("  └───────────────────────\n");

    let mut gbt_imp_sorted = gbt_importance;
    gbt_imp_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Some(ExtendedPipelineResult {
        linear_metrics: lin_metrics,
        logistic_metrics: log_metrics,
        gbt_metrics,
        linear_weights: lin_weights,
        logistic_weights: log_weights,
        gbt_importance: gbt_imp_sorted,
        best_direction_accuracy: *best_acc,
        best_model_name: best_name.to_string(),
        gbt_train_losses: gbt_model.train_losses,
        gbt_val_losses: gbt_model.val_losses,
    })
}

// ════════════════════════════════════════
// SECTION 8: Dashboard HTML for GBT
// ════════════════════════════════════════

/// Render GBT importance bars (matches report.rs style)
fn render_gbt_importance(html: &mut String, importance: &[(String, f64)]) {
    let max_imp = importance.iter()
        .map(|(_, v)| *v)
        .fold(0.0_f64, f64::max);

    for (name, imp) in importance.iter().take(14) {
        let bar_width = if max_imp > 0.0 {
            (imp / max_imp * 150.0) as u32
        } else { 0 };
        let hue = if *imp > 0.1 { 174 } else if *imp > 0.05 { 200 } else { 220 };

        html.push_str(&format!(
            "<p style='margin:3px 0;font-size:13px;'>\
             <span style='display:inline-block;width:110px;'>{}</span>\
             <span style='display:inline-block;width:55px;text-align:right;\
                    font-family:monospace;font-size:12px;'>{:.1}%</span> \
             <span class='weight-bar' style='width:{}px;\
                    background:hsl({},70%,55%);'></span></p>\n",
            name, imp * 100.0, bar_width, hue));
    }
}

/// Generate the GBT section for the HTML report
pub fn gbt_report_section(
    results: &[ExtendedPipelineResult],
    render_weights_fn: &dyn Fn(&mut String, &[(String, f64)]),
) -> String {
    let mut html = String::new();

    html.push_str("<h3>Gradient Boosted Trees</h3>\n");
    html.push_str("<p>Non-linear ensemble: shallow decision trees fitted to \
        log-loss gradients. Captures feature interactions that linear models miss.</p>\n");

    // Summary table
    html.push_str("<table>\n");
    html.push_str(
        "<tr><th>Symbol</th><th>LinReg %</th><th>LogReg %</th>\
         <th>GBT %</th><th>Best Model</th><th>Best %</th><th>Verdict</th></tr>\n");

    for r in results {
        let lin_c = acc_class(r.linear_metrics.direction_accuracy);
        let log_c = acc_class(r.logistic_metrics.direction_accuracy);
        let gbt_c = acc_class(r.gbt_metrics.direction_accuracy);
        let best_c = acc_class(r.best_direction_accuracy);
        let (badge, text) = acc_badge(r.best_direction_accuracy);

        html.push_str(&format!(
            "<tr><td>{}</td><td class='{}'>{:.1}%</td><td class='{}'>{:.1}%</td>\
             <td class='{}'>{:.1}%</td><td>{}</td><td class='{}'>{:.1}%</td>\
             <td><span class='{}'>{}</span></td></tr>\n",
            r.linear_metrics.symbol,
            lin_c, r.linear_metrics.direction_accuracy,
            log_c, r.logistic_metrics.direction_accuracy,
            gbt_c, r.gbt_metrics.direction_accuracy,
            r.best_model_name,
            best_c, r.best_direction_accuracy,
            badge, text,
        ));
    }
    html.push_str("</table>\n");

    // Feature importance per asset
    html.push_str("<h3>GBT Feature Importance</h3>\n");
    for r in results {
        if r.gbt_metrics.direction_accuracy < 50.0 { continue; }

        html.push_str(&format!(
            "<div class='card'>\n<h3>{} — GBT ({:.1}%)</h3>\n",
            r.gbt_metrics.symbol, r.gbt_metrics.direction_accuracy));

        html.push_str("<div class='model-compare'>\n");

        html.push_str("<div>\n<h3 style='font-size:14px;'>GBT Split Importance</h3>\n");
        render_gbt_importance(&mut html, &r.gbt_importance);
        html.push_str("</div>\n");

        html.push_str("<div>\n<h3 style='font-size:14px;'>Logistic Regression Weights</h3>\n");
        render_weights_fn(&mut html, &r.logistic_weights);
        html.push_str("</div>\n");

        html.push_str("</div>\n");

        html.push_str(&format!(
            "<p style='font-size:12px;color:#888;margin-top:10px;'>\
             Trees: {} | Final train loss: {:.4} | Final val loss: {}</p>\n",
            r.gbt_train_losses.len(),
            r.gbt_train_losses.last().unwrap_or(&0.0),
            r.gbt_val_losses.last()
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string()),
        ));
        html.push_str("</div>\n");
    }

    html
}

fn acc_class(acc: f64) -> &'static str {
    if acc > 55.0 { "positive" } else if acc > 50.0 { "neutral" } else { "negative" }
}

fn acc_badge(acc: f64) -> (&'static str, &'static str) {
    if acc > 55.0 { ("signal-bullish", "PROMISING") }
    else if acc > 50.0 { ("signal-neutral", "MARGINAL") }
    else { ("signal-bearish", "NO EDGE") }
}

// ════════════════════════════════════════
// TESTS
// ════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_tree_simple_split() {
        let x = vec![
            vec![0.1], vec![0.2], vec![0.3], vec![0.4],
            vec![0.6], vec![0.7], vec![0.8], vec![0.9],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let indices: Vec<usize> = (0..8).collect();
        let config = TreeConfig { max_depth: 2, min_samples_leaf: 1, min_samples_split: 2 };

        let tree = build_tree(&x, &y, &indices, &config, 0);
        assert!(tree.predict(&[0.15]) < 0.5);
        assert!(tree.predict(&[0.85]) > 0.5);
    }

    #[test]
    fn test_gbt_learns_simple_pattern() {
        let x: Vec<Vec<f64>> = vec![
            vec![0.1, 0.9], vec![0.2, 0.8], vec![0.3, 0.7],
            vec![0.7, 0.3], vec![0.8, 0.2], vec![0.9, 0.1],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = GBTConfig {
            n_trees: 50,
            learning_rate: 0.3,
            tree_config: TreeConfig { max_depth: 3, min_samples_leaf: 1, min_samples_split: 2 },
            subsample_ratio: 1.0,
            early_stopping_rounds: None,
        };

        let model = GradientBoostedClassifier::train(&x, &y, None, None, config);

        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = if model.predict_proba(xi) > 0.5 { 1.0 } else { 0.0 };
            assert_eq!(pred, yi, "Failed for {:?}", xi);
        }
    }
}
