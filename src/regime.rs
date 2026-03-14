/// Regime-Aware Ensemble — Novel Three-Layer Architecture
/// ========================================================
///
/// Layer 1 (Regime Detector):
///   K-means clustering on structural features (rolling volatility,
///   return autocorrelation, volume regime, VIX level). Outputs soft
///   probability distribution across 4 regimes:
///     - Trending-Up
///     - Trending-Down
///     - Mean-Reverting
///     - High-Volatility
///
/// Layer 2 (Specialists):
///   Separate LinReg, LogReg, and GBT instances trained on data from
///   their assigned regime only. Each sees regime-filtered training data.
///
/// Layer 3 (Adaptive Gating):
///   Final prediction = weighted sum of:
///     regime_probability × specialist_prediction × recency_accuracy
///   Recency accuracy is a rolling 20-prediction window that self-adjusts.
///
/// Implements train(), predict(), evaluate() matching the ml.rs interface.

use crate::ml::{self, Sample};
use crate::gbt::{GBTConfig, TreeConfig, GradientBoostedClassifier};

/// The 4 market regimes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regime {
    TrendingUp = 0,
    TrendingDown = 1,
    MeanReverting = 2,
    HighVolatility = 3,
}

const N_REGIMES: usize = 4;

/// Structural features used for regime detection
/// (indices into the full feature vector)
struct StructuralFeatureIndices {
    volatility_20d: usize,
    autocorr_1d: usize,
    volume_ratio_20d: usize,
    vix_level: usize,
    momentum_20d: usize,
}

/// Extract structural features for regime detection
fn structural_indices() -> StructuralFeatureIndices {
    let names = crate::features::feature_names();
    let find = |name: &str| -> usize {
        names.iter().position(|n| n == name).unwrap_or(0)
    };
    StructuralFeatureIndices {
        volatility_20d: find("volatility_20d"),
        autocorr_1d: find("autocorr_1d"),
        volume_ratio_20d: find("volume_ratio_20d"),
        vix_level: find("VIX_level"),
        momentum_20d: find("momentum_20d"),
    }
}

/// Extract 5 structural features from a sample
fn extract_structural(sample: &Sample, idx: &StructuralFeatureIndices) -> [f64; 5] {
    let f = &sample.features;
    let get = |i: usize| if i < f.len() { f[i] } else { 0.0 };
    [
        get(idx.volatility_20d),
        get(idx.autocorr_1d),
        get(idx.volume_ratio_20d),
        get(idx.vix_level),
        get(idx.momentum_20d),
    ]
}

// ════════════════════════════════════════
// Layer 1: K-Means Regime Detector
// ════════════════════════════════════════

/// K-means centroids for 4 regimes
#[derive(Debug, Clone)]
struct RegimeDetector {
    centroids: [[f64; 5]; N_REGIMES],
    /// Standard deviation per cluster per feature (for soft assignment)
    cluster_std: [[f64; 5]; N_REGIMES],
}

impl RegimeDetector {
    /// Train K-means on structural features
    fn train(samples: &[Sample]) -> Self {
        let idx = structural_indices();
        let features: Vec<[f64; 5]> = samples.iter()
            .map(|s| extract_structural(s, &idx))
            .collect();

        if features.is_empty() {
            return Self {
                centroids: [[0.0; 5]; N_REGIMES],
                cluster_std: [[1.0; 5]; N_REGIMES],
            };
        }

        // Initialize centroids with k-means++ style selection
        let mut centroids = Self::init_centroids(&features);

        // Run K-means for 50 iterations
        let mut assignments = vec![0_usize; features.len()];
        for _iter in 0..50 {
            // Assign each point to nearest centroid
            let mut changed = false;
            for (i, f) in features.iter().enumerate() {
                let nearest = Self::nearest_centroid(f, &centroids);
                if nearest != assignments[i] {
                    assignments[i] = nearest;
                    changed = true;
                }
            }
            if !changed { break; }

            // Update centroids
            for k in 0..N_REGIMES {
                let members: Vec<&[f64; 5]> = features.iter()
                    .zip(assignments.iter())
                    .filter(|(_, &a)| a == k)
                    .map(|(f, _)| f)
                    .collect();

                if !members.is_empty() {
                    for d in 0..5 {
                        centroids[k][d] = members.iter().map(|f| f[d]).sum::<f64>()
                            / members.len() as f64;
                    }
                }
            }
        }

        // Compute per-cluster standard deviation
        let mut cluster_std = [[1.0_f64; 5]; N_REGIMES];
        for k in 0..N_REGIMES {
            let members: Vec<&[f64; 5]> = features.iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == k)
                .map(|(f, _)| f)
                .collect();

            if members.len() >= 2 {
                for d in 0..5 {
                    let mean = centroids[k][d];
                    let var = members.iter()
                        .map(|f| (f[d] - mean).powi(2))
                        .sum::<f64>() / (members.len() - 1) as f64;
                    cluster_std[k][d] = var.sqrt().max(0.01);
                }
            }
        }

        // Label clusters by their characteristics
        // Sort by momentum feature to assign semantic labels
        let mut cluster_order: Vec<(usize, f64)> = (0..N_REGIMES)
            .map(|k| (k, centroids[k][4])) // momentum_20d
            .collect();
        cluster_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remap: highest momentum = TrendingUp, etc.
        let mut remapped_centroids = [[0.0; 5]; N_REGIMES];
        let mut remapped_std = [[1.0; 5]; N_REGIMES];

        // Highest vol cluster → HighVolatility
        let mut vol_scores: Vec<(usize, f64)> = (0..N_REGIMES)
            .map(|k| (k, centroids[k][0])) // volatility_20d
            .collect();
        vol_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let high_vol_k = vol_scores[0].0;
        remapped_centroids[Regime::HighVolatility as usize] = centroids[high_vol_k];
        remapped_std[Regime::HighVolatility as usize] = cluster_std[high_vol_k];

        // Among remaining, highest autocorr → MeanReverting
        let remaining: Vec<usize> = (0..N_REGIMES).filter(|&k| k != high_vol_k).collect();
        let mut autocorr_scores: Vec<(usize, f64)> = remaining.iter()
            .map(|&k| (k, centroids[k][1].abs())) // autocorr magnitude
            .collect();
        autocorr_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mean_rev_k = autocorr_scores[0].0;
        remapped_centroids[Regime::MeanReverting as usize] = centroids[mean_rev_k];
        remapped_std[Regime::MeanReverting as usize] = cluster_std[mean_rev_k];

        // Among remaining 2, higher momentum → TrendingUp, lower → TrendingDown
        let final_remaining: Vec<usize> = remaining.iter()
            .filter(|&&k| k != mean_rev_k)
            .copied()
            .collect();

        if final_remaining.len() >= 2 {
            let (k_a, k_b) = (final_remaining[0], final_remaining[1]);
            if centroids[k_a][4] >= centroids[k_b][4] {
                remapped_centroids[Regime::TrendingUp as usize] = centroids[k_a];
                remapped_std[Regime::TrendingUp as usize] = cluster_std[k_a];
                remapped_centroids[Regime::TrendingDown as usize] = centroids[k_b];
                remapped_std[Regime::TrendingDown as usize] = cluster_std[k_b];
            } else {
                remapped_centroids[Regime::TrendingUp as usize] = centroids[k_b];
                remapped_std[Regime::TrendingUp as usize] = cluster_std[k_b];
                remapped_centroids[Regime::TrendingDown as usize] = centroids[k_a];
                remapped_std[Regime::TrendingDown as usize] = cluster_std[k_a];
            }
        } else if final_remaining.len() == 1 {
            remapped_centroids[Regime::TrendingUp as usize] = centroids[final_remaining[0]];
            remapped_std[Regime::TrendingUp as usize] = cluster_std[final_remaining[0]];
        }

        Self {
            centroids: remapped_centroids,
            cluster_std: remapped_std,
        }
    }

    fn init_centroids(features: &[[f64; 5]]) -> [[f64; 5]; N_REGIMES] {
        let mut centroids = [[0.0; 5]; N_REGIMES];
        if features.is_empty() { return centroids; }

        // Pick first centroid from data
        centroids[0] = features[0];

        // Pick subsequent centroids maximizing min distance
        for k in 1..N_REGIMES {
            let mut best_idx = 0;
            let mut best_dist = 0.0_f64;
            for (i, f) in features.iter().enumerate() {
                let min_d: f64 = (0..k)
                    .map(|j| euclidean_dist(f, &centroids[j]))
                    .fold(f64::INFINITY, f64::min);
                if min_d > best_dist {
                    best_dist = min_d;
                    best_idx = i;
                }
            }
            centroids[k] = features[best_idx];
        }

        centroids
    }

    fn nearest_centroid(point: &[f64; 5], centroids: &[[f64; 5]; N_REGIMES]) -> usize {
        let mut best = 0;
        let mut best_dist = f64::INFINITY;
        for (k, c) in centroids.iter().enumerate() {
            let d = euclidean_dist(point, c);
            if d < best_dist {
                best_dist = d;
                best = k;
            }
        }
        best
    }

    /// Soft regime probabilities using Gaussian-like distance
    fn regime_probabilities(&self, sample: &Sample) -> [f64; N_REGIMES] {
        let idx = structural_indices();
        let f = extract_structural(sample, &idx);

        let mut log_probs = [0.0_f64; N_REGIMES];
        for k in 0..N_REGIMES {
            let mut log_p = 0.0;
            for d in 0..5 {
                let z = (f[d] - self.centroids[k][d]) / self.cluster_std[k][d];
                log_p -= 0.5 * z * z;
            }
            log_probs[k] = log_p;
        }

        // Softmax
        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs = [0.0_f64; N_REGIMES];
        let mut sum = 0.0;
        for k in 0..N_REGIMES {
            probs[k] = (log_probs[k] - max_lp).exp();
            sum += probs[k];
        }
        if sum > 0.0 {
            for k in 0..N_REGIMES {
                probs[k] /= sum;
            }
        } else {
            probs = [0.25; N_REGIMES];
        }

        probs
    }

    /// Hard regime assignment (for training data filtering)
    fn assign_regime(&self, sample: &Sample) -> usize {
        let probs = self.regime_probabilities(sample);
        probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k)
            .unwrap_or(0)
    }
}

fn euclidean_dist(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ════════════════════════════════════════
// Layer 2: Regime Specialists
// ════════════════════════════════════════

/// A specialist trained on regime-filtered data
struct RegimeSpecialist {
    linreg: ml::LinearRegression,
    logreg: ml::LogisticRegression,
    gbt: Option<GradientBoostedClassifier>,
    n_training_samples: usize,
}

impl RegimeSpecialist {
    fn train(regime_samples: &[Sample], n_features: usize) -> Self {
        let min_samples = 30;

        if regime_samples.len() < min_samples {
            // Not enough data — return dummy specialist
            return Self {
                linreg: ml::LinearRegression::new(n_features),
                logreg: ml::LogisticRegression::new(n_features),
                gbt: None,
                n_training_samples: regime_samples.len(),
            };
        }

        let mut train_data = regime_samples.to_vec();
        let (_means, _stds) = ml::normalise(&mut train_data);

        let mut linreg = ml::LinearRegression::new(n_features);
        linreg.train(&train_data, 0.005, 2000);

        let mut logreg = ml::LogisticRegression::new(n_features);
        logreg.train(&train_data, 0.01, 2000);

        let gbt = if regime_samples.len() >= 60 {
            let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
            let y_train: Vec<f64> = train_data.iter()
                .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
            let val_start = (x_train.len() as f64 * 0.85) as usize;
            let (x_t, x_v) = x_train.split_at(val_start);
            let (y_t, y_v) = y_train.split_at(val_start);

            let config = GBTConfig {
                n_trees: 50,
                learning_rate: 0.08,
                tree_config: TreeConfig {
                    max_depth: 3,
                    min_samples_leaf: 6,
                    min_samples_split: 12,
                },
                subsample_ratio: 0.8,
                early_stopping_rounds: Some(6),
            };

            Some(GradientBoostedClassifier::train(x_t, y_t, Some(x_v), Some(y_v), config))
        } else {
            None
        };

        Self {
            linreg,
            logreg,
            gbt,
            n_training_samples: regime_samples.len(),
        }
    }

    /// Predict P(up) — average of available models
    fn predict_proba(&self, features: &[f64]) -> f64 {
        if self.n_training_samples < 30 {
            return 0.5; // no signal
        }

        let raw_lin = self.linreg.predict(features);
        let lin_prob = 1.0 / (1.0 + (-raw_lin).exp());
        let log_prob = self.logreg.predict_probability(features);

        if let Some(ref gbt) = self.gbt {
            let gbt_prob = gbt.predict_proba(features);
            (lin_prob + log_prob + gbt_prob) / 3.0
        } else {
            (lin_prob + log_prob) / 2.0
        }
    }
}

// ════════════════════════════════════════
// Layer 3: Adaptive Gating + Full Model
// ════════════════════════════════════════

/// The complete Regime-Aware Ensemble model
pub struct RegimeEnsemble {
    detector: RegimeDetector,
    specialists: [Option<RegimeSpecialist>; N_REGIMES],
    /// Rolling accuracy per specialist (for adaptive gating)
    recency_accuracy: [f64; N_REGIMES],
    n_features: usize,
}

impl RegimeEnsemble {
    /// Train the full regime-aware ensemble
    pub fn train(samples: &[Sample]) -> Self {
        let n_features = if samples.is_empty() { 0 } else { samples[0].features.len() };

        // Layer 1: Train regime detector
        let detector = RegimeDetector::train(samples);

        // Assign each sample to its primary regime
        let assignments: Vec<usize> = samples.iter()
            .map(|s| detector.assign_regime(s))
            .collect();

        // Layer 2: Train specialists on regime-filtered data
        let mut specialists: [Option<RegimeSpecialist>; N_REGIMES] = [None, None, None, None];
        let regime_names = ["TrendingUp", "TrendingDown", "MeanReverting", "HighVolatility"];

        for k in 0..N_REGIMES {
            let regime_samples: Vec<Sample> = samples.iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == k)
                .map(|(s, _)| s.clone())
                .collect();

            println!("    [REGIME] {} — {} samples", regime_names[k], regime_samples.len());

            if regime_samples.len() >= 20 {
                specialists[k] = Some(RegimeSpecialist::train(&regime_samples, n_features));
            }
        }

        // Layer 3: Initialize recency accuracy (start at 50% = no information)
        let recency_accuracy = [0.5; N_REGIMES];

        Self {
            detector,
            specialists,
            recency_accuracy,
            n_features,
        }
    }

    /// Predict P(up) using regime-weighted specialists
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let sample = Sample { features: features.to_vec(), label: 0.0 };
        let regime_probs = self.detector.regime_probabilities(&sample);

        let mut weighted_pred = 0.0;
        let mut total_weight = 0.0;

        for k in 0..N_REGIMES {
            if let Some(ref specialist) = self.specialists[k] {
                let specialist_pred = specialist.predict_proba(features);
                let weight = regime_probs[k] * self.recency_accuracy[k];
                weighted_pred += weight * specialist_pred;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            (weighted_pred / total_weight).clamp(0.15, 0.85)
        } else {
            0.5
        }
    }

    /// Predict direction
    pub fn predict_direction(&self, features: &[f64]) -> bool {
        self.predict_proba(features) > 0.5
    }

    /// Update recency accuracy after observing outcomes.
    /// Uses exponential moving average over a rolling window.
    pub fn update_recency(&mut self, features: &[f64], actual_up: bool) {
        let sample = Sample { features: features.to_vec(), label: 0.0 };
        let regime_probs = self.detector.regime_probabilities(&sample);

        let predicted_up = self.predict_proba(features) > 0.5;
        let correct = predicted_up == actual_up;
        let correct_f = if correct { 1.0 } else { 0.0 };

        // Update each regime's recency accuracy weighted by its probability
        let alpha = 0.05; // EMA decay (effectively ~20-sample window)
        for k in 0..N_REGIMES {
            if regime_probs[k] > 0.1 {
                self.recency_accuracy[k] = (1.0 - alpha) * self.recency_accuracy[k]
                    + alpha * correct_f;
            }
        }
    }

    /// Evaluate on a test set with recency tracking
    pub fn evaluate(&mut self, test_samples: &[Sample]) -> f64 {
        let mut correct = 0_usize;

        for s in test_samples {
            let actual_up = s.label > 0.0;
            let predicted_up = self.predict_direction(&s.features);
            if predicted_up == actual_up { correct += 1; }
            self.update_recency(&s.features, actual_up);
        }

        if test_samples.is_empty() { return 50.0; }
        correct as f64 / test_samples.len() as f64 * 100.0
    }

    /// Get current regime probabilities for the latest sample
    pub fn current_regime(&self, features: &[f64]) -> [f64; N_REGIMES] {
        let sample = Sample { features: features.to_vec(), label: 0.0 };
        self.detector.regime_probabilities(&sample)
    }

    /// Get human-readable regime name
    pub fn regime_name(idx: usize) -> &'static str {
        match idx {
            0 => "Trending Up",
            1 => "Trending Down",
            2 => "Mean-Reverting",
            3 => "High Volatility",
            _ => "Unknown",
        }
    }

    /// Get the dominant regime
    pub fn dominant_regime(&self, features: &[f64]) -> (usize, &'static str, f64) {
        let probs = self.current_regime(features);
        let (idx, &prob) = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.25));
        (idx, Self::regime_name(idx), prob)
    }
}

// ════════════════════════════════════════
// Walk-Forward Evaluation
// ════════════════════════════════════════

pub struct RegimeWalkForwardResult {
    pub overall_accuracy: f64,
    pub recent_accuracy: f64,
    pub final_prob: f64,
    pub regime_probs: [f64; N_REGIMES],
    pub n_folds: usize,
    pub total_tested: usize,
}

/// Run walk-forward evaluation for the regime ensemble
pub fn walk_forward_regime(
    symbol: &str,
    samples: &[Sample],
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<RegimeWalkForwardResult> {
    if samples.len() < train_window + test_window + 10 {
        println!("  {} — not enough samples for regime walk-forward", symbol);
        return None;
    }

    println!("  {} — Regime ensemble walk-forward on {} samples", symbol, samples.len());

    let mut total_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0;
    let mut last_fold_correct = 0;
    let mut last_fold_size = 0;
    let mut last_prob = 0.5;
    let mut last_regime_probs = [0.25; N_REGIMES];

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        let train_data = &samples[start..train_end];
        let test_data = &samples[train_end..test_end];

        let mut model = RegimeEnsemble::train(train_data);
        let fold_acc = model.evaluate(test_data);

        let fold_correct = (fold_acc / 100.0 * test_data.len() as f64).round() as usize;
        total_correct += fold_correct;
        total_tested += test_data.len();
        n_folds += 1;
        last_fold_correct = fold_correct;
        last_fold_size = test_data.len();

        if let Some(last_sample) = test_data.last() {
            last_prob = model.predict_proba(&last_sample.features);
            last_regime_probs = model.current_regime(&last_sample.features);
        }

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        return None;
    }

    let overall_acc = total_correct as f64 / total_tested as f64 * 100.0;
    let recent_acc = last_fold_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    Regime ensemble: {:.1}% (recent: {:.1}%)", overall_acc, recent_acc);

    Some(RegimeWalkForwardResult {
        overall_accuracy: overall_acc,
        recent_accuracy: recent_acc,
        final_prob: last_prob,
        regime_probs: last_regime_probs,
        n_folds,
        total_tested,
    })
}
