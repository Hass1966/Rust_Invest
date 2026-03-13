/// Random Forest Classifier — Bagging + Random Feature Subsets
/// ============================================================
/// Uses existing CART tree building from gbt.rs with:
///   - Bootstrap sampling (bagging)
///   - Random feature subset at each tree (sqrt(n_features))
///   - Majority vote / average probability for prediction
///
/// Applies same per-asset gating as LSTM/GRU in the ensemble.

use crate::gbt::{self, TreeConfig, Node};
use crate::ml::{self, Sample};

/// Random Forest configuration
#[derive(Clone, Debug)]
pub struct RandomForestConfig {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub max_features_ratio: f64, // fraction of features to consider per tree
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 5,
            min_samples_leaf: 5,
            min_samples_split: 10,
            max_features_ratio: 0.5, // sqrt(n)/n ≈ 0.1 for 104 features; 0.5 is more stable
        }
    }
}

/// A single tree in the forest with its feature subset mapping
struct ForestTree {
    tree: Node,
    feature_indices: Vec<usize>, // which features this tree uses
}

/// Random Forest Classifier
pub struct RandomForestClassifier {
    trees: Vec<ForestTree>,
    n_features: usize,
}

/// Simple deterministic pseudo-random number generator (xorshift64)
/// We avoid pulling in rand crate — this is sufficient for bootstrap sampling
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

impl RandomForestClassifier {
    /// Train a random forest on classification data
    pub fn train(
        x: &[Vec<f64>],
        y: &[f64], // 1.0 = up, 0.0 = down
        config: RandomForestConfig,
    ) -> Self {
        let n_samples = x.len();
        let n_features = if n_samples > 0 { x[0].len() } else { 0 };
        let n_select = ((n_features as f64 * config.max_features_ratio) as usize).max(3).min(n_features);

        let tree_config = TreeConfig {
            max_depth: config.max_depth,
            min_samples_leaf: config.min_samples_leaf,
            min_samples_split: config.min_samples_split,
        };

        let mut rng = SimpleRng::new(42);
        let mut trees = Vec::with_capacity(config.n_trees);

        for _t in 0..config.n_trees {
            // Bootstrap sample (sample with replacement)
            let bootstrap_indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.next_usize(n_samples))
                .collect();

            // Random feature subset for this tree
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            // Fisher-Yates partial shuffle to select n_select features
            for i in 0..n_select {
                let j = i + rng.next_usize(n_features - i);
                feature_indices.swap(i, j);
            }
            feature_indices.truncate(n_select);
            feature_indices.sort();

            // Project data to selected features
            let x_proj: Vec<Vec<f64>> = bootstrap_indices.iter().map(|&idx| {
                feature_indices.iter().map(|&fi| x[idx][fi]).collect()
            }).collect();
            let y_boot: Vec<f64> = bootstrap_indices.iter().map(|&idx| y[idx]).collect();

            // Build tree on projected data
            let all_indices: Vec<usize> = (0..x_proj.len()).collect();
            let tree = gbt::build_tree(&x_proj, &y_boot, &all_indices, &tree_config, 0);

            trees.push(ForestTree { tree, feature_indices });
        }

        RandomForestClassifier { trees, n_features }
    }

    /// Predict probability P(up) by averaging tree predictions
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        if self.trees.is_empty() { return 0.5; }

        let sum: f64 = self.trees.iter().map(|ft| {
            // Project features to this tree's subset
            let proj: Vec<f64> = ft.feature_indices.iter().map(|&fi| {
                if fi < features.len() { features[fi] } else { 0.0 }
            }).collect();
            // Tree predicts raw value (residual-like), treat as probability indicator
            let raw = ft.tree.predict(&proj);
            // Clamp to [0, 1] — trees trained on 0/1 labels predict in that range
            raw.clamp(0.0, 1.0)
        }).sum();

        (sum / self.trees.len() as f64).clamp(0.05, 0.95)
    }

    /// Predict direction
    pub fn predict_direction(&self, features: &[f64]) -> bool {
        self.predict_proba(features) > 0.5
    }
}

/// Walk-forward result for Random Forest
pub struct RFWalkForwardResult {
    pub overall_accuracy: f64,
    pub recent_accuracy: f64,
    pub final_prob: f64,
    pub n_folds: usize,
    pub total_tested: usize,
}

/// Run Random Forest walk-forward evaluation
pub fn walk_forward_rf(
    symbol: &str,
    samples: &[Sample],
    config: &RandomForestConfig,
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<RFWalkForwardResult> {
    if samples.len() < train_window + test_window + 10 {
        return None;
    }

    let n_features = samples[0].features.len();
    println!("  {} — RF walk-forward on {} samples × {} features", symbol, samples.len(), n_features);

    let mut total_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;
    let mut last_fold_correct = 0_usize;
    let mut last_fold_size = 0_usize;
    let mut last_prob = 0.5_f64;

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;
        let test_len = test_end - train_end;

        let (train_data, test_data) = fold_samples.split_at_mut(train_len);
        let (means, stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &means, &stds);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let rf = RandomForestClassifier::train(&x_train, &y_train, config.clone());

        let mut fold_correct = 0;
        for s in test_data.iter() {
            let actual_up = s.label > 0.0;
            if rf.predict_direction(&s.features) == actual_up { fold_correct += 1; }
        }

        let fold_acc = fold_correct as f64 / test_len.max(1) as f64 * 100.0;
        if n_folds < 2 || start + step + train_window + test_window > samples.len() {
            println!("    [RF] Fold {}: {}/{} correct ({:.1}%)", n_folds + 1, fold_correct, test_len, fold_acc);
        }

        total_correct += fold_correct;
        total_tested += test_len;
        n_folds += 1;
        last_fold_correct = fold_correct;
        last_fold_size = test_len;

        if let Some(last_sample) = test_data.last() {
            last_prob = rf.predict_proba(&last_sample.features).clamp(0.15, 0.85);
        }

        start += step;
    }

    if n_folds == 0 || total_tested == 0 { return None; }

    let overall_acc = total_correct as f64 / total_tested as f64 * 100.0;
    let recent_acc = last_fold_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    RF walk-forward: {} folds, {} test samples", n_folds, total_tested);
    println!("      RF: {:.1}% (recent: {:.1}%)", overall_acc, recent_acc);

    Some(RFWalkForwardResult {
        overall_accuracy: overall_acc,
        recent_accuracy: recent_acc,
        final_prob: last_prob,
        n_folds,
        total_tested,
    })
}
