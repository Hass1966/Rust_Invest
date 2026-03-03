/// Diagnostics Module — Deep Ensemble Analysis
/// =============================================
/// Runs a diagnostic walk-forward that collects per-fold metrics
/// your main walk-forward doesn't capture:
///
///   1. Per-fold accuracy per model (not just the average)
///   2. Confusion matrix per model (TP, FP, TN, FN)
///   3. Bullish bias detection (does a model always predict UP?)
///   4. Feature importance ranking from GBT
///   5. Model contribution analysis (who helps, who drags)
///
/// Usage:
///   let diag = diagnostics::run_diagnostics(symbol, &samples, train_window, test_window, step);
///   diagnostics::print_diagnostics(&diag);           // console output
///   let html = diagnostics::diagnostics_html(&[diag]); // for report
///
/// This runs SEPARATELY from your production ensemble — it doesn't
/// change any signal logic. Think of it as an X-ray of your models.

use crate::ml::{self, Sample};
use crate::gbt::{self, GBTConfig, TreeConfig, GradientBoostedClassifier};
use crate::features;

// ════════════════════════════════════════
// Data structures
// ════════════════════════════════════════

/// Build the right feature name list based on the actual feature count.
/// - 83 features = rich feature set from features.rs (stocks)
/// - 14 features = basic from ml.rs (crypto fallback)
/// - 18 features = basic + 4 crossover features from gbt.rs (crypto)
/// - anything else = generic numbered names
fn build_feature_names(n_features: usize) -> Vec<String> {
    let rich = features::feature_names();
    if n_features == rich.len() {
        return rich;
    }

    // Basic features from ml.rs (10 base features)
    let basic: Vec<String> = crate::ml::FEATURE_NAMES.iter()
        .map(|s| s.to_string())
        .collect();

    if n_features == basic.len() {
        return basic;
    }

    // Extended basic: 10 base + 4 SMA crossover features
    if n_features == basic.len() + 4 {
        let mut names = basic;
        names.push("SMA_cross_signal".into());
        names.push("SMA_cross_strength".into());
        names.push("SMA_cross_days_since".into());
        names.push("SMA_cross_confirmed".into());
        return names;
    }

    // Fallback: generic numbered names
    (0..n_features).map(|i| format!("Feature_{}", i)).collect()
}


/// Confusion matrix for a binary classifier
#[derive(Debug, Clone, Default)]
pub struct ConfusionMatrix {
    pub tp: usize, // predicted UP, actual UP
    pub fp: usize, // predicted UP, actual DOWN
    pub tn: usize, // predicted DOWN, actual DOWN
    pub fn_: usize, // predicted DOWN, actual UP (fn is reserved keyword)
}

impl ConfusionMatrix {
    pub fn total(&self) -> usize { self.tp + self.fp + self.tn + self.fn_ }
    pub fn accuracy(&self) -> f64 {
        let t = self.total();
        if t == 0 { return 0.0; }
        (self.tp + self.tn) as f64 / t as f64 * 100.0
    }
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { return 0.0; }
        self.tp as f64 / denom as f64 * 100.0
    }
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 { return 0.0; }
        self.tp as f64 / denom as f64 * 100.0
    }
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { return 0.0; }
        2.0 * p * r / (p + r)
    }
    /// Bullish bias = % of all predictions that were UP
    pub fn bullish_rate(&self) -> f64 {
        let t = self.total();
        if t == 0 { return 0.0; }
        (self.tp + self.fp) as f64 / t as f64 * 100.0
    }
    /// Actual UP rate (base rate in the data)
    pub fn actual_up_rate(&self) -> f64 {
        let t = self.total();
        if t == 0 { return 0.0; }
        (self.tp + self.fn_) as f64 / t as f64 * 100.0
    }

    fn merge(&mut self, other: &ConfusionMatrix) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.tn += other.tn;
        self.fn_ += other.fn_;
    }
}

/// Per-fold metrics for one model
#[derive(Debug, Clone)]
pub struct FoldMetric {
    pub fold_idx: usize,
    pub accuracy: f64,
    pub test_size: usize,
    pub cm: ConfusionMatrix,
}

/// Complete diagnostics for one symbol
#[derive(Debug, Clone)]
pub struct SymbolDiagnostics {
    pub symbol: String,
    pub n_features: usize,
    pub n_folds: usize,
    pub total_samples: usize,

    // Per-model fold-by-fold metrics
    pub linear_folds: Vec<FoldMetric>,
    pub logistic_folds: Vec<FoldMetric>,
    pub gbt_folds: Vec<FoldMetric>,

    // Aggregate confusion matrices
    pub linear_cm: ConfusionMatrix,
    pub logistic_cm: ConfusionMatrix,
    pub gbt_cm: ConfusionMatrix,

    // Overall accuracies
    pub linear_accuracy: f64,
    pub logistic_accuracy: f64,
    pub gbt_accuracy: f64,

    // GBT feature importance (sorted descending)
    pub feature_importance: Vec<(String, f64)>,

    // Model contribution: does removing this model improve the ensemble?
    pub ensemble_accuracy: f64,         // all 3 models voting
    pub accuracy_without_linear: f64,   // LogReg + GBT only
    pub accuracy_without_logistic: f64, // LinReg + GBT only
    pub accuracy_without_gbt: f64,      // LinReg + LogReg only

    // Data balance
    pub actual_up_pct: f64,  // what % of test days were actually UP?

    // Per-fold ensemble accuracy for stability check
    pub ensemble_fold_accuracies: Vec<f64>,
}

// ════════════════════════════════════════
// Diagnostic walk-forward
// ════════════════════════════════════════

/// Run the full diagnostic walk-forward on pre-built samples.
/// This mirrors walk_forward_samples() but collects everything.
pub fn run_diagnostics(
    symbol: &str,
    samples: &[Sample],
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<SymbolDiagnostics> {
    if samples.len() < train_window + test_window + 10 {
        println!("  [DIAG] {} — not enough samples ({})", symbol, samples.len());
        return None;
    }

    let n_features = samples[0].features.len();
    println!("\n  ┌─── DIAGNOSTICS: {} ─── {} samples × {} features", symbol, samples.len(), n_features);

    let mut linear_folds = Vec::new();
    let mut logistic_folds = Vec::new();
    let mut gbt_folds = Vec::new();

    let mut linear_cm = ConfusionMatrix::default();
    let mut logistic_cm = ConfusionMatrix::default();
    let mut gbt_cm = ConfusionMatrix::default();

    // For ensemble contribution analysis
    let mut ensemble_correct = 0_usize;
    let mut without_lin_correct = 0_usize;
    let mut without_log_correct = 0_usize;
    let mut without_gbt_correct = 0_usize;
    let mut total_tested = 0_usize;

    let mut ensemble_fold_accuracies = Vec::new();

    // For feature importance: accumulate across folds then average
    let mut importance_accum: Vec<f64> = vec![0.0; n_features];
    let mut n_folds = 0_usize;

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        // Clone and normalise this fold
        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;
        let test_len = test_end - train_end;

        let (train_data, test_data) = fold_samples.split_at_mut(train_len);
        let (means, stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &means, &stds);

        // ── Train all 3 models (same as your ensemble.rs) ──
        let mut lin = ml::LinearRegression::new(n_features);
        lin.train(train_data, 0.005, 3000);

        let mut log = ml::LogisticRegression::new(n_features);
        log.train(train_data, 0.01, 3000);

        let x_train: Vec<Vec<f64>> = train_data.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_data.iter()
            .map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let val_start = (x_train.len() as f64 * 0.85) as usize;
        let (x_t, x_v) = x_train.split_at(val_start);
        let (y_t, y_v) = y_train.split_at(val_start);

        let gbt_config = GBTConfig {
            n_trees: 80,
            learning_rate: 0.08,
            tree_config: TreeConfig {
                max_depth: 4,
                min_samples_leaf: 8,
                min_samples_split: 16,
            },
            subsample_ratio: 0.8,
            early_stopping_rounds: Some(8),
        };

        let gbt = GradientBoostedClassifier::train(
            x_t, y_t, Some(x_v), Some(y_v), gbt_config,
        );

        // ── Evaluate with full confusion matrix tracking ──
        let mut fold_lin_cm = ConfusionMatrix::default();
        let mut fold_log_cm = ConfusionMatrix::default();
        let mut fold_gbt_cm = ConfusionMatrix::default();
        let mut fold_ensemble_correct = 0_usize;
        let mut fold_no_lin = 0_usize;
        let mut fold_no_log = 0_usize;
        let mut fold_no_gbt = 0_usize;

        for s in test_data.iter() {
            let actual_up = s.label > 0.0;

            let lin_up = lin.predict(&s.features) > 0.0;
            let log_up = log.predict_direction(&s.features);
            let gbt_up = gbt.predict_direction(&s.features);

            // Linear confusion matrix
            match (lin_up, actual_up) {
                (true, true) => fold_lin_cm.tp += 1,
                (true, false) => fold_lin_cm.fp += 1,
                (false, false) => fold_lin_cm.tn += 1,
                (false, true) => fold_lin_cm.fn_ += 1,
            }

            // Logistic confusion matrix
            match (log_up, actual_up) {
                (true, true) => fold_log_cm.tp += 1,
                (true, false) => fold_log_cm.fp += 1,
                (false, false) => fold_log_cm.tn += 1,
                (false, true) => fold_log_cm.fn_ += 1,
            }

            // GBT confusion matrix
            match (gbt_up, actual_up) {
                (true, true) => fold_gbt_cm.tp += 1,
                (true, false) => fold_gbt_cm.fp += 1,
                (false, false) => fold_gbt_cm.tn += 1,
                (false, true) => fold_gbt_cm.fn_ += 1,
            }

            // Majority vote ensemble (simple: 2 of 3 agree)
            let votes_up = [lin_up, log_up, gbt_up].iter().filter(|&&v| v).count();
            let ensemble_up = votes_up >= 2;
            if ensemble_up == actual_up { fold_ensemble_correct += 1; }

            // Leave-one-out ensembles
            let no_lin_up = [log_up, gbt_up].iter().filter(|&&v| v).count() >= 1;
            // With 2 models, tie = up (mirrors your 50/50 handling)
            let no_log_up = [lin_up, gbt_up].iter().filter(|&&v| v).count() >= 1;
            let no_gbt_up = [lin_up, log_up].iter().filter(|&&v| v).count() >= 1;

            if no_lin_up == actual_up { fold_no_lin += 1; }
            if no_log_up == actual_up { fold_no_log += 1; }
            if no_gbt_up == actual_up { fold_no_gbt += 1; }
        }

        // Record fold metrics
        let fold_idx = n_folds;

        linear_folds.push(FoldMetric {
            fold_idx, accuracy: fold_lin_cm.accuracy(), test_size: test_len,
            cm: fold_lin_cm.clone(),
        });
        logistic_folds.push(FoldMetric {
            fold_idx, accuracy: fold_log_cm.accuracy(), test_size: test_len,
            cm: fold_log_cm.clone(),
        });
        gbt_folds.push(FoldMetric {
            fold_idx, accuracy: fold_gbt_cm.accuracy(), test_size: test_len,
            cm: fold_gbt_cm.clone(),
        });

        let fold_ens_acc = fold_ensemble_correct as f64 / test_len as f64 * 100.0;
        ensemble_fold_accuracies.push(fold_ens_acc);

        // Merge into aggregate confusion matrices
        linear_cm.merge(&fold_lin_cm);
        logistic_cm.merge(&fold_log_cm);
        gbt_cm.merge(&fold_gbt_cm);

        ensemble_correct += fold_ensemble_correct;
        without_lin_correct += fold_no_lin;
        without_log_correct += fold_no_log;
        without_gbt_correct += fold_no_gbt;
        total_tested += test_len;

        // Accumulate GBT feature importance
        // Use rich feature names if 83 features, otherwise generate basic names
        let feat_names = build_feature_names(n_features);
        let feat_refs: Vec<&str> = feat_names.iter().map(|s| s.as_str()).collect();
        let fold_importance = gbt.feature_importance(&feat_refs);
        for (i, (_name, imp)) in fold_importance.iter().enumerate() {
            if i < importance_accum.len() {
                importance_accum[i] += imp;
            }
        }

        n_folds += 1;
        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        println!("  └─── No folds completed");
        return None;
    }

    // Average feature importance across folds
    let feat_names = build_feature_names(n_features);
    let mut feature_importance: Vec<(String, f64)> = feat_names.iter().enumerate()
        .map(|(i, name)| {
            let avg = if i < importance_accum.len() { importance_accum[i] / n_folds as f64 } else { 0.0 };
            (name.clone(), avg)
        })
        .collect();
    feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let lin_acc = linear_cm.accuracy();
    let log_acc = logistic_cm.accuracy();
    let gbt_acc = gbt_cm.accuracy();

    let ens_acc = ensemble_correct as f64 / total_tested as f64 * 100.0;
    let no_lin_acc = without_lin_correct as f64 / total_tested as f64 * 100.0;
    let no_log_acc = without_log_correct as f64 / total_tested as f64 * 100.0;
    let no_gbt_acc = without_gbt_correct as f64 / total_tested as f64 * 100.0;

    let actual_up_pct = linear_cm.actual_up_rate(); // same base rate for all models

    println!("  │ {} folds, {} test samples", n_folds, total_tested);
    println!("  └───────────────────────\n");

    Some(SymbolDiagnostics {
        symbol: symbol.to_string(),
        n_features,
        n_folds,
        total_samples: total_tested,
        linear_folds,
        logistic_folds,
        gbt_folds,
        linear_cm,
        logistic_cm,
        gbt_cm,
        linear_accuracy: lin_acc,
        logistic_accuracy: log_acc,
        gbt_accuracy: gbt_acc,
        feature_importance,
        ensemble_accuracy: ens_acc,
        accuracy_without_linear: no_lin_acc,
        accuracy_without_logistic: no_log_acc,
        accuracy_without_gbt: no_gbt_acc,
        actual_up_pct,
        ensemble_fold_accuracies,
    })
}

// ════════════════════════════════════════
// Console output — detailed diagnostics
// ════════════════════════════════════════

pub fn print_diagnostics(diag: &SymbolDiagnostics) {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  DIAGNOSTICS: {:<51}║", diag.symbol);
    println!("║  {} features, {} folds, {} test samples{:>20}║",
        diag.n_features, diag.n_folds, diag.total_samples, "");
    println!("╠═══════════════════════════════════════════════════════════════════╣");

    // ── Data balance ──
    println!("║                                                                   ║");
    println!("║  DATA BALANCE: {:.1}% of test days were UP                        ║", diag.actual_up_pct);
    if (diag.actual_up_pct - 50.0).abs() > 5.0 {
        println!("║  ⚠ IMBALANCED: >5% skew — raw accuracy may be misleading         ║");
    }

    // ── Per-fold accuracy ──
    println!("║                                                                   ║");
    println!("║  PER-FOLD ACCURACY:                                               ║");
    println!("║  {:>5} {:>8} {:>8} {:>8} {:>8} {:>6}                       ║",
        "Fold", "LinReg", "LogReg", "GBT", "Ensemb", "N");
    println!("║  ───── ──────── ──────── ──────── ──────── ──────                  ║");

    for i in 0..diag.n_folds {
        let lin = &diag.linear_folds[i];
        let log = &diag.logistic_folds[i];
        let gbt = &diag.gbt_folds[i];
        let ens = diag.ensemble_fold_accuracies[i];
        println!("║  {:>5} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>6}                  ║",
            i + 1, lin.accuracy, log.accuracy, gbt.accuracy, ens, lin.test_size);
    }

    // ── Overall accuracy ──
    println!("║                                                                   ║");
    println!("║  OVERALL:                                                          ║");
    println!("║    LinReg:   {:.1}%                                              ║", diag.linear_accuracy);
    println!("║    LogReg:   {:.1}%                                              ║", diag.logistic_accuracy);
    println!("║    GBT:      {:.1}%                                              ║", diag.gbt_accuracy);
    println!("║    Ensemble: {:.1}%                                              ║", diag.ensemble_accuracy);

    // ── Confusion matrices ──
    println!("║                                                                   ║");
    println!("║  CONFUSION MATRICES (aggregate across all folds):                  ║");
    print_cm("LinReg ", &diag.linear_cm);
    print_cm("LogReg ", &diag.logistic_cm);
    print_cm("GBT    ", &diag.gbt_cm);

    // ── Bullish bias ──
    println!("║                                                                   ║");
    println!("║  BULLISH BIAS CHECK:                                              ║");
    println!("║    Actual UP rate:     {:.1}%                                    ║", diag.actual_up_pct);
    println!("║    LinReg predicts UP: {:.1}%  {}                              ║",
        diag.linear_cm.bullish_rate(), bias_verdict(diag.linear_cm.bullish_rate(), diag.actual_up_pct));
    println!("║    LogReg predicts UP: {:.1}%  {}                              ║",
        diag.logistic_cm.bullish_rate(), bias_verdict(diag.logistic_cm.bullish_rate(), diag.actual_up_pct));
    println!("║    GBT predicts UP:    {:.1}%  {}                              ║",
        diag.gbt_cm.bullish_rate(), bias_verdict(diag.gbt_cm.bullish_rate(), diag.actual_up_pct));

    // ── Model contribution ──
    println!("║                                                                   ║");
    println!("║  MODEL CONTRIBUTION (does removing a model hurt or help?):         ║");
    println!("║    Full ensemble:           {:.1}%                               ║", diag.ensemble_accuracy);
    println!("║    Without LinReg:          {:.1}%  ({})                   ║",
        diag.accuracy_without_linear,
        contribution_verdict(diag.ensemble_accuracy, diag.accuracy_without_linear));
    println!("║    Without LogReg:          {:.1}%  ({})                   ║",
        diag.accuracy_without_logistic,
        contribution_verdict(diag.ensemble_accuracy, diag.accuracy_without_logistic));
    println!("║    Without GBT:             {:.1}%  ({})                   ║",
        diag.accuracy_without_gbt,
        contribution_verdict(diag.ensemble_accuracy, diag.accuracy_without_gbt));

    // ── Top 20 features ──
    println!("║                                                                   ║");
    println!("║  TOP 20 FEATURES (GBT importance, averaged across folds):          ║");
    println!("║  {:>4} {:<30} {:>8}                            ║", "Rank", "Feature", "Weight");
    println!("║  ──── ────────────────────────── ────────                           ║");
    for (i, (name, imp)) in diag.feature_importance.iter().take(20).enumerate() {
        let bar_len = (imp * 200.0).min(20.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("║  {:>4} {:<30} {:>7.4}  {}              ║",
            i + 1, name, imp, bar);
    }

    // ── Bottom 10 features (candidates for pruning) ──
    println!("║                                                                   ║");
    println!("║  BOTTOM 10 FEATURES (prune candidates):                            ║");
    let total_features = diag.feature_importance.len();
    for (name, imp) in diag.feature_importance.iter().rev().take(10) {
        println!("║    {:<30} {:>7.4}  {}                       ║",
            name, imp, if *imp < 0.005 { "← PRUNE" } else { "" });
    }

    println!("║                                                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");
}

fn print_cm(label: &str, cm: &ConfusionMatrix) {
    println!("║    {}:  TP={:>4}  FP={:>4}  TN={:>4}  FN={:>4}                 ║",
        label, cm.tp, cm.fp, cm.tn, cm.fn_);
    println!("║    {}   Prec={:.1}%  Recall={:.1}%  F1={:.1}%                 ║",
        "       ", cm.precision(), cm.recall(), cm.f1());
}

fn bias_verdict(predicted_up: f64, actual_up: f64) -> &'static str {
    let diff = predicted_up - actual_up;
    if diff > 15.0 { "⚠ HEAVY BULL BIAS" }
    else if diff > 8.0 { "⚠ BULL BIAS" }
    else if diff < -15.0 { "⚠ HEAVY BEAR BIAS" }
    else if diff < -8.0 { "⚠ BEAR BIAS" }
    else { "✓ Balanced" }
}

fn contribution_verdict(with: f64, without: f64) -> &'static str {
    let diff = with - without;
    if diff > 1.5 { "HELPS — keep it" }
    else if diff > 0.0 { "marginal help" }
    else if diff > -1.5 { "marginal drag" }
    else { "DRAGS — consider removing" }
}

// ════════════════════════════════════════
// HTML Diagnostic Report
// ════════════════════════════════════════

pub fn diagnostics_html(diagnostics: &[SymbolDiagnostics]) -> String {
    let mut html = String::new();

    html.push_str("<section id='diagnostics'>\n");
    html.push_str("<h2 class='section-title'><span>//</span> Model Diagnostics — Deep Analysis</h2>\n");
    html.push_str("<p style='color:var(--text-dim);font-size:13px;margin-bottom:20px;'>\
        Per-fold accuracy, confusion matrices, bias detection, feature importance, \
        and model contribution analysis. Use this to identify what to fix.</p>\n");

    for diag in diagnostics {
        html.push_str(&format!("<div class='card' style='margin-bottom:24px;'>\n"));
        html.push_str(&format!("<h3 style='font-size:18px;margin-bottom:12px;'>{} — {} features, {} folds, {} test samples</h3>\n",
            diag.symbol, diag.n_features, diag.n_folds, diag.total_samples));

        // ── Data balance warning ──
        if (diag.actual_up_pct - 50.0).abs() > 5.0 {
            html.push_str(&format!(
                "<div style='background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);border-radius:8px;padding:10px;margin-bottom:12px;font-size:12px;'>\
                 ⚠ <strong>Data Imbalance:</strong> {:.1}% of test days were UP (ideal ≈ 50%). \
                 Raw accuracy may overstate skill if models lean toward the majority class.</div>\n",
                diag.actual_up_pct));
        }

        // ── Per-fold accuracy table ──
        html.push_str("<h3>Per-Fold Accuracy</h3>\n");
        html.push_str("<table><thead><tr>\
            <th style='text-align:left;'>Fold</th><th>LinReg</th><th>LogReg</th><th>GBT</th><th>Ensemble</th><th>N</th>\
            </tr></thead><tbody>\n");
        for i in 0..diag.n_folds {
            let lin = &diag.linear_folds[i];
            let log = &diag.logistic_folds[i];
            let gbt = &diag.gbt_folds[i];
            let ens = diag.ensemble_fold_accuracies[i];
            html.push_str(&format!(
                "<tr><td style='text-align:left;'>Fold {}</td><td style='color:{};'>{:.1}%</td><td style='color:{};'>{:.1}%</td>\
                 <td style='color:{};'>{:.1}%</td><td style='color:{};font-weight:bold;'>{:.1}%</td><td>{}</td></tr>\n",
                i + 1,
                acc_color(lin.accuracy), lin.accuracy,
                acc_color(log.accuracy), log.accuracy,
                acc_color(gbt.accuracy), gbt.accuracy,
                acc_color(ens), ens,
                lin.test_size));
        }
        // Overall row
        html.push_str(&format!(
            "<tr style='border-top:2px solid var(--border);font-weight:bold;'>\
             <td style='text-align:left;'>OVERALL</td><td style='color:{};'>{:.1}%</td>\
             <td style='color:{};'>{:.1}%</td><td style='color:{};'>{:.1}%</td>\
             <td style='color:{};'>{:.1}%</td><td>{}</td></tr>\n",
            acc_color(diag.linear_accuracy), diag.linear_accuracy,
            acc_color(diag.logistic_accuracy), diag.logistic_accuracy,
            acc_color(diag.gbt_accuracy), diag.gbt_accuracy,
            acc_color(diag.ensemble_accuracy), diag.ensemble_accuracy,
            diag.total_samples));
        html.push_str("</tbody></table>\n");

        // ── Fold accuracy chart (inline SVG bar chart) ──
        html.push_str(&fold_accuracy_chart(diag));

        // ── Confusion matrices ──
        html.push_str("<h3>Confusion Matrices (Aggregate)</h3>\n");
        html.push_str("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;'>\n");
        html.push_str(&cm_card("Linear Regression", &diag.linear_cm));
        html.push_str(&cm_card("Logistic Regression", &diag.logistic_cm));
        html.push_str(&cm_card("Gradient Boosted Trees", &diag.gbt_cm));
        html.push_str("</div>\n");

        // ── Bullish bias ──
        html.push_str("<h3>Bullish Bias Detection</h3>\n");
        html.push_str(&bias_table(diag));

        // ── Model contribution ──
        html.push_str("<h3>Model Contribution Analysis</h3>\n");
        html.push_str(&contribution_table(diag));

        // ── Feature importance ──
        html.push_str("<h3>Feature Importance (GBT, averaged across folds)</h3>\n");
        html.push_str(&feature_importance_chart(diag));
        html.push_str(&feature_importance_table(diag));

        // ── Recommendations ──
        html.push_str("<h3>Diagnostic Recommendations</h3>\n");
        html.push_str(&recommendations_html(diag));

        html.push_str("</div>\n"); // close card
    }

    html.push_str("</section>\n");
    html
}

fn acc_color(acc: f64) -> &'static str {
    if acc >= 55.0 { "#00e676" }
    else if acc >= 52.0 { "#ffd740" }
    else if acc >= 50.0 { "#ff9800" }
    else { "#ff5252" }
}

fn cm_card(name: &str, cm: &ConfusionMatrix) -> String {
    let bias = cm.bullish_rate();
    let bias_label = if (bias - cm.actual_up_rate()).abs() > 15.0 { "⚠ HEAVY BIAS" }
        else if (bias - cm.actual_up_rate()).abs() > 8.0 { "⚠ BIAS" }
        else { "✓ Balanced" };

    format!(r#"<div style='background:rgba(10,16,24,0.7);border:1px solid var(--border);border-radius:8px;padding:14px;'>
  <div style='font-family:var(--mono);font-size:13px;font-weight:600;color:var(--teal);margin-bottom:10px;'>{}</div>
  <table style='font-size:12px;'>
    <tr><td></td><td style='font-weight:bold;color:#00e676;text-align:center;padding:4px 8px;'>Pred UP</td><td style='font-weight:bold;color:#ff5252;text-align:center;padding:4px 8px;'>Pred DOWN</td></tr>
    <tr><td style='font-weight:bold;color:#00e676;padding:4px 8px;'>Actual UP</td>
        <td style='text-align:center;background:rgba(0,230,118,0.1);padding:4px 8px;font-family:var(--mono);'>{}</td>
        <td style='text-align:center;background:rgba(255,82,82,0.08);padding:4px 8px;font-family:var(--mono);'>{}</td></tr>
    <tr><td style='font-weight:bold;color:#ff5252;padding:4px 8px;'>Actual DOWN</td>
        <td style='text-align:center;background:rgba(255,82,82,0.08);padding:4px 8px;font-family:var(--mono);'>{}</td>
        <td style='text-align:center;background:rgba(0,230,118,0.1);padding:4px 8px;font-family:var(--mono);'>{}</td></tr>
  </table>
  <div style='font-size:11px;color:var(--text-dim);margin-top:8px;'>
    Acc: <strong style='color:{};'>{:.1}%</strong> &nbsp; Prec: {:.1}% &nbsp; Recall: {:.1}% &nbsp; F1: {:.1}%<br>
    Predicts UP: {:.1}% &nbsp; <span style='color:{};'>{}</span>
  </div>
</div>
"#,
        name,
        cm.tp, cm.fn_,
        cm.fp, cm.tn,
        acc_color(cm.accuracy()), cm.accuracy(),
        cm.precision(), cm.recall(), cm.f1(),
        cm.bullish_rate(),
        if bias_label.contains("BIAS") { "#fbbf24" } else { "#00e676" },
        bias_label,
    )
}

fn bias_table(diag: &SymbolDiagnostics) -> String {
    let mut html = String::new();
    html.push_str("<table><thead><tr>\
        <th style='text-align:left;'>Model</th><th>Predicts UP %</th><th>Actual UP %</th><th>Difference</th><th>Verdict</th>\
        </tr></thead><tbody>\n");

    let models = [
        ("LinReg", diag.linear_cm.bullish_rate()),
        ("LogReg", diag.logistic_cm.bullish_rate()),
        ("GBT", diag.gbt_cm.bullish_rate()),
    ];

    for (name, bull_rate) in &models {
        let diff = bull_rate - diag.actual_up_pct;
        let verdict = bias_verdict(*bull_rate, diag.actual_up_pct);
        let color = if verdict.contains("BIAS") { "#fbbf24" } else { "#00e676" };
        html.push_str(&format!(
            "<tr><td style='text-align:left;'>{}</td><td>{:.1}%</td><td>{:.1}%</td>\
             <td style='color:{};'>{:+.1}%</td><td style='color:{};'>{}</td></tr>\n",
            name, bull_rate, diag.actual_up_pct,
            if diff.abs() > 8.0 { "#fbbf24" } else { "var(--text)" }, diff,
            color, verdict));
    }
    html.push_str("</tbody></table>\n");
    html
}

fn contribution_table(diag: &SymbolDiagnostics) -> String {
    let mut html = String::new();
    html.push_str("<table><thead><tr>\
        <th style='text-align:left;'>Configuration</th><th>Accuracy</th><th>Δ vs Full</th><th>Verdict</th>\
        </tr></thead><tbody>\n");

    let configs = [
        ("Full Ensemble (3 models)", diag.ensemble_accuracy, 0.0),
        ("Without LinReg", diag.accuracy_without_linear, diag.accuracy_without_linear - diag.ensemble_accuracy),
        ("Without LogReg", diag.accuracy_without_logistic, diag.accuracy_without_logistic - diag.ensemble_accuracy),
        ("Without GBT", diag.accuracy_without_gbt, diag.accuracy_without_gbt - diag.ensemble_accuracy),
    ];

    for (name, acc, delta) in &configs {
        let verdict = if name.starts_with("Full") { "—" }
            else { contribution_verdict(diag.ensemble_accuracy, diag.ensemble_accuracy + delta) };
        let delta_color = if *delta > 0.5 { "#ff5252" } // removing helps = model was dragging
            else if *delta < -0.5 { "#00e676" } // removing hurts = model was helping
            else { "var(--text-dim)" };
        html.push_str(&format!(
            "<tr><td style='text-align:left;'>{}</td><td style='color:{};'>{:.1}%</td>\
             <td style='color:{};'>{:+.1}%</td><td>{}</td></tr>\n",
            name, acc_color(*acc), acc, delta_color, delta, verdict));
    }
    html.push_str("</tbody></table>\n");
    html.push_str("<p style='font-size:11px;color:var(--text-muted);margin-top:6px;'>\
        If removing a model <strong>increases</strong> accuracy, that model is dragging the ensemble down. \
        Consider removing it or re-tuning its hyperparameters.</p>\n");
    html
}

fn feature_importance_chart(diag: &SymbolDiagnostics) -> String {
    // Horizontal bar chart as inline SVG
    let top_n = 25.min(diag.feature_importance.len());
    let bar_height = 18;
    let label_width = 180;
    let chart_width = 700;
    let total_height = top_n * (bar_height + 4) + 30;
    let max_imp = diag.feature_importance.iter().take(top_n)
        .map(|(_, v)| *v).fold(0.0_f64, f64::max);

    let mut svg = format!(
        "<svg width='100%' viewBox='0 0 {} {}' style='margin:10px 0;'>\n",
        chart_width, total_height);

    for (i, (name, imp)) in diag.feature_importance.iter().take(top_n).enumerate() {
        let y = i as i32 * (bar_height as i32 + 4) + 4;
        let bar_w = if max_imp > 0.0 { (imp / max_imp * (chart_width - label_width - 60) as f64) as i32 } else { 0 };
        let color = if *imp > 0.03 { "#00d4aa" }
            else if *imp > 0.01 { "#fbbf24" }
            else { "#4a5568" };

        svg.push_str(&format!(
            "  <text x='{}' y='{}' fill='#7a8a9e' font-family='JetBrains Mono, monospace' font-size='10' text-anchor='end' dominant-baseline='central'>{}</text>\n",
            label_width - 8, y + bar_height as i32 / 2, name));
        svg.push_str(&format!(
            "  <rect x='{}' y='{}' width='{}' height='{}' fill='{}' rx='2'/>\n",
            label_width, y, bar_w, bar_height, color));
        svg.push_str(&format!(
            "  <text x='{}' y='{}' fill='#e8edf2' font-family='JetBrains Mono, monospace' font-size='9' dominant-baseline='central'>{:.3}</text>\n",
            label_width + bar_w + 6, y + bar_height as i32 / 2, imp));
    }

    svg.push_str("</svg>\n");
    svg
}

fn feature_importance_table(diag: &SymbolDiagnostics) -> String {
    let mut html = String::new();

    // Show full table with prune recommendations
    html.push_str("<details><summary style='cursor:pointer;color:var(--teal);font-size:13px;margin:8px 0;'>Show all features with prune recommendations ▾</summary>\n");
    html.push_str("<table><thead><tr>\
        <th style='text-align:left;'>Rank</th><th style='text-align:left;'>Feature</th><th>Importance</th><th>Category</th><th>Recommendation</th>\
        </tr></thead><tbody>\n");

    for (i, (name, imp)) in diag.feature_importance.iter().enumerate() {
        let category = feature_category(name);
        let (rec, rec_color) = if *imp < 0.002 {
            ("PRUNE", "#ff5252")
        } else if *imp < 0.005 {
            ("Consider pruning", "#fbbf24")
        } else if *imp < 0.01 {
            ("Low signal", "#ff9800")
        } else if *imp >= 0.03 {
            ("★ HIGH VALUE", "#00e676")
        } else {
            ("Keep", "var(--text-dim)")
        };

        html.push_str(&format!(
            "<tr><td style='text-align:left;'>{}</td><td style='text-align:left;'>{}</td>\
             <td style='font-family:var(--mono);'>{:.4}</td><td style='color:var(--text-dim);'>{}</td>\
             <td style='color:{};font-weight:600;'>{}</td></tr>\n",
            i + 1, name, imp, category, rec_color, rec));
    }
    html.push_str("</tbody></table>\n</details>\n");
    html
}

fn feature_category(name: &str) -> &'static str {
    if name.starts_with("RSI") || name.starts_with("MACD") || name.starts_with("BB_")
        || name.starts_with("SMA") || name.starts_with("EMA") || name.starts_with("price_vs")
        || name == "daily_return" || name == "daily_range_pct" {
        "Technical"
    } else if name.starts_with("volume") || name.starts_with("obv") || name == "price_volume_corr_10d" {
        "Volume"
    } else if name.starts_with("volatility") || name.starts_with("vol_") || name.starts_with("atr")
        || name.starts_with("garman") || name.starts_with("max_drawdown") {
        "Volatility"
    } else if name.starts_with("momentum") || name.starts_with("return_") || name.starts_with("up_days") {
        "Momentum"
    } else if name.starts_with("day_of") || name.starts_with("month") || name == "is_month_end" {
        "Calendar"
    } else if name.starts_with("VIX") || name.starts_with("treasury") || name.starts_with("SPY")
        || name.starts_with("sector") || name.starts_with("gold") || name.starts_with("dollar")
        || name == "risk_on_off" {
        "Market Context"
    } else if name.starts_with("lag") {
        "Lagged"
    } else if name.starts_with("skew") || name.starts_with("kurtosis") || name.starts_with("autocorr")
        || name.starts_with("hurst") || name.starts_with("mean_reversion") {
        "Statistical"
    } else {
        "Other"
    }
}

fn fold_accuracy_chart(diag: &SymbolDiagnostics) -> String {
    // Grouped bar chart showing per-fold accuracy by model
    let n = diag.n_folds;
    if n == 0 { return String::new(); }

    let chart_width = 700;
    let chart_height = 200;
    let margin_left = 45;
    let margin_bottom = 30;
    let margin_top = 10;
    let plot_w = chart_width - margin_left - 20;
    let plot_h = chart_height - margin_bottom - margin_top;

    let group_width = plot_w / n as i32;
    let bar_width = (group_width as f64 * 0.2) as i32;
    let gap = 2;

    let mut svg = format!(
        "<svg width='100%' viewBox='0 0 {} {}' style='margin:10px 0;'>\n",
        chart_width, chart_height);

    // 50% reference line
    let y_50 = margin_top + (plot_h as f64 * (1.0 - (50.0 - 30.0) / 40.0)) as i32;
    svg.push_str(&format!(
        "  <line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#ff5252' stroke-dasharray='4,4' stroke-opacity='0.5'/>\n",
        margin_left, y_50, chart_width - 20, y_50));
    svg.push_str(&format!(
        "  <text x='{}' y='{}' fill='#ff5252' font-size='9' font-family='JetBrains Mono, monospace'>50%</text>\n",
        margin_left - 30, y_50 + 3));

    let colors = ["#4ade80", "#60a5fa", "#f59e0b", "#c084fc"]; // Lin, Log, GBT, Ensemble
    let labels = ["LinReg", "LogReg", "GBT", "Ens"];

    for (i, fold_idx) in (0..n).enumerate() {
        let accuracies = [
            diag.linear_folds[fold_idx].accuracy,
            diag.logistic_folds[fold_idx].accuracy,
            diag.gbt_folds[fold_idx].accuracy,
            diag.ensemble_fold_accuracies[fold_idx],
        ];

        let group_x = margin_left + (i as i32) * group_width + group_width / 8;

        for (j, acc) in accuracies.iter().enumerate() {
            // Scale: show 30% to 70% range
            let clamped = acc.clamp(30.0, 70.0);
            let bar_h = ((clamped - 30.0) / 40.0 * plot_h as f64) as i32;
            let x = group_x + (j as i32) * (bar_width + gap);
            let y = margin_top + plot_h - bar_h;

            svg.push_str(&format!(
                "  <rect x='{}' y='{}' width='{}' height='{}' fill='{}' rx='1' opacity='0.85'/>\n",
                x, y, bar_width, bar_h, colors[j]));
        }

        // Fold label
        svg.push_str(&format!(
            "  <text x='{}' y='{}' fill='#7a8a9e' font-size='9' font-family='JetBrains Mono, monospace' text-anchor='middle'>F{}</text>\n",
            group_x + (4 * (bar_width + gap)) / 2, chart_height - 8, i + 1));
    }

    // Legend
    for (j, (label, color)) in labels.iter().zip(colors.iter()).enumerate() {
        let lx = chart_width - 200 + j as i32 * 50;
        svg.push_str(&format!(
            "  <rect x='{}' y='4' width='8' height='8' fill='{}' rx='1'/>\n", lx, color));
        svg.push_str(&format!(
            "  <text x='{}' y='11' fill='#7a8a9e' font-size='8' font-family='JetBrains Mono, monospace'>{}</text>\n",
            lx + 11, label));
    }

    svg.push_str("</svg>\n");
    svg
}

fn recommendations_html(diag: &SymbolDiagnostics) -> String {
    let mut recs = Vec::new();

    // 1. Check for models worse than random
    if diag.linear_accuracy < 50.0 {
        recs.push(("🔴", format!("LinReg is worse than coin flip ({:.1}%). It's adding noise to the ensemble. Consider removing it or switching to a regularised version.", diag.linear_accuracy)));
    }
    if diag.logistic_accuracy < 50.0 {
        recs.push(("🔴", format!("LogReg is worse than coin flip ({:.1}%). It's hurting ensemble accuracy.", diag.logistic_accuracy)));
    }
    if diag.gbt_accuracy < 50.0 {
        recs.push(("🔴", format!("GBT is worse than coin flip ({:.1}%). Try reducing max_depth to 3 or increasing min_samples_leaf.", diag.gbt_accuracy)));
    }

    // 2. Check for heavy bias
    for (name, cm) in [("LinReg", &diag.linear_cm), ("LogReg", &diag.logistic_cm), ("GBT", &diag.gbt_cm)] {
        let diff = cm.bullish_rate() - diag.actual_up_pct;
        if diff.abs() > 15.0 {
            let direction = if diff > 0.0 { "bullish" } else { "bearish" };
            recs.push(("🟡", format!("{} has heavy {} bias (predicts UP {:.1}% vs actual {:.1}%). It may be memorising the base rate rather than learning patterns. Try adding class weighting or adjusting the decision threshold.",
                name, direction, cm.bullish_rate(), diag.actual_up_pct)));
        }
    }

    // 3. Feature pruning
    let low_features: Vec<&str> = diag.feature_importance.iter()
        .filter(|(_, imp)| *imp < 0.002)
        .map(|(name, _)| name.as_str())
        .collect();
    if !low_features.is_empty() {
        recs.push(("🟡", format!("{} features have near-zero importance and are likely noise. Prune them to reduce overfitting: {}",
            low_features.len(), low_features.join(", "))));
    }

    // 4. Model contribution
    if diag.accuracy_without_linear > diag.ensemble_accuracy + 0.5 {
        recs.push(("🟡", format!("Removing LinReg IMPROVES ensemble by {:.1}pp. It's dragging accuracy down.",
            diag.accuracy_without_linear - diag.ensemble_accuracy)));
    }
    if diag.accuracy_without_logistic > diag.ensemble_accuracy + 0.5 {
        recs.push(("🟡", format!("Removing LogReg IMPROVES ensemble by {:.1}pp.",
            diag.accuracy_without_logistic - diag.ensemble_accuracy)));
    }
    if diag.accuracy_without_gbt > diag.ensemble_accuracy + 0.5 {
        recs.push(("🟡", format!("Removing GBT IMPROVES ensemble by {:.1}pp.",
            diag.accuracy_without_gbt - diag.ensemble_accuracy)));
    }

    // 5. Fold stability
    if diag.ensemble_fold_accuracies.len() >= 2 {
        let min_fold = diag.ensemble_fold_accuracies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_fold = diag.ensemble_fold_accuracies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread = max_fold - min_fold;
        if spread > 15.0 {
            recs.push(("🟡", format!("High fold variance: {:.1}% to {:.1}% (spread {:.1}pp). The model isn't stable across time periods. Consider increasing training window or using regularisation.",
                min_fold, max_fold, spread)));
        }
    }

    // 6. Data balance
    if (diag.actual_up_pct - 50.0).abs() > 8.0 {
        recs.push(("🟡", format!("Data is {:.1}% UP days. Consider using balanced accuracy or class weighting to prevent the model from just predicting the majority class.",
            diag.actual_up_pct)));
    }

    // 7. Positive signals
    if diag.ensemble_accuracy > 55.0 {
        recs.push(("🟢", format!("Ensemble at {:.1}% — this is a meaningful edge above random.", diag.ensemble_accuracy)));
    }
    let best_individual = diag.linear_accuracy.max(diag.logistic_accuracy).max(diag.gbt_accuracy);
    if diag.ensemble_accuracy > best_individual + 0.5 {
        recs.push(("🟢", format!("Ensemble ({:.1}%) beats the best individual model ({:.1}%) — the combination is working.", diag.ensemble_accuracy, best_individual)));
    }

    // 8. High-value features
    let top_features: Vec<&str> = diag.feature_importance.iter()
        .take(5)
        .map(|(name, _)| name.as_str())
        .collect();
    recs.push(("ℹ️", format!("Top 5 features driving GBT decisions: {}", top_features.join(", "))));

    // Build HTML
    let mut html = String::new();
    html.push_str("<div style='background:rgba(10,16,24,0.7);border:1px solid var(--border);border-radius:8px;padding:16px;'>\n");

    if recs.is_empty() {
        html.push_str("<p style='color:var(--text-dim);'>No specific issues detected.</p>\n");
    } else {
        for (icon, text) in &recs {
            html.push_str(&format!(
                "<div style='margin:8px 0;padding:8px 12px;border-left:3px solid {};border-radius:0 4px 4px 0;background:rgba(255,255,255,0.02);font-size:12px;'>\
                 {} {}</div>\n",
                match *icon {
                    "🔴" => "#ef4444",
                    "🟡" => "#fbbf24",
                    "🟢" => "#10b981",
                    _ => "#7a8a9e",
                },
                icon, text));
        }
    }

    html.push_str("</div>\n");
    html
}
