/// LSTM Model for Sequence-Based Financial Prediction
/// ===================================================
/// Uses Hugging Face's candle-nn LSTM implementation.
///
/// Architecture:
///   Input(n_features) → LSTM(hidden_size) → Linear(hidden_size → 1) → Sigmoid
///
/// The LSTM sees sequences of days (e.g., 20 consecutive feature vectors)
/// and predicts next-day direction. This captures temporal patterns that
/// the pointwise models (LinReg, LogReg, GBT) miss.
///
/// Training uses AdamW optimizer with binary cross-entropy loss.
/// Model weights serialised via VarMap save/load for persistence.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarMap, VarBuilder, Optimizer, Module};
use candle_nn::rnn::{LSTM, LSTMConfig, LSTMState, RNN};
use candle_nn::linear;
use crate::ml::{self, Sample};

/// Sanitise a float: replace NaN/Inf with 0.0
fn sanitise_f32(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Sanitise a vector of f64 values for tensor creation
fn sanitise_vec(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&x| sanitise_f32(x as f32)).collect()
}

/// Count NaN/Inf values in a slice
fn count_bad_values(data: &[f64]) -> usize {
    data.iter().filter(|x| !x.is_finite()).count()
}

/// Select device for LSTM operations.
/// NOTE: candle 0.9 Metal backend is missing sigmoid/tanh kernels which LSTM
/// gates require, so we force CPU. The Metal feature is still useful for other
/// modules (e.g. matrix multiplications in GBT/TFT) but not LSTM.
fn select_device() -> Device {
    println!("    [LSTM] Using CPU (Metal lacks sigmoid/tanh kernels for LSTM gates)");
    Device::Cpu
}

/// Lazy-initialised device — CPU for LSTM (see select_device comment).
/// Using std::sync::OnceLock so we only probe once.
fn get_device() -> &'static Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(select_device)
}

/// LSTM configuration
#[derive(Clone, Debug)]
pub struct LSTMModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub seq_length: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Default for LSTMModelConfig {
    fn default() -> Self {
        Self {
            input_size: 30,
            hidden_size: 64,
            seq_length: 10,
            learning_rate: 0.0005,
            epochs: 50,
            batch_size: 32,
        }
    }
}

/// Trained LSTM model
pub struct LSTMModel {
    lstm: LSTM,
    fc: candle_nn::Linear,
    varmap: VarMap,
    config: LSTMModelConfig,
}

impl LSTMModel {
    /// Create a new untrained LSTM
    pub fn new(config: LSTMModelConfig) -> Result<Self, candle_core::Error> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, get_device());

        let lstm_config = LSTMConfig::default();
        let lstm = candle_nn::rnn::lstm(
            config.input_size,
            config.hidden_size,
            lstm_config,
            vb.pp("lstm"),
        )?;

        let fc = linear(config.hidden_size, 1, vb.pp("fc"))?;

        Ok(Self { lstm, fc, varmap, config })
    }

    /// Train on sequences built from samples
    /// If `feature_indices` is provided, only those feature columns are used
    pub fn train(&mut self, samples: &[Sample], val_samples: &[Sample], feature_indices: Option<&[usize]>) -> Result<TrainResult, candle_core::Error> {
        let seq_len = self.config.seq_length;
        let n_features = self.config.input_size;

        // Validate input data
        let train_bad = samples.iter()
            .flat_map(|s| s.features.iter())
            .filter(|x| !x.is_finite())
            .count();
        let val_bad = val_samples.iter()
            .flat_map(|s| s.features.iter())
            .filter(|x| !x.is_finite())
            .count();
        if train_bad > 0 || val_bad > 0 {
            println!("    [LSTM] WARNING: {} NaN/Inf in train, {} in val — sanitising", train_bad, val_bad);
        }

        // Build sequences: each sequence is seq_len consecutive samples
        let train_seqs = build_sequences_with_subset(samples, seq_len, feature_indices);
        let val_seqs = build_sequences_with_subset(val_samples, seq_len, feature_indices);

        println!("    [LSTM] Samples: {} train, {} val → {} train seqs, {} val seqs (seq_len={})",
            samples.len(), val_samples.len(), train_seqs.len(), val_seqs.len(), seq_len);

        if train_seqs.is_empty() || val_seqs.is_empty() {
            println!("    [LSTM] ERROR: empty sequences (need >{} samples per split)", seq_len + 1);
            return Ok(TrainResult {
                final_train_loss: f64::NAN,
                final_val_loss: f64::NAN,
                best_val_loss: f64::NAN,
                epochs_trained: 0,
                val_accuracy: 0.0,
            });
        }

        // Check label distribution
        let n_up = train_seqs.iter().filter(|s| s.label > 0.5).count();
        println!("    [LSTM] Label distribution: {:.1}% up, {:.1}% down",
            n_up as f64 / train_seqs.len() as f64 * 100.0,
            (train_seqs.len() - n_up) as f64 / train_seqs.len() as f64 * 100.0);

        let optim_config = candle_nn::ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(self.varmap.all_vars(), optim_config)?;

        let mut best_val_loss = f64::MAX;
        let mut patience = 0;
        let max_patience = 8;
        let mut final_train_loss = 0.0;
        let mut final_val_loss = 0.0;
        let mut epochs_trained = 0;
        let mut nan_loss_count = 0_usize;

        println!("    [LSTM] Training on {} sequences, validating on {}", train_seqs.len(), val_seqs.len());

        for epoch in 0..self.config.epochs {
            // Mini-batch training
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..train_seqs.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_seqs.len());
                let batch = &train_seqs[batch_start..batch_end];
                let batch_size = batch.len();

                // Build input tensor [batch_size, seq_len, n_features] with NaN sanitisation
                let mut input_data = Vec::with_capacity(batch_size * seq_len * n_features);
                let mut labels_data = Vec::with_capacity(batch_size);

                for seq in batch {
                    for step in &seq.features {
                        input_data.extend(sanitise_vec(step));
                    }
                    labels_data.push(sanitise_f32(seq.label as f32));
                }

                let input = Tensor::from_vec(
                    input_data,
                    (batch_size, seq_len, n_features),
                    get_device(),
                )?;
                let labels = Tensor::from_vec(
                    labels_data,
                    (batch_size, 1),
                    get_device(),
                )?;

                // Forward pass: process each timestep through LSTM
                let logits = self.forward_batch(&input)?;

                // Check for NaN in logits (indicates Metal GPU issue)
                let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
                let logit_nans = logits_vec.iter().filter(|x| !x.is_finite()).count();
                if logit_nans > 0 {
                    if epoch == 0 && n_batches == 0 {
                        println!("    [LSTM] ERROR: {} NaN/Inf in logits — GPU/Metal issue likely", logit_nans);
                        println!("    [LSTM] Logit sample: {:?}", &logits_vec[..logits_vec.len().min(5)]);
                    }
                    nan_loss_count += 1;
                    continue; // skip this batch
                }

                // Binary cross-entropy with logits
                let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;
                let loss_val = loss.to_scalar::<f32>()? as f64;

                // Validate loss
                if !loss_val.is_finite() {
                    nan_loss_count += 1;
                    if nan_loss_count <= 3 {
                        println!("    [LSTM] WARNING: NaN/Inf loss at epoch {} batch {} (loss={})",
                            epoch + 1, n_batches, loss_val);
                    }
                    continue; // skip this batch, don't update weights
                }

                // Backward + step
                optimizer.backward_step(&loss)?;

                epoch_loss += loss_val;
                n_batches += 1;
            }

            if n_batches == 0 {
                if epoch == 0 {
                    println!("    [LSTM] ERROR: no valid batches in epoch 1 — all produced NaN");
                }
                continue;
            }

            final_train_loss = epoch_loss / n_batches as f64;

            // Validation loss
            let val_loss = self.evaluate_loss(&val_seqs)?;
            final_val_loss = val_loss;

            if epoch == 0 || (epoch + 1) % 10 == 0 || epoch == self.config.epochs - 1 {
                println!("    [LSTM] Epoch {:>3}/{}: train_loss={:.4}, val_loss={:.4}{}",
                    epoch + 1, self.config.epochs, final_train_loss, val_loss,
                    if nan_loss_count > 0 { format!(" (nan_batches={})", nan_loss_count) } else { String::new() });
            }

            epochs_trained = epoch + 1;

            // Early stopping (only on valid loss)
            if val_loss.is_finite() && val_loss < best_val_loss - 0.001 {
                best_val_loss = val_loss;
                patience = 0;
            } else {
                patience += 1;
                if patience >= max_patience {
                    println!("    [LSTM] Early stopping at epoch {} (val_loss={:.4})", epoch + 1, val_loss);
                    break;
                }
            }
        }

        if nan_loss_count > 0 {
            println!("    [LSTM] Total NaN/Inf batches: {} out of ~{}", nan_loss_count,
                (train_seqs.len() / self.config.batch_size + 1) * epochs_trained);
        }

        // Compute validation accuracy
        let val_accuracy = self.evaluate_accuracy(&val_seqs)?;
        println!("    [LSTM] Validation accuracy: {:.1}% ({} val sequences)", val_accuracy * 100.0, val_seqs.len());

        // Log prediction distribution
        let val_probs = self.predict_probs_batch(&val_seqs)?;
        let n_pred_up = val_probs.iter().filter(|&&p| p > 0.5).count();
        let avg_prob = val_probs.iter().sum::<f64>() / val_probs.len().max(1) as f64;
        println!("    [LSTM] Val predictions: {:.1}% predict UP, avg_prob={:.3}",
            n_pred_up as f64 / val_probs.len().max(1) as f64 * 100.0, avg_prob);

        Ok(TrainResult {
            final_train_loss,
            final_val_loss,
            best_val_loss,
            epochs_trained,
            val_accuracy,
        })
    }

    /// Forward pass for a batch [batch_size, seq_len, n_features]
    fn forward_batch(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _n_features) = input.dims3()?;

        // Process each timestep through LSTM
        // candle LSTM expects individual timestep tensors
        let mut final_h = None;

        for t in 0..seq_len {
            let step = input.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [batch_size, n_features]

            // For the first step, create zero state
            if t == 0 {
                let h0 = Tensor::zeros((batch_size, self.config.hidden_size), DType::F32, get_device())?;
                let c0 = Tensor::zeros((batch_size, self.config.hidden_size), DType::F32, get_device())?;
                let state = LSTMState::new(h0, c0);
                let new_state = self.lstm.step(&step, &state)?;
                final_h = Some(new_state);
            } else if let Some(prev_state) = &final_h {
                let new_state = self.lstm.step(&step, prev_state)?;
                final_h = Some(new_state);
            }
        }

        // Take final hidden state → linear → output
        let h = final_h.unwrap().h().clone().contiguous()?; // [batch_size, hidden_size]
        let logits = self.fc.forward(&h)?; // [batch_size, 1]

        Ok(logits)
    }

    /// Evaluate loss on a set of sequences
    fn evaluate_loss(&self, seqs: &[Sequence]) -> Result<f64, candle_core::Error> {
        if seqs.is_empty() { return Ok(f64::NAN); }

        let n_features = self.config.input_size;
        let seq_len = self.config.seq_length;
        let batch_size = seqs.len();

        let mut input_data = Vec::with_capacity(batch_size * seq_len * n_features);
        let mut labels_data = Vec::with_capacity(batch_size);

        for seq in seqs {
            for step in &seq.features {
                input_data.extend(sanitise_vec(step));
            }
            labels_data.push(sanitise_f32(seq.label as f32));
        }

        let input = Tensor::from_vec(
            input_data,
            (batch_size, seq_len, n_features),
            get_device(),
        )?;
        let labels = Tensor::from_vec(
            labels_data,
            (batch_size, 1),
            get_device(),
        )?;

        let logits = self.forward_batch(&input)?;
        let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }

    /// Evaluate accuracy on a set of sequences
    fn evaluate_accuracy(&self, seqs: &[Sequence]) -> Result<f64, candle_core::Error> {
        if seqs.is_empty() { return Ok(0.0); }
        let probs_vec = self.predict_probs_batch(seqs)?;

        let mut correct = 0;
        for (i, seq) in seqs.iter().enumerate() {
            let predicted_up = probs_vec[i] > 0.5;
            let actual_up = seq.label > 0.5;
            if predicted_up == actual_up { correct += 1; }
        }

        Ok(correct as f64 / seqs.len() as f64)
    }

    /// Get prediction probabilities for a batch of sequences
    fn predict_probs_batch(&self, seqs: &[Sequence]) -> Result<Vec<f64>, candle_core::Error> {
        if seqs.is_empty() { return Ok(vec![]); }

        let n_features = self.config.input_size;
        let seq_len = self.config.seq_length;
        let batch_size = seqs.len();

        let mut input_data = Vec::with_capacity(batch_size * seq_len * n_features);
        for seq in seqs {
            for step in &seq.features {
                input_data.extend(sanitise_vec(step));
            }
        }

        let input = Tensor::from_vec(
            input_data,
            (batch_size, seq_len, n_features),
            get_device(),
        )?;

        let logits = self.forward_batch(&input)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
        Ok(probs_vec.iter().map(|&p| p as f64).collect())
    }

    /// Predict P(up) for a single sequence of features
    pub fn predict_proba(&self, sequence: &[Vec<f64>]) -> Result<f64, candle_core::Error> {
        let seq_len = sequence.len();
        let n_features = self.config.input_size;

        let mut input_data = Vec::with_capacity(seq_len * n_features);
        for step in sequence {
            input_data.extend(sanitise_vec(step));
        }

        let input = Tensor::from_vec(
            input_data,
            (1, seq_len, n_features),
            get_device(),
        )?;

        let logits = self.forward_batch(&input)?;
        let prob = candle_nn::ops::sigmoid(&logits)?;
        let val = prob.flatten_all()?.to_vec1::<f32>()?[0];

        Ok(sanitise_f32(val) as f64)
    }

    /// Predict direction for a single sequence
    pub fn predict_direction(&self, sequence: &[Vec<f64>]) -> Result<bool, candle_core::Error> {
        Ok(self.predict_proba(sequence)? > 0.5)
    }

    /// Save model weights to file
    pub fn save(&self, path: &str) -> Result<(), candle_core::Error> {
        self.varmap.save(path)?;
        println!("    [LSTM] Model saved to {}", path);
        Ok(())
    }

    /// Load model weights from file
    pub fn load(config: LSTMModelConfig, path: &str) -> Result<Self, candle_core::Error> {
        let mut model = Self::new(config)?;
        // Load saved weights into the varmap
        model.varmap.load(path)?;
        println!("    [LSTM] Model loaded from {}", path);
        Ok(model)
    }
}

/// Training result metrics
pub struct TrainResult {
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub best_val_loss: f64,
    pub epochs_trained: usize,
    pub val_accuracy: f64,
}

/// A sequence of feature vectors with a label
pub struct Sequence {
    pub features: Vec<Vec<f64>>,  // [seq_len][n_features]
    pub label: f64,               // 1.0 = up, 0.0 = down
}

/// Build sequences from flat samples
/// Each sequence is `seq_len` consecutive samples, label is from the last sample
/// If `feature_indices` is provided, only those feature columns are kept
pub fn build_sequences(samples: &[Sample], seq_len: usize) -> Vec<Sequence> {
    build_sequences_with_subset(samples, seq_len, None)
}

/// Build sequences with optional feature subsetting
pub fn build_sequences_with_subset(samples: &[Sample], seq_len: usize, feature_indices: Option<&[usize]>) -> Vec<Sequence> {
    if samples.len() < seq_len + 1 {
        return Vec::new();
    }

    let mut sequences = Vec::new();

    for i in seq_len..samples.len() {
        let features: Vec<Vec<f64>> = samples[i - seq_len..i]
            .iter()
            .map(|s| {
                if let Some(indices) = feature_indices {
                    indices.iter().map(|&idx| {
                        if idx < s.features.len() { s.features[idx] } else { 0.0 }
                    }).collect()
                } else {
                    s.features.clone()
                }
            })
            .collect();

        let label = if samples[i].label > 0.0 { 1.0 } else { 0.0 };

        sequences.push(Sequence { features, label });
    }

    sequences
}

/// Run LSTM walk-forward evaluation on pre-built samples
/// Returns (overall_accuracy, recent_accuracy, final_prob)
/// If `feature_indices` is provided, only those feature columns are fed to the LSTM
pub fn walk_forward_lstm(
    symbol: &str,
    samples: &[Sample],
    config: &LSTMModelConfig,
    train_window: usize,
    test_window: usize,
    step: usize,
    feature_indices: Option<&[usize]>,
) -> Option<LSTMWalkForwardResult> {
    let seq_len = config.seq_length;

    // Check NaN/Inf in input samples
    let bad_features: usize = samples.iter()
        .map(|s| count_bad_values(&s.features))
        .sum();
    if bad_features > 0 {
        println!("  {} — [LSTM] WARNING: {} NaN/Inf values in input features", symbol, bad_features);
    }

    if samples.len() < train_window + test_window + seq_len {
        println!("  {} — not enough samples for LSTM walk-forward ({} samples, need {})",
            symbol, samples.len(), train_window + test_window + seq_len);
        return None;
    }

    println!("  {} — LSTM walk-forward on {} samples × {} features (seq_len={}, train_win={}, test_win={}, step={})",
        symbol, samples.len(), config.input_size, seq_len, train_window, test_window, step);

    let mut total_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0_usize;
    let mut n_failed_folds = 0_usize;
    let mut last_fold_correct = 0_usize;
    let mut last_fold_size = 0_usize;
    let mut last_prob = 0.5_f64;

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        // Clone and normalise this fold's data (same as pointwise models)
        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;
        let (train_data, test_data) = fold_samples.split_at_mut(train_len);
        let (_means, _stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &_means, &_stds);

        // Split train into train/val (85/15)
        let val_split = (train_data.len() as f64 * 0.85) as usize;
        let (train_part, val_part) = train_data.split_at(val_split);

        let fold_num = n_folds + n_failed_folds + 1;
        if fold_num <= 2 {
            println!("    [LSTM] Fold {}: train={}, val={}, test={} (start={})",
                fold_num, train_part.len(), val_part.len(), test_data.len(), start);
        }

        // Train LSTM
        let mut model = match LSTMModel::new(config.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("    [LSTM] Failed to create model: {}", e);
                return None;
            }
        };

        match model.train(train_part, val_part, feature_indices) {
            Ok(result) => {
                if result.epochs_trained == 0 {
                    println!("    [LSTM] WARNING: 0 epochs trained (empty sequences?)");
                    n_failed_folds += 1;
                    start += step;
                    continue;
                }
            },
            Err(e) => {
                println!("    [LSTM] Training failed: {}", e);
                n_failed_folds += 1;
                start += step;
                continue;
            }
        }

        // Evaluate on test set (using normalised test data)
        let test_seqs = build_sequences_with_subset(test_data, seq_len, feature_indices);
        if test_seqs.is_empty() {
            println!("    [LSTM] WARNING: no test sequences (test_data={} < seq_len+1={})",
                test_data.len(), seq_len + 1);
            n_failed_folds += 1;
            start += step;
            continue;
        }

        let mut fold_correct = 0;

        for seq in &test_seqs {
            let prob = match model.predict_proba(&seq.features) {
                Ok(p) => {
                    if !p.is_finite() { 0.5 } else { p }
                },
                Err(e) => {
                    if fold_correct == 0 {
                        println!("    [LSTM] Prediction error: {}", e);
                    }
                    0.5
                },
            };
            let predicted_up = prob > 0.5;
            let actual_up = seq.label > 0.5;
            if predicted_up == actual_up { fold_correct += 1; }
        }

        let fold_size = test_seqs.len();
        let fold_acc = fold_correct as f64 / fold_size.max(1) as f64 * 100.0;
        println!("    [LSTM] Fold test: {}/{} correct ({:.1}%)", fold_correct, fold_size, fold_acc);

        total_correct += fold_correct;
        total_tested += fold_size;
        n_folds += 1;
        last_fold_correct = fold_correct;
        last_fold_size = fold_size;

        // Get final prediction
        if let Some(last_seq) = test_seqs.last() {
            last_prob = model.predict_proba(&last_seq.features).unwrap_or(0.5);
        }

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        println!("    [LSTM] FAILED: {} folds attempted, {} succeeded, {} total test sequences",
            n_folds + n_failed_folds, n_folds, total_tested);
        return None;
    }

    let overall_acc = total_correct as f64 / total_tested as f64 * 100.0;
    let recent_acc = last_fold_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    LSTM walk-forward: {} folds ({} failed), {} test sequences", n_folds, n_failed_folds, total_tested);
    println!("      LSTM: {:.1}% (recent: {:.1}%)", overall_acc, recent_acc);

    Some(LSTMWalkForwardResult {
        overall_accuracy: overall_acc,
        recent_accuracy: recent_acc,
        final_prob: last_prob.clamp(0.15, 0.85),
        n_folds,
        total_tested,
    })
}

pub struct LSTMWalkForwardResult {
    pub overall_accuracy: f64,
    pub recent_accuracy: f64,
    pub final_prob: f64,
    pub n_folds: usize,
    pub total_tested: usize,
}
