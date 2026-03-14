/// GRU Model for Sequence-Based Financial Prediction
/// ==================================================
/// Uses candle-nn GRU implementation. Near drop-in alongside LSTM.
///
/// Architecture:
///   Input(n_features) → GRU(hidden_size) → Linear(hidden_size → 1) → Sigmoid
///
/// Shares the same sequence-building, walk-forward, and gating logic as LSTM.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarMap, VarBuilder, Optimizer, Module};
use candle_nn::rnn::{GRU, GRUConfig, GRUState, RNN};
use candle_nn::linear;
use crate::ml::{self, Sample};
use crate::lstm::{build_sequences_with_subset, Sequence};

/// Sanitise a float: replace NaN/Inf with 0.0
fn sanitise_f32(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Sanitise a vector of f64 values for tensor creation
fn sanitise_vec(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&x| sanitise_f32(x as f32)).collect()
}

/// Select device — CPU for GRU (same Metal sigmoid/tanh limitation as LSTM).
/// TODO(metal): See lstm.rs select_device() — enable Metal when candle adds
///   sigmoid/tanh kernel support. Track: https://github.com/huggingface/candle/issues/2832
fn select_device() -> Device {
    Device::Cpu
}

fn get_device() -> &'static Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(select_device)
}

/// GRU configuration (mirrors LSTMModelConfig)
#[derive(Clone, Debug)]
pub struct GRUModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub seq_length: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Default for GRUModelConfig {
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

/// Trained GRU model
pub struct GRUModel {
    gru: GRU,
    fc: candle_nn::Linear,
    varmap: VarMap,
    config: GRUModelConfig,
}

impl GRUModel {
    pub fn new(config: GRUModelConfig) -> Result<Self, candle_core::Error> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, get_device());

        let gru_config = GRUConfig::default();
        let gru = candle_nn::rnn::gru(
            config.input_size,
            config.hidden_size,
            gru_config,
            vb.pp("gru"),
        )?;

        let fc = linear(config.hidden_size, 1, vb.pp("fc"))?;

        Ok(Self { gru, fc, varmap, config })
    }

    /// Train on sequences built from samples
    pub fn train(&mut self, samples: &[Sample], val_samples: &[Sample], feature_indices: Option<&[usize]>) -> Result<GRUTrainResult, candle_core::Error> {
        let seq_len = self.config.seq_length;
        let n_features = self.config.input_size;

        let train_seqs = build_sequences_with_subset(samples, seq_len, feature_indices);
        let val_seqs = build_sequences_with_subset(val_samples, seq_len, feature_indices);

        if train_seqs.is_empty() || val_seqs.is_empty() {
            println!("    [GRU] ERROR: empty sequences (need >{} samples per split)", seq_len + 1);
            return Ok(GRUTrainResult {
                final_train_loss: f64::NAN,
                final_val_loss: f64::NAN,
                best_val_loss: f64::NAN,
                epochs_trained: 0,
                val_accuracy: 0.0,
            });
        }

        let n_up = train_seqs.iter().filter(|s| s.label > 0.5).count();
        println!("    [GRU] Training: {} seqs ({:.1}% up), {} val seqs (seq_len={})",
            train_seqs.len(), n_up as f64 / train_seqs.len() as f64 * 100.0,
            val_seqs.len(), seq_len);

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

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..train_seqs.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_seqs.len());
                let batch = &train_seqs[batch_start..batch_end];
                let batch_size = batch.len();

                let mut input_data = Vec::with_capacity(batch_size * seq_len * n_features);
                let mut labels_data = Vec::with_capacity(batch_size);

                for seq in batch {
                    for step in &seq.features {
                        input_data.extend(sanitise_vec(step));
                    }
                    labels_data.push(sanitise_f32(seq.label as f32));
                }

                let input = Tensor::from_vec(input_data, (batch_size, seq_len, n_features), get_device())?;
                let labels = Tensor::from_vec(labels_data, (batch_size, 1), get_device())?;

                let logits = self.forward_batch(&input)?;

                let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
                if logits_vec.iter().any(|x| !x.is_finite()) {
                    continue;
                }

                let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;
                let loss_val = loss.to_scalar::<f32>()? as f64;

                if !loss_val.is_finite() { continue; }

                optimizer.backward_step(&loss)?;
                epoch_loss += loss_val;
                n_batches += 1;
            }

            if n_batches == 0 { continue; }

            final_train_loss = epoch_loss / n_batches as f64;
            let val_loss = self.evaluate_loss(&val_seqs)?;
            final_val_loss = val_loss;

            if epoch == 0 || (epoch + 1) % 10 == 0 || epoch == self.config.epochs - 1 {
                println!("    [GRU] Epoch {:>3}/{}: train_loss={:.4}, val_loss={:.4}",
                    epoch + 1, self.config.epochs, final_train_loss, val_loss);
            }

            epochs_trained = epoch + 1;

            if val_loss.is_finite() && val_loss < best_val_loss - 0.001 {
                best_val_loss = val_loss;
                patience = 0;
            } else {
                patience += 1;
                if patience >= max_patience {
                    println!("    [GRU] Early stopping at epoch {}", epoch + 1);
                    break;
                }
            }
        }

        let val_accuracy = self.evaluate_accuracy(&val_seqs)?;
        println!("    [GRU] Validation accuracy: {:.1}%", val_accuracy * 100.0);

        Ok(GRUTrainResult {
            final_train_loss,
            final_val_loss,
            best_val_loss,
            epochs_trained,
            val_accuracy,
        })
    }

    /// Forward pass for a batch [batch_size, seq_len, n_features]
    fn forward_batch(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _) = input.dims3()?;

        let mut state: Option<GRUState> = None;

        for t in 0..seq_len {
            let step = input.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;

            if state.is_none() {
                let h0 = Tensor::zeros((batch_size, self.config.hidden_size), DType::F32, get_device())?;
                let s = GRUState { h: h0 };
                state = Some(self.gru.step(&step, &s)?);
            } else {
                state = Some(self.gru.step(&step, state.as_ref().unwrap())?);
            }
        }

        let h = state.unwrap().h.clone().contiguous()?;
        let logits = self.fc.forward(&h)?;
        Ok(logits)
    }

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

        let input = Tensor::from_vec(input_data, (batch_size, seq_len, n_features), get_device())?;
        let labels = Tensor::from_vec(labels_data, (batch_size, 1), get_device())?;

        let logits = self.forward_batch(&input)?;
        let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;
        Ok(loss.to_scalar::<f32>()? as f64)
    }

    fn evaluate_accuracy(&self, seqs: &[Sequence]) -> Result<f64, candle_core::Error> {
        if seqs.is_empty() { return Ok(0.0); }
        let probs = self.predict_probs_batch(seqs)?;
        let mut correct = 0;
        for (i, seq) in seqs.iter().enumerate() {
            if (probs[i] > 0.5) == (seq.label > 0.5) { correct += 1; }
        }
        Ok(correct as f64 / seqs.len() as f64)
    }

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

        let input = Tensor::from_vec(input_data, (batch_size, seq_len, n_features), get_device())?;
        let logits = self.forward_batch(&input)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
        Ok(probs_vec.iter().map(|&p| p as f64).collect())
    }

    pub fn predict_proba(&self, sequence: &[Vec<f64>]) -> Result<f64, candle_core::Error> {
        let seq_len = sequence.len();
        let n_features = self.config.input_size;

        let mut input_data = Vec::with_capacity(seq_len * n_features);
        for step in sequence {
            input_data.extend(sanitise_vec(step));
        }

        let input = Tensor::from_vec(input_data, (1, seq_len, n_features), get_device())?;
        let logits = self.forward_batch(&input)?;
        let prob = candle_nn::ops::sigmoid(&logits)?;
        let val = prob.flatten_all()?.to_vec1::<f32>()?[0];
        Ok(sanitise_f32(val) as f64)
    }
}

/// GRU training result
pub struct GRUTrainResult {
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub best_val_loss: f64,
    pub epochs_trained: usize,
    pub val_accuracy: f64,
}

/// GRU walk-forward result
pub struct GRUWalkForwardResult {
    pub overall_accuracy: f64,
    pub recent_accuracy: f64,
    pub final_prob: f64,
    pub n_folds: usize,
    pub total_tested: usize,
}

/// Run GRU walk-forward evaluation on pre-built samples
pub fn walk_forward_gru(
    symbol: &str,
    samples: &[Sample],
    config: &GRUModelConfig,
    train_window: usize,
    test_window: usize,
    step: usize,
    feature_indices: Option<&[usize]>,
) -> Option<GRUWalkForwardResult> {
    let seq_len = config.seq_length;

    if samples.len() < train_window + test_window + seq_len {
        println!("  {} — not enough samples for GRU walk-forward ({} samples)", symbol, samples.len());
        return None;
    }

    println!("  {} — GRU walk-forward on {} samples × {} features (seq_len={})",
        symbol, samples.len(), config.input_size, seq_len);

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

        let mut fold_samples: Vec<Sample> = samples[start..test_end].to_vec();
        let train_len = train_window;
        let (train_data, test_data) = fold_samples.split_at_mut(train_len);
        let (_means, _stds) = ml::normalise(train_data);
        ml::apply_normalisation(test_data, &_means, &_stds);

        let val_split = (train_data.len() as f64 * 0.85) as usize;
        let (train_part, val_part) = train_data.split_at(val_split);

        let mut model = match GRUModel::new(config.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("    [GRU] Failed to create model: {}", e);
                return None;
            }
        };

        match model.train(train_part, val_part, feature_indices) {
            Ok(result) => {
                if result.epochs_trained == 0 {
                    n_failed_folds += 1;
                    start += step;
                    continue;
                }
            },
            Err(e) => {
                println!("    [GRU] Training failed: {}", e);
                n_failed_folds += 1;
                start += step;
                continue;
            }
        }

        let test_seqs = build_sequences_with_subset(test_data, seq_len, feature_indices);
        if test_seqs.is_empty() {
            n_failed_folds += 1;
            start += step;
            continue;
        }

        let mut fold_correct = 0;
        for seq in &test_seqs {
            let prob = model.predict_proba(&seq.features).unwrap_or(0.5);
            let prob = if prob.is_finite() { prob } else { 0.5 };
            if (prob > 0.5) == (seq.label > 0.5) { fold_correct += 1; }
        }

        let fold_size = test_seqs.len();
        let fold_acc = fold_correct as f64 / fold_size.max(1) as f64 * 100.0;
        println!("    [GRU] Fold test: {}/{} correct ({:.1}%)", fold_correct, fold_size, fold_acc);

        total_correct += fold_correct;
        total_tested += fold_size;
        n_folds += 1;
        last_fold_correct = fold_correct;
        last_fold_size = fold_size;

        if let Some(last_seq) = test_seqs.last() {
            last_prob = model.predict_proba(&last_seq.features).unwrap_or(0.5);
        }

        start += step;
    }

    if n_folds == 0 || total_tested == 0 {
        println!("    [GRU] FAILED: {} folds attempted, {} succeeded", n_folds + n_failed_folds, n_folds);
        return None;
    }

    let overall_acc = total_correct as f64 / total_tested as f64 * 100.0;
    let recent_acc = last_fold_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    GRU walk-forward: {} folds ({} failed), {} test sequences", n_folds, n_failed_folds, total_tested);
    println!("      GRU: {:.1}% (recent: {:.1}%)", overall_acc, recent_acc);

    Some(GRUWalkForwardResult {
        overall_accuracy: overall_acc,
        recent_accuracy: recent_acc,
        final_prob: last_prob.clamp(0.15, 0.85),
        n_folds,
        total_tested,
    })
}
