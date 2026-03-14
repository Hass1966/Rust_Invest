/// Temporal Fusion Transformer (Simplified) — Candle Implementation
/// ================================================================
///
/// A simplified TFT for multivariate time series classification:
///
///   Input: 30-day rolling window of all features per asset
///   → Variable Selection Network: learned attention over features
///   → Temporal Self-Attention: learns which past days matter most
///   → Output: P(up/down) for next day
///
/// This is a simplified version of the full TFT paper
/// (Lim et al., 2021) adapted for direction prediction.
///
/// Architecture:
///   [batch, seq_len, n_features]
///   → Variable Selection (linear + softmax → weighted features)
///   → Positional encoding (learned)
///   → Multi-head Self-Attention (2 heads)
///   → Feed-forward (GeLU activation)
///   → Linear → Sigmoid → P(up)
///
/// Implements train(), predict_proba(), evaluate() matching the ml.rs interface.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarMap, VarBuilder, Module, Optimizer};
use candle_nn::linear;
use crate::ml::Sample;

/// Lazy-initialised device
fn get_device() -> &'static Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        println!("    [TFT] Using CPU");
        Device::Cpu
    })
}

/// TFT configuration
#[derive(Clone, Debug)]
pub struct TFTConfig {
    pub n_features: usize,
    pub seq_length: usize,     // 30 days
    pub d_model: usize,        // internal dimension
    pub n_heads: usize,        // attention heads
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Default for TFTConfig {
    fn default() -> Self {
        Self {
            n_features: 83,
            seq_length: 30,
            d_model: 32,
            n_heads: 2,
            learning_rate: 0.001,
            epochs: 40,
            batch_size: 32,
        }
    }
}

/// The TFT model
pub struct TFTModel {
    // Variable selection: learns which features matter
    var_select_weights: candle_nn::Linear,  // n_features → n_features (then softmax)

    // Feature projection
    feature_proj: candle_nn::Linear,        // n_features → d_model

    // Positional encoding (learned)
    pos_encoding: Tensor,                   // [seq_length, d_model]

    // Self-attention components (simplified single-layer)
    query_proj: candle_nn::Linear,          // d_model → d_model
    key_proj: candle_nn::Linear,            // d_model → d_model
    value_proj: candle_nn::Linear,          // d_model → d_model

    // Feed-forward
    ff1: candle_nn::Linear,                 // d_model → d_model * 2
    ff2: candle_nn::Linear,                 // d_model * 2 → d_model

    // Output head
    output: candle_nn::Linear,              // d_model → 1

    varmap: VarMap,
    config: TFTConfig,
}

impl TFTModel {
    /// Create a new TFT model
    pub fn new(config: TFTConfig) -> Result<Self, candle_core::Error> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, get_device());

        let var_select_weights = linear(config.n_features, config.n_features, vb.pp("var_sel"))?;
        let feature_proj = linear(config.n_features, config.d_model, vb.pp("feat_proj"))?;

        // Learned positional encoding (init with small random values)
        let pos_encoding = Tensor::randn(
            0.0_f32, 0.02_f32,
            (config.seq_length, config.d_model),
            get_device(),
        )?;

        let query_proj = linear(config.d_model, config.d_model, vb.pp("q"))?;
        let key_proj = linear(config.d_model, config.d_model, vb.pp("k"))?;
        let value_proj = linear(config.d_model, config.d_model, vb.pp("v"))?;

        let ff1 = linear(config.d_model, config.d_model * 2, vb.pp("ff1"))?;
        let ff2 = linear(config.d_model * 2, config.d_model, vb.pp("ff2"))?;

        let output = linear(config.d_model, 1, vb.pp("out"))?;

        Ok(Self {
            var_select_weights,
            feature_proj,
            pos_encoding,
            query_proj,
            key_proj,
            value_proj,
            ff1,
            ff2,
            output,
            varmap,
            config,
        })
    }

    /// Forward pass: [batch, seq_len, n_features] → [batch, 1] logits
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _n_feat) = input.dims3()?;

        // Variable Selection: learn feature importance
        // Apply to each timestep: softmax(linear(x)) * x
        let reshaped = input.reshape((batch_size * seq_len, self.config.n_features))?;
        let var_weights = self.var_select_weights.forward(&reshaped)?;
        let var_weights = candle_nn::ops::softmax(&var_weights, D::Minus1)?;
        let selected = (reshaped * var_weights)?;
        let selected = selected.reshape((batch_size, seq_len, self.config.n_features))?;

        // Feature projection: n_features → d_model
        let projected = {
            let flat = selected.reshape((batch_size * seq_len, self.config.n_features))?;
            let proj = self.feature_proj.forward(&flat)?;
            proj.reshape((batch_size, seq_len, self.config.d_model))?
        };

        // Add positional encoding
        let pos = self.pos_encoding
            .unsqueeze(0)?
            .expand((batch_size, seq_len, self.config.d_model))?;
        let encoded = (projected + pos)?;

        // Self-Attention (simplified: single layer, single head)
        let d = self.config.d_model;
        let flat = encoded.reshape((batch_size * seq_len, d))?;

        let q = self.query_proj.forward(&flat)?.reshape((batch_size, seq_len, d))?;
        let k = self.key_proj.forward(&flat)?.reshape((batch_size, seq_len, d))?;
        let v = self.value_proj.forward(&flat)?.reshape((batch_size, seq_len, d))?;

        // Attention scores: Q * K^T / sqrt(d)
        let k_t = k.transpose(1, 2)?;
        let scale = (d as f64).sqrt();
        let scores = q.matmul(&k_t)?;
        let scores = (scores / scale)?;

        // Causal mask: only attend to past (lower triangular)
        // Build mask manually since tril may not be available
        let mut mask_data: Vec<f32> = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                mask_data.push(if j <= i { 0.0_f32 } else { -1e9_f32 });
            }
        }
        let mask = Tensor::from_vec(mask_data, (1, seq_len, seq_len), get_device())?
            .expand((batch_size, seq_len, seq_len))?;
        let scores = (scores + mask)?;

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attended = attn.matmul(&v)?;  // [batch, seq_len, d_model]

        // Residual connection
        let residual = (attended + encoded)?;

        // Feed-forward with GeLU
        let last_step = residual.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [batch, d_model]
        let ff_out = self.ff1.forward(&last_step)?;
        let ff_out = ff_out.gelu_erf()?;
        let ff_out = self.ff2.forward(&ff_out)?;

        // Residual + output
        let final_repr = (ff_out + last_step)?;
        let logits = self.output.forward(&final_repr)?;  // [batch, 1]

        Ok(logits)
    }

    /// Train on sequences
    pub fn train_model(
        &mut self,
        train_samples: &[Sample],
        val_samples: &[Sample],
    ) -> Result<TrainResult, candle_core::Error> {
        let seq_len = self.config.seq_length;
        let train_seqs = build_tft_sequences(train_samples, seq_len);
        let val_seqs = build_tft_sequences(val_samples, seq_len);

        if train_seqs.is_empty() || val_seqs.is_empty() {
            return Ok(TrainResult {
                final_train_loss: f64::NAN,
                final_val_loss: f64::NAN,
                epochs_trained: 0,
                val_accuracy: 0.0,
            });
        }

        let mut optimizer = candle_nn::AdamW::new_lr(
            self.varmap.all_vars(),
            self.config.learning_rate,
        )?;

        let mut best_val_loss = f64::MAX;
        let mut patience = 0;
        let max_patience = 8;
        let mut final_train_loss = 0.0;
        let mut final_val_loss = 0.0;
        let mut epochs_trained = 0;

        println!("    [TFT] Training on {} sequences, validating on {}", train_seqs.len(), val_seqs.len());

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..train_seqs.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_seqs.len());
                let batch = &train_seqs[batch_start..batch_end];
                let _bs = batch.len();

                let (input, labels) = build_tensors(batch, seq_len, self.config.n_features)?;
                let logits = self.forward(&input)?;
                let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;
                optimizer.backward_step(&loss)?;
                epoch_loss += loss.to_scalar::<f32>()? as f64;
                n_batches += 1;
            }

            final_train_loss = epoch_loss / n_batches.max(1) as f64;

            // Validation loss
            let (val_input, val_labels) = build_tensors(&val_seqs, seq_len, self.config.n_features)?;
            let val_logits = self.forward(&val_input)?;
            let val_loss_t = candle_nn::loss::binary_cross_entropy_with_logit(&val_logits, &val_labels)?;
            final_val_loss = val_loss_t.to_scalar::<f32>()? as f64;

            epochs_trained = epoch + 1;

            if epoch == 0 || (epoch + 1) % 10 == 0 {
                println!("    [TFT] Epoch {:>3}/{}: train_loss={:.4}, val_loss={:.4}",
                    epoch + 1, self.config.epochs, final_train_loss, final_val_loss);
            }

            if final_val_loss < best_val_loss - 0.001 {
                best_val_loss = final_val_loss;
                patience = 0;
            } else {
                patience += 1;
                if patience >= max_patience {
                    println!("    [TFT] Early stopping at epoch {}", epoch + 1);
                    break;
                }
            }
        }

        // Validation accuracy
        let val_accuracy = self.evaluate_accuracy(&val_seqs)?;

        Ok(TrainResult {
            final_train_loss,
            final_val_loss,
            epochs_trained,
            val_accuracy,
        })
    }

    /// Predict P(up) for a single sequence
    pub fn predict_proba(&self, sequence: &[Vec<f64>]) -> Result<f64, candle_core::Error> {
        let seq_len = sequence.len();
        let n_features = self.config.n_features;

        let mut input_data: Vec<f32> = Vec::with_capacity(seq_len * n_features);
        for step in sequence {
            for &v in step {
                input_data.push(v as f32);
            }
        }

        let input = Tensor::from_vec(input_data, (1, seq_len, n_features), get_device())?;
        let logits = self.forward(&input)?;
        let prob = candle_nn::ops::sigmoid(&logits)?;
        let val = prob.flatten_all()?.to_vec1::<f32>()?[0];

        Ok(val as f64)
    }

    /// Predict direction
    pub fn predict_direction(&self, sequence: &[Vec<f64>]) -> Result<bool, candle_core::Error> {
        Ok(self.predict_proba(sequence)? > 0.5)
    }

    fn evaluate_accuracy(&self, seqs: &[TFTSequence]) -> Result<f64, candle_core::Error> {
        if seqs.is_empty() { return Ok(0.0); }

        let (input, _) = build_tensors(seqs, self.config.seq_length, self.config.n_features)?;
        let logits = self.forward(&input)?;
        let probs = candle_nn::ops::sigmoid(&logits)?;
        let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;

        let mut correct = 0;
        for (i, seq) in seqs.iter().enumerate() {
            let predicted_up = probs_vec[i] > 0.5;
            let actual_up = seq.label > 0.5;
            if predicted_up == actual_up { correct += 1; }
        }

        Ok(correct as f64 / seqs.len() as f64)
    }

    /// Save model weights
    pub fn save(&self, path: &str) -> Result<(), candle_core::Error> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// Load model weights
    pub fn load(config: TFTConfig, path: &str) -> Result<Self, candle_core::Error> {
        let mut model = Self::new(config)?;
        model.varmap.load(path)?;
        Ok(model)
    }
}

// ════════════════════════════════════════
// Sequence Building
// ════════════════════════════════════════

pub struct TFTSequence {
    pub features: Vec<Vec<f64>>,  // [seq_len][n_features]
    pub label: f64,
}

pub struct TrainResult {
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub epochs_trained: usize,
    pub val_accuracy: f64,
}

pub fn build_tft_sequences(samples: &[Sample], seq_len: usize) -> Vec<TFTSequence> {
    if samples.len() < seq_len + 1 {
        return Vec::new();
    }

    let mut sequences = Vec::new();
    for i in seq_len..samples.len() {
        let features: Vec<Vec<f64>> = samples[i - seq_len..i]
            .iter()
            .map(|s| s.features.clone())
            .collect();
        let label = if samples[i].label > 0.0 { 1.0 } else { 0.0 };
        sequences.push(TFTSequence { features, label });
    }
    sequences
}

fn build_tensors(
    seqs: &[TFTSequence],
    seq_len: usize,
    n_features: usize,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let bs = seqs.len();

    let mut input_data: Vec<f32> = Vec::with_capacity(bs * seq_len * n_features);
    let mut labels_data: Vec<f32> = Vec::with_capacity(bs);

    for seq in seqs {
        for step in &seq.features {
            for &v in step {
                input_data.push(v as f32);
            }
        }
        labels_data.push(seq.label as f32);
    }

    let input = Tensor::from_vec(input_data, (bs, seq_len, n_features), get_device())?;
    let labels = Tensor::from_vec(labels_data, (bs, 1), get_device())?;

    Ok((input, labels))
}

// ════════════════════════════════════════
// Walk-Forward
// ════════════════════════════════════════

pub struct TFTWalkForwardResult {
    pub overall_accuracy: f64,
    pub recent_accuracy: f64,
    pub final_prob: f64,
    pub n_folds: usize,
    pub total_tested: usize,
}

/// Walk-forward evaluation for TFT
pub fn walk_forward_tft(
    symbol: &str,
    samples: &[Sample],
    config: &TFTConfig,
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<TFTWalkForwardResult> {
    let seq_len = config.seq_length;

    if samples.len() < train_window + test_window + seq_len {
        println!("  {} — not enough samples for TFT walk-forward", symbol);
        return None;
    }

    println!("  {} — TFT walk-forward on {} samples (seq_len={})", symbol, samples.len(), seq_len);

    let mut total_correct = 0_usize;
    let mut total_tested = 0_usize;
    let mut n_folds = 0;
    let mut last_fold_correct = 0;
    let mut last_fold_size = 0;
    let mut last_prob = 0.5;

    let mut start = 0;
    while start + train_window + test_window <= samples.len() {
        let train_end = start + train_window;
        let test_end = (train_end + test_window).min(samples.len());

        let train_samples = &samples[start..train_end];
        let test_samples = &samples[train_end..test_end];

        let val_split = (train_samples.len() as f64 * 0.85) as usize;
        let (train_part, val_part) = train_samples.split_at(val_split);

        let mut model = match TFTModel::new(config.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("    [TFT] Failed to create model: {}", e);
                return None;
            }
        };

        match model.train_model(train_part, val_part) {
            Ok(_) => {},
            Err(e) => {
                println!("    [TFT] Training failed: {}", e);
                start += step;
                continue;
            }
        }

        // Evaluate on test set
        let test_seqs = build_tft_sequences(test_samples, seq_len);
        let mut fold_correct = 0;

        for seq in &test_seqs {
            let prob = model.predict_proba(&seq.features).unwrap_or(0.5);
            let predicted_up = prob > 0.5;
            let actual_up = seq.label > 0.5;
            if predicted_up == actual_up { fold_correct += 1; }
        }

        let fold_size = test_seqs.len();
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
        return None;
    }

    let overall_acc = total_correct as f64 / total_tested as f64 * 100.0;
    let recent_acc = last_fold_correct as f64 / last_fold_size.max(1) as f64 * 100.0;

    println!("    TFT: {:.1}% (recent: {:.1}%)", overall_acc, recent_acc);

    Some(TFTWalkForwardResult {
        overall_accuracy: overall_acc,
        recent_accuracy: recent_acc,
        final_prob: last_prob.clamp(0.15, 0.85),
        n_folds,
        total_tested,
    })
}
