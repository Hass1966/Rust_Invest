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

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarMap, VarBuilder, Optimizer, Module};
use candle_nn::rnn::{LSTM, LSTMConfig, LSTMState, RNN};
use candle_nn::linear;
use crate::ml::Sample;

/// Select the best available device: Metal GPU if on Apple Silicon, else CPU.
/// Called once at startup, cached for all LSTM operations.
fn select_device() -> Device {
    // Try Metal GPU first (Apple Silicon M1/M2/M3/M4)
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                println!("    [LSTM] Using Metal GPU acceleration");
                return device;
            }
            Err(e) => {
                println!("    [LSTM] Metal GPU unavailable ({}), falling back to CPU", e);
            }
        }
    }
    println!("    [LSTM] Using CPU");
    Device::Cpu
}

/// Lazy-initialised device — Metal GPU if available, else CPU.
/// Using std::sync::OnceLock so we only probe Metal once.
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
            input_size: 83,
            hidden_size: 32,
            seq_length: 20,
            learning_rate: 0.001,
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
    pub fn train(&mut self, samples: &[Sample], val_samples: &[Sample]) -> Result<TrainResult, candle_core::Error> {
        let seq_len = self.config.seq_length;
        let n_features = self.config.input_size;

        // Build sequences: each sequence is seq_len consecutive samples
        let train_seqs = build_sequences(samples, seq_len);
        let val_seqs = build_sequences(val_samples, seq_len);

        if train_seqs.is_empty() || val_seqs.is_empty() {
            return Ok(TrainResult {
                final_train_loss: f64::NAN,
                final_val_loss: f64::NAN,
                best_val_loss: f64::NAN,
                epochs_trained: 0,
                val_accuracy: 0.0,
            });
        }

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

        println!("    [LSTM] Training on {} sequences, validating on {}", train_seqs.len(), val_seqs.len());

        for epoch in 0..self.config.epochs {
            // Mini-batch training
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..train_seqs.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_seqs.len());
                let batch = &train_seqs[batch_start..batch_end];
                let batch_size = batch.len();

                // Build input tensor [batch_size, seq_len, n_features]
                let mut input_data = Vec::with_capacity(batch_size * seq_len * n_features);
                let mut labels_data = Vec::with_capacity(batch_size);

                for seq in batch {
                    for step in &seq.features {
                        input_data.extend_from_slice(step);
                    }
                    labels_data.push(seq.label);
                }

                let input = Tensor::from_vec(
                    input_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                    (batch_size, seq_len, n_features),
                    get_device(),
                )?;
                let labels = Tensor::from_vec(
                    labels_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                    (batch_size, 1),
                    get_device(),
                )?;

                // Forward pass: process each timestep through LSTM
                let logits = self.forward_batch(&input)?;

                // Binary cross-entropy with logits
                let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;

                // Backward + step
                optimizer.backward_step(&loss)?;

                epoch_loss += loss.to_scalar::<f32>()? as f64;
                n_batches += 1;
            }

            final_train_loss = epoch_loss / n_batches.max(1) as f64;

            // Validation loss
            let val_loss = self.evaluate_loss(&val_seqs)?;
            final_val_loss = val_loss;

            if epoch == 0 || (epoch + 1) % 10 == 0 || epoch == self.config.epochs - 1 {
                println!("    [LSTM] Epoch {:>3}/{}: train_loss={:.4}, val_loss={:.4}",
                    epoch + 1, self.config.epochs, final_train_loss, val_loss);
            }

            epochs_trained = epoch + 1;

            // Early stopping
            if val_loss < best_val_loss - 0.001 {
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

        // Compute validation accuracy
        let val_accuracy = self.evaluate_accuracy(&val_seqs)?;
        println!("    [LSTM] Validation accuracy: {:.1}%", val_accuracy * 100.0);

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
                input_data.extend_from_slice(step);
            }
            labels_data.push(seq.label);
        }

        let input = Tensor::from_vec(
            input_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            (batch_size, seq_len, n_features),
            get_device(),
        )?;
        let labels = Tensor::from_vec(
            labels_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
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

        let n_features = self.config.input_size;
        let seq_len = self.config.seq_length;
        let batch_size = seqs.len();

        let mut input_data = Vec::with_capacity(batch_size * seq_len * n_features);
        for seq in seqs {
            for step in &seq.features {
                input_data.extend_from_slice(step);
            }
        }

        let input = Tensor::from_vec(
            input_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            (batch_size, seq_len, n_features),
            get_device(),
        )?;

        let logits = self.forward_batch(&input)?;
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

    /// Predict P(up) for a single sequence of features
    pub fn predict_proba(&self, sequence: &[Vec<f64>]) -> Result<f64, candle_core::Error> {
        let seq_len = sequence.len();
        let n_features = self.config.input_size;

        let mut input_data = Vec::with_capacity(seq_len * n_features);
        for step in sequence {
            input_data.extend_from_slice(step);
        }

        let input = Tensor::from_vec(
            input_data.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            (1, seq_len, n_features),
            get_device(),
        )?;

        let logits = self.forward_batch(&input)?;
        let prob = candle_nn::ops::sigmoid(&logits)?;
        let val = prob.flatten_all()?.to_vec1::<f32>()?[0];

        Ok(val as f64)
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
pub fn build_sequences(samples: &[Sample], seq_len: usize) -> Vec<Sequence> {
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

        sequences.push(Sequence { features, label });
    }

    sequences
}

/// Run LSTM walk-forward evaluation on pre-built samples
/// Returns (overall_accuracy, recent_accuracy, final_prob)
pub fn walk_forward_lstm(
    symbol: &str,
    samples: &[Sample],
    config: &LSTMModelConfig,
    train_window: usize,
    test_window: usize,
    step: usize,
) -> Option<LSTMWalkForwardResult> {
    let seq_len = config.seq_length;

    if samples.len() < train_window + test_window + seq_len {
        println!("  {} — not enough samples for LSTM walk-forward", symbol);
        return None;
    }

    println!("  {} — LSTM walk-forward on {} samples × {} features (seq_len={})",
        symbol, samples.len(), config.input_size, seq_len);

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

        let train_samples = &samples[start..train_end];
        let test_samples = &samples[train_end..test_end];

        // Split train into train/val (85/15)
        let val_split = (train_samples.len() as f64 * 0.85) as usize;
        let (train_part, val_part) = train_samples.split_at(val_split);

        // Train LSTM
        let mut model = match LSTMModel::new(config.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("    [LSTM] Failed to create model: {}", e);
                return None;
            }
        };

        match model.train(train_part, val_part) {
            Ok(_) => {},
            Err(e) => {
                println!("    [LSTM] Training failed: {}", e);
                start += step;
                continue;
            }
        }

        // Evaluate on test set
        let test_seqs = build_sequences(test_samples, seq_len);
        let mut fold_correct = 0;

        for seq in &test_seqs {
            let prob = match model.predict_proba(&seq.features) {
                Ok(p) => p,
                Err(_) => 0.5,
            };
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

        // Get final prediction
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

    println!("    LSTM walk-forward: {} folds, {} test sequences", n_folds, total_tested);
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
