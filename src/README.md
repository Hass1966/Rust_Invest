# Rust_Invest Diagnostics — Phase 1: Accuracy Deep Dive

## Files in This Patch

| File | Action | What Changed |
|------|--------|-------------|
| `Cargo.toml` | **REPLACED** — copy to project root | Adds `metal`, `accelerate`, `rayon` |
| `diagnostics.rs` | **NEW** — drop into `src/` | The entire diagnostics module |
| `main.rs` | **REPLACED** — copy over `src/main.rs` | 4 small additions (diagnostics wiring) |
| `report.rs` | **REPLACED** — copy over `src/report.rs` | 3 small additions (diagnostics HTML) |
| `lstm.rs` | **REPLACED** — copy over `src/lstm.rs` | Metal GPU device selection |
| `gbt.rs` | **REPLACED** — copy over `src/gbt.rs` | Rayon parallel feature scanning |
| `main.rs.backup` | Reference | Your original `main.rs` |
| `report.rs.backup` | Reference | Your original `report.rs` |
| `lstm.rs.backup` | Reference | Your original `lstm.rs` |

## How to Install

```bash
# From your Rust_Invest project root:

# Step 1: Replace Cargo.toml (adds Metal + Accelerate + rayon)
cp diagnostics_patch/Cargo.toml Cargo.toml

# Step 2: Drop in new/updated source files
cp diagnostics_patch/diagnostics.rs src/diagnostics.rs
cp diagnostics_patch/main.rs src/main.rs
cp diagnostics_patch/report.rs src/report.rs
cp diagnostics_patch/lstm.rs src/lstm.rs
cp diagnostics_patch/gbt.rs src/gbt.rs

# Step 3: Build (first build takes longer — compiling Metal shaders)
cargo build --release
```

Building with `--release` is important now. The rayon parallelism and Metal GPU really only shine with optimisations enabled. Debug builds won't show the speedup.

If cargo build fails, you can restore:
```bash
cp diagnostics_patch/main.rs.backup src/main.rs
cp diagnostics_patch/report.rs.backup src/report.rs
cp diagnostics_patch/lstm.rs.backup src/lstm.rs
rm src/diagnostics.rs
# Restore your original Cargo.toml from git
git checkout Cargo.toml src/gbt.rs
```

## GPU & Parallelism Acceleration

This patch adds three layers of acceleration targeting different parts of your pipeline:

### 1. Metal GPU for LSTM (candle-core `metal` feature)

**What changed:** `lstm.rs` — replaced `const DEVICE: Device = Device::Cpu` with runtime Metal detection via `Device::new_metal(0)`. All LSTM tensor operations (matmul in forward pass, gradient computation in backward pass, AdamW optimizer steps) now execute on your Mac Mini's GPU.

**Why it matters:** The LSTM is the most compute-intensive model in your ensemble. It processes 20-step sequences through matrix multiplications at every timestep. On CPU, this is the bottleneck. On Metal GPU, the tensor ops are dispatched to Apple's GPU shader cores which handle parallel matrix arithmetic natively.

**What you'll see:** Console output will print `[LSTM] Using Metal GPU acceleration` at startup. LSTM training epochs should be noticeably faster, especially the forward/backward passes per batch.

**How it works:** `OnceLock` probes Metal once at first use and caches the device for all subsequent operations. If Metal isn't available (e.g., building on Linux CI), it falls back to CPU automatically. Zero code changes needed for different platforms.

### 2. Apple Accelerate for CPU Linear Algebra (candle-core `accelerate` feature)

**What changed:** `Cargo.toml` — added `features = ["metal", "accelerate"]` to candle-core.

**Why it matters:** Even with Metal GPU for LSTM, the LinReg and LogReg models train on CPU (they use your hand-written gradient descent, not candle tensors). But candle's internal CPU operations still benefit from Accelerate — Apple's vecLib/BLAS gives hardware-optimised SIMD routines for any tensor ops that remain on CPU. This is a free speedup just from the feature flag.

### 3. Rayon Parallel Feature Scanning for GBT

**What changed:** `gbt.rs` — the `build_tree()` function's feature-scan loop now uses `rayon::par_iter()` instead of a sequential `for feat in 0..n_features`.

**Why it matters:** GBT builds 80 trees, and each tree evaluates all 83 features at every split node. The feature scan (sort indices by feature value, scan for best split point) is embarrassingly parallel — each feature is completely independent. With rayon, all CPU cores work simultaneously.

**Speedup estimate:** On your Mac Mini (likely 10+ cores), this should give roughly 4-6x speedup on GBT tree building. The improvement scales with core count and feature count. With 83 features, there's plenty of parallel work to distribute.

**How it works:** `(0..n_features).into_par_iter()` replaces `for feat in 0..n_features`. Each thread finds its best split for its subset of features, then rayon reduces to the global best. The algorithm produces identical results to the sequential version — same trees, same predictions, same accuracy. Just faster.

### Where Each Model Spends Time

| Model | Compute | Acceleration |
|-------|---------|-------------|
| LinReg | CPU gradient descent (3000 epochs) | No change (already fast, ~ms) |
| LogReg | CPU gradient descent (3000 epochs) | No change (already fast, ~ms) |
| GBT | CPU tree building (80 trees × 83 features) | **Rayon parallel** (~4-6x faster) |
| LSTM | Tensor matmul (20 timesteps × 40 epochs) | **Metal GPU** (~3-10x faster) |
| Diagnostics | Runs all 3 CPU models per fold | **Rayon parallel** GBT within each fold |

### Important: Build with --release

```bash
# Debug build (slow — no LLVM optimisations, rayon overhead dominates):
cargo build        # DON'T use for performance

# Release build (fast — LLVM optimises, rayon shines):
cargo build --release
cargo run --release
```

## What Changed in main.rs (4 additions)

1. **Line 11**: Added `mod diagnostics;`
2. **Line 424**: Added `let mut all_diagnostics: Vec<diagnostics::SymbolDiagnostics> = Vec::new();`
3. **Line ~458**: Before the existing `walk_forward_samples` call, added diagnostic collection:
   ```rust
   if let Some(diag) = diagnostics::run_diagnostics(...) {
       diagnostics::print_diagnostics(&diag);
       all_diagnostics.push(diag);
   }
   ```
4. **Line ~669**: Added `&all_diagnostics` parameter to `generate_html_report()`

## What Changed in report.rs (3 additions)

1. Added `use crate::diagnostics;`
2. Added `diagnostics_data: &[diagnostics::SymbolDiagnostics]` to function signature
3. Added diagnostics HTML section after trading signals
4. Added "Diagnostics" nav link

## What the Diagnostics Tell You

### 1. Per-Fold Accuracy (not just the average)

Your current code prints overall accuracy: "LinReg: 58.3%". But that hides whether the model is consistently 58% or whether it's 80% on some folds and 40% on others. The diagnostics print accuracy **for every fold**.

**What to look for:**
- If a model swings between 40% and 70% across folds → it's **unstable**, overfitting to specific time periods
- If a model is consistently 48-52% across all folds → it has **no edge**, it's noise
- If a model starts strong (early folds 60%+) but weakens (recent folds <52%) → the **market regime has changed** and the model hasn't adapted

### 2. Confusion Matrix (TP, FP, TN, FN)

Raw accuracy doesn't tell you if the model is right for the right reasons. The confusion matrix breaks down:

| | Predicted UP | Predicted DOWN |
|---|---|---|
| **Actual UP** | True Positive (TP) | False Negative (FN) |
| **Actual DOWN** | False Positive (FP) | True Negative (TN) |

**What to look for:**
- **Precision** (TP / (TP+FP)): When the model says BUY, how often is it right?
- **Recall** (TP / (TP+FN)): Of all UP days, how many did the model catch?
- A model with 60% accuracy but 45% precision is **dangerous** — it says BUY too often
- If FP >> FN, the model is **trigger-happy** (always predicts UP)

### 3. Bullish Bias Detection

This is probably the #1 issue at 56-61% accuracy. If the market went up 55% of days and your model predicts UP 80% of the time, it's not "learning" — it's just betting the base rate and getting lucky on the majority class.

**What to look for:**
- Compare "Predicts UP %" with "Actual UP %"
- If difference > 15%: **HEAVY BIAS** — the model is essentially a coin that always lands heads
- If difference > 8%: **BIAS** — needs threshold adjustment or class weighting

**How to fix:**
- Add class weighting: multiply down-day loss by `(n_up / n_down)` during training
- Adjust decision threshold: instead of `predict > 0.5`, use `predict > 0.55`
- Use balanced accuracy instead of raw accuracy

### 4. Feature Importance (GBT)

Shows which of your 83 features the GBT actually uses to make decisions.

**What to look for:**
- **Top 5 features**: These drive decisions. Are they sensible (RSI, volatility) or noise (day_of_week)?
- **Bottom 10 features** with importance < 0.002: These are **noise**. The GBT never splits on them. They add dimensionality without signal.
- **Calendar features** (day_of_week, month): If these rank high, the model may be overfitting to spurious patterns
- **Lagged features** (lag1_return, lag2_return): If these dominate, the model is doing simple mean-reversion, not learning deeper patterns

**How to fix:**
- Prune features with importance < 0.002
- If you have 83 features but only 15 matter, dropping the other 68 reduces overfitting
- After pruning, retrain and compare accuracy. You should see either no change (confirming they were noise) or improvement (they were hurting)

### 5. Model Contribution (Leave-One-Out)

Tests whether each model helps or hurts the ensemble by removing it and measuring accuracy without it.

**What to look for:**
- If removing LinReg **increases** accuracy: LinReg is a **drag**. Its predictions are worse than random when combined with the others.
- If removing a model barely changes accuracy: It's **redundant** — the other models already capture what it knows
- If removing a model **decreases** accuracy by 2%+: It's **valuable** — keep it

**How to fix:**
- Remove models that drag accuracy down
- A 2-model ensemble of LogReg + GBT might outperform a 3-model ensemble if LinReg is adding noise
- Alternatively, reduce the weight of the dragging model rather than removing it entirely

## Reading the Console Output

When you run `cargo run`, you'll see blocks like this for each stock:

```
╔═══════════════════════════════════════════════════════════════════╗
║  DIAGNOSTICS: AAPL                                               ║
║  83 features, 8 folds, 240 test samples                         ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  DATA BALANCE: 53.2% of test days were UP                        ║
║                                                                   ║
║  PER-FOLD ACCURACY:                                               ║
║   Fold   LinReg   LogReg      GBT   Ensemb      N                ║
║      1    56.7%    53.3%    60.0%    56.7%     30                 ║
║      2    50.0%    46.7%    53.3%    50.0%     30                 ║
║      ...                                                          ║
║                                                                   ║
║  CONFUSION MATRICES:                                              ║
║    LinReg :  TP= 72  FP= 38  TN= 68  FN= 62                    ║
║              Prec=65.5%  Recall=53.7%  F1=59.0%                  ║
║                                                                   ║
║  BULLISH BIAS CHECK:                                              ║
║    Actual UP rate:     53.2%                                      ║
║    LinReg predicts UP: 45.8%  ✓ Balanced                         ║
║    LogReg predicts UP: 72.1%  ⚠ BULL BIAS                       ║
║    GBT predicts UP:    55.0%  ✓ Balanced                         ║
║                                                                   ║
║  MODEL CONTRIBUTION:                                              ║
║    Full ensemble:           56.3%                                 ║
║    Without LinReg:          57.1%  (DRAGS — consider removing)   ║
║    Without LogReg:          55.8%  (marginal help)                ║
║    Without GBT:             52.9%  (HELPS — keep it)             ║
║                                                                   ║
║  TOP 20 FEATURES:                                                ║
║     1 volatility_20d            0.0834  ████████████████          ║
║     2 RSI_14                    0.0621  ████████████              ║
║     ...                                                           ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Reading the HTML Report

Open `report.html` and scroll to the **Diagnostics** section. You'll see:

1. **Per-fold accuracy table** with color coding (green ≥55%, yellow ≥52%, orange ≥50%, red <50%)
2. **Bar chart** showing fold-by-fold accuracy per model
3. **Confusion matrix cards** for each model with precision/recall/F1
4. **Bias detection table** comparing predicted vs actual UP rates
5. **Model contribution table** showing leave-one-out results
6. **Feature importance bar chart** (top 25 features)
7. **Expandable full feature table** with prune/keep recommendations
8. **Automated recommendations** with color-coded severity

## What to Fix Based on Results

### Scenario A: Heavy Bullish Bias Detected
→ Your model predicts UP 75% of the time but actual UP rate is 53%
→ **Fix**: Add class weighting to training. In `ensemble.rs`, weight down-day samples higher.
→ **Quick fix**: Change `ensemble_signal()` buy threshold from 0.55 to 0.60

### Scenario B: LinReg is Dragging the Ensemble
→ Removing LinReg improves accuracy by 2%+
→ **Fix**: Either remove LinReg from the ensemble entirely, or reduce its weight. In `ensemble_signal()`, you could multiply `lin_weight` by 0.5.

### Scenario C: 40+ Features Have Near-Zero Importance
→ Calendar and many lagged features contribute nothing
→ **Fix**: Create a `PRUNE_LIST` and skip those features in `build_rich_features()`. Start with features below 0.002 importance.

### Scenario D: High Fold Variance (40% to 70%)
→ Model is unstable across time periods
→ **Fix**: Increase training window (more data per fold), or add L2 regularisation to LinReg/LogReg, or reduce GBT max_depth from 4 to 3.

### Scenario E: Accuracy Barely Above 50%
→ Ensemble is 51-52%, no individual model above 53%
→ **Fix**: The features aren't predictive enough for this asset. Consider: adding fundamental data, using longer prediction horizons (5-day instead of 1-day), or focusing only on high-volatility regimes where technical signals have more edge.

## Performance Impact

The diagnostics module runs a **second** walk-forward pass per stock. Without GPU acceleration, this roughly doubles training time. With the Metal + rayon acceleration in this patch:

- **LSTM folds** run on Metal GPU (~3-10x faster per fold)
- **GBT tree building** runs on all CPU cores via rayon (~4-6x faster)
- **LinReg/LogReg** are already sub-second and aren't bottlenecks

Net result: the diagnostic pass should add roughly 30-60% extra wall-clock time instead of doubling it. On a Mac Mini M4, expect the full pipeline (signals + diagnostics) to complete in similar time to what signals-only took on CPU before.

If you want to skip diagnostics for quick production runs:
```bash
# Comment out the diagnostics blocks in main.rs (search for "run_diagnostics")
# Or keep them — with GPU acceleration the overhead is acceptable
```

## Next Step: Chat 2

Once accuracy is at 60%+ consistently:
- Split into two binaries (trainer + dashboard)
- Add Forex pairs
- Build training dashboard
- Public signal dashboard
- Lambda deployment

One thing at a time. Accuracy first.
