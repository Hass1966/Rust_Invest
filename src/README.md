# Rust Invest v0.4 — Look-Ahead Fix + LSTM + Model Serialisation

## What Changed

### 1. CRITICAL: Look-Ahead Bias Fix (features.rs)
The market context features (VIX, sector ETFs, treasury yields) were using
**same-day** data to predict same-day direction — which is cheating because
you can't know today's VIX close before the market closes.

**Fix:** Market context index shifted by 1 day: `mi = i - 1`

This means we now use **yesterday's** VIX/sectors/treasuries to predict
**today's** direction. Walk-forward accuracy will drop from the suspicious
~97% to something realistic (expect 55-65%), but these will be **real** gains.

### 2. LSTM Model (lstm.rs — NEW)
Fourth model in the ensemble using Hugging Face's `candle-nn` LSTM.

- Architecture: `Input(83) → LSTM(hidden=32) → Linear(1) → Sigmoid`
- Sees 20-day sequences (not individual rows like LinReg/LogReg/GBT)
- Captures temporal patterns: "VIX rising for 3 days while RSI dropping"
- Trained with AdamW optimizer, binary cross-entropy loss
- Early stopping on validation loss (patience=8)
- Walk-forward evaluated independently, then folded into ensemble

### 3. Model Serialisation (model_store.rs — NEW)
Save/load trained model weights to disk.

- LinReg/LogReg: weights + bias to JSON
- LSTM: candle VarMap to `.safetensors`
- Version-stamped: auto-invalidates when feature count changes
- Stored in `models/` directory

### 4. Updated Ensemble (ensemble.rs)
- 4-model ensemble: LinReg, LogReg, GBT, LSTM
- Agreement now X/4 (was X/3) for stocks with LSTM
- Crypto still 3-model (not enough history for LSTM sequences)
- HTML report shows LSTM column

## Files to Deploy

### NEW files (add to src/):
- `src/lstm.rs` — LSTM model via candle-nn
- `src/model_store.rs` — Model weight persistence

### MODIFIED files (replace):
- `src/features.rs` — Look-ahead bias fix (1-day lag on market context)
- `src/ensemble.rs` — 4-model ensemble with LSTM integration
- `src/main.rs` — New module declarations + model cache reporting

### Cargo.toml — add these dependencies:
```toml
candle-core = "0.9"
candle-nn = "0.9"
safetensors = "0.4"
```

## Deployment Steps

1. **Add new dependencies to Cargo.toml:**
   ```toml
   candle-core = "0.9"
   candle-nn = "0.9"
   safetensors = "0.4"
   ```

2. **Copy all files to src/:**
   - `lstm.rs` (new)
   - `model_store.rs` (new)
   - `features.rs` (replace)
   - `ensemble.rs` (replace)
   - `main.rs` (replace)

3. **Build:**
   ```bash
   cargo build --release
   ```
   First build will be slower (compiling candle crate ~2-3 min).

4. **Run:**
   ```bash
   cargo run --release
   ```

5. **NO need to delete rust_invest.db** — market data is reused.

## Expected Results After Fix

| Metric | Before (leaked) | After (honest) |
|--------|-----------------|----------------|
| SPY WF accuracy | 97.5% | 55-62% |
| QQQ WF accuracy | 92.3% | 54-60% |
| Stock confidence | 20-50% | 5-15% |
| Quality ratings | All HIGH | Mix of HIGH/MEDIUM |

The numbers will be lower, but they're **real**. 60% directional accuracy
on daily equity moves, sustained in walk-forward, is genuinely valuable.

## Architecture

```
                ┌─────────────┐
                │  83 Features │
                │  (lag-1 mkt) │
                └──────┬──────┘
                       │
        ┌──────┬──────┼──────┬──────┐
        │      │      │      │      │
    ┌───▼──┐┌──▼───┐┌─▼──┐┌─▼────┐
    │LinReg││LogReg││ GBT ││ LSTM │
    │      ││      ││     ││20-day│
    │ row  ││ row  ││ row ││ seq  │
    └───┬──┘└──┬───┘└──┬──┘└──┬───┘
        │      │       │      │
        └──────┴───┬───┴──────┘
                   │
            ┌──────▼──────┐
            │   Weighted  │
            │   Ensemble  │
            │ (acc² wts)  │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ BUY/HOLD/   │
            │ SELL + Conf% │
            └─────────────┘
```
