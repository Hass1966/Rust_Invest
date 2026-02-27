# Phase 6: Backtester + Model Persistence Wiring

## Files Modified/Created

### NEW: `backtester.rs` (703 lines)
Walk-forward backtester that replays ensemble signals to compute real P&L metrics.

**Architecture:**
- Replays the EXACT same walk-forward procedure as `ensemble.rs` — trains LinReg, LogReg, GBT on each fold's training window, generates signals day-by-day on the test window
- Enters/exits positions based on ensemble probability thresholds
- Tracks equity curve, computes benchmark (buy-and-hold) for comparison

**Metrics computed:**
- Total return, annualised return, benchmark return, excess return (strategy − B&H)
- Sharpe ratio (annualised), max drawdown, annualised volatility
- Win rate, average win/loss, profit factor (gross profit / gross loss), expectancy (avg P&L per trade)
- Days in market vs total days, number of walk-forward folds
- Full equity curve and daily returns for visualisation

**Configuration (`BacktestConfig`):**
- `initial_capital`: $100,000 default
- `transaction_cost`: 10 bps per trade (0.1%)
- `buy_threshold`: 0.55 ensemble P(up) → enter long
- `sell_threshold`: 0.45 ensemble P(up) → exit to cash
- `min_accuracy`: 51% walk-forward accuracy required to act
- `position_size`: 1.0 (fully invested when long)

**Console output:** Per-asset result cards + summary table with verdicts (✓/~/✗)

**HTML output:** Summary table + inline SVG equity curve sparklines per asset

### REWRITTEN: `model_store.rs` (475 lines)
Complete persistence layer for all 4 model types.

**What changed from Phase 5:**
- Added `SerializableNode` enum that converts `gbt::Node` ↔ serde-friendly form
- Added `SavedGBT` struct with full tree serialization + normalisation params
- Added `SavedLSTMMeta` for LSTM weight tracking
- Added `norm_means`/`norm_stds` to all saved models (required for prediction-time normalisation)
- Added staleness detection: models auto-invalidate after 7 days
- Added `has_valid_models()` to check if a symbol can skip training
- Added `print_cache_status()` for visual cache status table
- Bumped `MODEL_VERSION` to 3

**Retrain policy:**
- Models older than `RETRAIN_DAYS` (7) → retrain
- Feature count mismatch → retrain
- `MODEL_VERSION` bump → retrain all
- Future: `--retrain` CLI flag

**Save/Load API:**
```rust
// Linear/Logistic regression
model_store::save_weights(symbol, "linreg", &weights, bias, n_feat, n_samples, accuracy, &means, &stds)
model_store::load_weights(symbol, "linreg") → SavedWeights

// Gradient Boosted Trees
model_store::save_gbt(symbol, &classifier, n_samples, accuracy, &means, &stds)
model_store::load_gbt(symbol) → (SavedGBT, GradientBoostedClassifier)

// LSTM (meta only — weights via candle VarMap)
model_store::save_lstm_meta(symbol, n_feat, hidden, seq_len, n_samples, accuracy, &means, &stds)
model_store::load_lstm_meta(symbol) → SavedLSTMMeta
```

### MODIFIED: `main.rs` (+132 lines)
- Added `mod backtester`
- Added cache status display at ML section start
- **Model persistence wiring:** After each stock's walk-forward ensemble training, the final fold's models are retrained on the last window and saved:
  - LinReg → `models/{symbol}_linreg.json`
  - LogReg → `models/{symbol}_logreg.json`
  - GBT → `models/{symbol}_gbt.json`
- Added **PART 6b: Backtester** section that runs `backtester::run_backtest()` for all stocks and crypto
- Updated report generation call to include backtest results
- Updated summary to show backtest count and cached model count

### MODIFIED: `report.rs` (+5 lines)
- Added `use crate::backtester`
- Updated `generate_html_report()` signature to accept `&[backtester::BacktestResult]`
- Inserted `backtester::backtest_html()` output before footer

## Pipeline Flow (Updated)

```
PART 1: Fetch crypto prices
PART 2: Backfill crypto history
PART 3: Technical analysis (crypto)
PART 4: Cross-coin comparison
PART 5: Stock quotes + history
  5b: Market indicators (VIX, treasuries, sectors)
  5c: ML section
    → print_cache_status() shows which models are fresh/stale/missing
    → Legacy ML pipelines (LinReg, LogReg, GBT)
PART 6: Ensemble walk-forward (rich features)
    → For each stock: walk-forward train → generate signal
    → NEW: Save final-fold models (LinReg, LogReg, GBT) to disk
    → For each crypto: walk-forward train → generate signal
  6b: Backtester (walk-forward replay)            ← NEW
    → For each stock: replay signals, compute P&L
    → For each crypto: replay signals, compute P&L
    → Print summary table with Sharpe, drawdown, excess returns
PART 7: Generate HTML report (now includes backtest section)
PART 8: Summary
```

## Key Design Decisions

1. **Backtester retrains per fold** — same as ensemble.rs, ensuring no look-ahead bias. The backtester sees exactly what the live system would see.

2. **Long-only strategy** — the backtester enters long when ensemble P(up) > 55% and exits when P(up) < 45%. No short selling (appropriate for stocks/crypto).

3. **Model persistence saves the LAST fold's models** — these are the most recent and the ones you'd use for daily predictions. Older fold models are discarded.

4. **Normalisation params saved with models** — critical for prediction: you must apply the same normalisation that was used during training. Without saving means/stds, loaded models would produce garbage predictions.

5. **GBT serialization uses enum mirroring** — `SerializableNode` mirrors `gbt::Node` exactly but derives `Serialize`/`Deserialize`. The conversion is O(n) in tree nodes and preserves all split information.

6. **7-day retrain window** — financial models degrade quickly. Weekly retraining balances freshness against computation cost. The cache system means daily runs skip training and go straight to prediction.

## Next Steps (Phase 7+)

- **Prediction mode:** When valid cached models exist, skip walk-forward entirely — just load models, normalise today's features, predict, and generate signals
- **Paper trading via IBKR API:** Use the saved models + daily feature generation to submit paper orders
- **Performance tracking:** Store backtest results in the database for trend analysis
- **CLI flags:** `--retrain` (force), `--predict-only` (skip training), `--paper-trade` (IBKR mode)
