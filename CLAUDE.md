# Alpha Signal — Improvement Plan

## Status Key
- [ ] Not started
- [~] In progress
- [x] Completed

---

## 1. Simulator: Portfolio-Level Allocation Framework [x]

**File:** `frontend/src/pages/Simulator.tsx`

Replaced per-asset siloed cash model with a unified portfolio allocator. Capital now flows across assets — when AAPL gets a SELL signal, its capital is freed to the pool and reallocated to BUY assets.

### What was done
- Inverse-volatility weighting (60-day trailing log returns)
- Correlation penalty: pairs with r > 0.7 have the smaller-weighted asset shrunk by 20%
- 30% single-asset cap to prevent concentration risk
- 2% rebalance threshold to avoid churn
- Transaction costs: 10bps stocks/ETFs, 25bps crypto, applied on every rebalance
- Sell-first-then-buy execution order (frees cash before buying)
- Stacked area chart showing daily allocation percentages per asset + cash
- Allocation table showing final-day weights, signals, and values per asset
- Tx Costs metric card added to risk metrics grid

---

## 2. Simulator: Normalise Starting Capital [x]

**File:** `frontend/src/pages/Simulator.tsx`

### Problem
Buy & Hold starts at £60k, Alpha Signal at £40k, SPY at £100k. You cannot compare equity curves that start at different levels. The question "if I invested £100k 5 years ago" has no clear answer.

### Required changes
- All three strategies must start at the **same user-chosen amount** (default £100,000)
- Buy & Hold: £100k spread across selected assets (proportional to current DEFAULT_BH weights)
- Alpha Signal: £100k managed by the portfolio allocation framework
- SPY benchmark: £100k in S&P 500
- Remove separate `BH_TOTAL`, `AS_TOTAL`, and the implicit £100k for SPY
- Add a single `STARTING_CAPITAL` constant (default 100000) or let the user pick
- Update `SplitResults` to respect the same starting capital
- Summary cards should all show the same "Invested £100,000" label

### Acceptance criteria
- All three equity curves start at the same Y value on day 0
- Return percentages are directly comparable
- User can optionally change the starting amount

---

## 3. Track Record: Fix Signal Accuracy Metrics [x]

**Files:** `src/bin/serve.rs` (resolution logic), `frontend/src/pages/TrackRecord.tsx`

### Problem 1: "Correct" is defined too loosely
Current logic (serve.rs line ~2210):
```rust
"BUY"  => current_price > sig.price_at_signal   // +£0.01 = correct
"SELL"  => current_price < sig.price_at_signal   // -£0.01 = correct
"HOLD" => pct_change.abs() < 1.0                 // trivially true for low-vol assets
```

A BUY signal that captures a 0.02% move is not a useful prediction. In an upward-drifting market, random BUY signals are "correct" ~60% of the time. HOLD signals on FX pairs are almost always "correct" because they rarely move >1% in 4 hours.

### Problem 2: Resolution timing is arbitrary
Signals resolve at whatever the cached price is 4+ hours later. This is noisy, inconsistent, and doesn't represent any real trading decision.

### Problem 3: HOLD signals inflate headline accuracy
HOLD is the most common signal type and the easiest to be "correct" on. Including it in the headline "67.9% overall accuracy" makes the number meaningless.

### Required backend changes (serve.rs)

#### A. Minimum return thresholds for correctness
Replace the binary price comparison with asset-class-specific thresholds:

```rust
let min_threshold = match asset_class {
    "crypto" => 1.0,   // 1.0% minimum move
    "fx"     => 0.2,   // 0.2% minimum move
    _        => 0.5,   // 0.5% minimum move (stocks, ETFs, commodities)
};

let was_correct = match sig.signal_type.as_str() {
    "BUY"   => pct_change > min_threshold,
    "SELL"  | "SHORT" => pct_change < -min_threshold,
    "HOLD"  => pct_change.abs() < min_threshold,
    _ => false,
};
```

#### B. Fixed holding period resolution
Instead of resolving at arbitrary 4-hour cache snapshots, resolve at consistent time horizons:
- **Stocks/ETFs:** Next trading day close (US market close, 20:00 UTC)
- **Crypto:** 24 hours after signal timestamp
- **FX:** Next trading day close (17:00 UTC NY close)

Do not resolve intraday. Wait for the appropriate market close.

#### C. New metrics to compute and return in `/api/v1/signals/truth`
Add these fields to the response:

```rust
// Existing (keep)
overall_accuracy: f64,        // but now excludes HOLD from headline

// New fields
buy_accuracy: f64,            // % of BUY signals that rose > threshold
sell_accuracy: f64,           // % of SELL/SHORT signals that fell > threshold
hold_accuracy: f64,           // shown separately, not in headline
expected_value_bps: f64,      // average return (bps) per actionable signal
profit_factor: f64,           // sum(gains from winners) / sum(losses from losers)
actionable_signals: usize,    // count of BUY + SELL + SHORT (excludes HOLD)
actionable_accuracy: f64,     // correct actionable / total actionable
```

**Expected value per signal:**
```rust
let actionable: Vec<_> = resolved.iter()
    .filter(|r| r.signal_type != "HOLD")
    .collect();
let returns: Vec<f64> = actionable.iter().map(|r| {
    match r.signal_type.as_str() {
        "BUY" => r.pct_change,            // long: positive pct = profit
        "SELL" | "SHORT" => -r.pct_change, // short: negative pct = profit
        _ => 0.0,
    }
}).collect();
let expected_value = returns.iter().sum::<f64>() / returns.len() as f64;
let winners: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
let losers: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
let profit_factor = if losers > 0.0 { winners / losers } else { f64::INFINITY };
```

### Required frontend changes (TrackRecord.tsx)

#### A. Replace headline accuracy
The big "67.9%" number should show **actionable accuracy** (BUY + SELL only, with minimum thresholds). Below it, show:

| Metric | Description |
|--------|-------------|
| BUY Accuracy | X% (N signals) |
| SELL Accuracy | X% (N signals) |
| Expected Value | +X bps per signal |
| Profit Factor | X.Xx |
| HOLD Accuracy | X% (shown separately, greyed out, with note "non-actionable") |

#### B. Add "How we measure" explainer
Below the headline metrics, add a collapsible section explaining:
- BUY is correct only if price rose > 0.5% (stocks) / 1.0% (crypto) / 0.2% (FX)
- SELL is correct only if price fell by the same thresholds
- Resolution happens at next trading day close, not arbitrary intraday
- HOLD signals are excluded from headline accuracy

#### C. Keep existing features
- Rolling 7-day chart (but recompute with new accuracy definition)
- Bubble chart, best/worst assets, signal history table
- Per-asset grid

---

## 4. Walk-Forward Backtester: Zero Lookahead Bias [x]

**New file:** `src/bin/backtest_walkforward.rs`
**Modified:** `frontend/src/pages/Simulator.tsx` (to display walk-forward results)

### Why current backtesting has lookahead bias

The `train` binary trains all 6 models on the **full historical price dataset** (2021-2026). It learns patterns like "when RSI < 30 on AAPL, price recovers 72% of the time" — that 72% was computed across ALL 5 years.

When the simulator then generates "historical" signals for January 2022, those signals come from a model that learned from 4 years of future data (2022-2026). The features at each date are point-in-time correct (no future prices in RSI/SMA calculations), but the **model weights encode future knowledge**.

This is why the 5-year backtest shows unrealistically strong performance. The model literally knows the future when making "past" decisions.

The live tracker (since 15 March 2026) does NOT have this problem — those signals are genuinely out-of-sample.

### Walk-forward design

```
Timeline: |----2021----|----2022----|----2023----|----2024----|----2025----|--2026--|

Window 1:  [TRAIN: 2021      ] → TEST: 2022-Q1
Window 2:  [TRAIN: 2021-2022Q1] → TEST: 2022-Q2
Window 3:  [TRAIN: 2021-2022Q2] → TEST: 2022-Q3
...
Window N:  [TRAIN: 2021-2025  ] → TEST: 2026-Q1
```

At each test window, the model has **never seen** the test period data. Models are retrained from scratch using only prior data (expanding window), signals generated for the next quarter, then roll forward.

### Implementation

```
backtest_walkforward binary:

1. Load full price history for all assets
2. Define test windows: quarterly (Q1-Q4) from 2022 through 2025
   - Minimum 252 trading days of training data before first test window
3. For each test window:
   a. Slice training data: everything BEFORE the test window start
   b. Train all 6 models (LinReg, LogReg, GBT, LSTM, RegimeEnsemble, TFT) on training slice
   c. Save temporary model weights to memory (not disk)
   d. Generate signals for each trading day in the test window
      - Feature construction uses only data up to that day (already correct)
      - Model weights trained on pre-window data only (now also correct)
   e. Record: date, asset, signal, entry_price, exit_price (next-day close), pct_return
   f. Discard temporary weights, advance to next window
4. Concatenate all out-of-sample signals into a single results file
5. Save to: reports/walkforward_backtest.json
```

### Output schema (reports/walkforward_backtest.json)
```json
{
  "generated_at": "2026-03-29T12:00:00Z",
  "windows": [
    {
      "train_end": "2021-12-31",
      "test_start": "2022-01-03",
      "test_end": "2022-03-31",
      "signals_generated": 450,
      "buy_accuracy": 54.2,
      "sell_accuracy": 51.8
    }
  ],
  "signals": [
    {
      "date": "2022-01-03",
      "asset": "AAPL",
      "signal": "BUY",
      "entry_price": 182.01,
      "exit_price": 179.70,
      "pct_return": -1.27,
      "was_correct": false,
      "train_window_end": "2021-12-31"
    }
  ],
  "summary": {
    "total_signals": 5200,
    "buy_accuracy": 55.1,
    "sell_accuracy": 52.3,
    "expected_value_bps": 12.4,
    "profit_factor": 1.18,
    "sharpe_ratio": 0.82,
    "max_drawdown_pct": 18.4
  }
}
```

### Frontend integration
- Add a new data source: `GET /api/v1/simulator/walkforward`
  - Serves the pre-computed `reports/walkforward_backtest.json`
- In the Simulator, the "5-Year Backtest" tab should use walk-forward signals instead of current biased signals
- The equity curve will be worse — and that's the honest, correct result
- Add a note: "These signals were generated using models that had never seen the test period data"

### Runtime considerations
- Walk-forward retraining is expensive: 16 windows x 6 models x ~80 assets
- Run as an offline batch job (like `train`), not on-demand
- Schedule monthly or when models are retrained
- Cache results in `reports/walkforward_backtest.json`

---

## 5. Simulator: Serve Walk-Forward Signals [x]

**Files:** `src/bin/serve.rs`, `frontend/src/pages/Simulator.tsx`

### Backend
- New endpoint: `GET /api/v1/simulator/walkforward`
- Reads `reports/walkforward_backtest.json` and returns it
- If file doesn't exist, return 404 with message to run the walk-forward binary

### Frontend
- The "5-Year Backtest" tab fetches walk-forward data
- Constructs equity curve from walk-forward signals using the portfolio allocation framework
- Shows alongside Buy & Hold and SPY (all normalised to same starting capital)
- "Live Tracking" tab continues using real-time signals (already unbiased)
- Add badge/label distinguishing "Walk-Forward Backtest (no lookahead)" from "Live"

---

## 6. Track Record: Show Accuracy Trend Over Time [x]

**File:** `frontend/src/pages/TrackRecord.tsx`

### Problem
With only ~14 days of live data, rolling accuracy is noisy. As data accumulates, the trend matters more than the current snapshot.

### Required changes
- Extend the rolling accuracy chart to show BUY-only and SELL-only accuracy as separate lines
- Add a cumulative accuracy line (expanding window, not just 7-day rolling)
- Colour-code periods: green when above 55%, amber 50-55%, red below 50%
- Add sample size annotation: show number of signals in each rolling window

---

## Implementation Priority

### Phase 1 — Quick wins (frontend only, no model changes)
1. [x] Portfolio allocation framework (Simulator.tsx) — DONE
2. [x] Normalise starting capital to £100k across all strategies — DONE
3. [x] Frontend: separate BUY/SELL accuracy display, remove HOLD from headline — DONE

### Phase 2 — Backend accuracy improvements
4. [x] Minimum return thresholds for signal correctness — DONE
5. [x] Fixed holding period resolution (next-day close) — DONE
6. [x] New metrics: expected value, profit factor, actionable accuracy — DONE
7. [x] Track Record frontend updates to display new metrics — DONE

### Phase 3 — Walk-forward backtester (major)
8. [x] New `backtest_walkforward` binary — DONE
9. [x] Walk-forward endpoint in serve.rs — DONE
10. [x] Simulator frontend integration with walk-forward data — DONE
11. [ ] Monthly scheduling for walk-forward reruns

---

## Design Principles

- **Honesty over flattery:** Show real, unbiased numbers even if they look worse. Users trust honest platforms; they abandon ones that look too good to be true.
- **Actionable over comprehensive:** Headline metrics should reflect signals a user would actually trade on (BUY/SELL), not passive non-events (HOLD).
- **Apples-to-apples:** Every comparison (Alpha Signal vs Buy & Hold vs SPY) must use the same starting capital, same time period, same cost assumptions.
- **Out-of-sample only:** Any backtest result shown to users must come from models that never saw the test data during training.
