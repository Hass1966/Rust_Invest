# Honest Metrics Overhaul — Implementation Record

**Date:** 2026-03-30
**Commit:** `387689d`
**Branch:** `main`

---

## Overview

Transformed the Alpha Signal platform from inflated, lookahead-biased metrics to honest, out-of-sample performance reporting. The work covered 6 improvement items across 3 phases, touching 13 files (11 modified, 2 new).

---

## Phase 1: Frontend Quick Wins

### 1. Normalised Starting Capital

**File:** `frontend/src/pages/Simulator.tsx`

**Problem:** Buy & Hold started at £60k, Alpha Signal at £40k, SPY at £100k. You cannot compare equity curves that start at different levels.

**Changes:**
- Replaced `BH_TOTAL=60000`, `AS_TOTAL=40000`, `DEFAULT_CASH=4000` with a single `STARTING_CAPITAL = 100000`
- Converted `DEFAULT_BH` fixed amounts to proportional weights (e.g. AAPL £8,000/£60,000 → 0.133)
- Updated `runSimulation()` so all three strategies (Buy & Hold, Alpha Signal, SPY) start from the same user-configurable capital
- Added a starting capital input field (default £100,000)
- Updated `SimResult` interface: added `startingCapital`, removed `cashAmount`
- All summary cards now show the same "Invested £X" label

**Result:** All three equity curves start at the same Y value on day 0. Return percentages are directly comparable.

### 2. Actionable Accuracy Display

**File:** `frontend/src/pages/TrackRecord.tsx`

**Problem:** Headline "67.9% overall accuracy" was inflated by HOLD signals (trivially correct on low-volatility assets) and a loose definition of "correct" (any move > £0.01).

**Changes:**
- Added `actionableMetrics` useMemo computing BUY/SELL/HOLD accuracy, expected value, and profit factor from existing signal data
- Replaced headline "Overall Accuracy" with "Actionable Accuracy (BUY + SELL)"
- Added 5-card metrics grid:
  - BUY Accuracy (green)
  - SELL Accuracy (red)
  - Expected Value in basis points per signal
  - Profit Factor (sum of winners / sum of losers)
  - HOLD Accuracy (greyed out, labelled "non-actionable")
- Added collapsible "How we measure accuracy" section explaining thresholds
- Frontend prefers backend-provided values when available, falls back to client-side computation

### 3. Multi-Line Accuracy Trend Chart

**File:** `frontend/src/pages/TrackRecord.tsx`

**Problem:** Single rolling accuracy line didn't show BUY vs SELL performance separately.

**Changes:**
- Extended rolling chart to compute per-date:
  - 7-day rolling BUY-only accuracy (green, dashed)
  - 7-day rolling SELL-only accuracy (red, dashed)
  - 7-day rolling overall actionable accuracy (cyan, solid)
  - Cumulative accuracy from day 1 (purple, solid)
- Added 50% reference line
- Added chart legend

---

## Phase 2: Backend Accuracy Improvements

### 4. Minimum Return Thresholds + Fixed Holding Periods

**File:** `src/bin/serve.rs`

**Problem:** A BUY signal that captured a 0.02% move was marked "correct". In an upward-drifting market, random BUY signals are correct ~60% of the time. Resolution timing was arbitrary (whatever cached price existed 4+ hours later).

**Changes:**

#### A. Threshold-based correctness
Added `min_threshold_for_class()`:
| Asset Class | Minimum Move |
|-------------|-------------|
| Crypto | 1.0% |
| FX | 0.2% |
| Stocks/ETFs/Commodities | 0.5% |

- BUY correct only if price rose > threshold
- SELL/SHORT correct only if price fell > threshold
- HOLD correct only if |move| < threshold

#### B. Fixed holding period resolution
Added `resolution_ready()` function:
| Asset Class | Resolution Time |
|-------------|----------------|
| Stocks/ETFs | Next trading day close (20:00 UTC), skip weekends |
| FX | Next trading day close (17:00 UTC), skip weekends |
| Crypto | 24 hours after signal |

#### C. New API metrics
Extended `/api/v1/signals/truth` response with:
- `actionable_accuracy` — BUY + SELL + SHORT only
- `buy_accuracy`, `buy_signals`, `buy_correct`
- `sell_accuracy`, `sell_signals`, `sell_correct`
- `hold_accuracy`, `hold_signals`, `hold_correct`
- `expected_value_bps` — average directional return per actionable signal
- `profit_factor` — sum(winner returns) / sum(loser returns)

#### D. Updated types
Extended `SignalTruthData` interface in `frontend/src/lib/api.ts` with all new optional fields.

#### E. Database migration required
Old resolved signals must be cleared so they re-resolve with new thresholds:
```sql
UPDATE signal_history
SET outcome_price=NULL, pct_change=NULL, was_correct=NULL, resolution_ts=NULL
WHERE was_correct IS NOT NULL;
```

---

## Phase 3: Walk-Forward Backtester

### 5. Walk-Forward Binary

**New file:** `src/bin/backtest_walkforward.rs` (~500 lines)
**Modified:** `Cargo.toml` (added `[[bin]]` entry)

**Problem:** The `train` binary trains all models on the full 2021–2026 dataset. When the simulator generates "historical" signals for January 2022, those signals come from a model that learned from 4 years of future data. This is why the 5-year backtest showed unrealistically strong performance.

**Design:**
```
Window 1:  [TRAIN: 2021        ] → TEST: Q1 2022
Window 2:  [TRAIN: 2021–Q1 2022] → TEST: Q2 2022
Window 3:  [TRAIN: 2021–Q2 2022] → TEST: Q3 2022
...
Window 16: [TRAIN: 2021–Q3 2025] → TEST: Q4 2025
```

At each test window, models have **never seen** the test period data.

**Implementation:**
- 16 quarterly test windows (Q1 2022 → Q4 2025)
- Expanding-window training: each window uses ALL prior data
- Trains 3 models per window per asset: LinReg, LogReg, GBT
- Ensemble signal via accuracy-squared weighting (GBT gets 1.2x bonus)
- Uses Phase 2 thresholds for correctness determination
- Computes summary: buy/sell accuracy, expected value (bps), profit factor, Sharpe ratio, max drawdown
- Outputs `reports/walkforward_backtest.json`

**Runtime:** ~3–4 hours offline batch job. Not on-demand.

### 6. Walk-Forward API + Simulator Integration

**Files:** `src/bin/serve.rs`, `frontend/src/pages/Simulator.tsx`, `frontend/src/lib/api.ts`

**Backend:**
- New endpoint: `GET /api/v1/simulator/walkforward`
- Reads `reports/walkforward_backtest.json`, returns JSON (404 if missing)

**Frontend:**
- Simulator fetches walk-forward data on mount
- When walk-forward data available, "5-Year Backtest" tab uses walk-forward signals (no lookahead)
- Badge: "Walk-Forward Backtest (no lookahead)" vs "Live" on the live tab
- Green info banner: "These signals were generated using models that had never seen the test period data"
- Falls back gracefully to current biased backtest when walk-forward data unavailable

---

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `Cargo.toml` | Modified | Added `backtest_walkforward` binary entry |
| `src/bin/serve.rs` | Modified | Threshold correctness, fixed holding periods, new metrics, walk-forward endpoint |
| `src/bin/backtest_walkforward.rs` | **New** | Walk-forward backtester binary |
| `frontend/src/pages/Simulator.tsx` | Modified | Normalised capital, walk-forward integration, user capital input |
| `frontend/src/pages/TrackRecord.tsx` | Modified | Actionable accuracy, metrics grid, multi-line chart, methodology section |
| `frontend/src/lib/api.ts` | Modified | Extended types, new interfaces, `fetchWalkForwardData()` |
| `frontend/src/components/SignalCard.tsx` | Modified | Minor UI updates |
| `frontend/src/lib/plain-english.ts` | Modified | Minor updates |
| `frontend/src/pages/Dashboard.tsx` | Modified | Minor updates |
| `frontend/src/pages/DeepDive.tsx` | Modified | Minor updates |
| `reports/improved.json` | Modified | Updated report data |
| `reports/improvement_report.html` | Modified | Updated report |
| `CLAUDE.md` | **New** | Improvement plan with status tracking |

---

## Deployment Steps

After pulling the commit, run on the server:

```bash
# 1. Clear old resolved signals (re-resolve with new thresholds)
sqlite3 rust_invest.db "UPDATE signal_history SET outcome_price=NULL, pct_change=NULL, was_correct=NULL, resolution_ts=NULL WHERE was_correct IS NOT NULL;"

# 2. Retrain models
cargo run --release --bin train

# 3. Run walk-forward backtest (3-4 hours)
cargo run --release --bin backtest_walkforward

# 4. Rebuild frontend
cd frontend && npm run build && cd ..

# 5. Restart server
cargo run --release --bin serve
```

---

## Design Principles Applied

- **Honesty over flattery:** Real, unbiased numbers even if they look worse
- **Actionable over comprehensive:** Headline metrics reflect tradeable signals (BUY/SELL), not passive non-events (HOLD)
- **Apples-to-apples:** Same starting capital, same time period, same cost assumptions across all strategies
- **Out-of-sample only:** Walk-forward backtest signals come from models that never saw the test data
