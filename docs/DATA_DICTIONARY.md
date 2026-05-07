# Alpha Signal — Data Dictionary

Everything you need to connect to the data and know what each column means.

---

## Data Sources

| Source | Type | Location | Size | What it contains |
|--------|------|----------|------|-----------------|
| **PostgreSQL** | Database | `localhost:5434` / `alpha_signal` | ~9K rows | Live signals, portfolio, predictions |
| **Walk-forward JSON** | File | `reports/walkforward_backtest.json` | 368K signals | Out-of-sample backtest signals (2021-2025) |
| **Feature importance JSON** | File | `reports/feature_importance.json` | 30 features | LightGBM importance scores |
| **SQLite** | Database | `rust_invest.db` | ~285 assets | Historical prices (2021-present), read-only |

---

## 1. PostgreSQL — Connection Details

```
Host:     localhost
Port:     5434
Database: alpha_signal
User:     agent
Password: agent
```

### Tables

| Table | Rows | Purpose |
|-------|------|---------|
| `signals` | ~9,000 (growing daily) | One signal per asset per day, with resolution |
| `daily_portfolio` | ~32 | Daily portfolio value and returns |
| `predictions` | 0 (unused) | Reserved for prediction tracking |
| `signal_snapshots` | varies | Point-in-time UI snapshots |
| `sizing_rejections` | varies | Why position sizes were rejected |
| `agent_actions` | varies | Autonomous agent action log |
| `agent_config` | 1 | Agent configuration |
| `agent_metrics` | varies | Agent performance metrics |

---

## 2. `signals` Table (Main Table)

One row per asset per trading day. This is where all the analysis happens.

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| `id` | bigint | no | Auto-increment primary key | 4523 |
| `timestamp` | timestamptz | no | When signal was generated | 2026-05-07 08:30:00+01 |
| `asset` | text | no | Ticker symbol | "AAPL" |
| `asset_class` | text | no | Always "stock" (FX/crypto descoped) | "stock" |
| `signal_type` | text | no | The prediction | "BUY", "SELL", or "HOLD" |
| `price_at_signal` | float8 | no | Price when signal was made | 189.45 |
| `confidence` | float8 | no | Model confidence, 0.0 to 1.0 | 0.37 |
| `linreg_prob` | float8 | yes | Legacy: linear regression probability | 0.58 |
| `logreg_prob` | float8 | yes | Legacy: logistic regression probability | 0.62 |
| `gbt_prob` | float8 | yes | Legacy: gradient boosted tree probability | 0.55 |
| `outcome_price` | float8 | yes | Price at resolution. NULL = unresolved | 191.20 |
| `pct_change` | float8 | yes | % change: (outcome - signal) / signal * 100 | 0.92 |
| `was_correct` | boolean | yes | Did signal meet accuracy threshold? NULL = unresolved | true |
| `resolution_ts` | timestamptz | yes | When signal was resolved. NULL = pending | 2026-05-08 20:00:00+01 |
| `created_at` | timestamptz | yes | Row creation timestamp | 2026-05-07 08:30:00+01 |
| `signal_date` | date | yes | Calendar date of signal | 2026-05-07 |

### Indexes

| Index | Columns | Notes |
|-------|---------|-------|
| `signals_pkey` | `id` | Primary key |
| `idx_signals_asset_day` | `(asset, signal_date)` | **UNIQUE** — one signal per asset per day |
| `idx_signals_asset_ts` | `(asset, timestamp)` | Fast asset lookups |
| `idx_signals_ts` | `timestamp` | Fast time-range queries |
| `idx_signals_unresolved` | `resolution_ts WHERE NULL` | Partial index for pending signals |

### Correctness definitions

| Signal | Correct when | Threshold |
|--------|-------------|-----------|
| BUY | `pct_change > +0.5%` | Stocks/ETFs |
| SELL | `pct_change < -0.5%` | Stocks/ETFs |
| HOLD | `abs(pct_change) < 0.5%` | Stocks/ETFs |

Resolution timing: **next trading day close** (20:00 UTC for US stocks).

### Key queries

```sql
-- Total signals and date range
SELECT COUNT(*), MIN(signal_date), MAX(signal_date) FROM signals;

-- Accuracy by signal type (actionable only)
SELECT signal_type, COUNT(*) as n,
       ROUND(100.0 * AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 1) as accuracy
FROM signals WHERE was_correct IS NOT NULL
GROUP BY signal_type;

-- Per-asset accuracy (last 30 days, minimum 5 signals)
SELECT asset, COUNT(*) as n,
       ROUND(100.0 * AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 1) as accuracy
FROM signals
WHERE was_correct IS NOT NULL AND signal_type IN ('BUY','SELL')
  AND signal_date >= CURRENT_DATE - 30
GROUP BY asset HAVING COUNT(*) >= 5
ORDER BY accuracy DESC;

-- Unresolved signals (pending outcome)
SELECT asset, signal_type, confidence, price_at_signal, signal_date
FROM signals WHERE resolution_ts IS NULL ORDER BY signal_date DESC;
```

---

## 3. `daily_portfolio` Table

One row per calendar day. Tracks portfolio value over time.

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| `id` | bigint | no | Auto-increment PK | 32 |
| `date` | date | no | Calendar date (**UNIQUE**) | 2026-05-07 |
| `seed_value` | float8 | no | Starting portfolio value | 100000.0 |
| `portfolio_value` | float8 | no | End-of-day portfolio value | 102450.0 |
| `daily_return` | float8 | no | Day's return as decimal (0.01 = 1%) | 0.0082 |
| `cumulative_return` | float8 | no | Cumulative return since inception | 0.0245 |
| `signals_json` | jsonb | yes | All signals for this day (nested JSON) | {...} |
| `model_version` | integer | no | Model version used | 9 |
| `created_at` | timestamptz | yes | Row creation time | 2026-05-07 21:00:00+01 |

---

## 4. Walk-Forward Backtest JSON

**File:** `reports/walkforward_backtest.json`
**Size:** ~368,186 signals across 19 quarterly windows (2021-2025)
**Updated:** After running `cargo run --release --bin backtest_walkforward`

### Top-level structure

```
{
  "generated_at": "2026-05-07T07:52:57Z",
  "windows":  [ ... ],     ← 24 quarterly windows (some empty early ones)
  "signals":  [ ... ],     ← 368,186 individual signal objects
  "summary":  { ... }      ← Aggregate metrics
}
```

### `summary` object

| Field | Type | Description | Current value |
|-------|------|-------------|---------------|
| `total_signals` | int | Total out-of-sample signals | 368,186 |
| `buy_accuracy` | float | % of BUY signals correct | 57.4% |
| `sell_accuracy` | float | % of SELL signals correct | 59.5% |
| `expected_value_bps` | float | Avg return per signal (basis points) | +106.3 |
| `profit_factor` | float | sum(winners) / sum(losers) | 4.43x |
| `sharpe_ratio` | float | Risk-adjusted return | 7.21 |
| `max_drawdown_pct` | float | Maximum peak-to-trough decline | 36.4% |

### `windows[]` array — one per quarter

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `train_end` | string (date) | Last day of training data | "2021-12-31" |
| `test_start` | string (date) | First day of test window | "2022-01-03" |
| `test_end` | string (date) | Last day of test window | "2022-03-31" |
| `signals_generated` | int | Signals in this window | 19679 |
| `buy_accuracy` | float | BUY accuracy for this window (%) | 58.3 |
| `sell_accuracy` | float | SELL accuracy for this window (%) | 61.4 |

### `signals[]` array — one per asset per day (368K rows)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `date` | string (date) | Trading date | "2022-01-03" |
| `asset` | string | Ticker symbol | "AAPL" |
| `asset_class` | string | Always "stock" | "stock" |
| `signal` | string | "BUY", "SELL", or "HOLD" | "BUY" |
| `entry_price` | float | Price at signal | 182.01 |
| `exit_price` | float | Next-day close | 179.70 |
| `pct_return` | float | % change (exit/entry - 1) * 100 | -1.27 |
| `was_correct` | bool | Met 0.5% threshold? | false |
| `train_window_end` | string (date) | End of training data used | "2021-12-31" |
| `buy_probability` | float | P(up) from ensemble | 0.58 |
| `confidence` | float | Model confidence (0-1) | 0.32 |
| `ridge_return` | float | Ridge model's predicted return (%) | 0.85 |
| `lgbm_return` | float | LightGBM's predicted return (%) | 0.62 |
| `gru_return` | float | GRU's predicted return (disabled, noisy) | -0.43 |
| `ensemble_return` | float | Weighted average of models (%) | 0.71 |
| `linreg_prob` | float | Legacy (= buy_probability) | 0.58 |
| `logreg_prob` | float | Legacy (= buy_probability) | 0.58 |
| `gbt_prob` | float | Legacy (= buy_probability) | 0.58 |
| `lgbm_prob` | float | Legacy (= buy_probability) | 0.58 |
| `lstm_prob` | float | Legacy (= buy_probability) | 0.58 |
| `regime_prob` | float | Legacy (= buy_probability) | 0.58 |

**Notes:**
- `ridge_return` and `lgbm_return` are the useful model-level predictions
- `gru_return` is present but should be ignored (disabled from ensemble, produces noise)
- Legacy `*_prob` fields are all identical — they're placeholders from the old classification system
- `ensemble_return` is the combined prediction that drives the signal

---

## 5. Feature Importance JSON

**File:** `reports/feature_importance.json`
**Updated:** After running `cargo run --release --bin train`

### Structure

```
{
  "generated_at": "2026-05-06T05:44:39Z",
  "n_assets": 351,
  "n_features": 30,
  "ranked_features": [ ... ],
  "recommended_top_30": [ "BB_position", "volume_ratio_20d", ... ]
}
```

### `ranked_features[]` array

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | string | Feature name | "BB_position" |
| `rank` | int | Rank by importance (1 = most important) | 1 |
| `mean_importance` | float | Average LightGBM gain importance across all assets | 0.499 |
| `n_assets_present` | int | How many assets had this feature | 351 |

### Current top 30 features

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | BB_position | 0.499 | Bollinger Band position (0-1) |
| 2 | volume_ratio_20d | 0.023 | Today's volume / 20-day avg |
| 3 | vix_9d_ratio | 0.022 | VIX / 9-day VIX SMA |
| 4 | volume_ratio_5d | 0.021 | Today's volume / 5-day avg |
| 5 | vix_change_1d | 0.015 | 1-day VIX change |
| 6 | BB_width | 0.014 | Bollinger Band width (volatility proxy) |
| 7 | daily_range_pct | 0.013 | (High-Low)/Close as % |
| 8 | dollar_return_5d | 0.012 | 5-day dollar index return |
| 9 | volatility_5d | 0.012 | 5-day rolling std of returns |
| 10 | vix_term_spread | 0.012 | VIX - VIX3M spread |
| 11 | lag1_return | 0.011 | Yesterday's return |
| 12 | SPY_return_1d | 0.011 | S&P 500 1-day return |
| 13 | gold_return_5d | 0.011 | Gold 5-day return |
| 14 | rel_strength_vs_spy_1d | 0.011 | Stock return - SPY return (1d) |
| 15 | vol_ratio_5_20 | 0.010 | 5-day vol / 20-day vol |
| 16 | spy_ret_21d | 0.010 | SPY 21-day return |
| 17 | SPY_return_5d | 0.010 | S&P 500 5-day return |
| 18 | vix_sma10_dist | 0.009 | VIX distance from 10-day SMA |
| 19 | tnx_change_5d | 0.009 | 10Y Treasury yield 5-day change |
| 20 | atr_14d | 0.009 | 14-day Average True Range |
| 21 | RSI_delta_3d | 0.009 | 3-day RSI change |
| 22 | ret_63d | 0.009 | 63-day (quarterly) return |
| 23 | skew_delta_5d | 0.009 | 5-day change in CBOE Skew |
| 24 | obv_slope_10d | 0.009 | 10-day On-Balance Volume slope |
| 25 | ret_2d | 0.008 | 2-day return |
| 26 | gold_spy_ratio_10d | 0.008 | Gold/SPY price ratio 10d change |
| 27 | momentum_1d | 0.008 | 1-day price momentum |
| 28 | price_vs_52w_high | 0.008 | Price / 52-week high |
| 29 | hurst_exponent_est | 0.008 | Hurst exponent (mean-reversion proxy) |
| 30 | momentum_3d | 0.007 | 3-day price momentum |

**BB_position accounts for ~50% of total importance.** The model is primarily a Bollinger Band mean-reversion strategy.

---

## 6. SQLite — Price History (Read-Only)

**File:** `rust_invest.db`
**Used by:** `train` and `signal` binaries for historical prices

Not intended for R analysis (PostgreSQL is the live data source), but available if you need raw price history. Connect with any SQLite driver.

---

## 7. Computing Key Metrics

For reference, here's how the system defines the key metrics:

### Expected Value (per signal, in basis points)

```
For each actionable signal (BUY or SELL):
  if BUY:  trade_return = pct_change
  if SELL: trade_return = -pct_change

expected_value = mean(trade_return) * 100   (converts to bps)
```

### Profit Factor

```
winners = sum of trade_return where trade_return > 0
losers  = sum of abs(trade_return) where trade_return < 0
profit_factor = winners / losers
```

### Confidence Discrimination (diagnostic)

Split signals into confidence quartiles (Q1=lowest, Q4=highest). Healthy system: Q4 accuracy > Q1 accuracy by at least 5 percentage points.

### Walk-Forward Correctness

```
BUY correct:  pct_return > +0.5%
SELL correct: pct_return < -0.5%
HOLD correct: abs(pct_return) < 0.5%
```
