# Alpha Signal — System Guide

**Last updated:** 2026-05-07 | **Version:** Regression Ensemble v9.1

---

## TL;DR — The 60-Second Version

Alpha Signal is a **stock trading signal generator**. It trains ML models on historical prices, generates daily BUY/SELL/HOLD signals for ~285 assets, and serves them via a web dashboard.

```
train (weekly)  →  models/  →  signal (daily)  →  PostgreSQL  →  serve (always on)  →  Browser
```

**Start everything:** `sudo systemctl start rustinvest`
**Stop everything:** `sudo systemctl stop rustinvest`
**Check status:** `systemctl status rustinvest`

---

## Table of Contents

1. [How It Works (Plain English)](#1-how-it-works-plain-english)
2. [Architecture Diagram](#2-architecture-diagram)
3. [The 5 Binaries You Care About](#3-the-5-binaries-you-care-about)
4. [Starting, Stopping & Checking](#4-starting-stopping--checking)
5. [Training Models](#5-training-models)
6. [Generating Signals](#6-generating-signals)
7. [The Web Dashboard](#7-the-web-dashboard)
8. [How Often To Do Things](#8-how-often-to-do-things)
9. [The ML Pipeline (What The Models Actually Do)](#9-the-ml-pipeline)
10. [Database Layout](#10-database-layout)
11. [Feature Engineering](#11-feature-engineering)
12. [Walk-Forward Backtest](#12-walk-forward-backtest)
13. [File & Directory Map](#13-file--directory-map)
14. [Environment Variables](#14-environment-variables)
15. [Troubleshooting](#15-troubleshooting)
16. [Current Performance (2026-05-07)](#16-current-performance)

---

## 1. How It Works (Plain English)

1. **Once a week**, you run `train`. It downloads 5 years of stock prices, trains 2 ML models (Ridge regression + LightGBM) per stock, and saves the model files to `models/`.

2. **Once a day**, you run `signal`. It loads those saved models, looks at today's market data, and writes a BUY/SELL/HOLD signal for each stock into PostgreSQL.

3. **Always running** is `serve`. It's a web server that reads signals from PostgreSQL and shows them on a dashboard at `http://localhost:8888`.

That's it. Everything else is validation, monitoring, or diagnostics.

---

## 2. Architecture Diagram

```
                    ┌──────────────┐
                    │   train.rs   │  Weekly: ~4 hours
                    │  (offline)   │  Downloads prices, trains models
                    └──────┬───────┘
                           │ saves weights
                           ▼
                    ┌──────────────┐
                    │   models/    │  ~4,300 files
                    │  (on disk)   │  Ridge JSON + LightGBM text
                    └──────┬───────┘
                           │ loads weights
                           ▼
                    ┌──────────────┐
                    │  signal.rs   │  Daily: ~3 minutes
                    │  (offline)   │  Generates BUY/SELL/HOLD
                    └──────┬───────┘
                           │ writes signals
                           ▼
                    ┌──────────────┐
                    │  PostgreSQL  │  alpha_signal database
                    │  (port 5434) │  signals, portfolio, predictions
                    └──────┬───────┘
                           │ reads signals
                           ▼
┌───────────┐      ┌──────────────┐      ┌──────────────┐
│   nginx   │◄────►│  serve.rs    │      │  agent.rs    │
│ (port 8888)│      │ (port 8081)  │      │ (background) │
└─────┬─────┘      └──────────────┘      └──────────────┘
      │                                   Monitors accuracy,
      ▼                                   triggers retrains
┌──────────────┐
│   Browser    │
│  React SPA   │
│  Dashboard   │
└──────────────┘
```

---

## 3. The 5 Binaries You Care About

| Binary | What it does | How long | How often |
|--------|-------------|----------|-----------|
| `train` | Downloads prices, trains all models | ~4 hours | Weekly |
| `signal` | Loads models, generates today's signals | ~3 minutes | Daily |
| `serve` | Web API server (dashboard) | Runs forever | Always on |
| `backtest_walkforward` | Tests model accuracy without cheating | ~4 hours | Monthly |
| `agent` | Watches accuracy, triggers retrains | Runs forever | Always on |

### How to run each one

```bash
# Train all models (1-day horizon)
cargo run --release --bin train

# Train 5-day horizon models
cargo run --release --bin train -- --horizon 5

# Generate today's signals
cargo run --release --bin signal

# Start the web server
cargo run --release --bin serve

# Run walk-forward backtest
cargo run --release --bin backtest_walkforward
```

---

## 4. Starting, Stopping & Checking

### The serve binary (web dashboard)

```bash
# Start
sudo systemctl start rustinvest

# Stop
sudo systemctl stop rustinvest

# Restart (after retraining or code changes)
sudo systemctl restart rustinvest

# Check if it's running
systemctl status rustinvest

# View live logs
journalctl -u rustinvest -f

# View last 100 log lines
journalctl -u rustinvest -n 100
```

### The agent (autonomous monitor)

```bash
sudo systemctl start agent-alpha
sudo systemctl stop agent-alpha
systemctl status agent-alpha
```

### After code changes

```bash
# 1. Build the new binary
cargo build --release

# 2. Restart the service
sudo systemctl restart rustinvest
```

### Check everything is healthy

```bash
# Service status
systemctl status rustinvest agent-alpha nginx

# API health check
curl http://localhost:8081/api/v1/health

# PostgreSQL connection
psql -h localhost -p 5434 -U agent -d alpha_signal -c "SELECT count(*) FROM signals;"

# Model count
ls models/*.json models/*.txt 2>/dev/null | wc -l
```

---

## 5. Training Models

### Full retrain (both horizons)

This is the standard weekly procedure:

```bash
# Step 1: Train 1-day models (~4 hours)
cargo run --release --bin train 2>&1 | tee /tmp/retrain_1d.log

# Step 2: Train 5-day models (~6 hours)
cargo run --release --bin train -- --horizon 5 2>&1 | tee /tmp/retrain_5d.log

# Step 3: Restart serve to pick up new models
sudo systemctl restart rustinvest
```

### What training does

For each of ~285 stocks:
1. Loads price history (2021-present) from SQLite
2. Builds 30 features (Bollinger Bands, volume ratios, VIX, momentum, etc.)
3. Runs 20-fold walk-forward cross-validation with 5-day embargo
4. Trains **Ridge regression** (closed-form matrix solve, instant)
5. Trains **LightGBM** (gradient boosted trees, ~30 seconds per asset)
6. Saves model weights to `models/`
7. Saves feature importance to `reports/feature_importance.json`

### What gets saved

```
models/aapl_ridge.json          ← 1-day Ridge model
models/aapl_lgbm_reg.txt        ← 1-day LightGBM model
models/aapl_lgbm_mae.json       ← LightGBM validation error
models/5d_aapl_ridge.json       ← 5-day Ridge model
models/5d_aapl_lgbm_reg.txt     ← 5-day LightGBM model
```

### Training uses 6 CPU cores

Rayon is configured to use 6 threads, leaving 2 cores free for the OS and serve. You can still use the machine during training, but it'll be sluggish.

---

## 6. Generating Signals

### Daily signal generation

```bash
cargo run --release --bin signal
```

This takes ~3 minutes and:
1. Loads saved model weights from `models/`
2. Fetches latest prices from SQLite
3. For each stock, builds features and runs inference
4. **Multi-horizon check**: 1-day and 5-day models must agree on direction
5. **Model agreement**: Ridge and LightGBM must agree or signal becomes HOLD
6. **Confidence filter**: confidence < 0.25 becomes HOLD
7. Writes signals to PostgreSQL `signals` table

### Signal types

| Signal | Meaning | When emitted |
|--------|---------|-------------|
| **BUY** | Expected to rise >0.5% by next close | Both models agree UP, both horizons agree, confidence > 0.25 |
| **SELL** | Expected to fall >0.5% by next close | Both models agree DOWN, both horizons agree, confidence > 0.25 |
| **HOLD** | No confident prediction | Models disagree, horizons disagree, or low confidence |

### How accuracy is measured

- **BUY correct** = price actually rose > 0.5% by next trading day close
- **SELL correct** = price actually fell > 0.5% by next trading day close
- **HOLD** signals are excluded from headline accuracy (too easy to be "right")

---

## 7. The Web Dashboard

### URLs

| URL | What |
|-----|------|
| `http://localhost:8888` | Main dashboard (nginx proxy) |
| `http://localhost:8081` | Direct API access |
| `http://localhost:8888/app/` | React SPA |

### API Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/v1/signals` | Latest signals for all assets |
| `GET /api/v1/signals/truth` | Accuracy metrics (BUY%, SELL%, expected value, profit factor) |
| `GET /api/v1/portfolio` | Current portfolio state |
| `GET /api/v1/simulator/walkforward` | Walk-forward backtest results |
| `GET /api/v1/assets` | Asset list |
| `GET /api/v1/health` | System health |

### Frontend pages

| Page | What it shows |
|------|--------------|
| **Dashboard** | Today's signals, market overview |
| **Simulator** | 5-year backtest equity curves (walk-forward, no lookahead) |
| **Track Record** | Signal accuracy: BUY%, SELL%, expected value, profit factor |
| **My Portfolio** | Holdings, P&L (login required) |
| **System Health** | Agent status, data freshness |

---

## 8. How Often To Do Things

### Daily (automated or manual)

```bash
# Generate today's signals (~3 min)
cargo run --release --bin signal
```

Run this after US market close (20:00 UTC / 21:00 BST) so prices are final.

### Weekly (Sunday evening recommended)

```bash
# Full retrain: 1d + 5d models (~10 hours total)
cargo run --release --bin train 2>&1 | tee /tmp/retrain_1d.log
cargo run --release --bin train -- --horizon 5 2>&1 | tee /tmp/retrain_5d.log

# Restart serve to load new models
sudo systemctl restart rustinvest
```

### Monthly

```bash
# Walk-forward backtest (~4 hours) — validates model accuracy
cargo run --release --bin backtest_walkforward 2>&1 | tee /tmp/walkforward.log
```

### After code changes

```bash
cargo build --release
sudo systemctl restart rustinvest
```

### Recommended schedule

| When | What | Command |
|------|------|---------|
| Mon-Fri 21:00 BST | Generate signals | `cargo run --release --bin signal` |
| Sunday 18:00 BST | Full retrain (1d) | `cargo run --release --bin train` |
| Sunday 22:00 BST | Full retrain (5d) | `cargo run --release --bin train -- --horizon 5` |
| Monday 04:00 BST | Restart serve | `sudo systemctl restart rustinvest` |
| 1st of month | Walk-forward | `cargo run --release --bin backtest_walkforward` |

---

## 9. The ML Pipeline

### What the models predict

Each model predicts **next-day percentage return** for a given stock. Example: "AAPL will return +0.7% by tomorrow's close."

### Active models (v9.1)

| Model | Type | How it works | Speed |
|-------|------|-------------|-------|
| **Ridge** | Linear regression | Closed-form matrix solve (X'X + aI)^-1 X'y | Instant |
| **LightGBM** | Gradient boosted trees | 300 rounds, early stopping, 31 leaves | ~30s/asset |

**Disabled:** GRU (recurrent neural net) — produced 45-48% accuracy regardless of outcome, adding noise.

### How signals are generated from predictions

```
1. Ridge predicts:   +0.8%
2. LightGBM predicts: +0.6%
3. Ensemble average:  +0.7% (weighted by 1/MAE)

4. Model agreement check:
   - Ridge says UP, LightGBM says UP → AGREE → continue
   - If they disagree → HOLD (stop here)

5. Multi-horizon check:
   - 1-day model says UP
   - 5-day model says UP → AGREE → continue
   - If they disagree → HOLD (stop here)

6. Confidence check:
   - Confidence = 0.35 (> 0.25 threshold) → continue
   - If < 0.25 → HOLD (stop here)

7. Threshold check:
   - +0.7% > +0.5% threshold → BUY

Result: BUY with 35% confidence
```

### Ensemble weighting

Models are weighted by **inverse MAE** (mean absolute error). If Ridge has MAE=1.2 and LightGBM has MAE=0.9:
- Ridge weight = (1/1.2) / (1/1.2 + 1/0.9) = 0.43
- LightGBM weight = (1/0.9) / (1/1.2 + 1/0.9) = 0.57

Better models get more vote.

### Walk-forward training (no cheating)

Models are validated using **expanding window walk-forward**:

```
Window 1:  Train on 2021         → Test on Q2 2021 (never seen)
Window 2:  Train on 2021-Q2 2021 → Test on Q3 2021
Window 3:  Train on 2021-Q3 2021 → Test on Q4 2021
...
Window 19: Train on 2021-Q3 2025 → Test on Q4 2025

5-day embargo between train and test (prevents autocorrelation leak)
```

The model has **never seen** any test data during training. This is the honest accuracy number.

---

## 10. Database Layout

### PostgreSQL (alpha_signal, port 5434)

This is the **source of truth** for signals and portfolio.

```
Connection: psql -h localhost -p 5434 -U agent -d alpha_signal
Password: agent
```

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `signals` | One signal per asset per day | asset, signal_type, confidence, price_at_signal, was_correct |
| `signal_snapshots` | Point-in-time snapshots for UI | asset, signal, confidence, rsi, trend, price |
| `predictions` | For truth resolution tracking | asset, signal, price_at_prediction, was_correct |
| `daily_portfolio` | Portfolio weights per day | date, asset, allocation_weight, position_value |
| `sizing_rejections` | Why positions were rejected | asset, reason, requested_amount |

### SQLite (rust_invest.db, read-only)

Historical price data and cached market data. Used by `train` and `signal` binaries for reading prices. Not written to by the live system.

### Useful queries

```sql
-- Recent signals
SELECT asset, signal_type, confidence, price_at_signal, was_correct
FROM signals ORDER BY timestamp DESC LIMIT 20;

-- Accuracy by signal type
SELECT signal_type, COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy
FROM signals WHERE was_correct IS NOT NULL
GROUP BY signal_type;

-- Today's portfolio
SELECT asset, allocation_weight, position_value
FROM daily_portfolio WHERE date = CURRENT_DATE ORDER BY allocation_weight DESC;

-- Signal count by day
SELECT signal_date, COUNT(*) FROM signals
GROUP BY signal_date ORDER BY signal_date DESC LIMIT 14;
```

---

## 11. Feature Engineering

### The 30 active features (selected by LightGBM importance)

Reduced from 99 → 30 on 2026-05-07 based on cross-asset importance analysis.

| Category | Features | What they measure |
|----------|----------|------------------|
| **Price technical** | BB_position, BB_width, daily_range_pct, RSI_delta_3d, price_vs_52w_high | Where price sits relative to bands, ranges, highs |
| **Volume** | volume_ratio_20d, volume_ratio_5d, obv_slope_10d | Unusual volume activity |
| **Volatility** | volatility_5d, vol_ratio_5_20, atr_14d | Recent vs historical volatility |
| **Momentum** | momentum_1d, momentum_3d, lag1_return, ret_2d, ret_63d | Price momentum at multiple timeframes |
| **Market context** | SPY_return_1d/5d, spy_ret_21d, gold_return_5d, dollar_return_5d | Broad market and macro moves |
| **VIX** | vix_change_1d, vix_9d_ratio, vix_term_spread, vix_sma10_dist | Fear gauge and its term structure |
| **Cross-asset** | rel_strength_vs_spy_1d, gold_spy_ratio_10d | Stock vs market relative performance |
| **Statistical** | hurst_exponent_est, skew_delta_5d | Mean-reversion tendency, skew changes |
| **Macro** | tnx_change_5d | 10-year Treasury yield changes |

**BB_position is the dominant feature** (50% of LightGBM importance). This means the model heavily relies on Bollinger Band mean-reversion — buy when price is near the lower band, sell near the upper band.

---

## 12. Walk-Forward Backtest

### Latest results (2026-05-07, 30 features)

| Metric | Value |
|--------|-------|
| **BUY accuracy** | 57.4% |
| **SELL accuracy** | 59.5% |
| **Expected value** | +106 bps per signal |
| **Profit factor** | 4.43x |
| **Sharpe ratio** | 7.21 |
| **Max drawdown** | 36.4% |
| **Total signals** | 368,186 |

### Accuracy by quarter

| Period | BUY | SELL |
|--------|-----|------|
| Q2 2021 | 51.7% | 51.8% |
| Q3 2021 | 54.0% | 53.3% |
| Q4 2021 | 54.9% | 52.8% |
| Q1 2022 | 58.3% | 61.4% |
| Q2 2022 | 59.0% | 63.5% |
| Q3 2022 | 56.6% | 60.5% |
| Q4 2022 | 57.7% | 59.1% |
| Q1 2023 | 62.3% | 60.1% |
| Q2 2023 | 57.1% | 58.2% |
| Q3 2023 | 56.4% | 57.7% |
| Q4 2023 | 54.9% | 58.2% |
| Q1 2024 | 56.7% | 62.3% |
| Q2 2024 | 56.3% | 61.5% |
| Q3 2024 | 58.5% | 59.4% |
| Q4 2024 | 56.8% | 61.1% |
| Q1 2025 | 63.6% | 60.8% |
| Q2 2025 | 61.8% | 60.4% |
| Q3 2025 | 56.7% | 61.4% |
| Q4 2025 | 56.1% | 62.0% |

### How to re-run

```bash
cargo run --release --bin backtest_walkforward 2>&1 | tee /tmp/walkforward.log
# Results saved to: reports/walkforward_backtest.json
# Takes ~4 hours
```

---

## 13. File & Directory Map

```
Rust_Invest/
├── src/
│   ├── lib.rs              ← 50 modules, main library
│   ├── bin/
│   │   ├── train.rs        ← Weekly model training
│   │   ├── signal.rs       ← Daily signal generation
│   │   ├── serve.rs        ← Web API server
│   │   ├── agent.rs        ← Autonomous monitoring
│   │   ├── backtest_walkforward.rs  ← Honest backtester
│   │   ├── backtest_report.rs       ← Multi-frequency comparison
│   │   └── rebuild_portfolio.rs     ← One-time portfolio rebuild
│   ├── ensemble.rs         ← Signal generation logic, walk-forward, Platt calibration
│   ├── inference.rs        ← Model loading + forward pass
│   ├── features.rs         ← 30 active features (KEPT_FEATURES)
│   ├── ridge.rs            ← Ridge regression (closed-form)
│   ├── lgbm.rs             ← LightGBM wrapper
│   ├── ml.rs               ← Feature normalisation, sample building
│   ├── model_store.rs      ← Model save/load paths
│   ├── pg.rs               ← PostgreSQL connection & queries
│   ├── db.rs               ← SQLite connection (price history)
│   └── ...                 ← 40+ other modules
├── models/                 ← ~4,300 trained model files
│   ├── aapl_ridge.json
│   ├── aapl_lgbm_reg.txt
│   ├── 5d_aapl_ridge.json
│   └── ...
├── config/
│   ├── assets.json         ← Asset list (stocks, FX, crypto)
│   └── threshold_overrides.json  ← Per-asset signal thresholds
├── reports/
│   ├── walkforward_backtest.json  ← Walk-forward results
│   ├── feature_importance.json    ← Feature rankings
│   └── training_curves/           ← Loss curves per asset
├── frontend/               ← React 19 SPA
│   ├── src/pages/
│   │   ├── Dashboard.tsx
│   │   ├── Simulator.tsx
│   │   ├── TrackRecord.tsx
│   │   └── ...
│   └── dist/               ← Built frontend (served by Rust)
├── dashboard/              ← Original Plotly dashboard
│   └── index.html
├── deploy/                 ← Systemd service files
├── scripts/                ← Automation scripts
├── rust_invest.db          ← SQLite (price history, read-only)
├── .env                    ← API keys, secrets (DO NOT COMMIT)
├── Cargo.toml              ← Rust dependencies
├── CLAUDE.md               ← Improvement plan & acceptance criteria
└── SYSTEM_GUIDE.md         ← This file
```

---

## 14. Environment Variables

These live in `.env` in the project root:

```bash
# Required for price data
POLYGON_API_KEY=...

# Required for sentiment analysis
NEWSAPI_KEY=...
SERPER_API_KEY=...
LLM_PROVIDER=anthropic
LLM_API_KEY=...
LLM_MODEL=claude-3-5-sonnet-20241022

# Required for dashboard login (OAuth)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
MICROSOFT_CLIENT_ID=...
MICROSOFT_CLIENT_SECRET=...

# PostgreSQL (default: agent:agent@localhost:5434/alpha_signal)
DATABASE_URL_ALPHA=postgresql://agent:agent@localhost:5434/alpha_signal

# Email alerts (optional)
SMTP_HOST=...
SMTP_PORT=...
SMTP_USER=...
SMTP_PASSWORD=...
```

---

## 15. Troubleshooting

### "Service won't start"

```bash
journalctl -u rustinvest -n 50    # Check logs
lsof -i :8081                      # Port already in use?
systemctl status postgresql        # PostgreSQL running?
cargo build --release              # Binary built?
```

### "No signals being generated"

```bash
# Run signal manually and watch output
cargo run --release --bin signal 2>&1 | head -50

# Common causes:
# - No models in models/ → run train first
# - SQLite has no recent price data → train fetches this
# - PostgreSQL connection failed → check port 5434
```

### "Walk-forward already running"

```bash
ps aux | grep backtest_walkforward    # Find stale process
ls /tmp/walkforward_backtest.lock     # Check lock file
kill <PID>                            # Kill if stale
rm /tmp/walkforward_backtest.lock     # Remove lock
```

### "Models are stale"

```bash
ls -la models/spy_ridge.json          # Check model date
# If older than 1 week, retrain:
cargo run --release --bin train
cargo run --release --bin train -- --horizon 5
sudo systemctl restart rustinvest
```

### "Dashboard shows old data"

```bash
sudo systemctl restart rustinvest     # Refresh cache

# Check latest signal date
psql -h localhost -p 5434 -U agent -d alpha_signal \
  -c "SELECT MAX(signal_date) FROM signals;"
```

### "PostgreSQL connection refused"

```bash
systemctl status postgresql           # Is it running?
ss -tlnp | grep 5434                  # Is port open?
psql -h localhost -p 5434 -U agent -d alpha_signal -c "SELECT 1;"
```

---

## 16. Current Performance (2026-05-07)

### Architecture changes (v9.1)

| Change | Impact |
|--------|--------|
| Features reduced 99 → 30 | Less overfitting, faster training |
| GRU disabled | Removed noise source (was 45-48%) |
| Model agreement required | Ridge + LightGBM must agree or HOLD |
| Multi-horizon confirmation | 1d + 5d must agree or HOLD |
| Confidence filter (< 0.25) | Filters low-conviction signals |
| 5-day embargo in walk-forward | Honest accuracy, no autocorrelation leak |
| Platt calibration | Better-calibrated confidence scores |
| Zero-variance feature fix | No noise from constant features |
| LightGBM MAE tracked separately | Accurate ensemble weighting (was proxied from Ridge) |

### Walk-forward accuracy (honest, out-of-sample)

| Metric | Before (v8) | After (v9.1) |
|--------|-------------|-------------|
| BUY accuracy | 45.1% (live) | **57.4%** |
| SELL accuracy | 36.8% (live) | **59.5%** |
| Expected value | unknown | **+106 bps/signal** |
| Profit factor | unknown | **4.43x** |
| Features | 99 | **30** |
| Active models | 3 (Ridge+LGBM+GRU) | **2** (Ridge+LGBM) |

### What these numbers mean

- **57.4% BUY accuracy**: Out of every 100 BUY signals, ~57 correctly predicted a >0.5% rise
- **59.5% SELL accuracy**: Out of every 100 SELL signals, ~60 correctly predicted a >0.5% fall
- **+106 bps expected value**: Each actionable signal is worth ~1.06% on average
- **4.43x profit factor**: Winners earn 4.4x what losers cost
- These numbers are from walk-forward testing where models **never saw the test data**
