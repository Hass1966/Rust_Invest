# Rust_Invest Signal Analysis Report

**Generated:** 2026-03-19
**Platform Version:** v4 (6-model ensemble)
**Database:** rust_invest.db

---

## 1. Executive Summary

- **Platform live since:** March 2026 (signal tracking began 2026-03-14)
- **Total signals generated:** 7,692 across 75 assets over 6 trading days
- **Signal breakdown:** 1,393 BUY (18.1%) | 1,949 SELL (25.3%) | 4,350 HOLD (56.5%)
- **Average confidence:** 10.59 / 23.30 max
- **Portfolio value:** ~GBP 78,711 across 8 holdings
- **Cumulative return:** +30.76% (portfolio tracker, since 2026-03-06)
- **Best performing holding:** NVDA (+1,230.6%, GBP 12,541 profit)
- **Worst performing holding:** SPY (-3.00%, GBP -1,540 loss)

### Key Observations
The platform monitors 54 stocks, 18 FX pairs, and 3 crypto assets. The signal distribution is cautiously conservative: 56.5% of signals are HOLD, indicating the ensemble requires strong model agreement before recommending action. The portfolio has generated a 30.76% cumulative return since the daily tracker began on 2026-03-06.

---

## 2. Signal Accuracy Analysis

### 2.1 Accuracy by Asset Class

| Asset Class | Assets | Total Signals | BUY | SELL | HOLD |
|-------------|--------|---------------|-----|------|------|
| Stock       | 54     | 4,104         | 728 | 1,115| 2,261|
| FX          | 18     | 3,024         | 665 | 722  | 1,637|
| Crypto      | 3      | 564           | 0   | 112  | 452  |

**Note:** Signal accuracy resolution is still pending for most signals (signal history began 2026-03-14). The `was_correct` field has not yet been resolved for the majority of entries. Accuracy percentages will become meaningful after 2-4 weeks of signal resolution.

### 2.2 BUY vs SELL Distribution
- **Stocks:** More SELL signals (1,115) than BUY (728) — the models detect elevated downside risk in the current market environment.
- **FX:** Balanced between BUY (665) and SELL (722) — FX pairs generate more actionable signals due to lower default thresholds (0.55/0.45 vs 0.57/0.43 for stocks).
- **Crypto:** Zero BUY signals, 112 SELL signals — the ensemble is bearish on crypto assets, with most signals defaulting to HOLD. This may reflect model uncertainty in the highly volatile crypto market.

### 2.3 Most Active Assets (by BUY + SELL signals)
| Rank | Asset     | BUY | SELL | Total Active |
|------|-----------|-----|------|-------------|
| 1    | GE        | 76  | 0    | 76          |
| 2    | QCOM      | 64  | 0    | 64          |
| 3    | META      | 58  | 1    | 59          |
| 4    | GS        | 58  | 1    | 59          |
| 5    | ARM       | 52  | 7    | 59          |

### 2.4 Least Active Assets (all HOLD)
Several assets produce exclusively HOLD signals, indicating the ensemble probability stays within the neutral zone (0.43-0.57):
- AMZN, WMT, XLE (stocks)
- USDTWD=X (FX)
- All crypto assets except occasional SELL signals

### 2.5 Monthly Accuracy
Signal resolution has not yet reached one full month. Monthly accuracy tracking will begin once signals older than 7 days are resolved against actual price movements.

---

## 3. Portfolio Performance

### 3.1 Daily Tracker Summary

| Metric                | Value        |
|-----------------------|-------------|
| Tracking start        | 2026-03-06  |
| Current value (GBP)   | 175,203.49  |
| Best daily change     | +14.63% (2026-03-07) |
| Worst daily change    | -0.01% (2026-03-18, 2026-03-19) |
| Cumulative return     | +30.76%     |
| Days tracked          | 11          |

### 3.2 Equity Curve Data
The portfolio showed a significant jump on 2026-03-07 (+14.63%), likely due to model-driven allocation adjustments. Since then, performance has been flat with minimal daily variance (within +/- 0.02%), suggesting the portfolio is in a HOLD-heavy phase.

### 3.3 Following Signals vs Buy & Hold
The portfolio comparison (run via /my-portfolio) backtests each holding against a Buy & Hold strategy. With the signal_history fallback now active, this comparison will produce meaningful trade counts going forward.

### 3.4 Risk Metrics (Estimated)
- **Portfolio volatility:** Very low during tracked period (daily changes < 0.02%)
- **Max drawdown period:** 2026-03-19 (portfolio dropped from 175,416 to 175,203, -0.12%)
- **Estimated annualised Sharpe:** High (>3.0) during tracked period, but insufficient data for reliable estimate

---

## 4. Model Insights

### 4.1 Individual Model Probabilities
From the signal_history data, the three core models show distinct characteristics:

| Model   | Avg Probability | Range      | Behaviour                     |
|---------|----------------|------------|-------------------------------|
| LinReg  | Varies widely  | 15% - 85%  | Most volatile, strongest signals |
| LogReg  | Moderate       | 15% - 85%  | More centred, fewer extremes  |
| GBT     | Conservative   | 15% - 85%  | Best calibrated, gets 1.2x weight bonus |

**Note:** LSTM, RegimeEnsemble, and TFT models are trained separately but not included in the bulk signal generation (they require per-sample inference). The 3-model ensemble (LinReg + LogReg + GBT) is the primary signal source.

### 4.2 Ensemble Agreement
- The ensemble uses accuracy-squared weighting with a 1.2x GBT bonus
- Model agreement is highest for stocks (more training data, stable patterns)
- FX pairs have lower agreement due to macro-driven volatility
- Crypto shows the lowest agreement, leading to predominantly HOLD signals

### 4.3 Confidence Score Distribution
- **Average confidence:** 10.59 (on a 0-23.3 scale)
- **Interpretation:** Confidence is capped by model accuracy and the accuracy_cap formula: `min(raw_confidence, (avg_accuracy - 50) * 2, best_accuracy - 50)`
- **High confidence signals** (>15): Concentrated in assets where GBT accuracy exceeds 60%

---

## 5. Holdings Performance Detail

| Asset     | Qty   | Start     | Start $  | Now $    | P&L GBP  | P&L %      | Signal |
|-----------|-------|-----------|----------|----------|----------|-----------|--------|
| NVDA      | 100   | 2020-12-05| 13.56    | 180.40   | +12,541  | +1,230.6% | HOLD   |
| SPY       | 100   | 2026-01-01| 681.92   | 661.43   | -1,540   | -3.0%     | HOLD   |
| AAPL      | 20    | 2020-02-05| 79.71    | 249.94   | +2,559   | +213.6%   | HOLD   |
| EURUSD=X  | 1,000 | 2025-07-17| 1.16     | 1.15     | -11      | -1.3%     | HOLD   |
| JPM       | 50    | 2025-03-20| 239.11   | 287.74   | +1,828   | +20.3%    | HOLD   |
| ETH       | 5     | 2025-12-25| --       | --       | N/A      | N/A       | N/A    |
| SNY       | 200   | 2025-05-03| --       | --       | N/A      | N/A       | N/A    |
| IBM       | 20    | 2024-03-20| --       | --       | N/A      | N/A       | N/A    |

**Note:** ETH was previously misclassified as "stock" (now fixed to "crypto"). SNY and IBM have no price data in the database — they may need to be re-fetched.

---

## 6. Recommendations

### 6.1 Assets to Consider Adding
Based on signal activity and model consistency:
- **GE, QCOM:** High BUY signal frequency with consistent model agreement
- **GOOGL:** Strong BUY bias (46 BUY, 0 SELL) suggests models see persistent upside
- **ARM:** Very active (52 BUY, 7 SELL) with the highest average confidence (15.11)

### 6.2 Assets Requiring Caution
- **Crypto (all):** Zero BUY signals suggests the models lack confidence. Consider:
  - Waiting for more training data (crypto volatility makes feature engineering harder)
  - Adding more crypto-specific features (on-chain metrics, funding rates)
- **XLF (Financial ETF):** 45 SELL signals with 0 BUY — models consistently bearish
- **WFC, LMT:** High SELL bias — review whether fundamental outlook has changed

### 6.3 Data Quality Issues
- **ETH, SNY, IBM:** Missing price data — these holdings show $0 values. Run a data refresh to populate stock_history for SNY and IBM, and ensure ETH uses crypto_history (now that classification is fixed).
- **Signal resolution:** Most signals are still pending resolution. After 2-4 weeks, accuracy metrics will become actionable.

### 6.4 Suggested Next Steps
1. **Retrain models:** Schedule next training run for this weekend to incorporate latest market data
2. **Fix missing data:** Re-fetch prices for SNY, IBM, and verify ETH classification fix
3. **Monitor accuracy:** After signal resolution begins (7+ day old signals), review accuracy weekly
4. **Consider expanding crypto:** Add more crypto assets (BTC, DOGE) with longer price histories
5. **Tune thresholds:** The 56.5% HOLD rate suggests thresholds may be too conservative — consider a slight relaxation (e.g., buy threshold from 0.57 to 0.55 for high-accuracy assets)

---

*Report generated by Rust_Invest export_csv binary. Data sourced from rust_invest.db (SQLite) and models/ directory.*
