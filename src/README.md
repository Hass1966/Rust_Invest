# Phase 7 — Portfolio Allocator

## What's New
- **portfolio.rs** — New module. Distributes $100K across assets using 3 weighting schemes:
  - Sharpe-Weighted (more capital to better risk-adjusted performers)
  - Equal Weight ($10K each qualifying asset)
  - Inverse-Volatility (more capital to lower-risk assets)
- **report.rs** — Complete rewrite. Generates polished dark-theme HTML dashboard with:
  - Navigation bar with section links
  - KPI cards at top
  - Backtest results table with colour-coded verdicts
  - Portfolio equity curve (SVG) + allocation breakdown
  - Trading signals, correlation matrix, metric glossary
- **main.rs** — Updated to wire portfolio module after backtester

## Files to Replace/Add
Copy these 3 files into your `~/Rust_Invest/src/` directory:

```bash
cp src/main.rs ~/Rust_Invest/src/main.rs
cp src/report.rs ~/Rust_Invest/src/report.rs
cp src/portfolio.rs ~/Rust_Invest/src/portfolio.rs
```

## Build & Run

```bash
cd ~/Rust_Invest
cargo build --release 2>&1 | tail -20
./target/release/rust_invest
```

## Expected Output
The program will now show an additional section after backtest:

```
━━━ PORTFOLIO ALLOCATION ━━━

  [Portfolio] 10 assets qualify (of 15 tested)
  [Portfolio] Allocation (Sharpe-Weighted):
      NVDA:  14.9%  ($  14,909)  Sharpe: 4.49  Return: +621.5%
      QQQ:   12.7%  ($  12,715)  Sharpe: 3.83  Return: +136.4%
      TSLA:  12.5%  ($  12,483)  Sharpe: 3.76  Return: +870.3%
      ...

  ╔═══════════════════════════════════════╗
  ║  PORTFOLIO RESULT — Sharpe-Weighted   ║
  ║  Starting Capital:   $100,000         ║
  ║  Final Value:        $XXX,XXX         ║
  ║  Sharpe Ratio:       X.XX             ║
  ╚═══════════════════════════════════════╝
```

Then generates `report.html` with the beautiful dashboard format.

## Notes
- No model retraining needed — uses existing walk-forward signals
- Crypto excluded from portfolio (all 5 got NO EDGE)
- Transaction costs included (10 bps per trade)
- report.html opens in any browser — no server needed
