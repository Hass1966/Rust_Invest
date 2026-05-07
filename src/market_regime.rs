/// Market Regime Detection — SPY-based macro regime overlay
/// =========================================================
/// Computes a global market regime (Bull/Bear/Neutral/EarlyWarning/Crisis)
/// from SPY's 5d/10d/20d returns, then adjusts enriched signals accordingly.
///
/// This is a portfolio-level macro overlay, separate from the per-asset
/// K-means regime detector in regime.rs.

use serde::{Serialize, Deserialize};
use crate::enriched_signals::EnrichedSignal;

// ════════════════════════════════════════
// Types
// ════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Neutral,
    EarlyWarning,
    Crisis,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Bull => write!(f, "BULL"),
            MarketRegime::Bear => write!(f, "BEAR"),
            MarketRegime::Neutral => write!(f, "NEUTRAL"),
            MarketRegime::EarlyWarning => write!(f, "EARLY_WARNING"),
            MarketRegime::Crisis => write!(f, "CRISIS"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegimeState {
    pub regime: MarketRegime,
    pub spy_return_20d_pct: f64,
    pub spy_return_10d_pct: f64,
    pub spy_return_5d_pct: f64,
    pub risk_score: f64,
    pub regime_strength: String,
    pub spy_price_current: f64,
    pub spy_price_20d_ago: f64,
    pub defensive_assets: Vec<String>,
    pub timestamp: String,
}

// ════════════════════════════════════════
// Constants
// ════════════════════════════════════════

// Multi-timeframe thresholds
const FAST_DROP_WARNING: f64 = -2.0;  // 5-day drop triggers early warning
const FAST_DROP_CRISIS: f64 = -5.0;   // 5-day drop triggers crisis mode
const BEAR_10D_THRESHOLD: f64 = -2.0;

const BEAR_THRESHOLD: f64 = -3.0;     // 20d
const BULL_THRESHOLD: f64 = 3.0;      // 20d

pub const DEFAULT_DEFENSIVE: &[&str] = &["GLD", "SLV", "TLT", "BND", "USO"];

// BEAR regime thresholds
const BEAR_BUY_SUPPRESS_PROB: f64 = 62.0;     // Stock BUYs below this → HOLD
const BEAR_DEFENSIVE_PROMOTE_PROB: f64 = 50.0; // Defensive HOLDs above this → BUY
const BEAR_SHORT_PROMOTE_PROB: f64 = 42.0;     // Stock SELLs below this → SHORT

// EARLY_WARNING regime thresholds
const EARLY_WARNING_BUY_SUPPRESS_PROB: f64 = 65.0; // Stock BUYs below this → HOLD
const EARLY_WARNING_DEFENSIVE_PROMOTE_PROB: f64 = 50.0; // Same as Bear for defensive

// CRISIS regime thresholds
const CRISIS_DEFENSIVE_PROMOTE_PROB: f64 = 45.0; // Defensive HOLDs above this → BUY

// BULL regime thresholds
const BULL_DEFENSIVE_SUPPRESS_PROB: f64 = 58.0; // Defensive BUYs below this → HOLD

// ════════════════════════════════════════
// Regime Computation
// ════════════════════════════════════════

/// Compute market regime from SPY price history.
/// Requires at least 21 prices (20 trading days of returns).
pub fn compute_regime(spy_prices: &[f64]) -> Option<MarketRegimeState> {
    if spy_prices.len() < 21 {
        return None;
    }

    let len = spy_prices.len();
    let current = *spy_prices.last()?;

    // 20-day return (always available if len >= 21)
    let price_20d_ago = spy_prices[len - 21];
    if price_20d_ago <= 0.0 {
        return None;
    }
    let return_20d = ((current - price_20d_ago) / price_20d_ago) * 100.0;

    // 10-day return (need at least 11 prices)
    let return_10d = if len >= 11 {
        let price_10d_ago = spy_prices[len - 11];
        if price_10d_ago > 0.0 {
            ((current - price_10d_ago) / price_10d_ago) * 100.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 5-day return (need at least 6 prices)
    let return_5d = if len >= 6 {
        let price_5d_ago = spy_prices[len - 6];
        if price_5d_ago > 0.0 {
            ((current - price_5d_ago) / price_5d_ago) * 100.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    // ── Composite risk score (weighted blend) ──
    let risk_5d = ((-return_5d) / 7.0).clamp(0.0, 1.0);
    let risk_10d = ((-return_10d) / 7.0).clamp(0.0, 1.0);
    let risk_20d = ((-return_20d) / 10.0).clamp(0.0, 1.0);
    let risk_score = risk_5d * 0.50 + risk_10d * 0.30 + risk_20d * 0.20;

    // ── Regime priority (most urgent first) ──
    let regime = if return_5d < FAST_DROP_CRISIS {
        // 1. 5-day crash → Crisis
        MarketRegime::Crisis
    } else if return_20d < BEAR_THRESHOLD {
        // 2. 20-day sustained decline → Bear
        MarketRegime::Bear
    } else if return_5d < FAST_DROP_WARNING && return_20d > 0.0 {
        // 3. Sharp 5-day drop but 20d still positive → Early Warning
        MarketRegime::EarlyWarning
    } else if return_10d < BEAR_10D_THRESHOLD && return_20d < 0.0 {
        // 4. 10-day confirming nascent bear
        MarketRegime::Bear
    } else if return_20d > BULL_THRESHOLD {
        // 5. Sustained uptrend → Bull
        MarketRegime::Bull
    } else {
        // 6. Everything else
        MarketRegime::Neutral
    };

    // ── Regime strength: how many timeframes agree ──
    let bearish_count = [return_5d < 0.0, return_10d < 0.0, return_20d < 0.0]
        .iter().filter(|&&b| b).count();
    let bullish_count = [return_5d > 0.0, return_10d > 0.0, return_20d > 0.0]
        .iter().filter(|&&b| b).count();

    let regime_strength = match regime {
        MarketRegime::Bear | MarketRegime::Crisis | MarketRegime::EarlyWarning => {
            match bearish_count {
                3 => "strong",
                2 => "moderate",
                _ => "weak",
            }
        }
        MarketRegime::Bull => {
            match bullish_count {
                3 => "strong",
                2 => "moderate",
                _ => "weak",
            }
        }
        MarketRegime::Neutral => "moderate",
    };

    Some(MarketRegimeState {
        regime,
        spy_return_20d_pct: (return_20d * 100.0).round() / 100.0,
        spy_return_10d_pct: (return_10d * 100.0).round() / 100.0,
        spy_return_5d_pct: (return_5d * 100.0).round() / 100.0,
        risk_score: (risk_score * 1000.0).round() / 1000.0,
        regime_strength: regime_strength.to_string(),
        spy_price_current: current,
        spy_price_20d_ago: price_20d_ago,
        defensive_assets: DEFAULT_DEFENSIVE.iter().map(|s| s.to_string()).collect(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

// ════════════════════════════════════════
// Signal Overlay
// ════════════════════════════════════════

/// Apply regime-based signal adjustments to enriched signals.
/// Returns the number of signals modified.
pub fn apply_regime_overlay(
    regime: &MarketRegimeState,
    signals: &mut Vec<EnrichedSignal>,
    defensive_symbols: &[String],
) -> usize {
    let mut modified = 0;

    match regime.regime {
        MarketRegime::Crisis => {
            for sig in signals.iter_mut() {
                let is_defensive = defensive_symbols.contains(&sig.asset);
                let prob = sig.technical.probability_up;

                // Suppress ALL non-defensive BUYs
                if !is_defensive && sig.signal == "BUY" {
                    sig.signal = "HOLD".to_string();
                    sig.reason = format!(
                        "BUY suppressed (CRISIS regime, all non-defensive blocked) — {}",
                        sig.reason
                    );
                    modified += 1;
                }

                // Promote defensive HOLDs at relaxed 45% threshold
                if is_defensive && sig.signal == "HOLD" && prob > CRISIS_DEFENSIVE_PROMOTE_PROB {
                    sig.signal = "BUY".to_string();
                    sig.reason = format!(
                        "Promoted to BUY (CRISIS regime defensive rotation, prob {:.1}%) — {}",
                        prob, sig.reason
                    );
                    modified += 1;
                }

                // In crisis regime, non-defensive SELLs stay as SELL (SHORT disabled)
                // SHORT signals had 24.7% live accuracy — not worth the risk
            }
        }
        MarketRegime::Bear => {
            for sig in signals.iter_mut() {
                let is_defensive = defensive_symbols.contains(&sig.asset);
                let prob = sig.technical.probability_up;

                // Suppress marginal stock BUYs
                if !is_defensive && sig.signal == "BUY" && prob < BEAR_BUY_SUPPRESS_PROB {
                    sig.signal = "HOLD".to_string();
                    sig.reason = format!(
                        "BUY suppressed (BEAR regime, prob {:.1}% < {:.0}%) — {}",
                        prob, BEAR_BUY_SUPPRESS_PROB, sig.reason
                    );
                    modified += 1;
                }

                // Promote defensive assets with relaxed threshold
                if is_defensive && sig.signal == "HOLD" && prob > BEAR_DEFENSIVE_PROMOTE_PROB {
                    sig.signal = "BUY".to_string();
                    sig.reason = format!(
                        "Promoted to BUY (BEAR regime defensive rotation, prob {:.1}%) — {}",
                        prob, sig.reason
                    );
                    modified += 1;
                }

                // SHORT promotion disabled — 24.7% live accuracy
                // Weak SELLs remain as SELL in bear regime
            }
        }
        MarketRegime::EarlyWarning => {
            for sig in signals.iter_mut() {
                let is_defensive = defensive_symbols.contains(&sig.asset);
                let prob = sig.technical.probability_up;

                // Suppress stock BUYs below 65% prob
                if !is_defensive && sig.signal == "BUY" && prob < EARLY_WARNING_BUY_SUPPRESS_PROB {
                    sig.signal = "HOLD".to_string();
                    sig.reason = format!(
                        "BUY suppressed (EARLY_WARNING regime, prob {:.1}% < {:.0}%) — {}",
                        prob, EARLY_WARNING_BUY_SUPPRESS_PROB, sig.reason
                    );
                    modified += 1;
                }

                // Promote defensive HOLDs (same threshold as Bear)
                if is_defensive && sig.signal == "HOLD" && prob > EARLY_WARNING_DEFENSIVE_PROMOTE_PROB {
                    sig.signal = "BUY".to_string();
                    sig.reason = format!(
                        "Promoted to BUY (EARLY_WARNING regime defensive rotation, prob {:.1}%) — {}",
                        prob, sig.reason
                    );
                    modified += 1;
                }
            }
        }
        MarketRegime::Bull => {
            // Reduce defensive asset allocation — make defensive BUYs harder
            for sig in signals.iter_mut() {
                let is_defensive = defensive_symbols.contains(&sig.asset);
                if is_defensive && sig.signal == "BUY" {
                    let prob = sig.technical.probability_up;
                    if prob < BULL_DEFENSIVE_SUPPRESS_PROB {
                        sig.signal = "HOLD".to_string();
                        sig.reason = format!(
                            "Defensive BUY suppressed (BULL regime, prob {:.1}% < {:.0}%) — {}",
                            prob, BULL_DEFENSIVE_SUPPRESS_PROB, sig.reason
                        );
                        modified += 1;
                    }
                }
            }
        }
        MarketRegime::Neutral => {
            // Annotate GLD with hedge recommendation
            for sig in signals.iter_mut() {
                if sig.asset == "GLD" {
                    sig.suggested_action = format!(
                        "{} (Regime: NEUTRAL — maintain 10% GLD hedge allocation)",
                        sig.suggested_action
                    );
                }
            }
        }
    }

    modified
}
