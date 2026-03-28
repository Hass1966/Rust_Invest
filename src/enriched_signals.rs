/// Enriched Signal Generation — Rule-Based Decision Support
/// ==========================================================
/// Transforms raw TradingSignal into enriched JSON responses with
/// natural-language reasoning, risk context, and suggested actions.
/// All logic is rule-based (no LLM calls).

use serde::{Serialize, Deserialize};
use crate::ensemble::TradingSignal;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskContext {
    pub volatility_regime: String,
    pub drawdown_risk: String,
    pub trend_strength: String,
    pub days_to_earnings: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetail {
    pub probability_up: f64,
    pub weight: u32,
    pub vote: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDetail {
    pub confidence: f64,
    pub probability_up: f64,
    pub model_agreement: String,
    pub rsi: f64,
    pub trend: String,
    pub bb_position: Option<f64>,
    pub quality: String,
    pub walk_forward_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedSignal {
    pub asset: String,
    pub asset_class: String,
    pub signal: String,
    pub reason: String,
    pub explanation: String,
    pub risk_context: RiskContext,
    pub suggested_action: String,
    pub technical: TechnicalDetail,
    pub models: std::collections::HashMap<String, ModelDetail>,
    pub price: f64,
    pub timestamp: String,
    pub llm_sentiment: f64,
    pub llm_analysis: Option<String>,
}

/// Build an enriched signal from a raw TradingSignal
pub fn enrich_signal(
    signal: &TradingSignal,
    asset_class: &str,
    bb_position: Option<f64>,
    volatility_5d: Option<f64>,
    volatility_20d: Option<f64>,
) -> EnrichedSignal {
    let mut reasons = Vec::new();
    let mut risk_factors = Vec::new();

    // RSI-based reasoning
    if signal.rsi > 70.0 {
        reasons.push("RSI in overbought territory".to_string());
        risk_factors.push("overbought conditions increase reversal risk");
    } else if signal.rsi < 30.0 {
        reasons.push("RSI in oversold territory, potential bounce".to_string());
    } else if signal.rsi > 60.0 {
        reasons.push("Momentum improving".to_string());
    } else if signal.rsi < 40.0 {
        reasons.push("Momentum weakening".to_string());
    }

    // Bollinger Band position reasoning
    if let Some(bb) = bb_position {
        if bb > 0.8 {
            reasons.push("trading near upper Bollinger Band".to_string());
        } else if bb < 0.2 {
            reasons.push("trading near lower Bollinger Band, potentially oversold".to_string());
        }
    }

    // Volatility reasoning
    let vol_regime = match (volatility_5d, volatility_20d) {
        (Some(v5), Some(v20)) if v5 > v20 * 1.3 => {
            reasons.push("short-term volatility rising sharply".to_string());
            "high"
        }
        (Some(v5), Some(v20)) if v5 > v20 => {
            reasons.push("short-term volatility rising".to_string());
            "moderate"
        }
        (Some(v5), Some(v20)) if v5 < v20 * 0.7 => {
            reasons.push("volatility contracting".to_string());
            "low"
        }
        _ => "moderate",
    };

    // Model agreement reasoning
    if signal.models_agree == signal.n_models {
        reasons.push("strong model consensus".to_string());
    } else if signal.models_agree <= signal.n_models / 2 {
        reasons.push("models disagree — lower conviction signal".to_string());
    }

    // Trend reasoning
    match signal.sma_trend.as_str() {
        "BULLISH" => reasons.push("trend is bullish (SMA crossover positive)".to_string()),
        "BEARISH" => reasons.push("trend is bearish (SMA crossover negative)".to_string()),
        _ => {}
    }

    // LLM sentiment reasoning
    if signal.llm_sentiment > 0.3 {
        reasons.push("AI news analysis is bullish".to_string());
    } else if signal.llm_sentiment < -0.3 {
        reasons.push("AI news analysis is bearish".to_string());
    }

    let reason = if reasons.is_empty() {
        "No strong directional signal detected".to_string()
    } else {
        let first = reasons[0].clone();
        if reasons.len() > 1 {
            format!("{} with {}", capitalize_first(&first), reasons[1..].join(", "))
        } else {
            capitalize_first(&first)
        }
    };

    // Drawdown risk estimation
    let drawdown_risk = if signal.signal == "SHORT" {
        "Unlimited (short)"
    } else if signal.confidence > 5.0 && signal.signal == "BUY" {
        "3-5%"
    } else if vol_regime == "high" {
        "8-12%"
    } else if vol_regime == "moderate" {
        "5-8%"
    } else {
        "2-4%"
    };

    // Trend strength
    let trend_strength = if signal.models_agree == signal.n_models {
        "strong"
    } else if signal.models_agree > signal.n_models / 2 {
        "moderate"
    } else {
        "weak"
    };

    // Quality assessment
    let quality = if signal.walk_forward_accuracy > 62.0 {
        "HIGH"
    } else if signal.walk_forward_accuracy >= 55.0 {
        "MODERATE"
    } else if signal.walk_forward_accuracy >= 50.0 {
        "LOW"
    } else {
        "NO EDGE"
    };

    // Suggested action (with SHORT support)
    let suggested_action = build_suggested_action(
        &signal.signal,
        signal.confidence,
        quality,
        &signal.sma_trend,
        signal.rsi,
    );

    // Drawdown risk for SHORT signals
    if signal.signal == "SHORT" {
        risk_factors.push("short positions carry unlimited theoretical risk");
    }

    // Model details
    let mut models = std::collections::HashMap::new();
    models.insert("linreg".to_string(), ModelDetail {
        probability_up: (signal.linear_prob * 100.0 * 10.0).round() / 10.0,
        weight: (signal.linear_weight * 100.0).round() as u32,
        vote: if signal.linear_prob > 0.5 { "UP".to_string() } else { "DOWN".to_string() },
    });
    models.insert("logreg".to_string(), ModelDetail {
        probability_up: (signal.logistic_prob * 100.0 * 10.0).round() / 10.0,
        weight: (signal.logistic_weight * 100.0).round() as u32,
        vote: if signal.logistic_prob > 0.5 { "UP".to_string() } else { "DOWN".to_string() },
    });
    models.insert("gbt".to_string(), ModelDetail {
        probability_up: (signal.gbt_prob * 100.0 * 10.0).round() / 10.0,
        weight: (signal.gbt_weight * 100.0).round() as u32,
        vote: if signal.gbt_prob > 0.5 { "UP".to_string() } else { "DOWN".to_string() },
    });

    // Build explanation line
    let explanation = build_explanation(
        &signal.signal,
        signal.confidence,
        signal.models_agree,
        signal.n_models,
        signal.gbt_prob,
        signal.linear_prob,
        signal.logistic_prob,
    );

    EnrichedSignal {
        asset: signal.symbol.clone(),
        asset_class: asset_class.to_string(),
        signal: signal.signal.clone(),
        reason,
        explanation,
        risk_context: RiskContext {
            volatility_regime: vol_regime.to_string(),
            drawdown_risk: drawdown_risk.to_string(),
            trend_strength: trend_strength.to_string(),
            days_to_earnings: None,
        },
        suggested_action,
        technical: TechnicalDetail {
            confidence: (signal.confidence * 10.0).round() / 10.0,
            probability_up: (signal.ensemble_prob * 100.0 * 10.0).round() / 10.0,
            model_agreement: format!("{}/{}", signal.models_agree, signal.n_models),
            rsi: (signal.rsi * 10.0).round() / 10.0,
            trend: signal.sma_trend.clone(),
            bb_position,
            quality: quality.to_string(),
            walk_forward_accuracy: (signal.walk_forward_accuracy * 10.0).round() / 10.0,
        },
        models,
        price: (signal.current_price * 100.0).round() / 100.0,
        timestamp: chrono::Utc::now().to_rfc3339(),
        llm_sentiment: signal.llm_sentiment,
        llm_analysis: signal.llm_analysis.clone(),
    }
}

fn build_suggested_action(
    signal: &str,
    confidence: f64,
    quality: &str,
    trend: &str,
    rsi: f64,
) -> String {
    match (signal, quality) {
        ("BUY", "HIGH") if confidence > 5.0 => {
            if rsi > 65.0 {
                "Strong buy signal but RSI elevated. Consider scaling in gradually.".to_string()
            } else {
                "Strong buy signal with high conviction. Consider adding to position.".to_string()
            }
        }
        ("BUY", _) => {
            if trend == "BEARISH" {
                "Buy signal but against the trend. Use smaller position size.".to_string()
            } else {
                "Moderate buy signal. Consider accumulating on pullbacks.".to_string()
            }
        }
        ("SHORT", "HIGH") if confidence > 5.0 => {
            "Strong short signal. Consider opening a short position via CFD/spread bet. Use strict stop-loss.".to_string()
        }
        ("SHORT", _) => {
            if trend == "BULLISH" {
                "Short signal but against the trend. Higher risk — use tight stop-loss and small size.".to_string()
            } else {
                "Short signal aligns with bearish trend. Consider a measured short via CFD or spread bet.".to_string()
            }
        }
        ("SELL", "HIGH") if confidence > 5.0 => {
            "Strong sell signal. Consider reducing exposure or hedging.".to_string()
        }
        ("SELL", _) => {
            if trend == "BULLISH" {
                "Sell signal but trend is still positive. Tighten stops rather than exit fully.".to_string()
            } else {
                "Sell signal aligns with bearish trend. Consider reducing position.".to_string()
            }
        }
        ("HOLD", _) if quality == "NO EDGE" => {
            "No clear direction — stay patient. Models show no edge on this asset.".to_string()
        }
        ("HOLD", _) => {
            if rsi < 30.0 {
                "Hold current position. Oversold conditions may present opportunity soon.".to_string()
            } else if rsi > 70.0 {
                "Hold current position. Overbought — consider taking partial profits.".to_string()
            } else {
                "Hold current position. No strong directional signal at this time.".to_string()
            }
        }
        _ => "Monitor position. No actionable signal.".to_string(),
    }
}

fn build_explanation(
    signal: &str,
    confidence: f64,
    models_agree: usize,
    n_models: usize,
    gbt_prob: f64,
    linreg_prob: f64,
    logreg_prob: f64,
) -> String {
    let conf_pct = (confidence * 10.0).round() as i64; // confidence is 0-10, display as %

    // Determine primary reason
    let gbt_conf = ((gbt_prob - 0.5).abs() * 200.0).round() as i64;
    let linreg_up = linreg_prob > 0.5;
    let logreg_up = logreg_prob > 0.5;

    let primary_reason = if gbt_conf > 65 {
        "strong trend signal"
    } else if linreg_up == logreg_up {
        "consensus signal"
    } else if conf_pct > 70 {
        "high confidence signal"
    } else if conf_pct < 55 {
        "weak signal — use caution"
    } else {
        "quantitative momentum signal"
    };

    match signal {
        "BUY" => format!(
            "{} of {} models bullish · {}% confidence · {}",
            models_agree, n_models, conf_pct, primary_reason
        ),
        "SHORT" => format!(
            "{} of {} models bearish · {}% confidence · {} · Short positions carry higher risk. Not financial advice.",
            models_agree, n_models, conf_pct, primary_reason
        ),
        "SELL" => format!(
            "{} of {} models bearish · {}% confidence · {}",
            models_agree, n_models, conf_pct, primary_reason
        ),
        _ => "Models disagree or signal too weak to act".to_string(),
    }
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}
