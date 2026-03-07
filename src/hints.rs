/// Hints — Actionable, plain-English suggestions derived from today's signals
/// ============================================================================

use serde::Serialize;
use std::collections::HashMap;
use crate::enriched_signals::EnrichedSignal;

#[derive(Debug, Clone, Serialize)]
pub struct Hint {
    pub asset: String,
    pub category: String,
    pub urgency: String,
    pub title: String,
    pub reason: String,
    pub what_it_means: String,
    pub suggested_pct: f64,
}

const TECH_TICKERS: &[&str] = &["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "QQQ"];

pub fn generate_hints(signals: &HashMap<String, EnrichedSignal>) -> Vec<Hint> {
    let mut hints: Vec<Hint> = Vec::new();

    // Count signal types
    let buys: Vec<_> = signals.values().filter(|s| s.signal == "BUY").collect();
    let sells: Vec<_> = signals.values().filter(|s| s.signal == "SELL").collect();

    // Rule 1: Profit-taking hint for NVDA or TSLA with SELL signal
    for ticker in &["NVDA", "TSLA"] {
        if let Some(sig) = signals.get(*ticker) {
            if sig.signal == "SELL" {
                hints.push(Hint {
                    asset: ticker.to_string(),
                    category: "hedge".to_string(),
                    urgency: "high".to_string(),
                    title: format!("Consider taking some {} profit", ticker),
                    reason: format!(
                        "{} is showing a caution signal today. \
                         You don't have to sell everything — but taking 20-30% \
                         profit locks in real gains.",
                        ticker,
                    ),
                    what_it_means: format!(
                        "When our models flag a SELL on a stock that has been performing well, \
                         it often means the price has run ahead of what the data supports. \
                         Reducing your position by 20-30% lets you keep most of your exposure \
                         while banking some profit.",
                    ),
                    suggested_pct: 25.0,
                });
            }
        }
    }

    // Rule 2: Tech concentration
    let tech_buys = buys.iter().filter(|s| TECH_TICKERS.contains(&s.asset.as_str())).count();
    if buys.len() >= 3 && tech_buys as f64 / buys.len() as f64 > 0.5 {
        hints.push(Hint {
            asset: "XLV or XLI (defensive ETFs)".to_string(),
            category: "rebalance".to_string(),
            urgency: "medium".to_string(),
            title: "Your signals are heavily tech today".to_string(),
            reason: format!(
                "{} of today's {} BUY signals are tech stocks. Tech moves together — \
                 if one falls, they often all fall. Consider whether you have exposure \
                 to other sectors too.",
                tech_buys, buys.len(),
            ),
            what_it_means: "Sector concentration means your portfolio's returns depend \
                heavily on one industry. Spreading across healthcare, industrials, or \
                other sectors can reduce the impact of a tech-specific downturn.".to_string(),
            suggested_pct: 0.0,
        });
    }

    // Rule 3: More SELL signals than BUY
    if sells.len() > buys.len() {
        hints.push(Hint {
            asset: "Portfolio".to_string(),
            category: "warning".to_string(),
            urgency: "high".to_string(),
            title: "More caution signals than buy signals today".to_string(),
            reason: format!(
                "Today our models are seeing more reasons to be cautious ({} sell) \
                 than to buy ({} buy). This doesn't mean sell everything — it means \
                 be selective and don't rush into new positions.",
                sells.len(), buys.len(),
            ),
            what_it_means: "When the majority of signals are cautious, it usually \
                indicates broad market uncertainty. It's often better to wait for \
                clearer signals before making big moves.".to_string(),
            suggested_pct: 0.0,
        });
    }

    // Rule 4: GBP/EUR strengthening — currency hint
    for (ticker, currency) in &[("GBPUSD=X", "pound"), ("EURUSD=X", "euro")] {
        if let Some(sig) = signals.get(*ticker) {
            if sig.signal == "BUY" {
                hints.push(Hint {
                    asset: ticker.to_string(),
                    category: "opportunity".to_string(),
                    urgency: "low".to_string(),
                    title: format!("{} strengthening — affects your US stock returns",
                        if *currency == "pound" { "Pound" } else { "Euro" }),
                    reason: format!(
                        "When the {} rises against the dollar, your US stock returns \
                         are worth slightly less when converted back to {}. Worth knowing \
                         if you're thinking of buying more US stocks right now.",
                        currency,
                        if *currency == "pound" { "pounds" } else { "euros" },
                    ),
                    what_it_means: "Currency movements affect the real value of \
                        international investments. A stronger home currency means \
                        overseas returns are worth less when you bring them home.".to_string(),
                    suggested_pct: 0.0,
                });
            }
        }
    }

    hints
}
