/// Forex-Specific Features
/// ========================
/// Additional features for currency pair assets:
///
///   1. Interest rate differential (USD rate vs counter-currency rate)
///   2. Carry trade score (rate_high - rate_low, normalised)
///   3. Days to next central bank meeting (FOMC/ECB/BOE/BOJ/RBA/SNB calendars)
///
/// Non-USD rates are hardcoded initial values (updated periodically).
/// Central bank meeting dates are hardcoded for 2026.

/// Hardcoded central bank policy rates (as of early 2026)
/// These should be updated periodically or fetched from FRED/ECB/etc.
pub fn policy_rate(currency: &str) -> f64 {
    match currency {
        "USD" => 4.50,  // Fed funds target (upper bound)
        "EUR" => 2.75,  // ECB main refinancing rate
        "GBP" => 4.25,  // BOE bank rate
        "JPY" => 0.50,  // BOJ policy rate
        "AUD" => 4.10,  // RBA cash rate
        "CHF" => 1.00,  // SNB policy rate
        "CAD" => 3.75,  // BOC overnight rate
        _ => 3.0,       // fallback
    }
}

/// Compute interest rate differential for a forex pair.
/// For EUR/USD: USD rate - EUR rate.
/// Positive = USD has higher rate (carry in favour of USD).
pub fn rate_differential(pair_symbol: &str) -> f64 {
    let (base, quote) = parse_pair(pair_symbol);
    policy_rate(&quote) - policy_rate(&base)
}

/// Carry trade score: abs rate differential, normalised 0-1.
/// Higher = more carry opportunity = more interest in the trade.
pub fn carry_score(pair_symbol: &str) -> f64 {
    let diff = rate_differential(pair_symbol).abs();
    (diff / 5.0).min(1.0) // normalise: 5% diff = 1.0
}

/// Parse a forex pair symbol into (base_currency, quote_currency)
fn parse_pair(symbol: &str) -> (String, String) {
    // Yahoo Finance format: EURUSD=X, GBPUSD=X, JPY=X, AUDUSD=X, CHF=X
    let clean = symbol.trim_end_matches("=X").to_uppercase();
    match clean.as_str() {
        "EURUSD" => ("EUR".into(), "USD".into()),
        "GBPUSD" => ("GBP".into(), "USD".into()),
        "JPY" | "USDJPY" => ("USD".into(), "JPY".into()),
        "AUDUSD" => ("AUD".into(), "USD".into()),
        "CHF" | "USDCHF" => ("USD".into(), "CHF".into()),
        "USDCAD" => ("USD".into(), "CAD".into()),
        _ => {
            if clean.len() >= 6 {
                (clean[..3].to_string(), clean[3..6].to_string())
            } else {
                ("USD".to_string(), "USD".to_string())
            }
        }
    }
}

// ════════════════════════════════════════
// Central Bank Meeting Calendar 2026
// ════════════════════════════════════════

/// FOMC meeting dates for 2026 (MM-DD format)
const FOMC_2026: &[&str] = &[
    "01-28", "03-18", "05-06", "06-17",
    "07-29", "09-16", "11-04", "12-16",
];

/// ECB Governing Council meeting dates 2026 (rate decisions)
const ECB_2026: &[&str] = &[
    "01-30", "03-06", "04-17", "06-05",
    "07-17", "09-11", "10-23", "12-18",
];

/// BOE MPC meeting dates 2026
const BOE_2026: &[&str] = &[
    "02-06", "03-20", "05-08", "06-19",
    "08-07", "09-18", "11-06", "12-18",
];

/// BOJ meeting dates 2026
const BOJ_2026: &[&str] = &[
    "01-24", "03-14", "04-25", "06-13",
    "07-17", "09-19", "10-30", "12-18",
];

/// RBA meeting dates 2026
const RBA_2026: &[&str] = &[
    "02-17", "04-01", "05-20", "07-08",
    "08-19", "10-07", "11-25", "12-09",
];

/// SNB meeting dates 2026
const SNB_2026: &[&str] = &[
    "03-20", "06-19", "09-18", "12-11",
];

/// Get the relevant central bank meeting calendar for a currency pair.
/// Returns dates for both the base and quote currency's central bank.
pub fn relevant_meeting_dates(pair_symbol: &str) -> Vec<String> {
    let (base, quote) = parse_pair(pair_symbol);
    let mut dates = Vec::new();

    let add_calendar = |currency: &str, dates: &mut Vec<String>| {
        let cal = match currency {
            "USD" => FOMC_2026.to_vec(),
            "EUR" => ECB_2026.to_vec(),
            "GBP" => BOE_2026.to_vec(),
            "JPY" => BOJ_2026.to_vec(),
            "AUD" => RBA_2026.to_vec(),
            "CHF" => SNB_2026.to_vec(),
            _ => Vec::new(),
        };
        for d in cal {
            dates.push(format!("2026-{}", d));
        }
    };

    add_calendar(&base, &mut dates);
    add_calendar(&quote, &mut dates);
    dates.sort();
    dates
}

/// Compute days to next central bank meeting for a forex pair.
/// Returns the minimum days until any relevant central bank meets.
pub fn days_to_next_meeting(pair_symbol: &str, current_date: &str) -> f64 {
    let meetings = relevant_meeting_dates(pair_symbol);
    if meetings.is_empty() { return 30.0; }

    // Parse current date (YYYY-MM-DD or ISO format)
    let current = &current_date[..10]; // take YYYY-MM-DD part

    let mut min_days = 90.0_f64;
    for meeting in &meetings {
        if meeting.as_str() >= current {
            // Simple day difference estimate
            let diff = date_diff_days(current, meeting);
            if diff < min_days && diff >= 0.0 {
                min_days = diff;
            }
        }
    }

    min_days
}

/// Simple date difference in days (YYYY-MM-DD format)
fn date_diff_days(from: &str, to: &str) -> f64 {
    let parse = |s: &str| -> Option<(i32, u32, u32)> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() >= 3 {
            let y = parts[0].parse::<i32>().ok()?;
            let m = parts[1].parse::<u32>().ok()?;
            let d = parts[2].parse::<u32>().ok()?;
            Some((y, m, d))
        } else {
            None
        }
    };

    if let (Some((y1, m1, d1)), Some((y2, m2, d2))) = (parse(from), parse(to)) {
        // Approximate: 30.44 days/month
        let days1 = y1 as f64 * 365.25 + m1 as f64 * 30.44 + d1 as f64;
        let days2 = y2 as f64 * 365.25 + m2 as f64 * 30.44 + d2 as f64;
        days2 - days1
    } else {
        30.0 // fallback
    }
}

/// Compute all forex-specific features for a currency pair at a given date.
/// Returns: (rate_diff, carry_score, days_to_meeting) — all normalised.
pub fn forex_feature_vector(pair_symbol: &str, timestamp: &str) -> (f64, f64, f64) {
    let rate_diff = rate_differential(pair_symbol) / 5.0; // normalise to ~[-1, 1]
    let carry = carry_score(pair_symbol);
    let days_meeting = days_to_next_meeting(pair_symbol, timestamp) / 30.0; // normalise to ~[0, 3]

    (rate_diff, carry, days_meeting)
}
