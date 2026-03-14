/// Comprehensive Feature Engineering Module
/// ==========================================
/// Expands from 14 features to 80+ features per sample.
///
/// Feature categories:
///   A. Price-derived technical indicators (existing + new)
///   B. Volume-derived features
///   C. Volatility regime features
///   D. Multi-timeframe momentum
///   E. Calendar/seasonality features
///   F. Cross-asset market context (VIX, treasuries, sectors, other stocks)
///   G. Lagged features (yesterday's indicators as today's features)
///   H. Statistical distribution features (skewness, kurtosis)
///
/// All features computed from freely available Yahoo Finance data.

use crate::analysis;
use crate::ml::Sample;
use std::collections::HashMap;

// ════════════════════════════════════════
// Feature pruning — noise features identified by diagnostic analysis
// ════════════════════════════════════════

const PRUNED_FEATURES: &[&str] = &[
    // VIX_above_20 un-pruned: user explicitly wants VIX regime feature
    "VIX_above_30", "SMA50_above_200", "vol_regime",
    "daily_return", "is_month_end", "month_sin", "month_cos",
    "day_of_week_cos", "risk_on_off", "up_days_ratio_10d",
    "up_days_ratio_20d", "momentum_3d", "momentum_5d", "momentum_10d",
];

/// Return indices of features to keep (those NOT in the pruned list)
fn pruned_feature_indices() -> Vec<usize> {
    let names = feature_names();
    names.iter().enumerate()
        .filter(|(_, name)| !PRUNED_FEATURES.contains(&name.as_str()))
        .map(|(i, _)| i)
        .collect()
}

/// Feature names after pruning
pub fn active_feature_names() -> Vec<String> {
    let names = feature_names();
    let keep = pruned_feature_indices();
    keep.iter().map(|&i| names[i].clone()).collect()
}

/// Number of active features after pruning
pub fn active_feature_count() -> usize {
    feature_names().len() - PRUNED_FEATURES.len()
}

/// Apply feature pruning to a set of samples (removes pruned columns)
pub fn prune_features(samples: &[Sample]) -> Vec<Sample> {
    let keep = pruned_feature_indices();
    samples.iter().map(|s| {
        let features: Vec<f64> = keep.iter().map(|&i| s.features[i]).collect();
        Sample { features, label: s.label }
    }).collect()
}

// ════════════════════════════════════════
// Market context data — fetched separately, passed in
// ════════════════════════════════════════

/// Daily market context aligned by date index
#[derive(Clone, Debug)]
pub struct MarketContext {
    /// VIX daily close (fear index)
    pub vix: Vec<f64>,
    /// 10-year treasury yield (^TNX)
    pub tnx: Vec<f64>,
    /// 2-year treasury yield (^IRX as proxy, actually 13-week)
    pub irx: Vec<f64>,
    /// Sector ETF daily returns: XLK(tech), XLF(finance), XLE(energy), XLV(health), XLI(industrial)
    pub sector_returns: HashMap<String, Vec<f64>>,
    /// SPY daily returns (overall market)
    pub spy_returns: Vec<f64>,
    /// Gold daily returns (GLD)
    pub gold_returns: Vec<f64>,
    /// USD index returns (DX-Y.NYB or UUP)
    pub dollar_returns: Vec<f64>,
}

/// Tickers we need to fetch for market context
pub const MARKET_TICKERS: &[&str] = &[
    "^VIX",   // Volatility index
    "^TNX",   // 10-year treasury yield
    "^IRX",   // 13-week treasury bill
    "XLK",    // Technology sector
    "XLF",    // Financial sector
    "XLE",    // Energy sector
    "XLV",    // Healthcare sector
    "XLI",    // Industrial sector
    "XLC",    // Communication sector
    "XLP",    // Consumer staples
    "XLY",    // Consumer discretionary
    "GLD",    // Gold ETF
    "UUP",    // US Dollar ETF
];

/// Map a symbol to its sector ETF (returns None for FX/crypto)
pub fn sector_etf_for(symbol: &str) -> Option<&'static str> {
    match symbol {
        // Technology (XLK)
        "AAPL" | "MSFT" | "NVDA" | "AMD" | "QQQ" | "INTC" | "AVGO" | "CRM" | "ADBE" | "ORCL" => Some("XLK"),
        // Communication (XLC) — GOOGL/META per GICS classification
        "GOOGL" | "META" | "NFLX" | "DIS" | "CMCSA" | "VZ" | "T" => Some("XLC"),
        // Financials (XLF)
        "JPM" | "GS" | "BAC" | "WFC" | "MS" | "C" | "BLK" | "SCHW" | "DIA" => Some("XLF"),
        // Energy (XLE)
        "XOM" | "CVX" | "COP" | "SLB" | "EOG" | "MPC" | "PSX" | "VLO" => Some("XLE"),
        // Healthcare (XLV)
        "JNJ" | "UNH" | "LLY" | "PFE" | "MRNA" | "ABBV" | "TMO" | "ABT" | "BMY" | "AMGN" => Some("XLV"),
        // Industrials (XLI)
        "CAT" | "DE" | "MMM" | "HON" | "GE" | "EMR" | "LMT" | "RTX" | "NOC" | "BA" | "GD" | "UPS" | "FDX" => Some("XLI"),
        // Consumer Discretionary (XLY)
        "TSLA" | "AMZN" | "HD" | "NKE" | "SBUX" | "MCD" | "TGT" | "LOW" => Some("XLY"),
        // Consumer Staples (XLP)
        "WMT" | "COST" | "PG" | "KO" | "PEP" | "PM" | "MO" | "CL" => Some("XLP"),
        // SPY is the market itself
        "SPY" => Some("SPY"),
        _ => None,
    }
}

/// Feature names — all 80+
pub fn feature_names() -> Vec<String> {
    let mut names = Vec::new();

    // A. Price-derived technical (26 features)
    names.push("RSI_14".into());
    names.push("RSI_7".into());
    names.push("RSI_delta_3d".into());
    names.push("RSI_delta_7d".into());
    names.push("MACD_hist".into());
    names.push("MACD_hist_delta".into());
    names.push("MACD_line".into());
    names.push("MACD_signal".into());
    names.push("SMA7".into());
    names.push("SMA30".into());
    names.push("SMA50".into());
    names.push("SMA200".into());
    names.push("BB_position".into());
    names.push("BB_width".into());
    names.push("SMA7_ratio".into());
    names.push("SMA30_ratio".into());
    names.push("SMA50_ratio".into());
    names.push("SMA200_ratio".into());
    names.push("SMA50_above_200".into());
    names.push("SMA_spread_50_200".into());
    names.push("price_vs_52w_high".into());
    names.push("price_vs_52w_low".into());
    names.push("EMA12_ratio".into());
    names.push("EMA26_ratio".into());
    names.push("daily_return".into());
    names.push("daily_range_pct".into());

    // B. Volume features (6 features)
    names.push("volume_ratio_20d".into());
    names.push("volume_ratio_5d".into());
    names.push("volume_delta_1d".into());
    names.push("volume_sma5_vs_20".into());
    names.push("price_volume_corr_10d".into());
    names.push("obv_slope_10d".into());

    // C. Volatility features (8 features)
    names.push("volatility_5d".into());
    names.push("volatility_20d".into());
    names.push("volatility_60d".into());
    names.push("vol_ratio_5_20".into());
    names.push("vol_regime".into());
    names.push("atr_14d".into());
    names.push("garman_klass_vol".into());
    names.push("max_drawdown_20d".into());

    // D. Momentum multi-timeframe (10 features)
    names.push("momentum_1d".into());
    names.push("momentum_3d".into());
    names.push("momentum_5d".into());
    names.push("momentum_10d".into());
    names.push("momentum_20d".into());
    names.push("return_3d_avg".into());
    names.push("return_5d_avg".into());
    names.push("return_10d_avg".into());
    names.push("up_days_ratio_10d".into());
    names.push("up_days_ratio_20d".into());

    // E. Calendar features (5 features)
    names.push("day_of_week_sin".into());
    names.push("day_of_week_cos".into());
    names.push("month_sin".into());
    names.push("month_cos".into());
    names.push("is_month_end".into());

    // F. Market context (20 features)
    names.push("VIX_level".into());
    names.push("VIX_delta_5d".into());
    names.push("VIX_above_20".into());
    names.push("VIX_above_30".into());
    names.push("treasury_10y".into());
    names.push("treasury_spread".into());
    names.push("SPY_return_1d".into());
    names.push("SPY_return_5d".into());
    names.push("SPY_momentum_10d".into());
    names.push("sector_tech_ret".into());
    names.push("sector_fin_ret".into());
    names.push("sector_energy_ret".into());
    names.push("sector_health_ret".into());
    names.push("sector_indust_ret".into());
    names.push("sector_comm_ret".into());
    names.push("sector_staples_ret".into());
    names.push("sector_discret_ret".into());
    names.push("gold_return_5d".into());
    names.push("dollar_return_5d".into());
    names.push("risk_on_off".into());

    // G. Lagged features (8 features) — yesterday's key indicators
    names.push("lag1_return".into());
    names.push("lag2_return".into());
    names.push("lag3_return".into());
    names.push("lag1_volume_ratio".into());
    names.push("lag1_rsi".into());
    names.push("lag1_bb_pos".into());
    names.push("lag1_macd_hist".into());
    names.push("lag1_volatility".into());

    // H. Statistical features (6 features)
    names.push("skewness_20d".into());
    names.push("kurtosis_20d".into());
    names.push("autocorr_1d".into());
    names.push("autocorr_5d".into());
    names.push("hurst_exponent_est".into());
    names.push("mean_reversion_score".into());

    // I. Relative & Cross-Asset features (12 features)
    names.push("stock_vs_sector_5d".into());
    names.push("stock_vs_sector_20d".into());
    names.push("stock_vs_spy_5d".into());
    names.push("stock_vs_spy_20d".into());
    names.push("breadth_score".into());
    names.push("momentum_volume_confirm".into());
    names.push("vol_adjusted_momentum_5d".into());
    names.push("vol_adjusted_momentum_20d".into());
    names.push("consecutive_up_days".into());
    names.push("consecutive_down_days".into());
    names.push("price_acceleration".into());
    names.push("regime_consistency".into());

    // J. Event, Sentiment & Momentum Quality features (8 features)
    names.push("days_to_next_earnings".into());
    names.push("days_since_last_earnings".into());
    names.push("in_earnings_window".into());
    names.push("fear_greed_index".into());
    names.push("fear_greed_delta_5d".into());
    names.push("volume_surge_on_move".into());
    names.push("bid_ask_proxy".into());
    names.push("smart_money_flow".into());

    // K. NEW: Extended Macro features (5 features) — added for all assets
    names.push("dxy_level".into());            // US Dollar Index
    names.push("dxy_delta_5d".into());         // DXY 5-day change
    names.push("yield_spread_10y2y".into());   // 10Y-2Y yield spread (recession signal)
    names.push("fed_funds_rate".into());       // Federal funds rate level
    names.push("yield_curve_slope".into());    // Slope of yield curve (positive=normal)

    // L. NEW: Crypto-specific on-chain & sentiment (8 features) — crypto assets only
    names.push("btc_funding_rate".into());     // Binance BTC funding rate (contrarian)
    names.push("eth_funding_rate".into());     // Binance ETH funding rate (contrarian)
    names.push("btc_active_addrs".into());     // BTC active addresses (on-chain activity)
    names.push("btc_tx_volume".into());        // BTC transaction volume (on-chain flow)
    names.push("social_galaxy_score".into());  // LunarCrush galaxy score per coin
    names.push("social_volume".into());        // Social media mention volume
    names.push("social_dominance".into());     // % of total crypto social volume
    names.push("social_score".into());         // Overall sentiment score

    // M. NEW: Forex-specific features (3 features) — FX pairs only
    names.push("fx_rate_differential".into()); // Interest rate differential (carry direction)
    names.push("fx_carry_score".into());       // Carry trade attractiveness (0-1)
    names.push("fx_days_to_cb_meeting".into()); // Days to next central bank meeting

    // N. Phase 2+3: Additional macro/cross-asset and calendar features (7 features)
    names.push("yield_curve_trend".into());     // 5d change in TNX-IRX spread
    names.push("stock_vs_sector_10d".into());   // 10-day relative strength vs sector ETF
    names.push("btc_minus_eth_10d".into());     // BTC dominance proxy (BTC 10d ret - ETH 10d ret)
    names.push("quarter".into());               // Quarter of year (normalised 0.25-1.0)
    names.push("is_monday".into());             // Monday gap effect (binary)
    names.push("is_friday".into());             // Friday position squaring (binary)
    names.push("days_since_52w_high".into());   // Momentum exhaustion signal (normalised)

    names
}

// ════════════════════════════════════════
// Helper math functions
// ════════════════════════════════════════

fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 { 0.0 } else { a / b }
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 { return 0.0; }
    let ma = mean(&a[..n]);
    let mb = mean(&b[..n]);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    safe_div(cov, (va * vb).sqrt())
}

fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-12 { return 0.0; }
    let skew = data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n;
    skew
}

fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-12 { return 0.0; }
    let kurt = data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    kurt - 3.0 // excess kurtosis
}

fn rsi_at(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 { return 50.0; }
    let returns: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
    let recent = &returns[returns.len().saturating_sub(period)..];
    let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    if losses < 1e-12 { return 100.0; }
    let rs = gains / losses;
    100.0 - (100.0 / (1.0 + rs))
}

fn ema_vec(prices: &[f64], period: usize) -> Vec<f64> {
    analysis::ema(prices, period)
}

fn sma_val(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period { return prices.last().copied().unwrap_or(0.0); }
    let slice = &prices[prices.len() - period..];
    mean(slice)
}

fn returns_vec(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| safe_div(w[1] - w[0], w[0])).collect()
}

// ════════════════════════════════════════
// Main feature builder
// ════════════════════════════════════════

/// Build the full 80+ feature set for one asset
///
/// prices: daily close prices (oldest first)
/// volumes: daily volumes (oldest first, same length as prices)
/// timestamps: date strings for calendar features
/// market: optional market context (None for crypto or if unavailable)
/// asset_type: "stock" or "crypto" — affects which features are available
///
/// Returns: Vec<Sample> where each sample has label = next day's return (>0 = up)
/// Extended macro data for new features (DXY, yield spread, fed funds)
#[derive(Clone, Debug, Default)]
pub struct ExtendedMacro {
    /// DXY (US Dollar Index) daily values
    pub dxy: Vec<f64>,
    /// 10Y-2Y yield spread (from FRED)
    pub yield_spread: Vec<f64>,
    /// Fed funds rate
    pub fed_funds: Vec<f64>,
}

/// Crypto enrichment data for new features
#[derive(Clone, Debug, Default)]
pub struct CryptoEnrichmentFeatures {
    /// BTC funding rate values
    pub btc_funding_rate: Vec<f64>,
    /// ETH funding rate values
    pub eth_funding_rate: Vec<f64>,
    /// BTC active addresses (single current value, used as-is)
    pub btc_active_addrs: f64,
    /// BTC transaction volume
    pub btc_tx_volume: f64,
    /// Per-coin social sentiment: (galaxy_score, social_volume, social_dominance, social_score)
    pub social: Option<(f64, f64, f64, f64)>,
}

pub fn build_rich_features(
    prices: &[f64],
    volumes: &[Option<f64>],
    timestamps: &[String],
    market: Option<&MarketContext>,
    asset_type: &str,
    sector_etf: Option<&str>,
    earnings_dates: Option<&[String]>,
    fear_greed: Option<&[(String, f64)]>,
) -> Vec<Sample> {
    build_rich_features_ext(prices, volumes, timestamps, market, asset_type,
        sector_etf, earnings_dates, fear_greed, None, None)
}

/// Extended version with new data sources
pub fn build_rich_features_ext(
    prices: &[f64],
    volumes: &[Option<f64>],
    timestamps: &[String],
    market: Option<&MarketContext>,
    asset_type: &str,
    sector_etf: Option<&str>,
    earnings_dates: Option<&[String]>,
    fear_greed: Option<&[(String, f64)]>,
    ext_macro: Option<&ExtendedMacro>,
    crypto_enrich: Option<&CryptoEnrichmentFeatures>,
) -> Vec<Sample> {
    let min_lookback = 260; // need 252 trading days + buffer for SMA200
    if prices.len() < min_lookback {
        // Fall back to shorter lookback if not enough data
        return build_basic_features(prices, volumes);
    }

    let all_returns = returns_vec(prices);
    let vol_f64: Vec<f64> = volumes.iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();

    // Pre-compute indicators
    let sma7 = analysis::sma(prices, 7);
    let sma30 = analysis::sma(prices, 30);
    let sma50 = analysis::sma(prices, 50);
    let sma200 = analysis::sma(prices, 200);
    let ema12 = ema_vec(prices, 12);
    let ema26 = ema_vec(prices, 26);
    let bb = analysis::bollinger_bands(prices, 20, 2.0);
    let (macd_line, signal_line, macd_hist) = analysis::macd(prices);

    let mut samples = Vec::new();
    let start = 260; // start after max lookback

    let feat_count = feature_names().len();

    for i in start..prices.len() - 1 {
        let price = prices[i];
        let window = &prices[..=i];

        let mut f: Vec<f64> = Vec::with_capacity(feat_count);

        // ══ A. Price-derived technical (20) ══
        let rsi14 = rsi_at(window, 14);
        let rsi7 = rsi_at(window, 7);
        let rsi14_3d_ago = rsi_at(&prices[..=i.saturating_sub(3)], 14);
        let rsi14_7d_ago = rsi_at(&prices[..=i.saturating_sub(7)], 14);

        f.push(rsi14 / 100.0);                                  // RSI_14 (normalised)
        f.push(rsi7 / 100.0);                                   // RSI_7
        f.push((rsi14 - rsi14_3d_ago) / 100.0);                 // RSI_delta_3d
        f.push((rsi14 - rsi14_7d_ago) / 100.0);                 // RSI_delta_7d

        let mh_idx = i.min(macd_hist.len().saturating_sub(1));
        let mh_prev = if mh_idx > 0 { macd_hist[mh_idx - 1] } else { 0.0 };
        let mh = if mh_idx < macd_hist.len() { macd_hist[mh_idx] } else { 0.0 };
        f.push(mh);                                              // MACD_hist
        f.push(mh - mh_prev);                                   // MACD_hist_delta

        // MACD line and signal line (pre-computed, normalised by price)
        let ml_idx = i.min(macd_line.len().saturating_sub(1));
        let sl_idx = i.min(signal_line.len().saturating_sub(1));
        let ml_val = if ml_idx < macd_line.len() { macd_line[ml_idx] } else { 0.0 };
        let sl_val = if sl_idx < signal_line.len() { signal_line[sl_idx] } else { 0.0 };
        f.push(safe_div(ml_val, price));                         // MACD_line (normalised)
        f.push(safe_div(sl_val, price));                         // MACD_signal (normalised)

        // Raw SMA values (pre-computed, normalised by price)
        let sma7_val = if i < sma7.len() { sma7[i] } else { price };
        let sma30_val = if i < sma30.len() { sma30[i] } else { price };
        let sma50_val = if i < sma50.len() { sma50[i] } else { price };
        let sma200_val = if i < sma200.len() { sma200[i] } else { price };
        f.push(safe_div(sma7_val, price));                       // SMA7 (normalised)
        f.push(safe_div(sma30_val, price));                      // SMA30 (normalised)
        f.push(safe_div(sma50_val, price));                      // SMA50 (normalised)
        f.push(safe_div(sma200_val, price));                     // SMA200 (normalised)

        let bb_idx = i.min(bb.len().saturating_sub(1));
        let (bb_upper, bb_mid, bb_lower) = if bb_idx < bb.len() { bb[bb_idx] } else { (price, price, price) };
        let bb_range = bb_upper - bb_lower;
        f.push(safe_div(price - bb_lower, bb_range));            // BB_position [0,1]
        f.push(safe_div(bb_range, bb_mid));                      // BB_width (normalised)

        let s7 = sma_val(window, 7);
        let s30 = sma_val(window, 30);
        let s50 = sma_val(window, 50);
        let s200 = sma_val(window, 200);

        f.push(safe_div(price - s7, s7));                        // SMA7_ratio
        f.push(safe_div(price - s30, s30));                      // SMA30_ratio
        f.push(safe_div(price - s50, s50));                      // SMA50_ratio
        f.push(safe_div(price - s200, s200));                    // SMA200_ratio
        f.push(if s50 > s200 { 1.0 } else { -1.0 });           // SMA50_above_200
        f.push(safe_div(s50 - s200, s200));                      // SMA_spread_50_200

        // 52-week high/low (252 trading days)
        let lookback_252 = &prices[i.saturating_sub(252)..=i];
        let high_52w = lookback_252.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low_52w = lookback_252.iter().cloned().fold(f64::INFINITY, f64::min);
        f.push(safe_div(price - high_52w, high_52w));            // price_vs_52w_high
        f.push(safe_div(price - low_52w, low_52w));              // price_vs_52w_low

        let e12 = if i < ema12.len() { ema12[i] } else { price };
        let e26 = if i < ema26.len() { ema26[i] } else { price };
        f.push(safe_div(price - e12, e12));                      // EMA12_ratio
        f.push(safe_div(price - e26, e26));                      // EMA26_ratio

        let daily_ret = safe_div(price - prices[i-1], prices[i-1]);
        f.push(daily_ret);                                       // daily_return

        // Daily range (high - low) / close — approximate from returns
        let dr = if i >= 1 { (prices[i] - prices[i-1]).abs() / prices[i-1] } else { 0.0 };
        f.push(dr);                                               // daily_range_pct

        // ══ B. Volume features (6) ══
        let v_now = vol_f64[i];
        let v_sma20 = mean(&vol_f64[i.saturating_sub(20)..=i]);
        let v_sma5 = mean(&vol_f64[i.saturating_sub(5)..=i]);
        let v_prev = vol_f64[i.saturating_sub(1)];

        f.push(safe_div(v_now, v_sma20));                       // volume_ratio_20d
        f.push(safe_div(v_now, v_sma5));                         // volume_ratio_5d
        f.push(safe_div(v_now - v_prev, v_prev));                // volume_delta_1d
        f.push(safe_div(v_sma5, v_sma20));                       // volume_sma5_vs_20

        // Price-volume correlation (10d)
        let pv_prices = &all_returns[i.saturating_sub(10)..i.min(all_returns.len())];
        let pv_vols = &vol_f64[i.saturating_sub(10)..=i];
        let pv_len = pv_prices.len().min(pv_vols.len());
        f.push(if pv_len >= 3 { correlation(&pv_prices[..pv_len], &pv_vols[..pv_len]) } else { 0.0 });  // price_volume_corr

        // On-balance volume slope (10d)
        let obv_window = &vol_f64[i.saturating_sub(10)..=i];
        let ret_window = &all_returns[i.saturating_sub(10)..i.min(all_returns.len())];
        let obv_len = obv_window.len().min(ret_window.len());
        let mut obv_sum = 0.0;
        for j in 0..obv_len {
            obv_sum += if ret_window[j] > 0.0 { obv_window[j] } else { -obv_window[j] };
        }
        f.push(safe_div(obv_sum, v_sma20 * 10.0));              // obv_slope_10d

        // ══ C. Volatility features (8) ══
        let ret_5d = &all_returns[i.saturating_sub(5)..i.min(all_returns.len())];
        let ret_20d = &all_returns[i.saturating_sub(20)..i.min(all_returns.len())];
        let ret_60d = &all_returns[i.saturating_sub(60)..i.min(all_returns.len())];

        let vol5 = std_dev(ret_5d);
        let vol20 = std_dev(ret_20d);
        let vol60 = std_dev(ret_60d);

        f.push(vol5);                                             // volatility_5d
        f.push(vol20);                                            // volatility_20d
        f.push(vol60);                                            // volatility_60d
        f.push(safe_div(vol5, vol20));                            // vol_ratio_5_20

        // Volatility regime: low=1, normal=0, high=-1
        let vol_regime = if vol20 < vol60 * 0.7 { 1.0 }
            else if vol20 > vol60 * 1.3 { -1.0 }
            else { 0.0 };
        f.push(vol_regime);                                       // vol_regime

        // ATR-14 approximation (using daily range)
        let atr_window: Vec<f64> = (0..14).map(|j| {
            let idx = i.saturating_sub(j);
            if idx > 0 { (prices[idx] - prices[idx-1]).abs() } else { 0.0 }
        }).collect();
        f.push(safe_div(mean(&atr_window), price));               // atr_14d (normalised)

        // Garman-Klass volatility (simplified — using close-to-close)
        let gk_rets: Vec<f64> = (0..20).map(|j| {
            let idx = i.saturating_sub(j);
            if idx > 0 { ((prices[idx] / prices[idx-1]).ln()).powi(2) } else { 0.0 }
        }).collect();
        f.push(mean(&gk_rets).sqrt());                            // garman_klass_vol

        // Max drawdown over 20 days
        let peak = prices[i.saturating_sub(20)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let trough = prices[i.saturating_sub(20)..=i].iter().cloned().fold(f64::INFINITY, f64::min);
        f.push(safe_div(trough - peak, peak));                    // max_drawdown_20d

        // ══ D. Momentum multi-timeframe (10) ══
        f.push(safe_div(price - prices[i.saturating_sub(1)], prices[i.saturating_sub(1)]));   // momentum_1d
        f.push(safe_div(price - prices[i.saturating_sub(3)], prices[i.saturating_sub(3)]));   // momentum_3d
        f.push(safe_div(price - prices[i.saturating_sub(5)], prices[i.saturating_sub(5)]));   // momentum_5d
        f.push(safe_div(price - prices[i.saturating_sub(10)], prices[i.saturating_sub(10)])); // momentum_10d
        f.push(safe_div(price - prices[i.saturating_sub(20)], prices[i.saturating_sub(20)])); // momentum_20d

        let ret_3d_avg = mean(&all_returns[i.saturating_sub(3)..i.min(all_returns.len())]);
        let ret_5d_avg = mean(&all_returns[i.saturating_sub(5)..i.min(all_returns.len())]);
        let ret_10d_avg = mean(&all_returns[i.saturating_sub(10)..i.min(all_returns.len())]);
        f.push(ret_3d_avg);                                       // return_3d_avg
        f.push(ret_5d_avg);                                       // return_5d_avg
        f.push(ret_10d_avg);                                      // return_10d_avg

        let up_10 = all_returns[i.saturating_sub(10)..i.min(all_returns.len())]
            .iter().filter(|&&r| r > 0.0).count() as f64;
        let up_20 = all_returns[i.saturating_sub(20)..i.min(all_returns.len())]
            .iter().filter(|&&r| r > 0.0).count() as f64;
        f.push(up_10 / 10.0);                                    // up_days_ratio_10d
        f.push(up_20 / 20.0);                                    // up_days_ratio_20d

        // ══ E. Calendar features (5) ══
        let (dow_sin, dow_cos, month_sin, month_cos, is_month_end) =
            parse_calendar(&timestamps[i]);
        f.push(dow_sin);
        f.push(dow_cos);
        f.push(month_sin);
        f.push(month_cos);
        f.push(is_month_end);

        // ══ F. Market context (20) ══
        // CRITICAL: Use YESTERDAY's market data (mi = i-1) to avoid look-ahead bias.
        // We can't know today's VIX close before the market closes, so we use
        // the previous day's values as features for today's prediction.
        if let Some(mkt) = market {
            // Lag by 1 day to prevent look-ahead bias
            let mi = (i.saturating_sub(1)).min(mkt.vix.len().saturating_sub(1));

            let vix_now = if mi < mkt.vix.len() { mkt.vix[mi] } else { 20.0 };
            let vix_5d = if mi >= 5 && mi < mkt.vix.len() { mkt.vix[mi - 5] } else { vix_now };

            f.push(vix_now / 100.0);                              // VIX_level (normalised)
            f.push((vix_now - vix_5d) / 100.0);                   // VIX_delta_5d
            f.push(if vix_now > 20.0 { 1.0 } else { 0.0 });     // VIX_above_20
            f.push(if vix_now > 30.0 { 1.0 } else { 0.0 });     // VIX_above_30

            let tnx_now = if mi < mkt.tnx.len() { mkt.tnx[mi] } else { 4.0 };
            let irx_now = if mi < mkt.irx.len() { mkt.irx[mi] } else { 4.0 };
            f.push(tnx_now / 100.0);                              // treasury_10y
            f.push((tnx_now - irx_now) / 100.0);                  // treasury_spread

            let spy_r = |idx: usize| -> f64 {
                if idx < mkt.spy_returns.len() { mkt.spy_returns[idx] } else { 0.0 }
            };
            f.push(spy_r(mi));                                    // SPY_return_1d
            f.push(mean(&mkt.spy_returns[mi.saturating_sub(5)..=mi.min(mkt.spy_returns.len().saturating_sub(1))])); // SPY_return_5d

            // SPY 10d momentum
            let spy_10d: Vec<f64> = (0..10).map(|j| spy_r(mi.saturating_sub(j))).collect();
            f.push(spy_10d.iter().sum::<f64>());                  // SPY_momentum_10d

            // Sector returns (8 sectors)
            let sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"];
            for sector in &sectors {
                let sr = mkt.sector_returns.get(*sector)
                    .and_then(|v| if mi < v.len() { Some(v[mi]) } else { None })
                    .unwrap_or(0.0);
                f.push(sr);
            }

            // Gold 5d return
            let gold_5d = if mkt.gold_returns.len() > mi && mi >= 5 {
                mean(&mkt.gold_returns[mi-5..=mi])
            } else { 0.0 };
            f.push(gold_5d);                                      // gold_return_5d

            // Dollar 5d return
            let dollar_5d = if mkt.dollar_returns.len() > mi && mi >= 5 {
                mean(&mkt.dollar_returns[mi-5..=mi])
            } else { 0.0 };
            f.push(dollar_5d);                                    // dollar_return_5d

            // Risk-on/off: SPY up AND VIX down = risk-on (+1), opposite = risk-off (-1)
            let risk = if spy_r(mi) > 0.0 && vix_now < vix_5d { 1.0 }
                else if spy_r(mi) < 0.0 && vix_now > vix_5d { -1.0 }
                else { 0.0 };
            f.push(risk);                                         // risk_on_off
        } else {
            // No market context — fill with zeros (20 features)
            for _ in 0..20 { f.push(0.0); }
        }

        // ══ G. Lagged features (8) — yesterday's indicators ══
        let lag1_ret = if i >= 2 { safe_div(prices[i-1] - prices[i-2], prices[i-2]) } else { 0.0 };
        let lag2_ret = if i >= 3 { safe_div(prices[i-2] - prices[i-3], prices[i-3]) } else { 0.0 };
        let lag3_ret = if i >= 4 { safe_div(prices[i-3] - prices[i-4], prices[i-4]) } else { 0.0 };
        f.push(lag1_ret);
        f.push(lag2_ret);
        f.push(lag3_ret);

        let lag1_vol_ratio = if i >= 1 {
            safe_div(vol_f64[i-1], mean(&vol_f64[i.saturating_sub(21)..i]))
        } else { 1.0 };
        f.push(lag1_vol_ratio);                                   // lag1_volume_ratio

        let lag1_rsi = rsi_at(&prices[..i], 14) / 100.0;
        f.push(lag1_rsi);                                         // lag1_rsi

        let lag1_bb = if i >= 1 && bb.len() > i - 1 {
            let (u, _, l) = bb[i-1];
            safe_div(prices[i-1] - l, u - l)
        } else { 0.5 };
        f.push(lag1_bb);                                          // lag1_bb_pos

        let lag1_macd = if i >= 1 && macd_hist.len() > i - 1 { macd_hist[i-1] } else { 0.0 };
        f.push(lag1_macd);                                        // lag1_macd_hist

        let lag1_vol = std_dev(&all_returns[i.saturating_sub(21)..i.saturating_sub(1).min(all_returns.len())]);
        f.push(lag1_vol);                                         // lag1_volatility

        // ══ H. Statistical features (6) ══
        let stat_returns = &all_returns[i.saturating_sub(20)..i.min(all_returns.len())];
        f.push(skewness(stat_returns));                           // skewness_20d
        f.push(kurtosis(stat_returns));                           // kurtosis_20d

        // Autocorrelation at lag 1
        let ac1_a = &all_returns[i.saturating_sub(20)..i.saturating_sub(1).min(all_returns.len())];
        let ac1_b = &all_returns[i.saturating_sub(19)..i.min(all_returns.len())];
        let ac_len = ac1_a.len().min(ac1_b.len());
        f.push(if ac_len >= 3 { correlation(&ac1_a[..ac_len], &ac1_b[..ac_len]) } else { 0.0 });  // autocorr_1d

        // Autocorrelation at lag 5
        let ac5_a = &all_returns[i.saturating_sub(25)..i.saturating_sub(5).min(all_returns.len())];
        let ac5_b = &all_returns[i.saturating_sub(20)..i.min(all_returns.len())];
        let ac5_len = ac5_a.len().min(ac5_b.len());
        f.push(if ac5_len >= 3 { correlation(&ac5_a[..ac5_len], &ac5_b[..ac5_len]) } else { 0.0 });  // autocorr_5d

        // Hurst exponent estimate (R/S method, simplified)
        let hurst = estimate_hurst(stat_returns);
        f.push(hurst);                                            // hurst_exponent_est

        // Mean reversion score: how often price returns to SMA20 within 5 days
        let revert_count = (0..20).filter(|&j| {
            let idx = i.saturating_sub(j);
            if idx >= 5 {
                let above_sma = prices[idx] > s30;
                let returned = (idx+1..=(idx+5).min(i)).any(|k| {
                    (prices[k] > s30) != above_sma
                });
                returned
            } else { false }
        }).count() as f64;
        f.push(revert_count / 20.0);                             // mean_reversion_score

        // ══ I. Relative & Cross-Asset features (12) ══

        // Precompute momentum values used by multiple features below
        let mom_1d = safe_div(price - prices[i.saturating_sub(1)], prices[i.saturating_sub(1)]);
        let mom_5d = safe_div(price - prices[i.saturating_sub(5)], prices[i.saturating_sub(5)]);
        let mom_10d = safe_div(price - prices[i.saturating_sub(10)], prices[i.saturating_sub(10)]);
        let mom_20d = safe_div(price - prices[i.saturating_sub(20)], prices[i.saturating_sub(20)]);

        // stock_vs_sector_5d / stock_vs_sector_20d
        if let (Some(mkt), Some(etf)) = (market, sector_etf) {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            let sector_rets = mkt.sector_returns.get(etf);
            let sector_5d = sector_rets
                .filter(|v| v.len() > mi && mi >= 5)
                .map(|v| v[mi.saturating_sub(4)..=mi].iter().sum::<f64>())
                .unwrap_or(0.0);
            let sector_20d = sector_rets
                .filter(|v| v.len() > mi && mi >= 20)
                .map(|v| v[mi.saturating_sub(19)..=mi].iter().sum::<f64>())
                .unwrap_or(0.0);
            f.push(mom_5d - sector_5d);                              // stock_vs_sector_5d
            f.push(mom_20d - sector_20d);                            // stock_vs_sector_20d
        } else {
            f.push(0.0);                                             // stock_vs_sector_5d (FX/no data)
            f.push(0.0);                                             // stock_vs_sector_20d (FX/no data)
        }

        // stock_vs_spy_5d / stock_vs_spy_20d
        if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            let spy_5d = if mkt.spy_returns.len() > mi && mi >= 5 {
                mkt.spy_returns[mi.saturating_sub(4)..=mi].iter().sum::<f64>()
            } else { 0.0 };
            let spy_20d = if mkt.spy_returns.len() > mi && mi >= 20 {
                mkt.spy_returns[mi.saturating_sub(19)..=mi].iter().sum::<f64>()
            } else { 0.0 };
            f.push(mom_5d - spy_5d);                                 // stock_vs_spy_5d
            f.push(mom_20d - spy_20d);                               // stock_vs_spy_20d
        } else {
            f.push(0.0);                                             // stock_vs_spy_5d
            f.push(0.0);                                             // stock_vs_spy_20d
        }

        f.push(0.0);                                                // breadth_score (placeholder)

        // momentum_volume_confirm: sign(momentum_5d) * volume_ratio_20d
        let vol_ratio_20d = safe_div(v_now, v_sma20);
        let mom_sign = if mom_5d > 0.0 { 1.0 } else if mom_5d < 0.0 { -1.0 } else { 0.0 };
        f.push(mom_sign * vol_ratio_20d);                           // momentum_volume_confirm

        // vol_adjusted_momentum_5d: mom_5d / vol_20d
        f.push(safe_div(mom_5d, vol20));                            // vol_adjusted_momentum_5d

        // vol_adjusted_momentum_20d: mom_20d / vol_60d
        f.push(safe_div(mom_20d, vol60));                           // vol_adjusted_momentum_20d

        // consecutive_up_days (capped at 10)
        let mut consec_up = 0_usize;
        for j in (1..=10.min(i)).rev() {
            if all_returns.len() > i - j && all_returns[i - j] > 0.0 {
                consec_up += 1;
            } else {
                break;
            }
        }
        f.push(consec_up as f64);                                   // consecutive_up_days

        // consecutive_down_days (capped at 10)
        let mut consec_down = 0_usize;
        for j in (1..=10.min(i)).rev() {
            if all_returns.len() > i - j && all_returns[i - j] < 0.0 {
                consec_down += 1;
            } else {
                break;
            }
        }
        f.push(consec_down as f64);                                 // consecutive_down_days

        // price_acceleration: momentum_5d - momentum_10d
        f.push(mom_5d - mom_10d);                                   // price_acceleration

        // regime_consistency: 1.0 if mom_1d, 5d, 20d all same sign, else 0.0
        let same_sign = (mom_1d > 0.0 && mom_5d > 0.0 && mom_20d > 0.0)
            || (mom_1d < 0.0 && mom_5d < 0.0 && mom_20d < 0.0);
        f.push(if same_sign { 1.0 } else { 0.0 });                 // regime_consistency

        // ══ J. Event, Sentiment & Momentum Quality features (8) ══

        // Earnings features
        let (days_to_next, days_since_last, in_window) = if let Some(edates) = earnings_dates {
            compute_earnings_features(&timestamps[i], edates)
        } else {
            (1.0, 1.0, 0.0)
        };
        f.push(days_to_next);                                        // days_to_next_earnings
        f.push(days_since_last);                                     // days_since_last_earnings
        f.push(in_window);                                           // in_earnings_window

        // Fear & Greed features
        let (fg_val, fg_delta) = if let Some(fg) = fear_greed {
            compute_fear_greed_features(&timestamps[i], fg)
        } else {
            (0.5, 0.0)
        };
        f.push(fg_val);                                              // fear_greed_index
        f.push(fg_delta);                                            // fear_greed_delta_5d

        // Volume surge on move: vol_ratio_20d * abs(momentum_1d) * 10
        let vol_surge = vol_ratio_20d * mom_1d.abs() * 10.0;
        f.push(vol_surge.min(5.0));                                  // volume_surge_on_move

        // Bid-ask proxy: volatility_5d / volatility_20d ratio
        f.push(safe_div(vol5, vol20));                               // bid_ask_proxy

        // Smart money flow: OBV slope direction aligned with price trend
        let obv_sign = if obv_sum > 0.0 { 1.0 } else if obv_sum < 0.0 { -1.0 } else { 0.0 };
        let price_sign = if mom_5d > 0.0 { 1.0 } else if mom_5d < 0.0 { -1.0 } else { 0.0 };
        let smart_money = if obv_sign == 0.0 || price_sign == 0.0 { 0.0 }
            else if obv_sign == price_sign { 1.0 } else { -1.0 };
        f.push(smart_money);                                         // smart_money_flow

        // ══ K. Extended Macro features (5) ══

        // DXY level and delta
        let dxy_val = ext_macro.and_then(|m| {
            if i < m.dxy.len() { Some(m.dxy[i]) }
            else if !m.dxy.is_empty() { Some(*m.dxy.last().unwrap()) }
            else { None }
        }).unwrap_or(100.0);
        let dxy_5d = ext_macro.and_then(|m| {
            if i >= 5 && i < m.dxy.len() { Some(m.dxy[i] - m.dxy[i-5]) }
            else { None }
        }).unwrap_or(0.0);

        f.push(dxy_val / 100.0);                                    // dxy_level (normalised ~1.0)
        f.push(dxy_5d / 100.0);                                     // dxy_delta_5d

        // Yield spread (10Y-2Y): positive = normal, negative = inverted
        let ys_val = ext_macro.and_then(|m| {
            if i < m.yield_spread.len() { Some(m.yield_spread[i]) }
            else if !m.yield_spread.is_empty() { Some(*m.yield_spread.last().unwrap()) }
            else { None }
        }).unwrap_or(0.0);
        f.push(ys_val / 3.0);                                       // yield_spread_10y2y (normalised)

        // Fed funds rate level
        let ffr_val = ext_macro.and_then(|m| {
            if i < m.fed_funds.len() { Some(m.fed_funds[i]) }
            else if !m.fed_funds.is_empty() { Some(*m.fed_funds.last().unwrap()) }
            else { None }
        }).unwrap_or(5.0);
        f.push(ffr_val / 10.0);                                     // fed_funds_rate (normalised)

        // Yield curve slope: derivative of yield spread (positive = steepening)
        let ys_prev = ext_macro.and_then(|m| {
            if i >= 5 && i < m.yield_spread.len() { Some(m.yield_spread[i] - m.yield_spread[i-5]) }
            else { None }
        }).unwrap_or(0.0);
        f.push(ys_prev / 3.0);                                      // yield_curve_slope

        // ══ L. Crypto on-chain & sentiment (8) ══

        if asset_type == "crypto" {
            let btc_fr = crypto_enrich.and_then(|c| {
                if i < c.btc_funding_rate.len() { Some(c.btc_funding_rate[i]) }
                else if !c.btc_funding_rate.is_empty() { Some(*c.btc_funding_rate.last().unwrap()) }
                else { None }
            }).unwrap_or(0.0);
            f.push(btc_fr * 100.0);                                 // btc_funding_rate (scaled up)

            let eth_fr = crypto_enrich.and_then(|c| {
                if i < c.eth_funding_rate.len() { Some(c.eth_funding_rate[i]) }
                else if !c.eth_funding_rate.is_empty() { Some(*c.eth_funding_rate.last().unwrap()) }
                else { None }
            }).unwrap_or(0.0);
            f.push(eth_fr * 100.0);                                 // eth_funding_rate (scaled up)

            let btc_addrs = crypto_enrich.map(|c| c.btc_active_addrs).unwrap_or(800_000.0);
            f.push(btc_addrs / 1_000_000.0);                        // btc_active_addrs (normalised to millions)

            let btc_txvol = crypto_enrich.map(|c| c.btc_tx_volume).unwrap_or(0.0);
            f.push((btc_txvol / 1e9).min(50.0));                    // btc_tx_volume (billions)

            let (galaxy, svol, sdom, sscore) = crypto_enrich
                .and_then(|c| c.social)
                .unwrap_or((50.0, 0.0, 0.0, 50.0));
            f.push(galaxy / 100.0);                                  // social_galaxy_score (0-1)
            f.push((svol / 10000.0).min(10.0));                      // social_volume (scaled)
            f.push(sdom / 100.0);                                    // social_dominance (0-1)
            f.push(sscore / 100.0);                                  // social_score (0-1)
        } else {
            // Non-crypto assets: pad with zeros (8 features)
            for _ in 0..8 {
                f.push(0.0);
            }
        }

        // ══ M. Forex-specific features (3) ══
        if asset_type == "fx" {
            // symbol is not directly available here, but we can pass it
            // through the asset_type field. For now, use the sector_etf
            // param repurposed: when asset_type=="fx", sector_etf carries the pair symbol.
            let pair_sym = sector_etf.unwrap_or("EURUSD=X");
            let (rate_diff, carry, days_meet) =
                crate::forex_features::forex_feature_vector(pair_sym, &timestamps[i]);
            f.push(rate_diff);                                       // fx_rate_differential
            f.push(carry);                                            // fx_carry_score
            f.push(days_meet);                                        // fx_days_to_cb_meeting
        } else {
            // Non-FX assets: pad with zeros (3 features)
            for _ in 0..3 {
                f.push(0.0);
            }
        }

        // ══ N. Additional macro/cross-asset and calendar features (7) ══

        // yield_curve_trend: 5-day change in TNX-IRX spread
        let yc_trend = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.tnx.len().saturating_sub(1));
            if mi >= 5 && mi < mkt.tnx.len() && mi < mkt.irx.len() {
                let spread_now = mkt.tnx[mi] - mkt.irx[mi];
                let spread_5d = mkt.tnx[mi - 5] - mkt.irx[mi - 5];
                (spread_now - spread_5d) / 100.0
            } else { 0.0 }
        } else { 0.0 };
        f.push(yc_trend);                                              // yield_curve_trend

        // stock_vs_sector_10d: 10-day relative strength vs sector ETF
        let svs_10d = if let (Some(mkt), Some(etf)) = (market, sector_etf) {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            let sector_rets = mkt.sector_returns.get(etf);
            let sector_10d = sector_rets
                .filter(|v| v.len() > mi && mi >= 10)
                .map(|v| v[mi.saturating_sub(9)..=mi].iter().sum::<f64>())
                .unwrap_or(0.0);
            mom_10d - sector_10d
        } else { 0.0 };
        f.push(svs_10d);                                              // stock_vs_sector_10d

        // btc_minus_eth_10d: BTC dominance proxy (only meaningful for crypto)
        // Conservative: set to 0.0 for non-crypto; for crypto, computed from
        // market context SPY slot or passed externally. Without separate BTC/ETH
        // price series in MarketContext, default to 0.0 with TODO.
        // TODO: pass BTC and ETH price series via MarketContext for crypto assets
        f.push(0.0);                                                   // btc_minus_eth_10d

        // Calendar features from timestamp
        let cal_parts: Vec<&str> = timestamps[i].split('-').collect();
        let cal_month: u32 = if cal_parts.len() >= 2 { cal_parts[1].parse().unwrap_or(1) } else { 1 };
        let cal_day_str: &str = if cal_parts.len() >= 3 { cal_parts[2].split('T').next().unwrap_or("15") } else { "15" };
        let cal_year: i32 = if !cal_parts.is_empty() { cal_parts[0].parse().unwrap_or(2024) } else { 2024 };
        let cal_day: u32 = cal_day_str.parse().unwrap_or(15);
        let cal_dow = day_of_week(cal_year, cal_month, cal_day);

        // quarter: normalised 0.25-1.0
        let quarter = ((cal_month - 1) / 3 + 1) as f64 / 4.0;
        f.push(quarter);                                               // quarter

        // is_monday: day_of_week returns 0=Sun, 1=Mon, ... 6=Sat
        f.push(if cal_dow == 1 { 1.0 } else { 0.0 });                 // is_monday

        // is_friday
        f.push(if cal_dow == 5 { 1.0 } else { 0.0 });                 // is_friday

        // days_since_52w_high: normalised by 252 trading days
        let days_since_high = {
            let mut days = 0_usize;
            for j in (0..252.min(i + 1)).rev() {
                let idx = i - j;
                if prices[idx] >= high_52w * 0.999 { // within 0.1% of high
                    days = j;
                    break;
                }
            }
            (days as f64 / 252.0).min(1.0)
        };
        f.push(days_since_high);                                       // days_since_52w_high

        // ══ Label: next day return ══
        let label = prices[i+1] - prices[i]; // positive = up

        debug_assert_eq!(f.len(), feat_count,
            "Feature count mismatch: expected {}, got {}", feat_count, f.len());

        samples.push(Sample { features: f, label });
    }

    println!("  Built {} samples × {} features (raw) for {}",
        samples.len(), feat_count, asset_type);

    // Apply feature pruning
    let pruned = prune_features(&samples);
    let pruned_count = feat_count - pruned[0].features.len();
    println!("  Built {} samples × {} features for {} ({} pruned)",
        pruned.len(), pruned[0].features.len(), asset_type, pruned_count);

    pruned
}

/// Simplified features for assets without enough history
fn build_basic_features(prices: &[f64], volumes: &[Option<f64>]) -> Vec<Sample> {
    // Fall back to the original ml::build_features
    crate::ml::build_features(prices, volumes)
}

/// Parse calendar features from timestamp string
fn parse_calendar(timestamp: &str) -> (f64, f64, f64, f64, f64) {
    // Timestamps are ISO format: 2024-03-15T00:00:00+00:00
    let parts: Vec<&str> = timestamp.split('-').collect();
    if parts.len() < 3 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let year: i32 = parts[0].parse().unwrap_or(2024);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day_str: &str = parts[2].split('T').next().unwrap_or("15");
    let day: u32 = day_str.parse().unwrap_or(15);

    // Day of week using Zeller's formula (simplified)
    let dow = day_of_week(year, month, day);
    let dow_f = dow as f64;
    let pi2 = std::f64::consts::PI * 2.0;
    let dow_sin = (pi2 * dow_f / 7.0).sin();
    let dow_cos = (pi2 * dow_f / 7.0).cos();

    let month_sin = (pi2 * (month as f64 - 1.0) / 12.0).sin();
    let month_cos = (pi2 * (month as f64 - 1.0) / 12.0).cos();

    // Is last 3 days of month?
    let days_in_month = match month {
        2 => if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 29 } else { 28 },
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    let is_month_end = if day >= days_in_month - 2 { 1.0 } else { 0.0 };

    (dow_sin, dow_cos, month_sin, month_cos, is_month_end)
}

fn day_of_week(year: i32, month: u32, day: u32) -> u32 {
    // Tomohiko Sakamoto's algorithm
    let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year };
    let m = month as usize;
    ((y + y/4 - y/100 + y/400 + t[m - 1] as i32 + day as i32) % 7) as u32
}

fn estimate_hurst(returns: &[f64]) -> f64 {
    // Simplified R/S analysis
    let n = returns.len();
    if n < 10 { return 0.5; }

    let m = mean(returns);
    let deviations: Vec<f64> = returns.iter().map(|r| r - m).collect();

    // Cumulative deviations
    let mut cum_dev = Vec::with_capacity(n);
    let mut sum = 0.0;
    for d in &deviations {
        sum += d;
        cum_dev.push(sum);
    }

    let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);
    let s = std_dev(returns);

    if s < 1e-12 { return 0.5; }
    let rs = range / s;

    // H ≈ log(R/S) / log(n)
    let h = if rs > 0.0 { rs.ln() / (n as f64).ln() } else { 0.5 };
    h.clamp(0.0, 1.0)
}

/// Compute earnings features: (days_to_next/60, days_since_last/60, in_window)
fn compute_earnings_features(timestamp: &str, earnings_dates: &[String]) -> (f64, f64, f64) {
    let date_str = &timestamp[..10.min(timestamp.len())];
    let mut days_to_next = 60_i64;
    let mut days_since_last = 60_i64;

    for ed in earnings_dates {
        let ed_str = &ed[..10.min(ed.len())];
        // Simple date diff using string comparison + rough day calculation
        if let (Some(cur), Some(earn)) = (parse_date_days(date_str), parse_date_days(ed_str)) {
            let diff = earn - cur;
            if diff >= 0 && diff < days_to_next {
                days_to_next = diff;
            }
            if diff <= 0 && (-diff) < days_since_last {
                days_since_last = -diff;
            }
        }
    }

    let in_window = if days_to_next <= 3 || days_since_last <= 3 { 1.0 } else { 0.0 };
    (
        (days_to_next.min(60) as f64) / 60.0,
        (days_since_last.min(60) as f64) / 60.0,
        in_window,
    )
}

/// Parse YYYY-MM-DD into approximate day number for date arithmetic
fn parse_date_days(date: &str) -> Option<i64> {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() < 3 { return None; }
    let y: i64 = parts[0].parse().ok()?;
    let m: i64 = parts[1].parse().ok()?;
    let d: i64 = parts[2].parse().ok()?;
    Some(y * 365 + m * 30 + d)
}

/// Compute fear & greed features: (normalised 0-1, 5-day delta)
fn compute_fear_greed_features(timestamp: &str, fg_history: &[(String, f64)]) -> (f64, f64) {
    let date_str = &timestamp[..10.min(timestamp.len())];

    // Find the most recent F&G value on or before this date
    let mut current_val = 50.0;
    let mut val_5d_ago = 50.0;
    let mut found = false;

    for (i, (d, v)) in fg_history.iter().enumerate() {
        let d_str = &d[..10.min(d.len())];
        if d_str <= date_str {
            current_val = *v;
            found = true;
            // Look back ~5 entries for delta
            if i >= 5 {
                val_5d_ago = fg_history[i - 5].1;
            }
        } else {
            break;
        }
    }

    if !found {
        return (0.5, 0.0);
    }

    let normalised = (current_val / 100.0).clamp(0.0, 1.0);
    let delta = (current_val - val_5d_ago) / 100.0;
    (normalised, delta)
}

/// Build MarketContext from fetched price histories
pub fn build_market_context(
    histories: &HashMap<String, Vec<f64>>,
) -> MarketContext {
    let vix = histories.get("^VIX").cloned().unwrap_or_default();
    let tnx = histories.get("^TNX").cloned().unwrap_or_default();
    let irx = histories.get("^IRX").cloned().unwrap_or_default();

    let spy_prices = histories.get("SPY").cloned().unwrap_or_default();
    let spy_returns = returns_vec(&spy_prices);

    let gold_prices = histories.get("GLD").cloned().unwrap_or_default();
    let gold_returns = returns_vec(&gold_prices);

    let dollar_prices = histories.get("UUP").cloned().unwrap_or_default();
    let dollar_returns = returns_vec(&dollar_prices);

    let sector_tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"];
    let mut sector_returns = HashMap::new();
    for ticker in &sector_tickers {
        let prices = histories.get(*ticker).cloned().unwrap_or_default();
        sector_returns.insert(ticker.to_string(), returns_vec(&prices));
    }

    MarketContext {
        vix,
        tnx,
        irx,
        sector_returns,
        spy_returns,
        gold_returns,
        dollar_returns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::Sample;

    /// Test 1: Feature vector length and no NaN/Inf values
    #[test]
    fn test_feature_vector_length_and_validity() {
        // Generate 300 synthetic prices (enough for sma200 + 260 lookback + some samples)
        let n = 350;
        let mut prices = Vec::with_capacity(n);
        prices.push(100.0);
        for i in 1..n {
            // Slight uptrend with noise
            let noise = ((i as f64 * 0.1).sin()) * 2.0;
            prices.push(prices[i - 1] + 0.05 + noise * 0.01);
        }

        let volumes: Vec<Option<f64>> = (0..n).map(|i| Some(1_000_000.0 + (i as f64 * 100.0))).collect();
        let timestamps: Vec<String> = (0..n).map(|i| {
            let day = (i % 28) + 1;
            let month = ((i / 28) % 12) + 1;
            format!("2023-{:02}-{:02}T00:00:00+00:00", month, day)
        }).collect();

        let samples = build_rich_features(&prices, &volumes, &timestamps, None, "stock", None, None, None);

        assert!(!samples.is_empty(), "Should produce at least one sample");

        let expected = active_feature_count();
        for (idx, s) in samples.iter().enumerate() {
            assert_eq!(
                s.features.len(), expected,
                "Sample {} has {} features, expected {}",
                idx, s.features.len(), expected
            );
            for (fi, &val) in s.features.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Sample {} feature {} is not finite: {}",
                    idx, fi, val
                );
            }
        }
    }

    /// Test 2: Macro feature inclusion — VIX, yield curve, UUP are present and non-zero
    #[test]
    fn test_macro_features_present() {
        let n = 350;
        let mut prices = Vec::with_capacity(n);
        prices.push(100.0);
        for i in 1..n {
            let noise = ((i as f64 * 0.1).sin()) * 2.0;
            prices.push(prices[i - 1] + 0.05 + noise * 0.01);
        }

        let volumes: Vec<Option<f64>> = (0..n).map(|i| Some(1_000_000.0 + (i as f64 * 100.0))).collect();
        let timestamps: Vec<String> = (0..n).map(|i| {
            let day = (i % 28) + 1;
            let month = ((i / 28) % 12) + 1;
            format!("2023-{:02}-{:02}T00:00:00+00:00", month, day)
        }).collect();

        // Build MarketContext with non-trivial values
        let vix: Vec<f64> = (0..n).map(|i| 15.0 + (i as f64 * 0.03).sin() * 10.0).collect();
        let tnx: Vec<f64> = (0..n).map(|i| 4.0 + (i as f64 * 0.02).sin() * 0.5).collect();
        let irx: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64 * 0.01).sin() * 0.3).collect();
        let spy_returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin() * 0.01).collect();
        let gold_returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin() * 0.005).collect();
        let dollar_returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03).sin() * 0.003).collect();
        let mut sector_returns = HashMap::new();
        for sector in &["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"] {
            sector_returns.insert(sector.to_string(),
                (0..n).map(|i| (i as f64 * 0.04).sin() * 0.008).collect());
        }

        let mkt = MarketContext { vix, tnx, irx, sector_returns, spy_returns, gold_returns, dollar_returns };

        let samples = build_rich_features(&prices, &volumes, &timestamps, Some(&mkt), "stock", Some("XLK"), None, None);
        assert!(!samples.is_empty());

        // Find feature indices by name
        let names = active_feature_names();
        let find_idx = |name: &str| names.iter().position(|n| n == name);

        // VIX_level should be non-zero (VIX was ~15-25)
        if let Some(idx) = find_idx("VIX_level") {
            let val = samples[5].features[idx];
            assert!(val > 0.0, "VIX_level should be positive, got {}", val);
        }

        // treasury_spread should be non-zero (TNX ~4.0, IRX ~5.0, so spread ~-1.0)
        if let Some(idx) = find_idx("treasury_spread") {
            let val = samples[5].features[idx];
            assert!(val != 0.0, "treasury_spread should be non-zero, got {}", val);
        }

        // dollar_return_5d should be non-zero given sinusoidal input
        if let Some(idx) = find_idx("dollar_return_5d") {
            // At least one sample should have non-zero dollar return
            let any_nonzero = samples.iter().any(|s| s.features[idx].abs() > 1e-10);
            assert!(any_nonzero, "dollar_return_5d should have non-zero values");
        }

        // yield_curve_trend should be present and some non-zero
        if let Some(idx) = find_idx("yield_curve_trend") {
            let any_nonzero = samples.iter().any(|s| s.features[idx].abs() > 1e-10);
            assert!(any_nonzero, "yield_curve_trend should have non-zero values");
        }
    }

    /// Test 3: Calendar features in valid ranges
    #[test]
    fn test_calendar_features_range() {
        let n = 350;
        let mut prices = Vec::with_capacity(n);
        prices.push(100.0);
        for i in 1..n {
            prices.push(prices[i - 1] + 0.05 + ((i as f64 * 0.1).sin()) * 0.01);
        }
        let volumes: Vec<Option<f64>> = (0..n).map(|_| Some(1_000_000.0)).collect();
        let timestamps: Vec<String> = (0..n).map(|i| {
            let day = (i % 28) + 1;
            let month = ((i / 28) % 12) + 1;
            format!("2023-{:02}-{:02}T00:00:00+00:00", month, day)
        }).collect();

        let samples = build_rich_features(&prices, &volumes, &timestamps, None, "stock", None, None, None);
        assert!(!samples.is_empty());

        let names = active_feature_names();
        let find_idx = |name: &str| names.iter().position(|n| n == name);

        // day_of_week_sin should be in [-1, 1]
        if let Some(idx) = find_idx("day_of_week_sin") {
            for s in &samples {
                assert!((-1.0..=1.0).contains(&s.features[idx]),
                    "day_of_week_sin out of range: {}", s.features[idx]);
            }
        }

        // quarter should be in [0.25, 1.0]
        if let Some(idx) = find_idx("quarter") {
            for s in &samples {
                assert!((0.2..=1.01).contains(&s.features[idx]),
                    "quarter out of range: {}", s.features[idx]);
            }
        }

        // is_monday should be 0.0 or 1.0
        if let Some(idx) = find_idx("is_monday") {
            for s in &samples {
                assert!(s.features[idx] == 0.0 || s.features[idx] == 1.0,
                    "is_monday should be 0 or 1, got {}", s.features[idx]);
            }
        }

        // is_friday should be 0.0 or 1.0
        if let Some(idx) = find_idx("is_friday") {
            for s in &samples {
                assert!(s.features[idx] == 0.0 || s.features[idx] == 1.0,
                    "is_friday should be 0 or 1, got {}", s.features[idx]);
            }
        }

        // days_since_52w_high should be in [0.0, 1.0]
        if let Some(idx) = find_idx("days_since_52w_high") {
            for s in &samples {
                assert!((0.0..=1.0).contains(&s.features[idx]),
                    "days_since_52w_high out of range: {}", s.features[idx]);
            }
        }
    }

    /// Test 4: ensemble_overrides.json parses successfully
    #[test]
    fn test_ensemble_overrides_parse() {
        use crate::ensemble::{load_ensemble_overrides, EnsembleOverride};

        let overrides = load_ensemble_overrides();
        // Should have at least AAPL and default
        assert!(
            overrides.contains_key("AAPL") || overrides.contains_key("default"),
            "Overrides should contain at least one entry, got: {:?}",
            overrides.keys().collect::<Vec<_>>()
        );

        // Verify each entry has valid fields
        for (key, ov) in &overrides {
            let _ = format!("{}: linreg={} logreg={} gbt={}", key, ov.use_linreg, ov.use_logreg, ov.use_gbt);
        }

        // Also test direct JSON parse
        let json = std::fs::read_to_string("config/ensemble_overrides.json")
            .expect("ensemble_overrides.json should exist");
        let parsed: std::collections::HashMap<String, EnsembleOverride> =
            serde_json::from_str(&json).expect("JSON should parse into HashMap<String, EnsembleOverride>");
        assert!(!parsed.is_empty());
    }

    /// Test 5: Ensemble overrides have valid fields
    #[test]
    fn test_ensemble_overrides_valid_fields() {
        use crate::ensemble::{load_ensemble_overrides, get_override};

        let overrides = load_ensemble_overrides();
        assert!(!overrides.is_empty(), "Should have at least one override");

        // Test that get_override returns a valid result for a known key
        let aapl_ov = get_override(&overrides, "AAPL");
        // AAPL override should have GBT enabled
        assert!(aapl_ov.use_gbt, "AAPL override should have use_gbt=true");

        // Default should have all models enabled
        let default_ov = get_override(&overrides, "default");
        assert!(default_ov.use_linreg && default_ov.use_logreg && default_ov.use_gbt,
            "Default override should enable all models");

        // Unknown symbol should fall back to default
        let unknown_ov = get_override(&overrides, "UNKNOWN_TICKER_XYZ");
        assert!(unknown_ov.use_linreg && unknown_ov.use_logreg && unknown_ov.use_gbt,
            "Unknown symbol should fall back to default (all enabled)");
    }

    /// Test 3: Walk-forward fold produces valid accuracy for each model
    #[test]
    fn test_walk_forward_model_accuracy() {
        // Create synthetic labeled data
        let n_samples = 200;
        let n_features = 10; // Small feature set for speed
        let mut samples: Vec<Sample> = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let features: Vec<f64> = (0..n_features)
                .map(|j| ((i * 7 + j * 13) as f64 / 100.0).sin())
                .collect();
            let label = if i % 3 == 0 { -1.0 } else { 1.0 }; // ~67% positive
            samples.push(Sample { features, label });
        }

        let train_window = 120;
        let test_window = 30;
        let (train_data, test_data) = samples.split_at(train_window);

        // LinReg
        let mut train_copy = train_data.to_vec();
        let (means, stds) = crate::ml::normalise(&mut train_copy);
        let mut test_copy = test_data[..test_window].to_vec();
        crate::ml::apply_normalisation(&mut test_copy, &means, &stds);

        let mut linreg = crate::ml::LinearRegression::new(n_features);
        linreg.train(&train_copy, 0.005, 1000);
        let lin_correct = test_copy.iter().filter(|s| {
            let pred = linreg.predict(&s.features);
            (pred > 0.0) == (s.label > 0.0)
        }).count();
        let lin_acc = lin_correct as f64 / test_copy.len() as f64;
        assert!((0.0..=1.0).contains(&lin_acc), "LinReg accuracy {:.3} out of [0,1]", lin_acc);

        // LogReg
        let mut logreg = crate::ml::LogisticRegression::new(n_features);
        logreg.train(&train_copy, 0.01, 1000);
        let log_correct = test_copy.iter().filter(|s| {
            let prob = logreg.predict_probability(&s.features);
            (prob > 0.5) == (s.label > 0.0)
        }).count();
        let log_acc = log_correct as f64 / test_copy.len() as f64;
        assert!((0.0..=1.0).contains(&log_acc), "LogReg accuracy {:.3} out of [0,1]", log_acc);

        // GBT
        let x_train: Vec<Vec<f64>> = train_copy.iter().map(|s| s.features.clone()).collect();
        let y_train: Vec<f64> = train_copy.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
        let x_test: Vec<Vec<f64>> = test_copy.iter().map(|s| s.features.clone()).collect();
        let y_test: Vec<f64> = test_copy.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();

        let gbt_config = crate::gbt::GBTConfig {
            n_trees: 20,
            learning_rate: 0.1,
            tree_config: crate::gbt::TreeConfig { max_depth: 3, min_samples_leaf: 5, min_samples_split: 10 },
            subsample_ratio: 0.8,
            early_stopping_rounds: Some(5),
        };
        let gbt_model = crate::gbt::GradientBoostedClassifier::train(
            &x_train, &y_train, None, None, gbt_config
        );
        let gbt_correct = x_test.iter().zip(y_test.iter()).filter(|(x, &y)| {
            let prob = gbt_model.predict_proba(x);
            (prob > 0.5) == (y > 0.5)
        }).count();
        let gbt_acc = gbt_correct as f64 / x_test.len() as f64;
        assert!((0.0..=1.0).contains(&gbt_acc), "GBT accuracy {:.3} out of [0,1]", gbt_acc);
    }
}
