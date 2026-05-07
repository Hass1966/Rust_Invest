/// Comprehensive Feature Engineering Module
/// ==========================================
/// Expands from 14 features to 84 active features per sample (172 raw, 84 whitelisted).
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
// Feature selection — whitelist of features validated by Python comparison
// ════════════════════════════════════════

/// Top 30 features by LightGBM importance (2026-05-06 retrain, 351 assets).
/// Excluded: autocorr_1d (#30, leaks at train/test boundary),
///           consecutive_up_days (#81), consecutive_down_days (#83).
/// Replaced with momentum_3d (#31).
const KEPT_FEATURES: &[&str] = &[
    // Price-derived technical (5)
    "BB_position", "BB_width", "daily_range_pct", "RSI_delta_3d", "price_vs_52w_high",
    // Volume (3)
    "volume_ratio_20d", "volume_ratio_5d", "obv_slope_10d",
    // Volatility (3)
    "volatility_5d", "vol_ratio_5_20", "atr_14d",
    // Momentum & returns (5)
    "momentum_1d", "momentum_3d", "lag1_return", "ret_2d", "ret_63d",
    // Market context — SPY & gold (5)
    "SPY_return_1d", "SPY_return_5d", "spy_ret_21d", "gold_return_5d", "dollar_return_5d",
    // VIX features (4)
    "vix_change_1d", "vix_9d_ratio", "vix_term_spread", "vix_sma10_dist",
    // Cross-asset relative (2)
    "rel_strength_vs_spy_1d", "gold_spy_ratio_10d",
    // Statistical (2)
    "hurst_exponent_est", "skew_delta_5d",
    // Macro (1)
    "tnx_change_5d",
];

/// Return indices of features to keep (those IN the kept list)
fn kept_feature_indices() -> Vec<usize> {
    let names = feature_names();
    names.iter().enumerate()
        .filter(|(_, name)| KEPT_FEATURES.contains(&name.as_str()))
        .map(|(i, _)| i)
        .collect()
}

/// Feature names after selection
pub fn active_feature_names() -> Vec<String> {
    let names = feature_names();
    let keep = kept_feature_indices();
    keep.iter().map(|&i| names[i].clone()).collect()
}

/// Number of active features after selection
pub fn active_feature_count() -> usize {
    kept_feature_indices().len()
}

/// Apply feature selection to a set of samples (keeps only whitelisted columns)
pub fn prune_features(samples: &[Sample]) -> Vec<Sample> {
    let keep = kept_feature_indices();
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
    /// 3-month VIX (^VIX3M) — term structure
    pub vix_3m: Vec<f64>,
    /// 9-day VIX (^VIX9D) — near-term fear
    pub vix_9d: Vec<f64>,
    /// CBOE SKEW index (^SKEW) — tail risk gauge
    pub skew: Vec<f64>,
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
    /// ICE BofA HY credit spread (from FRED BAMLH0A0HYM2)
    pub hy_spread: Vec<f64>,
    /// 5-year breakeven inflation (from FRED T5YIE)
    pub breakeven_5y: Vec<f64>,
}

/// Tickers we need to fetch for market context
pub const MARKET_TICKERS: &[&str] = &[
    "^VIX",   // Volatility index
    "^VIX3M", // 3-month VIX (term structure)
    "^VIX9D", // 9-day VIX (near-term)
    "^SKEW",  // CBOE SKEW index (tail risk)
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
    "CL=F",   // WTI Crude Oil futures
    "UUP",    // US Dollar ETF
];

/// Map a symbol to its sector ETF (returns None for FX/crypto)
pub fn sector_etf_for(symbol: &str) -> Option<&'static str> {
    match symbol {
        // Technology (XLK)
        "AAPL" | "MSFT" | "NVDA" | "AMD" | "QQQ" | "INTC" | "AVGO" | "CRM" | "ADBE" | "ORCL"
        | "SAP.DE" | "ARM" | "QCOM" | "TSM" | "IBM"
        | "NOW" | "INTU" | "AMAT" | "MU" | "LRCX" | "KLAC" | "SNPS" | "CDNS"
        | "FTNT" | "PANW" | "CRWD" | "TEAM" | "ZS" | "DDOG" | "NET" | "MDB" | "SNOW"
        | "FICO" | "DELL" | "HPQ" | "KEYS" | "TRMB" | "ANSS" | "TYL" | "MSTR" | "CPRT"
        | "DARK.L" | "AUTO.L" => Some("XLK"),
        // Communication (XLC) — GOOGL/META per GICS classification
        "GOOGL" | "META" | "NFLX" | "DIS" | "CMCSA" | "VZ" | "T"
        | "VOD.L" | "BT-A.L" | "WPP.L"
        | "CHTR" | "TMUS" | "EA" | "TTWO" => Some("XLC"),
        // Financials (XLF)
        "JPM" | "GS" | "BAC" | "WFC" | "MS" | "C" | "BLK" | "SCHW" | "DIA"
        | "HSBA.L" | "LLOY.L" | "BARC.L" | "NWG.L" | "LGEN.L" | "III.L" | "MNG.L"
        | "ALV.DE" | "V" | "MA" | "BRK-B" | "MMC" | "TRV" | "AFL"
        | "AXP" | "COF" | "USB" | "PNC" | "TFC" | "SPGI" | "MCO" | "ICE" | "CME" | "CB"
        | "RJF" | "FITB" | "HBAN" | "CFG" | "MTB" | "ZION" | "KEY" | "NDAQ" | "CBOE"
        | "AIG" | "ALL" | "MET" | "PRU" | "HIG" | "GL" | "FNF" | "WRB"
        | "MSCI" | "MKTX"
        | "STAN.L" | "AV.L" => Some("XLF"),
        // Energy (XLE)
        "XOM" | "CVX" | "COP" | "SLB" | "EOG" | "MPC" | "PSX" | "VLO"
        | "BP.L" | "SHEL.L" | "CNA.L" | "DCC.L" | "VDE" | "CL=F" | "USO"
        | "HAL" | "BKR" | "DVN" | "FANG"
        | "OXY" | "CTRA" | "MRO" | "APA" | "HES" => Some("XLE"),
        // Healthcare (XLV)
        "JNJ" | "UNH" | "LLY" | "PFE" | "MRNA" | "ABBV" | "TMO" | "ABT" | "BMY" | "AMGN"
        | "AZN.L" | "GSK.L" | "SAN.PA" | "MRK" | "CVS" | "CI" | "VHT" | "IHI" | "SNY"
        | "ELV" | "HUM" | "MOH" | "ISRG" | "BSX" | "SYK" | "MDT" | "EW"
        | "REGN" | "VRTX" | "BIIB" | "GILD"
        | "STE" | "WAT" | "A" | "DHR" | "ZBH" | "HOLX" | "DXCM" | "ALGN" | "TECH" | "BIO"
        | "IDXX" | "WST" => Some("XLV"),
        // Industrials (XLI)
        "CAT" | "DE" | "MMM" | "HON" | "GE" | "EMR" | "LMT" | "RTX" | "NOC" | "BA" | "GD" | "UPS" | "FDX"
        | "RR.L" | "AIR.PA" | "SAF.PA" | "SIE.DE" | "MBG.DE" | "QQ.L" | "EXPN.L"
        | "CSX" | "NSC" | "UNP" | "ETN" | "PH" | "ROK" | "IR" | "CARR" | "OTIS"
        | "PCAR" | "GNRC" | "SWK" | "TT" | "DOV" | "NDSN" | "GWW" | "FAST"
        | "WM" | "RSG"
        | "WEIR.L" | "RS1.L" => Some("XLI"),
        // Consumer Discretionary (XLY)
        "TSLA" | "AMZN" | "HD" | "NKE" | "SBUX" | "MCD" | "TGT" | "LOW"
        | "OR.PA" | "MC.PA" | "PSON.L"
        | "TJX" | "ORLY" | "AZO" | "DLTR" | "DG" | "YUM" | "CMG"
        | "ABNB" | "BKNG" | "MAR" | "HLT"
        | "ROST" | "ETSY" | "W" | "LULU" | "DPZ" | "WYNN" | "LVS" | "NCLH" | "RCL" | "CCL" | "POOL"
        | "FRAS.L" | "BRBY.L" | "JD.L" | "PSN.L" | "BDEV.L" | "TW.L" => Some("XLY"),
        // Consumer Staples (XLP)
        "WMT" | "COST" | "PG" | "KO" | "PEP" | "PM" | "MO" | "CL"
        | "ULVR.L" | "IMB.L" | "BATS.L" | "DGE.L" | "TSCO.L" | "SBRY.L" | "CPG.L"
        | "GIS" => Some("XLP"),
        // Utilities (XLU)
        "NG.L" | "SSE.L" | "DUK" | "NEE" | "SO" | "VPU" | "IDU"
        | "AWK" | "ED" | "AES" | "WEC" | "ES" | "CMS" | "DTE" | "EIX" | "FE"
        | "PEG" | "PPL" | "AEP" | "XEL" | "EVRG" => Some("XLU"),
        // Materials (XLB)
        "GLEN.L" | "AAL.L" | "ANTO.L" | "BAS.DE" | "SMDS.L" | "MNDI.L"
        | "LIN" | "APD" | "ECL" | "NEM" | "FCX" | "NUE" | "ALB"
        | "SHW" | "DD" | "PPG" | "VMC" | "MLM" | "CF" | "MOS" | "BALL" | "PKG" | "IP"
        | "RIO.L" | "JMAT.L" | "CRDA.L" | "EVR.L" => Some("XLB"),
        // Real Estate (XLRE)
        "AMT" | "VNQ" | "PLD" | "EQIX" | "PSA" | "EXR" | "AVB" | "EQR" | "O"
        | "CCI" | "DLR" | "WELL" | "SPG" | "ARE" | "MAA" | "UDR" | "KIM"
        | "LAND.L" | "SGRO.L" => Some("XLRE"),
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

    // O. News & Sentiment features (4 features)
    names.push("news_sentiment_3d".into());    // 3-day rolling avg news score
    names.push("reddit_mentions_norm".into()); // Normalised reddit mention count (0-1)
    names.push("reddit_sentiment".into());     // Reddit sentiment score (-1 to 1)
    names.push("sentiment_momentum".into());   // Today vs yesterday sentiment change

    // P. Python-validated cross-asset & multi-horizon features (12 features)
    // These features ranked in the top 20 by importance in the Python walk-forward backtest
    names.push("vix_change_1d".into());           // VIX 1-day percentage change (#1 Python feature)
    names.push("vix_sma10_dist".into());          // VIX distance from 10-day SMA as % (#2)
    names.push("tnx_change_5d".into());           // 10Y Treasury yield 5-day change (#3)
    names.push("irx_level".into());               // 3M Treasury yield level (#10)
    names.push("spy_ret_21d".into());             // SPY 21-day cumulative return (#4)
    names.push("rel_strength_vs_spy_1d".into());  // Asset 1d return minus SPY 1d return
    names.push("ret_2d".into());                  // Raw 2-day return (#17)
    names.push("ret_21d".into());                 // Raw 21-day return
    names.push("ret_63d".into());                 // Raw 63-day (quarterly) return
    names.push("vol_ratio_21_63".into());         // 21d volatility / 63d volatility (#19)
    names.push("day_of_week_raw".into());         // Trading day 0-4 normalised (#12)
    names.push("month_raw".into());               // Month 1-12 normalised (#16)

    // Q. External data source features (6 features)
    names.push("boe_rate".into());               // Bank of England base rate
    names.push("uk_gilt_10y".into());            // UK 10-year gilt yield
    names.push("ecb_rate".into());               // ECB main refinancing rate
    names.push("eu_inflation".into());           // EU HICP inflation rate
    names.push("insider_score".into());          // SEC insider buying score (US stocks, 0-1)
    names.push("short_interest_ratio".into());   // FINRA short interest ratio (US stocks, 0-1)

    // R. Python-alignment features (10 features)
    names.push("vix_regime".into());             // VIX regime categorical: 0=low(<15), 1=medium(15-25), 2=high(>25)
    names.push("logret_1d".into());              // Log return 1-day (%)
    names.push("logret_5d".into());              // Log return 5-day (%)
    names.push("logret_21d".into());             // Log return 21-day (%)
    names.push("zscore_20".into());              // Price z-score vs 20-day mean
    names.push("zscore_50".into());              // Price z-score vs 50-day mean
    names.push("stoch_k".into());               // Stochastic %K (14-day) [0-100]
    names.push("stoch_d".into());               // Stochastic %D (3-day SMA of %K) [0-100]
    names.push("price_pos_20d".into());          // Price position in 20-day range [0-1]
    names.push("price_pos_63d".into());          // Price position in 63-day range [0-1]

    // S. Sector momentum & cross-asset features (8 features) — regime-aware rotation
    names.push("sector_momentum_10d".into());    // 10-day return of asset's sector ETF (%)
    names.push("sector_momentum_20d".into());    // 20-day return of asset's sector ETF (%)
    names.push("sector_rank".into());            // Rank of sector among all sectors by 10d momentum (normalised 0-1)
    names.push("sector_vs_spy_10d".into());      // Sector ETF 10d return minus SPY 10d return (%)
    names.push("corr_with_sector_30d".into());   // 30-day rolling correlation with sector ETF
    names.push("corr_with_spy_30d".into());      // 30-day rolling correlation with SPY
    names.push("gold_spy_ratio_10d".into());     // 10-day change in GLD/SPY ratio (risk sentiment, %)
    names.push("market_breadth".into());          // Fraction of sector ETFs with positive 10d momentum [0-1]

    // T. VIX term structure, SKEW, VRP, FRED macro (10 features)
    names.push("vix_term_slope".into());          // VIX3M/VIX - 1 (contango>0 = complacent, backwardation<0 = panic)
    names.push("vix_9d_ratio".into());            // VIX9D/VIX (near-term skew)
    names.push("vix_term_spread".into());         // VIX3M - VIX (absolute spread in vol points)
    names.push("skew_level".into());              // CBOE SKEW index level (tail risk gauge)
    names.push("skew_delta_5d".into());           // SKEW 5-day change
    names.push("vrp".into());                     // Variance Risk Premium: VIX^2 - realised vol 20d^2
    names.push("hy_spread".into());               // ICE BofA HY credit spread (FRED BAMLH0A0HYM2)
    names.push("hy_spread_delta_5d".into());      // HY spread 5-day change
    names.push("breakeven_5y".into());            // 5-year breakeven inflation (FRED T5YIE)
    names.push("breakeven_delta_5d".into());      // Breakeven inflation 5-day change

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

fn mean_std(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 { return (data.first().copied().unwrap_or(0.0), 1.0); }
    let m = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, var.sqrt().max(1e-10))
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
    /// BOE base rate (Bank of England)
    pub boe_rate: Vec<f64>,
    /// UK 10-year gilt yield
    pub uk_10y_gilt: Vec<f64>,
    /// ECB main refinancing rate
    pub ecb_rate: Vec<f64>,
    /// EU inflation (HICP annual rate)
    pub eu_inflation: Vec<f64>,
    /// SEC insider trading score per asset (0-1)
    pub insider_score: f64,
    /// FINRA short interest ratio per asset (0-1)
    pub short_interest_ratio: f64,
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
    build_rich_features_horizon(prices, volumes, timestamps, market, asset_type,
        sector_etf, earnings_dates, fear_greed, ext_macro, crypto_enrich, 1)
}

/// Build features with configurable prediction horizon (in trading days).
/// horizon=1: predict next-day return (default)
/// horizon=5: predict 5-day forward return (smoother, more predictable)
pub fn build_rich_features_horizon(
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
    horizon: usize,
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

    for i in start..prices.len().saturating_sub(horizon) {
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

        f.push(safe_div(price - s7, s7) * 100.0);                 // SMA7_ratio (%)
        f.push(safe_div(price - s30, s30) * 100.0);             // SMA30_ratio (%)
        f.push(safe_div(price - s50, s50) * 100.0);             // SMA50_ratio (%)
        f.push(safe_div(price - s200, s200) * 100.0);           // SMA200_ratio (%)
        f.push(if s50 > s200 { 1.0 } else { -1.0 });           // SMA50_above_200
        f.push(safe_div(s50 - s200, s200) * 100.0);             // SMA_spread_50_200 (%)

        // 52-week high/low (252 trading days)
        let lookback_252 = &prices[i.saturating_sub(252)..=i];
        let high_52w = lookback_252.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low_52w = lookback_252.iter().cloned().fold(f64::INFINITY, f64::min);
        f.push(safe_div(price - high_52w, high_52w) * 100.0);     // price_vs_52w_high (%)
        f.push(safe_div(price - low_52w, low_52w) * 100.0);     // price_vs_52w_low (%)

        let e12 = if i < ema12.len() { ema12[i] } else { price };
        let e26 = if i < ema26.len() { ema26[i] } else { price };
        f.push(safe_div(price - e12, e12) * 100.0);               // EMA12_ratio (%)
        f.push(safe_div(price - e26, e26) * 100.0);               // EMA26_ratio (%)

        let daily_ret = safe_div(price - prices[i-1], prices[i-1]) * 100.0;
        f.push(daily_ret);                                       // daily_return (%)

        // Daily range (high - low) / close — approximate from returns
        let dr = if i >= 1 { (prices[i] - prices[i-1]).abs() / prices[i-1] * 100.0 } else { 0.0 };
        f.push(dr);                                               // daily_range_pct (%)

        // ══ B. Volume features (6) ══
        let v_now = vol_f64[i];
        let v_sma20 = mean(&vol_f64[i.saturating_sub(20)..=i]);
        let v_sma5 = mean(&vol_f64[i.saturating_sub(5)..=i]);
        let v_prev = vol_f64[i.saturating_sub(1)];

        f.push(safe_div(v_now, v_sma20));                       // volume_ratio_20d
        f.push(safe_div(v_now, v_sma5));                         // volume_ratio_5d
        f.push(safe_div(v_now - v_prev, v_prev) * 100.0);         // volume_delta_1d (%)
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

        let vol5_raw = std_dev(ret_5d);
        let vol20_raw = std_dev(ret_20d);
        let vol60_raw = std_dev(ret_60d);
        // Annualise: × √252 × 100 to get percentage annualised volatility
        let ann_factor = (252.0_f64).sqrt() * 100.0;
        let vol5 = vol5_raw * ann_factor;
        let vol20 = vol20_raw * ann_factor;
        let vol60 = vol60_raw * ann_factor;

        f.push(vol5);                                             // volatility_5d (ann %)
        f.push(vol20);                                            // volatility_20d (ann %)
        f.push(vol60);                                            // volatility_60d (ann %)
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
        f.push(safe_div(price - prices[i.saturating_sub(1)], prices[i.saturating_sub(1)]) * 100.0);   // momentum_1d (%)
        f.push(safe_div(price - prices[i.saturating_sub(3)], prices[i.saturating_sub(3)]) * 100.0);   // momentum_3d (%)
        f.push(safe_div(price - prices[i.saturating_sub(5)], prices[i.saturating_sub(5)]) * 100.0);   // momentum_5d (%)
        f.push(safe_div(price - prices[i.saturating_sub(10)], prices[i.saturating_sub(10)]) * 100.0); // momentum_10d (%)
        f.push(safe_div(price - prices[i.saturating_sub(20)], prices[i.saturating_sub(20)]) * 100.0); // momentum_20d (%)

        let ret_3d_avg = mean(&all_returns[i.saturating_sub(3)..i.min(all_returns.len())]) * 100.0;
        let ret_5d_avg = mean(&all_returns[i.saturating_sub(5)..i.min(all_returns.len())]) * 100.0;
        let ret_10d_avg = mean(&all_returns[i.saturating_sub(10)..i.min(all_returns.len())]) * 100.0;
        f.push(ret_3d_avg);                                       // return_3d_avg (%)
        f.push(ret_5d_avg);                                       // return_5d_avg (%)
        f.push(ret_10d_avg);                                      // return_10d_avg (%)

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

            f.push(vix_now);                                       // VIX_level (raw, matches Python)
            f.push(safe_div(vix_now - vix_5d, vix_5d) * 100.0);   // VIX_delta_5d (% change)
            f.push(if vix_now > 20.0 { 1.0 } else { 0.0 });     // VIX_above_20
            f.push(if vix_now > 30.0 { 1.0 } else { 0.0 });     // VIX_above_30

            let tnx_now = if mi < mkt.tnx.len() { mkt.tnx[mi] } else { 4.0 };
            let irx_now = if mi < mkt.irx.len() { mkt.irx[mi] } else { 4.0 };
            f.push(tnx_now);                                       // treasury_10y (raw, matches Python tnx_level)
            f.push(tnx_now - irx_now);                             // treasury_spread (raw)

            let spy_r = |idx: usize| -> f64 {
                if idx < mkt.spy_returns.len() { mkt.spy_returns[idx] } else { 0.0 }
            };
            f.push(spy_r(mi) * 100.0);                             // SPY_return_1d (%)
            let spy_5d_cum = if mkt.spy_returns.is_empty() { 0.0 } else {
                let end = mi.min(mkt.spy_returns.len() - 1);
                let start = mi.saturating_sub(4).min(end);
                mkt.spy_returns[start..=end].iter().sum::<f64>() * 100.0
            };
            f.push(spy_5d_cum);                                      // SPY_return_5d (% cumulative)

            // SPY 10d momentum
            let spy_10d: Vec<f64> = (0..10).map(|j| spy_r(mi.saturating_sub(j))).collect();
            f.push(spy_10d.iter().sum::<f64>() * 100.0);          // SPY_momentum_10d (%)

            // Sector returns (8 sectors)
            let sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"];
            for sector in &sectors {
                let sr = mkt.sector_returns.get(*sector)
                    .and_then(|v| if mi < v.len() { Some(v[mi]) } else { None })
                    .unwrap_or(0.0);
                f.push(sr * 100.0);                                // sector return (%)
            }

            // Gold 5d return (cumulative %)
            let gold_5d = if mkt.gold_returns.len() > mi && mi >= 5 {
                mkt.gold_returns[mi-4..=mi].iter().sum::<f64>() * 100.0
            } else { 0.0 };
            f.push(gold_5d);                                      // gold_return_5d (%)

            // Dollar 5d return (cumulative %)
            let dollar_5d = if mkt.dollar_returns.len() > mi && mi >= 5 {
                mkt.dollar_returns[mi-4..=mi].iter().sum::<f64>() * 100.0
            } else { 0.0 };
            f.push(dollar_5d);                                    // dollar_return_5d (%)

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
        let lag1_ret = if i >= 2 { safe_div(prices[i-1] - prices[i-2], prices[i-2]) * 100.0 } else { 0.0 };
        let lag2_ret = if i >= 3 { safe_div(prices[i-2] - prices[i-3], prices[i-3]) * 100.0 } else { 0.0 };
        let lag3_ret = if i >= 4 { safe_div(prices[i-3] - prices[i-4], prices[i-4]) * 100.0 } else { 0.0 };
        f.push(lag1_ret);                                           // lag1_return (%)
        f.push(lag2_ret);                                           // lag2_return (%)
        f.push(lag3_ret);                                           // lag3_return (%)

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

        let lag1_vol = std_dev(&all_returns[i.saturating_sub(21)..i.saturating_sub(1).min(all_returns.len())]) * ann_factor;
        f.push(lag1_vol);                                         // lag1_volatility (ann %)

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

        // Precompute raw (fractional) momentum values used by multiple features below
        let mom_1d_raw = safe_div(price - prices[i.saturating_sub(1)], prices[i.saturating_sub(1)]);
        let mom_5d_raw = safe_div(price - prices[i.saturating_sub(5)], prices[i.saturating_sub(5)]);
        let mom_10d_raw = safe_div(price - prices[i.saturating_sub(10)], prices[i.saturating_sub(10)]);
        let mom_20d_raw = safe_div(price - prices[i.saturating_sub(20)], prices[i.saturating_sub(20)]);
        // Percentage versions for feature output
        let mom_1d = mom_1d_raw * 100.0;
        let mom_5d = mom_5d_raw * 100.0;
        let mom_10d = mom_10d_raw * 100.0;
        let mom_20d = mom_20d_raw * 100.0;

        // stock_vs_sector_5d / stock_vs_sector_20d (all in %)
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
            f.push((mom_5d_raw - sector_5d) * 100.0);                // stock_vs_sector_5d (%)
            f.push((mom_20d_raw - sector_20d) * 100.0);              // stock_vs_sector_20d (%)
        } else {
            f.push(0.0);                                             // stock_vs_sector_5d (FX/no data)
            f.push(0.0);                                             // stock_vs_sector_20d (FX/no data)
        }

        // stock_vs_spy_5d / stock_vs_spy_20d (all in %)
        if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            let spy_5d = if mkt.spy_returns.len() > mi && mi >= 5 {
                mkt.spy_returns[mi.saturating_sub(4)..=mi].iter().sum::<f64>()
            } else { 0.0 };
            let spy_20d = if mkt.spy_returns.len() > mi && mi >= 20 {
                mkt.spy_returns[mi.saturating_sub(19)..=mi].iter().sum::<f64>()
            } else { 0.0 };
            f.push((mom_5d_raw - spy_5d) * 100.0);                   // stock_vs_spy_5d (%)
            f.push((mom_20d_raw - spy_20d) * 100.0);                 // stock_vs_spy_20d (%)
        } else {
            f.push(0.0);                                             // stock_vs_spy_5d
            f.push(0.0);                                             // stock_vs_spy_20d
        }

        f.push(0.0);                                                // breadth_score (placeholder)

        // momentum_volume_confirm: sign(momentum_5d) * volume_ratio_20d
        let vol_ratio_20d = safe_div(v_now, v_sma20);
        let mom_sign = if mom_5d_raw > 0.0 { 1.0 } else if mom_5d_raw < 0.0 { -1.0 } else { 0.0 };
        f.push(mom_sign * vol_ratio_20d);                           // momentum_volume_confirm

        // vol_adjusted_momentum: mom / vol (both in %, so ratio is unitless Sharpe-like)
        f.push(safe_div(mom_5d, vol20));                            // vol_adjusted_momentum_5d
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

        f.push(dxy_val);                                              // dxy_level (raw)
        f.push(dxy_5d);                                              // dxy_delta_5d (raw)

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

        // Yield curve slope: raw yield spread (TNX - IRX), positive = normal curve
        let yc_slope = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.tnx.len().saturating_sub(1));
            let tnx_now = if mi < mkt.tnx.len() { mkt.tnx[mi] } else { 4.0 };
            let irx_now = if mi < mkt.irx.len() { mkt.irx[mi] } else { 4.0 };
            tnx_now - irx_now
        } else {
            ext_macro.and_then(|m| {
                if i < m.yield_spread.len() { Some(m.yield_spread[i]) } else { None }
            }).unwrap_or(0.0)
        };
        f.push(yc_slope);                                           // yield_curve_slope (raw spread)

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
                spread_now - spread_5d
            } else { 0.0 }
        } else { 0.0 };
        f.push(yc_trend);                                              // yield_curve_trend

        // stock_vs_sector_10d: 10-day relative strength vs sector ETF (%)
        let svs_10d = if let (Some(mkt), Some(etf)) = (market, sector_etf) {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            let sector_rets = mkt.sector_returns.get(etf);
            let sector_10d = sector_rets
                .filter(|v| v.len() > mi && mi >= 10)
                .map(|v| v[mi.saturating_sub(9)..=mi].iter().sum::<f64>())
                .unwrap_or(0.0);
            (mom_10d_raw - sector_10d) * 100.0
        } else { 0.0 };
        f.push(svs_10d);                                              // stock_vs_sector_10d (%)

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

        // O. News & Sentiment features (4 features) — default 0.0
        // These are populated from the news_sentiment table at inference time
        f.push(0.0); // news_sentiment_3d
        f.push(0.0); // reddit_mentions_norm
        f.push(0.0); // reddit_sentiment
        f.push(0.0); // sentiment_momentum

        // ══ P. Python-validated cross-asset & multi-horizon features (12) ══

        // VIX 1-day percentage change (top Python feature by importance)
        let vix_change_1d = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.vix.len().saturating_sub(1));
            let vix_now = if mi < mkt.vix.len() { mkt.vix[mi] } else { 20.0 };
            let vix_prev = if mi >= 1 && mi < mkt.vix.len() { mkt.vix[mi - 1] } else { vix_now };
            safe_div(vix_now - vix_prev, vix_prev) * 100.0
        } else { 0.0 };
        f.push(vix_change_1d);                                            // vix_change_1d (%)

        // VIX distance from 10-day SMA (as percentage)
        let vix_sma10_dist = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.vix.len().saturating_sub(1));
            let vix_now = if mi < mkt.vix.len() { mkt.vix[mi] } else { 20.0 };
            let vix_sma10 = if mi >= 9 && mi < mkt.vix.len() {
                mean(&mkt.vix[mi.saturating_sub(9)..=mi])
            } else { vix_now };
            safe_div(vix_now - vix_sma10, vix_sma10) * 100.0
        } else { 0.0 };
        f.push(vix_sma10_dist);                                           // vix_sma10_dist (%)

        // 10Y Treasury yield 5-day change
        let tnx_change_5d_val = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.tnx.len().saturating_sub(1));
            let tnx_now = if mi < mkt.tnx.len() { mkt.tnx[mi] } else { 4.0 };
            let tnx_5d = if mi >= 5 && mi < mkt.tnx.len() { mkt.tnx[mi - 5] } else { tnx_now };
            tnx_now - tnx_5d  // raw, matches Python
        } else { 0.0 };
        f.push(tnx_change_5d_val);                                        // tnx_change_5d

        // 3M Treasury yield (IRX) level
        let irx_level_val = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.irx.len().saturating_sub(1));
            if mi < mkt.irx.len() { mkt.irx[mi] } else { 4.0 }
        } else { 0.0 };
        f.push(irx_level_val);                                            // irx_level

        // SPY 21-day cumulative return (%)
        let spy_ret_21d_val = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            if mkt.spy_returns.len() > mi && mi >= 20 {
                mkt.spy_returns[mi.saturating_sub(20)..=mi].iter().sum::<f64>() * 100.0
            } else { 0.0 }
        } else { 0.0 };
        f.push(spy_ret_21d_val);                                          // spy_ret_21d (%)

        // Relative strength vs SPY: asset 1d return - SPY 1d return (%)
        let rel_vs_spy_1d = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            let spy_1d = if mi < mkt.spy_returns.len() { mkt.spy_returns[mi] } else { 0.0 };
            (mom_1d_raw - spy_1d) * 100.0
        } else { mom_1d_raw * 100.0 };
        f.push(rel_vs_spy_1d);                                            // rel_strength_vs_spy_1d (%)

        // Multi-horizon raw returns (%)
        f.push(safe_div(price - prices[i.saturating_sub(2)], prices[i.saturating_sub(2)]) * 100.0);    // ret_2d (%)
        f.push(safe_div(price - prices[i.saturating_sub(21)], prices[i.saturating_sub(21)]) * 100.0);  // ret_21d (%)
        f.push(safe_div(price - prices[i.saturating_sub(63)], prices[i.saturating_sub(63)]) * 100.0);  // ret_63d (%)

        // Volatility ratio: 21d / 63d (vol expansion/contraction signal)
        f.push(safe_div(vol20, vol60));                                    // vol_ratio_21_63

        // Calendar raw values (integer encodings, normalised)
        // cal_dow: 0=Sun, 1=Mon, ..., 5=Fri, 6=Sat → map to trading day 0-4
        let trading_dow = if cal_dow >= 1 && cal_dow <= 5 { (cal_dow - 1) as f64 } else { 2.0 };
        f.push(trading_dow / 4.0);                                         // day_of_week_raw
        f.push(cal_month as f64 / 12.0);                                   // month_raw

        // ══ Q. External data source features (6) ══

        // BOE base rate (forward-filled, normalised to ~0-1 range)
        let boe_val = ext_macro.and_then(|m| {
            if i < m.boe_rate.len() { Some(m.boe_rate[i]) }
            else if !m.boe_rate.is_empty() { Some(*m.boe_rate.last().unwrap()) }
            else { None }
        }).unwrap_or(4.5);
        f.push(boe_val / 10.0);                                           // boe_rate

        // UK 10-year gilt yield
        let gilt_val = ext_macro.and_then(|m| {
            if i < m.uk_10y_gilt.len() { Some(m.uk_10y_gilt[i]) }
            else if !m.uk_10y_gilt.is_empty() { Some(*m.uk_10y_gilt.last().unwrap()) }
            else { None }
        }).unwrap_or(4.0);
        f.push(gilt_val / 10.0);                                          // uk_gilt_10y

        // ECB main refinancing rate
        let ecb_val = ext_macro.and_then(|m| {
            if i < m.ecb_rate.len() { Some(m.ecb_rate[i]) }
            else if !m.ecb_rate.is_empty() { Some(*m.ecb_rate.last().unwrap()) }
            else { None }
        }).unwrap_or(4.0);
        f.push(ecb_val / 10.0);                                           // ecb_rate

        // EU inflation (HICP annual %)
        let eu_infl_val = ext_macro.and_then(|m| {
            if i < m.eu_inflation.len() { Some(m.eu_inflation[i]) }
            else if !m.eu_inflation.is_empty() { Some(*m.eu_inflation.last().unwrap()) }
            else { None }
        }).unwrap_or(2.0);
        f.push(eu_infl_val / 10.0);                                       // eu_inflation

        // SEC insider score (per-asset, static for all days)
        let insider = ext_macro.map(|m| m.insider_score).unwrap_or(0.0);
        f.push(insider);                                                    // insider_score

        // FINRA short interest ratio (per-asset, static)
        let short_int = ext_macro.map(|m| m.short_interest_ratio).unwrap_or(0.0);
        f.push(short_int);                                                  // short_interest_ratio

        // ══ R. Python-alignment features (10) ══

        // vix_regime: categorical 0=low(<15), 1=medium(15-25), 2=high(>25)
        let vix_regime_val = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.vix.len().saturating_sub(1));
            let v = if mi < mkt.vix.len() { mkt.vix[mi] } else { 20.0 };
            if v < 15.0 { 0.0 } else if v <= 25.0 { 1.0 } else { 2.0 }
        } else { 1.0 };
        f.push(vix_regime_val);                                             // vix_regime

        // Log returns (% scale)
        let lp = price.max(1e-10).ln();
        let logret_1d = (lp - prices[i.saturating_sub(1)].max(1e-10).ln()) * 100.0;
        let logret_5d = (lp - prices[i.saturating_sub(5)].max(1e-10).ln()) * 100.0;
        let logret_21d = (lp - prices[i.saturating_sub(21)].max(1e-10).ln()) * 100.0;
        f.push(logret_1d);                                                  // logret_1d (%)
        f.push(logret_5d);                                                  // logret_5d (%)
        f.push(logret_21d);                                                 // logret_21d (%)

        // Z-scores: (price - mean) / std
        let zs_start_20 = i.saturating_sub(19);
        let (zm_20, zs_20) = mean_std(&prices[zs_start_20..=i]);
        let zs_start_50 = i.saturating_sub(49);
        let (zm_50, zs_50) = mean_std(&prices[zs_start_50..=i]);
        f.push(safe_div(price - zm_20, zs_20));                            // zscore_20
        f.push(safe_div(price - zm_50, zs_50));                            // zscore_50

        // Stochastic %K (14-day) and %D (3-day SMA of %K)
        let stoch_window = &prices[i.saturating_sub(13)..=i];
        let s_lo = stoch_window.iter().cloned().fold(f64::INFINITY, f64::min);
        let s_hi = stoch_window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sk = safe_div(price - s_lo, s_hi - s_lo) * 100.0;
        f.push(sk);                                                         // stoch_k [0-100]
        // %D: 3-period SMA of %K
        let sk_prev = |offset: usize| -> f64 {
            let end = i.saturating_sub(offset);
            let start = end.saturating_sub(13);
            let w = &prices[start..=end];
            let lo = w.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            safe_div(prices[end] - lo, hi - lo) * 100.0
        };
        let sd = (sk + sk_prev(1) + sk_prev(2)) / 3.0;
        f.push(sd);                                                         // stoch_d [0-100]

        // Price position in range: (price - low) / (high - low)
        let price_pos = |n: usize| -> f64 {
            let w = &prices[i.saturating_sub(n)..=i];
            let lo = w.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            safe_div(price - lo, hi - lo)
        };
        f.push(price_pos(19));                                              // price_pos_20d [0,1]
        f.push(price_pos(62));                                              // price_pos_63d [0,1]

        // ══ S. Sector momentum & cross-asset features (8) ══

        // Compute all sector 10d cumulative returns for ranking
        let all_sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY", "XLB", "XLU", "XLRE"];
        let sector_10d_rets: Vec<(&str, f64)> = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            all_sector_etfs.iter().map(|&etf| {
                let ret = mkt.sector_returns.get(etf)
                    .filter(|v| v.len() > mi && mi >= 10)
                    .map(|v| v[mi.saturating_sub(9)..=mi].iter().sum::<f64>())
                    .unwrap_or(0.0);
                (etf, ret)
            }).collect()
        } else {
            all_sector_etfs.iter().map(|&etf| (etf, 0.0)).collect()
        };

        // sector_momentum_10d: 10-day return of asset's sector ETF (%)
        let my_sector_10d = if let Some(etf) = sector_etf {
            sector_10d_rets.iter().find(|(e, _)| *e == etf).map(|(_, r)| *r).unwrap_or(0.0)
        } else { 0.0 };
        f.push(my_sector_10d * 100.0);                                      // sector_momentum_10d (%)

        // sector_momentum_20d: 20-day return of asset's sector ETF (%)
        let my_sector_20d = if let (Some(mkt), Some(etf)) = (market, sector_etf) {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            mkt.sector_returns.get(etf)
                .filter(|v| v.len() > mi && mi >= 20)
                .map(|v| v[mi.saturating_sub(19)..=mi].iter().sum::<f64>())
                .unwrap_or(0.0)
        } else { 0.0 };
        f.push(my_sector_20d * 100.0);                                      // sector_momentum_20d (%)

        // sector_rank: rank of this asset's sector among all sectors by 10d momentum (normalised 0-1)
        let sector_rank = if let Some(etf) = sector_etf {
            let mut sorted_rets: Vec<f64> = sector_10d_rets.iter().map(|(_, r)| *r).collect();
            sorted_rets.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // descending
            let my_ret = sector_10d_rets.iter().find(|(e, _)| *e == etf).map(|(_, r)| *r).unwrap_or(0.0);
            let rank = sorted_rets.iter().position(|&r| (r - my_ret).abs() < 1e-12).unwrap_or(sorted_rets.len()) + 1;
            rank as f64 / sorted_rets.len().max(1) as f64
        } else { 0.5 };
        f.push(sector_rank);                                                 // sector_rank [0-1] (lower = better)

        // sector_vs_spy_10d: sector ETF 10d return minus SPY 10d return (%)
        let spy_10d_cum = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            if mkt.spy_returns.len() > mi && mi >= 10 {
                mkt.spy_returns[mi.saturating_sub(9)..=mi].iter().sum::<f64>()
            } else { 0.0 }
        } else { 0.0 };
        f.push((my_sector_10d - spy_10d_cum) * 100.0);                      // sector_vs_spy_10d (%)

        // corr_with_sector_30d: 30-day rolling correlation with sector ETF
        let corr_sector = if let (Some(mkt), Some(etf)) = (market, sector_etf) {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            if let Some(sector_v) = mkt.sector_returns.get(etf) {
                if mi >= 29 && mi < sector_v.len() && i >= 30 {
                    let asset_rets: Vec<f64> = (0..30).map(|j| {
                        let idx = i.saturating_sub(j);
                        if idx > 0 { safe_div(prices[idx] - prices[idx-1], prices[idx-1]) } else { 0.0 }
                    }).collect();
                    let sector_rets_30: Vec<f64> = (0..30).map(|j| {
                        sector_v[mi.saturating_sub(j)]
                    }).collect();
                    correlation(&asset_rets, &sector_rets_30)
                } else { 0.0 }
            } else { 0.0 }
        } else { 0.0 };
        f.push(corr_sector);                                                 // corr_with_sector_30d [-1, 1]

        // corr_with_spy_30d: 30-day rolling correlation with SPY
        let corr_spy = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            if mi >= 29 && i >= 30 {
                let asset_rets: Vec<f64> = (0..30).map(|j| {
                    let idx = i.saturating_sub(j);
                    if idx > 0 { safe_div(prices[idx] - prices[idx-1], prices[idx-1]) } else { 0.0 }
                }).collect();
                let spy_rets_30: Vec<f64> = (0..30).map(|j| {
                    if mi.saturating_sub(j) < mkt.spy_returns.len() { mkt.spy_returns[mi.saturating_sub(j)] } else { 0.0 }
                }).collect();
                correlation(&asset_rets, &spy_rets_30)
            } else { 0.0 }
        } else { 0.0 };
        f.push(corr_spy);                                                    // corr_with_spy_30d [-1, 1]

        // gold_spy_ratio_10d: 10-day change in GLD/SPY price ratio (risk sentiment)
        let gold_spy_ratio_change = if let Some(mkt) = market {
            let mi = (i.saturating_sub(1)).min(mkt.spy_returns.len().saturating_sub(1));
            if mi >= 10 && !mkt.gold_returns.is_empty() && !mkt.spy_returns.is_empty() {
                // Cumulative returns as proxies for price ratio change
                let gold_10d: f64 = mkt.gold_returns[mi.saturating_sub(9)..=mi.min(mkt.gold_returns.len()-1)].iter().sum();
                let spy_10d_r: f64 = mkt.spy_returns[mi.saturating_sub(9)..=mi].iter().sum();
                (gold_10d - spy_10d_r) * 100.0  // positive = gold outperforming (risk-off)
            } else { 0.0 }
        } else { 0.0 };
        f.push(gold_spy_ratio_change);                                       // gold_spy_ratio_10d (%)

        // market_breadth: fraction of sector ETFs with positive 10d momentum [0-1]
        let positive_sectors = sector_10d_rets.iter().filter(|(_, r)| *r > 0.0).count();
        let breadth = positive_sectors as f64 / sector_10d_rets.len().max(1) as f64;
        f.push(breadth);                                                      // market_breadth [0-1]

        // ══ T. VIX term structure, SKEW, VRP, FRED macro (10 features) ══
        if let Some(ctx) = market {
            let mi = (i.saturating_sub(1)).min(ctx.vix.len().saturating_sub(1));

            // vix_term_slope: VIX3M/VIX - 1 (contango=complacent, backwardation=panic)
            let vix_now = if mi < ctx.vix.len() { ctx.vix[mi] } else { 0.0 };
            let vix3m_now = if mi < ctx.vix_3m.len() { ctx.vix_3m[mi] } else { 0.0 };
            let vix9d_now = if mi < ctx.vix_9d.len() { ctx.vix_9d[mi] } else { 0.0 };
            let vix_term_slope = if vix_now > 0.0 && vix3m_now > 0.0 { vix3m_now / vix_now - 1.0 } else { 0.0 };
            f.push(vix_term_slope);                                           // vix_term_slope

            // vix_9d_ratio: VIX9D/VIX (near-term panic ratio)
            let vix_9d_ratio = if vix_now > 0.0 && vix9d_now > 0.0 { vix9d_now / vix_now } else { 1.0 };
            f.push(vix_9d_ratio);                                             // vix_9d_ratio

            // vix_term_spread: VIX3M - VIX (absolute vol points)
            let vix_term_spread = if vix3m_now > 0.0 { vix3m_now - vix_now } else { 0.0 };
            f.push(vix_term_spread);                                          // vix_term_spread

            // skew_level: CBOE SKEW index (tail risk, normalised around 100)
            let skew_now = if mi < ctx.skew.len() { ctx.skew[mi] } else { 0.0 };
            f.push(skew_now / 100.0);                                         // skew_level (normalised)

            // skew_delta_5d: SKEW 5-day change
            let skew_5ago = if mi >= 5 && mi - 5 < ctx.skew.len() { ctx.skew[mi - 5] } else { skew_now };
            f.push(if skew_5ago > 0.0 { (skew_now - skew_5ago) / skew_5ago * 100.0 } else { 0.0 }); // skew_delta_5d

            // vrp: Variance Risk Premium = VIX^2 - realised_vol_20d^2
            let realised_vol_20d_sq = if i >= 20 {
                let recent_rets: Vec<f64> = (i-19..=i).map(|j| (prices[j] - prices[j-1]) / prices[j-1]).collect();
                let rv_mean = recent_rets.iter().sum::<f64>() / recent_rets.len() as f64;
                let rv_var = recent_rets.iter().map(|r| (r - rv_mean).powi(2)).sum::<f64>() / (recent_rets.len() - 1) as f64;
                rv_var * 252.0 * 10000.0 // annualised variance in pct^2
            } else { 0.0 };
            let vix_sq = vix_now * vix_now;
            f.push((vix_sq - realised_vol_20d_sq) / 100.0);                  // vrp (scaled)

            // hy_spread: HY credit spread from FRED
            let hy_now = if mi < ctx.hy_spread.len() { ctx.hy_spread[mi] } else { 0.0 };
            f.push(hy_now);                                                   // hy_spread

            // hy_spread_delta_5d
            let hy_5ago = if mi >= 5 && mi - 5 < ctx.hy_spread.len() { ctx.hy_spread[mi - 5] } else { hy_now };
            f.push(hy_now - hy_5ago);                                         // hy_spread_delta_5d

            // breakeven_5y: 5-year breakeven inflation from FRED
            let be_now = if mi < ctx.breakeven_5y.len() { ctx.breakeven_5y[mi] } else { 0.0 };
            f.push(be_now);                                                   // breakeven_5y

            // breakeven_delta_5d
            let be_5ago = if mi >= 5 && mi - 5 < ctx.breakeven_5y.len() { ctx.breakeven_5y[mi - 5] } else { be_now };
            f.push(be_now - be_5ago);                                         // breakeven_delta_5d
        } else {
            // No market context — push 10 zeros
            for _ in 0..10 { f.push(0.0); }
        }

        // ══ Label: forward percentage return over `horizon` days ══
        let label = (prices[i + horizon] - prices[i]) / prices[i] * 100.0; // percentage return

        debug_assert_eq!(f.len(), feat_count,
            "Feature count mismatch: expected {}, got {}", feat_count, f.len());

        samples.push(Sample { features: f, label });
    }

    println!("  Built {} samples × {} features (raw) for {}",
        samples.len(), feat_count, asset_type);

    // Apply feature selection (whitelist)
    let selected = prune_features(&samples);
    let dropped_count = feat_count - selected[0].features.len();
    println!("  Built {} samples × {} features for {} ({} dropped by whitelist)",
        selected.len(), selected[0].features.len(), asset_type, dropped_count);

    selected
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

    let vix_3m = histories.get("^VIX3M").cloned().unwrap_or_default();
    let vix_9d = histories.get("^VIX9D").cloned().unwrap_or_default();
    let skew = histories.get("^SKEW").cloned().unwrap_or_default();

    // FRED series stored via market_history (populated by train)
    let hy_spread = histories.get("HY_SPREAD").cloned().unwrap_or_default();
    let breakeven_5y = histories.get("BREAKEVEN_5Y").cloned().unwrap_or_default();

    MarketContext {
        vix,
        vix_3m,
        vix_9d,
        skew,
        tnx,
        irx,
        sector_returns,
        spy_returns,
        gold_returns,
        dollar_returns,
        hy_spread,
        breakeven_5y,
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

        // vix_change_1d should be present and some non-zero
        if let Some(idx) = find_idx("vix_change_1d") {
            let any_nonzero = samples.iter().any(|s| s.features[idx].abs() > 1e-10);
            assert!(any_nonzero, "vix_change_1d should have non-zero values");
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

        // day_of_week_raw should be in [0, 1] (normalised trading day / 4)
        if let Some(idx) = find_idx("day_of_week_raw") {
            for s in &samples {
                assert!((0.0..=1.0).contains(&s.features[idx]),
                    "day_of_week_raw out of range: {}", s.features[idx]);
            }
        }

        // quarter should be in [0.25, 1.0]
        if let Some(idx) = find_idx("quarter") {
            for s in &samples {
                assert!((0.2..=1.01).contains(&s.features[idx]),
                    "quarter out of range: {}", s.features[idx]);
            }
        }

        // month_raw should be in (0, 1] (month/12)
        if let Some(idx) = find_idx("month_raw") {
            for s in &samples {
                assert!((0.0..=1.01).contains(&s.features[idx]),
                    "month_raw out of range: {}", s.features[idx]);
            }
        }

        // days_since_52w_high should be in [0.0, 1.0]
        if let Some(idx) = find_idx("days_since_52w_high") {
            for s in &samples {
                assert!((0.0..=1.0).contains(&s.features[idx]),
                    "days_since_52w_high out of range: {}", s.features[idx]);
            }
        }

        // price_pos_20d should be in [0.0, 1.0]
        if let Some(idx) = find_idx("price_pos_20d") {
            for s in &samples {
                assert!((0.0..=1.001).contains(&s.features[idx]),
                    "price_pos_20d out of range: {}", s.features[idx]);
            }
        }

        // stoch_k should be in [0, 100]
        if let Some(idx) = find_idx("stoch_k") {
            for s in &samples {
                assert!((-0.001..=100.001).contains(&s.features[idx]),
                    "stoch_k out of range: {}", s.features[idx]);
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

        let gbt_config = crate::gbt::GBTConfig::default();
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

// ════════════════════════════════════════
// Label classification for SHORT signals
// ════════════════════════════════════════

/// Signal class for training data — used for class distribution analysis and weighting
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalClass {
    Buy,
    Short,
    Sell,
    Hold,
}

/// Classify a training sample into BUY/SHORT/SELL/HOLD based on return and volatility
/// - BUY: return > vol_threshold (strong upward move)
/// - SHORT: return < -vol_threshold (strong downward move, mirrors BUY)
/// - SELL: return < 0 but not strong enough for SHORT
/// - HOLD: return ~ flat (within 20% of vol_threshold)
pub fn classify_label(label: f64, vol_threshold: f64) -> SignalClass {
    let threshold = vol_threshold.max(0.001); // avoid division by zero
    if label > threshold {
        SignalClass::Buy
    } else if label < -threshold {
        SignalClass::Short
    } else if label < -threshold * 0.2 {
        SignalClass::Sell
    } else {
        SignalClass::Hold
    }
}

/// Compute volatility threshold for an asset: median absolute daily return * 1.5
pub fn compute_volatility_threshold(samples: &[Sample]) -> f64 {
    let mut abs_returns: Vec<f64> = samples.iter().map(|s| s.label.abs()).collect();
    abs_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if abs_returns.is_empty() {
        0.01
    } else {
        abs_returns[abs_returns.len() / 2]
    };
    median * 1.5
}

/// Count class distribution for a set of samples
pub fn class_distribution(samples: &[Sample], vol_threshold: f64) -> (usize, usize, usize, usize) {
    let mut buy = 0;
    let mut short = 0;
    let mut sell = 0;
    let mut hold = 0;
    for s in samples {
        match classify_label(s.label, vol_threshold) {
            SignalClass::Buy => buy += 1,
            SignalClass::Short => short += 1,
            SignalClass::Sell => sell += 1,
            SignalClass::Hold => hold += 1,
        }
    }
    (buy, short, sell, hold)
}

/// Compute class weights for binary classification (UP=1, DOWN=0)
/// If SHORT labels are underrepresented relative to BUY, upweight DOWN class samples
/// Returns (weight_for_down, weight_for_up) for use in weighted training
pub fn compute_class_weights(samples: &[Sample], vol_threshold: f64) -> (f64, f64) {
    let (buy, short, _sell, _hold) = class_distribution(samples, vol_threshold);
    let n = samples.len() as f64;
    let n_up = samples.iter().filter(|s| s.label > 0.0).count() as f64;
    let n_down = n - n_up;

    if n_up == 0.0 || n_down == 0.0 {
        return (1.0, 1.0);
    }

    // Base class weight: inverse of frequency
    let w_up = n / (2.0 * n_up);
    let w_down = n / (2.0 * n_down);

    // If SHORT is underrepresented relative to BUY, boost DOWN weight further
    let short_boost = if short > 0 && buy > 0 {
        let ratio = buy as f64 / short as f64;
        if ratio > 1.5 {
            1.0 + (ratio - 1.0).min(1.0) * 0.3 // Up to 30% extra boost
        } else {
            1.0
        }
    } else {
        1.0
    };

    (w_down * short_boost, w_up)
}
