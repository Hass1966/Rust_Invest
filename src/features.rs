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

/// Feature names — all 80+
pub fn feature_names() -> Vec<String> {
    let mut names = Vec::new();

    // A. Price-derived technical (20 features)
    names.push("RSI_14".into());
    names.push("RSI_7".into());
    names.push("RSI_delta_3d".into());
    names.push("RSI_delta_7d".into());
    names.push("MACD_hist".into());
    names.push("MACD_hist_delta".into());
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
pub fn build_rich_features(
    prices: &[f64],
    volumes: &[Option<f64>],
    timestamps: &[String],
    market: Option<&MarketContext>,
    asset_type: &str,
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

        // ══ Label: next day return ══
        let label = prices[i+1] - prices[i]; // positive = up

        debug_assert_eq!(f.len(), feat_count,
            "Feature count mismatch: expected {}, got {}", feat_count, f.len());

        samples.push(Sample { features: f, label });
    }

    println!("  Built {} samples × {} features for {}",
        samples.len(), feat_count, asset_type);

    samples
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
