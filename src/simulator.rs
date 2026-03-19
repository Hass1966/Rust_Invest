/// simulator — Core simulation logic extracted from validate.rs
/// =============================================================
/// Replays historical signals against real prices to answer:
/// "What if you had followed our signals for the last N days?"

use serde::Serialize;
use std::collections::HashMap;
use crate::*;

#[derive(Debug, Clone, Serialize)]
pub struct SimResult {
    pub days: usize,
    pub starting_capital: f64,
    pub final_value: f64,
    pub total_return_pct: f64,
    pub vs_buy_and_hold_pct: f64,
    pub signal_accuracy_pct: f64,
    pub inception_date: String,
    pub daily: Vec<SimDay>,
    pub per_asset: Vec<SimAsset>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimDay {
    pub date: String,
    pub value: f64,
    pub daily_return_pct: f64,
    pub correct: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimAsset {
    pub asset: String,
    pub signal_accuracy_pct: f64,
    pub contribution_pct: f64,
}

// ── Cached model infrastructure ──────────────────────────────

/// Pre-loaded model weights for a single asset (avoids re-reading JSON per date)
pub struct CachedModels {
    pub linreg: model_store::SavedWeights,
    pub logreg: model_store::SavedWeights,
    pub gbt_saved: model_store::SavedGBT,
    pub gbt_classifier: gbt::GradientBoostedClassifier,
}

/// Load all 3 model files for a symbol once
pub fn load_models_for_symbol(symbol: &str) -> Option<CachedModels> {
    let linreg = model_store::load_weights(symbol, "linreg").ok()?;
    let logreg = model_store::load_weights(symbol, "logreg").ok()?;
    let (gbt_saved, gbt_classifier) = model_store::load_gbt(symbol).ok()?;
    Some(CachedModels { linreg, logreg, gbt_saved, gbt_classifier })
}

// ── Shared helpers ───────────────────────────────────────────

/// Build market context from DB (shared by simulator + portfolio backtest)
pub fn build_market_context_from_db(database: &db::Database) -> features::MarketContext {
    let mut market_histories: HashMap<String, Vec<f64>> = HashMap::new();
    let spy_prices: Vec<f64> = database.get_stock_history("SPY")
        .unwrap_or_default().iter().map(|p| p.price).collect();
    market_histories.insert("SPY".to_string(), spy_prices);
    for ticker in features::MARKET_TICKERS {
        let prices = database.get_market_prices(ticker).unwrap_or_default();
        market_histories.insert(ticker.to_string(), prices);
    }
    features::build_market_context(&market_histories)
}

/// Load full price history for a single asset as Vec<(date, price)>
pub fn load_asset_prices(
    database: &db::Database,
    symbol: &str,
    asset_class: &str,
) -> Vec<(String, f64)> {
    match asset_class {
        "stock" => database.get_stock_history(symbol)
            .unwrap_or_default().into_iter()
            .map(|p| (p.timestamp[..10].to_string(), p.price))
            .collect(),
        "fx" => database.get_fx_history(symbol)
            .unwrap_or_default().into_iter()
            .map(|p| (p.timestamp[..10].to_string(), p.price))
            .collect(),
        "crypto" => database.get_coin_history(symbol)
            .unwrap_or_default().into_iter()
            .map(|p| (p.timestamp[..10].to_string(), p.price))
            .collect(),
        _ => Vec::new(),
    }
}

// ── Signal generation (public for portfolio backtest) ────────

/// Generate a signal for a specific historical date using cached models.
/// Cuts off price data at `date` to avoid look-ahead bias.
pub fn generate_signal_cached(
    symbol: &str,
    asset_class: &str,
    date: &str,
    asset_prices: &HashMap<String, Vec<(String, f64)>>,
    market_context: &features::MarketContext,
    models: &CachedModels,
) -> Option<String> {
    let points_full = asset_prices.get(symbol)?;

    let cutoff_points: Vec<analysis::PricePoint> = points_full.iter()
        .filter(|(d, _)| d.as_str() <= date)
        .map(|(d, p)| analysis::PricePoint {
            timestamp: d.clone(),
            price: *p,
            volume: None,
        })
        .collect();

    if cutoff_points.len() < 100 { return None; }

    let prices: Vec<f64> = cutoff_points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = cutoff_points.iter().map(|_| None).collect();
    let timestamps: Vec<String> = cutoff_points.iter().map(|p| p.timestamp.clone()).collect();

    let se: Option<&str> = match asset_class {
        "stock" => features::sector_etf_for(symbol),
        "fx" => Some(symbol),
        _ => None,
    };
    let samples = features::build_rich_features(
        &prices, &volumes, &timestamps,
        Some(market_context), asset_class,
        se,
        None, None,
    );

    if samples.is_empty() { return None; }

    let wf = infer_with_cached(symbol, &samples, models)?;

    let result = analysis::analyse_coin(symbol, &cutoff_points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(
        symbol, &wf, result.current_price,
        result.rsi_14.unwrap_or(50.0), trend,
    );

    Some(signal.signal.clone())
}

/// Run inference using pre-loaded model weights (no disk reads)
pub fn infer_with_cached(
    symbol: &str,
    samples: &[ml::Sample],
    models: &CachedModels,
) -> Option<ensemble::WalkForwardResult> {
    if samples.is_empty() { return None; }
    let n_features = samples[0].features.len();
    let feat = &samples.last().unwrap().features;

    let lin_feat = norm(feat, &models.linreg.norm_means, &models.linreg.norm_stds);
    let raw_lin: f64 = models.linreg.bias + models.linreg.weights.iter().zip(lin_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
    let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);

    let log_feat = norm(feat, &models.logreg.norm_means, &models.logreg.norm_stds);
    let log_z: f64 = models.logreg.bias + models.logreg.weights.iter().zip(log_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
    let log_prob = (1.0 / (1.0 + (-log_z).exp())).clamp(0.15, 0.85);

    let gbt_feat = norm(feat, &models.gbt_saved.norm_means, &models.gbt_saved.norm_stds);
    let gbt_prob = models.gbt_classifier.predict_proba(&gbt_feat).clamp(0.15, 0.85);

    Some(ensemble::WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: models.linreg.meta.walk_forward_accuracy,
        logistic_accuracy: models.logreg.meta.walk_forward_accuracy,
        gbt_accuracy: models.gbt_saved.meta.walk_forward_accuracy,
        lstm_accuracy: 50.0,
        gru_accuracy: 50.0,
        rf_accuracy: 50.0,
        n_folds: 1,
        total_test_samples: 0,
        linear_recent: models.linreg.meta.walk_forward_accuracy,
        logistic_recent: models.logreg.meta.walk_forward_accuracy,
        gbt_recent: models.gbt_saved.meta.walk_forward_accuracy,
        lstm_recent: 50.0,
        gru_recent: 50.0,
        rf_recent: 50.0,
        final_linear_prob: lin_prob,
        final_logistic_prob: log_prob,
        final_gbt_prob: gbt_prob,
        final_lstm_prob: 0.5,
        final_gru_prob: 0.5,
        final_rf_prob: 0.5,
        gbt_importance: Vec::new(),
        n_features,
        has_lstm: false,
        has_gru: false,
        has_rf: false,
        stacking_weights: None,
    })
}

pub fn norm(features: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    features.iter().enumerate().map(|(i, &f)| {
        let mean = means.get(i).copied().unwrap_or(0.0);
        let std = stds.get(i).copied().unwrap_or(1.0);
        if std == 0.0 { f - mean } else { (f - mean) / std }
    }).collect()
}

// ── Bulk signal generation (builds features once for all dates) ──

/// Pre-generate signals for ALL dates in the price history at once.
/// Returns a map of date → signal ("BUY"/"SELL"/"HOLD").
/// This is O(n) instead of O(n²) compared to calling generate_signal_cached per-date.
pub fn generate_signals_bulk(
    symbol: &str,
    asset_class: &str,
    all_prices: &[(String, f64)],
    market_context: &features::MarketContext,
    models: &CachedModels,
) -> std::collections::HashMap<String, String> {
    let mut signal_map = std::collections::HashMap::new();

    if all_prices.len() < 261 { return signal_map; }

    // Build features ONCE from full price history
    let prices_vec: Vec<f64> = all_prices.iter().map(|(_, p)| *p).collect();
    let volumes_vec: Vec<Option<f64>> = vec![None; all_prices.len()];
    let timestamps_vec: Vec<String> = all_prices.iter().map(|(d, _)| d.clone()).collect();

    let sector_etf: Option<&str> = match asset_class {
        "stock" => features::sector_etf_for(symbol),
        "fx" => Some(symbol),
        _ => None,
    };
    let all_samples = features::build_rich_features(
        &prices_vec, &volumes_vec, &timestamps_vec,
        Some(market_context), asset_class,
        sector_etf,
        None, None,
    );

    if all_samples.is_empty() { return signal_map; }

    // Pre-compute SMA for trend detection
    let sma_7 = analysis::sma(&prices_vec, 7);
    let sma_30 = analysis::sma(&prices_vec, 30);

    // build_rich_features starts at index 260, so:
    // sample[k] has features using data through prices[260 + k]
    // For signal on date at index j, use sample[j - 1 - 260]
    //   (features through j-1, avoiding look-ahead bias)
    let sample_offset: usize = 260;

    let n_features = all_samples[0].features.len();

    // Generate signal for each eligible date
    // Start from sample_offset + 1 (need at least one sample before the signal date)
    for j in (sample_offset + 1)..all_prices.len() {
        let sample_idx = j - 1 - sample_offset;
        if sample_idx >= all_samples.len() { break; }

        let feat = &all_samples[sample_idx].features;

        // Run inference with cached models
        let lin_feat = norm(feat, &models.linreg.norm_means, &models.linreg.norm_stds);
        let raw_lin: f64 = models.linreg.bias + models.linreg.weights.iter().zip(lin_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
        let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);

        let log_feat = norm(feat, &models.logreg.norm_means, &models.logreg.norm_stds);
        let log_z: f64 = models.logreg.bias + models.logreg.weights.iter().zip(log_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
        let log_prob = (1.0 / (1.0 + (-log_z).exp())).clamp(0.15, 0.85);

        let gbt_feat = norm(feat, &models.gbt_saved.norm_means, &models.gbt_saved.norm_stds);
        let gbt_prob = models.gbt_classifier.predict_proba(&gbt_feat).clamp(0.15, 0.85);

        let wf = ensemble::WalkForwardResult {
            symbol: symbol.to_string(),
            linear_accuracy: models.linreg.meta.walk_forward_accuracy,
            logistic_accuracy: models.logreg.meta.walk_forward_accuracy,
            gbt_accuracy: models.gbt_saved.meta.walk_forward_accuracy,
            lstm_accuracy: 50.0,
            gru_accuracy: 50.0,
            rf_accuracy: 50.0,
            n_folds: 1,
            total_test_samples: 0,
            linear_recent: models.linreg.meta.walk_forward_accuracy,
            logistic_recent: models.logreg.meta.walk_forward_accuracy,
            gbt_recent: models.gbt_saved.meta.walk_forward_accuracy,
            lstm_recent: 50.0,
            gru_recent: 50.0,
            rf_recent: 50.0,
            final_linear_prob: lin_prob,
            final_logistic_prob: log_prob,
            final_gbt_prob: gbt_prob,
            final_lstm_prob: 0.5,
            final_gru_prob: 0.5,
            final_rf_prob: 0.5,
            gbt_importance: Vec::new(),
            n_features,
            has_lstm: false,
            has_gru: false,
            has_rf: false,
            stacking_weights: None,
        };

        // Trend at j-1 (data available before signal date)
        let trend_idx = (j - 1).min(sma_7.len().saturating_sub(1));
        let trend = match (sma_7.get(trend_idx), sma_30.get(trend_idx)) {
            (Some(&s), Some(&l)) if s > l => "BULLISH",
            _ => "BEARISH",
        };

        // RSI at j-1
        let rsi = analysis::rsi(&prices_vec[..j], 14).unwrap_or(50.0);

        let signal = ensemble::ensemble_signal(
            symbol, &wf, prices_vec[j - 1], rsi, trend,
        );

        signal_map.insert(all_prices[j].0.clone(), signal.signal.clone());
    }

    signal_map
}

// ── Original simulation (uses uncached path for backward compat) ──

pub fn run_simulation(
    days: usize,
    capital: f64,
    db_path: &str,
) -> Result<SimResult, String> {
    let database = db::Database::new(db_path)
        .map_err(|e| format!("DB error: {}", e))?;

    let model_version = model_store::MODEL_VERSION;

    // Load Sharpe-weighted allocations
    let allocations = database.get_portfolio_allocations(model_version, "sharpe")
        .map_err(|e| format!("Allocations error: {}", e))?;
    if allocations.is_empty() {
        return Err("No Sharpe allocations found. Run train first.".to_string());
    }

    // Normalise weights
    let total_weight: f64 = allocations.iter().map(|a| a.weight).sum();
    let weights: HashMap<String, f64> = allocations.iter()
        .map(|a| (a.asset.clone(), a.weight / total_weight))
        .collect();

    // Market context
    let market_context = build_market_context_from_db(&database);

    // Load price history
    let mut asset_prices: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for alloc in &allocations {
        let symbol = &alloc.asset;
        let points = if stocks::STOCK_LIST.iter().any(|s| s.symbol == symbol.as_str()) {
            database.get_stock_history(symbol)
                .unwrap_or_default().into_iter()
                .map(|p| (p.timestamp[..10].to_string(), p.price))
                .collect::<Vec<_>>()
        } else if stocks::FX_LIST.iter().any(|s| s.symbol == symbol.as_str()) {
            database.get_fx_history(symbol)
                .unwrap_or_default().into_iter()
                .map(|p| (p.timestamp[..10].to_string(), p.price))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if !points.is_empty() {
            asset_prices.insert(symbol.clone(), points);
        }
    }

    // Determine trading days — use ALL available history, then limit output to `days`
    let mut all_dates: Vec<String> = {
        let first_asset = match asset_prices.values().next() {
            Some(v) => v,
            None => return Err("No price data available".to_string()),
        };
        first_asset.iter()
            .map(|(d, _)| d.clone())
            .collect()
    };
    all_dates.sort();
    all_dates.dedup();

    // Trim to the requested number of days (from the end of history)
    if all_dates.len() > days {
        all_dates = all_dates[all_dates.len() - days..].to_vec();
    }
    if all_dates.is_empty() {
        return Err("No price data found for the requested period.".to_string());
    }

    let inception_date = all_dates.first().cloned().unwrap_or_default();

    // Price lookup
    let price_lookup: HashMap<String, HashMap<String, f64>> = asset_prices.iter()
        .map(|(asset, points)| {
            let by_date: HashMap<String, f64> = points.iter().cloned().collect();
            (asset.clone(), by_date)
        })
        .collect();

    // Asset classes
    let asset_class: HashMap<String, &str> = allocations.iter()
        .map(|a| {
            let cls = if stocks::STOCK_LIST.iter().any(|s| s.symbol == a.asset.as_str()) {
                "stock"
            } else if stocks::FX_LIST.iter().any(|s| s.symbol == a.asset.as_str()) {
                "fx"
            } else {
                "crypto"
            };
            (a.asset.clone(), cls)
        })
        .collect();

    // Day-by-day simulation
    let mut portfolio_value = capital;
    let mut sim_days: Vec<SimDay> = Vec::new();
    let mut total_correct = 0_usize;
    let mut total_signals = 0_usize;
    let mut asset_stats: HashMap<String, (usize, usize, f64)> = HashMap::new();

    // Also track buy & hold
    let first_date = &all_dates[0];
    let mut bh_start_prices: HashMap<String, f64> = HashMap::new();
    for alloc in &allocations {
        if let Some(price) = price_lookup.get(&alloc.asset).and_then(|m| m.get(first_date.as_str())) {
            bh_start_prices.insert(alloc.asset.clone(), *price);
        }
    }

    for (day_idx, date) in all_dates.iter().enumerate() {
        let next_date = match all_dates.get(day_idx + 1) {
            Some(d) => d.clone(),
            None => continue, // skip last day (no next-day price to evaluate)
        };

        let mut day_weighted_return = 0.0_f64;
        let mut day_correct = 0_usize;
        let mut day_total = 0_usize;

        for alloc in &allocations {
            let symbol = &alloc.asset;
            let weight = weights[symbol];
            let cls = asset_class.get(symbol).copied().unwrap_or("stock");

            let price_today = price_lookup.get(symbol).and_then(|m| m.get(date.as_str())).copied();
            let price_next = price_lookup.get(symbol).and_then(|m| m.get(next_date.as_str())).copied();

            let (price_today, price_next) = match (price_today, price_next) {
                (Some(t), Some(n)) => (t, n),
                _ => continue,
            };

            let actual_return = (price_next - price_today) / price_today;
            let signal = generate_signal_for_date(symbol, cls, date, &asset_prices, &market_context)
                .unwrap_or_else(|| "HOLD".to_string());

            let contribution = match signal.as_str() {
                "SELL" => 0.0,
                _ => weight * actual_return,
            };
            day_weighted_return += contribution;

            let was_correct = match signal.as_str() {
                "BUY" => actual_return > 0.0,
                "SELL" => actual_return < 0.0,
                "HOLD" => true,
                _ => false,
            };

            if signal != "HOLD" {
                if was_correct { day_correct += 1; }
                day_total += 1;
            }

            let entry = asset_stats.entry(symbol.clone()).or_insert((0, 0, 0.0));
            entry.2 += contribution * 100.0;
            if signal != "HOLD" {
                entry.1 += 1;
                if was_correct { entry.0 += 1; }
            }
        }

        portfolio_value *= 1.0 + day_weighted_return;
        total_correct += day_correct;
        total_signals += day_total;

        sim_days.push(SimDay {
            date: date.clone(),
            value: (portfolio_value * 100.0).round() / 100.0,
            daily_return_pct: (day_weighted_return * 100.0 * 100.0).round() / 100.0,
            correct: day_correct,
            total: day_total,
        });
    }

    // Buy & hold comparison
    let last_date = all_dates.last().cloned().unwrap_or_default();
    let mut bh_return = 0.0_f64;
    let mut bh_count = 0;
    for alloc in &allocations {
        if let (Some(start_p), Some(end_p)) = (
            bh_start_prices.get(&alloc.asset),
            price_lookup.get(&alloc.asset).and_then(|m| m.get(last_date.as_str())),
        ) {
            let weight = weights[&alloc.asset];
            bh_return += weight * (end_p - start_p) / start_p;
            bh_count += 1;
        }
    }
    let bh_pct = if bh_count > 0 { bh_return * 100.0 } else { 0.0 };

    let total_return = (portfolio_value - capital) / capital * 100.0;
    let accuracy = if total_signals > 0 {
        total_correct as f64 / total_signals as f64 * 100.0
    } else { 0.0 };

    // Per-asset breakdown
    let mut per_asset: Vec<SimAsset> = asset_stats.into_iter().map(|(asset, (correct, total, contrib))| {
        let acc = if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 };
        SimAsset {
            asset,
            signal_accuracy_pct: (acc * 10.0).round() / 10.0,
            contribution_pct: (contrib * 100.0).round() / 100.0,
        }
    }).collect();
    per_asset.sort_by(|a, b| b.contribution_pct.partial_cmp(&a.contribution_pct).unwrap_or(std::cmp::Ordering::Equal));

    Ok(SimResult {
        days: sim_days.len(),
        starting_capital: capital,
        final_value: (portfolio_value * 100.0).round() / 100.0,
        total_return_pct: (total_return * 100.0).round() / 100.0,
        vs_buy_and_hold_pct: (bh_pct * 100.0).round() / 100.0,
        signal_accuracy_pct: (accuracy * 10.0).round() / 10.0,
        inception_date,
        daily: sim_days,
        per_asset,
    })
}

// ── Private helpers for run_simulation (uncached path) ───────

fn generate_signal_for_date(
    symbol: &str,
    asset_class: &str,
    date: &str,
    asset_prices: &HashMap<String, Vec<(String, f64)>>,
    market_context: &features::MarketContext,
) -> Option<String> {
    let points_full = asset_prices.get(symbol)?;

    let cutoff_points: Vec<analysis::PricePoint> = points_full.iter()
        .filter(|(d, _)| d.as_str() <= date)
        .map(|(d, p)| analysis::PricePoint {
            timestamp: d.clone(),
            price: *p,
            volume: None,
        })
        .collect();

    if cutoff_points.len() < 100 { return None; }

    let prices: Vec<f64> = cutoff_points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = cutoff_points.iter().map(|_| None).collect();
    let timestamps: Vec<String> = cutoff_points.iter().map(|p| p.timestamp.clone()).collect();

    let se2: Option<&str> = match asset_class {
        "stock" => features::sector_etf_for(symbol),
        "fx" => Some(symbol),
        _ => None,
    };
    let samples = features::build_rich_features(
        &prices, &volumes, &timestamps,
        Some(market_context), asset_class,
        se2,
        None, None,
    );

    if samples.is_empty() { return None; }

    let wf = infer_quiet(symbol, &samples)?;

    let result = analysis::analyse_coin(symbol, &cutoff_points);
    let sma_7 = analysis::sma(&prices, 7);
    let sma_30 = analysis::sma(&prices, 30);
    let trend = match (sma_7.last(), sma_30.last()) {
        (Some(s), Some(l)) if s > l => "BULLISH",
        _ => "BEARISH",
    };

    let signal = ensemble::ensemble_signal(
        symbol, &wf, result.current_price,
        result.rsi_14.unwrap_or(50.0), trend,
    );

    Some(signal.signal.clone())
}

fn infer_quiet(symbol: &str, samples: &[ml::Sample]) -> Option<ensemble::WalkForwardResult> {
    if samples.is_empty() { return None; }
    let n_features = samples[0].features.len();

    let linreg_saved = model_store::load_weights(symbol, "linreg").ok()?;
    let logreg_saved = model_store::load_weights(symbol, "logreg").ok()?;
    let (gbt_saved, gbt_classifier) = model_store::load_gbt(symbol).ok()?;

    let feat = &samples.last().unwrap().features;

    let lin_feat = norm(feat, &linreg_saved.norm_means, &linreg_saved.norm_stds);
    let raw_lin: f64 = linreg_saved.bias + linreg_saved.weights.iter().zip(lin_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
    let lin_prob = (1.0 / (1.0 + (-raw_lin).exp())).clamp(0.15, 0.85);

    let log_feat = norm(feat, &logreg_saved.norm_means, &logreg_saved.norm_stds);
    let log_z: f64 = logreg_saved.bias + logreg_saved.weights.iter().zip(log_feat.iter()).map(|(w, f)| w * f).sum::<f64>();
    let log_prob = (1.0 / (1.0 + (-log_z).exp())).clamp(0.15, 0.85);

    let gbt_feat = norm(feat, &gbt_saved.norm_means, &gbt_saved.norm_stds);
    let gbt_prob = gbt_classifier.predict_proba(&gbt_feat).clamp(0.15, 0.85);

    Some(ensemble::WalkForwardResult {
        symbol: symbol.to_string(),
        linear_accuracy: linreg_saved.meta.walk_forward_accuracy,
        logistic_accuracy: logreg_saved.meta.walk_forward_accuracy,
        gbt_accuracy: gbt_saved.meta.walk_forward_accuracy,
        lstm_accuracy: 50.0,
        gru_accuracy: 50.0,
        rf_accuracy: 50.0,
        n_folds: 1,
        total_test_samples: 0,
        linear_recent: linreg_saved.meta.walk_forward_accuracy,
        logistic_recent: logreg_saved.meta.walk_forward_accuracy,
        gbt_recent: gbt_saved.meta.walk_forward_accuracy,
        lstm_recent: 50.0,
        gru_recent: 50.0,
        rf_recent: 50.0,
        final_linear_prob: lin_prob,
        final_logistic_prob: log_prob,
        final_gbt_prob: gbt_prob,
        final_lstm_prob: 0.5,
        final_gru_prob: 0.5,
        final_rf_prob: 0.5,
        gbt_importance: Vec::new(),
        n_features,
        has_lstm: false,
        has_gru: false,
        has_rf: false,
        stacking_weights: None,
    })
}
