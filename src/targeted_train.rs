/// Targeted Training — Single-asset retraining for the agent
/// =========================================================
/// Extracts the per-asset training logic from train.rs into a shared function
/// that both the weekly train binary and the agent can use.
///
/// Usage: targeted_train::train_single_asset("AAPL", &database)

use crate::{db, features, ensemble, model_store, ml, gbt, regime, stocks};
use std::collections::HashMap;

/// Result of a single-asset training run
#[derive(Debug, Clone)]
pub struct TrainResult {
    pub symbol: String,
    pub linreg_accuracy: f64,
    pub logreg_accuracy: f64,
    pub gbt_accuracy: f64,
    pub lstm_accuracy: f64,
    pub regime_accuracy: f64,
    pub ensemble_accuracy: f64,
    pub pre_accuracy: f64,
    pub post_accuracy: f64,
    pub success: bool,
    pub error: Option<String>,
}

/// Train all models for a single asset. Returns training accuracy results.
///
/// This is the same walk-forward + model-save logic as train.rs, but for one asset.
/// The caller (agent) is responsible for backup/restore.
pub fn train_single_asset(symbol: &str, db_path: &str) -> TrainResult {
    let mut result = TrainResult {
        symbol: symbol.to_string(),
        linreg_accuracy: 0.0,
        logreg_accuracy: 0.0,
        gbt_accuracy: 0.0,
        lstm_accuracy: 0.0,
        regime_accuracy: 0.0,
        ensemble_accuracy: 0.0,
        pre_accuracy: 0.0,
        post_accuracy: 0.0,
        success: false,
        error: None,
    };

    let database = match db::Database::new(db_path) {
        Ok(d) => { d.set_wal_mode(); d }
        Err(e) => {
            result.error = Some(format!("DB open failed: {}", e));
            return result;
        }
    };

    // Pre-retrain accuracy baseline
    result.pre_accuracy = model_store::load_model_accuracy(symbol);

    // Determine asset class and load price data
    let asset_class = detect_asset_class(symbol);
    let points = match load_price_data(&database, symbol, &asset_class) {
        Ok(p) => p,
        Err(e) => {
            result.error = Some(e);
            return result;
        }
    };

    if points.len() < 300 {
        result.error = Some(format!("Insufficient data: {} points (need 300+)", points.len()));
        return result;
    }

    // Build market context from DB
    let market_context = build_market_context_from_db(&database);
    let ext_macro = build_ext_macro_from_db(&database);

    let prices: Vec<f64> = points.iter().map(|p| p.price).collect();
    let volumes: Vec<Option<f64>> = points.iter().map(|p| p.volume).collect();
    let timestamps: Vec<String> = points.iter().map(|p| p.timestamp.clone()).collect();

    let sector_etf = if asset_class == "stock" {
        features::sector_etf_for(symbol)
    } else if asset_class == "fx" {
        Some(symbol)
    } else {
        None
    };

    let samples = features::build_rich_features_ext(
        &prices, &volumes, &timestamps,
        Some(&market_context), &asset_class,
        sector_etf, None, None,
        Some(&ext_macro), None,
    );

    if samples.len() < 100 {
        result.error = Some(format!("Too few samples after feature engineering: {}", samples.len()));
        return result;
    }

    let vol_threshold = features::compute_volatility_threshold(&samples);
    let (w_down, w_up) = features::compute_class_weights(&samples, vol_threshold);

    let n_feat = samples[0].features.len();
    let train_window = (samples.len() as f64 * 0.6) as usize;
    let test_window = 30.min(samples.len() / 10);
    let step = test_window;

    // Load ensemble overrides
    let ensemble_overrides = ensemble::load_ensemble_overrides();

    // Walk-forward evaluation
    let wf = match ensemble::walk_forward_samples(symbol, &samples, train_window, test_window, step) {
        Some(wf) => wf,
        None => {
            result.error = Some("Walk-forward returned None".to_string());
            return result;
        }
    };

    result.linreg_accuracy = wf.linear_accuracy;
    result.logreg_accuracy = wf.logistic_accuracy;
    result.gbt_accuracy = wf.gbt_accuracy;
    result.lstm_accuracy = if wf.has_lstm { wf.lstm_accuracy } else { 0.0 };

    // Regime model
    if let Some(rw) = regime::walk_forward_regime(symbol, &samples, train_window, test_window, step) {
        result.regime_accuracy = rw.overall_accuracy;
    }

    let ov = ensemble::get_override(&ensemble_overrides, symbol);
    result.ensemble_accuracy = compute_ensemble_accuracy(&wf, &ov);

    // Train and save final-fold models
    let last_train_end = {
        let mut s = 0;
        let mut last = 0;
        while s + train_window + test_window <= samples.len() {
            last = s + train_window;
            s += step;
        }
        last
    };
    let mut last_fold: Vec<ml::Sample> = samples[last_train_end.saturating_sub(train_window)..last_train_end].to_vec();
    let (means, stds) = ml::normalise(&mut last_fold);

    let recency_weights = ensemble::compute_recency_weights(last_fold.len());
    let class_adjusted_weights: Vec<f64> = last_fold.iter().zip(recency_weights.iter()).map(|(s, &rw)| {
        let cw = if s.label > 0.0 { w_up } else { w_down };
        rw * cw
    }).collect();

    // LinReg
    let mut lin = ml::LinearRegression::new(n_feat);
    lin.train_weighted(&last_fold, Some(&class_adjusted_weights), 0.005, 3000);
    if let Err(e) = model_store::save_weights(symbol, "linreg", &lin.weights, lin.bias, n_feat, last_fold.len(), wf.linear_accuracy, &means, &stds) {
        result.error = Some(format!("Failed to save linreg: {}", e));
        return result;
    }

    // LogReg
    let mut log = ml::LogisticRegression::new(n_feat);
    log.train_weighted(&last_fold, Some(&class_adjusted_weights), 0.01, 3000);
    if let Err(e) = model_store::save_weights(symbol, "logreg", &log.weights, log.bias, n_feat, last_fold.len(), wf.logistic_accuracy, &means, &stds) {
        result.error = Some(format!("Failed to save logreg: {}", e));
        return result;
    }

    // GBT
    let x_train: Vec<Vec<f64>> = last_fold.iter().map(|s| s.features.clone()).collect();
    let y_train: Vec<f64> = last_fold.iter().map(|s| if s.label > 0.0 { 1.0 } else { 0.0 }).collect();
    let val_start = (x_train.len() as f64 * 0.85) as usize;
    let (x_t, x_v) = x_train.split_at(val_start);
    let (y_t, y_v) = y_train.split_at(val_start);
    let gbt_recency = &class_adjusted_weights[..x_t.len()];
    let gbt_config = gbt::GBTConfig::default();
    let gbt_model = gbt::GradientBoostedClassifier::train_weighted(x_t, y_t, Some(gbt_recency), Some(x_v), Some(y_v), gbt_config);
    if let Err(e) = model_store::save_gbt(symbol, &gbt_model, last_fold.len(), wf.gbt_accuracy, &means, &stds) {
        result.error = Some(format!("Failed to save gbt: {}", e));
        return result;
    }

    result.post_accuracy = (wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy) / 3.0;
    result.success = true;

    // Log to retrain_log table
    let (buy_count, short_count, sell_count, hold_count) = features::class_distribution(&samples, vol_threshold);
    let _ = database.insert_retrain_log(
        symbol, "ensemble",
        result.pre_accuracy, result.post_accuracy,
        buy_count as i64, sell_count as i64, short_count as i64, hold_count as i64,
    );

    println!("  [TargetedTrain] {} complete: LinReg:{:.1}% LogReg:{:.1}% GBT:{:.1}%",
        symbol, wf.linear_accuracy, wf.logistic_accuracy, wf.gbt_accuracy);

    result
}

/// Detect asset class from symbol
pub fn detect_asset_class(symbol: &str) -> String {
    // Check FX list
    for fx in stocks::FX_LIST {
        if fx.symbol == symbol { return "fx".to_string(); }
    }
    // Crypto IDs are lowercase without special chars
    // But for targeted retrain, we only support stock/FX currently
    // (crypto requires cross-asset enrichment which needs all coins)
    "stock".to_string()
}

/// Load price data for an asset from the database
fn load_price_data(database: &db::Database, symbol: &str, asset_class: &str) -> Result<Vec<crate::analysis::PricePoint>, String> {
    match asset_class {
        "fx" => database.get_fx_history(symbol).map_err(|e| format!("FX data error: {}", e)),
        _ => database.get_stock_history(symbol).map_err(|e| format!("Stock data error: {}", e)),
    }
}

/// Build market context from database (same as train.rs)
fn build_market_context_from_db(database: &db::Database) -> features::MarketContext {
    let tickers = ["^VIX", "^TNX", "^TYX", "SPY", "XLF", "XLK", "XLE", "XLV", "XLI", "DIA", "QQQ", "IWM"];
    let mut market_histories = HashMap::new();
    for ticker in &tickers {
        if let Ok(prices) = database.get_market_prices(ticker) {
            market_histories.insert(ticker.to_string(), prices);
        }
    }
    market_histories.insert("HY_SPREAD".to_string(), database.get_market_prices("HY_SPREAD").unwrap_or_default());
    market_histories.insert("BREAKEVEN_5Y".to_string(), database.get_market_prices("BREAKEVEN_5Y").unwrap_or_default());
    features::build_market_context(&market_histories)
}

/// Build extended macro data from database
fn build_ext_macro_from_db(database: &db::Database) -> features::ExtendedMacro {
    features::ExtendedMacro {
        dxy: database.get_market_prices("UUP").unwrap_or_default(),
        yield_spread: vec![],
        fed_funds: vec![],
        boe_rate: vec![],
        uk_10y_gilt: vec![],
        ecb_rate: vec![],
        eu_inflation: vec![],
        insider_score: 0.0,
        short_interest_ratio: 0.0,
    }
}

/// Compute ensemble accuracy (same logic as train.rs)
fn compute_ensemble_accuracy(wf: &ensemble::WalkForwardResult, ov: &ensemble::EnsembleOverride) -> f64 {
    let mut accs = Vec::new();
    if ov.use_linreg { accs.push(wf.linear_accuracy); }
    if ov.use_logreg { accs.push(wf.logistic_accuracy); }
    if ov.use_gbt { accs.push(wf.gbt_accuracy); }
    if wf.has_lstm { accs.push(wf.lstm_accuracy); }
    if accs.is_empty() {
        (wf.linear_accuracy + wf.logistic_accuracy + wf.gbt_accuracy) / 3.0
    } else {
        accs.iter().sum::<f64>() / accs.len() as f64
    }
}
