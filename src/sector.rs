/// Sector Rotation Layer
/// ====================
/// Groups assets into sectors, calculates momentum scores from existing signals,
/// and provides sector-weighted allocation multipliers.

use serde::Serialize;
use std::collections::HashMap;

/// Sector categories for grouping assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Sector {
    Tech,
    Energy,
    Financials,
    Healthcare,
    Industrials,
    Consumer,
    Commodities,
    Crypto,
    FX,
    Defensive,
    RealEstate,
}

impl Sector {
    pub fn label(&self) -> &'static str {
        match self {
            Sector::Tech => "Technology",
            Sector::Energy => "Energy",
            Sector::Financials => "Financials",
            Sector::Healthcare => "Healthcare",
            Sector::Industrials => "Industrials",
            Sector::Consumer => "Consumer",
            Sector::Commodities => "Commodities",
            Sector::Crypto => "Crypto",
            Sector::FX => "FX",
            Sector::Defensive => "Defensive",
            Sector::RealEstate => "Real Estate",
        }
    }
}

/// Map an asset symbol to its sector.
pub fn classify_sector(symbol: &str) -> Sector {
    match symbol {
        // ── Technology ──
        "AAPL" | "MSFT" | "GOOGL" | "AMZN" | "NVDA" | "META" | "TSLA" | "AMD" | "AVGO"
        | "NFLX" | "CRM" | "ARM" | "INTC" | "IBM" | "QCOM" | "TSM" | "SHOP" | "UBER"
        | "PLTR" | "ADBE" | "NOW" | "INTU" | "AMAT" | "MU" | "LRCX" | "KLAC" | "SNPS"
        | "CDNS" | "FTNT" | "PANW" | "CRWD" | "TEAM" | "ZS" | "DDOG" | "NET" | "MDB"
        | "SNOW" | "ORCL" | "SAP.DE"
        | "QQQ" | "XLK" => Sector::Tech,

        // ── Financials ──
        "JPM" | "GS" | "BAC" | "WFC" | "V" | "MA" | "MS" | "BLK" | "SCHW" | "AXP"
        | "COF" | "USB" | "PNC" | "TFC" | "SPGI" | "MCO" | "ICE" | "CME" | "CB"
        | "BRK-B" | "MMC" | "TRV" | "AFL" | "COIN"
        | "HSBA.L" | "LLOY.L" | "BARC.L" | "NWG.L" | "LGEN.L" | "III.L" | "MNG.L"
        | "ALV.DE"
        | "XLF" => Sector::Financials,

        // ── Healthcare ──
        "SNY" | "JNJ" | "UNH" | "LLY" | "PFE" | "MRNA" | "ABBV" | "MRK" | "BMY"
        | "CVS" | "CI" | "ELV" | "HUM" | "MOH" | "ISRG" | "BSX" | "SYK" | "MDT"
        | "EW" | "REGN" | "VRTX" | "BIIB" | "GILD"
        | "AZN.L" | "GSK.L" | "SAN.PA"
        | "XLV" | "VHT" | "IHI" => Sector::Healthcare,

        // ── Energy ──
        "XOM" | "CVX" | "COP" | "SLB" | "HAL" | "BKR" | "DVN" | "FANG" | "MPC"
        | "PSX" | "VLO"
        | "BP.L" | "SHEL.L" | "CNA.L"
        | "XLE" | "VDE" => Sector::Energy,

        // ── Industrials / Defense ──
        "LMT" | "RTX" | "NOC" | "BA" | "GD" | "CAT" | "DE" | "MMM" | "HON" | "GE"
        | "EMR" | "UPS" | "FDX" | "CSX" | "NSC" | "UNP" | "ETN" | "PH" | "ROK"
        | "IR" | "CARR" | "OTIS"
        | "RR.L" | "QQ.L" | "AIR.PA" | "SAF.PA" | "SIE.DE" | "MBG.DE"
        | "XLI" => Sector::Industrials,

        // ── Consumer ──
        "WMT" | "TGT" | "COST" | "HD" | "NKE" | "MCD" | "SBUX" | "TJX" | "ORLY"
        | "AZO" | "DLTR" | "DG" | "YUM" | "CMG" | "ABNB" | "BKNG" | "MAR" | "HLT"
        | "ULVR.L" | "DGE.L" | "CPG.L" | "TSCO.L" | "SBRY.L"
        | "OR.PA" | "MC.PA" => Sector::Consumer,

        // ── Defensive (Staples + Utilities + Bonds + Telco + Tobacco) ──
        "KO" | "PEP" | "PG" | "GIS" | "CL" | "MO" | "T" | "VZ" | "DUK" | "NEE"
        | "SO" | "AMT"
        | "TLT" | "AGG" | "BND" | "IBTL.L" | "VGLT" | "SHY"
        | "NG.L" | "SSE.L" | "IMB.L" | "BATS.L" | "PSON.L" | "VOD.L" | "BT-A.L"
        | "DCC.L" | "REL.L" | "WPP.L" | "EXPN.L"
        | "BAS.DE"
        | "XLP" | "XLU" | "VPU" | "IDU"
        | "SPY" | "DIA" | "ISF.L" | "SH" => Sector::Defensive,

        // ── Commodities ──
        "GLD" | "CL=F" | "USO" | "SLV" | "CPER"
        | "GLEN.L" | "AAL.L" | "ANTO.L" | "SMDS.L" | "MNDI.L"
        | "LIN" | "APD" | "ECL" | "NEM" | "FCX" | "NUE" | "ALB"
        | "PDBC" | "DBC" => Sector::Commodities,

        // ── Real Estate ──
        "PLD" | "EQIX" | "PSA" | "EXR" | "AVB" | "EQR" | "O"
        | "XLRE" | "VNQ" => Sector::RealEstate,

        // ── FX (all =X pairs) ──
        s if s.ends_with("=X") => Sector::FX,

        // ── Crypto (detected by asset_class, but also by symbol) ──
        "BTC-USD" | "ETH-USD" | "SOL-USD" | "ADA-USD" | "DOT-USD"
        | "DOGE-USD" | "AVAX-USD" | "MATIC-USD" | "XRP-USD" | "LINK-USD"
        | "UNI-USD" | "AAVE-USD" | "ATOM-USD" | "NEAR-USD" | "FTM-USD"
        | "ARB-USD" | "OP-USD" | "APT-USD" | "SUI-USD" | "SEI-USD"
        | "RENDER-USD" | "FET-USD" | "JASMY-USD" | "INJ-USD" | "TIA-USD" => Sector::Crypto,

        // Default: classify by asset_class if available, else Defensive
        _ => Sector::Defensive,
    }
}

/// Overload: classify by symbol with an asset_class fallback
pub fn classify_sector_with_class(symbol: &str, asset_class: &str) -> Sector {
    match asset_class {
        "crypto" => Sector::Crypto,
        "fx" => Sector::FX,
        _ => classify_sector(symbol),
    }
}

/// A signal's contribution to sector scoring
pub struct SignalInput {
    pub asset: String,
    pub asset_class: String,
    pub signal: String,          // "BUY", "SELL", "SHORT", "HOLD"
    pub probability_up: f64,     // 0-100
    pub confidence: f64,         // 0-10
}

/// Sector-level aggregated metrics
#[derive(Debug, Clone, Serialize)]
pub struct SectorScore {
    pub sector: Sector,
    pub label: String,
    pub asset_count: usize,
    pub buy_count: usize,
    pub sell_count: usize,
    pub hold_count: usize,
    /// Momentum score: -100 (all strong sells) to +100 (all strong buys)
    pub momentum_score: f64,
    /// Average probability_up across all assets in sector (0-100)
    pub avg_probability_up: f64,
    /// Average confidence across all assets (0-10)
    pub avg_confidence: f64,
    /// Recommended allocation weight multiplier: 0.5 (weak) to 1.5 (strong)
    pub weight_multiplier: f64,
}

/// Calculate sector scores from a list of signal inputs.
pub fn calculate_sector_scores(signals: &[SignalInput]) -> Vec<SectorScore> {
    // Group signals by sector
    let mut sector_signals: HashMap<Sector, Vec<&SignalInput>> = HashMap::new();
    for sig in signals {
        let sector = classify_sector_with_class(&sig.asset, &sig.asset_class);
        sector_signals.entry(sector).or_default().push(sig);
    }

    let mut scores: Vec<SectorScore> = sector_signals
        .into_iter()
        .map(|(sector, sigs)| {
            let asset_count = sigs.len();
            let buy_count = sigs.iter().filter(|s| s.signal == "BUY").count();
            let sell_count = sigs.iter().filter(|s| s.signal == "SELL" || s.signal == "SHORT").count();
            let hold_count = sigs.iter().filter(|s| s.signal == "HOLD").count();

            // Momentum = confidence-weighted average of signal direction
            // BUY = +1, HOLD = 0, SELL/SHORT = -1, weighted by confidence
            let (weighted_sum, weight_total) = sigs.iter().fold((0.0, 0.0), |(sum, wt), s| {
                let direction = match s.signal.as_str() {
                    "BUY" => 1.0,
                    "SELL" | "SHORT" => -1.0,
                    _ => 0.0,
                };
                let w = s.confidence.max(0.1); // min weight to avoid division by zero
                (sum + direction * w, wt + w)
            });
            let momentum_score = if weight_total > 0.0 {
                (weighted_sum / weight_total) * 100.0
            } else {
                0.0
            };

            let avg_probability_up = if asset_count > 0 {
                sigs.iter().map(|s| s.probability_up).sum::<f64>() / asset_count as f64
            } else {
                50.0
            };

            let avg_confidence = if asset_count > 0 {
                sigs.iter().map(|s| s.confidence).sum::<f64>() / asset_count as f64
            } else {
                5.0
            };

            // Weight multiplier: map momentum from [-100, 100] to [0.5, 1.5]
            let weight_multiplier = 1.0 + (momentum_score / 200.0); // -100 → 0.5, 0 → 1.0, +100 → 1.5
            let weight_multiplier = weight_multiplier.clamp(0.5, 1.5);

            SectorScore {
                sector,
                label: sector.label().to_string(),
                asset_count,
                buy_count,
                sell_count,
                hold_count,
                momentum_score: (momentum_score * 10.0).round() / 10.0,
                avg_probability_up: (avg_probability_up * 10.0).round() / 10.0,
                avg_confidence: (avg_confidence * 10.0).round() / 10.0,
                weight_multiplier: (weight_multiplier * 100.0).round() / 100.0,
            }
        })
        .collect();

    // Sort by momentum descending
    scores.sort_by(|a, b| b.momentum_score.partial_cmp(&a.momentum_score).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Summary response for the API
#[derive(Debug, Clone, Serialize)]
pub struct SectorOverview {
    pub sectors: Vec<SectorScore>,
    pub strongest_sector: String,
    pub weakest_sector: String,
    pub total_assets: usize,
}

/// Build a complete sector overview from signals
pub fn build_sector_overview(signals: &[SignalInput]) -> SectorOverview {
    let scores = calculate_sector_scores(signals);
    let strongest = scores.first().map(|s| s.label.clone()).unwrap_or_default();
    let weakest = scores.last().map(|s| s.label.clone()).unwrap_or_default();
    let total_assets = signals.len();

    SectorOverview {
        sectors: scores,
        strongest_sector: strongest,
        weakest_sector: weakest,
        total_assets,
    }
}
