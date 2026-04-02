use serde::Deserialize;

pub struct StockInfo {
    pub symbol: &'static str,
    pub name: &'static str,
}

pub const STOCK_LIST: &[StockInfo] = &[
    // Indices
    StockInfo { symbol: "SPY", name: "S&P 500 ETF" },
    StockInfo { symbol: "QQQ", name: "Nasdaq 100 ETF" },
    StockInfo { symbol: "DIA", name: "Dow Jones ETF" },
    StockInfo { symbol: "XLF", name: "Financial Select ETF" },
    StockInfo { symbol: "XLE", name: "Energy Select ETF" },
    StockInfo { symbol: "XLV", name: "Healthcare Select ETF" },
    StockInfo { symbol: "XLI", name: "Industrial Select ETF" },
    // Big Tech
    StockInfo { symbol: "AAPL", name: "Apple" },
    StockInfo { symbol: "MSFT", name: "Microsoft" },
    StockInfo { symbol: "GOOGL", name: "Alphabet" },
    StockInfo { symbol: "AMZN", name: "Amazon" },
    StockInfo { symbol: "NVDA", name: "NVIDIA" },
    StockInfo { symbol: "META", name: "Meta" },
    StockInfo { symbol: "TSLA", name: "Tesla" },
    StockInfo { symbol: "AMD", name: "AMD" },
    StockInfo { symbol: "AVGO", name: "Broadcom" },
    StockInfo { symbol: "NFLX", name: "Netflix" },
    StockInfo { symbol: "CRM", name: "Salesforce" },
    StockInfo { symbol: "ARM", name: "ARM Holdings" },
    StockInfo { symbol: "INTC", name: "Intel" },
    StockInfo { symbol: "QCOM", name: "Qualcomm" },
    StockInfo { symbol: "TSM", name: "Taiwan Semiconductor" },
    // Finance
    StockInfo { symbol: "JPM", name: "JPMorgan Chase" },
    StockInfo { symbol: "GS", name: "Goldman Sachs" },
    StockInfo { symbol: "BAC", name: "Bank of America" },
    StockInfo { symbol: "WFC", name: "Wells Fargo" },
    StockInfo { symbol: "V", name: "Visa" },
    StockInfo { symbol: "MA", name: "Mastercard" },
    // Healthcare & Pharma
    StockInfo { symbol: "JNJ", name: "Johnson & Johnson" },
    StockInfo { symbol: "UNH", name: "UnitedHealth" },
    StockInfo { symbol: "LLY", name: "Eli Lilly" },
    StockInfo { symbol: "PFE", name: "Pfizer" },
    StockInfo { symbol: "MRNA", name: "Moderna" },
    StockInfo { symbol: "ABBV", name: "AbbVie" },
    // Defense
    StockInfo { symbol: "LMT", name: "Lockheed Martin" },
    StockInfo { symbol: "RTX", name: "Raytheon" },
    StockInfo { symbol: "NOC", name: "Northrop Grumman" },
    StockInfo { symbol: "BA", name: "Boeing" },
    StockInfo { symbol: "GD", name: "General Dynamics" },
    // Manufacturing & Industrial
    StockInfo { symbol: "CAT", name: "Caterpillar" },
    StockInfo { symbol: "DE", name: "John Deere" },
    StockInfo { symbol: "MMM", name: "3M" },
    StockInfo { symbol: "HON", name: "Honeywell" },
    StockInfo { symbol: "GE", name: "GE Aerospace" },
    StockInfo { symbol: "EMR", name: "Emerson Electric" },
    // Retail & Consumer
    StockInfo { symbol: "WMT", name: "Walmart" },
    StockInfo { symbol: "TGT", name: "Target" },
    StockInfo { symbol: "COST", name: "Costco" },
    StockInfo { symbol: "HD", name: "Home Depot" },
    StockInfo { symbol: "NKE", name: "Nike" },
    StockInfo { symbol: "MCD", name: "McDonald's" },
    // Energy
    StockInfo { symbol: "XOM", name: "ExxonMobil" },
    StockInfo { symbol: "CVX", name: "Chevron" },
    StockInfo { symbol: "COP", name: "ConocoPhillips" },
    // Commodities ETFs
    StockInfo { symbol: "GLD", name: "SPDR Gold Trust" },
    StockInfo { symbol: "CL=F", name: "Oil (WTI Crude)" },
    StockInfo { symbol: "USO", name: "US Oil Fund" },
    StockInfo { symbol: "SLV", name: "iShares Silver Trust" },
    StockInfo { symbol: "CPER", name: "US Copper Index Fund" },
    // UK Stocks
    StockInfo { symbol: "HSBA.L", name: "HSBC Holdings" },
    StockInfo { symbol: "BP.L", name: "BP plc" },
    StockInfo { symbol: "SHEL.L", name: "Shell plc" },
    StockInfo { symbol: "RR.L", name: "Rolls-Royce" },
    StockInfo { symbol: "AZN.L", name: "AstraZeneca" },
    // New US Stocks
    StockInfo { symbol: "SHOP", name: "Shopify" },
    StockInfo { symbol: "UBER", name: "Uber" },
    StockInfo { symbol: "COIN", name: "Coinbase" },
    StockInfo { symbol: "PLTR", name: "Palantir" },
    // UK Stocks (FTSE 100)
    StockInfo { symbol: "ISF.L", name: "iShares FTSE 100 ETF" },
    StockInfo { symbol: "GSK.L", name: "GSK" },
    StockInfo { symbol: "ULVR.L", name: "Unilever" },
    StockInfo { symbol: "DGE.L", name: "Diageo" },
    StockInfo { symbol: "VOD.L", name: "Vodafone" },
    StockInfo { symbol: "BT-A.L", name: "BT Group" },
    StockInfo { symbol: "LLOY.L", name: "Lloyds Banking Group" },
    StockInfo { symbol: "BARC.L", name: "Barclays" },
    StockInfo { symbol: "NWG.L", name: "NatWest Group" },
    StockInfo { symbol: "REL.L", name: "RELX" },
    StockInfo { symbol: "CPG.L", name: "Compass Group" },
    // Defensive US Sectors
    StockInfo { symbol: "XLP", name: "Consumer Staples ETF" },
    StockInfo { symbol: "XLU", name: "Utilities ETF" },
    StockInfo { symbol: "XLRE", name: "Real Estate ETF" },
    StockInfo { symbol: "XLK", name: "Technology ETF" },
    // Bonds
    StockInfo { symbol: "TLT", name: "iShares 20yr Treasury Bond ETF" },
    StockInfo { symbol: "AGG", name: "iShares Core US Aggregate Bond ETF" },
    StockInfo { symbol: "BND", name: "Vanguard Total Bond Market ETF" },
    StockInfo { symbol: "IBTL.L", name: "iShares UK Index-Linked Gilts ETF" },
    StockInfo { symbol: "VGLT", name: "Vanguard Long-Term Treasury ETF" },
    StockInfo { symbol: "SHY", name: "iShares 1-3yr Treasury Bond ETF" },
    // FTSE 100 — Defensives and High Value
    StockInfo { symbol: "NG.L", name: "National Grid" },
    StockInfo { symbol: "SSE.L", name: "SSE" },
    StockInfo { symbol: "CNA.L", name: "Centrica" },
    StockInfo { symbol: "IMB.L", name: "Imperial Brands" },
    StockInfo { symbol: "BATS.L", name: "British American Tobacco" },
    StockInfo { symbol: "PSON.L", name: "Pearson" },
    StockInfo { symbol: "LGEN.L", name: "Legal and General" },
    StockInfo { symbol: "III.L", name: "3i Group" },
    StockInfo { symbol: "EXPN.L", name: "Experian" },
    StockInfo { symbol: "GLEN.L", name: "Glencore" },
    StockInfo { symbol: "AAL.L", name: "Anglo American" },
    StockInfo { symbol: "ANTO.L", name: "Antofagasta" },
    StockInfo { symbol: "WPP.L", name: "WPP" },
    StockInfo { symbol: "QQ.L", name: "QinetiQ" },
    StockInfo { symbol: "MNG.L", name: "M&G" },
    StockInfo { symbol: "SMDS.L", name: "DS Smith" },
    StockInfo { symbol: "MNDI.L", name: "Mondi" },
    StockInfo { symbol: "TSCO.L", name: "Tesco" },
    StockInfo { symbol: "SBRY.L", name: "Sainsbury" },
    StockInfo { symbol: "DCC.L", name: "DCC" },
    // European Stocks
    StockInfo { symbol: "AIR.PA", name: "Airbus" },
    StockInfo { symbol: "SAF.PA", name: "Safran" },
    StockInfo { symbol: "SAN.PA", name: "Sanofi" },
    StockInfo { symbol: "OR.PA", name: "L'Oreal" },
    StockInfo { symbol: "MC.PA", name: "LVMH" },
    StockInfo { symbol: "SIE.DE", name: "Siemens" },
    StockInfo { symbol: "ALV.DE", name: "Allianz" },
    StockInfo { symbol: "BAS.DE", name: "BASF" },
    StockInfo { symbol: "MBG.DE", name: "Mercedes-Benz" },
    StockInfo { symbol: "SAP.DE", name: "SAP" },
    // US Stocks — Defensives and Staples
    StockInfo { symbol: "KO", name: "Coca-Cola" },
    StockInfo { symbol: "PEP", name: "PepsiCo" },
    StockInfo { symbol: "PG", name: "Procter & Gamble" },
    StockInfo { symbol: "GIS", name: "General Mills" },
    StockInfo { symbol: "CL", name: "Colgate-Palmolive" },
    StockInfo { symbol: "MO", name: "Altria" },
    StockInfo { symbol: "T", name: "AT&T" },
    StockInfo { symbol: "VZ", name: "Verizon" },
    StockInfo { symbol: "DUK", name: "Duke Energy" },
    StockInfo { symbol: "NEE", name: "NextEra Energy" },
    StockInfo { symbol: "SO", name: "Southern Company" },
    StockInfo { symbol: "AMT", name: "American Tower" },
    StockInfo { symbol: "BRK-B", name: "Berkshire Hathaway" },
    StockInfo { symbol: "MRK", name: "Merck" },
    StockInfo { symbol: "BMY", name: "Bristol Myers Squibb" },
    StockInfo { symbol: "CVS", name: "CVS Health" },
    StockInfo { symbol: "CI", name: "Cigna" },
    StockInfo { symbol: "MMC", name: "Marsh McLennan" },
    StockInfo { symbol: "TRV", name: "Travelers" },
    StockInfo { symbol: "AFL", name: "Aflac" },
    // Energy/Commodities ETFs
    StockInfo { symbol: "VDE", name: "Vanguard Energy ETF" },
    StockInfo { symbol: "PDBC", name: "Invesco Commodity ETF" },
    StockInfo { symbol: "DBC", name: "Invesco DB Commodity Index" },
    // Healthcare ETFs
    StockInfo { symbol: "VHT", name: "Vanguard Health Care ETF" },
    StockInfo { symbol: "IHI", name: "iShares Medical Devices ETF" },
    // Utilities ETFs
    StockInfo { symbol: "VPU", name: "Vanguard Utilities ETF" },
    StockInfo { symbol: "IDU", name: "iShares US Utilities ETF" },
    // Real Estate
    StockInfo { symbol: "VNQ", name: "Vanguard Real Estate ETF" },
];

pub const FX_LIST: &[StockInfo] = &[
    // Major pairs
    StockInfo { symbol: "EURUSD=X", name: "EUR/USD" },
    StockInfo { symbol: "GBPUSD=X", name: "GBP/USD" },
    StockInfo { symbol: "USDJPY=X", name: "USD/JPY" },
    StockInfo { symbol: "AUDUSD=X", name: "AUD/USD" },
    StockInfo { symbol: "USDCHF=X", name: "USD/CHF" },
    StockInfo { symbol: "USDCAD=X", name: "USD/CAD" },
    StockInfo { symbol: "NZDUSD=X", name: "NZD/USD" },
    StockInfo { symbol: "EURGBP=X", name: "EUR/GBP" },
    StockInfo { symbol: "USDSGD=X", name: "USD/SGD" },
    StockInfo { symbol: "USDMXN=X", name: "USD/MXN" },
    // Far East
    StockInfo { symbol: "USDCNH=X", name: "USD/CNH" },
    StockInfo { symbol: "USDKRW=X", name: "USD/KRW" },
    StockInfo { symbol: "USDHKD=X", name: "USD/HKD" },
    StockInfo { symbol: "USDTWD=X", name: "USD/TWD" },
    StockInfo { symbol: "USDTHB=X", name: "USD/THB" },
    StockInfo { symbol: "USDIDR=X", name: "USD/IDR" },
    StockInfo { symbol: "USDMYR=X", name: "USD/MYR" },
    StockInfo { symbol: "USDPHP=X", name: "USD/PHP" },
    StockInfo { symbol: "USDINR=X", name: "USD/INR" },
    // Additional FX pairs
    StockInfo { symbol: "XAUUSD=X", name: "Gold/USD" },
    StockInfo { symbol: "EURJPY=X", name: "EUR/JPY" },
    StockInfo { symbol: "GBPJPY=X", name: "GBP/JPY" },
    StockInfo { symbol: "USDNOK=X", name: "USD/NOK" },
    StockInfo { symbol: "USDSEK=X", name: "USD/SEK" },
    StockInfo { symbol: "USDZAR=X", name: "USD/ZAR" },
    StockInfo { symbol: "EURCHF=X", name: "EUR/CHF" },
    StockInfo { symbol: "GBPCHF=X", name: "GBP/CHF" },
    StockInfo { symbol: "AUDJPY=X", name: "AUD/JPY" },
    StockInfo { symbol: "NZDJPY=X", name: "NZD/JPY" },
    StockInfo { symbol: "CADUSD=X", name: "CAD/USD" },
];

// ── Yahoo Finance response types ──

#[derive(Debug, Deserialize)]
pub struct YahooResponse {
    pub chart: ChartData,
}

#[derive(Debug, Deserialize)]
pub struct ChartData {
    pub result: Option<Vec<ChartResult>>,
    pub error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChartResult {
    pub meta: MetaData,
    pub timestamp: Option<Vec<i64>>,
    pub indicators: Indicators,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetaData {
    pub symbol: String,
    pub regular_market_price: Option<f64>,
    pub previous_close: Option<f64>,
    pub regular_market_day_high: Option<f64>,
    pub regular_market_day_low: Option<f64>,
    pub regular_market_volume: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct Indicators {
    pub quote: Vec<QuoteData>,
}

#[derive(Debug, Deserialize)]
pub struct QuoteData {
    pub open: Option<Vec<Option<f64>>>,
    pub high: Option<Vec<Option<f64>>>,
    pub low: Option<Vec<Option<f64>>>,
    pub close: Option<Vec<Option<f64>>>,
    pub volume: Option<Vec<Option<u64>>>,
}

pub struct StockQuoteResult {
    pub symbol: String,
    pub price: f64,
    pub change: f64,
    pub change_percent: f64,
    pub high: f64,
    pub low: f64,
    pub volume: u64,
}

pub async fn fetch_quote(
    client: &reqwest::Client,
    symbol: &str,
) -> Result<StockQuoteResult, Box<dyn std::error::Error>> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range=1d",
        symbol
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")
        .send()
        .await?;

    let text = resp.text().await?;
    let data: YahooResponse = serde_json::from_str(&text)?;

    let result = data.chart.result
        .ok_or("No chart result")?
        .into_iter()
        .next()
        .ok_or("Empty result")?;

    let meta = &result.meta;
    let price = meta.regular_market_price.unwrap_or(0.0);
    let prev_close = meta.previous_close.unwrap_or(price);
    let change = price - prev_close;
    let change_pct = if prev_close != 0.0 {
        (change / prev_close) * 100.0
    } else {
        0.0
    };

    Ok(StockQuoteResult {
        symbol: meta.symbol.clone(),
        price,
        change,
        change_percent: change_pct,
        high: meta.regular_market_day_high.unwrap_or(0.0),
        low: meta.regular_market_day_low.unwrap_or(0.0),
        volume: meta.regular_market_volume.unwrap_or(0),
    })
}

pub async fn fetch_history(
    client: &reqwest::Client,
    symbol: &str,
    range: &str,
) -> Result<Vec<(i64, f64, Option<u64>)>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range={}",
        symbol, range
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")
        .send()
        .await?;

    let text = resp.text().await?;
    let data: YahooResponse = serde_json::from_str(&text)?;

    let result = data.chart.result
        .ok_or("No chart result")?
        .into_iter()
        .next()
        .ok_or("Empty result")?;

    let timestamps = result.timestamp.unwrap_or_default();
    let closes = result.indicators.quote.first()
        .and_then(|q| q.close.as_ref())
        .cloned()
        .unwrap_or_default();
    let volumes = result.indicators.quote.first()
        .and_then(|q| q.volume.as_ref())
        .cloned()
        .unwrap_or_default();

    let mut points = Vec::new();
    for (i, ts) in timestamps.iter().enumerate() {
        if let Some(Some(close)) = closes.get(i) {
            let vol = volumes.get(i).and_then(|v| *v);
            points.push((*ts, *close, vol));
        }
    }

    Ok(points)
}
