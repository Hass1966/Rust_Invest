import { useState, useEffect } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { TrendingUp, Shield, DollarSign, Activity, BarChart3, Gauge } from 'lucide-react'

type AssetTab = 'SPY' | 'GLD' | 'CL=F' | 'bitcoin'

interface DeepDiveData {
  asset: string
  asset_class: string
  signal: { type: string; confidence: number; probability_up: number; timestamp: string; price: number } | null
  accuracy_7d: { correct: number; total: number; accuracy: number }
  macro: {
    vix: { value: number; timestamp: string } | null
    tnx: { value: number; timestamp: string } | null
    irx: { value: number; timestamp: string } | null
    uup: { value: number; timestamp: string } | null
    sectors: Record<string, { value: number; timestamp: string; change_pct: number }>
  }
  fred: {
    fed_funds_rate: { value: number; date: string } | null
    yield_curve: { value: number; date: string } | null
  }
  fear_greed: { value: number; date: string } | null
  sentiment: { news_score: number; reddit_score: number; combined_score: number; llm_analysis: string | null; date: string } | null
  price_history: { date: string; price: number }[]
}

function MacroCard({ icon: Icon, title, value, subtitle, accent }: {
  icon: React.ElementType
  title: string
  value: string
  subtitle: string
  accent: string
}) {
  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-4 space-y-2">
      <div className="flex items-center gap-2 text-gray-400 text-xs">
        <Icon className={`w-4 h-4 ${accent}`} />
        <span>{title}</span>
      </div>
      <p className={`text-2xl font-bold ${accent}`}>{value}</p>
      <p className="text-xs text-gray-500 leading-relaxed">{subtitle}</p>
    </div>
  )
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {[...Array(6)].map((_, i) => (
        <div key={i} className="bg-[#111827] border border-[#1f2937] rounded-xl p-4 space-y-3 animate-pulse">
          <div className="h-3 w-24 bg-gray-700 rounded" />
          <div className="h-8 w-20 bg-gray-700 rounded" />
          <div className="h-3 w-full bg-gray-800 rounded" />
        </div>
      ))}
    </div>
  )
}

function formatPrice(price: number, asset: string) {
  if (asset === 'bitcoin') return price >= 1000 ? `$${(price / 1000).toFixed(1)}k` : `$${price.toFixed(0)}`
  return `$${price.toFixed(2)}`
}

function getSignalColor(type: string) {
  const t = type.toUpperCase()
  if (t === 'BUY' || t === 'STRONG BUY') return 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30'
  if (t === 'SHORT') return 'text-orange-400 bg-orange-500/15 border-orange-500/30'
  if (t === 'SELL' || t === 'STRONG SELL') return 'text-red-400 bg-red-500/15 border-red-500/30'
  return 'text-amber-400 bg-amber-500/15 border-amber-500/30'
}

function computeVerdict(data: DeepDiveData, tab: AssetTab): { text: string; color: string; border: string } {
  let bullish = 0
  let bearish = 0

  // VIX: below 20 = bullish, above 25 = bearish
  if (data.macro.vix) {
    if (data.macro.vix.value < 20) bullish++
    else if (data.macro.vix.value > 25) bearish++
  }

  // Dollar strength: UUP down = bullish for assets
  if (data.macro.uup) {
    if (data.macro.uup.value < 28) bullish++
    else if (data.macro.uup.value > 29) bearish++
  }

  // Yield curve: positive = bullish, negative = recession risk
  if (data.fred.yield_curve) {
    if (data.fred.yield_curve.value > 0) bullish++
    else bearish++
  }

  // Sentiment
  if (data.sentiment) {
    if (data.sentiment.combined_score > 0.1) bullish++
    else if (data.sentiment.combined_score < -0.1) bearish++
  }

  // Signal
  if (data.signal) {
    const t = data.signal.type.toUpperCase()
    if (t === 'BUY' || t === 'STRONG BUY') bullish++
    else if (t === 'SELL' || t === 'STRONG SELL' || t === 'SHORT') bearish++
  }

  // Fear & Greed (BTC specific)
  if (tab === 'bitcoin' && data.fear_greed) {
    if (data.fear_greed.value > 60) bullish++
    else if (data.fear_greed.value < 40) bearish++
  }

  const net = bullish - bearish
  const assetName = tab === 'SPY' ? 'equities' : tab === 'GLD' ? 'gold' : tab === 'CL=F' ? 'oil' : 'bitcoin'
  if (net >= 2) return { text: `Macro conditions appear bullish for ${assetName} — ${bullish} of ${bullish + bearish} indicators positive.`, color: 'text-emerald-400', border: 'border-emerald-500/30' }
  if (net <= -2) return { text: `Macro headwinds detected for ${assetName} — ${bearish} of ${bullish + bearish} indicators negative.`, color: 'text-red-400', border: 'border-red-500/30' }
  return { text: `Mixed macro signals for ${assetName} — no strong directional bias from indicators.`, color: 'text-amber-400', border: 'border-amber-500/30' }
}

function getSPYCards(data: DeepDiveData) {
  const sectors = data.macro.sectors || {}
  const sectorEntries = Object.entries(sectors)
  const best = sectorEntries.length > 0 ? sectorEntries.reduce((a, b) => a[1].change_pct > b[1].change_pct ? a : b) : null
  const worst = sectorEntries.length > 0 ? sectorEntries.reduce((a, b) => a[1].change_pct < b[1].change_pct ? a : b) : null

  return [
    { icon: Gauge, title: 'VIX — Fear Gauge', value: data.macro.vix ? data.macro.vix.value.toFixed(1) : 'N/A', subtitle: data.macro.vix ? (data.macro.vix.value < 20 ? 'Low volatility — market complacency' : data.macro.vix.value > 25 ? 'Elevated fear — potential opportunity or risk' : 'Normal range') : 'No data', accent: data.macro.vix && data.macro.vix.value > 25 ? 'text-red-400' : 'text-emerald-400' },
    { icon: TrendingUp, title: 'Treasury Yield (10Y)', value: data.macro.tnx ? `${data.macro.tnx.value.toFixed(2)}%` : 'N/A', subtitle: 'Higher yields compete with equities for capital', accent: 'text-cyan-400' },
    { icon: DollarSign, title: 'Fed Funds Rate', value: data.fred.fed_funds_rate ? `${data.fred.fed_funds_rate.value.toFixed(2)}%` : 'Unavailable', subtitle: data.fred.fed_funds_rate ? `As of ${data.fred.fed_funds_rate.date}` : 'FRED API key required', accent: 'text-purple-400' },
    { icon: Activity, title: 'Yield Curve (10Y-2Y)', value: data.fred.yield_curve ? `${data.fred.yield_curve.value > 0 ? '+' : ''}${data.fred.yield_curve.value.toFixed(2)}%` : 'Unavailable', subtitle: data.fred.yield_curve ? (data.fred.yield_curve.value < 0 ? 'Inverted — historically signals recession' : 'Normal — economy expanding') : 'FRED API key required', accent: data.fred.yield_curve && data.fred.yield_curve.value < 0 ? 'text-red-400' : 'text-emerald-400' },
    { icon: DollarSign, title: 'Dollar Strength (UUP)', value: data.macro.uup ? `$${data.macro.uup.value.toFixed(2)}` : 'N/A', subtitle: 'Strong dollar headwind for multinational earnings', accent: 'text-amber-400' },
    { icon: BarChart3, title: 'Sector Rotation', value: best && worst && best[0] !== worst[0] ? `${best[0]} vs ${worst[0]}` : 'N/A', subtitle: best && worst && best[0] !== worst[0] ? `Best: ${best[0]} (${best[1].change_pct > 0 ? '+' : ''}${best[1].change_pct.toFixed(1)}%) | Worst: ${worst[0]} (${worst[1].change_pct > 0 ? '+' : ''}${worst[1].change_pct.toFixed(1)}%)` : 'Sector data unavailable', accent: 'text-cyan-400' },
  ]
}

function getGoldCards(data: DeepDiveData) {
  const prices = data.price_history.map(p => p.price)
  const high90 = prices.length > 0 ? Math.max(...prices) : 0
  const low90 = prices.length > 0 ? Math.min(...prices) : 0
  const current = prices.length > 0 ? prices[prices.length - 1] : 0

  return [
    { icon: DollarSign, title: 'Dollar Strength (UUP)', value: data.macro.uup ? `$${data.macro.uup.value.toFixed(2)}` : 'N/A', subtitle: 'Weak dollar is bullish for gold (inverse correlation)', accent: 'text-amber-400' },
    { icon: TrendingUp, title: 'Real Yields (10Y)', value: data.macro.tnx ? `${data.macro.tnx.value.toFixed(2)}%` : 'N/A', subtitle: 'Lower real yields increase gold attractiveness', accent: 'text-cyan-400' },
    { icon: Gauge, title: 'VIX — Risk Sentiment', value: data.macro.vix ? data.macro.vix.value.toFixed(1) : 'N/A', subtitle: 'High VIX drives safe-haven demand for gold', accent: data.macro.vix && data.macro.vix.value > 25 ? 'text-emerald-400' : 'text-gray-400' },
    { icon: Shield, title: 'Central Bank Buying', value: 'Record Levels', subtitle: 'Global central banks adding gold reserves at fastest pace since 1960s', accent: 'text-emerald-400' },
    { icon: Activity, title: 'Gold Price Range (90d)', value: prices.length > 0 ? `$${low90.toFixed(0)} – $${high90.toFixed(0)}` : 'N/A', subtitle: prices.length > 0 ? `Current: $${current.toFixed(2)} (${((current - low90) / (high90 - low90) * 100).toFixed(0)}% of range)` : 'No data', accent: 'text-amber-400' },
    { icon: TrendingUp, title: 'Why Gold at ATH?', value: 'Multi-factor', subtitle: 'De-dollarisation, geopolitical risk, central bank buying, and real-rate expectations all support', accent: 'text-purple-400' },
  ]
}

function getOilCards(data: DeepDiveData) {
  const prices = data.price_history.map(p => p.price)
  const high90 = prices.length > 0 ? Math.max(...prices) : 0
  const low90 = prices.length > 0 ? Math.min(...prices) : 0
  const current = prices.length > 0 ? prices[prices.length - 1] : 0

  return [
    { icon: Activity, title: 'Oil & The Economy', value: 'Macro Driver', subtitle: 'Oil prices affect inflation, transport costs, and energy stocks globally. Rising oil = higher inflation pressure, potential headwind for equities. Falling oil = relief for consumers but may signal slowing demand. Particularly relevant given current geopolitical tensions.', accent: 'text-amber-400' },
    { icon: Gauge, title: 'VIX — Risk Sentiment', value: data.macro.vix ? data.macro.vix.value.toFixed(1) : 'N/A', subtitle: 'Geopolitical risk spikes push both VIX and oil higher', accent: data.macro.vix && data.macro.vix.value > 25 ? 'text-red-400' : 'text-emerald-400' },
    { icon: DollarSign, title: 'Dollar Strength (UUP)', value: data.macro.uup ? `$${data.macro.uup.value.toFixed(2)}` : 'N/A', subtitle: 'Oil is priced in USD — strong dollar weighs on crude prices', accent: 'text-amber-400' },
    { icon: BarChart3, title: 'Energy Sector (XLE)', value: data.macro.sectors?.XLE ? `$${data.macro.sectors.XLE.value.toFixed(2)}` : 'N/A', subtitle: data.macro.sectors?.XLE ? `Change: ${data.macro.sectors.XLE.change_pct > 0 ? '+' : ''}${data.macro.sectors.XLE.change_pct.toFixed(1)}%` : 'Energy equities track crude closely', accent: 'text-cyan-400' },
    { icon: Activity, title: 'Oil Price Range (90d)', value: prices.length > 0 ? `$${low90.toFixed(0)} – $${high90.toFixed(0)}` : 'N/A', subtitle: prices.length > 0 ? `Current: $${current.toFixed(2)} (${((current - low90) / (high90 - low90) * 100).toFixed(0)}% of range)` : 'No data', accent: 'text-amber-400' },
    { icon: TrendingUp, title: 'Treasury Yield (10Y)', value: data.macro.tnx ? `${data.macro.tnx.value.toFixed(2)}%` : 'N/A', subtitle: 'Rising yields and rising oil both signal inflation expectations', accent: 'text-cyan-400' },
  ]
}

function getBitcoinCards(data: DeepDiveData) {
  return [
    { icon: Gauge, title: 'Fear & Greed Index', value: data.fear_greed ? `${data.fear_greed.value.toFixed(0)}` : 'N/A', subtitle: data.fear_greed ? (data.fear_greed.value > 75 ? 'Extreme greed — historically precedes corrections' : data.fear_greed.value > 55 ? 'Greed — momentum is positive' : data.fear_greed.value > 40 ? 'Neutral zone' : 'Fear — contrarian buy signal historically') : 'No data', accent: data.fear_greed && data.fear_greed.value > 60 ? 'text-emerald-400' : data.fear_greed && data.fear_greed.value < 40 ? 'text-red-400' : 'text-amber-400' },
    { icon: DollarSign, title: 'Dollar Strength (UUP)', value: data.macro.uup ? `$${data.macro.uup.value.toFixed(2)}` : 'N/A', subtitle: 'Weak dollar historically bullish for BTC', accent: 'text-amber-400' },
    { icon: Activity, title: 'BTC-SPY Correlation', value: '~0.4–0.6', subtitle: 'Bitcoin increasingly correlated with risk assets during macro stress', accent: 'text-cyan-400' },
    { icon: TrendingUp, title: '200-Week Moving Avg', value: 'Support Zone', subtitle: 'BTC has never closed below the 200-week MA on a sustained basis', accent: 'text-purple-400' },
    { icon: BarChart3, title: 'Funding Rates', value: 'Varies', subtitle: 'Positive funding = longs pay shorts. Extreme values signal overleveraged markets', accent: 'text-cyan-400' },
    { icon: Shield, title: 'Honest Assessment', value: 'High Risk', subtitle: 'Crypto is volatile. Our models have limited crypto history. Size positions accordingly.', accent: 'text-red-400' },
  ]
}

export default function DeepDive() {
  const [tab, setTab] = useState<AssetTab>('SPY')
  const [data, setData] = useState<DeepDiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`/api/v1/deep-dive/${tab}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [tab])

  const tabLabels: Record<AssetTab, string> = { SPY: 'S&P 500', GLD: 'Gold', 'CL=F': 'Oil', bitcoin: 'Bitcoin' }

  const cards = data ? (tab === 'SPY' ? getSPYCards(data) : tab === 'GLD' ? getGoldCards(data) : tab === 'CL=F' ? getOilCards(data) : getBitcoinCards(data)) : []
  const verdict = data ? computeVerdict(data, tab) : null

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">Deep Dive</h2>
        <p className="text-sm text-gray-500 mt-1">
          Macro context for key assets — powered by 6 Rust ML models analysing 123 features.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-2">
        {(['SPY', 'GLD', 'CL=F', 'bitcoin'] as AssetTab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
              tab === t
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
            }`}
          >
            {tabLabels[t]}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
          Failed to load data: {error}
        </div>
      )}

      {loading ? (
        <SkeletonGrid />
      ) : data ? (
        <>
          {/* Current Signal */}
          <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-medium text-gray-400">Current Signal — {tabLabels[tab]}</h3>
            {data.signal ? (
              <div className="flex flex-wrap items-center gap-4">
                <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${getSignalColor(data.signal.type)}`}>
                  {data.signal.type.toUpperCase()}
                </span>
                <span className="text-gray-300 text-sm">
                  Confidence: <span className="text-white font-medium">{data.signal.confidence.toFixed(1)}%</span>
                </span>
                <span className="text-gray-300 text-sm">
                  P(up): <span className="text-white font-medium">{data.signal.probability_up.toFixed(1)}%</span>
                </span>
                <span className="text-gray-300 text-sm">
                  Price: <span className="text-white font-medium">{formatPrice(data.signal.price, tab)}</span>
                </span>
                {data.accuracy_7d.total > 0 && (
                  <span className="text-gray-300 text-sm">
                    7d accuracy: <span className={`font-medium ${data.accuracy_7d.accuracy >= 60 ? 'text-emerald-400' : data.accuracy_7d.accuracy >= 40 ? 'text-amber-400' : 'text-red-400'}`}>
                      {data.accuracy_7d.accuracy.toFixed(0)}% ({data.accuracy_7d.correct}/{data.accuracy_7d.total})
                    </span>
                  </span>
                )}
              </div>
            ) : (
              <div>
                {tab === 'CL=F' ? (
                  <div className="flex items-center gap-3">
                    <span className="px-3 py-1 rounded-full text-sm font-semibold border text-cyan-400 bg-cyan-500/15 border-cyan-500/30">MONITORING</span>
                    <span className="text-gray-500 text-sm">Tracking started today — signal data will appear after the next pipeline run.</span>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">No signal data available. Signals are generated during the daily pipeline run.</p>
                )}
              </div>
            )}

            {/* Sentiment summary */}
            {data.sentiment && (
              <div className="pt-2 border-t border-[#1f2937]">
                <p className="text-xs text-gray-500">
                  Sentiment ({data.sentiment.date}): News {data.sentiment.news_score > 0 ? '+' : ''}{data.sentiment.news_score.toFixed(2)} | Reddit {data.sentiment.reddit_score > 0 ? '+' : ''}{data.sentiment.reddit_score.toFixed(2)} | Combined {data.sentiment.combined_score > 0 ? '+' : ''}{data.sentiment.combined_score.toFixed(2)}
                </p>
                {data.sentiment.llm_analysis && (
                  <p className="text-xs text-gray-400 mt-1 italic line-clamp-2">{data.sentiment.llm_analysis}</p>
                )}
              </div>
            )}
          </div>

          {/* Macro Cards */}
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-3">Macro Context</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {cards.map((c, i) => (
                <MacroCard key={i} icon={c.icon} title={c.title} value={c.value} subtitle={c.subtitle} accent={c.accent} />
              ))}
            </div>
          </div>

          {/* Macro Verdict */}
          {verdict && (
            <div className={`bg-[#111827] border ${verdict.border} rounded-xl p-4`}>
              <h3 className="text-sm font-medium text-gray-400 mb-2">Macro Verdict</h3>
              <p className={`text-sm ${verdict.color}`}>{verdict.text}</p>
            </div>
          )}

          {/* 90-Day Price Chart */}
          {data.price_history.length > 0 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-5">
              <h3 className="text-sm font-medium text-gray-400 mb-4">90-Day Price History — {tabLabels[tab]}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={data.price_history}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickFormatter={d => d.slice(5)}
                    interval="preserveStartEnd"
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickFormatter={v => tab === 'bitcoin' ? `$${(v / 1000).toFixed(0)}k` : `$${v}`}
                    domain={['auto', 'auto']}
                    axisLine={false}
                    tickLine={false}
                    width={60}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px', fontSize: '12px' }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={(value: number | undefined) => [value != null ? formatPrice(value, tab) : '', 'Price']}
                  />
                  <Area
                    type="monotone"
                    dataKey="price"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    fill="url(#priceGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Footer */}
          <div className="text-xs text-gray-600 space-y-1 pb-4">
            <p>Data: Yahoo Finance (prices, VIX, yields), FRED (Fed Funds, yield curve), Alternative.me (Fear & Greed), Serper/NewsAPI/Reddit (sentiment).</p>
            <p>Not financial advice. Past performance does not guarantee future results.</p>
          </div>
        </>
      ) : null}
    </div>
  )
}
