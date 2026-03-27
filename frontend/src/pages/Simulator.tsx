import { useEffect, useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'

// ─── Types ───

interface PricePoint { date: string; price: number }
interface SignalPoint { date: string; signal: string; price: number; was_correct: boolean | null; outcome_price: number | null }
interface SimulatorData {
  price_history: Record<string, PricePoint[]>
  signal_history: Record<string, SignalPoint[]>
}

// ─── Config ───

const LIVE_START = '2026-03-15'
const BH_ASSETS: { asset: string; amount: number }[] = [
  { asset: 'AAPL', amount: 8000 },
  { asset: 'MSFT', amount: 8000 },
  { asset: 'GOOGL', amount: 6000 },
  { asset: 'JPM', amount: 5000 },
  { asset: 'HSBA.L', amount: 6000 },
  { asset: 'AZN.L', amount: 6000 },
  { asset: 'XOM', amount: 5000 },
  { asset: 'GLD', amount: 7000 },
  { asset: 'bitcoin', amount: 5000 },
  { asset: 'ethereum', amount: 4000 },
]
const CASH_AMOUNT = 4000
const BH_TOTAL = 60000
const AS_TOTAL = 40000
const AS_PER_ASSET = 4000

// ─── Helpers ───

function fmtGBP(n: number): string { return '\u00a3' + Math.round(n).toLocaleString() }
function fmtPct(n: number): string { return (n >= 0 ? '+' : '') + n.toFixed(2) + '%' }

function buildPriceMap(points: PricePoint[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const p of points) m.set(p.date, p.price)
  return m
}

function getPrice(priceMap: Map<string, number>, date: string): number | null {
  const p = priceMap.get(date)
  if (p !== undefined) return p
  // Find nearest earlier date
  let best: number | null = null
  let bestDate = ''
  for (const [d, v] of priceMap) {
    if (d <= date && d > bestDate) { best = v; bestDate = d }
  }
  return best
}

function getAllDates(priceHistory: Record<string, PricePoint[]>, fromDate?: string): string[] {
  const dateSet = new Set<string>()
  for (const points of Object.values(priceHistory)) {
    for (const p of points) {
      if (!fromDate || p.date >= fromDate) dateSet.add(p.date)
    }
  }
  return Array.from(dateSet).sort()
}

// ─── Simulation Logic ───

interface SimResult {
  chartData: { date: string; buyHold: number; alphaSignal: number; spy: number }[]
  bhBreakdown: { asset: string; invested: number; currentValue: number; returnPct: number }[]
  bhTotal: number
  bhReturn: number
  asTotal: number
  asReturn: number
  spyTotal: number
  spyReturn: number
  asTotalSignals: number
  asCorrect: number
  asIncorrect: number
  asBest: { asset: string; returnPct: number } | null
  asWorst: { asset: string; returnPct: number } | null
}

function runSimulation(data: SimulatorData, fromDate?: string): SimResult | null {
  const dates = getAllDates(data.price_history, fromDate)
  if (dates.length < 2) return null

  const priceMaps: Record<string, Map<string, number>> = {}
  for (const [k, v] of Object.entries(data.price_history)) {
    priceMaps[k] = buildPriceMap(v)
  }

  const startDate = dates[0]

  // ── Buy & Hold ──
  const bhShares: Record<string, number> = {}
  for (const { asset, amount } of BH_ASSETS) {
    const startPrice = getPrice(priceMaps[asset], startDate)
    bhShares[asset] = startPrice ? amount / startPrice : 0
  }

  // ── Alpha Signal ──
  const asState: Record<string, { shares: number; cash: number; invested: boolean }> = {}
  for (const { asset } of BH_ASSETS) {
    const startPrice = getPrice(priceMaps[asset], startDate)
    // Start fully invested (BUY default)
    asState[asset] = {
      shares: startPrice ? AS_PER_ASSET / startPrice : 0,
      cash: 0,
      invested: true,
    }
  }

  // Pre-process signals by asset and date
  const signalsByAssetDate: Record<string, Map<string, string>> = {}
  for (const [asset, signals] of Object.entries(data.signal_history)) {
    const m = new Map<string, string>()
    for (const s of signals) {
      if (!fromDate || s.date >= fromDate) m.set(s.date, s.signal)
    }
    signalsByAssetDate[asset] = m
  }

  // Count signals
  let totalSignals = 0
  let correctSignals = 0
  let incorrectSignals = 0
  for (const signals of Object.values(data.signal_history)) {
    for (const s of signals) {
      if (fromDate && s.date < fromDate) continue
      totalSignals++
      if (s.was_correct === true) correctSignals++
      else if (s.was_correct === false) incorrectSignals++
    }
  }

  const chartData: SimResult['chartData'] = []

  for (const date of dates) {
    // Process Alpha Signal trades
    for (const { asset } of BH_ASSETS) {
      const signal = signalsByAssetDate[asset]?.get(date)
      if (!signal) continue
      const price = getPrice(priceMaps[asset], date)
      if (!price) continue
      const st = asState[asset]

      if (signal === 'BUY' && !st.invested) {
        st.shares = st.cash / price
        st.cash = 0
        st.invested = true
      } else if (signal === 'SELL' && st.invested) {
        st.cash = st.shares * price
        st.shares = 0
        st.invested = false
      }
    }

    // Calculate portfolio values
    let bhValue = CASH_AMOUNT
    for (const { asset } of BH_ASSETS) {
      const price = getPrice(priceMaps[asset], date)
      bhValue += price ? bhShares[asset] * price : 0
    }

    let asValue = 0
    for (const { asset } of BH_ASSETS) {
      const price = getPrice(priceMaps[asset], date)
      const st = asState[asset]
      asValue += st.invested && price ? st.shares * price : st.cash
    }

    const spyPrice = getPrice(priceMaps['SPY'], date)
    const spyStartPrice = getPrice(priceMaps['SPY'], startDate)
    const spyValue = spyStartPrice && spyPrice ? 100000 * (spyPrice / spyStartPrice) : 100000

    chartData.push({ date, buyHold: Math.round(bhValue), alphaSignal: Math.round(asValue), spy: Math.round(spyValue) })
  }

  // Final values
  const lastDate = dates[dates.length - 1]
  const bhBreakdown: SimResult['bhBreakdown'] = BH_ASSETS.map(({ asset, amount }) => {
    const price = getPrice(priceMaps[asset], lastDate)
    const currentValue = price ? bhShares[asset] * price : 0
    return { asset, invested: amount, currentValue: Math.round(currentValue), returnPct: amount > 0 ? ((currentValue - amount) / amount) * 100 : 0 }
  })

  const last = chartData[chartData.length - 1]
  const bhTotal = last.buyHold
  const asTotal = last.alphaSignal
  const spyTotal = last.spy

  // Alpha Signal per-asset performance
  const asPerAsset = BH_ASSETS.map(({ asset }) => {
    const price = getPrice(priceMaps[asset], lastDate)
    const st = asState[asset]
    const value = st.invested && price ? st.shares * price : st.cash
    return { asset, returnPct: ((value - AS_PER_ASSET) / AS_PER_ASSET) * 100 }
  })
  const asBest = asPerAsset.length ? asPerAsset.reduce((a, b) => a.returnPct > b.returnPct ? a : b) : null
  const asWorst = asPerAsset.length ? asPerAsset.reduce((a, b) => a.returnPct < b.returnPct ? a : b) : null

  return {
    chartData,
    bhBreakdown,
    bhTotal,
    bhReturn: ((bhTotal - BH_TOTAL) / BH_TOTAL) * 100,
    asTotal,
    asReturn: ((asTotal - AS_TOTAL) / AS_TOTAL) * 100,
    spyTotal,
    spyReturn: ((spyTotal - 100000) / 100000) * 100,
    asTotalSignals: totalSignals,
    asCorrect: correctSignals,
    asIncorrect: incorrectSignals,
    asBest,
    asWorst,
  }
}

// ─── Main Component ───

export default function Simulator() {
  const [data, setData] = useState<SimulatorData | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<'backtest' | 'live'>('live')

  useEffect(() => {
    fetch('/api/v1/simulator/data')
      .then(r => r.json())
      .then(d => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [])

  const backtestResult = useMemo(() => data ? runSimulation(data) : null, [data])
  const liveResult = useMemo(() => data ? runSimulation(data, LIVE_START) : null, [data])

  const result = tab === 'backtest' ? backtestResult : liveResult
  const daysSinceLive = Math.floor((Date.now() - new Date(LIVE_START).getTime()) / 86400000)

  if (loading) return <div className="text-gray-500 p-8 text-center">Loading simulator...</div>
  if (!data) return <div className="text-gray-500 p-8 text-center">Simulator data unavailable. Ensure the backend has historical price data.</div>

  return (
    <div className="space-y-6">
      {/* Hero explanation */}
      <div className="bg-gradient-to-r from-[#0f1729] to-[#111827] rounded-2xl border border-[#1f2937] p-6 sm:p-8">
        <h2 className="text-2xl font-bold text-white mb-3">Investment Simulator</h2>
        <p className="text-sm text-gray-400 leading-relaxed max-w-3xl">
          We simulated investing {fmtGBP(100000)} the classic way &mdash; 60% in a diversified buy-and-hold
          portfolio across 10 assets, 40% following Alpha Signal recommendations on those same assets.
          The only difference: buy-and-hold buys on day one and never touches it. Alpha Signal actively
          trades based on signals. Which did better?
        </p>
      </div>

      {/* Tab selector */}
      <div className="flex gap-2">
        <button
          onClick={() => setTab('backtest')}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
            tab === 'backtest'
              ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
              : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
          }`}
        >
          5-Year Backtest
        </button>
        <button
          onClick={() => setTab('live')}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer flex items-center gap-2 ${
            tab === 'live'
              ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
              : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
          }`}
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          Live Tracking
        </button>
      </div>

      {/* Warning banners */}
      {tab === 'backtest' && (
        <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg px-4 py-3 text-sm text-amber-300">
          Simulated &mdash; these models were trained on this historical data. Results are optimistic and not a reliable indicator of future performance.
        </div>
      )}
      {tab === 'live' && (
        <div className="bg-green-900/20 border border-green-500/30 rounded-lg px-4 py-3 text-sm text-green-300 flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          Live &mdash; Day {daysSinceLive} of live tracking (since 15 March 2026)
        </div>
      )}

      {!result || result.chartData.length < 2 ? (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-8 text-center text-gray-500">
          Not enough data for this scenario yet. {tab === 'live' ? 'Live tracking data accumulates daily from 15 March 2026.' : 'Historical price data needed.'}
        </div>
      ) : (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <SummaryCard
              label={`Buy & Hold ${fmtGBP(BH_TOTAL)}`}
              value={fmtGBP(result.bhTotal)}
              returnPct={result.bhReturn}
              borderColor="border-[#1f2937]"
              valueColor="text-white"
            />
            <SummaryCard
              label={`Alpha Signal ${fmtGBP(AS_TOTAL)}`}
              value={fmtGBP(result.asTotal)}
              returnPct={result.asReturn}
              borderColor="border-cyan-500/30"
              valueColor="text-cyan-400"
            />
            <SummaryCard
              label={`S&P 500 Benchmark ${fmtGBP(100000)}`}
              value={fmtGBP(result.spyTotal)}
              returnPct={result.spyReturn}
              borderColor="border-gray-700"
              valueColor="text-gray-300"
            />
          </div>

          {/* Main chart */}
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Portfolio Value Over Time</h3>
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={result.chartData} margin={{ left: 10, right: 10, top: 4, bottom: 0 }}>
                <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `\u00a3${(v / 1000).toFixed(0)}k`} width={55} />
                <Tooltip
                  contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                  labelStyle={{ color: '#9ca3af' }}
                  formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '']}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                <Line type="monotone" dataKey="alphaSignal" name="Alpha Signal" stroke="#06b6d4" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="buyHold" name="Buy & Hold" stroke="#e5e7eb" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="spy" name="S&P 500" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Buy & Hold breakdown */}
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Buy & Hold Breakdown</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-[#1f2937]">
                    <th className="text-left py-2 px-2">Asset</th>
                    <th className="text-right py-2 px-2">Invested</th>
                    <th className="text-right py-2 px-2">Current Value</th>
                    <th className="text-right py-2 px-2">Return %</th>
                  </tr>
                </thead>
                <tbody>
                  {result.bhBreakdown.map(a => (
                    <tr key={a.asset} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                      <td className="py-1.5 px-2 text-gray-300 font-medium">{a.asset}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{fmtGBP(a.invested)}</td>
                      <td className="py-1.5 px-2 text-right text-gray-300">{fmtGBP(a.currentValue)}</td>
                      <td className={`py-1.5 px-2 text-right font-medium ${a.returnPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {fmtPct(a.returnPct)}
                      </td>
                    </tr>
                  ))}
                  <tr className="border-b border-[#1f2937]/50">
                    <td className="py-1.5 px-2 text-gray-500">Cash</td>
                    <td className="py-1.5 px-2 text-right text-gray-500">{fmtGBP(CASH_AMOUNT)}</td>
                    <td className="py-1.5 px-2 text-right text-gray-500">{fmtGBP(CASH_AMOUNT)}</td>
                    <td className="py-1.5 px-2 text-right text-gray-600">0.00%</td>
                  </tr>
                </tbody>
                <tfoot>
                  <tr className="border-t border-[#374151] font-semibold">
                    <td className="py-2 px-2 text-white">Total</td>
                    <td className="py-2 px-2 text-right text-gray-300">{fmtGBP(BH_TOTAL)}</td>
                    <td className="py-2 px-2 text-right text-white">{fmtGBP(result.bhTotal)}</td>
                    <td className={`py-2 px-2 text-right ${result.bhReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {fmtPct(result.bhReturn)}
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>

          {/* Alpha Signal breakdown */}
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
            <h3 className="text-sm font-medium text-cyan-400 mb-3">Alpha Signal Breakdown</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 mb-1">Total Signals</div>
                <div className="text-xl font-bold text-white">{result.asTotalSignals}</div>
              </div>
              <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 mb-1">Correct</div>
                <div className="text-xl font-bold text-green-400">{result.asCorrect}</div>
              </div>
              <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 mb-1">Incorrect</div>
                <div className="text-xl font-bold text-red-400">{result.asIncorrect}</div>
              </div>
              <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 mb-1">Portfolio Value</div>
                <div className="text-xl font-bold text-cyan-400">{fmtGBP(result.asTotal)}</div>
              </div>
            </div>
            {(result.asBest || result.asWorst) && (
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                {result.asBest && (
                  <div className="bg-green-500/5 border border-green-500/20 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <div className="text-xs text-gray-500">Best Performer</div>
                      <div className="text-white font-medium">{result.asBest.asset}</div>
                    </div>
                    <div className="text-green-400 font-bold">{fmtPct(result.asBest.returnPct)}</div>
                  </div>
                )}
                {result.asWorst && (
                  <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <div className="text-xs text-gray-500">Worst Performer</div>
                      <div className="text-white font-medium">{result.asWorst.asset}</div>
                    </div>
                    <div className="text-red-400 font-bold">{fmtPct(result.asWorst.returnPct)}</div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Disclaimer */}
          <div className="bg-amber-900/10 border border-amber-500/20 rounded-xl p-5">
            <p className="text-sm text-amber-300/90 leading-relaxed">
              This is a simulation. No real money was invested.{' '}
              {tab === 'backtest'
                ? 'Scenario 1 uses historical data the models were trained on \u2014 results are optimistic.'
                : 'Scenario 2 uses real signals from 15 March 2026.'}{' '}
              This is not financial advice. Past performance does not guarantee future results.
            </p>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Sub-components ───

function SummaryCard({ label, value, returnPct, borderColor, valueColor }: {
  label: string; value: string; returnPct: number; borderColor: string; valueColor: string
}) {
  return (
    <div className={`bg-[#111827] rounded-xl border ${borderColor} p-5`}>
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-3xl font-bold ${valueColor}`}>{value}</div>
      <div className={`text-sm mt-1 ${returnPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {fmtPct(returnPct)}
      </div>
    </div>
  )
}
