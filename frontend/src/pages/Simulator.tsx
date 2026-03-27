import { useEffect, useState, useMemo, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import { Loader2, Plus, Trash2 } from 'lucide-react'
import { fetchSimulation } from '../lib/api'
import type { SimResult as WhatIfResult } from '../lib/types'

// ─── Types ───

interface PricePoint { date: string; price: number }
interface SignalPoint { date: string; signal: string; price: number; was_correct: boolean | null; outcome_price: number | null }
interface SimulatorData {
  price_history: Record<string, PricePoint[]>
  signal_history: Record<string, SignalPoint[]>
}

// ─── Config ───

const LIVE_START = '2026-03-15'
const AVAILABLE_ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'HSBA.L', 'AZN.L', 'XOM', 'GLD', 'bitcoin', 'ethereum']
const DEFAULT_BH: { asset: string; amount: number }[] = [
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
const DEFAULT_CASH = 4000
const BH_TOTAL = 60000
const AS_TOTAL = 40000
const MAX_CUSTOM = 5

interface Allocation { asset: string; pct: number }

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
  cashAmount: number
  bhSharpe: number
  asSharpe: number
  bhMaxDrawdown: number
  asMaxDrawdown: number
}

function runSimulation(
  data: SimulatorData,
  bhAssets: { asset: string; amount: number }[],
  cashAmount: number,
  fromDate?: string,
): SimResult | null {
  const dates = getAllDates(data.price_history, fromDate)
  if (dates.length < 2) return null

  const priceMaps: Record<string, Map<string, number>> = {}
  for (const [k, v] of Object.entries(data.price_history)) {
    priceMaps[k] = buildPriceMap(v)
  }

  const startDate = dates[0]
  const bhTotalInvested = bhAssets.reduce((s, a) => s + a.amount, 0) + cashAmount
  const asPerAssetAmount = AS_TOTAL / (bhAssets.length || 1)

  // ── Buy & Hold ──
  const bhShares: Record<string, number> = {}
  for (const { asset, amount } of bhAssets) {
    const startPrice = getPrice(priceMaps[asset], startDate)
    bhShares[asset] = startPrice ? amount / startPrice : 0
  }

  // ── Alpha Signal ──
  const asState: Record<string, { shares: number; cash: number; invested: boolean }> = {}
  for (const { asset } of bhAssets) {
    const startPrice = getPrice(priceMaps[asset], startDate)
    asState[asset] = {
      shares: startPrice ? asPerAssetAmount / startPrice : 0,
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

  // Count signals for selected assets only
  let totalSignals = 0
  let correctSignals = 0
  let incorrectSignals = 0
  const selectedAssets = new Set(bhAssets.map(a => a.asset))
  for (const [asset, signals] of Object.entries(data.signal_history)) {
    if (!selectedAssets.has(asset)) continue
    for (const s of signals) {
      if (fromDate && s.date < fromDate) continue
      totalSignals++
      if (s.was_correct === true) correctSignals++
      else if (s.was_correct === false) incorrectSignals++
    }
  }

  const chartData: SimResult['chartData'] = []

  for (const date of dates) {
    for (const { asset } of bhAssets) {
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

    let bhValue = cashAmount
    for (const { asset } of bhAssets) {
      const price = getPrice(priceMaps[asset], date)
      bhValue += price ? bhShares[asset] * price : 0
    }

    let asValue = 0
    for (const { asset } of bhAssets) {
      const price = getPrice(priceMaps[asset], date)
      const st = asState[asset]
      asValue += st.invested && price ? st.shares * price : st.cash
    }

    const spyPrice = getPrice(priceMaps['SPY'], date)
    const spyStartPrice = getPrice(priceMaps['SPY'], startDate)
    const spyValue = spyStartPrice && spyPrice ? 100000 * (spyPrice / spyStartPrice) : 100000

    chartData.push({ date, buyHold: Math.round(bhValue), alphaSignal: Math.round(asValue), spy: Math.round(spyValue) })
  }

  const lastDate = dates[dates.length - 1]
  const bhBreakdown: SimResult['bhBreakdown'] = bhAssets.map(({ asset, amount }) => {
    const price = getPrice(priceMaps[asset], lastDate)
    const currentValue = price ? bhShares[asset] * price : 0
    return { asset, invested: amount, currentValue: Math.round(currentValue), returnPct: amount > 0 ? ((currentValue - amount) / amount) * 100 : 0 }
  })

  const last = chartData[chartData.length - 1]
  const bhTotal = last.buyHold
  const asTotal = last.alphaSignal
  const spyTotal = last.spy

  const asPerAsset = bhAssets.map(({ asset }) => {
    const price = getPrice(priceMaps[asset], lastDate)
    const st = asState[asset]
    const value = st.invested && price ? st.shares * price : st.cash
    return { asset, returnPct: ((value - asPerAssetAmount) / asPerAssetAmount) * 100 }
  })
  const asBest = asPerAsset.length ? asPerAsset.reduce((a, b) => a.returnPct > b.returnPct ? a : b) : null
  const asWorst = asPerAsset.length ? asPerAsset.reduce((a, b) => a.returnPct < b.returnPct ? a : b) : null

  // Compute daily returns, Sharpe ratios and max drawdowns
  const RISK_FREE_DAILY = 0.045 / 252 // 4.5% annual risk-free rate

  function dailyReturns(series: number[]): number[] {
    const r: number[] = []
    for (let i = 1; i < series.length; i++) {
      r.push(series[i - 1] > 0 ? (series[i] - series[i - 1]) / series[i - 1] : 0)
    }
    return r
  }

  function sharpeRatio(returns: number[]): number {
    if (returns.length < 2) return 0
    const mean = returns.reduce((s, v) => s + v, 0) / returns.length
    const variance = returns.reduce((s, v) => s + (v - mean) ** 2, 0) / returns.length
    const std = Math.sqrt(variance)
    if (std === 0) return 0
    return ((mean - RISK_FREE_DAILY) / std) * Math.sqrt(252) // annualised
  }

  function maxDrawdown(series: number[]): number {
    let peak = series[0] || 0
    let maxDd = 0
    for (const v of series) {
      if (v > peak) peak = v
      const dd = peak > 0 ? (peak - v) / peak : 0
      if (dd > maxDd) maxDd = dd
    }
    return maxDd * 100 // as percentage
  }

  const bhValues = chartData.map(d => d.buyHold)
  const asValues = chartData.map(d => d.alphaSignal)

  return {
    chartData,
    bhBreakdown,
    bhTotal,
    bhReturn: bhTotalInvested > 0 ? ((bhTotal - bhTotalInvested) / bhTotalInvested) * 100 : 0,
    asTotal,
    asReturn: AS_TOTAL > 0 ? ((asTotal - AS_TOTAL) / AS_TOTAL) * 100 : 0,
    spyTotal,
    spyReturn: ((spyTotal - 100000) / 100000) * 100,
    asTotalSignals: totalSignals,
    asCorrect: correctSignals,
    asIncorrect: incorrectSignals,
    asBest,
    asWorst,
    cashAmount,
    bhSharpe: sharpeRatio(dailyReturns(bhValues)),
    asSharpe: sharpeRatio(dailyReturns(asValues)),
    bhMaxDrawdown: maxDrawdown(bhValues),
    asMaxDrawdown: maxDrawdown(asValues),
  }
}

// ═══════════════════════════════════════
// Main Component
// ═══════════════════════════════════════

type TopTab = 'backtest' | 'live' | 'whatif'

export default function Simulator() {
  const [data, setData] = useState<SimulatorData | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<TopTab>('live')

  // Custom portfolio state
  const [useCustom, setUseCustom] = useState(false)
  const [allocations, setAllocations] = useState<Allocation[]>([
    { asset: 'AAPL', pct: 30 },
    { asset: 'MSFT', pct: 30 },
    { asset: 'GLD', pct: 20 },
    { asset: 'bitcoin', pct: 20 },
  ])

  useEffect(() => {
    fetch('/api/v1/simulator/data')
      .then(r => r.json())
      .then(d => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [])

  // Compute BH assets from allocations
  const customBhAssets = useMemo(() => {
    if (!useCustom) return DEFAULT_BH
    return allocations.filter(a => a.asset && a.pct > 0).map(a => ({
      asset: a.asset,
      amount: Math.round((a.pct / 100) * BH_TOTAL),
    }))
  }, [useCustom, allocations])

  const customCash = useMemo(() => {
    if (!useCustom) return DEFAULT_CASH
    const totalPct = allocations.reduce((s, a) => s + a.pct, 0)
    return Math.round(((100 - totalPct) / 100) * BH_TOTAL)
  }, [useCustom, allocations])

  const backtestResult = useMemo(() => data ? runSimulation(data, customBhAssets, customCash) : null, [data, customBhAssets, customCash])
  const liveResult = useMemo(() => data ? runSimulation(data, customBhAssets, customCash, LIVE_START) : null, [data, customBhAssets, customCash])

  const result = tab === 'backtest' ? backtestResult : tab === 'live' ? liveResult : null
  const daysSinceLive = Math.floor((Date.now() - new Date(LIVE_START).getTime()) / 86400000)

  const totalPct = allocations.reduce((s, a) => s + a.pct, 0)
  const pctValid = totalPct > 0 && totalPct <= 100
  const usedAssets = new Set(allocations.map(a => a.asset))

  // Allocation handlers
  const addAllocation = () => {
    if (allocations.length >= MAX_CUSTOM) return
    const remaining = AVAILABLE_ASSETS.filter(a => !usedAssets.has(a))
    if (remaining.length === 0) return
    setAllocations([...allocations, { asset: remaining[0], pct: 0 }])
  }

  const removeAllocation = (idx: number) => {
    setAllocations(allocations.filter((_, i) => i !== idx))
  }

  const updateAllocation = (idx: number, field: 'asset' | 'pct', value: string | number) => {
    setAllocations(allocations.map((a, i) => i === idx ? { ...a, [field]: value } : a))
  }

  if (loading) return <div className="text-gray-500 p-8 text-center">Loading simulator...</div>
  if (!data) return <div className="text-gray-500 p-8 text-center">Simulator data unavailable. Ensure the backend has historical price data.</div>

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="bg-gradient-to-r from-[#0f1729] to-[#111827] rounded-2xl border border-[#1f2937] p-6 sm:p-8">
        <h2 className="text-2xl font-bold text-white mb-3">Investment Simulator</h2>
        <p className="text-sm text-gray-400 leading-relaxed max-w-3xl">
          Compare buy-and-hold vs Alpha Signal recommendations, or run a what-if simulation
          to see what your capital would be worth if you had followed every signal.
        </p>
      </div>

      {/* Tab selector */}
      <div className="flex gap-2 flex-wrap">
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
        <button
          onClick={() => setTab('whatif')}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
            tab === 'whatif'
              ? 'bg-purple-500/15 text-purple-400 border border-purple-500/30'
              : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
          }`}
        >
          What-If Simulator
        </button>
      </div>

      {/* What-If tab */}
      {tab === 'whatif' && <WhatIfSimulator />}

      {/* Backtest / Live tabs */}
      {(tab === 'backtest' || tab === 'live') && (
        <>
          {/* Warning banners */}
          {tab === 'backtest' && (
            <div className="bg-yellow-900/30 border-2 border-yellow-500/50 rounded-lg px-5 py-4 text-sm">
              <div className="flex items-start gap-3">
                <span className="text-xl leading-none flex-shrink-0">{'\u26A0\uFE0F'}</span>
                <div>
                  <div className="font-semibold text-yellow-300 mb-1">Lookahead bias warning</div>
                  <p className="text-yellow-200/80">These models were trained on this historical data. The backtest is illustrative only and significantly overstates real-world performance.</p>
                </div>
              </div>
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

          {/* Custom Portfolio Builder */}
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-400">Portfolio Allocation</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setUseCustom(false)}
                  className={`px-3 py-1.5 rounded text-xs font-medium cursor-pointer transition-colors ${
                    !useCustom ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30' : 'text-gray-400 bg-[#0a0e17] border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  Default (10 assets)
                </button>
                <button
                  onClick={() => setUseCustom(true)}
                  className={`px-3 py-1.5 rounded text-xs font-medium cursor-pointer transition-colors ${
                    useCustom ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30' : 'text-gray-400 bg-[#0a0e17] border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  Custom
                </button>
              </div>
            </div>

            {useCustom && (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  Pick up to {MAX_CUSTOM} assets and set the % of {fmtGBP(BH_TOTAL)} to allocate. Remaining goes to cash.
                </p>

                {allocations.map((alloc, idx) => {
                  return (
                    <div key={idx} className="flex items-center gap-2">
                      <select
                        value={alloc.asset}
                        onChange={e => updateAllocation(idx, 'asset', e.target.value)}
                        className="bg-[#0a0e17] border border-[#1f2937] rounded px-3 py-2 text-sm text-gray-300 flex-1 min-w-0"
                      >
                        {AVAILABLE_ASSETS.map(a => {
                          const taken = allocations.some((al, i) => i !== idx && al.asset === a)
                          return (
                            <option key={a} value={a} disabled={taken}>
                              {a}{taken ? ' (used)' : ''}
                            </option>
                          )
                        })}
                      </select>
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          min={0}
                          max={100}
                          value={alloc.pct}
                          onChange={e => updateAllocation(idx, 'pct', Math.max(0, Math.min(100, parseInt(e.target.value) || 0)))}
                          className="w-16 bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-2 text-sm text-white text-center outline-none focus:border-cyan-500/30"
                        />
                        <span className="text-gray-500 text-sm">%</span>
                      </div>
                      <span className="text-xs text-gray-600 w-16 text-right">
                        {fmtGBP(Math.round((alloc.pct / 100) * BH_TOTAL))}
                      </span>
                      <button
                        onClick={() => removeAllocation(idx)}
                        className="text-gray-600 hover:text-red-400 p-1 cursor-pointer transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )
                })}

                {allocations.length < MAX_CUSTOM && AVAILABLE_ASSETS.some(a => !usedAssets.has(a)) && (
                  <button
                    onClick={addAllocation}
                    className="flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    Add asset
                  </button>
                )}

                <div className="flex items-center justify-between pt-2 border-t border-[#1f2937]">
                  <div className="text-xs text-gray-500">
                    Allocated: <span className={totalPct <= 100 ? 'text-cyan-400' : 'text-red-400'}>{totalPct}%</span>
                    {totalPct < 100 && <> &middot; Cash: <span className="text-gray-400">{100 - totalPct}% ({fmtGBP(Math.round(((100 - totalPct) / 100) * BH_TOTAL))})</span></>}
                  </div>
                  {!pctValid && (
                    <span className="text-xs text-red-400">
                      {totalPct > 100 ? 'Total exceeds 100%' : 'Add at least one allocation'}
                    </span>
                  )}
                </div>
              </div>
            )}

            {!useCustom && (
              <div className="text-xs text-gray-500">
                Using default 60/40 allocation across 10 assets. Switch to Custom to choose your own.
              </div>
            )}
          </div>

          {/* Simulation results */}
          {!result || result.chartData.length < 2 ? (
            <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-8 text-center text-gray-500">
              {!pctValid && useCustom
                ? 'Fix allocation percentages to run simulation.'
                : tab === 'live'
                  ? 'Not enough data yet. Live tracking data accumulates daily from 15 March 2026.'
                  : 'Not enough historical price data available.'}
            </div>
          ) : (
            <InvestmentResults result={result} />
          )}
        </>
      )}
    </div>
  )
}

// ═══════════════════════════════════════
// Investment Sim Results
// ═══════════════════════════════════════

function InvestmentResults({ result }: { result: SimResult }) {
  const bhTotalInvested = result.bhBreakdown.reduce((s, a) => s + a.invested, 0) + result.cashAmount

  return (
    <>
      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard
          label={`Buy & Hold ${fmtGBP(bhTotalInvested)}`}
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

      {/* Risk metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCard label="Sharpe Ratio (B&H)" value={result.bhSharpe.toFixed(2)} color={result.bhSharpe >= 1 ? 'text-green-400' : result.bhSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Sharpe Ratio (Alpha)" value={result.asSharpe.toFixed(2)} color={result.asSharpe >= 1 ? 'text-green-400' : result.asSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max Drawdown (B&H)" value={`-${result.bhMaxDrawdown.toFixed(1)}%`} color={result.bhMaxDrawdown < 10 ? 'text-green-400' : result.bhMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max Drawdown (Alpha)" value={`-${result.asMaxDrawdown.toFixed(1)}%`} color={result.asMaxDrawdown < 10 ? 'text-green-400' : result.asMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
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
              {result.cashAmount > 0 && (
                <tr className="border-b border-[#1f2937]/50">
                  <td className="py-1.5 px-2 text-gray-500">Cash</td>
                  <td className="py-1.5 px-2 text-right text-gray-500">{fmtGBP(result.cashAmount)}</td>
                  <td className="py-1.5 px-2 text-right text-gray-500">{fmtGBP(result.cashAmount)}</td>
                  <td className="py-1.5 px-2 text-right text-gray-600">0.00%</td>
                </tr>
              )}
            </tbody>
            <tfoot>
              <tr className="border-t border-[#374151] font-semibold">
                <td className="py-2 px-2 text-white">Total</td>
                <td className="py-2 px-2 text-right text-gray-300">{fmtGBP(bhTotalInvested)}</td>
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

      {/* Methodology section */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <h3 className="text-lg font-semibold text-white mb-4">For the technically minded</h3>
        <div className="space-y-3 text-sm text-gray-400 leading-relaxed">
          <div className="flex gap-3">
            <span className="text-cyan-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Walk-forward validation:</span> Models are tested on data they haven&apos;t seen during training.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-amber-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Backtest caveat:</span> The 5-year backtest uses data the models were trained on &mdash; treat results as optimistic upper bounds, not predictions.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-500 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Transaction costs:</span> Not modelled &mdash; real trading would reduce returns by 0.1&ndash;0.5% per trade.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-green-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Live data:</span> Only 12 days of real signals &mdash; statistically early, check back in 90 days.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-500 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Overfitting check:</span> Some FX pairs show very high accuracy &mdash; this reflects low volatility in those pairs during the tracked period, not model overfitting.</p>
          </div>
          <div className="mt-4 pt-4 border-t border-[#1f2937] text-gray-500 text-xs">
            This is a learning and research project &mdash; not a production trading system or financial advice.
          </div>
        </div>
      </div>
    </>
  )
}

// ═══════════════════════════════════════
// What-If Simulator (moved from Explore)
// ═══════════════════════════════════════

const CAPITAL_OPTIONS = [1_000, 5_000, 10_000, 50_000]
const PERIOD_OPTIONS = [7, 14, 30, 60, 90]

function WhatIfSimulator() {
  const [capital, setCapital] = useState(10_000)
  const [customCapital, setCustomCapital] = useState('')
  const [useCustomCap, setUseCustomCap] = useState(false)
  const [period, setPeriod] = useState(30)
  const [wLoading, setWLoading] = useState(false)
  const [wResult, setWResult] = useState<WhatIfResult | null>(null)
  const [wError, setWError] = useState<string | null>(null)

  const runSim = useCallback(async () => {
    const cap = useCustomCap ? parseFloat(customCapital) || 10_000 : capital
    setWLoading(true)
    setWError(null)
    try {
      const res = await fetchSimulation(period, cap)
      setWResult(res)
    } catch {
      setWError('Simulation failed. Make sure models are trained and the server is running.')
    } finally {
      setWLoading(false)
    }
  }, [useCustomCap, customCapital, capital, period])

  const returnPct = wResult ? wResult.total_return_pct : 0
  const bhPct = wResult ? wResult.vs_buy_and_hold_pct : 0
  const beatsBH = returnPct > bhPct

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-white text-lg font-semibold">What if you had followed our signals?</h3>
        <p className="text-gray-500 text-sm mt-1">
          See exactly what would have happened if you bought when we said buy, sold when we said sell, and held when we said hold.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="text-gray-500 text-xs mb-2">Starting capital</div>
            <div className="flex gap-2 flex-wrap">
              {CAPITAL_OPTIONS.map(c => (
                <button
                  key={c}
                  onClick={() => { setCapital(c); setUseCustomCap(false) }}
                  className={`px-3 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                    !useCustomCap && capital === c
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  {fmtGBP(c)}
                </button>
              ))}
              <div className="flex items-center gap-1">
                <span className="text-gray-500 text-sm">{'\u00A3'}</span>
                <input
                  type="number"
                  value={customCapital}
                  onChange={e => { setCustomCapital(e.target.value); setUseCustomCap(true) }}
                  onFocus={() => setUseCustomCap(true)}
                  placeholder="Custom"
                  className={`w-24 bg-[#0a0e17] border rounded px-2 py-1.5 text-sm text-white outline-none ${
                    useCustomCap ? 'border-cyan-500/30' : 'border-[#1f2937]'
                  }`}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="text-gray-500 text-xs mb-2">Period</div>
            <div className="flex gap-2">
              {PERIOD_OPTIONS.map(p => (
                <button
                  key={p}
                  onClick={() => setPeriod(p)}
                  className={`px-3 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                    period === p
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  {p}d
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-end">
            <button
              onClick={runSim}
              disabled={wLoading}
              className="px-6 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg font-medium text-sm hover:bg-cyan-500/30 transition-colors disabled:opacity-50 cursor-pointer flex items-center gap-2"
            >
              {wLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {wLoading ? 'Running...' : 'Run Simulation'}
            </button>
          </div>
        </div>
      </div>

      {wError && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-red-400 text-sm">{wError}</div>
      )}

      {wResult && (
        <>
          <div className="bg-[#111827] border border-cyan-500/20 rounded-lg p-8 text-center shadow-[0_0_30px_rgba(6,182,212,0.05)]">
            <p className="text-gray-400 text-sm mb-2">{fmtGBP(wResult.starting_capital)} would be worth</p>
            <div className={`text-4xl font-bold mb-2 ${returnPct >= 0 ? 'text-cyan-400' : 'text-red-400'}`}>
              {fmtGBP(wResult.final_value)}
            </div>
            <p className="text-gray-500 text-sm">
              {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}% return &middot; {wResult.days} days &middot; {wResult.signal_accuracy_pct.toFixed(1)}% of signals were correct
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
            <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Performance comparison</div>
            <div className="grid grid-cols-3 gap-4">
              <CompareBar label="Our signals" value={returnPct} highlight={beatsBH} />
              <CompareBar label="Buy & Hold" value={bhPct} highlight={!beatsBH} />
              <CompareBar label="Cash" value={0} highlight={false} />
            </div>
          </div>

          {wResult.daily.length > 1 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Portfolio value over time</div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={wResult.daily} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `\u00A3${(v / 1000).toFixed(1)}k`} width={55} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                    labelStyle={{ color: '#e5e7eb' }}
                    formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '', 'Value']}
                  />
                  <ReferenceLine y={wResult.starting_capital} stroke="#374151" strokeDasharray="4 4" label={{ value: 'Start', fill: '#6b7280', fontSize: 10 }} />
                  <Line type="monotone" dataKey="value" stroke="#06b6d4" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#06b6d4' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {wResult.per_asset.length > 0 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Per-asset breakdown</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
                    <th className="text-left py-2">Asset</th>
                    <th className="text-right py-2">Signal Accuracy</th>
                    <th className="text-right py-2">Contribution to Return</th>
                  </tr>
                </thead>
                <tbody>
                  {wResult.per_asset.map(a => (
                    <tr key={a.asset} className="border-b border-[#1f2937]/50">
                      <td className="py-2 text-white font-medium">{a.asset}</td>
                      <td className={`py-2 text-right ${a.signal_accuracy_pct >= 60 ? 'text-emerald-400' : a.signal_accuracy_pct >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
                        {a.signal_accuracy_pct.toFixed(1)}%
                      </td>
                      <td className={`py-2 text-right ${a.contribution_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {a.contribution_pct >= 0 ? '+' : ''}{a.contribution_pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="bg-[#0a0e17] border border-[#1f2937] rounded-lg p-4 text-gray-500 text-xs leading-relaxed">
            This simulation uses the same models that generated live signals. It is not a true backtest — the models were trained on data that includes this period, so results may be optimistic. Live performance since {wResult.inception_date} is the more reliable measure.
          </div>
        </>
      )}
    </div>
  )
}

// ═══════════════════════════════════════
// Shared Sub-components
// ═══════════════════════════════════════

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

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
    </div>
  )
}

function CompareBar({ label, value, highlight }: { label: string; value: number; highlight: boolean }) {
  const isPositive = value >= 0
  const barColor = highlight ? (isPositive ? '#10b981' : '#ef4444') : '#374151'

  return (
    <div className="text-center">
      <div className={`text-lg font-bold ${highlight ? (isPositive ? 'text-emerald-400' : 'text-red-400') : 'text-gray-500'}`}>
        {isPositive ? '+' : ''}{value.toFixed(2)}%
      </div>
      <div className="text-gray-500 text-xs mt-1">{label}</div>
      <div className="h-1.5 mt-2 bg-[#0a0e17] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(Math.abs(value) * 2, 100)}%`, background: barColor }}
        />
      </div>
    </div>
  )
}
