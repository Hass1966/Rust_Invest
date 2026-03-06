import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, BarChart, Bar, Cell,
} from 'recharts'

const BASE = ''

// ─── Types ───

interface PortfolioPoint {
  date: string
  value: number
  daily_return: number
  cumulative_return: number
}

interface PortfolioHistory {
  has_data: boolean
  note?: string
  seed_value?: number
  current_value?: number
  cumulative_return?: number
  days?: number
  points?: PortfolioPoint[]
}

interface SignalEntry {
  date: string
  signal: string
  confidence?: number
  probability_up?: number
  price?: number
  rsi?: number
  asset_class?: string
}

interface SignalSummary {
  total: number
  buys: number
  sells: number
  holds: number
}

interface SignalsHistory {
  has_data: boolean
  note?: string
  days?: number
  signals?: Record<string, SignalEntry[]>
  accuracy?: Record<string, SignalSummary>
}

// ─── Helpers ───

function fmt(n: number, dp = 2): string { return n.toFixed(dp) }
function fmtGBP(n: number): string { return '£' + Math.round(n).toLocaleString() }

const SIGNAL_COLORS: Record<string, string> = {
  BUY: '#10b981',
  SELL: '#ef4444',
  HOLD: '#f59e0b',
  'N/A': '#374151',
}

const CLASS_COLORS: Record<string, string> = {
  stock: '#06b6d4',
  fx: '#10b981',
  crypto: '#f59e0b',
}

// ─── Main Component ───

export default function History() {
  const [portfolio, setPortfolio] = useState<PortfolioHistory | null>(null)
  const [signals, setSignals] = useState<SignalsHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState<7 | 14 | 30 | 90>(30)
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      fetch(`${BASE}/api/v1/history/portfolio`).then(r => r.json()).catch(() => null),
      fetch(`${BASE}/api/v1/history/signals?days=${timeRange}`).then(r => r.json()).catch(() => null),
    ]).then(([p, s]) => {
      if (p) setPortfolio(p)
      if (s) setSignals(s)
    }).finally(() => setLoading(false))
  }, [timeRange])

  if (loading) return <div className="text-gray-500 p-8">Loading history...</div>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-white text-xl font-semibold">History</h2>
        <div className="flex gap-1">
          {([7, 14, 30, 90] as const).map(d => (
            <button
              key={d}
              onClick={() => setTimeRange(d)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                timeRange === d
                  ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                  : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
              }`}
            >
              {d}D
            </button>
          ))}
        </div>
      </div>

      {/* Portfolio equity curve */}
      <PortfolioChart portfolio={portfolio} />

      {/* Daily returns bar chart */}
      <DailyReturnsChart portfolio={portfolio} />

      {/* Signal heatmap */}
      <SignalHeatmap
        signals={signals}
        selectedAsset={selectedAsset}
        onSelectAsset={setSelectedAsset}
      />

      {/* Asset signal timeline */}
      {selectedAsset && signals?.signals?.[selectedAsset] && (
        <AssetTimeline
          asset={selectedAsset}
          entries={signals.signals[selectedAsset]}
          summary={signals.accuracy?.[selectedAsset]}
          onClose={() => setSelectedAsset(null)}
        />
      )}
    </div>
  )
}

// ─── Portfolio Equity Curve ───

function PortfolioChart({ portfolio }: { portfolio: PortfolioHistory | null }) {
  if (!portfolio?.has_data || !portfolio.points?.length) {
    return (
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-8 text-center">
        <div className="text-gray-400 font-medium mb-1">Portfolio Equity Curve</div>
        <p className="text-gray-500 text-sm">
          {portfolio?.note || 'Data accumulates hourly. Check back once serve has been running for a few hours.'}
        </p>
      </div>
    )
  }

  const points = portfolio.points!
  const seed = portfolio.seed_value!
  const current = portfolio.current_value!
  const cumReturn = portfolio.cumulative_return!
  const isUp = cumReturn >= 0

  const minVal = Math.min(...points.map(p => p.value)) * 0.999
  const maxVal = Math.max(...points.map(p => p.value)) * 1.001

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="text-gray-400 text-sm mb-1">Portfolio Value</div>
          <div className="text-cyan-400 text-3xl font-bold">{fmtGBP(current)}</div>
          <div className={`text-sm mt-1 ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
            {isUp ? '+' : ''}{fmt(cumReturn)}% since inception · seeded from {fmtGBP(seed)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-gray-500 text-xs">{portfolio.days} day{portfolio.days !== 1 ? 's' : ''} tracked</div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={points} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
          <XAxis
            dataKey="date"
            tick={{ fill: '#4b5563', fontSize: 11 }}
            tickFormatter={(v) => v.slice(5)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#4b5563', fontSize: 11 }}
            tickFormatter={(v) => `£${(v / 1000).toFixed(0)}k`}
            width={52}
            domain={[minVal, maxVal]}
          />
          <Tooltip
            contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(v: number | undefined, name?: string) => [
              v == null ? '' : name === 'value' ? fmtGBP(v) : `${fmt(v)}%`,
              name === 'value' ? 'Portfolio' : 'Cumulative Return'
            ]}
          />
          <ReferenceLine y={seed} stroke="#374151" strokeDasharray="4 4" label={{ value: 'Seed', fill: '#4b5563', fontSize: 10 }} />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#06b6d4"
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 5, fill: '#06b6d4', strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Daily Returns Bar Chart ───

function DailyReturnsChart({ portfolio }: { portfolio: PortfolioHistory | null }) {
  if (!portfolio?.has_data || !portfolio.points?.length || portfolio.points.length < 2) return null

  const points = portfolio.points!.slice(-30) // last 30 days max

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <div className="text-gray-400 text-sm font-medium mb-4">Daily Returns</div>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={points} margin={{ left: 0, right: 0 }}>
          <XAxis
            dataKey="date"
            tick={{ fill: '#4b5563', fontSize: 10 }}
            tickFormatter={(v) => v.slice(5)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#4b5563', fontSize: 10 }}
            tickFormatter={(v) => `${v > 0 ? '+' : ''}${fmt(v, 1)}%`}
            width={48}
          />
          <Tooltip
            contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
            formatter={(v: number | undefined) => [`${v != null && v > 0 ? '+' : ''}${v != null ? fmt(v) : ''}%`, 'Daily Return']}
          />
          <ReferenceLine y={0} stroke="#374151" />
          <Bar dataKey="daily_return" radius={[2, 2, 0, 0]}>
            {points.map((p, i) => (
              <Cell key={i} fill={p.daily_return >= 0 ? '#10b981' : '#ef4444'} opacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Signal Heatmap ───

function SignalHeatmap({ signals, selectedAsset, onSelectAsset }: {
  signals: SignalsHistory | null
  selectedAsset: string | null
  onSelectAsset: (a: string) => void
}) {
  if (!signals?.has_data || !signals.signals) {
    return (
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-8 text-center">
        <div className="text-gray-400 font-medium mb-1">Signal History</div>
        <p className="text-gray-500 text-sm">
          {signals?.note || 'Signal history accumulates as serve runs. Check back after a few hours.'}
        </p>
      </div>
    )
  }

  // Collect all unique dates across all assets, sorted
  const allDates = Array.from(
    new Set(
      Object.values(signals.signals).flatMap(entries => entries.map(e => e.date))
    )
  ).sort()

  // Only show last 14 dates to keep the grid readable
  const dates = allDates.slice(-14)

  // Sort assets by class then name
  const assets = Object.keys(signals.signals).sort((a, b) => {
    const classA = signals.signals![a]?.[0]?.asset_class || ''
    const classB = signals.signals![b]?.[0]?.asset_class || ''
    if (classA !== classB) return classA.localeCompare(classB)
    return a.localeCompare(b)
  })

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="text-gray-400 text-sm font-medium">Signal Heatmap</div>
        <div className="flex gap-3 text-xs text-gray-500">
          <span><span className="inline-block w-3 h-3 rounded-sm mr-1" style={{ background: SIGNAL_COLORS.BUY }} />BUY</span>
          <span><span className="inline-block w-3 h-3 rounded-sm mr-1" style={{ background: SIGNAL_COLORS.HOLD }} />HOLD</span>
          <span><span className="inline-block w-3 h-3 rounded-sm mr-1" style={{ background: SIGNAL_COLORS.SELL }} />SELL</span>
          <span><span className="inline-block w-3 h-3 rounded-sm mr-1 bg-[#374151]" />No data</span>
        </div>
      </div>
      <p className="text-gray-600 text-xs mb-4">Click an asset row to see detailed signal timeline</p>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left text-gray-500 pb-2 pr-4 font-normal w-28">Asset</th>
              {dates.map(d => (
                <th key={d} className="text-gray-600 pb-2 font-normal text-center" style={{ minWidth: 28 }}>
                  {d.slice(5)} {/* MM-DD */}
                </th>
              ))}
              <th className="text-gray-500 pb-2 font-normal text-right pl-4">BUY%</th>
            </tr>
          </thead>
          <tbody>
            {assets.map(asset => {
              const entries = signals.signals![asset] || []
              const byDate = Object.fromEntries(entries.map(e => [e.date, e]))
              const summary = signals.accuracy?.[asset]
              const buyPct = summary && summary.total > 0
                ? Math.round(summary.buys / summary.total * 100)
                : 0
              const assetClass = entries[0]?.asset_class || 'stock'
              const isSelected = selectedAsset === asset

              return (
                <tr
                  key={asset}
                  onClick={() => onSelectAsset(asset)}
                  className={`cursor-pointer transition-colors ${isSelected ? 'bg-cyan-500/5' : 'hover:bg-white/[0.02]'}`}
                >
                  <td className="py-1 pr-4">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: CLASS_COLORS[assetClass] || '#6b7280' }}
                      />
                      <span className={`font-medium ${isSelected ? 'text-cyan-400' : 'text-gray-300'}`}>
                        {asset}
                      </span>
                    </div>
                  </td>
                  {dates.map(d => {
                    const entry = byDate[d]
                    const sig = entry?.signal || 'N/A'
                    const color = SIGNAL_COLORS[sig] || SIGNAL_COLORS['N/A']
                    return (
                      <td key={d} className="py-1 text-center">
                        <div
                          className="w-5 h-5 rounded mx-auto flex items-center justify-center text-[9px] font-bold"
                          style={{ background: `${color}25`, color }}
                          title={`${asset} ${d}: ${sig}${entry?.price ? ` @ $${entry.price.toFixed(2)}` : ''}`}
                        >
                          {sig === 'N/A' ? '·' : sig[0]}
                        </div>
                      </td>
                    )
                  })}
                  <td className="py-1 text-right pl-4">
                    <span className={`font-medium ${buyPct >= 50 ? 'text-emerald-400' : 'text-gray-500'}`}>
                      {buyPct}%
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Asset Signal Timeline ───

function AssetTimeline({ asset, entries, summary, onClose }: {
  asset: string
  entries: SignalEntry[]
  summary?: SignalSummary
  onClose: () => void
}) {
  const sorted = [...entries].sort((a, b) => a.date.localeCompare(b.date))

  return (
    <div className="bg-[#111827] border border-cyan-500/20 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-white font-semibold">{asset}</span>
          <span className="text-gray-500 text-sm">Signal Timeline</span>
        </div>
        <div className="flex items-center gap-4">
          {summary && (
            <div className="flex gap-3 text-xs">
              <span className="text-emerald-400">{summary.buys} BUY</span>
              <span className="text-amber-400">{summary.holds} HOLD</span>
              <span className="text-red-400">{summary.sells} SELL</span>
            </div>
          )}
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 text-xs px-2 py-1 rounded border border-[#1f2937] hover:border-[#374151]"
          >
            Close
          </button>
        </div>
      </div>

      <div className="space-y-1.5">
        {sorted.map((e, i) => (
          <div key={i} className="flex items-center gap-4 bg-[#0a0e17] rounded px-3 py-2 text-xs">
            <span className="text-gray-500 w-20 flex-shrink-0">{e.date}</span>
            <span
              className="font-bold px-2 py-0.5 rounded w-12 text-center"
              style={{
                color: SIGNAL_COLORS[e.signal] || SIGNAL_COLORS['N/A'],
                background: `${SIGNAL_COLORS[e.signal] || SIGNAL_COLORS['N/A']}20`,
              }}
            >
              {e.signal}
            </span>
            {e.price && (
              <span className="text-gray-400">${e.price.toFixed(2)}</span>
            )}
            {e.probability_up !== undefined && (
              <span className="text-gray-500">P(↑) {fmt(e.probability_up, 1)}%</span>
            )}
            {e.confidence !== undefined && (
              <span className="text-gray-500">Conf {fmt(e.confidence, 1)}%</span>
            )}
            {e.rsi !== undefined && (
              <span className={`${e.rsi > 70 ? 'text-red-400' : e.rsi < 30 ? 'text-emerald-400' : 'text-gray-500'}`}>
                RSI {fmt(e.rsi, 1)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
