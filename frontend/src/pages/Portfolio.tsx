import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, Legend, LineChart, Line, ReferenceLine,
} from 'recharts'
import { fetchPortfolio, fetchDailyTracker } from '../lib/api'
import UserHoldings from '../components/UserHoldings'
import type { PortfolioResult, StrategyResult, AssetBacktest, DailyTrackerResult } from '../lib/types'

const STRATEGY_LABELS: Record<string, string> = {
  sharpe: 'Sharpe-Weighted',
  equal: 'Equal Weight',
  inverse_volatility: 'Inverse Volatility',
}

const CLASS_COLORS: Record<string, string> = {
  stock: '#06b6d4',
  fx: '#10b981',
  crypto: '#f59e0b',
  unknown: '#6b7280',
}

const SIGNAL_COLORS: Record<string, string> = {
  BUY: '#10b981',
  SELL: '#ef4444',
  HOLD: '#f59e0b',
  'N/A': '#6b7280',
}

const TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'QQQ', 'TSLA']

function fmt(n: number, dp = 2): string {
  return n.toFixed(dp)
}

function fmtGBP(n: number): string {
  return '£' + Math.round(n).toLocaleString()
}

export default function Portfolio() {
  const [data, setData] = useState<PortfolioResult | null>(null)
  const [tracker, setTracker] = useState<DailyTrackerResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedStrategy, setSelectedStrategy] = useState('sharpe')
  const [expandedAsset, setExpandedAsset] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetchPortfolio().catch(() => null),
      fetchDailyTracker().catch(() => null),
    ]).then(([portfolio, tracker]) => {
      if (portfolio) setData(portfolio)
      if (tracker) setTracker(tracker)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-500 p-8">Loading portfolio data...</div>
  if (!data) return <div className="text-gray-500 p-8">Failed to load portfolio data.</div>

  if (!data.has_data || !data.strategies) {
    return (
      <div className="p-8">
        <h2 className="text-white text-xl font-semibold mb-4">Portfolio</h2>
        <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-8 text-center">
          <p className="text-gray-400 text-lg mb-2">No backtest data available yet</p>
          <p className="text-gray-500 text-sm">{data.note || 'Run cargo run --release --bin train to generate backtest data.'}</p>
        </div>
      </div>
    )
  }

  const strategies = data.strategies
  const current = strategies[selectedStrategy]
  const backtests = data.per_asset_backtest || []

  if (!current) return <div className="text-gray-500 p-8">Strategy data missing.</div>

  return (
    <div className="space-y-6">
      <h2 className="text-white text-xl font-semibold">Portfolio</h2>

      {/* Strategy selector cards */}
      <StrategySelector
        strategies={strategies}
        selected={selectedStrategy}
        onSelect={setSelectedStrategy}
      />

      {/* Headline */}
      <Headline capital={data.starting_capital} strategy={current} />

      {/* Part 2 — Daily live tracker */}
      <DailyTracker tracker={tracker} />

      {/* User holdings */}
      <UserHoldings />

      {/* Allocation table + bar chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AllocationTable allocations={current.allocations} />
        <AllocationBarChart allocations={current.allocations} />
      </div>

      {/* Exposure breakdown + strategy comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ExposureBreakdown allocations={current.allocations} />
        <StrategyComparison strategies={strategies} />
      </div>

      {/* Per-asset backtest cards */}
      <BacktestCards
        backtests={backtests}
        expanded={expandedAsset}
        onToggle={(a) => setExpandedAsset(expandedAsset === a ? null : a)}
      />
    </div>
  )
}

// ─── Daily Tracker (Part 2) ───

function DailyTracker({ tracker }: { tracker: DailyTrackerResult | null }) {
  if (!tracker) return null

  if (!tracker.has_data) {
    return (
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          <h3 className="text-white font-semibold">Live Signal Portfolio</h3>
          <span className="text-xs px-2 py-0.5 rounded bg-amber-400/10 text-amber-400">Awaiting Data</span>
        </div>
        <p className="text-gray-500 text-sm">
          {tracker.note || 'Daily tracking begins on the first run after training. The portfolio compounds from the backtest seed value above.'}
        </p>
      </div>
    )
  }

  const seed = tracker.seed_value ?? 0
  const current = tracker.current_value ?? 0
  const dailyReturn = tracker.daily_return ?? 0
  const cumReturn = tracker.cumulative_return ?? 0
  const accuracy = tracker.model_accuracy_pct ?? 0
  const days = tracker.days_tracked ?? 0
  const signals = tracker.today_signals ?? []
  const curve = tracker.equity_curve ?? []

  const isUp = dailyReturn >= 0
  const isCumUp = cumReturn >= 0

  return (
    <div className="bg-[#111827] border border-cyan-500/20 rounded-lg p-6 shadow-[0_0_30px_rgba(6,182,212,0.05)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <h3 className="text-white font-semibold">Live Signal Portfolio</h3>
          <span className="text-xs px-2 py-0.5 rounded bg-cyan-400/10 text-cyan-400">Part 2</span>
        </div>
        <div className="text-gray-500 text-xs">
          Since {tracker.inception_date} · {days} day{days !== 1 ? 's' : ''}
        </div>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="bg-[#0a0e17] rounded-lg p-4">
          <div className="text-gray-500 text-xs mb-1">Current Value</div>
          <div className="text-cyan-400 text-2xl font-bold">{fmtGBP(current)}</div>
          <div className="text-gray-500 text-xs mt-1">from {fmtGBP(seed)}</div>
        </div>
        <div className="bg-[#0a0e17] rounded-lg p-4">
          <div className="text-gray-500 text-xs mb-1">Today</div>
          <div className={`text-2xl font-bold ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
            {isUp ? '+' : ''}{fmt(dailyReturn)}%
          </div>
          <div className="text-gray-500 text-xs mt-1">
            {isUp ? '+' : ''}{fmtGBP((current - current / (1 + dailyReturn / 100)))}
          </div>
        </div>
        <div className="bg-[#0a0e17] rounded-lg p-4">
          <div className="text-gray-500 text-xs mb-1">Since Inception</div>
          <div className={`text-2xl font-bold ${isCumUp ? 'text-emerald-400' : 'text-red-400'}`}>
            {isCumUp ? '+' : ''}{fmt(cumReturn)}%
          </div>
          <div className="text-gray-500 text-xs mt-1">
            {isCumUp ? '+' : ''}{fmtGBP(current - seed)}
          </div>
        </div>
        <div className="bg-[#0a0e17] rounded-lg p-4">
          <div className="text-gray-500 text-xs mb-1">Signal Accuracy</div>
          <div className={`text-2xl font-bold ${accuracy >= 55 ? 'text-emerald-400' : accuracy >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
            {fmt(accuracy, 1)}%
          </div>
          <div className="text-gray-500 text-xs mt-1">direction correct</div>
        </div>
      </div>

      {/* Equity curve + today's signals */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Mini equity curve */}
        {curve.length > 1 && (
          <div>
            <div className="text-gray-400 text-xs mb-3 uppercase tracking-wider">Portfolio Growth</div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={curve} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#4b5563', fontSize: 10 }}
                  tickFormatter={(v) => v.slice(5)} // MM-DD
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fill: '#4b5563', fontSize: 10 }}
                  tickFormatter={(v) => `£${(v / 1000).toFixed(0)}k`}
                  width={48}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                  labelStyle={{ color: '#e5e7eb' }}
                  formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '', 'Value']}
                />
                <ReferenceLine y={seed} stroke="#374151" strokeDasharray="4 4" />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#06b6d4' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Today's signals breakdown */}
        {signals.length > 0 && (
          <div>
            <div className="text-gray-400 text-xs mb-3 uppercase tracking-wider">
              Today's Signals · {tracker.last_updated}
            </div>
            <div className="space-y-2">
              {signals.map((s: { asset: string; signal: string; weight: number; price_return: number; contribution: number }) => (
                <div key={s.asset} className="flex items-center justify-between bg-[#0a0e17] rounded px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs font-bold px-1.5 py-0.5 rounded"
                      style={{
                        color: SIGNAL_COLORS[s.signal] || SIGNAL_COLORS['N/A'],
                        background: `${SIGNAL_COLORS[s.signal] || SIGNAL_COLORS['N/A']}20`,
                      }}
                    >
                      {s.signal}
                    </span>
                    <span className="text-white text-sm font-medium">{s.asset}</span>
                    <span className="text-gray-500 text-xs">{fmt(s.weight, 1)}%</span>
                  </div>
                  <div className="flex gap-4 text-xs">
                    <span className={s.price_return >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {s.price_return >= 0 ? '+' : ''}{fmt(s.price_return)}%
                    </span>
                    <span className={`${s.contribution >= 0 ? 'text-emerald-400' : 'text-red-400'} text-gray-500`}>
                      {s.contribution >= 0 ? '+' : ''}{fmt(s.contribution, 3)}% contrib
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Strategy Selector ───

function StrategySelector({
  strategies, selected, onSelect,
}: {
  strategies: Record<string, StrategyResult>
  selected: string
  onSelect: (s: string) => void
}) {
  return (
    <div className="grid grid-cols-3 gap-4">
      {Object.entries(strategies).map(([key, s]) => (
        <button
          key={key}
          onClick={() => onSelect(key)}
          className={`bg-[#111827] border rounded-lg p-4 text-left transition-all ${
            selected === key
              ? 'border-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
              : 'border-[#1f2937] hover:border-[#374151]'
          }`}
        >
          <div className="text-gray-400 text-xs mb-1">{STRATEGY_LABELS[key] || key}</div>
          <div className="text-white text-lg font-bold">{fmtGBP(s.final_value)}</div>
          <div className="flex gap-3 mt-2 text-xs">
            <span className={s.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}>
              {fmt(s.total_return, 1)}%
            </span>
            <span className="text-gray-500">Sharpe {fmt(s.sharpe_ratio)}</span>
            <span className="text-gray-500">DD {fmt(s.max_drawdown, 1)}%</span>
          </div>
        </button>
      ))}
    </div>
  )
}

// ─── Headline ───

function Headline({ capital, strategy: s }: { capital: number; strategy: StrategyResult }) {
  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-8">
      <p className="text-gray-400 text-sm mb-2">If you invested {fmtGBP(capital)} three years ago...</p>
      <div className="text-4xl font-bold text-cyan-400 mb-4">{fmtGBP(s.final_value)}</div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
        <MetricBox label="Total Return" value={`${fmt(s.total_return, 1)}%`} positive={s.total_return >= 0} />
        <MetricBox label="Sharpe Ratio" value={fmt(s.sharpe_ratio)} />
        <MetricBox label="Max Drawdown" value={`${fmt(s.max_drawdown, 1)}%`} negative />
        <MetricBox label="vs Buy & Hold" value={`${s.excess_return >= 0 ? '+' : ''}${fmt(s.excess_return, 1)}%`} positive={s.excess_return >= 0} />
      </div>
    </div>
  )
}

function MetricBox({ label, value, positive, negative }: {
  label: string; value: string; positive?: boolean; negative?: boolean
}) {
  const color = positive !== undefined
    ? (positive ? 'text-emerald-400' : 'text-red-400')
    : negative ? 'text-amber-400' : 'text-white'
  return (
    <div>
      <div className="text-gray-500 text-xs mb-1">{label}</div>
      <div className={`text-xl font-semibold ${color}`}>{value}</div>
    </div>
  )
}

// ─── Allocation Table ───

function AllocationTable({ allocations }: { allocations: StrategyResult['allocations'] }) {
  const sorted = [...allocations].sort((a, b) => b.contribution - a.contribution)
  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <h3 className="text-white font-semibold mb-4">Capital Allocation</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs border-b border-[#1f2937]">
              <th className="text-left py-2">Asset</th>
              <th className="text-left py-2">Class</th>
              <th className="text-right py-2">Weight</th>
              <th className="text-right py-2">Allocated</th>
              <th className="text-right py-2">Return</th>
              <th className="text-right py-2">Contribution</th>
              <th className="text-center py-2">Signal</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((a) => (
              <tr key={a.asset} className="border-b border-[#1f2937]/50 hover:bg-[#1a2332]">
                <td className="py-2 text-white font-medium">{a.asset}</td>
                <td className="py-2">
                  <span className="text-xs px-1.5 py-0.5 rounded" style={{
                    color: CLASS_COLORS[a.asset_class] || CLASS_COLORS.unknown,
                    background: `${CLASS_COLORS[a.asset_class] || CLASS_COLORS.unknown}15`,
                  }}>{a.asset_class}</span>
                </td>
                <td className="py-2 text-right text-gray-300">{fmt(a.weight, 1)}%</td>
                <td className="py-2 text-right text-gray-300">{fmtGBP(a.allocated)}</td>
                <td className={`py-2 text-right ${a.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {a.return >= 0 ? '+' : ''}{fmt(a.return, 1)}%
                </td>
                <td className={`py-2 text-right ${a.contribution >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {a.contribution >= 0 ? '+' : ''}{fmt(a.contribution, 1)}%
                </td>
                <td className="py-2 text-center">
                  <span className="text-xs font-bold px-2 py-0.5 rounded" style={{
                    color: SIGNAL_COLORS[a.signal] || SIGNAL_COLORS['N/A'],
                    background: `${SIGNAL_COLORS[a.signal] || SIGNAL_COLORS['N/A']}20`,
                  }}>{a.signal}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Allocation Bar Chart ───

function AllocationBarChart({ allocations }: { allocations: StrategyResult['allocations'] }) {
  const sorted = [...allocations].sort((a, b) => b.weight - a.weight)
  const chartData = sorted.map(a => ({
    asset: a.asset,
    weight: a.weight,
    fill: CLASS_COLORS[a.asset_class] || CLASS_COLORS.unknown,
  }))

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <h3 className="text-white font-semibold mb-4">Asset Weights</h3>
      <ResponsiveContainer width="100%" height={Math.max(250, chartData.length * 28)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 60, right: 20 }}>
          <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
          <YAxis type="category" dataKey="asset" tick={{ fill: '#e5e7eb', fontSize: 12 }} width={55} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px' }}
            labelStyle={{ color: '#e5e7eb' }}
            formatter={(v) => [`${fmt(Number(v), 1)}%`, 'Weight']}
          />
          <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
            {chartData.map((d, i) => (
              <Cell key={i} fill={d.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-3 text-xs text-gray-500">
        <span><span className="inline-block w-2.5 h-2.5 rounded-sm mr-1" style={{ background: CLASS_COLORS.stock }} />Stocks</span>
        <span><span className="inline-block w-2.5 h-2.5 rounded-sm mr-1" style={{ background: CLASS_COLORS.fx }} />FX</span>
        <span><span className="inline-block w-2.5 h-2.5 rounded-sm mr-1" style={{ background: CLASS_COLORS.crypto }} />Crypto</span>
      </div>
    </div>
  )
}

// ─── Exposure Breakdown ───

function ExposureBreakdown({ allocations }: { allocations: StrategyResult['allocations'] }) {
  let stockWeight = 0, fxWeight = 0, cryptoWeight = 0, techWeight = 0
  for (const a of allocations) {
    if (a.asset_class === 'stock') stockWeight += a.weight
    else if (a.asset_class === 'fx') fxWeight += a.weight
    else if (a.asset_class === 'crypto') cryptoWeight += a.weight
    if (TECH_STOCKS.includes(a.asset)) techWeight += a.weight
  }

  const rows = [
    { label: 'Stocks', pct: stockWeight, color: CLASS_COLORS.stock },
    { label: 'FX', pct: fxWeight, color: CLASS_COLORS.fx },
    { label: 'Crypto', pct: cryptoWeight, color: CLASS_COLORS.crypto },
    { label: 'Tech Exposure', pct: techWeight, color: '#a78bfa' },
    { label: 'Defensive', pct: 0, color: '#6b7280' },
  ]

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <h3 className="text-white font-semibold mb-4">Exposure Breakdown</h3>
      <div className="space-y-3">
        {rows.map((r) => (
          <div key={r.label}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">{r.label}</span>
              <span className="text-white">{fmt(r.pct, 1)}%</span>
            </div>
            <div className="h-2 bg-[#0a0e17] rounded-full">
              <div className="h-full rounded-full transition-all" style={{
                width: `${Math.min(r.pct, 100)}%`,
                background: r.color,
              }} />
            </div>
          </div>
        ))}
      </div>
      {techWeight > 50 && (
        <p className="text-amber-400/70 text-xs mt-4 italic">
          Tech exposure is {fmt(techWeight, 0)}% — consider diversifying into defensive sectors.
        </p>
      )}
      {fxWeight === 0 && cryptoWeight === 0 && (
        <p className="text-gray-500 text-xs mt-4 italic">
          Portfolio is 100% equities. FX and crypto assets did not meet quality thresholds.
        </p>
      )}
    </div>
  )
}

// ─── Strategy Comparison ───

function StrategyComparison({ strategies }: { strategies: Record<string, StrategyResult> }) {
  const chartData = Object.entries(strategies).map(([key, s]) => ({
    strategy: STRATEGY_LABELS[key] || key,
    'Total Return': s.total_return,
    'Sharpe Ratio': s.sharpe_ratio,
    'Max Drawdown': s.max_drawdown,
  }))

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      <h3 className="text-white font-semibold mb-4">Strategy Comparison</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData} margin={{ left: 10, right: 10 }}>
          <XAxis dataKey="strategy" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px' }}
            labelStyle={{ color: '#e5e7eb' }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
          <Bar dataKey="Total Return" fill="#06b6d4" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Sharpe Ratio" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Max Drawdown" fill="#10b981" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Per-Asset Backtest Cards ───

function BacktestCards({ backtests, expanded, onToggle }: {
  backtests: AssetBacktest[]
  expanded: string | null
  onToggle: (asset: string) => void
}) {
  const sorted = [...backtests].sort((a, b) => b.sharpe_ratio - a.sharpe_ratio)

  const verdictColor = (v: string) =>
    v === 'EDGE' ? 'text-emerald-400 bg-emerald-400/10' :
    v === 'MARGINAL' ? 'text-amber-400 bg-amber-400/10' :
    'text-red-400 bg-red-400/10'

  return (
    <div>
      <h3 className="text-white font-semibold mb-4">Per-Asset Backtest Results</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {sorted.map((b) => {
          const isExpanded = expanded === b.asset
          return (
            <div
              key={b.asset}
              className="bg-[#111827] border border-[#1f2937] rounded-lg overflow-hidden cursor-pointer hover:border-[#374151] transition-colors"
              onClick={() => onToggle(b.asset)}
            >
              {/* Header */}
              <div className="p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-white font-semibold">{b.asset}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded" style={{
                    color: CLASS_COLORS[b.asset_class] || CLASS_COLORS.unknown,
                    background: `${CLASS_COLORS[b.asset_class] || CLASS_COLORS.unknown}15`,
                  }}>{b.asset_class}</span>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${verdictColor(b.verdict)}`}>
                  {b.verdict}
                </span>
              </div>

              {/* Summary row always visible */}
              <div className="px-4 pb-3 flex gap-4 text-xs">
                <span className={b.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                  {b.total_return >= 0 ? '+' : ''}{fmt(b.total_return, 1)}%
                </span>
                <span className="text-gray-500">Sharpe {fmt(b.sharpe_ratio)}</span>
                <span className="text-gray-500">Win {fmt(b.win_rate, 0)}%</span>
              </div>

              {/* Expanded details */}
              {isExpanded && (
                <div className="border-t border-[#1f2937] p-4 grid grid-cols-2 gap-y-2 gap-x-4 text-xs">
                  <Stat label="Total Return" value={`${fmt(b.total_return, 2)}%`} positive={b.total_return >= 0} />
                  <Stat label="Buy & Hold" value={`${fmt(b.buy_hold_return, 2)}%`} />
                  <Stat label="Excess Return" value={`${fmt(b.excess_return, 2)}%`} positive={b.excess_return >= 0} />
                  <Stat label="Sharpe Ratio" value={fmt(b.sharpe_ratio)} />
                  <Stat label="Max Drawdown" value={`${fmt(b.max_drawdown, 2)}%`} />
                  <Stat label="Win Rate" value={`${fmt(b.win_rate, 1)}%`} />
                  <Stat label="Profit Factor" value={fmt(b.profit_factor)} />
                  <Stat label="Expectancy" value={`${fmt(b.expectancy, 3)}%`} />
                  <Stat label="Days in Market" value={`${b.days_in_market} / ${b.total_days}`} />
                  <Stat label="Market Time" value={`${fmt(b.total_days > 0 ? (b.days_in_market / b.total_days) * 100 : 0, 0)}%`} />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Stat({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  const color = positive !== undefined
    ? (positive ? 'text-emerald-400' : 'text-red-400')
    : 'text-white'
  return (
    <div>
      <div className="text-gray-500">{label}</div>
      <div className={`font-medium ${color}`}>{value}</div>
    </div>
  )
}
