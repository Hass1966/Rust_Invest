import { useEffect, useState, useCallback } from 'react'
import { TrendingUp, TrendingDown, Minus, Activity, RefreshCw, Gauge, ChevronDown, ChevronUp } from 'lucide-react'
import { fetchSignals, fetchMorningBriefing, fetchHints, comparePortfolio, fetchUserHoldings } from '../lib/api'
import type { EnrichedSignal, Hint } from '../lib/types'
import type { PortfolioComparison } from '../lib/api'
import { translateSignalSummary, confidenceLabel, convictionInfo } from '../lib/plain-english'

type Filter = 'All' | 'Stocks' | 'FX' | 'Crypto'

export default function Dashboard() {
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [briefing, setBriefing] = useState<string | null>(null)
  const [briefingLoading, setBriefingLoading] = useState(true)
  const [briefingTime, setBriefingTime] = useState<string | null>(null)
  const [briefingError, setBriefingError] = useState(false)
  const [hints, setHints] = useState<Hint[]>([])
  const [signalsError, setSignalsError] = useState(false)
  const [filter, setFilter] = useState<Filter>('All')
  const [comparison, setComparison] = useState<PortfolioComparison | null>(null)
  const [hasHoldings, setHasHoldings] = useState(false)

  const loadBriefing = useCallback(() => {
    setBriefingLoading(true)
    setBriefingError(false)
    fetchMorningBriefing()
      .then(text => {
        setBriefing(text)
        setBriefingTime(new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }))
      })
      .catch(() => setBriefingError(true))
      .finally(() => setBriefingLoading(false))
  }, [])

  useEffect(() => {
    fetchSignals()
      .then(setSignals)
      .catch(() => setSignalsError(true))
      .finally(() => setLoading(false))
    loadBriefing()
    fetchHints().then(setHints).catch(() => {})
    // Load portfolio quick-view
    fetchUserHoldings().then(h => {
      if (h.length > 0) {
        setHasHoldings(true)
        comparePortfolio('weekly').then(setComparison).catch(() => {})
      }
    }).catch(() => {})
  }, [loadBriefing])

  const filtered = signals.filter(s => {
    if (filter === 'All') return true
    if (filter === 'Stocks') return s.asset_class === 'stock'
    if (filter === 'FX') return s.asset_class === 'fx'
    if (filter === 'Crypto') return s.asset_class === 'crypto'
    return true
  })

  const buys = signals.filter(s => s.signal === 'BUY')
  const sells = signals.filter(s => s.signal === 'SELL')
  const holds = signals.filter(s => s.signal === 'HOLD')

  const avgBuyConf = buys.length > 0
    ? buys.reduce((sum, s) => sum + s.technical.confidence, 0) / buys.length
    : 0
  const signalQualityLabel = avgBuyConf > 15 ? 'High' : avgBuyConf >= 8 ? 'Moderate' : 'Low'
  const signalQualityColor = avgBuyConf > 15 ? 'text-emerald-400' : avgBuyConf >= 8 ? 'text-amber-400' : 'text-gray-400'
  const signalQualityBg = avgBuyConf > 15 ? 'bg-emerald-500/10' : avgBuyConf >= 8 ? 'bg-amber-500/10' : 'bg-gray-500/10'

  return (
    <div>
      {/* Morning briefing */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6 mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-white text-lg font-semibold">Morning Briefing</h2>
          <button
            onClick={loadBriefing}
            disabled={briefingLoading}
            className="text-gray-500 hover:text-cyan-400 transition-colors p-1 rounded hover:bg-white/5 disabled:opacity-30 cursor-pointer"
            title="Refresh briefing"
          >
            <RefreshCw className={`w-4 h-4 ${briefingLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
        {briefingLoading ? (
          <div className="space-y-3">
            <div className="h-4 bg-gray-700/50 rounded w-full animate-pulse" />
            <div className="h-4 bg-gray-700/50 rounded w-4/5 animate-pulse" />
          </div>
        ) : briefingError ? (
          <div className="text-gray-500 text-sm">
            <p>Couldn't load briefing. Is the LLM configured?</p>
            <button onClick={loadBriefing} className="text-cyan-400 text-xs mt-1 hover:underline cursor-pointer">Retry</button>
          </div>
        ) : (
          <>
            <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">{briefing}</p>
            {briefingTime && (
              <p className="text-gray-600 text-xs mt-3">Generated at {briefingTime}</p>
            )}
          </>
        )}
      </div>

      {/* Hints panel */}
      {hints.length > 0 && <HintsPanel hints={hints} />}

      {/* Portfolio quick-view */}
      {hasHoldings && comparison?.has_data && (
        <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Portfolio Quick View</h3>
            <a href="/my-portfolio" className="text-xs text-cyan-400 hover:underline">View details</a>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <div className="text-xs text-gray-500">Signal Value</div>
              <div className="text-lg font-bold text-white">{fmtCurrency(comparison.signal_value ?? 0)}</div>
              <div className={`text-xs ${(comparison.signal_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(comparison.signal_return_pct ?? 0) >= 0 ? '+' : ''}{(comparison.signal_return_pct ?? 0).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Buy & Hold</div>
              <div className="text-lg font-bold text-white">{fmtCurrency(comparison.buy_hold_value ?? 0)}</div>
              <div className={`text-xs ${(comparison.buy_hold_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(comparison.buy_hold_return_pct ?? 0) >= 0 ? '+' : ''}{(comparison.buy_hold_return_pct ?? 0).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Win Rate</div>
              <div className="text-lg font-bold text-white">{(comparison.overall_win_rate_pct ?? 0).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Verdict</div>
              <div className={`text-lg font-bold ${
                comparison.verdict === 'signals_win' ? 'text-green-400' :
                comparison.verdict === 'buy_hold_wins' ? 'text-red-400' : 'text-amber-400'
              }`}>
                {comparison.verdict === 'signals_win' ? 'Signals Win' :
                 comparison.verdict === 'buy_hold_wins' ? 'B&H Wins' : 'Even'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
        <SummaryCard icon={Activity} label="Assets Monitored" value={signals.length} color="text-cyan-400" bg="bg-cyan-500/10" loading={loading} />
        <SummaryCard icon={TrendingUp} label="BUY Signals" value={buys.length} color="text-emerald-400" bg="bg-emerald-500/10" loading={loading} />
        <SummaryCard icon={TrendingDown} label="SELL Signals" value={sells.length} color="text-red-400" bg="bg-red-500/10" loading={loading} />
        <SummaryCard icon={Minus} label="HOLD" value={holds.length} color="text-amber-400" bg="bg-amber-500/10" loading={loading} />
        <SummaryCard icon={Gauge} label="Signal Quality" value={signalQualityLabel} color={signalQualityColor} bg={signalQualityBg} loading={loading} />
      </div>

      {/* Filter row */}
      <div className="flex gap-2 mb-6">
        {(['All', 'Stocks', 'FX', 'Crypto'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-1.5 rounded-lg text-sm transition-colors cursor-pointer ${
              filter === f
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
            }`}
          >
            {f} {f !== 'All' && `(${signals.filter(s =>
              f === 'Stocks' ? s.asset_class === 'stock' :
              f === 'FX' ? s.asset_class === 'fx' :
              s.asset_class === 'crypto'
            ).length})`}
          </button>
        ))}
      </div>

      {/* Signal cards grid */}
      {signalsError ? (
        <div className="text-gray-500 p-8 text-center">
          <p>Couldn't load signals. Is the server running?</p>
          <button onClick={() => window.location.reload()} className="text-cyan-400 text-xs mt-2 hover:underline cursor-pointer">Retry</button>
        </div>
      ) : loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6, 7, 8].map(i => (
            <div key={i} className="bg-[#111827] border border-[#1f2937] rounded-lg p-4 space-y-3">
              <div className="h-5 bg-gray-700/50 rounded w-24 animate-pulse" />
              <div className="h-4 bg-gray-700/50 rounded w-full animate-pulse" />
              <div className="h-4 bg-gray-700/50 rounded w-3/4 animate-pulse" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(s => <DashboardCard key={s.asset} signal={s} />)}
        </div>
      )}
    </div>
  )
}

// ── Dashboard Signal Card ──

const borderColors: Record<string, string> = {
  BUY: 'border-l-emerald-500',
  SHORT: 'border-l-orange-500',
  SELL: 'border-l-red-500',
  HOLD: 'border-l-amber-500',
}

const pillColors: Record<string, string> = {
  BUY: 'text-emerald-400 bg-emerald-500/15',
  SHORT: 'text-orange-400 bg-orange-500/15',
  SELL: 'text-red-400 bg-red-500/15',
  HOLD: 'text-amber-400 bg-amber-500/15',
}

function ConfidenceArc({ confidence, signal }: { confidence: number; signal: string }) {
  const radius = 20
  const circumference = 2 * Math.PI * radius
  const pct = Math.min(Math.max(confidence, 0), 100)
  const offset = circumference - (pct / 100) * circumference
  const color = signal === 'BUY' ? '#10b981' : signal === 'SHORT' ? '#f97316' : signal === 'SELL' ? '#ef4444' : '#f59e0b'

  return (
    <svg width="50" height="50" viewBox="0 0 50 50" className="flex-shrink-0">
      <circle cx="25" cy="25" r={radius} fill="none" stroke="#1f2937" strokeWidth="4" />
      <circle
        cx="25" cy="25" r={radius} fill="none"
        stroke={color} strokeWidth="4" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={offset}
        transform="rotate(-90 25 25)"
      />
      <text x="25" y="25" textAnchor="middle" dominantBaseline="central"
        fill="#e5e7eb" fontSize="10" fontWeight="bold">
        {Math.round(pct)}
      </text>
    </svg>
  )
}

function DashboardCard({ signal: s }: { signal: EnrichedSignal }) {
  const [expanded, setExpanded] = useState(false)
  const plainReason = translateSignalSummary(s.reason, s.signal, s.asset)
  const conf = confidenceLabel(s.technical.confidence)

  // Calculate price change from technical data if available
  const priceChangeStr = s.technical.rsi > 0
    ? `RSI ${s.technical.rsi.toFixed(0)}`
    : ''

  return (
    <div
      className={`bg-[#111827] border border-[#1f2937] border-l-4 ${borderColors[s.signal] || 'border-l-gray-500'} rounded-lg p-4 cursor-pointer hover:border-[#374151] transition-colors`}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Top row: symbol + pill */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-white font-semibold">{s.asset}</div>
          <div className="text-gray-500 text-xs uppercase">{s.asset_class}</div>
        </div>
        <span className={`px-2.5 py-0.5 rounded text-xs font-bold ${pillColors[s.signal] || 'text-gray-400 bg-gray-500/15'}`}>
          {s.signal}
        </span>
      </div>

      {/* Middle: signal strength arc + price */}
      <div className="flex items-center gap-3 mb-2">
        <ConfidenceArc confidence={s.technical.confidence * 10} signal={s.signal} />
        <div className="flex-1 min-w-0">
          <div className="text-white font-mono text-lg">${s.price.toFixed(2)}</div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2 text-xs">
              <span className={conf.color}>{conf.text}: {s.technical.confidence.toFixed(1)}%</span>
              {priceChangeStr && <span className="text-gray-500">{priceChangeStr}</span>}
            </div>
            <span className="text-gray-600 text-[10px]">Model agreement on direction</span>
          </div>
        </div>
      </div>

      {/* Signal explanation */}
      {s.explanation && (
        <p className="text-gray-400 text-xs mb-2 font-mono leading-relaxed">{s.explanation}</p>
      )}

      {/* Expand indicator */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>{s.technical.quality} quality</span>
        {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </div>

      {/* Expanded model votes */}
      {expanded && (
        <div className="mt-3 pt-3 border-t border-[#1f2937]">
          <div className="text-gray-400 text-xs uppercase tracking-wider mb-2">Model Votes</div>
          <div className="space-y-1.5">
            {Object.entries(s.models).map(([name, model]) => {
              const cv = convictionInfo(model.probability_up)
              return (
                <div key={name} className="flex items-center gap-2 bg-[#0a0e17] rounded px-2 py-1.5">
                  <span className="text-gray-500 text-[10px] uppercase w-12 flex-shrink-0 font-medium">{name}</span>
                  <span className="text-white font-mono text-[11px] w-10 flex-shrink-0">{model.probability_up.toFixed(1)}%</span>
                  <span className={`text-[10px] font-bold w-14 flex-shrink-0 ${cv.textColor}`}>
                    {cv.direction === 'UP' ? '\u2191' : '\u2193'} {cv.direction}
                  </span>
                  <span className="inline-flex gap-px flex-shrink-0">
                    {Array.from({ length: 10 }, (_, i) => (
                      <span key={i} className={`w-1 h-2 rounded-sm ${i < cv.filledBars ? cv.barColor : 'bg-[#1f2937]'}`} />
                    ))}
                  </span>
                  <span className="text-gray-500 text-[10px]">{cv.label}</span>
                </div>
              )
            })}
          </div>
          <p className="mt-1.5 text-[10px] text-gray-600 leading-relaxed">
            Percentages show probability of price going UP. Below 50% = bearish. Further from 50% = stronger conviction.
          </p>
          <div className="mt-2 text-xs text-gray-500 space-y-1">
            <p>RSI: {s.technical.rsi.toFixed(1)} | Trend: {s.technical.trend}</p>
            <p className="text-gray-400">{plainReason}</p>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Hints Panel ──

function HintsPanel({ hints }: { hints: Hint[] }) {
  const [expanded, setExpanded] = useState<number | null>(null)
  const urgencyColor: Record<string, string> = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#3b82f6',
  }

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6 mb-6">
      <h3 className="text-white font-semibold mb-4">Things worth knowing today</h3>
      <div className="space-y-3">
        {hints.map((hint, i) => (
          <div key={i} className="bg-[#0a0e17] rounded-lg overflow-hidden flex">
            <div className="w-1 flex-shrink-0" style={{ background: urgencyColor[hint.urgency] || '#6b7280' }} />
            <div className="p-4 flex-1">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h4 className="text-white text-sm font-medium">{hint.title}</h4>
                  <p className="text-gray-400 text-sm mt-1">{hint.reason}</p>
                </div>
                {hint.suggested_pct > 0 && (
                  <span className="text-xs px-2 py-1 rounded-full bg-white/5 text-gray-400 whitespace-nowrap flex-shrink-0">
                    ~{hint.suggested_pct}% reduction
                  </span>
                )}
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setExpanded(expanded === i ? null : i) }}
                className="text-cyan-400 text-xs mt-2 hover:underline cursor-pointer"
              >
                {expanded === i ? 'Hide detail' : 'What this means'}
              </button>
              {expanded === i && (
                <p className="text-gray-500 text-xs mt-2 leading-relaxed">{hint.what_it_means}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Summary Card ──

function SummaryCard({ icon: Icon, label, value, color, bg, loading }: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: number | string
  color: string
  bg: string
  loading: boolean
}) {
  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-lg ${bg} flex items-center justify-center`}>
          <Icon className={`w-5 h-5 ${color}`} />
        </div>
        <div>
          <div className="text-gray-500 text-xs">{label}</div>
          <div className="text-white text-2xl font-bold">{loading ? '-' : value}</div>
        </div>
      </div>
    </div>
  )
}

function fmtCurrency(val: number): string {
  return val.toLocaleString(undefined, { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 })
}
