import { useEffect, useState, useCallback } from 'react'
import { TrendingUp, TrendingDown, Minus, Activity, RefreshCw } from 'lucide-react'
import { fetchSignals, fetchMorningBriefing, fetchHints } from '../lib/api'
import type { EnrichedSignal, Hint } from '../lib/types'

export default function Overview() {
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [briefing, setBriefing] = useState<string | null>(null)
  const [briefingLoading, setBriefingLoading] = useState(true)
  const [briefingTime, setBriefingTime] = useState<string | null>(null)
  const [briefingError, setBriefingError] = useState(false)
  const [hints, setHints] = useState<Hint[]>([])

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
      .catch(() => {})
      .finally(() => setLoading(false))
    loadBriefing()
    fetchHints().then(setHints).catch(() => {})
  }, [loadBriefing])

  const buys = signals.filter(s => s.signal === 'BUY')
  const sells = signals.filter(s => s.signal === 'SELL')
  const holds = signals.filter(s => s.signal === 'HOLD')

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

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <SummaryCard icon={Activity} label="Assets Monitored" value={signals.length} color="text-cyan-400" bg="bg-cyan-500/10" loading={loading} />
        <SummaryCard icon={TrendingUp} label="BUY Signals" value={buys.length} color="text-emerald-400" bg="bg-emerald-500/10" loading={loading} />
        <SummaryCard icon={TrendingDown} label="SELL Signals" value={sells.length} color="text-red-400" bg="bg-red-500/10" loading={loading} />
        <SummaryCard icon={Minus} label="HOLD" value={holds.length} color="text-amber-400" bg="bg-amber-500/10" loading={loading} />
      </div>

      {/* Signal table */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-[#1f2937]">
          <h3 className="text-white font-semibold">All Signals</h3>
        </div>
        {loading ? (
          <div className="p-8 text-center text-gray-500">Loading signals...</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
                <th className="text-left px-6 py-3">Asset</th>
                <th className="text-left px-4 py-3">Class</th>
                <th className="text-left px-4 py-3">Signal</th>
                <th className="text-right px-4 py-3">Price</th>
                <th className="text-right px-4 py-3">Confidence</th>
                <th className="text-right px-4 py-3">P(Up)</th>
                <th className="text-center px-4 py-3">Agreement</th>
                <th className="text-left px-4 py-3">Quality</th>
                <th className="text-right px-6 py-3">RSI</th>
              </tr>
            </thead>
            <tbody>
              {signals.map(s => (
                <tr key={s.asset} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                  <td className="px-6 py-3 text-white font-medium">{s.asset}</td>
                  <td className="px-4 py-3 text-gray-500 uppercase text-xs">{s.asset_class}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                      s.signal === 'BUY' ? 'text-emerald-400 bg-emerald-500/15'
                        : s.signal === 'SELL' ? 'text-red-400 bg-red-500/15'
                        : 'text-amber-400 bg-amber-500/15'
                    }`}>{s.signal}</span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-white">${s.price.toFixed(2)}</td>
                  <td className="px-4 py-3 text-right text-gray-300">{s.technical.confidence.toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right text-gray-300">{s.technical.probability_up.toFixed(1)}%</td>
                  <td className="px-4 py-3 text-center text-gray-400">{s.technical.model_agreement}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs ${s.technical.quality === 'HIGH' ? 'text-emerald-400' : s.technical.quality === 'MODERATE' ? 'text-amber-400' : 'text-red-400'}`}>
                      {s.technical.quality}
                    </span>
                  </td>
                  <td className="px-6 py-3 text-right text-gray-300">{s.technical.rsi.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

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

function SummaryCard({ icon: Icon, label, value, color, bg, loading }: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: number
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
