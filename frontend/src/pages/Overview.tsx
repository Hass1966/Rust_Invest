import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react'
import { fetchSignals } from '../lib/api'
import type { EnrichedSignal } from '../lib/types'

export default function Overview() {
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchSignals()
      .then(setSignals)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const buys = signals.filter(s => s.signal === 'BUY')
  const sells = signals.filter(s => s.signal === 'SELL')
  const holds = signals.filter(s => s.signal === 'HOLD')

  return (
    <div>
      {/* Morning briefing */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6 mb-6">
        <h2 className="text-white text-lg font-semibold mb-2">Morning Briefing</h2>
        <p className="text-gray-400 text-sm leading-relaxed">
          Market overview and AI-generated briefing will appear here. The system monitors {signals.length} assets
          across stocks, FX, and crypto, running inference through a 3-model ensemble (Linear Regression,
          Logistic Regression, Gradient Boosted Trees) trained with walk-forward evaluation on 83+ features.
        </p>
        <p className="text-gray-500 text-xs mt-3 italic">
          LLM-powered briefings coming in Phase 6. Use the Chat panel for interactive analysis.
        </p>
      </div>

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
