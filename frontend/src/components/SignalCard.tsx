import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { EnrichedSignal } from '../lib/types'
import { translateSignalSummary, confidenceLabel } from '../lib/plain-english'

const signalColors: Record<string, string> = {
  BUY: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30',
  SELL: 'text-red-400 bg-red-500/15 border-red-500/30',
  HOLD: 'text-amber-400 bg-amber-500/15 border-amber-500/30',
}

const signalBorder: Record<string, string> = {
  BUY: 'border-l-emerald-500',
  SELL: 'border-l-red-500',
  HOLD: 'border-l-amber-500',
}

function TrafficLight({ signal }: { signal: string }) {
  const lit = (target: string) => signal === target
  return (
    <div className="flex flex-col gap-1 bg-[#0a0e17] rounded-full px-1.5 py-1.5">
      <div
        className="w-2.5 h-2.5 rounded-full transition-all"
        style={{
          background: lit('SELL') ? '#ef4444' : '#1f2937',
          boxShadow: lit('SELL') ? '0 0 6px #ef4444' : 'none',
        }}
      />
      <div
        className="w-2.5 h-2.5 rounded-full transition-all"
        style={{
          background: lit('HOLD') ? '#f59e0b' : '#1f2937',
          boxShadow: lit('HOLD') ? '0 0 6px #f59e0b' : 'none',
        }}
      />
      <div
        className="w-2.5 h-2.5 rounded-full transition-all"
        style={{
          background: lit('BUY') ? '#10b981' : '#1f2937',
          boxShadow: lit('BUY') ? '0 0 6px #10b981' : 'none',
        }}
      />
    </div>
  )
}

export default function SignalCard({ signal }: { signal: EnrichedSignal }) {
  const [expanded, setExpanded] = useState(false)
  const s = signal
  const plainReason = translateSignalSummary(s.reason, s.signal, s.asset)
  const conf = confidenceLabel(s.technical.confidence)

  return (
    <div
      className={`bg-[#111827] border border-[#1f2937] border-l-4 ${signalBorder[s.signal] || 'border-l-gray-500'} rounded-lg p-4 cursor-pointer hover:border-[#374151] transition-colors`}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <TrafficLight signal={s.signal} />
          <div>
            <span className="text-white font-semibold text-lg">{s.asset}</span>
            <span className="text-gray-500 text-xs ml-2 uppercase">{s.asset_class}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded text-sm font-bold border ${signalColors[s.signal] || 'text-gray-400'}`}>
            {s.signal}
          </span>
          {expanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
        </div>
      </div>

      {/* Price + confidence */}
      <div className="flex items-center gap-4 text-sm mb-2">
        <span className="text-gray-400">Price: <span className="text-white font-mono">${s.price.toFixed(2)}</span></span>
        <span className={conf.color} title={`${s.technical.confidence.toFixed(1)}%`}>
          {conf.text}
        </span>
        <span className="text-gray-400">RSI: <span className="text-white">{s.technical.rsi.toFixed(1)}</span></span>
        <span className={`text-xs px-2 py-0.5 rounded ${s.technical.quality === 'HIGH' ? 'bg-emerald-500/15 text-emerald-400' : s.technical.quality === 'MODERATE' ? 'bg-amber-500/15 text-amber-400' : 'bg-red-500/15 text-red-400'}`}>
          {s.technical.quality}
        </span>
      </div>

      {/* Plain English reason */}
      <p className="text-gray-400 text-sm mb-2">{plainReason}</p>

      {/* Risk context */}
      <div className="flex gap-3 text-xs text-gray-500">
        <span>Vol: {s.risk_context.volatility_regime}</span>
        <span>Drawdown: {s.risk_context.drawdown_risk}</span>
        <span>Trend: {s.risk_context.trend_strength}</span>
      </div>

      {/* Suggested action */}
      <p className="text-cyan-400/80 text-sm mt-2 italic">{s.suggested_action}</p>

      {/* Expanded model breakdown */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-[#1f2937]">
          <h4 className="text-gray-400 text-xs uppercase tracking-wider mb-3">Model Breakdown</h4>
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(s.models).map(([name, model]) => (
              <div key={name} className="bg-[#0a0e17] rounded p-3">
                <div className="text-gray-500 text-xs uppercase mb-1">{name}</div>
                <div className="flex items-center justify-between">
                  <span className="text-white font-mono text-sm">{model.probability_up.toFixed(1)}%</span>
                  <span className={`text-xs font-bold ${model.vote === 'UP' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {model.vote === 'UP' ? '\u25B2' : '\u25BC'} {model.vote}
                  </span>
                </div>
                <div className="text-gray-500 text-xs mt-1">Weight: {model.weight}%</div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs text-gray-500">
            <span>Agreement: {s.technical.model_agreement}</span>
            <span className="ml-4">WF Accuracy: {s.technical.walk_forward_accuracy.toFixed(1)}%</span>
            <span className="ml-4">P(Up): {s.technical.probability_up.toFixed(1)}%</span>
            <span className="ml-4">Trend: {s.technical.trend}</span>
          </div>
        </div>
      )}
    </div>
  )
}
