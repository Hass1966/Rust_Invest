import { useState, useEffect } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { EnrichedSignal } from '../lib/types'
import { translateSignalSummary, confidenceLabel, convictionInfo } from '../lib/plain-english'

const signalColors: Record<string, string> = {
  BUY: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30',
  SHORT: 'text-orange-400 bg-orange-500/15 border-orange-500/30',
  SELL: 'text-red-400 bg-red-500/15 border-red-500/30',
  HOLD: 'text-amber-400 bg-amber-500/15 border-amber-500/30',
}

const signalBorder: Record<string, string> = {
  BUY: 'border-l-emerald-500',
  SHORT: 'border-l-orange-500',
  SELL: 'border-l-red-500',
  HOLD: 'border-l-amber-500',
}

function getShortTooltip(assetClass: string, asset: string): string {
  if (assetClass === 'fx') return 'How to act: Sell the pair directly through your FX broker.'
  if (assetClass === 'crypto') return 'How to act: Use crypto futures (e.g. Binance/Bybit perpetuals) or a CFD broker.'
  if (asset === 'GLD' || asset === 'SLV') return 'How to act: Buy inverse gold ETF (GLL) or use a CFD/spread bet.'
  if (asset === 'USO' || asset === 'CPER') return 'How to act: Use a CFD or spread betting platform to short the commodity.'
  return 'How to act: Use CFDs or spread betting (UK). Stocks cannot be directly shorted in a standard ISA/SIPP.'
}

interface SentimentEntry {
  news_score: number
  reddit_mentions: number
  reddit_score: number
  combined_score: number
}

function TrafficLight({ signal }: { signal: string }) {
  const lit = (target: string) => signal === target
  return (
    <div className="flex flex-col gap-1 bg-[#0a0e17] rounded-full px-1.5 py-1.5">
      <div
        className="w-2.5 h-2.5 rounded-full transition-all"
        style={{
          background: lit('SELL') ? '#ef4444' : lit('SHORT') ? '#f97316' : '#1f2937',
          boxShadow: lit('SELL') ? '0 0 6px #ef4444' : lit('SHORT') ? '0 0 6px #f97316' : 'none',
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

function SentimentBar({ score, label }: { score: number; label: string }) {
  // score is -1 to 1, map to 0-10 for bar display
  const filled = Math.round((score + 1) * 5)
  const barColor = score > 0.1 ? 'bg-emerald-500' : score < -0.1 ? 'bg-red-500' : 'bg-gray-500'
  const textColor = score > 0.1 ? 'text-emerald-400' : score < -0.1 ? 'text-red-400' : 'text-gray-400'

  return (
    <span className="inline-flex items-center gap-1.5 text-xs">
      <span className="text-gray-500">{label}:</span>
      <span className="inline-flex gap-px">
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className={`w-1 h-2.5 rounded-sm ${i < filled ? barColor : 'bg-[#1f2937]'}`} />
        ))}
      </span>
      <span className={textColor}>{score > 0 ? '+' : ''}{score.toFixed(2)}</span>
    </span>
  )
}

export default function SignalCard({ signal }: { signal: EnrichedSignal }) {
  const [expanded, setExpanded] = useState(false)
  const [sentiment, setSentiment] = useState<SentimentEntry | null>(null)
  const s = signal
  const plainReason = translateSignalSummary(s.reason, s.signal, s.asset)
  const conf = confidenceLabel(s.technical.confidence)

  useEffect(() => {
    fetch(`/api/v1/sentiment/${encodeURIComponent(s.asset)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.data?.length > 0) {
          setSentiment(data.data[0])
        }
      })
      .catch(() => {})
  }, [s.asset])

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

      {/* Price + signal strength */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm mb-2">
        <span className="text-gray-400">Price: <span className="text-white font-mono">${s.price.toFixed(2)}</span></span>
        <span className="inline-flex flex-col">
          <span className={conf.color}>Signal Strength: {s.technical.confidence.toFixed(1)}%</span>
          <span className="text-gray-600 text-[10px]">Model agreement on direction</span>
        </span>
        <span className="text-gray-400">RSI: <span className="text-white">{s.technical.rsi.toFixed(1)}</span></span>
        <span className={`text-xs px-2 py-0.5 rounded ${s.technical.quality === 'HIGH' ? 'bg-emerald-500/15 text-emerald-400' : s.technical.quality === 'MODERATE' ? 'bg-amber-500/15 text-amber-400' : 'bg-red-500/15 text-red-400'}`}>
          {s.technical.quality}
        </span>
      </div>

      {/* Signal explanation */}
      {s.explanation && (
        <p className="text-gray-300 text-xs mb-2 font-mono">{s.explanation}</p>
      )}

      {/* Plain English reason */}
      <p className="text-gray-400 text-sm mb-2">{plainReason}</p>

      {/* Sentiment indicator */}
      {sentiment && (
        <div className="flex flex-wrap gap-4 text-xs mb-2">
          <SentimentBar score={sentiment.news_score} label="News" />
          <span className="inline-flex items-center gap-1 text-xs">
            <span className="text-gray-500">Reddit:</span>
            <span className={sentiment.reddit_score > 0.1 ? 'text-emerald-400' : sentiment.reddit_score < -0.1 ? 'text-red-400' : 'text-gray-400'}>
              {sentiment.reddit_score > 0.1 ? 'bullish' : sentiment.reddit_score < -0.1 ? 'bearish' : 'neutral'}
            </span>
            <span className="text-gray-600">({sentiment.reddit_mentions} mentions)</span>
          </span>
        </div>
      )}

      {/* Risk context */}
      <div className="flex gap-3 text-xs text-gray-500">
        <span>Vol: {s.risk_context.volatility_regime}</span>
        <span>Drawdown: {s.risk_context.drawdown_risk}</span>
        <span>Trend: {s.risk_context.trend_strength}</span>
      </div>

      {/* Suggested action */}
      <p className="text-cyan-400/80 text-sm mt-2 italic">{s.suggested_action}</p>

      {/* SHORT signal tooltip and risk disclaimer */}
      {s.signal === 'SHORT' && (
        <div className="mt-2 space-y-1">
          <p className="text-orange-400/90 text-xs bg-orange-500/10 border border-orange-500/20 rounded px-3 py-2">
            {getShortTooltip(s.asset_class, s.asset)}
          </p>
          <p className="text-red-400/80 text-xs italic">
            Short positions carry higher risk. Not financial advice.
          </p>
        </div>
      )}

      {/* Expanded model votes */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-[#1f2937]">
          <h4 className="text-gray-400 text-xs uppercase tracking-wider mb-3">Model Votes</h4>
          <div className="space-y-2">
            {Object.entries(s.models).map(([name, model]) => {
              const cv = convictionInfo(model.probability_up)
              return (
                <div key={name} className="flex items-center gap-3 bg-[#0a0e17] rounded px-3 py-2">
                  <span className="text-gray-500 text-xs uppercase w-16 flex-shrink-0 font-medium">{name}</span>
                  <span className="text-white font-mono text-xs w-12 flex-shrink-0">{model.probability_up.toFixed(1)}%</span>
                  <span className={`text-xs font-bold w-16 flex-shrink-0 ${cv.textColor}`}>
                    {cv.direction === 'UP' ? '\u2191' : '\u2193'} {cv.direction}
                  </span>
                  <span className="inline-flex gap-px flex-shrink-0">
                    {Array.from({ length: 10 }, (_, i) => (
                      <span key={i} className={`w-1.5 h-2.5 rounded-sm ${i < cv.filledBars ? cv.barColor : 'bg-[#1f2937]'}`} />
                    ))}
                  </span>
                  <span className="text-gray-500 text-xs">{cv.label}</span>
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-[10px] text-gray-600 leading-relaxed">
            Percentages show probability of price going UP. Below 50% = bearish. Further from 50% = stronger conviction.
          </p>
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
