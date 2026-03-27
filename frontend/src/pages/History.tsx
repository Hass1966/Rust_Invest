import { useEffect, useState } from 'react'

const BASE = ''
const LIVE_START = '2026-03-15'

// ─── Types ───

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
  const [signals, setSignals] = useState<SignalsHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState<7 | 14 | 30 | 90>(30)
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetch(`${BASE}/api/v1/history/signals?days=${timeRange}`).then(r => r.json()).catch(() => null)
      .then(s => { if (s) setSignals(s) })
      .finally(() => setLoading(false))
  }, [timeRange])

  if (loading) return <div className="text-gray-500 p-8">Loading history...</div>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-white text-xl font-semibold">Signal History</h2>
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

  // Find the index where live tracking begins
  const liveStartIdx = dates.findIndex(d => d >= LIVE_START)

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
      <p className="text-gray-600 text-xs mb-4">Click an asset row to see detailed signal timeline. Dates before 15 Mar 2026 are simulated (backtest).</p>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left text-gray-500 pb-2 pr-4 font-normal w-28">Asset</th>
              {dates.map((d, i) => {
                const isLiveBoundary = liveStartIdx >= 0 && i === liveStartIdx
                const isBacktest = d < LIVE_START
                return (
                  <th key={d} className={`pb-2 font-normal text-center relative ${isBacktest ? 'text-gray-700' : 'text-gray-500'}`} style={{ minWidth: 28 }}>
                    {isLiveBoundary && (
                      <div className="absolute -left-px top-0 bottom-0 w-0.5 bg-cyan-500" title="Live tracking begins here" />
                    )}
                    {d.slice(5)}
                  </th>
                )
              })}
              <th className="text-gray-500 pb-2 font-normal text-right pl-4">BUY%</th>
            </tr>
            {liveStartIdx >= 0 && (
              <tr>
                <th />
                {dates.map((d, i) => (
                  <th key={d} className="pb-1 text-center" style={{ minWidth: 28 }}>
                    {i === liveStartIdx && (
                      <span className="text-[8px] text-cyan-400 whitespace-nowrap">LIVE</span>
                    )}
                  </th>
                ))}
                <th />
              </tr>
            )}
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
                  {dates.map((d, i) => {
                    const entry = byDate[d]
                    const sig = entry?.signal || 'N/A'
                    const color = SIGNAL_COLORS[sig] || SIGNAL_COLORS['N/A']
                    const isBacktest = d < LIVE_START
                    const isLiveBoundary = liveStartIdx >= 0 && i === liveStartIdx
                    return (
                      <td key={d} className="py-1 text-center relative">
                        {isLiveBoundary && (
                          <div className="absolute -left-px top-0 bottom-0 w-0.5 bg-cyan-500/40" />
                        )}
                        <div
                          className={`w-5 h-5 rounded mx-auto flex items-center justify-center text-[9px] font-bold ${isBacktest ? 'opacity-40' : ''}`}
                          style={{
                            background: `${color}25`,
                            color,
                            ...(isBacktest ? { borderStyle: 'dashed', borderWidth: 1, borderColor: `${color}40` } : {}),
                          }}
                          title={`${asset} ${d}: ${sig}${isBacktest ? ' (backtest)' : ' (live)'}${entry?.price ? ` @ $${entry.price.toFixed(2)}` : ''}`}
                        >
                          {sig === 'N/A' ? '\u00b7' : sig[0]}
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
        {sorted.map((e, i) => {
          const isBacktest = e.date < LIVE_START
          return (
            <div key={i} className={`flex items-center gap-4 bg-[#0a0e17] rounded px-3 py-2 text-xs ${isBacktest ? 'opacity-50 border border-dashed border-[#1f2937]' : ''}`}>
              <span className="text-gray-500 w-20 flex-shrink-0">
                {e.date}
                {isBacktest && <span className="text-gray-700 ml-1">(bt)</span>}
              </span>
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
                <span className="text-gray-500">P(up) {fmt(e.probability_up, 1)}%</span>
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
          )
        })}
      </div>
    </div>
  )
}
