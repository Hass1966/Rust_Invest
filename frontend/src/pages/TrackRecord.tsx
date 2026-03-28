import { useEffect, useState, useCallback, useMemo } from 'react'
import { fetchSignalTruth, forceResolveSignals } from '../lib/api'
import type { SignalTruthData, SignalTruthRecord } from '../lib/api'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, LineChart, Line, Cell,
} from 'recharts'

export default function TrackRecord() {
  const [data, setData] = useState<SignalTruthData | null>(null)
  const [loading, setLoading] = useState(true)
  const [resolving, setResolving] = useState(false)
  const [filter, setFilter] = useState<string>('all')
  const [classFilter, setClassFilter] = useState<string>('all')
  const [signalFilter, setSignalFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'accuracy' | 'total' | 'asset'>('accuracy')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const d = await fetchSignalTruth()
      setData(d)
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleForceResolve = async () => {
    setResolving(true)
    try {
      await forceResolveSignals()
      await load()
    } catch { /* ignore */ }
    setResolving(false)
  }

  useEffect(() => { load() }, [load])

  // Auto-refresh every hour
  useEffect(() => {
    const interval = setInterval(load, 3600_000)
    return () => clearInterval(interval)
  }, [load])

  // ── Compute rolling 7-day accuracy data from signals ──
  const rollingData = useMemo(() => {
    if (!data?.signals) return []
    const resolved = data.signals.filter(s => s.was_correct !== null)
    if (resolved.length === 0) return []

    // Group by date
    const byDate = new Map<string, { correct: number; total: number }>()
    for (const s of resolved) {
      const date = s.timestamp.slice(0, 10)
      const entry = byDate.get(date) || { correct: 0, total: 0 }
      entry.total++
      if (s.was_correct) entry.correct++
      byDate.set(date, entry)
    }

    const dates = Array.from(byDate.keys()).sort().filter(d => d >= '2026-03-15')
    if (dates.length === 0) return []

    // Compute 7-day rolling accuracy
    const allDates = Array.from(byDate.keys()).sort()
    return dates.map(date => {
      const windowStart = new Date(date)
      windowStart.setDate(windowStart.getDate() - 6)
      const startStr = windowStart.toISOString().slice(0, 10)

      let correct = 0, total = 0
      for (const d of allDates) {
        if (d >= startStr && d <= date) {
          const entry = byDate.get(d)!
          correct += entry.correct
          total += entry.total
        }
      }

      const accuracy = total > 0 ? (correct / total) * 100 : 0
      return { date, accuracy: Number(accuracy.toFixed(1)), total }
    })
  }, [data])

  if (loading && !data) {
    return <div className="text-gray-500 p-8 text-center">Loading track record...</div>
  }

  if (!data || data.total_signals === 0) {
    return (
      <div className="text-gray-500 p-8 text-center">
        <p>No signals recorded yet. Run the serve binary to start generating and tracking signals.</p>
      </div>
    )
  }

  const { rolling, by_signal_type, by_asset_class, per_asset, signals } = data
  const totalWrong = data.total_resolved - data.total_correct
  const accColor = (acc: number, resolved: number) =>
    resolved === 0 ? 'text-gray-600' : acc >= 57 ? 'text-green-400' : acc >= 50 ? 'text-amber-400' : 'text-red-400'

  // Per-asset sorted
  const sortedAssets = [...per_asset].filter(a => a.total > 0).sort((a, b) => {
    if (sortBy === 'accuracy') return b.accuracy - a.accuracy
    if (sortBy === 'total') return b.total - a.total
    return a.asset.localeCompare(b.asset)
  })

  // Best/worst assets
  const bestAssets = sortedAssets.filter(a => a.total >= 3).slice(0, 5)
  const worstAssets = [...sortedAssets].filter(a => a.total >= 3).sort((a, b) => a.accuracy - b.accuracy).slice(0, 5)

  // Bubble chart data
  const bubbleData = sortedAssets.map(a => ({
    asset: a.asset,
    accuracy: Number(a.accuracy.toFixed(1)),
    signals: a.total,
    correct: a.correct,
    incorrect: a.total - a.correct,
    fill: a.accuracy >= 70 ? '#22c55e' : a.accuracy >= 50 ? '#f59e0b' : '#ef4444',
  }))

  // Filtered signals for the table
  const filtered = signals.filter(s => {
    if (filter !== 'all' && s.asset !== filter) return false
    if (classFilter !== 'all' && s.asset_class !== classFilter) return false
    if (signalFilter !== 'all' && s.signal_type !== signalFilter) return false
    return true
  })
  const uniqueAssets = [...new Set(signals.map(s => s.asset))].sort()

  return (
    <div className="space-y-6">
      {/* ── Hero Banner ── */}
      <div className="bg-gradient-to-r from-[#0f1729] to-[#111827] rounded-2xl border border-[#1f2937] p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">Model Scorecard</h2>
            <p className="text-sm text-gray-400">
              Fully transparent. Every signal tracked — good and bad.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleForceResolve}
              disabled={resolving}
              className="px-4 py-2 rounded-lg text-sm bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 hover:bg-cyan-500/20 transition-colors cursor-pointer disabled:opacity-50"
            >
              {resolving ? 'Resolving...' : 'Resolve Pending'}
            </button>
            <button
              onClick={load}
              className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white bg-[#111827] border border-[#1f2937] transition-colors cursor-pointer"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Big accuracy number */}
        <div className="mt-6 flex flex-col sm:flex-row sm:items-end gap-6">
          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Overall Accuracy</div>
            <div className={`text-5xl font-black ${accColor(data.overall_accuracy, data.total_resolved)}`}>
              {data.total_resolved > 0 ? `${data.overall_accuracy.toFixed(1)}%` : '--'}
            </div>
          </div>
          <div className="flex-1 text-sm text-gray-400 leading-relaxed">
            {data.total_resolved > 0 ? (
              <>
                We analysed <span className="text-white font-semibold">{data.total_signals.toLocaleString()}</span> signals.{' '}
                <span className="text-green-400 font-semibold">{data.total_correct.toLocaleString()}</span> predictions were correct,{' '}
                <span className="text-red-400 font-semibold">{totalWrong.toLocaleString()}</span> were wrong.{' '}
                {data.total_pending > 0 && <span className="text-amber-400">{data.total_pending.toLocaleString()} still pending resolution.</span>}
              </>
            ) : (
              <span className="text-amber-400">All {data.total_signals.toLocaleString()} signals are awaiting resolution. Click "Resolve Pending" to score them against current prices.</span>
            )}
          </div>
        </div>
      </div>

      {/* ── Rolling 7-Day Accuracy Line Chart (Change 2) ── */}
      {rollingData.length > 1 && (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-1">Is the model improving over time?</h3>
          <p className="text-xs text-gray-600 mb-4">7-day rolling accuracy since 15 March 2026</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rollingData} margin={{ left: 0, right: 10, top: 4, bottom: 0 }}>
              <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
              <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} domain={[0, 100]} tickFormatter={v => `${v}%`} width={42} />
              <Tooltip
                contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(v: number | undefined) => [v != null ? `${v.toFixed(1)}%` : '', '7-Day Accuracy']}
              />
              <ReferenceLine y={50} stroke="#6b7280" strokeDasharray="4 4" label={{ value: 'Random baseline', fill: '#6b7280', fontSize: 10, position: 'insideBottomRight' }} />
              <ReferenceLine y={65} stroke="#22c55e" strokeDasharray="4 4" label={{ value: 'Target', fill: '#22c55e', fontSize: 10, position: 'insideTopRight' }} />
              <Line type="monotone" dataKey="accuracy" stroke="#06b6d4" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: '#06b6d4', strokeWidth: 0 }}>
                {rollingData.map((d, i) => (
                  <Cell key={i} stroke={d.accuracy >= 65 ? '#22c55e' : d.accuracy >= 50 ? '#f59e0b' : '#ef4444'} />
                ))}
              </Line>
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Rolling Accuracy Cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <RollingCard label="Today" stats={rolling.today} />
        <RollingCard label="This Week" stats={rolling.this_week} />
        <RollingCard label="All Time" stats={rolling.all_time} />
      </div>

      {/* ── By Signal Type & Asset Class ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">By Signal Type</h3>
          <div className="space-y-3">
            {by_signal_type.map(st => (
              <AccuracyBar
                key={st.signal_type}
                label={st.signal_type}
                correct={st.correct}
                total={st.total}
                accuracy={st.accuracy}
                color={st.signal_type === 'BUY' ? 'bg-green-500' : st.signal_type === 'SHORT' ? 'bg-orange-500' : st.signal_type === 'SELL' ? 'bg-red-500' : 'bg-amber-500'}
                labelColor={st.signal_type === 'BUY' ? 'text-green-400' : st.signal_type === 'SHORT' ? 'text-orange-400' : st.signal_type === 'SELL' ? 'text-red-400' : 'text-amber-400'}
              />
            ))}
          </div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">By Asset Class</h3>
          <div className="space-y-3">
            {by_asset_class.map(ac => (
              <AccuracyBar
                key={ac.asset_class}
                label={ac.asset_class}
                correct={ac.correct}
                total={ac.total}
                accuracy={ac.accuracy}
                color="bg-cyan-500"
                labelColor="text-gray-300"
              />
            ))}
          </div>
        </div>
      </div>

      {/* ── Bubble Chart: Signal Accuracy vs Evidence (Change 1) ── */}
      {bubbleData.length > 0 && (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-1">Signal Accuracy vs Evidence</h3>
          <p className="text-xs text-gray-600 mb-4">Each bubble is one asset. Size = number of signals.</p>
          <ResponsiveContainer width="100%" height={380}>
            <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
              <XAxis
                type="number"
                dataKey="accuracy"
                name="Accuracy"
                domain={[0, 100]}
                tick={{ fill: '#6b7280', fontSize: 11 }}
                label={{ value: 'Accuracy %', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="signals"
                name="Signals"
                tick={{ fill: '#6b7280', fontSize: 11 }}
                label={{ value: 'Number of Signals', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }}
              />
              <ZAxis type="number" dataKey="signals" range={[40, 400]} />
              <Tooltip
                contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                cursor={{ strokeDasharray: '3 3' }}
                content={({ payload }) => {
                  if (!payload?.length) return null
                  const d = payload[0].payload as typeof bubbleData[0]
                  return (
                    <div className="bg-[#0a0e17] border border-[#1f2937] rounded-lg p-3 text-xs">
                      <div className="text-white font-semibold mb-1">{d.asset}</div>
                      <div className="text-gray-400">Accuracy: <span className="text-white">{d.accuracy}%</span></div>
                      <div className="text-gray-400">Signals: <span className="text-white">{d.signals}</span></div>
                      <div className="text-gray-400">Correct: <span className="text-green-400">{d.correct}</span> / Incorrect: <span className="text-red-400">{d.incorrect}</span></div>
                    </div>
                  )
                }}
              />
              <ReferenceLine
                x={50}
                stroke="#6b7280"
                strokeDasharray="4 4"
                label={{ value: 'Random baseline', fill: '#6b7280', fontSize: 10, angle: -90, position: 'insideTopLeft' }}
              />
              <Scatter data={bubbleData}>
                {bubbleData.map((d, i) => (
                  <Cell key={i} fill={d.fill} fillOpacity={0.7} stroke={d.fill} strokeWidth={1} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex items-center justify-center gap-4 mt-2 text-xs text-gray-500">
            <span><span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-1" />Above 70%</span>
            <span><span className="inline-block w-3 h-3 rounded-full bg-amber-500 mr-1" />50-70%</span>
            <span><span className="inline-block w-3 h-3 rounded-full bg-red-500 mr-1" />Below 50%</span>
          </div>
        </div>
      )}

      {/* ── Best & Worst Assets ── */}
      {(bestAssets.length > 0 || worstAssets.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {bestAssets.length > 0 && (
            <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
              <h3 className="text-sm font-medium text-green-400 mb-3">Best Performing Assets</h3>
              <div className="space-y-2">
                {bestAssets.map((a, i) => (
                  <div key={a.asset} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-600 w-4">{i + 1}.</span>
                      <span className="text-sm font-medium text-white">{a.asset}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-24 bg-[#0a0e17] rounded-full h-1.5">
                        <div className="h-1.5 rounded-full bg-green-500" style={{ width: `${Math.min(a.accuracy, 100)}%` }} />
                      </div>
                      <span className="text-sm text-green-400 font-semibold w-14 text-right">{a.accuracy.toFixed(1)}%</span>
                      <span className="text-xs text-gray-600 w-10 text-right">{a.correct}/{a.total}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {worstAssets.length > 0 && (
            <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
              <h3 className="text-sm font-medium text-red-400 mb-3">Weakest Assets</h3>
              <div className="space-y-2">
                {worstAssets.map((a, i) => (
                  <div key={a.asset} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-600 w-4">{i + 1}.</span>
                      <span className="text-sm font-medium text-white">{a.asset}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-24 bg-[#0a0e17] rounded-full h-1.5">
                        <div className={`h-1.5 rounded-full ${a.accuracy >= 50 ? 'bg-amber-500' : 'bg-red-500'}`} style={{ width: `${Math.min(a.accuracy, 100)}%` }} />
                      </div>
                      <span className={`text-sm font-semibold w-14 text-right ${a.accuracy >= 50 ? 'text-amber-400' : 'text-red-400'}`}>{a.accuracy.toFixed(1)}%</span>
                      <span className="text-xs text-gray-600 w-10 text-right">{a.correct}/{a.total}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Full Per-Asset Accuracy Grid ── */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-400">All Assets ({sortedAssets.length})</h3>
          <div className="flex gap-1">
            {(['accuracy', 'total', 'asset'] as const).map(s => (
              <button
                key={s}
                onClick={() => setSortBy(s)}
                className={`px-2 py-1 text-xs rounded cursor-pointer ${sortBy === s ? 'text-cyan-400 bg-cyan-500/10' : 'text-gray-500 hover:text-gray-300'}`}
              >
                {s === 'accuracy' ? 'Best' : s === 'total' ? 'Most signals' : 'A-Z'}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 max-h-80 overflow-y-auto">
          {sortedAssets.map(a => (
            <div key={a.asset} className="bg-[#0a0e17] rounded-lg p-3 text-center">
              <div className="text-xs text-gray-500 truncate">{a.asset}</div>
              <div className={`text-lg font-bold ${accColor(a.accuracy, a.total)}`}>
                {a.accuracy.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600">{a.correct}/{a.total}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Signal History Table ── */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <h3 className="text-sm font-medium text-gray-400">Recent Signals</h3>
          <div className="flex flex-wrap items-center gap-2">
            <select value={classFilter} onChange={e => setClassFilter(e.target.value)} className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300">
              <option value="all">All classes</option>
              <option value="stock">Stocks</option>
              <option value="fx">FX</option>
              <option value="crypto">Crypto</option>
            </select>
            <select value={signalFilter} onChange={e => setSignalFilter(e.target.value)} className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300">
              <option value="all">All signals</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
            </select>
            <select value={filter} onChange={e => setFilter(e.target.value)} className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300">
              <option value="all">All assets</option>
              {uniqueAssets.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
        </div>
        <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-[#111827] z-10">
              <tr className="text-gray-500 border-b border-[#1f2937]">
                <th className="text-left py-2 px-2">Time</th>
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-left py-2 px-2">Class</th>
                <th className="text-left py-2 px-2">Signal</th>
                <th className="text-right py-2 px-2">Entry Price</th>
                <th className="text-right py-2 px-2">Outcome</th>
                <th className="text-right py-2 px-2">Change</th>
                <th className="text-center py-2 px-2">Result</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 300).map(s => <SignalRow key={s.id} signal={s} />)}
            </tbody>
          </table>
        </div>
        {filtered.length > 300 && <p className="text-xs text-gray-600 mt-2 text-center">Showing 300 of {filtered.length}</p>}
        {filtered.length === 0 && <p className="text-gray-600 text-center py-8">No signals match the current filters.</p>}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════
// Sub-components
// ═══════════════════════════════════════

function RollingCard({ label, stats }: { label: string; stats: { resolved: number; correct: number; accuracy: number } }) {
  const wrong = stats.resolved - stats.correct
  const color = stats.resolved === 0 ? 'text-gray-600' : stats.accuracy >= 57 ? 'text-green-400' : stats.accuracy >= 50 ? 'text-amber-400' : 'text-red-400'

  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-3xl font-bold ${color}`}>
        {stats.resolved > 0 ? `${stats.accuracy.toFixed(1)}%` : '--'}
      </div>
      {stats.resolved > 0 && (
        <div className="mt-2 flex items-center gap-3 text-xs">
          <span className="text-green-400">{stats.correct} correct</span>
          <span className="text-red-400">{wrong} wrong</span>
          <span className="text-gray-600">{stats.resolved} total</span>
        </div>
      )}
      {stats.resolved > 0 && (
        <div className="mt-2 w-full bg-[#0a0e17] rounded-full h-1.5">
          <div
            className={`h-1.5 rounded-full ${stats.accuracy >= 50 ? 'bg-green-500' : 'bg-red-500'}`}
            style={{ width: `${Math.min(stats.accuracy, 100)}%` }}
          />
        </div>
      )}
    </div>
  )
}

function AccuracyBar({ label, correct, total, accuracy, color, labelColor }: {
  label: string; correct: number; total: number; accuracy: number; color: string; labelColor: string
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className={`text-sm font-medium w-14 capitalize ${labelColor}`}>{label}</span>
        <div className="w-32 bg-[#0a0e17] rounded-full h-2">
          <div className={`h-2 rounded-full ${total > 0 && accuracy < 50 ? 'bg-red-500' : color}`} style={{ width: `${total > 0 ? Math.min(accuracy, 100) : 0}%` }} />
        </div>
      </div>
      <span className="text-sm text-gray-400">
        {total > 0 ? `${accuracy.toFixed(1)}%` : '--'}{' '}
        <span className="text-gray-600">({correct}/{total})</span>
      </span>
    </div>
  )
}

function SignalRow({ signal: s }: { signal: SignalTruthRecord }) {
  const time = new Date(s.timestamp).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  const signalColor = s.signal_type === 'BUY' ? 'text-green-400' : s.signal_type === 'SHORT' ? 'text-orange-400' : s.signal_type === 'SELL' ? 'text-red-400' : 'text-amber-400'

  let resultIcon: React.ReactNode
  if (s.was_correct === null) resultIcon = <span className="text-amber-400/60 text-xs">pending</span>
  else if (s.was_correct) resultIcon = <span className="text-green-400 font-bold">&#10003;</span>
  else resultIcon = <span className="text-red-400 font-bold">&#10007;</span>

  const pctStr = s.pct_change != null ? `${s.pct_change >= 0 ? '+' : ''}${s.pct_change.toFixed(2)}%` : '--'
  const pctColor = s.pct_change == null ? 'text-gray-600' : s.pct_change > 0 ? 'text-green-400' : s.pct_change < 0 ? 'text-red-400' : 'text-gray-400'

  return (
    <tr className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
      <td className="py-1.5 px-2 text-gray-400 text-xs whitespace-nowrap">{time}</td>
      <td className="py-1.5 px-2 text-gray-300 font-medium">{s.asset}</td>
      <td className="py-1.5 px-2 text-gray-500 text-xs capitalize">{s.asset_class}</td>
      <td className={`py-1.5 px-2 font-medium ${signalColor}`}>{s.signal_type}</td>
      <td className="py-1.5 px-2 text-right text-gray-400">{formatPrice(s.price_at_signal)}</td>
      <td className="py-1.5 px-2 text-right text-gray-400">
        {s.outcome_price != null ? formatPrice(s.outcome_price) : <span className="text-amber-400/40">--</span>}
      </td>
      <td className={`py-1.5 px-2 text-right ${pctColor}`}>{pctStr}</td>
      <td className="py-1.5 px-2 text-center">{resultIcon}</td>
    </tr>
  )
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toFixed(0)
  if (price >= 1) return price.toFixed(2)
  if (price >= 0.01) return price.toFixed(4)
  return price.toFixed(6)
}
