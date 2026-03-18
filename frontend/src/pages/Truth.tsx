import { useEffect, useState, useCallback } from 'react'
import { fetchSignalTruth, fetchHistoricalSignalAccuracy, submitSignalFeedback } from '../lib/api'
import type { SignalTruthData, SignalTruthRecord, HistoricalSignalAccuracy } from '../lib/api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts'

export default function Truth() {
  const [data, setData] = useState<SignalTruthData | null>(null)
  const [historical, setHistorical] = useState<HistoricalSignalAccuracy | null>(null)
  const [histLoading, setHistLoading] = useState(true)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')
  const [classFilter, setClassFilter] = useState<string>('all')
  const [signalFilter, setSignalFilter] = useState<string>('all')

  const load = useCallback(() => {
    setLoading(true)
    fetchSignalTruth()
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [])

  const loadHistorical = useCallback(() => {
    setHistLoading(true)
    fetchHistoricalSignalAccuracy('weekly')
      .then(setHistorical)
      .catch(() => setHistorical(null))
      .finally(() => setHistLoading(false))
  }, [])

  useEffect(() => { load(); loadHistorical() }, [load, loadHistorical])

  // Auto-refresh live data every 60 seconds
  useEffect(() => {
    const interval = setInterval(load, 60_000)
    return () => clearInterval(interval)
  }, [load])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">Signal Truth</h2>
        <p className="text-sm text-gray-500 mt-1">
          Fully transparent, unfiltered track record. Every signal shown &mdash; good and bad.
        </p>
      </div>

      {/* Historical Backtest Section */}
      <HistoricalBacktest data={historical} loading={histLoading} onRefresh={loadHistorical} />

      {/* Divider */}
      <div className="border-t border-[#1f2937] pt-2">
        <h3 className="text-lg font-semibold text-white">Live Signal Tracker</h3>
        <p className="text-xs text-gray-500 mt-1">Forward-looking signals recorded since launch</p>
      </div>

      {/* Live tracker section (unchanged) */}
      <LiveTracker data={data} loading={loading} load={load}
        filter={filter} setFilter={setFilter}
        classFilter={classFilter} setClassFilter={setClassFilter}
        signalFilter={signalFilter} setSignalFilter={setSignalFilter}
      />
    </div>
  )
}

// ═══════════════════════════════════════
// Historical Backtest Section
// ═══════════════════════════════════════

function HistoricalBacktest({ data, loading, onRefresh }: {
  data: HistoricalSignalAccuracy | null
  loading: boolean
  onRefresh: () => void
}) {
  if (loading && !data) {
    return (
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <div className="text-gray-500 text-center">Loading historical backtest...</div>
      </div>
    )
  }

  if (!data || !data.has_data) {
    return (
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <h3 className="text-lg font-semibold text-white mb-2">Historical Signal Accuracy</h3>
        <p className="text-gray-500 text-sm">
          {data?.note || 'No holdings in portfolio. Add holdings to see historical signal accuracy.'}
        </p>
      </div>
    )
  }

  const accColor = (acc: number) =>
    acc >= 55 ? 'text-green-400' : acc >= 50 ? 'text-amber-400' : 'text-red-400'

  const buyStats = data.by_signal_type.find(s => s.signal_type === 'BUY')
  const sellStats = data.by_signal_type.find(s => s.signal_type === 'SELL')
  const holdStats = data.by_signal_type.find(s => s.signal_type === 'HOLD')

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">Historical Signal Accuracy</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Retrospective backtest across all portfolio holdings
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-600">
            {data.generated_at ? `Updated ${new Date(data.generated_at).toLocaleString()}` : ''}
          </span>
          <button onClick={onRefresh} className="text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer">
            Refresh
          </button>
        </div>
      </div>

      {/* Overall stats */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Overall Accuracy</div>
          <div className={`text-2xl font-bold ${accColor(data.overall_accuracy)}`}>
            {data.total_resolved > 0 ? `${data.overall_accuracy.toFixed(1)}%` : '--'}
          </div>
          <div className="text-xs text-gray-600 mt-1">{data.total_correct}/{data.total_resolved - (holdStats?.total || 0)} BUY/SELL correct</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Total Signals</div>
          <div className="text-2xl font-bold text-white">{data.total_signals.toLocaleString()}</div>
          <div className="text-xs text-gray-600 mt-1">{data.total_pending} pending</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">BUY Accuracy</div>
          <div className={`text-2xl font-bold ${buyStats && buyStats.total > 0 ? accColor(buyStats.accuracy) : 'text-gray-600'}`}>
            {buyStats && buyStats.total > 0 ? `${buyStats.accuracy.toFixed(1)}%` : '--'}
          </div>
          <div className="text-xs text-gray-600 mt-1">{buyStats?.correct || 0}/{buyStats?.total || 0} correct</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">SELL Accuracy</div>
          <div className={`text-2xl font-bold ${sellStats && sellStats.total > 0 ? accColor(sellStats.accuracy) : 'text-gray-600'}`}>
            {sellStats && sellStats.total > 0 ? `${sellStats.accuracy.toFixed(1)}%` : '--'}
          </div>
          <div className="text-xs text-gray-600 mt-1">{sellStats?.correct || 0}/{sellStats?.total || 0} correct</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">HOLD Signals</div>
          <div className="text-2xl font-bold text-amber-400">{holdStats?.total_including_pending || 0}</div>
          <div className="text-xs text-gray-600 mt-1">excluded from accuracy</div>
        </div>
      </div>

      {/* Accuracy by asset class */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h4 className="text-sm font-medium text-gray-400 mb-3">Accuracy by Asset Class</h4>
          <div className="space-y-2">
            {data.by_asset_class.map(ac => (
              <div key={ac.asset_class} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium w-14 text-gray-300 capitalize">{ac.asset_class}</span>
                  <div className="w-32 bg-[#0a0e17] rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${ac.accuracy >= 50 ? 'bg-cyan-500' : 'bg-red-500'}`}
                      style={{ width: `${Math.min(ac.accuracy, 100)}%` }}
                    />
                  </div>
                </div>
                <span className="text-sm text-gray-400">
                  {ac.total > 0 ? `${ac.accuracy.toFixed(1)}%` : '--'}{' '}
                  <span className="text-gray-600">({ac.correct}/{ac.total})</span>
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Per-asset table */}
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h4 className="text-sm font-medium text-gray-400 mb-3">Per-Asset Accuracy</h4>
          <div className="overflow-y-auto max-h-48">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-[#1f2937]">
                  <th className="text-left py-1 px-1">Asset</th>
                  <th className="text-left py-1 px-1">Class</th>
                  <th className="text-right py-1 px-1">Signals</th>
                  <th className="text-right py-1 px-1">Accuracy</th>
                  <th className="text-right py-1 px-1 hidden sm:table-cell">Range</th>
                </tr>
              </thead>
              <tbody>
                {data.per_asset.map(a => (
                  <tr key={a.asset} className="border-b border-[#1f2937]/30">
                    <td className="py-1 px-1 text-gray-300 font-medium">{a.asset}</td>
                    <td className="py-1 px-1 text-gray-500 text-xs capitalize">{a.asset_class}</td>
                    <td className="py-1 px-1 text-right text-gray-400">{a.total_signals}</td>
                    <td className={`py-1 px-1 text-right font-medium ${a.total > 0 ? accColor(a.accuracy) : 'text-gray-600'}`}>
                      {a.total > 0 ? `${a.accuracy.toFixed(1)}%` : '--'}
                    </td>
                    <td className="py-1 px-1 text-right text-gray-600 text-xs hidden sm:table-cell">
                      {a.date_from.slice(0, 7)} - {a.date_to.slice(0, 7)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Monthly accuracy chart */}
      {data.monthly_accuracy.length > 0 && (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h4 className="text-sm font-medium text-gray-400 mb-3">
            Monthly Accuracy Timeline
            <span className="text-gray-600 font-normal ml-2">
              ({data.monthly_accuracy[0]?.month} to {data.monthly_accuracy[data.monthly_accuracy.length - 1]?.month})
            </span>
          </h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.monthly_accuracy.map(m => ({
              month: m.month,
              accuracy: Number(m.accuracy.toFixed(1)),
              total: m.total,
            }))}>
              <XAxis
                dataKey="month"
                tick={{ fill: '#6b7280', fontSize: 10 }}
                interval={Math.max(0, Math.floor(data.monthly_accuracy.length / 12) - 1)}
                angle={-45}
                textAnchor="end"
                height={50}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#6b7280', fontSize: 11 }}
                width={35}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value: unknown, _name: unknown, props: unknown) => {
                  const v = value as number
                  const p = props as { payload: { total: number } }
                  return [`${v}% (${p.payload.total} signals)`, 'Accuracy']
                }}
              />
              <ReferenceLine y={50} stroke="#4b5563" strokeDasharray="3 3" />
              <Bar dataKey="accuracy" radius={[2, 2, 0, 0]}>
                {data.monthly_accuracy.map((m, idx) => (
                  <Cell key={idx} fill={m.accuracy >= 55 ? '#22c55e' : m.accuracy >= 50 ? '#f59e0b' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════
// Live Signal Tracker (existing functionality)
// ═══════════════════════════════════════

function LiveTracker({ data, loading, load, filter, setFilter, classFilter, setClassFilter, signalFilter, setSignalFilter }: {
  data: SignalTruthData | null
  loading: boolean
  load: () => void
  filter: string
  setFilter: (v: string) => void
  classFilter: string
  setClassFilter: (v: string) => void
  signalFilter: string
  setSignalFilter: (v: string) => void
}) {
  if (loading && !data) return <div className="text-gray-500 p-8">Loading live signals...</div>

  if (!data || data.total_signals === 0) {
    return (
      <div className="text-gray-500 p-8 text-center">
        <p className="text-lg mb-2">No live signal history yet</p>
        <p className="text-sm">
          Signals will be recorded automatically each hourly cycle.
        </p>
      </div>
    )
  }

  const { rolling, by_signal_type, by_asset_class, signals } = data

  // Filter signals
  const filtered = signals.filter(s => {
    if (filter !== 'all' && s.asset !== filter) return false
    if (classFilter !== 'all' && s.asset_class !== classFilter) return false
    if (signalFilter !== 'all' && s.signal_type !== signalFilter) return false
    return true
  })

  const uniqueAssets = [...new Set(signals.map(s => s.asset))].sort()

  return (
    <>
      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Total Signals" value={data.total_signals.toString()} sub={`${data.total_pending} pending`} color="text-white" />
        <StatCard
          label="Overall Accuracy"
          value={data.total_resolved > 0 ? `${data.overall_accuracy.toFixed(1)}%` : '--'}
          sub={`${data.total_correct}/${data.total_resolved} correct`}
          color={data.overall_accuracy >= 55 ? 'text-green-400' : data.overall_accuracy >= 50 ? 'text-amber-400' : data.total_resolved === 0 ? 'text-gray-600' : 'text-red-400'}
        />
        <StatCard
          label="Resolved"
          value={data.total_resolved.toString()}
          sub={`${data.total_pending} still pending`}
          color="text-cyan-400"
        />
        <StatCard
          label="Pending"
          value={data.total_pending.toString()}
          sub="awaiting next cycle"
          color="text-amber-400"
        />
      </div>

      {/* Rolling accuracy */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <RollingCard label="Today" stats={rolling.today} />
        <RollingCard label="This Week" stats={rolling.this_week} />
        <RollingCard label="All Time" stats={rolling.all_time} />
      </div>

      {/* Accuracy by signal type */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Accuracy by Signal Type</h3>
          <div className="space-y-2">
            {by_signal_type.map(st => (
              <div key={st.signal_type} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium w-12 ${
                    st.signal_type === 'BUY' ? 'text-green-400' :
                    st.signal_type === 'SELL' ? 'text-red-400' : 'text-amber-400'
                  }`}>{st.signal_type}</span>
                  <div className="w-32 bg-[#0a0e17] rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${st.accuracy >= 50 ? 'bg-green-500' : 'bg-red-500'}`}
                      style={{ width: `${Math.min(st.accuracy, 100)}%` }}
                    />
                  </div>
                </div>
                <span className="text-sm text-gray-400">
                  {st.total > 0 ? `${st.accuracy.toFixed(1)}%` : '--'} <span className="text-gray-600">({st.correct}/{st.total})</span>
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Accuracy by Asset Class</h3>
          <div className="space-y-2">
            {by_asset_class.map(ac => (
              <div key={ac.asset_class} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium w-12 text-gray-300 capitalize">{ac.asset_class}</span>
                  <div className="w-32 bg-[#0a0e17] rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${ac.accuracy >= 50 ? 'bg-cyan-500' : 'bg-red-500'}`}
                      style={{ width: `${Math.min(ac.accuracy, 100)}%` }}
                    />
                  </div>
                </div>
                <span className="text-sm text-gray-400">
                  {ac.total > 0 ? `${ac.accuracy.toFixed(1)}%` : '--'} <span className="text-gray-600">({ac.correct}/{ac.total})</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Signal history table */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <h3 className="text-sm font-medium text-gray-400">Full Signal History</h3>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={classFilter}
              onChange={e => setClassFilter(e.target.value)}
              className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300"
            >
              <option value="all">All classes</option>
              <option value="stock">Stocks</option>
              <option value="fx">FX</option>
              <option value="crypto">Crypto</option>
            </select>
            <select
              value={signalFilter}
              onChange={e => setSignalFilter(e.target.value)}
              className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300"
            >
              <option value="all">All signals</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
            </select>
            <select
              value={filter}
              onChange={e => setFilter(e.target.value)}
              className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300"
            >
              <option value="all">All assets</option>
              {uniqueAssets.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
            <button onClick={load} className="text-xs text-cyan-400 hover:text-cyan-300 px-2 py-1 cursor-pointer">
              Refresh
            </button>
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
                <th className="text-right py-2 px-2">Price</th>
                <th className="text-right py-2 px-2">Outcome</th>
                <th className="text-right py-2 px-2">Change</th>
                <th className="text-center py-2 px-2">Result</th>
                <th className="text-right py-2 px-2 hidden sm:table-cell">LinR</th>
                <th className="text-right py-2 px-2 hidden sm:table-cell">LogR</th>
                <th className="text-right py-2 px-2 hidden sm:table-cell">GBT</th>
                <th className="text-center py-2 px-2"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 200).map(s => (
                <SignalRow key={s.id} signal={s} />
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length > 200 && (
          <p className="text-xs text-gray-600 mt-2 text-center">
            Showing 200 of {filtered.length} signals
          </p>
        )}
        {filtered.length === 0 && (
          <p className="text-gray-600 text-center py-8">No signals match the current filters.</p>
        )}
      </div>
    </>
  )
}

// ═══════════════════════════════════════
// Shared Components
// ═══════════════════════════════════════

function StatCard({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-xs text-gray-600 mt-1">{sub}</div>
    </div>
  )
}

function RollingCard({ label, stats }: { label: string; stats: { resolved: number; correct: number; accuracy: number } }) {
  const color = stats.resolved === 0 ? 'text-gray-600'
    : stats.accuracy >= 55 ? 'text-green-400'
    : stats.accuracy >= 50 ? 'text-amber-400'
    : 'text-red-400'

  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {stats.resolved > 0 ? `${stats.accuracy.toFixed(1)}%` : '--'}
      </div>
      <div className="text-xs text-gray-600 mt-1">
        {stats.correct}/{stats.resolved} correct
      </div>
    </div>
  )
}

function SignalRow({ signal: s }: { signal: SignalTruthRecord }) {
  const [voted, setVoted] = useState<'up' | 'down' | null>(null)
  const time = new Date(s.timestamp).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  const signalColor = s.signal_type === 'BUY' ? 'text-green-400'
    : s.signal_type === 'SELL' ? 'text-red-400'
    : 'text-amber-400'

  const classLabel = s.asset_class === 'stock' ? 'Stock'
    : s.asset_class === 'fx' ? 'FX'
    : s.asset_class === 'crypto' ? 'Crypto'
    : s.asset_class

  // Result icon
  let resultIcon: React.ReactNode
  if (s.was_correct === null) {
    resultIcon = <span className="text-amber-400 text-lg" title="Pending">&#9679;</span>
  } else if (s.was_correct) {
    resultIcon = <span className="text-green-400 text-lg" title="Correct">&#10003;</span>
  } else {
    resultIcon = <span className="text-red-400 text-lg" title="Incorrect">&#10007;</span>
  }

  const pctStr = s.pct_change != null ? `${s.pct_change >= 0 ? '+' : ''}${s.pct_change.toFixed(2)}%` : '--'
  const pctColor = s.pct_change == null ? 'text-gray-600'
    : s.pct_change > 0 ? 'text-green-400'
    : s.pct_change < 0 ? 'text-red-400'
    : 'text-gray-400'

  const handleVote = (reaction: 'up' | 'down') => {
    setVoted(reaction)
    submitSignalFeedback(s.asset, s.signal_type, reaction).catch(() => {})
  }

  return (
    <tr className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
      <td className="py-1.5 px-2 text-gray-400 text-xs whitespace-nowrap">{time}</td>
      <td className="py-1.5 px-2 text-gray-300 font-medium">{s.asset}</td>
      <td className="py-1.5 px-2 text-gray-500 text-xs">{classLabel}</td>
      <td className={`py-1.5 px-2 font-medium ${signalColor}`}>{s.signal_type}</td>
      <td className="py-1.5 px-2 text-right text-gray-400">{formatPrice(s.price_at_signal)}</td>
      <td className="py-1.5 px-2 text-right text-gray-400">
        {s.outcome_price != null ? formatPrice(s.outcome_price) : <span className="text-amber-400/60">pending</span>}
      </td>
      <td className={`py-1.5 px-2 text-right ${pctColor}`}>{pctStr}</td>
      <td className="py-1.5 px-2 text-center">{resultIcon}</td>
      <td className="py-1.5 px-2 text-right text-gray-500 hidden sm:table-cell">
        {s.linreg_prob != null ? `${s.linreg_prob.toFixed(1)}%` : '--'}
      </td>
      <td className="py-1.5 px-2 text-right text-gray-500 hidden sm:table-cell">
        {s.logreg_prob != null ? `${s.logreg_prob.toFixed(1)}%` : '--'}
      </td>
      <td className="py-1.5 px-2 text-right text-gray-500 hidden sm:table-cell">
        {s.gbt_prob != null ? `${s.gbt_prob.toFixed(1)}%` : '--'}
      </td>
      <td className="py-1.5 px-2 text-center">
        {voted ? (
          <span className={voted === 'up' ? 'text-green-400' : 'text-red-400'}>
            {voted === 'up' ? '\u{1F44D}' : '\u{1F44E}'}
          </span>
        ) : (
          <span className="inline-flex gap-1">
            <button onClick={() => handleVote('up')} className="text-gray-600 hover:text-green-400 cursor-pointer" title="Thumbs up">&#x1F44D;</button>
            <button onClick={() => handleVote('down')} className="text-gray-600 hover:text-red-400 cursor-pointer" title="Thumbs down">&#x1F44E;</button>
          </span>
        )}
      </td>
    </tr>
  )
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toFixed(0)
  if (price >= 1) return price.toFixed(2)
  if (price >= 0.01) return price.toFixed(4)
  return price.toFixed(6)
}
