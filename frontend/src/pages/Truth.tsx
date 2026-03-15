import { useEffect, useState, useCallback } from 'react'
import { fetchSignalTruth, submitSignalFeedback } from '../lib/api'
import type { SignalTruthData, SignalTruthRecord } from '../lib/api'

export default function Truth() {
  const [data, setData] = useState<SignalTruthData | null>(null)
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

  useEffect(() => { load() }, [load])

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(load, 60_000)
    return () => clearInterval(interval)
  }, [load])

  if (loading && !data) return <div className="text-gray-500 p-8">Loading signal truth...</div>

  if (!data || data.total_signals === 0) {
    return (
      <div className="text-gray-500 p-8 text-center">
        <p className="text-lg mb-2">No signal history yet</p>
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
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">Signal Truth</h2>
        <p className="text-sm text-gray-500 mt-1">
          Fully transparent, unfiltered track record. Every signal shown &mdash; good and bad.
        </p>
      </div>

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
    </div>
  )
}

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
