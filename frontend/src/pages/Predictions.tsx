import { useEffect, useState } from 'react'
import { fetchPredictions } from '../lib/api'
import type { PredictionsData, PredictionRecord } from '../lib/api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function Predictions() {
  const [data, setData] = useState<PredictionsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')

  const load = () => {
    setLoading(true)
    fetchPredictions()
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])
  // Auto-refresh every hour
  useEffect(() => {
    const interval = setInterval(load, 3600_000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="text-gray-500 p-8">Loading predictions...</div>

  if (!data || data.predictions.length === 0) {
    return (
      <div className="text-gray-500 p-8 text-center">
        <p className="text-lg mb-2">No predictions yet</p>
        <p className="text-sm">
          Run <code className="bg-gray-800 px-2 py-1 rounded text-cyan-400">cargo run --release --bin signal</code> to
          generate predictions using saved models (inference only, no retraining).
          Models must be trained first with <code className="bg-gray-800 px-2 py-1 rounded text-cyan-400">cargo run --release --bin train</code>.
        </p>
      </div>
    )
  }

  const { stats, per_asset, confidence_bands, predictions } = data

  const filteredPredictions = filter === 'all'
    ? predictions
    : predictions.filter(p => p.asset === filter)

  const uniqueAssets = [...new Set(predictions.map(p => p.asset))].sort()

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-white">Prediction Tracker</h2>

      {/* Rolling accuracy stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Overall" stats={{ resolved: stats.total_resolved, correct: stats.total_correct, accuracy: stats.overall_accuracy }} />
        <StatCard label="Last 24h" stats={stats.last_24h} />
        <StatCard label="Last 7 days" stats={stats.last_7d} />
        <StatCard label="Last 30 days" stats={stats.last_30d} />
      </div>

      {/* Confidence calibration chart */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Confidence Calibration</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={confidence_bands.filter(b => b.predictions > 0)}>
              <XAxis dataKey="band" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#e5e7eb' }}
                formatter={(value?: number, name?: string) => [`${(value ?? 0).toFixed(1)}%`, name === 'accuracy' ? 'Actual accuracy' : (name ?? '')]}
              />
              <Bar dataKey="accuracy" name="Actual accuracy" radius={[4, 4, 0, 0]}>
                {confidence_bands.filter(b => b.predictions > 0).map((entry, i) => (
                  <Cell key={i} fill={entry.accuracy >= 50 ? '#34d399' : '#f87171'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-gray-500 mt-2">Shows actual accuracy per confidence band. Green = above 50%, Red = below.</p>
      </div>

      {/* Per-asset breakdown */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Per-Asset Accuracy</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
          {per_asset.map(a => (
            <div key={a.asset} className="bg-[#0a0e17] rounded-lg p-3 text-center">
              <div className="text-xs text-gray-500">{a.asset}</div>
              <div className={`text-lg font-bold ${a.accuracy >= 55 ? 'text-green-400' : a.accuracy >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                {a.accuracy.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600">{a.correct}/{a.total}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Predictions table */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-400">Predictions Log</h3>
          <div className="flex items-center gap-2">
            <select
              value={filter}
              onChange={e => setFilter(e.target.value)}
              className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-300"
            >
              <option value="all">All assets</option>
              {uniqueAssets.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
            <button onClick={load} className="text-xs text-cyan-400 hover:text-cyan-300 px-2 py-1">Refresh</button>
          </div>
        </div>

        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-[#111827]">
              <tr className="text-gray-500 border-b border-[#1f2937]">
                <th className="text-left py-2 px-2">Time</th>
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-left py-2 px-2">Signal</th>
                <th className="text-right py-2 px-2">Confidence</th>
                <th className="text-right py-2 px-2">Price</th>
                <th className="text-left py-2 px-2">Actual</th>
                <th className="text-center py-2 px-2">Result</th>
              </tr>
            </thead>
            <tbody>
              {filteredPredictions.slice(0, 100).map(p => (
                <PredictionRow key={p.id} prediction={p} />
              ))}
            </tbody>
          </table>
        </div>
        {filteredPredictions.length > 100 && (
          <p className="text-xs text-gray-600 mt-2 text-center">Showing first 100 of {filteredPredictions.length}</p>
        )}
      </div>
    </div>
  )
}

function StatCard({ label, stats }: { label: string; stats: { resolved: number; correct: number; accuracy: number } }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${stats.accuracy >= 55 ? 'text-green-400' : stats.accuracy >= 50 ? 'text-yellow-400' : stats.resolved === 0 ? 'text-gray-600' : 'text-red-400'}`}>
        {stats.resolved > 0 ? `${stats.accuracy.toFixed(1)}%` : '--'}
      </div>
      <div className="text-xs text-gray-600 mt-1">
        {stats.correct}/{stats.resolved} correct
      </div>
    </div>
  )
}

function PredictionRow({ prediction: p }: { prediction: PredictionRecord }) {
  const time = new Date(p.timestamp).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  const signalColor = p.signal === 'BUY' ? 'text-green-400' : p.signal === 'SHORT' ? 'text-orange-400' : p.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'

  const resultIcon = p.was_correct === null
    ? <span className="text-gray-500" title="Pending">&#9203;</span>
    : p.was_correct
      ? <span className="text-green-400" title="Correct">&#10003;</span>
      : <span className="text-red-400" title="Wrong">&#10007;</span>

  const priceChange = p.price_at_outcome
    ? ((p.price_at_outcome - p.price_at_prediction) / p.price_at_prediction * 100).toFixed(2)
    : null

  return (
    <tr className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
      <td className="py-1.5 px-2 text-gray-400 text-xs">{time}</td>
      <td className="py-1.5 px-2 text-gray-300 font-medium">{p.asset}</td>
      <td className={`py-1.5 px-2 font-medium ${signalColor}`}>{p.signal}</td>
      <td className="py-1.5 px-2 text-right text-gray-400">{p.confidence.toFixed(1)}%</td>
      <td className="py-1.5 px-2 text-right text-gray-400">
        {p.price_at_prediction.toFixed(2)}
        {priceChange && (
          <span className={`ml-1 text-xs ${Number(priceChange) > 0 ? 'text-green-500' : 'text-red-500'}`}>
            ({priceChange}%)
          </span>
        )}
      </td>
      <td className="py-1.5 px-2 text-gray-400">{p.actual_direction || '-'}</td>
      <td className="py-1.5 px-2 text-center text-lg">{resultIcon}</td>
    </tr>
  )
}
