import { useEffect, useState } from 'react'
import { fetchTrainingResults, fetchModels } from '../lib/api'
import type { TrainingResults, ModelManifest } from '../lib/types'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, Cell } from 'recharts'

const MODEL_COLORS: Record<string, string> = {
  linreg: '#60a5fa',
  logreg: '#a78bfa',
  gbt: '#34d399',
  lstm: '#fbbf24',
  regime: '#f472b6',
  tft: '#fb923c',
}

const MODEL_LABELS: Record<string, string> = {
  linreg: 'LinReg',
  logreg: 'LogReg',
  gbt: 'GBT',
  lstm: 'LSTM',
  regime: 'Regime',
  tft: 'TFT',
}

export default function Training() {
  const [results, setResults] = useState<TrainingResults | null>(null)
  const [manifest, setManifest] = useState<ModelManifest | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetchTrainingResults().catch(() => null),
      fetchModels().catch(() => null),
    ]).then(([r, m]) => {
      setResults(r)
      setManifest(m)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-500 p-8">Loading training data...</div>

  if (!results || !results.assets || Object.keys(results.assets).length === 0) {
    return (
      <div className="text-gray-500 p-8 text-center">
        <p className="text-lg mb-2">No training results found</p>
        <p className="text-sm">Run <code className="bg-gray-800 px-2 py-1 rounded text-cyan-400">cargo run --release --bin train</code> to train all 6 models.</p>
      </div>
    )
  }

  const assets = Object.entries(results.assets).sort(([a], [b]) => a.localeCompare(b))
  const models = ['linreg', 'logreg', 'gbt', 'lstm', 'regime', 'tft'] as const

  // Compute averages
  const avgByModel: Record<string, number> = {}
  for (const m of models) {
    const vals = assets.map(([, a]) => a[m]).filter(v => v > 0)
    avgByModel[m] = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
  }

  // Best model per asset
  const bestModel = (a: Record<string, number>): string => {
    let best = 'linreg'
    let bestVal = 0
    for (const m of models) {
      if ((a[m] ?? 0) > bestVal) { bestVal = a[m]; best = m }
    }
    return best
  }

  // Chart data: average accuracy per model
  const avgChartData = models.map(m => ({
    model: MODEL_LABELS[m],
    accuracy: Number(avgByModel[m].toFixed(1)),
    fill: MODEL_COLORS[m],
  }))

  // Per-asset chart data for the heatmap-style view
  const assetChartData = assets.map(([symbol, a]) => ({
    asset: symbol.replace('=X', ''),
    ...Object.fromEntries(models.map(m => [m, Number((a[m] ?? 0).toFixed(1))])),
  }))

  const stockAssets = assets.filter(([s]) => !s.includes('=X') && !['bitcoin', 'ethereum', 'solana'].includes(s))
  const fxAssets = assets.filter(([s]) => s.includes('=X'))
  const cryptoAssets = assets.filter(([s]) => ['bitcoin', 'ethereum', 'solana'].includes(s))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-white text-xl font-semibold">Training Dashboard</h2>
        <p className="text-gray-500 text-sm">
          Version {results.version} &middot; {results.date} &middot; {results.features} active features &middot; 6 models
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard label="Assets Trained" value={String(assets.length)} sub={`${stockAssets.length} stocks, ${fxAssets.length} FX, ${cryptoAssets.length} crypto`} />
        <SummaryCard label="Features" value={String(results.features)} sub="after pruning noise" />
        <SummaryCard label="Models" value="6" sub="LinReg, LogReg, GBT, LSTM, Regime, TFT" />
        <SummaryCard
          label="Best Avg Model"
          value={MODEL_LABELS[Object.entries(avgByModel).sort(([,a],[,b]) => b - a)[0]?.[0] ?? 'gbt']}
          sub={`${Math.max(...Object.values(avgByModel)).toFixed(1)}% avg accuracy`}
        />
      </div>

      {/* Average Accuracy Chart */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
        <h3 className="text-white font-semibold mb-4">Average Accuracy by Model</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={avgChartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <XAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis domain={[40, 80]} tick={{ fill: '#9ca3af', fontSize: 12 }} tickFormatter={v => `${v}%`} />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(v: number | undefined) => [`${v ?? 0}%`, 'Accuracy']}
            />
            <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
              {avgChartData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-Asset Grouped Bar Chart */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
        <h3 className="text-white font-semibold mb-4">Per-Asset Model Comparison</h3>
        <ResponsiveContainer width="100%" height={Math.max(300, assets.length * 35)}>
          <BarChart data={assetChartData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
            <XAxis type="number" domain={[40, 80]} tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={v => `${v}%`} />
            <YAxis type="category" dataKey="asset" tick={{ fill: '#9ca3af', fontSize: 11 }} width={70} />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(v: number | undefined, name?: string) => [`${v ?? 0}%`, MODEL_LABELS[name ?? ''] || name || '']}
            />
            <Legend formatter={(v: string) => MODEL_LABELS[v] || v} />
            {models.map(m => (
              <Bar key={m} dataKey={m} fill={MODEL_COLORS[m]} radius={[0, 2, 2, 0]} barSize={4} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Table */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg overflow-hidden">
        <h3 className="text-white font-semibold px-6 pt-4 pb-2">6-Model Accuracy Table</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
              <th className="text-left px-6 py-3">Asset</th>
              <th className="text-left px-3 py-3">Class</th>
              {models.map(m => (
                <th key={m} className="text-right px-3 py-3" style={{ color: MODEL_COLORS[m] }}>
                  {MODEL_LABELS[m]}
                </th>
              ))}
              <th className="text-right px-3 py-3">Ensemble</th>
              <th className="text-center px-6 py-3">Best</th>
            </tr>
          </thead>
          <tbody>
            {assets.map(([symbol, a]) => {
              const best = bestModel(a as unknown as Record<string, number>)
              const assetClass = symbol.includes('=X') ? 'FX' : ['bitcoin', 'ethereum', 'solana'].includes(symbol) ? 'Crypto' : 'Stock'
              return (
                <tr key={symbol} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                  <td className="px-6 py-3 text-white font-medium">{symbol}</td>
                  <td className="px-3 py-3 text-gray-500 text-xs">{assetClass}</td>
                  {models.map(m => (
                    <td key={m} className="px-3 py-3 text-right font-mono">
                      <AccCell value={a[m]} isBest={m === best} />
                    </td>
                  ))}
                  <td className="px-3 py-3 text-right font-mono">
                    <AccCell value={a.ensemble} isBest={false} />
                  </td>
                  <td className="px-6 py-3 text-center">
                    <span className="text-xs px-2 py-0.5 rounded" style={{
                      backgroundColor: MODEL_COLORS[best] + '20',
                      color: MODEL_COLORS[best],
                    }}>
                      {MODEL_LABELS[best]}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Model Status from Manifest */}
      {manifest && (
        <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
          <h3 className="text-white font-semibold mb-3">Saved Model Status</h3>
          <p className="text-gray-500 text-sm mb-4">
            Version {manifest.version} &middot; Generated {new Date(manifest.generated_at).toLocaleString()}
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(manifest.assets).sort(([a],[b]) => a.localeCompare(b)).map(([sym, a]) => (
              <div key={sym} className="bg-[#0a0e17] border border-[#1f2937] rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-white text-sm font-medium">{sym}</span>
                  {a.weights_present ? (
                    <span className="text-emerald-400 text-xs">Ready</span>
                  ) : (
                    <span className="text-red-400 text-xs">Missing</span>
                  )}
                </div>
                <div className="text-gray-500 text-xs">
                  {a.last_trained ? `Trained ${new Date(a.last_trained).toLocaleDateString()}` : 'Not trained'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SummaryCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
      <p className="text-gray-500 text-xs uppercase mb-1">{label}</p>
      <p className="text-white text-2xl font-bold">{value}</p>
      <p className="text-gray-500 text-xs mt-1">{sub}</p>
    </div>
  )
}

function AccCell({ value, isBest }: { value: number; isBest: boolean }) {
  if (!value || value === 0) return <span className="text-gray-600">-</span>
  const color = value >= 60 ? 'text-emerald-400' : value >= 55 ? 'text-amber-400' : 'text-gray-400'
  return (
    <span className={`${color} ${isBest ? 'font-bold underline decoration-dotted' : ''}`}>
      {value.toFixed(1)}%
    </span>
  )
}
