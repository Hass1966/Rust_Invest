import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { fetchModels, reloadModels } from '../lib/api'
import type { ModelManifest } from '../lib/types'

export default function Diagnostics() {
  const [manifest, setManifest] = useState<ModelManifest | null>(null)
  const [loading, setLoading] = useState(true)
  const [reloading, setReloading] = useState(false)

  function load() {
    setLoading(true)
    fetchModels()
      .then(setManifest)
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  async function handleReload() {
    setReloading(true)
    try {
      await reloadModels()
      load()
    } catch { /* ignore */ }
    setReloading(false)
  }

  if (loading) return <div className="text-gray-500 p-8">Loading model data...</div>
  if (!manifest) return <div className="text-gray-500 p-8">Failed to load model manifest.</div>

  const assets = Object.entries(manifest.assets).sort(([a], [b]) => a.localeCompare(b))

  // Count models with data
  const totalReady = assets.filter(([, a]) => a.weights_present).length
  const withLstm = assets.filter(([, a]) => a.lstm_accuracy && a.lstm_accuracy > 0).length
  const withRegime = assets.filter(([, a]) => a.regime_accuracy && a.regime_accuracy > 0).length
  const withTft = assets.filter(([, a]) => a.tft_accuracy && a.tft_accuracy > 0).length

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-white text-xl font-semibold">Model Diagnostics</h2>
          <p className="text-gray-500 text-sm">
            Version {manifest.version} &middot; Generated {new Date(manifest.generated_at).toLocaleString()}
            &middot; {totalReady} assets ready
            {withLstm > 0 && ` \u00b7 ${withLstm} w/ LSTM`}
            {withRegime > 0 && ` \u00b7 ${withRegime} w/ Regime`}
            {withTft > 0 && ` \u00b7 ${withTft} w/ TFT`}
          </p>
        </div>
        <button
          onClick={handleReload}
          disabled={reloading}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/15 text-cyan-400 rounded-lg text-sm hover:bg-cyan-500/25 disabled:opacity-50 transition-colors cursor-pointer"
        >
          <RefreshCw className={`w-4 h-4 ${reloading ? 'animate-spin' : ''}`} />
          Reload Models
        </button>
      </div>

      <div className="bg-[#111827] border border-[#1f2937] rounded-lg overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
              <th className="text-left px-6 py-3">Asset</th>
              <th className="text-right px-3 py-3">LinReg</th>
              <th className="text-right px-3 py-3">LogReg</th>
              <th className="text-right px-3 py-3">GBT</th>
              <th className="text-right px-3 py-3">LSTM</th>
              <th className="text-right px-3 py-3">Regime</th>
              <th className="text-right px-3 py-3">TFT</th>
              <th className="text-right px-3 py-3">Ensemble</th>
              <th className="text-left px-3 py-3">Trained</th>
              <th className="text-center px-6 py-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {assets.map(([symbol, asset]) => (
              <tr key={symbol} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                <td className="px-6 py-3 text-white font-medium">{symbol}</td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.linreg_accuracy} />
                </td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.logreg_accuracy} />
                </td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.gbt_accuracy} />
                </td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.lstm_accuracy} />
                </td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.regime_accuracy} />
                </td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.tft_accuracy} />
                </td>
                <td className="px-3 py-3 text-right font-mono">
                  <AccuracyCell value={asset.ensemble_accuracy} />
                </td>
                <td className="px-3 py-3 text-gray-400 text-xs">
                  {asset.last_trained ? new Date(asset.last_trained).toLocaleDateString() : '-'}
                </td>
                <td className="px-6 py-3 text-center">
                  {asset.weights_present ? (
                    <span className="text-emerald-400 text-xs">Ready</span>
                  ) : (
                    <span className="text-red-400 text-xs">Missing</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function AccuracyCell({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined || value === 0) return <span className="text-gray-600">-</span>
  const color = value >= 60 ? 'text-emerald-400' : value >= 55 ? 'text-amber-400' : 'text-gray-400'
  return <span className={color}>{value.toFixed(1)}%</span>
}
