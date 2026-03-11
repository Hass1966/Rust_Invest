import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Plus, Loader2 } from 'lucide-react'

interface AssetEntry {
  symbol: string
  name: string
  enabled: boolean
}

interface AssetConfig {
  stocks: AssetEntry[]
  fx: AssetEntry[]
  crypto: AssetEntry[]
}

const BASE = ''

export default function Admin() {
  const [config, setConfig] = useState<AssetConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [reloading, setReloading] = useState(false)
  const [reloadMsg, setReloadMsg] = useState('')
  const [toggling, setToggling] = useState<string | null>(null)

  // Add form state
  const [newSymbol, setNewSymbol] = useState('')
  const [newName, setNewName] = useState('')
  const [newClass, setNewClass] = useState('stock')
  const [newEnabled, setNewEnabled] = useState(true)
  const [addError, setAddError] = useState('')
  const [adding, setAdding] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch(`${BASE}/api/v1/config/assets`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: AssetConfig = await res.json()
      setConfig(data)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchConfig() }, [fetchConfig])

  const handleToggle = async (symbol: string, enabled: boolean) => {
    setToggling(symbol)
    try {
      const res = await fetch(`${BASE}/api/v1/admin/assets/${encodeURIComponent(symbol)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      if (!res.ok) {
        const err = await res.json()
        alert(err.error || 'Failed to toggle')
        return
      }
      await fetchConfig()
    } finally {
      setToggling(null)
    }
  }

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault()
    setAddError('')
    if (!newSymbol.trim() || !newName.trim()) {
      setAddError('Symbol and name are required')
      return
    }
    setAdding(true)
    try {
      const res = await fetch(`${BASE}/api/v1/admin/assets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: newSymbol.trim().toUpperCase(),
          name: newName.trim(),
          class: newClass,
          enabled: newEnabled,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        setAddError(err.error || 'Failed to add asset')
        return
      }
      setNewSymbol('')
      setNewName('')
      setNewClass('stock')
      setNewEnabled(true)
      await fetchConfig()
    } finally {
      setAdding(false)
    }
  }

  const handleReload = async () => {
    setReloading(true)
    setReloadMsg('')
    try {
      const res = await fetch(`${BASE}/api/v1/models/reload`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setReloadMsg(`Reload complete — ${data.assets_found} signals generated`)
    } catch {
      setReloadMsg('Reload failed')
    } finally {
      setReloading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    )
  }

  const allAssets: (AssetEntry & { class: string })[] = [
    ...(config?.stocks || []).map(a => ({ ...a, class: 'stock' })),
    ...(config?.fx || []).map(a => ({ ...a, class: 'fx' })),
    ...(config?.crypto || []).map(a => ({ ...a, class: 'crypto' })),
  ]

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Asset Management</h2>
        <button
          onClick={handleReload}
          disabled={reloading}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg text-sm transition-colors cursor-pointer"
        >
          {reloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          Save &amp; Reload Models
        </button>
      </div>
      {reloadMsg && (
        <div className="px-4 py-2 rounded bg-cyan-900/40 text-cyan-300 text-sm">{reloadMsg}</div>
      )}

      {/* Asset Table */}
      <div className="bg-[#1f2937] rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              <th className="text-left px-4 py-3">Symbol</th>
              <th className="text-left px-4 py-3">Name</th>
              <th className="text-left px-4 py-3">Class</th>
              <th className="text-center px-4 py-3">Enabled</th>
            </tr>
          </thead>
          <tbody>
            {allAssets.map(a => (
              <tr key={`${a.class}-${a.symbol}`} className="border-b border-gray-800 hover:bg-white/5">
                <td className="px-4 py-2 text-white font-mono">{a.symbol}</td>
                <td className="px-4 py-2 text-gray-300">{a.name}</td>
                <td className="px-4 py-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    a.class === 'stock' ? 'bg-blue-900/50 text-blue-300' :
                    a.class === 'fx' ? 'bg-purple-900/50 text-purple-300' :
                    'bg-orange-900/50 text-orange-300'
                  }`}>{a.class}</span>
                </td>
                <td className="px-4 py-2 text-center">
                  <button
                    onClick={() => handleToggle(a.symbol, !a.enabled)}
                    disabled={toggling === a.symbol}
                    className={`relative w-10 h-5 rounded-full transition-colors cursor-pointer ${
                      a.enabled ? 'bg-cyan-500' : 'bg-gray-600'
                    }`}
                  >
                    <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                      a.enabled ? 'left-5' : 'left-0.5'
                    }`} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Add Asset Form */}
      <div className="bg-[#1f2937] rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Plus className="w-5 h-5" /> Add Asset
        </h3>
        <form onSubmit={handleAdd} className="grid grid-cols-2 md:grid-cols-5 gap-4 items-end">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Symbol</label>
            <input
              type="text"
              value={newSymbol}
              onChange={e => setNewSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Name</label>
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="Apple Inc."
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Class</label>
            <select
              value={newClass}
              onChange={e => setNewClass(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
            >
              <option value="stock">Stock</option>
              <option value="etf">ETF</option>
              <option value="fx">FX</option>
              <option value="crypto">Crypto</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="block text-xs text-gray-400">Enabled</label>
            <input
              type="checkbox"
              checked={newEnabled}
              onChange={e => setNewEnabled(e.target.checked)}
              className="w-4 h-4"
            />
          </div>
          <button
            type="submit"
            disabled={adding}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white rounded text-sm transition-colors cursor-pointer"
          >
            {adding ? 'Adding...' : 'Add Asset'}
          </button>
        </form>
        {addError && <p className="mt-2 text-red-400 text-sm">{addError}</p>}
      </div>
    </div>
  )
}
