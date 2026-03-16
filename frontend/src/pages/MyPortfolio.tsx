import { useEffect, useState, useCallback } from 'react'
import { Trash2, Plus, RefreshCw, Edit2, Check, X } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import {
  fetchUserHoldings, addUserHolding, updateUserHolding,
  deleteUserHolding, comparePortfolio,
} from '../lib/api'
import type { UserHolding, PortfolioComparison, AssetComparison } from '../lib/api'

type Frequency = 'weekly' | 'daily' | 'hourly'

export default function MyPortfolio() {
  const [holdings, setHoldings] = useState<UserHolding[]>([])
  const [comparison, setComparison] = useState<PortfolioComparison | null>(null)
  const [loading, setLoading] = useState(true)
  const [comparing, setComparing] = useState(false)
  const [frequency, setFrequency] = useState<Frequency>('weekly')
  const [editId, setEditId] = useState<number | null>(null)
  const [editQty, setEditQty] = useState('')
  const [editDate, setEditDate] = useState('')

  // Add form state
  const [newSymbol, setNewSymbol] = useState('')
  const [newQty, setNewQty] = useState('')
  const [newDate, setNewDate] = useState('')
  const [addError, setAddError] = useState('')

  const loadHoldings = useCallback(() => {
    fetchUserHoldings()
      .then(setHoldings)
      .catch(() => setHoldings([]))
      .finally(() => setLoading(false))
  }, [])

  const runComparison = useCallback(() => {
    if (frequency === 'hourly') return
    setComparing(true)
    comparePortfolio(frequency)
      .then(setComparison)
      .catch(() => setComparison(null))
      .finally(() => setComparing(false))
  }, [frequency])

  useEffect(() => { loadHoldings() }, [loadHoldings])

  // Auto-run comparison when holdings or frequency change
  useEffect(() => {
    if (holdings.length > 0) runComparison()
    else setComparison(null)
  }, [holdings, runComparison])

  const handleAdd = async () => {
    if (!newSymbol.trim() || !newQty || !newDate) {
      setAddError('All fields required')
      return
    }
    const qty = parseFloat(newQty)
    if (isNaN(qty) || qty <= 0) {
      setAddError('Invalid quantity')
      return
    }
    setAddError('')
    try {
      await addUserHolding(newSymbol.trim(), qty, newDate)
      setNewSymbol('')
      setNewQty('')
      setNewDate('')
      loadHoldings()
    } catch {
      setAddError('Failed to add holding')
    }
  }

  const handleDelete = async (id: number, symbol: string) => {
    if (!confirm(`Are you sure you want to remove ${symbol} from your portfolio?`)) return
    setAddError('')
    try {
      await deleteUserHolding(id)
      setHoldings(prev => prev.filter(h => h.id !== id))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setAddError(`Failed to delete ${symbol}: ${msg}`)
    }
  }

  const handleEdit = (h: UserHolding) => {
    setEditId(h.id)
    setEditQty(h.quantity.toString())
    setEditDate(h.start_date)
  }

  const handleSaveEdit = async (id: number) => {
    const qty = parseFloat(editQty)
    if (isNaN(qty) || qty <= 0) return
    await updateUserHolding(id, qty, editDate)
    setEditId(null)
    loadHoldings()
  }

  if (loading) return <div className="text-gray-500 p-8">Loading portfolio...</div>

  const verdictColor = comparison?.verdict === 'signals_win' ? 'text-green-400'
    : comparison?.verdict === 'buy_hold_wins' ? 'text-red-400'
    : 'text-amber-400'

  const verdictBg = comparison?.verdict === 'signals_win' ? 'border-green-500/30 bg-green-500/5'
    : comparison?.verdict === 'buy_hold_wins' ? 'border-red-500/30 bg-red-500/5'
    : 'border-amber-500/30 bg-amber-500/5'

  const verdictText = comparison?.verdict === 'signals_win'
    ? 'Following signals beat buy & hold'
    : comparison?.verdict === 'buy_hold_wins'
    ? 'Buy & hold beat following signals'
    : 'Roughly equal performance'

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">My Portfolio</h2>
        <p className="text-sm text-gray-500 mt-1">
          Compare what your portfolio would be worth following signals vs buy &amp; hold.
        </p>
      </div>

      {/* Caveat */}
      <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
        <p className="text-sm text-amber-300/90">
          <strong>Note:</strong> Signals are retroactively generated using trained models.
          This is a simulated backtest, not out-of-sample results.
        </p>
      </div>

      {/* Add holding form */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Add Holding</h3>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            placeholder="Symbol (e.g. AAPL, bitcoin, EURUSD=X)"
            value={newSymbol}
            onChange={e => setNewSymbol(e.target.value)}
            className="bg-[#0a0e17] border border-[#1f2937] rounded-lg px-3 py-2 text-sm text-gray-200 flex-1 min-w-0"
          />
          <input
            type="number"
            placeholder="Quantity"
            value={newQty}
            onChange={e => setNewQty(e.target.value)}
            step="any"
            min="0"
            className="bg-[#0a0e17] border border-[#1f2937] rounded-lg px-3 py-2 text-sm text-gray-200 w-full sm:w-28"
          />
          <input
            type="date"
            value={newDate}
            onChange={e => setNewDate(e.target.value)}
            className="bg-[#0a0e17] border border-[#1f2937] rounded-lg px-3 py-2 text-sm text-gray-200 w-full sm:w-40"
          />
          <button
            onClick={handleAdd}
            className="flex items-center justify-center gap-2 bg-cyan-500/15 text-cyan-400 hover:bg-cyan-500/25 rounded-lg px-4 py-2 text-sm transition-colors cursor-pointer"
          >
            <Plus className="w-4 h-4" /> Add
          </button>
        </div>
        {addError && <p className="text-red-400 text-xs mt-2">{addError}</p>}
        <p className="text-xs text-gray-600 mt-2">
          Stocks: use ticker (AAPL, MSFT) &middot; FX: use pair (EURUSD=X) &middot; Crypto: use CoinGecko ID (bitcoin, ethereum, solana)
        </p>
      </div>

      {/* Holdings table */}
      {holdings.length > 0 && (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Your Holdings ({holdings.length})</h3>
            <button
              onClick={runComparison}
              disabled={comparing}
              className="flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${comparing ? 'animate-spin' : ''}`} />
              {comparing ? 'Backtesting...' : 'Refresh'}
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-[#1f2937]">
                  <th className="text-left py-2 px-2">Symbol</th>
                  <th className="text-left py-2 px-2">Class</th>
                  <th className="text-right py-2 px-2">Quantity</th>
                  <th className="text-left py-2 px-2">Start Date</th>
                  <th className="text-center py-2 px-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {holdings.map(h => (
                  <tr key={h.id} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                    <td className="py-2 px-2 text-gray-200 font-medium">{h.symbol}</td>
                    <td className="py-2 px-2 text-gray-500 text-xs capitalize">{h.asset_class}</td>
                    <td className="py-2 px-2 text-right text-gray-300">
                      {editId === h.id ? (
                        <input type="number" value={editQty} onChange={e => setEditQty(e.target.value)}
                          step="any" className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-200 w-24 text-right" />
                      ) : h.quantity}
                    </td>
                    <td className="py-2 px-2 text-gray-400 text-xs">
                      {editId === h.id ? (
                        <input type="date" value={editDate} onChange={e => setEditDate(e.target.value)}
                          className="bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1 text-sm text-gray-200" />
                      ) : h.start_date}
                    </td>
                    <td className="py-2 px-2 text-center">
                      {editId === h.id ? (
                        <div className="flex justify-center gap-1">
                          <button onClick={() => handleSaveEdit(h.id)} className="text-green-400 hover:text-green-300 cursor-pointer p-1">
                            <Check className="w-4 h-4" />
                          </button>
                          <button onClick={() => setEditId(null)} className="text-gray-400 hover:text-gray-300 cursor-pointer p-1">
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <div className="flex justify-center gap-1">
                          <button onClick={() => handleEdit(h)} className="text-gray-400 hover:text-gray-300 cursor-pointer p-1">
                            <Edit2 className="w-4 h-4" />
                          </button>
                          <button onClick={() => handleDelete(h.id, h.symbol)} className="text-red-400/60 hover:text-red-400 cursor-pointer p-1">
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Frequency tabs */}
      {holdings.length > 0 && (
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Signal frequency</span>
          <div className="flex gap-2">
            {(['weekly', 'daily', 'hourly'] as const).map(f => (
              <button
                key={f}
                onClick={() => f !== 'hourly' && setFrequency(f)}
                disabled={f === 'hourly'}
                className={`px-4 py-1.5 rounded text-sm transition-colors ${
                  f === 'hourly'
                    ? 'bg-[#0a0e17] text-gray-600 border border-[#1f2937] cursor-not-allowed'
                    : frequency === f
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 cursor-pointer'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151] cursor-pointer'
                }`}
                title={f === 'hourly' ? 'Available after next training run' : undefined}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
          {comparing && <span className="text-xs text-gray-500 animate-pulse">Generating signals...</span>}
          {frequency === 'daily' && !comparing && (
            <span className="text-xs text-gray-600">Daily mode may take 30-60s</span>
          )}
        </div>
      )}

      {/* Comparison results */}
      {comparison?.has_data && (
        <>
          {/* Verdict banner */}
          <div className={`rounded-xl border p-4 ${verdictBg}`}>
            <div className={`text-lg font-semibold ${verdictColor} text-center`}>
              {verdictText}
            </div>
          </div>

          {/* Metrics row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard label="Sharpe (Signals)" value={comparison.sharpe_signals?.toFixed(2) ?? '-'} />
            <MetricCard label="Sharpe (B&H)" value={comparison.sharpe_buy_hold?.toFixed(2) ?? '-'} />
            <MetricCard label="Win Rate" value={comparison.overall_win_rate_pct != null ? `${comparison.overall_win_rate_pct.toFixed(1)}%` : '-'} />
            <MetricCard label="Total Trades" value={comparison.total_trades?.toString() ?? '-'} />
          </div>

          {/* Equity curve chart */}
          {comparison.equity_curve && comparison.equity_curve.length > 1 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">
                Portfolio equity curve
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={comparison.equity_curve} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#4b5563', fontSize: 11 }}
                    tickFormatter={v => v.slice(5)}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: '#4b5563', fontSize: 11 }}
                    tickFormatter={v => `$${(v / 1000).toFixed(1)}k`}
                    width={60}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                    labelStyle={{ color: '#e5e7eb' }}
                    formatter={(v: number | undefined, name?: string) => [formatCurrency(v ?? 0), name === 'signal_value' ? 'Follow Signals' : 'Buy & Hold']}
                  />
                  <Legend formatter={v => v === 'signal_value' ? 'Follow Signals' : 'Buy & Hold'} />
                  <Line type="monotone" dataKey="signal_value" name="signal_value"
                    stroke="#06b6d4" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="buy_hold_value" name="buy_hold_value"
                    stroke="#6b7280" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Side by side comparison */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Follow Signals */}
            <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
              <h3 className="text-sm font-medium text-gray-400 mb-1">Follow The Signals</h3>
              <p className="text-xs text-gray-600 mb-3">Trade on every BUY/SELL/HOLD signal</p>
              <div className="text-3xl font-bold text-white mb-1">
                {formatCurrency(comparison.signal_value!)}
              </div>
              <div className={`text-sm font-medium ${comparison.signal_return_pct! >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {comparison.signal_return_pct! >= 0 ? '+' : ''}{comparison.signal_return_pct!.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-600 mt-1">
                from {formatCurrency(comparison.total_cost!)} invested
              </div>
            </div>

            {/* Buy & Hold */}
            <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
              <h3 className="text-sm font-medium text-gray-400 mb-1">Buy &amp; Hold</h3>
              <p className="text-xs text-gray-600 mb-3">No trading, just hold from start date</p>
              <div className="text-3xl font-bold text-white mb-1">
                {formatCurrency(comparison.buy_hold_value!)}
              </div>
              <div className={`text-sm font-medium ${comparison.buy_hold_return_pct! >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {comparison.buy_hold_return_pct! >= 0 ? '+' : ''}{comparison.buy_hold_return_pct!.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-600 mt-1">
                from {formatCurrency(comparison.total_cost!)} invested
              </div>
            </div>
          </div>

          {/* Per-asset breakdown */}
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Per-Asset Breakdown</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-[#1f2937]">
                    <th className="text-left py-2 px-2">Asset</th>
                    <th className="text-right py-2 px-2">Qty</th>
                    <th className="text-right py-2 px-2">Start</th>
                    <th className="text-right py-2 px-2">Now</th>
                    <th className="text-right py-2 px-2">B&amp;H %</th>
                    <th className="text-right py-2 px-2">Signal %</th>
                    <th className="text-center py-2 px-2">Winner</th>
                    <th className="text-right py-2 px-2 hidden sm:table-cell">Trades</th>
                    <th className="text-right py-2 px-2 hidden sm:table-cell">Win %</th>
                    <th className="text-right py-2 px-2 hidden md:table-cell">Sharpe</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.per_asset!.map(a => (
                    <AssetRow key={a.symbol} asset={a} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {comparison && !comparison.has_data && (
        <div className="text-gray-500 text-center py-8">
          {comparison.note || 'No data available for comparison.'}
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4 text-center">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-lg font-semibold text-white">{value}</div>
    </div>
  )
}

function AssetRow({ asset: a }: { asset: AssetComparison }) {
  const signalBetter = a.signal_return_pct > a.buy_hold_return_pct + 1
  const holdBetter = a.buy_hold_return_pct > a.signal_return_pct + 1
  const winner = signalBetter ? 'Signals' : holdBetter ? 'B&H' : 'Even'
  const winnerColor = signalBetter ? 'text-green-400' : holdBetter ? 'text-red-400' : 'text-amber-400'

  return (
    <tr className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
      <td className="py-2 px-2">
        <div className="text-gray-200 font-medium">{a.symbol}</div>
        <div className="text-xs text-gray-600 capitalize">{a.asset_class}</div>
        {a.note && <div className="text-xs text-amber-400/70">{a.note}</div>}
      </td>
      <td className="py-2 px-2 text-right text-gray-400">{a.quantity}</td>
      <td className="py-2 px-2 text-right text-gray-400">{formatPrice(a.start_price)}</td>
      <td className="py-2 px-2 text-right text-gray-300">{formatPrice(a.current_price)}</td>
      <td className={`py-2 px-2 text-right ${a.buy_hold_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {a.buy_hold_return_pct >= 0 ? '+' : ''}{a.buy_hold_return_pct.toFixed(2)}%
      </td>
      <td className={`py-2 px-2 text-right ${a.signal_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {a.signal_return_pct >= 0 ? '+' : ''}{a.signal_return_pct.toFixed(2)}%
      </td>
      <td className={`py-2 px-2 text-center font-medium ${winnerColor}`}>{winner}</td>
      <td className="py-2 px-2 text-right text-gray-500 hidden sm:table-cell">{a.total_trades}</td>
      <td className="py-2 px-2 text-right text-gray-500 hidden sm:table-cell">{a.win_rate_pct.toFixed(1)}%</td>
      <td className="py-2 px-2 text-right text-gray-500 hidden md:table-cell">{a.sharpe_signals.toFixed(2)}</td>
    </tr>
  )
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (price >= 1) return price.toFixed(2)
  if (price >= 0.01) return price.toFixed(4)
  return price.toFixed(6)
}

function formatCurrency(val: number): string {
  return val.toLocaleString(undefined, { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 })
}
