import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { Loader2 } from 'lucide-react'
import { fetchSimulation } from '../lib/api'
import type { SimResult } from '../lib/types'

const CAPITAL_OPTIONS = [1_000, 5_000, 10_000, 50_000]
const PERIOD_OPTIONS = [7, 14, 30, 60, 90]

function fmtGBP(n: number): string {
  return '\u00A3' + Math.round(n).toLocaleString()
}

export default function Simulate() {
  const [capital, setCapital] = useState(10_000)
  const [customCapital, setCustomCapital] = useState('')
  const [useCustom, setUseCustom] = useState(false)
  const [period, setPeriod] = useState(30)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SimResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function runSim() {
    const cap = useCustom ? parseFloat(customCapital) || 10_000 : capital
    setLoading(true)
    setError(null)
    try {
      const res = await fetchSimulation(period, cap)
      setResult(res)
    } catch {
      setError('Simulation failed. Make sure models are trained and the server is running.')
    } finally {
      setLoading(false)
    }
  }

  const returnPct = result ? result.total_return_pct : 0
  const bhPct = result ? result.vs_buy_and_hold_pct : 0
  const beatsBH = returnPct > bhPct

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-white text-xl font-semibold">What if you had followed our signals?</h2>
        <p className="text-gray-500 text-sm mt-1">
          This shows exactly what would have happened if you had bought when we said buy, sold when we said sell, and held when we said hold.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-6">
          {/* Capital */}
          <div>
            <div className="text-gray-500 text-xs mb-2">Starting capital</div>
            <div className="flex gap-2 flex-wrap">
              {CAPITAL_OPTIONS.map(c => (
                <button
                  key={c}
                  onClick={() => { setCapital(c); setUseCustom(false) }}
                  className={`px-3 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                    !useCustom && capital === c
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  {fmtGBP(c)}
                </button>
              ))}
              <div className="flex items-center gap-1">
                <span className="text-gray-500 text-sm">{'\u00A3'}</span>
                <input
                  type="number"
                  value={customCapital}
                  onChange={e => { setCustomCapital(e.target.value); setUseCustom(true) }}
                  onFocus={() => setUseCustom(true)}
                  placeholder="Custom"
                  className={`w-24 bg-[#0a0e17] border rounded px-2 py-1.5 text-sm text-white outline-none ${
                    useCustom ? 'border-cyan-500/30' : 'border-[#1f2937]'
                  }`}
                />
              </div>
            </div>
          </div>

          {/* Period */}
          <div>
            <div className="text-gray-500 text-xs mb-2">Period</div>
            <div className="flex gap-2">
              {PERIOD_OPTIONS.map(p => (
                <button
                  key={p}
                  onClick={() => setPeriod(p)}
                  className={`px-3 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                    period === p
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  {p}d
                </button>
              ))}
            </div>
          </div>

          {/* Run button */}
          <div className="flex items-end">
            <button
              onClick={runSim}
              disabled={loading}
              className="px-6 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg font-medium text-sm hover:bg-cyan-500/30 transition-colors disabled:opacity-50 cursor-pointer flex items-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {loading ? 'Running...' : 'Run Simulation'}
            </button>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Big number card */}
          <div className="bg-[#111827] border border-cyan-500/20 rounded-lg p-8 text-center shadow-[0_0_30px_rgba(6,182,212,0.05)]">
            <p className="text-gray-400 text-sm mb-2">
              {fmtGBP(result.starting_capital)} would be worth
            </p>
            <div className={`text-4xl font-bold mb-2 ${returnPct >= 0 ? 'text-cyan-400' : 'text-red-400'}`}>
              {fmtGBP(result.final_value)}
            </div>
            <p className="text-gray-500 text-sm">
              {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}% return · {result.days} days · {result.signal_accuracy_pct.toFixed(1)}% of signals were correct
            </p>
          </div>

          {/* Comparison bar */}
          <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
            <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Performance comparison</div>
            <div className="grid grid-cols-3 gap-4">
              <CompareBar label="Our signals" value={returnPct} highlight={beatsBH} />
              <CompareBar label="Buy & Hold" value={bhPct} highlight={!beatsBH} />
              <CompareBar label="Cash" value={0} highlight={false} />
            </div>
          </div>

          {/* Equity chart */}
          {result.daily.length > 1 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Portfolio value over time</div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={result.daily} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#4b5563', fontSize: 11 }}
                    tickFormatter={v => v.slice(5)}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: '#4b5563', fontSize: 11 }}
                    tickFormatter={v => `\u00A3${(v / 1000).toFixed(1)}k`}
                    width={55}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                    labelStyle={{ color: '#e5e7eb' }}
                    formatter={(v: number) => [fmtGBP(v), 'Value']}
                  />
                  <ReferenceLine
                    y={result.starting_capital}
                    stroke="#374151"
                    strokeDasharray="4 4"
                    label={{ value: 'Start', fill: '#6b7280', fontSize: 10 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: '#06b6d4' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Per-asset breakdown */}
          {result.per_asset.length > 0 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Per-asset breakdown</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
                    <th className="text-left py-2">Asset</th>
                    <th className="text-right py-2">Signal Accuracy</th>
                    <th className="text-right py-2">Contribution to Return</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_asset.map(a => (
                    <tr key={a.asset} className="border-b border-[#1f2937]/50">
                      <td className="py-2 text-white font-medium">{a.asset}</td>
                      <td className={`py-2 text-right ${a.signal_accuracy_pct >= 60 ? 'text-emerald-400' : a.signal_accuracy_pct >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
                        {a.signal_accuracy_pct.toFixed(1)}%
                      </td>
                      <td className={`py-2 text-right ${a.contribution_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {a.contribution_pct >= 0 ? '+' : ''}{a.contribution_pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Caveat */}
          <div className="bg-[#0a0e17] border border-[#1f2937] rounded-lg p-4 text-gray-500 text-xs leading-relaxed">
            This simulation uses the same models that generated live signals. It is not a true backtest — the models were trained on data that includes this period, so results may be optimistic. Live performance since {result.inception_date} is the more reliable measure.
          </div>
        </>
      )}
    </div>
  )
}

function CompareBar({ label, value, highlight }: { label: string; value: number; highlight: boolean }) {
  const isPositive = value >= 0
  const barColor = highlight
    ? (isPositive ? '#10b981' : '#ef4444')
    : '#374151'

  return (
    <div className="text-center">
      <div className={`text-lg font-bold ${highlight ? (isPositive ? 'text-emerald-400' : 'text-red-400') : 'text-gray-500'}`}>
        {isPositive ? '+' : ''}{value.toFixed(2)}%
      </div>
      <div className="text-gray-500 text-xs mt-1">{label}</div>
      <div className="h-1.5 mt-2 bg-[#0a0e17] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${Math.min(Math.abs(value) * 2, 100)}%`,
            background: barColor,
          }}
        />
      </div>
    </div>
  )
}
