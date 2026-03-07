import { useState, useEffect } from 'react'
import { Plus, X, MessageSquare } from 'lucide-react'
import type { EnrichedSignal } from '../lib/types'
import { fetchSignals, sendChat } from '../lib/api'

// ─── Types ───

interface Holding {
  ticker: string
  shares: number
  avgPrice: number
  account: string
  addedDate: string
}

const STORAGE_KEY = 'rust_invest_holdings'
const TECH_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'QQQ', 'TSLA']
const ACCOUNT_OPTIONS = ['ISA', 'GIA', 'SIPP', 'Trading 212', 'Freetrade', 'HL', 'Other']

function loadHoldings(): Holding[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveHoldings(holdings: Holding[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(holdings))
}

// ─── Main Component ───

export default function UserHoldings() {
  const [holdings, setHoldings] = useState<Holding[]>(loadHoldings)
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [showForm, setShowForm] = useState(false)
  const [askingAI, setAskingAI] = useState<string | null>(null)
  const [aiResponse, setAiResponse] = useState<string | null>(null)

  // Form state
  const [ticker, setTicker] = useState('')
  const [shares, setShares] = useState('')
  const [avgPrice, setAvgPrice] = useState('')
  const [account, setAccount] = useState('ISA')

  useEffect(() => {
    fetchSignals().then(setSignals).catch(() => {})
  }, [])

  const signalMap: Record<string, EnrichedSignal> = {}
  for (const s of signals) signalMap[s.asset] = s

  function addHolding() {
    const t = ticker.trim().toUpperCase()
    if (!t || !shares || !avgPrice) return
    const newHolding: Holding = {
      ticker: t,
      shares: parseFloat(shares),
      avgPrice: parseFloat(avgPrice),
      account,
      addedDate: new Date().toISOString().slice(0, 10),
    }
    const updated = [...holdings, newHolding]
    setHoldings(updated)
    saveHoldings(updated)
    setTicker('')
    setShares('')
    setAvgPrice('')
    setShowForm(false)
  }

  function removeHolding(idx: number) {
    const updated = holdings.filter((_, i) => i !== idx)
    setHoldings(updated)
    saveHoldings(updated)
  }

  async function askAI(h: Holding) {
    const sig = signalMap[h.ticker]
    const sigText = sig ? sig.signal : 'unknown'
    const msg = `I hold ${h.shares} shares of ${h.ticker} at avg price £${h.avgPrice}. Current signal is ${sigText}. What should I do?`
    setAskingAI(h.ticker)
    setAiResponse(null)
    try {
      const resp = await sendChat(msg, 'portfolio')
      setAiResponse(resp)
    } catch {
      setAiResponse('Could not get AI response. Is the server running?')
    }
  }

  // Derive "What to do today" from holdings × signals
  const consider_adding = holdings.filter(h => signalMap[h.ticker]?.signal === 'BUY')
  const hold_watch = holdings.filter(h => signalMap[h.ticker]?.signal === 'HOLD')
  const consider_reducing = holdings.filter(h => {
    const sig = signalMap[h.ticker]
    if (!sig || sig.signal !== 'SELL') return false
    return sig.price > h.avgPrice // only if in profit
  })

  // Tech concentration
  const totalValue = holdings.reduce((sum, h) => {
    const price = signalMap[h.ticker]?.price || h.avgPrice
    return sum + h.shares * price
  }, 0)
  const techValue = holdings
    .filter(h => TECH_TICKERS.includes(h.ticker))
    .reduce((sum, h) => {
      const price = signalMap[h.ticker]?.price || h.avgPrice
      return sum + h.shares * price
    }, 0)
  const techPct = totalValue > 0 ? (techValue / totalValue) * 100 : 0

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold text-lg">My Holdings</h3>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-1 text-sm px-3 py-1.5 rounded-lg bg-cyan-500/15 text-cyan-400 hover:bg-cyan-500/25 transition-colors cursor-pointer"
        >
          <Plus className="w-4 h-4" /> Add
        </button>
      </div>

      {/* Add form */}
      {showForm && (
        <div className="bg-[#0a0e17] rounded-lg p-4 mb-4 space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div>
              <label className="text-gray-500 text-xs block mb-1">Ticker</label>
              <input
                value={ticker}
                onChange={e => setTicker(e.target.value.toUpperCase())}
                placeholder="AAPL"
                className="w-full bg-[#111827] border border-[#1f2937] rounded px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/50"
              />
            </div>
            <div>
              <label className="text-gray-500 text-xs block mb-1">Shares</label>
              <input
                type="number"
                value={shares}
                onChange={e => setShares(e.target.value)}
                placeholder="10"
                className="w-full bg-[#111827] border border-[#1f2937] rounded px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/50"
              />
            </div>
            <div>
              <label className="text-gray-500 text-xs block mb-1">Avg Price Paid</label>
              <input
                type="number"
                value={avgPrice}
                onChange={e => setAvgPrice(e.target.value)}
                placeholder="150.00"
                className="w-full bg-[#111827] border border-[#1f2937] rounded px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/50"
              />
            </div>
            <div>
              <label className="text-gray-500 text-xs block mb-1">Account</label>
              <select
                value={account}
                onChange={e => setAccount(e.target.value)}
                className="w-full bg-[#111827] border border-[#1f2937] rounded px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/50"
              >
                {ACCOUNT_OPTIONS.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={addHolding} className="px-4 py-1.5 bg-cyan-500/15 text-cyan-400 text-sm rounded hover:bg-cyan-500/25 cursor-pointer">
              Save
            </button>
            <button onClick={() => setShowForm(false)} className="px-4 py-1.5 text-gray-500 text-sm rounded hover:text-gray-300 cursor-pointer">
              Cancel
            </button>
          </div>
        </div>
      )}

      {holdings.length === 0 ? (
        <p className="text-gray-500 text-sm text-center py-8">
          Add your first holding above to get personalised signals
        </p>
      ) : (
        <>
          {/* What to do today */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <ActionCard title="Consider Adding" items={consider_adding.map(h => h.ticker)} color="emerald" />
            <ActionCard title="Hold & Watch" items={hold_watch.map(h => h.ticker)} color="amber" />
            <ActionCard title="Consider Reducing" items={consider_reducing.map(h => h.ticker)} color="red" />
          </div>

          {/* Holdings table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
                  <th className="text-left py-2 pr-2">Asset</th>
                  <th className="text-right py-2 px-2">Shares</th>
                  <th className="text-right py-2 px-2">Avg Price</th>
                  <th className="text-right py-2 px-2">Current</th>
                  <th className="text-right py-2 px-2">Value</th>
                  <th className="text-right py-2 px-2">P&L</th>
                  <th className="text-right py-2 px-2">P&L %</th>
                  <th className="text-center py-2 px-2">Signal</th>
                  <th className="text-center py-2 pl-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((h, i) => {
                  const sig = signalMap[h.ticker]
                  const currentPrice = sig?.price || h.avgPrice
                  const value = h.shares * currentPrice
                  const pnl = (currentPrice - h.avgPrice) * h.shares
                  const pnlPct = h.avgPrice > 0 ? ((currentPrice - h.avgPrice) / h.avgPrice) * 100 : 0
                  const signalText = sig?.signal || '—'
                  const signalColor = signalText === 'BUY' ? 'text-emerald-400 bg-emerald-500/15'
                    : signalText === 'SELL' ? 'text-red-400 bg-red-500/15'
                    : signalText === 'HOLD' ? 'text-amber-400 bg-amber-500/15'
                    : 'text-gray-500'
                  return (
                    <tr key={i} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                      <td className="py-2 pr-2">
                        <div className="flex items-center gap-2">
                          <span className="text-white font-medium">{h.ticker}</span>
                          <button onClick={() => removeHolding(i)} className="text-gray-600 hover:text-red-400 cursor-pointer">
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                        <span className="text-gray-600 text-xs">{h.account}</span>
                      </td>
                      <td className="py-2 px-2 text-right text-gray-300">{h.shares}</td>
                      <td className="py-2 px-2 text-right text-gray-300 font-mono">${h.avgPrice.toFixed(2)}</td>
                      <td className="py-2 px-2 text-right text-white font-mono">${currentPrice.toFixed(2)}</td>
                      <td className="py-2 px-2 text-right text-white">${value.toFixed(0)}</td>
                      <td className={`py-2 px-2 text-right ${pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {pnl >= 0 ? '+' : ''}{pnl.toFixed(0)}
                      </td>
                      <td className={`py-2 px-2 text-right ${pnlPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
                      </td>
                      <td className="py-2 px-2 text-center">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded ${signalColor}`}>{signalText}</span>
                      </td>
                      <td className="py-2 pl-2 text-center">
                        <button
                          onClick={() => askAI(h)}
                          className="text-xs px-2 py-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 cursor-pointer flex items-center gap-1 mx-auto"
                        >
                          <MessageSquare className="w-3 h-3" /> Ask AI
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* AI response */}
          {askingAI && (
            <div className="mt-4 bg-[#0a0e17] border border-[#1f2937] rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-400 text-sm">AI advice for {askingAI}</span>
                <button onClick={() => { setAskingAI(null); setAiResponse(null) }} className="text-gray-600 hover:text-gray-400 cursor-pointer">
                  <X className="w-4 h-4" />
                </button>
              </div>
              {aiResponse ? (
                <p className="text-gray-300 text-sm whitespace-pre-line">{aiResponse}</p>
              ) : (
                <div className="space-y-2">
                  <div className="h-3 bg-gray-700/50 rounded w-full animate-pulse" />
                  <div className="h-3 bg-gray-700/50 rounded w-3/4 animate-pulse" />
                </div>
              )}
            </div>
          )}

          {/* Risk summary */}
          <div className="mt-4 bg-[#0a0e17] rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Your tech exposure</span>
              <span className="text-white text-sm font-medium">{techPct.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-[#1f2937] rounded-full">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${Math.min(techPct, 100)}%`,
                  background: techPct > 70 ? '#ef4444' : techPct > 50 ? '#f59e0b' : '#10b981',
                }}
              />
            </div>
            {techPct > 70 && (
              <p className="text-red-400/70 text-xs mt-2">High concentration risk — consider adding non-tech positions</p>
            )}
            {techPct > 50 && techPct <= 70 && (
              <p className="text-amber-400/70 text-xs mt-2">Consider adding non-tech positions for balance</p>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function ActionCard({ title, items, color }: { title: string; items: string[]; color: string }) {
  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    emerald: { bg: 'bg-emerald-500/5', border: 'border-emerald-500/20', text: 'text-emerald-400' },
    amber: { bg: 'bg-amber-500/5', border: 'border-amber-500/20', text: 'text-amber-400' },
    red: { bg: 'bg-red-500/5', border: 'border-red-500/20', text: 'text-red-400' },
  }
  const c = colorMap[color] || colorMap.amber
  return (
    <div className={`${c.bg} border ${c.border} rounded-lg p-3`}>
      <div className={`text-xs font-medium ${c.text} mb-1`}>{title}</div>
      <div className="text-white text-sm">
        {items.length > 0 ? items.join(', ') : <span className="text-gray-600">None today</span>}
      </div>
    </div>
  )
}
