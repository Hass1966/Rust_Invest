import { useEffect, useState } from 'react'
import { fetchStockSignals } from '../lib/api'
import type { EnrichedSignal } from '../lib/types'
import SignalCard from '../components/SignalCard'

export default function Stocks() {
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetchStockSignals()
      .then(setSignals)
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="space-y-4 p-8">
      {[1, 2, 3, 4].map(i => (
        <div key={i} className="bg-[#111827] border border-[#1f2937] rounded-lg p-4 space-y-3">
          <div className="h-5 bg-gray-700/50 rounded w-32 skeleton-pulse" />
          <div className="h-4 bg-gray-700/50 rounded w-full skeleton-pulse" />
          <div className="h-4 bg-gray-700/50 rounded w-3/4 skeleton-pulse" />
        </div>
      ))}
    </div>
  )

  if (error) return (
    <div className="text-gray-500 p-8 text-center">
      <p>Couldn't load this data. Is the server running?</p>
      <button onClick={() => window.location.reload()} className="text-cyan-400 text-xs mt-2 hover:underline cursor-pointer">Retry</button>
    </div>
  )

  return (
    <div>
      <h2 className="text-white text-xl font-semibold mb-4">Stock Signals</h2>
      <p className="text-gray-500 text-sm mb-6">
        {signals.length} stocks monitored. Click a card to see per-model breakdown.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {signals.map(s => <SignalCard key={s.asset} signal={s} />)}
      </div>
    </div>
  )
}
