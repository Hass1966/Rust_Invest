import { useEffect, useState } from 'react'
import { fetchStockSignals } from '../lib/api'
import type { EnrichedSignal } from '../lib/types'
import SignalCard from '../components/SignalCard'

export default function Stocks() {
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStockSignals()
      .then(setSignals)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-500 p-8">Loading stock signals...</div>

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
