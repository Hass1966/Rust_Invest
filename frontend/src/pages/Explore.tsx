import { useState } from 'react'
import Simulate from './Simulate'
import Advisor from './Advisor'
import History from './History'

type Tab = 'simulate' | 'advisor' | 'history'

export default function Explore() {
  const [tab, setTab] = useState<Tab>('simulate')

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">Explore</h2>
        <p className="text-sm text-gray-500 mt-1">
          Simulate strategies, ask the AI advisor, or review signal history.
        </p>
      </div>

      {/* Tab selector */}
      <div className="flex gap-2">
        {([
          { key: 'simulate' as Tab, label: 'What-If Simulator' },
          { key: 'advisor' as Tab, label: 'AI Advisor' },
          { key: 'history' as Tab, label: 'Signal History' },
        ]).map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
              tab === t.key
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'simulate' && <Simulate />}
      {tab === 'advisor' && <Advisor />}
      {tab === 'history' && <History />}
    </div>
  )
}
