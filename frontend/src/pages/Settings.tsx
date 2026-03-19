import { useState } from 'react'
import Admin from './Admin'
import Training from './Training'
import Diagnostics from './Diagnostics'
import Feedback from './Feedback'
import About from './About'

type Tab = 'admin' | 'training' | 'diagnostics' | 'feedback' | 'about'

export default function Settings() {
  const [tab, setTab] = useState<Tab>('admin')

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">Settings</h2>
        <p className="text-sm text-gray-500 mt-1">
          Manage assets, view training results, diagnostics, and more.
        </p>
      </div>

      {/* Tab selector */}
      <div className="flex gap-2 flex-wrap">
        {([
          { key: 'admin' as Tab, label: 'Assets' },
          { key: 'training' as Tab, label: 'Training' },
          { key: 'diagnostics' as Tab, label: 'Diagnostics' },
          { key: 'feedback' as Tab, label: 'Feedback' },
          { key: 'about' as Tab, label: 'About' },
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

      {tab === 'admin' && <Admin />}
      {tab === 'training' && <Training />}
      {tab === 'diagnostics' && <Diagnostics />}
      {tab === 'feedback' && <Feedback />}
      {tab === 'about' && <About />}
    </div>
  )
}
