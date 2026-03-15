import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'
import { BarChart3, TrendingUp, DollarSign, Briefcase, Cpu, MessageSquare, X, Clock, Bot, PlayCircle, Bitcoin, GraduationCap, Target, ShieldCheck, Wallet, Info, MessageCircle } from 'lucide-react'
import Overview from './pages/Overview'
import Stocks from './pages/Stocks'
import FX from './pages/FX'
import Crypto from './pages/Crypto'
import Portfolio from './pages/Portfolio'
import History from './pages/History'
import Training from './pages/Training'
import Diagnostics from './pages/Diagnostics'
import Advisor from './pages/Advisor'
import Simulate from './pages/Simulate'
import Admin from './pages/Admin'
import Predictions from './pages/Predictions'
import Truth from './pages/Truth'
import MyPortfolio from './pages/MyPortfolio'
import About from './pages/About'
import Feedback from './pages/Feedback'
import ChatPanel from './components/ChatPanel'

const tabs = [
  { path: '/', label: 'Overview', icon: BarChart3 },
  { path: '/stocks', label: 'Stocks', icon: TrendingUp },
  { path: '/fx', label: 'FX', icon: DollarSign },
  { path: '/crypto', label: 'Crypto', icon: Bitcoin },
  { path: '/portfolio', label: 'Portfolio', icon: Briefcase },
  { path: '/my-portfolio', label: 'My Portfolio', icon: Wallet },
  { path: '/history', label: 'History', icon: Clock },
  { path: '/simulate', label: 'Simulate', icon: PlayCircle },
  { path: '/advisor', label: 'Advisor', icon: Bot },
  { path: '/predictions', label: 'Predictions', icon: Target },
  { path: '/truth', label: 'Signal Truth', icon: ShieldCheck },
  { path: '/training', label: 'Training', icon: GraduationCap },
  { path: '/diagnostics', label: 'Diagnostics', icon: Cpu },
  { path: '/feedback', label: 'Feedback', icon: MessageCircle },
  { path: '/about', label: 'About', icon: Info },
]

export default function App() {
  const [chatOpen, setChatOpen] = useState(false)
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const location = useLocation()

  const currentTab = tabs.find(t =>
    t.path === '/' ? location.pathname === '/' : location.pathname.startsWith(t.path)
  )?.label.toLowerCase() || 'overview'

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[#1f2937] bg-[#111827] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-cyan-500/20 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
          </div>
          <h1 className="text-lg font-semibold text-white">Rust Invest</h1>
        </div>

        <nav className="flex gap-1 overflow-x-auto scrollbar-hide">
          {tabs.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        <button
          onClick={() => setChatOpen(!chatOpen)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
            chatOpen ? 'bg-cyan-500/15 text-cyan-400' : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
          }`}
        >
          {chatOpen ? <X className="w-4 h-4" /> : <MessageSquare className="w-4 h-4" />}
          Chat
        </button>
      </header>

      {/* Beta Banner */}
      {!bannerDismissed && (
        <div className="bg-cyan-900/30 border-b border-cyan-800/40 px-4 py-2 flex items-center justify-between text-sm">
          <span className="text-cyan-300/90">
            Beta &mdash; Signal tracking started 15 March 2026. Signals shown transparently &mdash; good and bad. Not financial advice.
          </span>
          <button
            onClick={() => setBannerDismissed(true)}
            className="text-cyan-400/60 hover:text-cyan-300 ml-4 cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Content */}
      <div className="flex flex-1 overflow-hidden">
        <main className={`flex-1 overflow-y-auto p-6 transition-all ${chatOpen ? 'mr-96' : ''}`}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/stocks" element={<Stocks />} />
            <Route path="/fx" element={<FX />} />
            <Route path="/crypto" element={<Crypto />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/my-portfolio" element={<MyPortfolio />} />
            <Route path="/history" element={<History />} />
            <Route path="/simulate" element={<Simulate />} />
            <Route path="/advisor" element={<Advisor />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/truth" element={<Truth />} />
            <Route path="/training" element={<Training />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/feedback" element={<Feedback />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>

        {chatOpen && (
          <aside className="w-96 fixed right-0 top-[57px] bottom-0 border-l border-[#1f2937] bg-[#111827]">
            <ChatPanel tabContext={currentTab} />
          </aside>
        )}
      </div>
    </div>
  )
}
