import { Routes, Route, NavLink, useLocation, Navigate } from 'react-router-dom'
import { useState } from 'react'
import { BarChart3, TrendingUp, Wallet, Target, Compass, Settings as SettingsIcon, MessageSquare, X, LogOut, LogIn } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import MyPortfolio from './pages/MyPortfolio'
import TrackRecord from './pages/TrackRecord'
import Explore from './pages/Explore'
import Settings from './pages/Settings'
import Login from './pages/Login'
import AuthCallback from './pages/AuthCallback'
import ChatPanel from './components/ChatPanel'
import { useAuth } from './lib/auth'

const tabs = [
  { path: '/', label: 'Dashboard', icon: BarChart3 },
  { path: '/my-portfolio', label: 'My Portfolio', icon: Wallet, protected: true },
  { path: '/track-record', label: 'Track Record', icon: Target },
  { path: '/explore', label: 'Explore', icon: Compass },
]

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return null
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

export default function App() {
  const [chatOpen, setChatOpen] = useState(false)
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const location = useLocation()
  const { user, logout } = useAuth()

  const currentTab = tabs.find(t =>
    t.path === '/' ? location.pathname === '/' : location.pathname.startsWith(t.path)
  )?.label.toLowerCase() || 'dashboard'

  // Full-page auth routes (no nav)
  if (location.pathname === '/login' || location.pathname === '/auth/callback') {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
      </Routes>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[#1f2937] bg-[#111827] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-cyan-500/20 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
          </div>
          <h1 className="text-lg font-semibold text-white">Alpha Signal</h1>
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

          {/* Settings */}
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-cyan-500/15 text-cyan-400'
                  : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
              }`
            }
          >
            <SettingsIcon className="w-4 h-4" />
          </NavLink>
        </nav>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setChatOpen(!chatOpen)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
              chatOpen ? 'bg-cyan-500/15 text-cyan-400' : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
            }`}
          >
            {chatOpen ? <X className="w-4 h-4" /> : <MessageSquare className="w-4 h-4" />}
            Chat
          </button>

          {user ? (
            <button
              onClick={logout}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-colors cursor-pointer"
              title={user.email}
            >
              <LogOut className="w-4 h-4" />
            </button>
          ) : (
            <NavLink
              to="/login"
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400 hover:text-cyan-400 hover:bg-cyan-500/10 transition-colors"
            >
              <LogIn className="w-4 h-4" />
              Sign In
            </NavLink>
          )}
        </div>
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
            <Route path="/" element={<Dashboard />} />
            <Route path="/my-portfolio" element={<ProtectedRoute><MyPortfolio /></ProtectedRoute>} />
            <Route path="/track-record" element={<TrackRecord />} />
            <Route path="/explore" element={<Explore />} />
            <Route path="/settings" element={<Settings />} />
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
