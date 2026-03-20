import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { TrendingUp, UserPlus } from 'lucide-react'

export default function Register() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { register } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (password !== confirm) {
      setError('Passwords do not match')
      return
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }

    setLoading(true)
    try {
      await register(email, password)
      navigate('/my-portfolio')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0a0e17]">
      <div className="w-full max-w-md p-8">
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="w-10 h-10 rounded bg-cyan-500/20 flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-cyan-400" />
          </div>
          <h1 className="text-2xl font-bold text-white">Rust Invest</h1>
        </div>

        <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
            <UserPlus className="w-5 h-5 text-cyan-400" />
            Create Account
          </h2>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-2 rounded-lg mb-4 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="w-full px-4 py-2 bg-[#0a0e17] border border-[#1f2937] rounded-lg text-white focus:outline-none focus:border-cyan-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full px-4 py-2 bg-[#0a0e17] border border-[#1f2937] rounded-lg text-white focus:outline-none focus:border-cyan-500"
                required
                minLength={8}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Confirm Password</label>
              <input
                type="password"
                value={confirm}
                onChange={e => setConfirm(e.target.value)}
                className="w-full px-4 py-2 bg-[#0a0e17] border border-[#1f2937] rounded-lg text-white focus:outline-none focus:border-cyan-500"
                required
                minLength={8}
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 cursor-pointer"
            >
              {loading ? 'Creating account...' : 'Create Account'}
            </button>
          </form>

          <p className="text-gray-500 text-sm mt-4 text-center">
            Already have an account?{' '}
            <Link to="/login" className="text-cyan-400 hover:text-cyan-300">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
