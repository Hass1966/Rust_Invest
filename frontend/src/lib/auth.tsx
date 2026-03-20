import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import { setAuthToken } from './api'

const BASE = ''

export interface User {
  id: number
  email: string
}

interface AuthState {
  user: User | null
  token: string | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // Check for existing session on mount
  useEffect(() => {
    // Token is stored in memory only — no localStorage
    setLoading(false)
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const res = await fetch(`${BASE}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Login failed')
    setToken(data.token)
    setAuthToken(data.token)
    setUser(data.user)
  }, [])

  const register = useCallback(async (email: string, password: string) => {
    const res = await fetch(`${BASE}/api/v1/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Registration failed')
    setToken(data.token)
    setAuthToken(data.token)
    setUser(data.user)
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setAuthToken(null)
    setUser(null)
    fetch(`${BASE}/api/v1/auth/logout`, { method: 'POST' }).catch(() => {})
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}

/** Get auth headers for API calls */
export function authHeaders(token: string | null): Record<string, string> {
  if (!token) return {}
  return { Authorization: `Bearer ${token}` }
}
