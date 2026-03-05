import type { EnrichedSignal, ModelManifest, PortfolioResult } from './types'

const BASE = ''

export async function fetchSignals(): Promise<EnrichedSignal[]> {
  const res = await fetch(`${BASE}/api/v1/signals/current`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchStockSignals(): Promise<EnrichedSignal[]> {
  const res = await fetch(`${BASE}/api/v1/signals/current/stocks`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchFxSignals(): Promise<EnrichedSignal[]> {
  const res = await fetch(`${BASE}/api/v1/signals/current/fx`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchModels(): Promise<ModelManifest> {
  const res = await fetch(`${BASE}/api/v1/models/current`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function reloadModels(): Promise<{ status: string; assets_found: number }> {
  const res = await fetch(`${BASE}/api/v1/models/reload`, { method: 'POST' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchPortfolio(): Promise<PortfolioResult> {
  const res = await fetch(`${BASE}/api/v1/portfolio/simulate`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function sendChat(message: string, tabContext: string): Promise<string> {
  const res = await fetch(`${BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, tab_context: tabContext }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.response
}
