import type { EnrichedSignal, ModelManifest, PortfolioResult, DailyTrackerResult, Hint, SimResult, TrainingResults } from './types'

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

export async function fetchDailyTracker(): Promise<DailyTrackerResult> {
  const res = await fetch(`${BASE}/api/v1/portfolio/daily-tracker`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchSimulation(days: number, capital: number): Promise<SimResult> {
  const res = await fetch(`${BASE}/api/v1/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ days, capital }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchHints(): Promise<Hint[]> {
  const res = await fetch(`${BASE}/api/v1/hints`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchCryptoSignals(): Promise<EnrichedSignal[]> {
  const res = await fetch(`${BASE}/api/v1/signals/current/crypto`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchTrainingResults(): Promise<TrainingResults> {
  const res = await fetch(`${BASE}/api/v1/training/results`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchMorningBriefing(): Promise<string> {
  const res = await fetch(`${BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'morning_briefing', tab_context: 'overview' }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.response
}

export async function fetchPredictions(): Promise<PredictionsData> {
  const res = await fetch(`${BASE}/api/v1/predictions/history?limit=500`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export interface PredictionsData {
  predictions: PredictionRecord[]
  stats: {
    total_predictions: number
    total_resolved: number
    total_correct: number
    overall_accuracy: number
    last_24h: AccuracyStats
    last_7d: AccuracyStats
    last_30d: AccuracyStats
  }
  per_asset: AssetAccuracy[]
  confidence_bands: ConfidenceBand[]
}

export interface PredictionRecord {
  id: number
  timestamp: string
  asset: string
  signal: string
  confidence: number
  price_at_prediction: number
  actual_direction: string | null
  was_correct: boolean | null
  price_at_outcome: number | null
  outcome_timestamp: string | null
}

export interface AccuracyStats {
  predictions: number
  resolved: number
  correct: number
  accuracy: number
}

export interface AssetAccuracy {
  asset: string
  correct: number
  total: number
  accuracy: number
}

export interface ConfidenceBand {
  band: string
  predictions: number
  correct: number
  accuracy: number
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
