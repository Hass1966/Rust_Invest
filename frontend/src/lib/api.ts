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

export interface SignalTruthData {
  total_signals: number
  total_resolved: number
  total_pending: number
  total_correct: number
  overall_accuracy: number
  by_signal_type: { signal_type: string; correct: number; total: number; accuracy: number }[]
  by_asset_class: { asset_class: string; correct: number; total: number; accuracy: number }[]
  rolling: {
    today: { resolved: number; correct: number; accuracy: number }
    this_week: { resolved: number; correct: number; accuracy: number }
    all_time: { resolved: number; correct: number; accuracy: number }
  }
  per_asset: { asset: string; correct: number; total: number; accuracy: number }[]
  signals: SignalTruthRecord[]
}

export interface SignalTruthRecord {
  id: number
  timestamp: string
  asset: string
  asset_class: string
  signal_type: string
  price_at_signal: number
  confidence: number
  linreg_prob: number | null
  logreg_prob: number | null
  gbt_prob: number | null
  outcome_price: number | null
  pct_change: number | null
  was_correct: boolean | null
  resolution_ts: string | null
}

export async function fetchSignalTruth(): Promise<SignalTruthData> {
  const res = await fetch(`${BASE}/api/v1/signals/truth?limit=1000`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── Historical Signal Accuracy ──

export interface HistoricalSignalAccuracy {
  has_data: boolean
  note?: string
  frequency: string
  total_signals: number
  total_resolved: number
  total_pending: number
  total_correct: number
  overall_accuracy: number
  by_signal_type: { signal_type: string; correct: number; total: number; total_including_pending: number; accuracy: number }[]
  by_asset_class: { asset_class: string; correct: number; total: number; accuracy: number }[]
  per_asset: { asset: string; asset_class: string; correct: number; total: number; total_signals: number; accuracy: number; date_from: string; date_to: string }[]
  monthly_accuracy: { month: string; correct: number; total: number; accuracy: number }[]
  generated_at: string
}

export async function fetchHistoricalSignalAccuracy(frequency: string = 'weekly'): Promise<HistoricalSignalAccuracy> {
  const res = await fetch(`${BASE}/api/v1/signals/truth/historical?frequency=${frequency}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── User Portfolio Tracker ──

export interface UserHolding {
  id: number
  symbol: string
  quantity: number
  start_date: string
  asset_class: string
  created_at: string | null
}

export interface EquityCurvePoint {
  date: string
  signal_value: number
  buy_hold_value: number
}

export interface AssetComparison {
  symbol: string
  asset_class: string
  quantity: number
  start_date: string
  actual_start_date: string
  start_price: number
  current_price: number
  cost_basis: number
  buy_hold_value: number
  buy_hold_return_pct: number
  signal_value: number
  signal_return_pct: number
  signals_used: number
  total_trades: number
  win_rate_pct: number
  sharpe_signals: number
  sharpe_buy_hold: number
  equity_curve: EquityCurvePoint[]
  note: string | null
}

export interface PortfolioComparison {
  has_data: boolean
  note?: string
  frequency?: string
  total_cost?: number
  buy_hold_value?: number
  buy_hold_return_pct?: number
  signal_value?: number
  signal_return_pct?: number
  verdict?: 'signals_win' | 'buy_hold_wins' | 'roughly_equal'
  sharpe_signals?: number
  sharpe_buy_hold?: number
  overall_win_rate_pct?: number
  total_trades?: number
  equity_curve?: EquityCurvePoint[]
  per_asset?: AssetComparison[]
}

export async function fetchUserHoldings(): Promise<UserHolding[]> {
  const res = await fetch(`${BASE}/api/v1/user-portfolio`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function addUserHolding(symbol: string, quantity: number, start_date: string): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/user-portfolio`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, quantity, start_date }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
}

export async function updateUserHolding(id: number, quantity: number, start_date: string): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/user-portfolio/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ quantity, start_date }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
}

export async function deleteUserHolding(id: number): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/user-portfolio/${id}`, { method: 'DELETE' })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Server returned ${res.status}`)
  }
}

export async function comparePortfolio(frequency: string = 'weekly'): Promise<PortfolioComparison> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 120_000)
  try {
    const res = await fetch(`${BASE}/api/v1/user-portfolio/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frequency }),
      signal: controller.signal,
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
  } finally {
    clearTimeout(timeout)
  }
}

// ── Asset Config ──

export interface AssetEntry {
  symbol: string
  name: string
  enabled: boolean
}

export interface AssetConfig {
  stocks: AssetEntry[]
  fx: AssetEntry[]
  crypto: AssetEntry[]
}

export async function fetchAssetConfig(): Promise<AssetConfig> {
  const res = await fetch(`${BASE}/api/v1/config/assets`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── Feedback ──

export async function submitSignalFeedback(asset: string, signalType: string, reaction: 'up' | 'down'): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/feedback/signal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ asset, signal_type: signalType, reaction }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
}

export interface SurveyFeedback {
  q_understand?: string
  q_check_daily?: string
  q_trust_more?: string
  q_missing?: string
  q_would_pay?: string
}

export async function submitSurveyFeedback(survey: SurveyFeedback): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/feedback/survey`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(survey),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
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
