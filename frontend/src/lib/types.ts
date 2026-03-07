export interface ModelDetail {
  probability_up: number
  weight: number
  vote: string
}

export interface RiskContext {
  volatility_regime: string
  drawdown_risk: string
  trend_strength: string
  days_to_earnings: number | null
}

export interface TechnicalDetail {
  confidence: number
  probability_up: number
  model_agreement: string
  rsi: number
  trend: string
  bb_position: number | null
  quality: string
  walk_forward_accuracy: number
}

export interface EnrichedSignal {
  asset: string
  asset_class: string
  signal: string
  reason: string
  risk_context: RiskContext
  suggested_action: string
  technical: TechnicalDetail
  models: Record<string, ModelDetail>
  price: number
  timestamp: string
}

export interface ManifestAsset {
  linreg_accuracy: number | null
  logreg_accuracy: number | null
  gbt_accuracy: number | null
  ensemble_accuracy: number | null
  last_trained: string | null
  weights_present: boolean
}

export interface ModelManifest {
  version: number
  generated_at: string
  assets: Record<string, ManifestAsset>
}

export interface PortfolioAllocation {
  asset: string
  asset_class: string
  weight: number
  allocated: number
  return: number
  contribution: number
  sharpe: number
  signal: string
}

export interface StrategyResult {
  final_value: number
  total_return: number
  annualised_return: number
  benchmark_return: number
  excess_return: number
  sharpe_ratio: number
  max_drawdown: number
  volatility: number
  n_assets: number
  allocations: PortfolioAllocation[]
}

export interface AssetBacktest {
  asset: string
  asset_class: string
  total_return: number
  buy_hold_return: number
  excess_return: number
  annualised_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  profit_factor: number
  expectancy: number
  days_in_market: number
  total_days: number
  verdict: string
}

export interface PortfolioResult {
  starting_capital: number
  has_data: boolean
  note?: string
  strategies?: Record<string, StrategyResult>
  per_asset_backtest?: AssetBacktest[]
}

export interface TrackerSignal {
  asset: string
  signal: string
  weight: number
  price_return: number
  contribution: number
}

export interface EquityCurvePoint {
  date: string
  value: number
  daily_return: number
}

export interface DailyTrackerResult {
  has_data: boolean
  note?: string
  seed_value?: number
  current_value?: number
  daily_return?: number
  cumulative_return?: number
  inception_date?: string
  last_updated?: string
  days_tracked?: number
  model_accuracy_pct?: number
  today_signals?: TrackerSignal[]
  equity_curve?: EquityCurvePoint[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Hint {
  asset: string
  category: string
  urgency: string
  title: string
  reason: string
  what_it_means: string
  suggested_pct: number
}
