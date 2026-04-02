import { useEffect, useState, useMemo, useCallback } from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import { Loader2, Plus, Trash2 } from 'lucide-react'
import { fetchSimulation, fetchWalkForwardData, fetchManagedSimulation } from '../lib/api'
import type { WalkForwardData, ManagedSimData, ManagedSimSignal } from '../lib/api'
import type { SimResult as WhatIfResult } from '../lib/types'

// ─── Types ───

interface PricePoint { date: string; price: number }
interface SignalPoint { date: string; signal: string; price: number; was_correct: boolean | null; outcome_price: number | null }
interface SimulatorData {
  price_history: Record<string, PricePoint[]>
  signal_history: Record<string, SignalPoint[]>
}

// ─── Config ───

const LIVE_START = '2026-03-15'
const AVAILABLE_ASSETS = [
  'AAPL', 'MSFT', 'JPM', 'XOM', 'BRK-B',          // US stocks (30%)
  'HSBA.L', 'AZN.L',                                 // UK stocks (10%)
  'TLT', 'AGG', 'BND',                               // Bonds (15%)
  'GLD',                                              // Gold (8%)
  'USO', 'CPER',                                      // Commodities (4%)
  'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURJPY=X',   // FX (13%)
  'bitcoin', 'ethereum',                              // Crypto (10%)
]
const STARTING_CAPITAL = 100000
const DEFAULT_BH_WEIGHTS: { asset: string; weight: number }[] = [
  // US Stocks (30%)
  { asset: 'AAPL', weight: 0.06 },
  { asset: 'MSFT', weight: 0.06 },
  { asset: 'JPM', weight: 0.06 },
  { asset: 'XOM', weight: 0.06 },
  { asset: 'BRK-B', weight: 0.06 },
  // UK Stocks (10%)
  { asset: 'HSBA.L', weight: 0.05 },
  { asset: 'AZN.L', weight: 0.05 },
  // Bonds (15%)
  { asset: 'TLT', weight: 0.08 },
  { asset: 'AGG', weight: 0.04 },
  { asset: 'BND', weight: 0.03 },
  // Gold (8%)
  { asset: 'GLD', weight: 0.08 },
  // Commodities (4%)
  { asset: 'USO', weight: 0.02 },
  { asset: 'CPER', weight: 0.02 },
  // FX (13%)
  { asset: 'EURUSD=X', weight: 0.04 },
  { asset: 'GBPUSD=X', weight: 0.04 },
  { asset: 'USDJPY=X', weight: 0.03 },
  { asset: 'EURJPY=X', weight: 0.02 },
  // Crypto (10%)
  { asset: 'bitcoin', weight: 0.05 },
  { asset: 'ethereum', weight: 0.05 },
]
const MAX_CUSTOM = 5

interface Allocation { asset: string; pct: number }

// ─── Portfolio Allocation Types ───

interface PortfolioState {
  cash: number
  positions: Record<string, number>   // asset → shares held
  weights: Record<string, number>     // asset → current target weight [0,1]
}

interface DaySnapshot {
  date: string
  totalValue: number
  cashValue: number
  cashPct: number
  positions: { asset: string; shares: number; value: number; weight: number; targetWeight: number; signal: string }[]
  txCostsToday: number
  cumulativeTxCosts: number
}

interface AllocationChartPoint {
  date: string
  cash: number
  [asset: string]: number | string
}

const VOL_LOOKBACK = 60
const MIN_VOL_DAYS = 20
const MAX_SINGLE_WEIGHT = 0.30
const REBALANCE_THRESHOLD = 0.02
const CORR_THRESHOLD = 0.7
const CORR_PENALTY = 0.2

// ─── Helpers ───

function fmtGBP(n: number): string { return '\u00a3' + Math.round(n).toLocaleString() }
function fmtPct(n: number): string { return (n >= 0 ? '+' : '') + n.toFixed(2) + '%' }

function buildPriceMap(points: PricePoint[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const p of points) m.set(p.date, p.price)
  return m
}

function getPrice(priceMap: Map<string, number>, date: string): number | null {
  const p = priceMap.get(date)
  if (p !== undefined) return p
  let best: number | null = null
  let bestDate = ''
  for (const [d, v] of priceMap) {
    if (d <= date && d > bestDate) { best = v; bestDate = d }
  }
  return best
}

function getAllDates(priceHistory: Record<string, PricePoint[]>, fromDate?: string): string[] {
  const dateSet = new Set<string>()
  for (const points of Object.values(priceHistory)) {
    for (const p of points) {
      if (!fromDate || p.date >= fromDate) dateSet.add(p.date)
    }
  }
  return Array.from(dateSet).sort()
}

// ─── Risk metric helpers ───

const RISK_FREE_DAILY = 0.045 / 252 // 4.5% annual risk-free rate

function dailyReturns(series: number[]): number[] {
  const r: number[] = []
  for (let i = 1; i < series.length; i++) {
    r.push(series[i - 1] > 0 ? (series[i] - series[i - 1]) / series[i - 1] : 0)
  }
  return r
}

function sharpeRatio(returns: number[]): number {
  if (returns.length < 2) return 0
  const mean = returns.reduce((s, v) => s + v, 0) / returns.length
  const variance = returns.reduce((s, v) => s + (v - mean) ** 2, 0) / returns.length
  const std = Math.sqrt(variance)
  if (std === 0) return 0
  return ((mean - RISK_FREE_DAILY) / std) * Math.sqrt(252) // annualised
}

function maxDrawdown(series: number[]): number {
  let peak = series[0] || 0
  let maxDd = 0
  for (const v of series) {
    if (v > peak) peak = v
    const dd = peak > 0 ? (peak - v) / peak : 0
    if (dd > maxDd) maxDd = dd
  }
  return maxDd * 100 // as percentage
}

// ─── Portfolio Allocation Helpers ───

function computeTrailingVol(priceMap: Map<string, number>, dates: string[], dateIdx: number): number {
  const start = Math.max(0, dateIdx - VOL_LOOKBACK)
  const logReturns: number[] = []
  let prevPrice: number | null = null
  for (let i = start; i <= dateIdx; i++) {
    const p = priceMap.get(dates[i])
    if (p && prevPrice && prevPrice > 0 && p > 0) {
      logReturns.push(Math.log(p / prevPrice))
    }
    if (p) prevPrice = p
  }
  if (logReturns.length < MIN_VOL_DAYS) return Infinity
  const mean = logReturns.reduce((s, v) => s + v, 0) / logReturns.length
  const variance = logReturns.reduce((s, v) => s + (v - mean) ** 2, 0) / logReturns.length
  return Math.sqrt(variance)
}

function pearsonCorrelation(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length)
  if (n < 10) return 0
  const meanA = a.slice(0, n).reduce((s, v) => s + v, 0) / n
  const meanB = b.slice(0, n).reduce((s, v) => s + v, 0) / n
  let cov = 0, varA = 0, varB = 0
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA
    const db = b[i] - meanB
    cov += da * db
    varA += da * da
    varB += db * db
  }
  const denom = Math.sqrt(varA * varB)
  return denom > 0 ? cov / denom : 0
}

function computeTargetWeights(
  assets: string[],
  dateIdx: number,
  dates: string[],
  priceMaps: Record<string, Map<string, number>>,
  lastSignal: Record<string, string>,
  prevWeights: Record<string, number>,
): Record<string, number> {
  const weights: Record<string, number> = {}
  const invVols: Record<string, number> = {}
  let totalInvVol = 0

  for (const asset of assets) {
    const sig = lastSignal[asset] || 'HOLD'
    if (sig === 'SELL' || sig === 'SHORT') {
      weights[asset] = 0
    } else if (sig === 'HOLD') {
      weights[asset] = prevWeights[asset] ?? 0
    } else {
      // BUY: use inverse volatility
      const vol = computeTrailingVol(priceMaps[asset], dates, dateIdx)
      if (vol === Infinity || vol <= 0) {
        invVols[asset] = 1 // fallback: equal weight
      } else {
        invVols[asset] = 1 / vol
      }
      totalInvVol += invVols[asset]
    }
  }

  // Distribute remaining weight (1 - holdWeight - 0) among BUY assets via inverse-vol
  const holdWeight = Object.values(weights).reduce((s, w) => s + w, 0)
  const availableWeight = Math.max(0, 1 - holdWeight)

  if (totalInvVol > 0) {
    for (const asset of assets) {
      if (invVols[asset] !== undefined) {
        weights[asset] = (invVols[asset] / totalInvVol) * availableWeight
      }
    }
  }

  // Apply single-asset cap and re-normalize
  let excess = 0
  let uncappedCount = 0
  for (const asset of assets) {
    if ((weights[asset] || 0) > MAX_SINGLE_WEIGHT) {
      excess += weights[asset] - MAX_SINGLE_WEIGHT
      weights[asset] = MAX_SINGLE_WEIGHT
    } else if ((weights[asset] || 0) > 0) {
      uncappedCount++
    }
  }
  if (excess > 0 && uncappedCount > 0) {
    const boost = excess / uncappedCount
    for (const asset of assets) {
      if ((weights[asset] || 0) > 0 && weights[asset] < MAX_SINGLE_WEIGHT) {
        weights[asset] = Math.min(MAX_SINGLE_WEIGHT, weights[asset] + boost)
      }
    }
  }

  return weights
}

function applyCorrelationPenalty(
  weights: Record<string, number>,
  dateIdx: number,
  dates: string[],
  priceMaps: Record<string, Map<string, number>>,
): Record<string, number> {
  const assets = Object.keys(weights).filter(a => weights[a] > 0)
  if (assets.length < 2) return weights

  // Build return arrays for lookback window
  const start = Math.max(0, dateIdx - VOL_LOOKBACK)
  const returnArrays: Record<string, number[]> = {}
  for (const asset of assets) {
    const rets: number[] = []
    let prev: number | null = null
    for (let i = start; i <= dateIdx; i++) {
      const p = priceMaps[asset]?.get(dates[i])
      if (p && prev && prev > 0) {
        rets.push((p - prev) / prev)
      }
      if (p) prev = p
    }
    returnArrays[asset] = rets
  }

  const penalized = { ...weights }
  for (let i = 0; i < assets.length; i++) {
    for (let j = i + 1; j < assets.length; j++) {
      const corr = pearsonCorrelation(returnArrays[assets[i]], returnArrays[assets[j]])
      if (Math.abs(corr) > CORR_THRESHOLD) {
        // Shrink the smaller-weighted asset
        const [larger, smaller] = penalized[assets[i]] >= penalized[assets[j]]
          ? [assets[i], assets[j]] : [assets[j], assets[i]]
        penalized[smaller] *= (1 - CORR_PENALTY)
        void larger // keep larger unchanged
      }
    }
  }

  // Re-normalize to sum to original total
  const origTotal = Object.values(weights).reduce((s, w) => s + w, 0)
  const newTotal = Object.values(penalized).reduce((s, w) => s + w, 0)
  if (newTotal > 0 && origTotal > 0) {
    const scale = origTotal / newTotal
    for (const asset of Object.keys(penalized)) {
      penalized[asset] *= scale
    }
  }

  return penalized
}

function txCostBps(asset: string): number {
  const cryptoAssets = new Set(['bitcoin', 'ethereum', 'solana', 'dogecoin', 'cardano', 'xrp', 'polkadot', 'chainlink', 'avalanche', 'polygon', 'uniswap', 'litecoin', 'stellar', 'cosmos', 'algorand'])
  return cryptoAssets.has(asset.toLowerCase()) ? 0.0025 : 0.001
}

function rebalancePortfolio(
  portfolio: PortfolioState,
  targetWeights: Record<string, number>,
  priceMaps: Record<string, Map<string, number>>,
  date: string,
  assets: string[],
): number {
  // Calculate current portfolio value
  let totalValue = portfolio.cash
  for (const asset of assets) {
    const price = getPrice(priceMaps[asset], date)
    if (price && portfolio.positions[asset]) {
      totalValue += portfolio.positions[asset] * price
    }
  }
  if (totalValue <= 0) return 0

  let txCosts = 0

  // Phase 1: Sell overweight positions first (frees cash)
  for (const asset of assets) {
    const price = getPrice(priceMaps[asset], date)
    if (!price) continue
    const currentValue = (portfolio.positions[asset] || 0) * price
    const currentWeight = currentValue / totalValue
    const target = targetWeights[asset] || 0

    if (currentWeight > target + REBALANCE_THRESHOLD) {
      const sellValue = (currentWeight - target) * totalValue
      const sellShares = sellValue / price
      const cost = sellValue * txCostBps(asset)
      portfolio.positions[asset] = Math.max(0, (portfolio.positions[asset] || 0) - sellShares)
      portfolio.cash += sellValue - cost
      txCosts += cost
    }
  }

  // Phase 2: Buy underweight positions
  for (const asset of assets) {
    const price = getPrice(priceMaps[asset], date)
    if (!price) continue
    const currentValue = (portfolio.positions[asset] || 0) * price
    const currentWeight = currentValue / totalValue
    const target = targetWeights[asset] || 0

    if (target > currentWeight + REBALANCE_THRESHOLD) {
      const buyValue = Math.min((target - currentWeight) * totalValue, portfolio.cash)
      if (buyValue <= 0) continue
      const cost = buyValue * txCostBps(asset)
      const netBuyValue = buyValue - cost
      const buyShares = netBuyValue / price
      portfolio.positions[asset] = (portfolio.positions[asset] || 0) + buyShares
      portfolio.cash -= buyValue
      txCosts += cost
    }
  }

  portfolio.weights = { ...targetWeights }
  return txCosts
}

// ─── Simulation Logic ───

interface SimResult {
  chartData: { date: string; buyHold: number; alphaSignal: number; spy: number }[]
  bhBreakdown: { asset: string; invested: number; currentValue: number; returnPct: number }[]
  startingCapital: number
  bhTotal: number
  bhReturn: number
  asTotal: number
  asReturn: number
  spyTotal: number
  spyReturn: number
  asTotalSignals: number
  asCorrect: number
  asIncorrect: number
  asBest: { asset: string; returnPct: number } | null
  asWorst: { asset: string; returnPct: number } | null
  bhSharpe: number
  asSharpe: number
  bhMaxDrawdown: number
  asMaxDrawdown: number
  allocationHistory: DaySnapshot[]
  cumulativeTxCosts: number
  allocationChartData: AllocationChartPoint[]
}

function runSimulation(
  data: SimulatorData,
  bhAssets: { asset: string; amount: number }[],
  capital: number,
  fromDate?: string,
): SimResult | null {
  const dates = getAllDates(data.price_history, fromDate)
  if (dates.length < 2) return null

  const priceMaps: Record<string, Map<string, number>> = {}
  for (const [k, v] of Object.entries(data.price_history)) {
    priceMaps[k] = buildPriceMap(v)
  }

  const startDate = dates[0]
  const assetNames = bhAssets.map(a => a.asset)

  // ── Buy & Hold: allocate capital proportionally ──
  const bhShares: Record<string, number> = {}
  for (const { asset, amount } of bhAssets) {
    const startPrice = getPrice(priceMaps[asset], startDate)
    bhShares[asset] = startPrice ? amount / startPrice : 0
  }

  // ── Alpha Signal: Unified Portfolio (same starting capital) ──
  const portfolio: PortfolioState = {
    cash: capital,
    positions: {},
    weights: {},
  }
  for (const asset of assetNames) {
    portfolio.positions[asset] = 0
    portfolio.weights[asset] = 0
  }

  // Pre-process signals by asset and date
  const signalsByAssetDate: Record<string, Map<string, string>> = {}
  for (const [asset, signals] of Object.entries(data.signal_history)) {
    const m = new Map<string, string>()
    for (const s of signals) {
      if (!fromDate || s.date >= fromDate) m.set(s.date, s.signal)
    }
    signalsByAssetDate[asset] = m
  }

  // Count signals for selected assets only
  let totalSignals = 0
  let correctSignals = 0
  let incorrectSignals = 0
  const selectedAssets = new Set(assetNames)
  for (const [asset, signals] of Object.entries(data.signal_history)) {
    if (!selectedAssets.has(asset)) continue
    for (const s of signals) {
      if (fromDate && s.date < fromDate) continue
      totalSignals++
      if (s.was_correct === true) correctSignals++
      else if (s.was_correct === false) incorrectSignals++
    }
  }

  const chartData: SimResult['chartData'] = []
  const allocationHistory: DaySnapshot[] = []
  const lastSignal: Record<string, string> = {}
  let cumulativeTxCosts = 0

  // Initial buy on day 0: equal weight across all assets (treated as BUY)
  for (const asset of assetNames) {
    lastSignal[asset] = 'BUY'
  }
  const initWeights = computeTargetWeights(assetNames, 0, dates, priceMaps, lastSignal, portfolio.weights)
  const initCorrWeights = applyCorrelationPenalty(initWeights, 0, dates, priceMaps)
  const initTxCost = rebalancePortfolio(portfolio, initCorrWeights, priceMaps, dates[0], assetNames)
  cumulativeTxCosts += initTxCost

  for (let di = 0; di < dates.length; di++) {
    const date = dates[di]

    // 1. Update lastSignal map from today's signals
    for (const asset of assetNames) {
      const signal = signalsByAssetDate[asset]?.get(date)
      if (signal) lastSignal[asset] = signal
    }

    // 2. Compute target weights (skip day 0 — already initialized)
    let txCostsToday = di === 0 ? initTxCost : 0
    if (di > 0) {
      const targetWeights = computeTargetWeights(assetNames, di, dates, priceMaps, lastSignal, portfolio.weights)
      const corrWeights = applyCorrelationPenalty(targetWeights, di, dates, priceMaps)
      txCostsToday = rebalancePortfolio(portfolio, corrWeights, priceMaps, date, assetNames)
      cumulativeTxCosts += txCostsToday
    }

    // 3. Value portfolio
    let asValue = portfolio.cash
    const positionDetails: DaySnapshot['positions'] = []
    for (const asset of assetNames) {
      const price = getPrice(priceMaps[asset], date)
      const shares = portfolio.positions[asset] || 0
      const value = price ? shares * price : 0
      asValue += value
      positionDetails.push({
        asset,
        shares,
        value,
        weight: 0, // filled after total known
        targetWeight: portfolio.weights[asset] || 0,
        signal: lastSignal[asset] || 'HOLD',
      })
    }
    // Fill actual weights
    for (const pos of positionDetails) {
      pos.weight = asValue > 0 ? pos.value / asValue : 0
    }

    allocationHistory.push({
      date,
      totalValue: asValue,
      cashValue: portfolio.cash,
      cashPct: asValue > 0 ? portfolio.cash / asValue : 0,
      positions: positionDetails,
      txCostsToday,
      cumulativeTxCosts,
    })

    // 4. Buy & Hold value
    let bhValue = 0
    for (const { asset } of bhAssets) {
      const price = getPrice(priceMaps[asset], date)
      bhValue += price ? bhShares[asset] * price : 0
    }

    // 5. SPY benchmark (normalised to same starting capital)
    const spyPrice = getPrice(priceMaps['SPY'], date)
    const spyStartPrice = getPrice(priceMaps['SPY'], startDate)
    const spyValue = spyStartPrice && spyPrice ? capital * (spyPrice / spyStartPrice) : capital

    chartData.push({ date, buyHold: Math.round(bhValue), alphaSignal: Math.round(asValue), spy: Math.round(spyValue) })
  }

  // Compute allocation chart data (percentages)
  const allocationChartData: AllocationChartPoint[] = allocationHistory.map(snap => {
    const point: AllocationChartPoint = { date: snap.date, cash: snap.cashPct * 100 }
    for (const pos of snap.positions) {
      point[pos.asset] = pos.weight * 100
    }
    return point
  })

  const lastDate = dates[dates.length - 1]
  const bhBreakdown: SimResult['bhBreakdown'] = bhAssets.map(({ asset, amount }) => {
    const price = getPrice(priceMaps[asset], lastDate)
    const currentValue = price ? bhShares[asset] * price : 0
    return { asset, invested: amount, currentValue: Math.round(currentValue), returnPct: amount > 0 ? ((currentValue - amount) / amount) * 100 : 0 }
  })

  const last = chartData[chartData.length - 1]
  const bhTotal = last.buyHold
  const asTotal = last.alphaSignal
  const spyTotal = last.spy

  // Best/worst from final allocation snapshot
  const finalSnap = allocationHistory[allocationHistory.length - 1]
  const asPerAsset = finalSnap.positions.map(pos => ({
    asset: pos.asset,
    returnPct: capital > 0 && pos.value > 0 ? ((pos.value / (pos.targetWeight * capital || 1)) - 1) * 100 : 0,
  }))
  const asBest = asPerAsset.length ? asPerAsset.reduce((a, b) => a.returnPct > b.returnPct ? a : b) : null
  const asWorst = asPerAsset.length ? asPerAsset.reduce((a, b) => a.returnPct < b.returnPct ? a : b) : null

  const bhValues = chartData.map(d => d.buyHold)
  const asValues = chartData.map(d => d.alphaSignal)

  return {
    chartData,
    bhBreakdown,
    startingCapital: capital,
    bhTotal,
    bhReturn: capital > 0 ? ((bhTotal - capital) / capital) * 100 : 0,
    asTotal,
    asReturn: capital > 0 ? ((asTotal - capital) / capital) * 100 : 0,
    spyTotal,
    spyReturn: capital > 0 ? ((spyTotal - capital) / capital) * 100 : 0,
    asTotalSignals: totalSignals,
    asCorrect: correctSignals,
    asIncorrect: incorrectSignals,
    asBest,
    asWorst,
    bhSharpe: sharpeRatio(dailyReturns(bhValues)),
    asSharpe: sharpeRatio(dailyReturns(asValues)),
    bhMaxDrawdown: maxDrawdown(bhValues),
    asMaxDrawdown: maxDrawdown(asValues),
    allocationHistory,
    cumulativeTxCosts,
    allocationChartData,
  }
}

// ─── Managed Portfolio Simulation ───

interface ManagedTrade {
  date: string
  action: 'BUY' | 'SELL'
  asset: string
  price: number
  shares: number
  value: number
  reason: string
}

interface ManagedHolding {
  asset: string
  shares: number
  avgCost: number
  currentPrice: number
  value: number
  pnlPct: number
}

interface ManagedResult {
  chartData: { date: string; managed: number; buyHold: number; spy: number }[]
  startingCapital: number
  managedTotal: number
  managedReturn: number
  bhTotal: number
  bhReturn: number
  spyTotal: number
  spyReturn: number
  totalTrades: number
  numHoldings: number
  cashPosition: number
  cashPct: number
  bestTrade: { asset: string; pct: number } | null
  worstTrade: { asset: string; pct: number } | null
  holdings: ManagedHolding[]
  trades: ManagedTrade[]
  managedSharpe: number
  bhSharpe: number
  managedMaxDrawdown: number
  bhMaxDrawdown: number
  cumulativeTxCosts: number
}

const MANAGED_TX_COST = 0.002 // 0.2%
const MANAGED_MAX_POSITION = 0.15 // 15% cap
const MANAGED_MAX_BUYS = 8
const MANAGED_CONFIDENCE_MIN = 10 // 10% minimum confidence

function runManagedSimulation(
  managedData: ManagedSimData,
  capital: number,
): ManagedResult | null {
  const { price_history, signals } = managedData

  // Build price maps for all assets
  const priceMaps: Record<string, Map<string, number>> = {}
  for (const [asset, points] of Object.entries(price_history)) {
    priceMaps[asset] = buildPriceMap(points)
  }

  // Get all dates from price history (using all assets), sorted
  const dateSet = new Set<string>()
  for (const points of Object.values(price_history)) {
    for (const p of points) dateSet.add(p.date)
  }
  const dates = Array.from(dateSet).sort()
  if (dates.length < 2) return null

  // Group signals by date
  const signalsByDate: Record<string, ManagedSimSignal[]> = {}
  for (const s of signals) {
    if (!signalsByDate[s.date]) signalsByDate[s.date] = []
    signalsByDate[s.date].push(s)
  }

  // Default BH assets
  const bhAssets = [
    { asset: 'AAPL', weight: 0.1 }, { asset: 'MSFT', weight: 0.1 }, { asset: 'GOOGL', weight: 0.1 },
    { asset: 'JPM', weight: 0.1 }, { asset: 'HSBA.L', weight: 0.1 }, { asset: 'AZN.L', weight: 0.1 },
    { asset: 'XOM', weight: 0.1 }, { asset: 'GLD', weight: 0.1 },
    { asset: 'bitcoin', weight: 0.1 }, { asset: 'ethereum', weight: 0.1 },
  ]

  // Helper: get first available price for an asset (handles weekends/holidays)
  function getFirstPrice(pm: Map<string, number>): number | null {
    let earliest = ''
    let price: number | null = null
    for (const [d, v] of pm) {
      if (!earliest || d < earliest) { earliest = d; price = v }
    }
    return price
  }

  // Initialize BH positions — use first available price per asset (handles weekend start dates)
  const bhShares: Record<string, number> = {}
  for (const bh of bhAssets) {
    const pm = priceMaps[bh.asset]
    if (!pm) continue
    const startPrice = getPrice(pm, dates[0]) ?? getFirstPrice(pm)
    if (startPrice && startPrice > 0) {
      bhShares[bh.asset] = (capital * bh.weight) / startPrice
    }
  }

  // Managed portfolio state
  let cash = 0
  const positions: Record<string, number> = {} // asset -> shares
  const avgCosts: Record<string, number> = {} // asset -> average cost basis
  const trades: ManagedTrade[] = []
  let cumulativeTxCosts = 0
  const tradeReturns: { asset: string; pct: number }[] = []

  // Day 0: buy initial 10 assets equally — use first available price per asset
  for (const bh of bhAssets) {
    const pm = priceMaps[bh.asset]
    if (!pm) { cash += capital * bh.weight; continue }
    const startPrice = getPrice(pm, dates[0]) ?? getFirstPrice(pm)
    if (startPrice && startPrice > 0) {
      const investAmount = capital * bh.weight
      const txCost = investAmount * MANAGED_TX_COST
      const netAmount = investAmount - txCost
      const shares = netAmount / startPrice
      positions[bh.asset] = shares
      avgCosts[bh.asset] = startPrice
      cumulativeTxCosts += txCost
      trades.push({
        date: dates[0], action: 'BUY', asset: bh.asset,
        price: startPrice, shares, value: investAmount,
        reason: 'Initial allocation (10% equal weight)',
      })
    } else {
      cash += capital * bh.weight
    }
  }

  // Track last known signal per asset
  const lastSignal: Record<string, { signal: string; confidence: number }> = {}

  const chartData: { date: string; managed: number; buyHold: number; spy: number }[] = []
  const spyMap = priceMaps['SPY']
  const spyStart = spyMap ? (getPrice(spyMap, dates[0]) ?? getFirstPrice(spyMap)) : null

  for (let di = 0; di < dates.length; di++) {
    const date = dates[di]
    const daySignals = signalsByDate[date] || []

    // Update last known signals
    for (const s of daySignals) {
      lastSignal[s.asset] = { signal: s.signal, confidence: s.confidence * 100 }
    }

    if (di > 0) {
      // Compute portfolio value before rebalance
      let totalValue = cash
      for (const [asset, shares] of Object.entries(positions)) {
        const pm = priceMaps[asset]
        const price = pm ? getPrice(pm, date) : null
        if (price) totalValue += shares * price
      }

      // Phase 1: SELL — sell assets with SELL/SHORT signals and confidence > 10%
      const sellAssets: string[] = []
      for (const s of daySignals) {
        if ((s.signal === 'SELL' || s.signal === 'SHORT') && s.confidence * 100 > MANAGED_CONFIDENCE_MIN) {
          if (positions[s.asset] && positions[s.asset] > 0) {
            sellAssets.push(s.asset)
          }
        }
      }

      for (const asset of sellAssets) {
        const pm = priceMaps[asset]
        const price = pm ? getPrice(pm, date) : null
        if (!price || !positions[asset]) continue
        const shares = positions[asset]
        const saleValue = shares * price
        const txCost = saleValue * MANAGED_TX_COST
        cash += saleValue - txCost
        cumulativeTxCosts += txCost

        // Track trade return
        const costBasis = avgCosts[asset] || price
        const tradePnl = ((price - costBasis) / costBasis) * 100
        tradeReturns.push({ asset, pct: tradePnl })

        trades.push({
          date, action: 'SELL', asset, price, shares,
          value: saleValue,
          reason: `${lastSignal[asset]?.signal} signal ${(lastSignal[asset]?.confidence || 0).toFixed(1)}% confidence`,
        })
        delete positions[asset]
        delete avgCosts[asset]
      }

      // Phase 2: BUY — find all BUY signals for assets not currently held
      const buySignals = daySignals
        .filter(s => s.signal === 'BUY' && !positions[s.asset] && s.confidence * 100 > MANAGED_CONFIDENCE_MIN)
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, MANAGED_MAX_BUYS)

      if (buySignals.length > 0 && cash > 0) {
        // Recalculate total value after sells
        let newTotalValue = cash
        for (const [asset, shares] of Object.entries(positions)) {
          const pm = priceMaps[asset]
          const price = pm ? getPrice(pm, date) : null
          if (price) newTotalValue += shares * price
        }

        const maxPerPosition = newTotalValue * MANAGED_MAX_POSITION
        const perBuy = Math.min(cash / buySignals.length, maxPerPosition)

        for (const s of buySignals) {
          if (cash <= 0) break
          const pm = priceMaps[s.asset]
          const price = pm ? getPrice(pm, date) : null
          if (!price || price <= 0) continue

          const investAmount = Math.min(perBuy, cash)
          const txCost = investAmount * MANAGED_TX_COST
          const netAmount = investAmount - txCost
          const shares = netAmount / price
          positions[s.asset] = (positions[s.asset] || 0) + shares
          avgCosts[s.asset] = price
          cash -= investAmount
          cumulativeTxCosts += txCost

          trades.push({
            date, action: 'BUY', asset: s.asset, price, shares,
            value: investAmount,
            reason: `BUY signal ${(s.confidence * 100).toFixed(1)}% confidence`,
          })
        }
      }
    }

    // Compute end-of-day values
    let managedValue = cash
    for (const [asset, shares] of Object.entries(positions)) {
      const pm = priceMaps[asset]
      const price = pm ? getPrice(pm, date) : null
      if (price) managedValue += shares * price
    }

    let bhValue = 0
    for (const [asset, shares] of Object.entries(bhShares)) {
      const pm = priceMaps[asset]
      const price = pm ? getPrice(pm, date) : null
      if (price) bhValue += shares * price
    }

    const spyPrice = spyMap ? getPrice(spyMap, date) : null
    const spyValue = spyStart && spyPrice ? capital * (spyPrice / spyStart) : capital

    chartData.push({
      date,
      managed: Math.round(managedValue),
      buyHold: Math.round(bhValue),
      spy: Math.round(spyValue),
    })
  }

  if (chartData.length < 2) return null

  const last = chartData[chartData.length - 1]
  const managedValues = chartData.map(d => d.managed)
  const bhValues = chartData.map(d => d.buyHold)

  // Build current holdings
  const lastDate = dates[dates.length - 1]
  const holdings: ManagedHolding[] = []
  for (const [asset, shares] of Object.entries(positions)) {
    const pm = priceMaps[asset]
    const price = pm ? getPrice(pm, lastDate) : null
    if (!price || shares <= 0) continue
    const cost = avgCosts[asset] || price
    holdings.push({
      asset,
      shares,
      avgCost: cost,
      currentPrice: price,
      value: shares * price,
      pnlPct: ((price - cost) / cost) * 100,
    })
  }
  holdings.sort((a, b) => b.value - a.value)

  const bestTrade = tradeReturns.length > 0
    ? tradeReturns.reduce((best, t) => t.pct > best.pct ? t : best)
    : null
  const worstTrade = tradeReturns.length > 0
    ? tradeReturns.reduce((worst, t) => t.pct < worst.pct ? t : worst)
    : null

  return {
    chartData,
    startingCapital: capital,
    managedTotal: last.managed,
    managedReturn: ((last.managed - capital) / capital) * 100,
    bhTotal: last.buyHold,
    bhReturn: ((last.buyHold - capital) / capital) * 100,
    spyTotal: last.spy,
    spyReturn: ((last.spy - capital) / capital) * 100,
    totalTrades: trades.length,
    numHoldings: holdings.length,
    cashPosition: cash,
    cashPct: last.managed > 0 ? (cash / last.managed) * 100 : 0,
    bestTrade,
    worstTrade,
    holdings,
    trades: trades.slice().reverse().slice(0, 20), // last 20, newest first
    managedSharpe: sharpeRatio(dailyReturns(managedValues)),
    bhSharpe: sharpeRatio(dailyReturns(bhValues)),
    managedMaxDrawdown: maxDrawdown(managedValues),
    bhMaxDrawdown: maxDrawdown(bhValues),
    cumulativeTxCosts,
  }
}

// ═══════════════════════════════════════
// Main Component
// ═══════════════════════════════════════

type TopTab = 'backtest' | 'live' | 'whatif'

export default function Simulator() {
  const [data, setData] = useState<SimulatorData | null>(null)
  const [wfData, setWfData] = useState<WalkForwardData | null>(null)
  const [managedSimData, setManagedSimData] = useState<ManagedSimData | null>(null)
  const [managedLoading, setManagedLoading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<TopTab>('live')

  // Starting capital (user-configurable)
  const [startingCapital, setStartingCapital] = useState(STARTING_CAPITAL)

  // Portfolio allocation mode
  const [allocMode, setAllocMode] = useState<'default' | 'custom' | 'split' | 'managed'>('default')
  const [splitTotal, setSplitTotal] = useState(100000)
  const [allocations, setAllocations] = useState<Allocation[]>([
    { asset: 'AAPL', pct: 30 },
    { asset: 'MSFT', pct: 30 },
    { asset: 'GLD', pct: 20 },
    { asset: 'bitcoin', pct: 20 },
  ])

  useEffect(() => {
    Promise.all([
      fetch('/api/v1/simulator/data').then(r => {
        const ct = r.headers.get('content-type') || ''
        if (!r.ok || !ct.includes('application/json')) return null
        return r.json()
      }).catch(() => null),
      fetchWalkForwardData(),
    ]).then(([simData, wf]) => {
      setData(simData)
      setWfData(wf)
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])

  // Fetch managed simulation data when mode is selected
  useEffect(() => {
    if (allocMode === 'managed' && !managedSimData && !managedLoading) {
      setManagedLoading(true)
      fetchManagedSimulation()
        .then(d => setManagedSimData(d))
        .finally(() => setManagedLoading(false))
    }
  }, [allocMode, managedSimData, managedLoading])

  // Compute managed portfolio result
  const managedResult = useMemo(() => {
    if (!managedSimData) return null
    return runManagedSimulation(managedSimData, startingCapital)
  }, [managedSimData, startingCapital])

  // Compute BH assets from weights, scaled to startingCapital
  const bhAssets = useMemo(() => {
    if (allocMode === 'custom') {
      return allocations.filter(a => a.asset && a.pct > 0).map(a => ({
        asset: a.asset,
        amount: Math.round((a.pct / 100) * startingCapital),
      }))
    }
    return DEFAULT_BH_WEIGHTS.map(w => ({
      asset: w.asset,
      amount: Math.round(w.weight * startingCapital),
    }))
  }, [allocMode, allocations, startingCapital])

  const backtestResult = useMemo(() => data ? runSimulation(data, bhAssets, startingCapital) : null, [data, bhAssets, startingCapital])
  const liveResult = useMemo(() => data ? runSimulation(data, bhAssets, startingCapital, LIVE_START) : null, [data, bhAssets, startingCapital])

  // When walk-forward data is available, override backtest signals for honest equity curve
  const wfBacktestResult = useMemo(() => {
    if (!wfData || !data) return null
    // Build a modified SimulatorData using walk-forward signals instead of biased ones
    const wfSignalHistory: Record<string, SignalPoint[]> = {}
    for (const sig of wfData.signals) {
      if (!wfSignalHistory[sig.asset]) wfSignalHistory[sig.asset] = []
      wfSignalHistory[sig.asset].push({
        date: sig.date,
        signal: sig.signal,
        price: sig.entry_price,
        was_correct: sig.was_correct,
        outcome_price: sig.exit_price,
      })
    }
    // Merge: use walk-forward signals where available, keep original price history
    const wfSimData: SimulatorData = {
      price_history: data.price_history,
      signal_history: wfSignalHistory,
    }
    return runSimulation(wfSimData, bhAssets, startingCapital)
  }, [wfData, data, bhAssets, startingCapital])

  const result = tab === 'backtest' ? (wfBacktestResult ?? backtestResult) : tab === 'live' ? liveResult : null
  const isWalkForward = tab === 'backtest' && wfBacktestResult !== null
  const daysSinceLive = Math.floor((Date.now() - new Date(LIVE_START).getTime()) / 86400000)

  // 50/50 split: scale existing BH and AS to equal halves of splitTotal
  const splitData = useMemo(() => {
    if (allocMode !== 'split' || !result || result.chartData.length < 2) return null
    const cap = result.startingCapital
    const bhScale = (splitTotal / 2) / cap
    const asScale = (splitTotal / 2) / cap
    const pureBhScale = splitTotal / cap
    const spyScale = splitTotal / cap

    const chartData = result.chartData.map(d => ({
      date: d.date,
      blended: Math.round(d.buyHold * bhScale + d.alphaSignal * asScale),
      pureBuyHold: Math.round(d.buyHold * pureBhScale),
      spy: Math.round(d.spy * spyScale),
    }))

    const last = chartData[chartData.length - 1]
    const blendedValues = chartData.map(d => d.blended)
    const pureBhValues = chartData.map(d => d.pureBuyHold)

    return {
      chartData,
      splitTotal,
      blendedTotal: last.blended,
      blendedReturn: ((last.blended - splitTotal) / splitTotal) * 100,
      pureBhTotal: last.pureBuyHold,
      pureBhReturn: ((last.pureBuyHold - splitTotal) / splitTotal) * 100,
      spyTotal: last.spy,
      spyReturn: ((last.spy - splitTotal) / splitTotal) * 100,
      blendedSharpe: sharpeRatio(dailyReturns(blendedValues)),
      blendedMaxDrawdown: maxDrawdown(blendedValues),
      pureBhSharpe: sharpeRatio(dailyReturns(pureBhValues)),
      pureBhMaxDrawdown: maxDrawdown(pureBhValues),
    }
  }, [allocMode, result, splitTotal])

  const totalPct = allocations.reduce((s, a) => s + a.pct, 0)
  const pctValid = totalPct > 0 && totalPct <= 100
  const usedAssets = new Set(allocations.map(a => a.asset))

  // Allocation handlers
  const addAllocation = () => {
    if (allocations.length >= MAX_CUSTOM) return
    const remaining = AVAILABLE_ASSETS.filter(a => !usedAssets.has(a))
    if (remaining.length === 0) return
    setAllocations([...allocations, { asset: remaining[0], pct: 0 }])
  }

  const removeAllocation = (idx: number) => {
    setAllocations(allocations.filter((_, i) => i !== idx))
  }

  const updateAllocation = (idx: number, field: 'asset' | 'pct', value: string | number) => {
    setAllocations(allocations.map((a, i) => i === idx ? { ...a, [field]: value } : a))
  }

  if (loading) return <div className="text-gray-500 p-8 text-center">Loading simulator...</div>
  if (!data) return <div className="text-gray-500 p-8 text-center">Simulator data unavailable. Ensure the backend has historical price data.</div>

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="bg-gradient-to-r from-[#0f1729] to-[#111827] rounded-2xl border border-[#1f2937] p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-3">Investment Simulator</h2>
            <p className="text-sm text-gray-400 leading-relaxed max-w-3xl">
              Compare buy-and-hold vs Alpha Signal recommendations, or run a what-if simulation
              to see what your capital would be worth if you had followed every signal.
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-xs text-gray-500">Starting capital:</span>
            <div className="flex items-center gap-1">
              <span className="text-gray-500 text-sm">{'\u00A3'}</span>
              <input
                type="number"
                min={1000}
                step={10000}
                value={startingCapital}
                onChange={e => setStartingCapital(Math.max(1000, parseInt(e.target.value) || STARTING_CAPITAL))}
                className="w-28 bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-500/30"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Tab selector */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setTab('backtest')}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer flex items-center gap-2 ${
            tab === 'backtest'
              ? wfData ? 'bg-green-500/15 text-green-400 border border-green-500/30' : 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
              : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
          }`}
        >
          {wfData ? 'Walk-Forward Backtest' : '5-Year Backtest'}
          {wfData && <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">no lookahead</span>}
        </button>
        <button
          onClick={() => setTab('live')}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer flex items-center gap-2 ${
            tab === 'live'
              ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
              : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
          }`}
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          Live Tracking
        </button>
        <button
          onClick={() => setTab('whatif')}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
            tab === 'whatif'
              ? 'bg-purple-500/15 text-purple-400 border border-purple-500/30'
              : 'text-gray-400 hover:text-gray-200 bg-[#111827] border border-[#1f2937]'
          }`}
        >
          What-If Simulator
        </button>
      </div>

      {/* What-If tab */}
      {tab === 'whatif' && <WhatIfSimulator />}

      {/* Backtest / Live tabs */}
      {(tab === 'backtest' || tab === 'live') && (
        <>
          {/* Warning banners */}
          {tab === 'backtest' && (
            isWalkForward ? (
              <div className="bg-green-900/20 border border-green-500/30 rounded-lg px-5 py-4 text-sm">
                <div className="flex items-start gap-3">
                  <span className="text-xl leading-none flex-shrink-0 inline-flex items-center gap-1.5">
                    <span className="inline-block w-2.5 h-2.5 rounded-full bg-green-500" />
                  </span>
                  <div>
                    <div className="font-semibold text-green-300 mb-1">Walk-Forward Backtest (no lookahead)</div>
                    <p className="text-green-200/70">These signals were generated using models that had never seen the test period data. Each quarterly window was tested with models trained only on prior data.</p>
                    {wfData && <p className="text-green-300/50 text-xs mt-1">Generated: {new Date(wfData.generated_at).toLocaleDateString()} &middot; {wfData.summary.total_signals.toLocaleString()} signals across {wfData.windows.length} windows</p>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-yellow-900/30 border-2 border-yellow-500/50 rounded-lg px-5 py-4 text-sm">
                <div className="flex items-start gap-3">
                  <span className="text-xl leading-none flex-shrink-0">{'\u26A0\uFE0F'}</span>
                  <div>
                    <div className="font-semibold text-yellow-300 mb-1">Lookahead bias warning</div>
                    <p className="text-yellow-200/80">These models were trained on this historical data. The backtest is illustrative only and significantly overstates real-world performance. Run the walk-forward backtester to see unbiased results.</p>
                  </div>
                </div>
              </div>
            )
          )}
          {tab === 'live' && (
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg px-4 py-3 text-sm text-green-300 flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              Live &mdash; Day {daysSinceLive} of live tracking (since 15 March 2026)
            </div>
          )}

          {/* Custom Portfolio Builder */}
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-400">Portfolio Allocation</h3>
              <div className="flex gap-2">
                {(['default', 'managed', 'split', 'custom'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setAllocMode(mode)}
                    className={`px-3 py-1.5 rounded text-xs font-medium cursor-pointer transition-colors ${
                      allocMode === mode
                        ? mode === 'split'
                          ? 'bg-violet-500/15 text-violet-400 border border-violet-500/30'
                          : mode === 'managed'
                            ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                            : 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                        : 'text-gray-400 bg-[#0a0e17] border border-[#1f2937] hover:border-[#374151]'
                    }`}
                  >
                    {mode === 'default' ? 'Default (10 assets)' : mode === 'managed' ? 'Managed Portfolio' : mode === 'split' ? '50/50 Split' : 'Custom'}
                  </button>
                ))}
              </div>
            </div>

            {allocMode === 'custom' && (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  Pick up to {MAX_CUSTOM} assets and set the % of {fmtGBP(startingCapital)} to allocate. Remaining goes to cash.
                </p>

                {allocations.map((alloc, idx) => {
                  return (
                    <div key={idx} className="flex items-center gap-2">
                      <select
                        value={alloc.asset}
                        onChange={e => updateAllocation(idx, 'asset', e.target.value)}
                        className="bg-[#0a0e17] border border-[#1f2937] rounded px-3 py-2 text-sm text-gray-300 flex-1 min-w-0"
                      >
                        {AVAILABLE_ASSETS.map(a => {
                          const taken = allocations.some((al, i) => i !== idx && al.asset === a)
                          return (
                            <option key={a} value={a} disabled={taken}>
                              {a}{taken ? ' (used)' : ''}
                            </option>
                          )
                        })}
                      </select>
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          min={0}
                          max={100}
                          value={alloc.pct}
                          onChange={e => updateAllocation(idx, 'pct', Math.max(0, Math.min(100, parseInt(e.target.value) || 0)))}
                          className="w-16 bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-2 text-sm text-white text-center outline-none focus:border-cyan-500/30"
                        />
                        <span className="text-gray-500 text-sm">%</span>
                      </div>
                      <span className="text-xs text-gray-600 w-16 text-right">
                        {fmtGBP(Math.round((alloc.pct / 100) * startingCapital))}
                      </span>
                      <button
                        onClick={() => removeAllocation(idx)}
                        className="text-gray-600 hover:text-red-400 p-1 cursor-pointer transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )
                })}

                {allocations.length < MAX_CUSTOM && AVAILABLE_ASSETS.some(a => !usedAssets.has(a)) && (
                  <button
                    onClick={addAllocation}
                    className="flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    Add asset
                  </button>
                )}

                <div className="flex items-center justify-between pt-2 border-t border-[#1f2937]">
                  <div className="text-xs text-gray-500">
                    Allocated: <span className={totalPct <= 100 ? 'text-cyan-400' : 'text-red-400'}>{totalPct}%</span>
                    {totalPct < 100 && <> &middot; Cash: <span className="text-gray-400">{100 - totalPct}% ({fmtGBP(Math.round(((100 - totalPct) / 100) * startingCapital))})</span></>}
                  </div>
                  {!pctValid && (
                    <span className="text-xs text-red-400">
                      {totalPct > 100 ? 'Total exceeds 100%' : 'Add at least one allocation'}
                    </span>
                  )}
                </div>
              </div>
            )}

            {allocMode === 'default' && (
              <div className="text-xs text-gray-500">
                Diversified allocation across US stocks, UK stocks, bonds, gold, commodities, FX and crypto &mdash; 10% cash buffer. Alpha Signal rotates across all ~165 assets.
              </div>
            )}

            {allocMode === 'managed' && (
              <div className="text-xs text-gray-500 space-y-1">
                <p>Alpha Signal manages your entire portfolio across ~165 assets &mdash; stocks (US, UK, European), bonds, FX, gold, commodities and crypto. Capital rotates daily into the strongest signals across all asset classes. In a falling equity market, capital can rotate into bonds, defensive sectors, or FX positions.</p>
                <p className="text-gray-600">Rules: 15% max per position &middot; Top 8 BUY signals by confidence &middot; 0.2% tx cost per trade &middot; SELL/SHORT signals with &gt;10% confidence trigger exits</p>
              </div>
            )}

            {allocMode === 'split' && (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  Split your capital 50/50: half follows buy &amp; hold across default assets, half follows Alpha Signal recommendations.
                </p>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-gray-500">Total capital:</span>
                  <div className="flex items-center gap-1">
                    <span className="text-gray-500 text-sm">{'\u00A3'}</span>
                    <input
                      type="number"
                      min={1000}
                      step={1000}
                      value={splitTotal}
                      onChange={e => setSplitTotal(Math.max(1000, parseInt(e.target.value) || 100000))}
                      className="w-28 bg-[#0a0e17] border border-[#1f2937] rounded px-2 py-1.5 text-sm text-white outline-none focus:border-violet-500/30"
                    />
                  </div>
                  <span className="text-xs text-gray-600">
                    {fmtGBP(splitTotal / 2)} buy &amp; hold + {fmtGBP(splitTotal / 2)} Alpha Signal
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Simulation results */}
          {allocMode === 'managed' ? (
            managedLoading ? (
              <div className="bg-[#111827] rounded-xl border border-emerald-500/20 p-8 text-center">
                <Loader2 className="w-6 h-6 animate-spin text-emerald-400 mx-auto mb-3" />
                <div className="text-emerald-400 text-sm font-medium">Alpha Signal is managing your portfolio...</div>
              </div>
            ) : managedResult ? (
              <ManagedPortfolioResults result={managedResult} />
            ) : (
              <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-8 text-center text-gray-500">
                Managed portfolio data unavailable. Ensure the backend has signal history from 15 March 2026.
              </div>
            )
          ) : !result || result.chartData.length < 2 ? (
            <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-8 text-center text-gray-500">
              {!pctValid && allocMode === 'custom'
                ? 'Fix allocation percentages to run simulation.'
                : tab === 'live'
                  ? 'Not enough data yet. Live tracking data accumulates daily from 15 March 2026.'
                  : 'Not enough historical price data available.'}
            </div>
          ) : allocMode === 'split' && splitData ? (
            <SplitResults data={splitData} />
          ) : (
            <InvestmentResults result={result} />
          )}
        </>
      )}
    </div>
  )
}

// ═══════════════════════════════════════
// Investment Sim Results
// ═══════════════════════════════════════

interface SplitData {
  chartData: { date: string; blended: number; pureBuyHold: number; spy: number }[]
  splitTotal: number
  blendedTotal: number
  blendedReturn: number
  pureBhTotal: number
  pureBhReturn: number
  spyTotal: number
  spyReturn: number
  blendedSharpe: number
  blendedMaxDrawdown: number
  pureBhSharpe: number
  pureBhMaxDrawdown: number
}

function SplitResults({ data }: { data: SplitData }) {
  return (
    <>
      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard
          label={`50/50 Blended ${fmtGBP(data.splitTotal)}`}
          value={fmtGBP(data.blendedTotal)}
          returnPct={data.blendedReturn}
          borderColor="border-violet-500/30"
          valueColor="text-violet-400"
        />
        <SummaryCard
          label={`Pure Buy & Hold ${fmtGBP(data.splitTotal)}`}
          value={fmtGBP(data.pureBhTotal)}
          returnPct={data.pureBhReturn}
          borderColor="border-[#1f2937]"
          valueColor="text-white"
        />
        <SummaryCard
          label={`S&P 500 Benchmark ${fmtGBP(data.splitTotal)}`}
          value={fmtGBP(data.spyTotal)}
          returnPct={data.spyReturn}
          borderColor="border-gray-700"
          valueColor="text-gray-300"
        />
      </div>

      {/* Risk metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCard label="Sharpe (50/50)" value={data.blendedSharpe.toFixed(2)} color={data.blendedSharpe >= 1 ? 'text-green-400' : data.blendedSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Sharpe (Pure B&H)" value={data.pureBhSharpe.toFixed(2)} color={data.pureBhSharpe >= 1 ? 'text-green-400' : data.pureBhSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max DD (50/50)" value={`-${data.blendedMaxDrawdown.toFixed(1)}%`} color={data.blendedMaxDrawdown < 10 ? 'text-green-400' : data.blendedMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max DD (Pure B&H)" value={`-${data.pureBhMaxDrawdown.toFixed(1)}%`} color={data.pureBhMaxDrawdown < 10 ? 'text-green-400' : data.pureBhMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
      </div>

      {/* Chart */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Portfolio Value Over Time</h3>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={data.chartData} margin={{ left: 10, right: 10, top: 4, bottom: 0 }}>
            <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
            <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `\u00a3${(v / 1000).toFixed(0)}k`} width={55} />
            <Tooltip
              contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '']}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <ReferenceLine y={data.splitTotal} stroke="#374151" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="blended" name="50/50 Blended" stroke="#a78bfa" strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="pureBuyHold" name="Pure Buy & Hold" stroke="#e5e7eb" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="spy" name="S&P 500" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Explanation */}
      <div className="bg-violet-500/5 border border-violet-500/20 rounded-lg p-4 text-sm text-gray-400 leading-relaxed">
        <span className="text-violet-400 font-medium">How it works:</span> {fmtGBP(data.splitTotal / 2)} is invested
        buy-and-hold across the default 10 assets, and the other {fmtGBP(data.splitTotal / 2)} follows
        Alpha Signal&apos;s BUY/SELL/HOLD recommendations. The blended line shows the combined portfolio value.
      </div>
    </>
  )
}

function InvestmentResults({ result }: { result: SimResult }) {
  const cap = result.startingCapital

  return (
    <>
      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard
          label={`Buy & Hold ${fmtGBP(cap)}`}
          value={fmtGBP(result.bhTotal)}
          returnPct={result.bhReturn}
          borderColor="border-[#1f2937]"
          valueColor="text-white"
        />
        <SummaryCard
          label={`Alpha Signal ${fmtGBP(cap)}`}
          value={fmtGBP(result.asTotal)}
          returnPct={result.asReturn}
          borderColor="border-cyan-500/30"
          valueColor="text-cyan-400"
        />
        <SummaryCard
          label={`S&P 500 Benchmark ${fmtGBP(cap)}`}
          value={fmtGBP(result.spyTotal)}
          returnPct={result.spyReturn}
          borderColor="border-gray-700"
          valueColor="text-gray-300"
        />
      </div>

      {/* Risk metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <MetricCard label="Sharpe Ratio (B&H)" value={result.bhSharpe.toFixed(2)} color={result.bhSharpe >= 1 ? 'text-green-400' : result.bhSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Sharpe Ratio (Alpha)" value={result.asSharpe.toFixed(2)} color={result.asSharpe >= 1 ? 'text-green-400' : result.asSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max Drawdown (B&H)" value={`-${result.bhMaxDrawdown.toFixed(1)}%`} color={result.bhMaxDrawdown < 10 ? 'text-green-400' : result.bhMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max Drawdown (Alpha)" value={`-${result.asMaxDrawdown.toFixed(1)}%`} color={result.asMaxDrawdown < 10 ? 'text-green-400' : result.asMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Tx Costs (Alpha)" value={fmtGBP(result.cumulativeTxCosts)} color="text-orange-400" />
      </div>

      {/* Main chart */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Portfolio Value Over Time</h3>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={result.chartData} margin={{ left: 10, right: 10, top: 4, bottom: 0 }}>
            <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
            <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `\u00a3${(v / 1000).toFixed(0)}k`} width={55} />
            <Tooltip
              contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '']}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <Line type="monotone" dataKey="alphaSignal" name="Alpha Signal" stroke="#06b6d4" strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="buyHold" name="Buy & Hold" stroke="#e5e7eb" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="spy" name="S&P 500" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Stacked area allocation chart */}
      {result.allocationChartData.length > 1 && (() => {
        const ALLOC_COLORS = ['#06b6d4', '#a78bfa', '#f59e0b', '#10b981', '#ef4444', '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#6366f1']
        const assetKeys = result.allocationHistory[0]?.positions.map(p => p.asset) || []
        return (
          <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Alpha Signal &mdash; Daily Allocation</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={result.allocationChartData} margin={{ left: 10, right: 10, top: 4, bottom: 0 }}>
                <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `${Math.round(v)}%`} width={45} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                  labelStyle={{ color: '#9ca3af' }}
                  formatter={(v: number | undefined) => [v != null ? `${v.toFixed(1)}%` : '']}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
                {assetKeys.map((asset, i) => (
                  <Area key={asset} type="monotone" dataKey={asset} stackId="1" fill={ALLOC_COLORS[i % ALLOC_COLORS.length]} stroke={ALLOC_COLORS[i % ALLOC_COLORS.length]} fillOpacity={0.8} />
                ))}
                <Area type="monotone" dataKey="cash" stackId="1" fill="#374151" stroke="#374151" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )
      })()}

      {/* Buy & Hold breakdown */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Buy & Hold Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 border-b border-[#1f2937]">
                <th className="text-left py-2 px-2">Asset</th>
                <th className="text-right py-2 px-2">Invested</th>
                <th className="text-right py-2 px-2">Current Value</th>
                <th className="text-right py-2 px-2">Return %</th>
              </tr>
            </thead>
            <tbody>
              {result.bhBreakdown.map(a => (
                <tr key={a.asset} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                  <td className="py-1.5 px-2 text-gray-300 font-medium">{a.asset}</td>
                  <td className="py-1.5 px-2 text-right text-gray-400">{fmtGBP(a.invested)}</td>
                  <td className="py-1.5 px-2 text-right text-gray-300">{fmtGBP(a.currentValue)}</td>
                  <td className={`py-1.5 px-2 text-right font-medium ${a.returnPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {fmtPct(a.returnPct)}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-[#374151] font-semibold">
                <td className="py-2 px-2 text-white">Total</td>
                <td className="py-2 px-2 text-right text-gray-300">{fmtGBP(cap)}</td>
                <td className="py-2 px-2 text-right text-white">{fmtGBP(result.bhTotal)}</td>
                <td className={`py-2 px-2 text-right ${result.bhReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {fmtPct(result.bhReturn)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Alpha Signal breakdown */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
        <h3 className="text-sm font-medium text-cyan-400 mb-3">Alpha Signal Allocation</h3>
        {/* Signal accuracy stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500 mb-1">Total Signals</div>
            <div className="text-xl font-bold text-white">{result.asTotalSignals}</div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500 mb-1">Correct</div>
            <div className="text-xl font-bold text-green-400">{result.asCorrect}</div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500 mb-1">Incorrect</div>
            <div className="text-xl font-bold text-red-400">{result.asIncorrect}</div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3 text-center">
            <div className="text-xs text-gray-500 mb-1">Portfolio Value</div>
            <div className="text-xl font-bold text-cyan-400">{fmtGBP(result.asTotal)}</div>
          </div>
        </div>
        {/* Final day allocation table */}
        {result.allocationHistory.length > 0 && (() => {
          const finalSnap = result.allocationHistory[result.allocationHistory.length - 1]
          return (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-[#1f2937]">
                    <th className="text-left py-2 px-2">Asset</th>
                    <th className="text-center py-2 px-2">Signal</th>
                    <th className="text-right py-2 px-2">Target %</th>
                    <th className="text-right py-2 px-2">Actual %</th>
                    <th className="text-right py-2 px-2">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {finalSnap.positions.map(pos => (
                    <tr key={pos.asset} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                      <td className="py-1.5 px-2 text-gray-300 font-medium">{pos.asset}</td>
                      <td className="py-1.5 px-2 text-center">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          pos.signal === 'BUY' ? 'bg-green-500/15 text-green-400' :
                          pos.signal === 'SELL' || pos.signal === 'SHORT' ? 'bg-red-500/15 text-red-400' :
                          'bg-gray-500/15 text-gray-400'
                        }`}>{pos.signal}</span>
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{(pos.targetWeight * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right text-gray-300">{(pos.weight * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right text-gray-300">{fmtGBP(pos.value)}</td>
                    </tr>
                  ))}
                  <tr className="border-b border-[#1f2937]/50">
                    <td className="py-1.5 px-2 text-gray-500">Cash</td>
                    <td className="py-1.5 px-2 text-center text-gray-600">&mdash;</td>
                    <td className="py-1.5 px-2 text-right text-gray-500">&mdash;</td>
                    <td className="py-1.5 px-2 text-right text-gray-500">{(finalSnap.cashPct * 100).toFixed(1)}%</td>
                    <td className="py-1.5 px-2 text-right text-gray-500">{fmtGBP(finalSnap.cashValue)}</td>
                  </tr>
                </tbody>
                <tfoot>
                  <tr className="border-t border-[#374151] font-semibold">
                    <td className="py-2 px-2 text-white">Total</td>
                    <td className="py-2 px-2"></td>
                    <td className="py-2 px-2"></td>
                    <td className="py-2 px-2 text-right text-gray-300">100%</td>
                    <td className="py-2 px-2 text-right text-white">{fmtGBP(finalSnap.totalValue)}</td>
                  </tr>
                </tfoot>
              </table>
            </div>
          )
        })()}
      </div>

      {/* Methodology section */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <h3 className="text-lg font-semibold text-white mb-4">For the technically minded</h3>
        <div className="space-y-3 text-sm text-gray-400 leading-relaxed">
          <div className="flex gap-3">
            <span className="text-cyan-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Walk-forward validation:</span> Models are tested on data they haven&apos;t seen during training.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-amber-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Backtest caveat:</span> The 5-year backtest uses data the models were trained on &mdash; treat results as optimistic upper bounds, not predictions.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-500 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Transaction costs:</span> 10bps per trade for stocks/ETFs, 25bps for crypto, applied on every rebalance.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-violet-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Allocation:</span> Inverse-volatility weighting (60-day trailing returns). Correlated pairs (r&gt;0.7) penalized. 30% single-asset cap. 2% rebalance threshold to avoid churn.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-green-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Live data:</span> Only 12 days of real signals &mdash; statistically early, check back in 90 days.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-500 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Overfitting check:</span> Some FX pairs show very high accuracy &mdash; this reflects low volatility in those pairs during the tracked period, not model overfitting.</p>
          </div>
          <div className="flex gap-3">
            <span className="text-orange-400 font-bold mt-0.5 flex-shrink-0">&bull;</span>
            <p><span className="text-gray-300 font-medium">Built in Rust, not Python:</span> This entire ML pipeline &mdash; 6-model ensemble (LinReg, LogReg, GBT, LSTM, GRU, TFT), feature engineering, walk-forward validation &mdash; is written in pure Rust using the <a href="https://github.com/huggingface/candle" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2">candle</a> framework. One goal of this project is to demonstrate that Python is not the only viable language for machine learning, despite the conventional wisdom.</p>
          </div>
          <div className="mt-4 pt-4 border-t border-[#1f2937] text-gray-500 text-xs">
            This is a learning and research project &mdash; not a production trading system or financial advice.
          </div>
        </div>
      </div>
    </>
  )
}

// ═══════════════════════════════════════
// Managed Portfolio Results
// ═══════════════════════════════════════

function ManagedPortfolioResults({ result }: { result: ManagedResult }) {
  const [showTradeLog, setShowTradeLog] = useState(false)
  const cap = result.startingCapital

  return (
    <>
      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard
          label={`Managed Portfolio ${fmtGBP(cap)}`}
          value={fmtGBP(result.managedTotal)}
          returnPct={result.managedReturn}
          borderColor="border-emerald-500/30"
          valueColor="text-emerald-400"
        />
        <SummaryCard
          label={`Buy & Hold ${fmtGBP(cap)}`}
          value={fmtGBP(result.bhTotal)}
          returnPct={result.bhReturn}
          borderColor="border-[#1f2937]"
          valueColor="text-white"
        />
        <SummaryCard
          label={`S&P 500 Benchmark ${fmtGBP(cap)}`}
          value={fmtGBP(result.spyTotal)}
          returnPct={result.spyReturn}
          borderColor="border-gray-700"
          valueColor="text-gray-300"
        />
      </div>

      {/* Risk metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <MetricCard label="Sharpe (Managed)" value={result.managedSharpe.toFixed(2)} color={result.managedSharpe >= 1 ? 'text-green-400' : result.managedSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Sharpe (B&H)" value={result.bhSharpe.toFixed(2)} color={result.bhSharpe >= 1 ? 'text-green-400' : result.bhSharpe >= 0 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max DD (Managed)" value={`-${result.managedMaxDrawdown.toFixed(1)}%`} color={result.managedMaxDrawdown < 10 ? 'text-green-400' : result.managedMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Max DD (B&H)" value={`-${result.bhMaxDrawdown.toFixed(1)}%`} color={result.bhMaxDrawdown < 10 ? 'text-green-400' : result.bhMaxDrawdown < 25 ? 'text-amber-400' : 'text-red-400'} />
        <MetricCard label="Tx Costs" value={fmtGBP(result.cumulativeTxCosts)} color="text-orange-400" />
      </div>

      {/* Three-line chart */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Portfolio Value Over Time</h3>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={result.chartData} margin={{ left: 10, right: 10, top: 4, bottom: 0 }}>
            <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
            <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `\u00a3${(v / 1000).toFixed(0)}k`} width={55} />
            <Tooltip
              contentStyle={{ background: '#0a0e17', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '']}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            <ReferenceLine y={cap} stroke="#374151" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="managed" name="Managed Portfolio" stroke="#10b981" strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="buyHold" name="Buy & Hold" stroke="#e5e7eb" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="spy" name="S&P 500" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Total Trades</div>
          <div className="text-xl font-bold text-white">{result.totalTrades}</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Current Holdings</div>
          <div className="text-xl font-bold text-white">{result.numHoldings}</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Cash Position</div>
          <div className="text-xl font-bold text-white">{fmtGBP(result.cashPosition)}</div>
          <div className="text-xs text-gray-500 mt-0.5">{result.cashPct.toFixed(1)}% of portfolio</div>
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Best Trade</div>
          {result.bestTrade ? (
            <>
              <div className="text-xl font-bold text-green-400">{fmtPct(result.bestTrade.pct)}</div>
              <div className="text-xs text-gray-500 mt-0.5">{result.bestTrade.asset}</div>
            </>
          ) : <div className="text-gray-600">N/A</div>}
        </div>
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <div className="text-xs text-gray-500 mb-1">Worst Trade</div>
          {result.worstTrade ? (
            <>
              <div className="text-xl font-bold text-red-400">{fmtPct(result.worstTrade.pct)}</div>
              <div className="text-xs text-gray-500 mt-0.5">{result.worstTrade.asset}</div>
            </>
          ) : <div className="text-gray-600">N/A</div>}
        </div>
      </div>

      {/* Current holdings table */}
      {result.holdings.length > 0 && (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <h3 className="text-sm font-medium text-emerald-400 mb-3">Current Holdings</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-[#1f2937]">
                  <th className="text-left py-2 px-2">Asset</th>
                  <th className="text-right py-2 px-2">Shares</th>
                  <th className="text-right py-2 px-2">Avg Cost</th>
                  <th className="text-right py-2 px-2">Current Price</th>
                  <th className="text-right py-2 px-2">Value</th>
                  <th className="text-right py-2 px-2">P&amp;L %</th>
                </tr>
              </thead>
              <tbody>
                {result.holdings.map(h => (
                  <tr key={h.asset} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                    <td className="py-1.5 px-2 text-gray-300 font-medium">{h.asset}</td>
                    <td className="py-1.5 px-2 text-right text-gray-400">{h.shares.toFixed(4)}</td>
                    <td className="py-1.5 px-2 text-right text-gray-400">{'\u00A3'}{h.avgCost.toFixed(2)}</td>
                    <td className="py-1.5 px-2 text-right text-gray-300">{'\u00A3'}{h.currentPrice.toFixed(2)}</td>
                    <td className="py-1.5 px-2 text-right text-gray-300">{fmtGBP(h.value)}</td>
                    <td className={`py-1.5 px-2 text-right font-medium ${h.pnlPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {fmtPct(h.pnlPct)}
                    </td>
                  </tr>
                ))}
                <tr className="border-b border-[#1f2937]/50">
                  <td className="py-1.5 px-2 text-gray-500">Cash</td>
                  <td className="py-1.5 px-2 text-right text-gray-600">&mdash;</td>
                  <td className="py-1.5 px-2 text-right text-gray-600">&mdash;</td>
                  <td className="py-1.5 px-2 text-right text-gray-600">&mdash;</td>
                  <td className="py-1.5 px-2 text-right text-gray-500">{fmtGBP(result.cashPosition)}</td>
                  <td className="py-1.5 px-2 text-right text-gray-600">&mdash;</td>
                </tr>
              </tbody>
              <tfoot>
                <tr className="border-t border-[#374151] font-semibold">
                  <td className="py-2 px-2 text-white">Total</td>
                  <td className="py-2 px-2"></td>
                  <td className="py-2 px-2"></td>
                  <td className="py-2 px-2"></td>
                  <td className="py-2 px-2 text-right text-white">{fmtGBP(result.managedTotal)}</td>
                  <td className={`py-2 px-2 text-right ${result.managedReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {fmtPct(result.managedReturn)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      )}

      {/* Trade log (collapsible) */}
      {result.trades.length > 0 && (
        <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
          <button
            onClick={() => setShowTradeLog(!showTradeLog)}
            className="flex items-center gap-2 text-sm font-medium text-gray-400 hover:text-gray-200 cursor-pointer transition-colors w-full"
          >
            <span className={`transition-transform ${showTradeLog ? 'rotate-90' : ''}`}>{'\u25B6'}</span>
            Trade Log (last 20 trades)
          </button>
          {showTradeLog && (
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-[#1f2937]">
                    <th className="text-left py-2 px-2">Date</th>
                    <th className="text-center py-2 px-2">Action</th>
                    <th className="text-left py-2 px-2">Asset</th>
                    <th className="text-right py-2 px-2">Price</th>
                    <th className="text-right py-2 px-2">Shares</th>
                    <th className="text-right py-2 px-2">Value</th>
                    <th className="text-left py-2 px-2">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.map((t, i) => (
                    <tr key={i} className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]">
                      <td className="py-1.5 px-2 text-gray-400 text-xs">{t.date}</td>
                      <td className="py-1.5 px-2 text-center">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          t.action === 'BUY' ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'
                        }`}>{t.action}</span>
                      </td>
                      <td className="py-1.5 px-2 text-gray-300 font-medium">{t.asset}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{'\u00A3'}{t.price.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{t.shares.toFixed(4)}</td>
                      <td className="py-1.5 px-2 text-right text-gray-300">{fmtGBP(t.value)}</td>
                      <td className="py-1.5 px-2 text-gray-500 text-xs">{t.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Methodology */}
      <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-4 text-sm text-gray-400 leading-relaxed">
        <span className="text-emerald-400 font-medium">How the Managed Portfolio works:</span> Alpha Signal starts with {fmtGBP(cap)} diversified
        across US stocks, UK stocks, bonds, gold, commodities, FX and crypto, then actively manages the portfolio using live signals across all ~165 tracked assets.
        In a falling equity market, capital can rotate into bonds, defensive sectors, or FX positions.
        SELL/SHORT signals with &gt;10% confidence trigger exits; freed capital is redeployed to the top 8 BUY signals
        ranked by confidence, with a 15% single-position cap and 0.2% transaction cost per trade.
      </div>
    </>
  )
}

// ═══════════════════════════════════════
// What-If Simulator (moved from Explore)
// ═══════════════════════════════════════

const CAPITAL_OPTIONS = [1_000, 5_000, 10_000, 50_000]
const PERIOD_OPTIONS = [7, 14, 30, 60, 90]

function WhatIfSimulator() {
  const [capital, setCapital] = useState(10_000)
  const [customCapital, setCustomCapital] = useState('')
  const [useCustomCap, setUseCustomCap] = useState(false)
  const [period, setPeriod] = useState(30)
  const [wLoading, setWLoading] = useState(false)
  const [wResult, setWResult] = useState<WhatIfResult | null>(null)
  const [wError, setWError] = useState<string | null>(null)

  const runSim = useCallback(async () => {
    const cap = useCustomCap ? parseFloat(customCapital) || 10_000 : capital
    setWLoading(true)
    setWError(null)
    try {
      const res = await fetchSimulation(period, cap)
      setWResult(res)
    } catch {
      setWError('Simulation failed. Make sure models are trained and the server is running.')
    } finally {
      setWLoading(false)
    }
  }, [useCustomCap, customCapital, capital, period])

  const returnPct = wResult ? wResult.total_return_pct : 0
  const bhPct = wResult ? wResult.vs_buy_and_hold_pct : 0
  const beatsBH = returnPct > bhPct

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-white text-lg font-semibold">What if you had followed our signals?</h3>
        <p className="text-gray-500 text-sm mt-1">
          See exactly what would have happened if you bought when we said buy, sold when we said sell, and held when we said hold.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="text-gray-500 text-xs mb-2">Starting capital</div>
            <div className="flex gap-2 flex-wrap">
              {CAPITAL_OPTIONS.map(c => (
                <button
                  key={c}
                  onClick={() => { setCapital(c); setUseCustomCap(false) }}
                  className={`px-3 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                    !useCustomCap && capital === c
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  {fmtGBP(c)}
                </button>
              ))}
              <div className="flex items-center gap-1">
                <span className="text-gray-500 text-sm">{'\u00A3'}</span>
                <input
                  type="number"
                  value={customCapital}
                  onChange={e => { setCustomCapital(e.target.value); setUseCustomCap(true) }}
                  onFocus={() => setUseCustomCap(true)}
                  placeholder="Custom"
                  className={`w-24 bg-[#0a0e17] border rounded px-2 py-1.5 text-sm text-white outline-none ${
                    useCustomCap ? 'border-cyan-500/30' : 'border-[#1f2937]'
                  }`}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="text-gray-500 text-xs mb-2">Period</div>
            <div className="flex gap-2">
              {PERIOD_OPTIONS.map(p => (
                <button
                  key={p}
                  onClick={() => setPeriod(p)}
                  className={`px-3 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                    period === p
                      ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                      : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-[#374151]'
                  }`}
                >
                  {p}d
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-end">
            <button
              onClick={runSim}
              disabled={wLoading}
              className="px-6 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg font-medium text-sm hover:bg-cyan-500/30 transition-colors disabled:opacity-50 cursor-pointer flex items-center gap-2"
            >
              {wLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {wLoading ? 'Running...' : 'Run Simulation'}
            </button>
          </div>
        </div>
      </div>

      {wError && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-red-400 text-sm">{wError}</div>
      )}

      {wResult && (
        <>
          <div className="bg-[#111827] border border-cyan-500/20 rounded-lg p-8 text-center shadow-[0_0_30px_rgba(6,182,212,0.05)]">
            <p className="text-gray-400 text-sm mb-2">{fmtGBP(wResult.starting_capital)} would be worth</p>
            <div className={`text-4xl font-bold mb-2 ${returnPct >= 0 ? 'text-cyan-400' : 'text-red-400'}`}>
              {fmtGBP(wResult.final_value)}
            </div>
            <p className="text-gray-500 text-sm">
              {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}% return &middot; {wResult.days} days &middot; {wResult.signal_accuracy_pct.toFixed(1)}% of signals were correct
            </p>
          </div>

          <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
            <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Performance comparison</div>
            <div className="grid grid-cols-3 gap-4">
              <CompareBar label="Our signals" value={returnPct} highlight={beatsBH} />
              <CompareBar label="Buy & Hold" value={bhPct} highlight={!beatsBH} />
              <CompareBar label="Cash" value={0} highlight={false} />
            </div>
          </div>

          {wResult.daily.length > 1 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Portfolio value over time</div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={wResult.daily} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="date" tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: '#4b5563', fontSize: 11 }} tickFormatter={v => `\u00A3${(v / 1000).toFixed(1)}k`} width={55} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: '8px', fontSize: 12 }}
                    labelStyle={{ color: '#e5e7eb' }}
                    formatter={(v: number | undefined) => [v != null ? fmtGBP(v) : '', 'Value']}
                  />
                  <ReferenceLine y={wResult.starting_capital} stroke="#374151" strokeDasharray="4 4" label={{ value: 'Start', fill: '#6b7280', fontSize: 10 }} />
                  <Line type="monotone" dataKey="value" stroke="#06b6d4" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#06b6d4' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {wResult.per_asset.length > 0 && (
            <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-6">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-3">Per-asset breakdown</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
                    <th className="text-left py-2">Asset</th>
                    <th className="text-right py-2">Signal Accuracy</th>
                    <th className="text-right py-2">Contribution to Return</th>
                  </tr>
                </thead>
                <tbody>
                  {wResult.per_asset.map(a => (
                    <tr key={a.asset} className="border-b border-[#1f2937]/50">
                      <td className="py-2 text-white font-medium">{a.asset}</td>
                      <td className={`py-2 text-right ${a.signal_accuracy_pct >= 60 ? 'text-emerald-400' : a.signal_accuracy_pct >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
                        {a.signal_accuracy_pct.toFixed(1)}%
                      </td>
                      <td className={`py-2 text-right ${a.contribution_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {a.contribution_pct >= 0 ? '+' : ''}{a.contribution_pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="bg-[#0a0e17] border border-[#1f2937] rounded-lg p-4 text-gray-500 text-xs leading-relaxed">
            This simulation uses the same models that generated live signals. It is not a true backtest — the models were trained on data that includes this period, so results may be optimistic. Live performance since {wResult.inception_date} is the more reliable measure.
          </div>
        </>
      )}
    </div>
  )
}

// ═══════════════════════════════════════
// Shared Sub-components
// ═══════════════════════════════════════

function SummaryCard({ label, value, returnPct, borderColor, valueColor }: {
  label: string; value: string; returnPct: number; borderColor: string; valueColor: string
}) {
  return (
    <div className={`bg-[#111827] rounded-xl border ${borderColor} p-5`}>
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-3xl font-bold ${valueColor}`}>{value}</div>
      <div className={`text-sm mt-1 ${returnPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {fmtPct(returnPct)}
      </div>
    </div>
  )
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
    </div>
  )
}

function CompareBar({ label, value, highlight }: { label: string; value: number; highlight: boolean }) {
  const isPositive = value >= 0
  const barColor = highlight ? (isPositive ? '#10b981' : '#ef4444') : '#374151'

  return (
    <div className="text-center">
      <div className={`text-lg font-bold ${highlight ? (isPositive ? 'text-emerald-400' : 'text-red-400') : 'text-gray-500'}`}>
        {isPositive ? '+' : ''}{value.toFixed(2)}%
      </div>
      <div className="text-gray-500 text-xs mt-1">{label}</div>
      <div className="h-1.5 mt-2 bg-[#0a0e17] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(Math.abs(value) * 2, 100)}%`, background: barColor }}
        />
      </div>
    </div>
  )
}
