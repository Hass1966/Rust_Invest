import { useEffect, useState, useCallback } from 'react'
import { RefreshCw } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  AreaChart, Area,
} from 'recharts'
import {
  fetchAgentStatus,
  fetchAgentActions,
  fetchAgentMetrics,
  fetchReflections,
  fetchPaperPortfolio,
} from '../lib/api'
import type {
  AgentStatus,
  AgentAction,
  AgentMetrics,
  Reflection,
  PaperPortfolioData,
} from '../lib/api'

// ── Tooltip styling (matches existing pages) ──

const tooltipStyle = {
  background: '#111827',
  border: '1px solid #1f2937',
  borderRadius: '8px',
  fontSize: 12,
}

// ── Section wrapper with independent loading/error ──

function Section({
  title,
  loading,
  error,
  onRetry,
  children,
  className = '',
}: {
  title: string
  loading: boolean
  error: string | null
  onRetry?: () => void
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={`bg-[#111827] border border-[#1f2937] rounded-lg p-6 ${className}`}>
      <h3 className="text-white font-semibold mb-4">{title}</h3>
      {loading ? (
        <div className="space-y-3">
          <div className="h-4 bg-gray-700/50 rounded w-full animate-pulse" />
          <div className="h-4 bg-gray-700/50 rounded w-4/5 animate-pulse" />
          <div className="h-4 bg-gray-700/50 rounded w-3/5 animate-pulse" />
        </div>
      ) : error ? (
        <div className="text-gray-500 text-sm">
          <p>{error}</p>
          {onRetry && (
            <button onClick={onRetry} className="text-cyan-400 text-xs mt-1 hover:underline cursor-pointer">
              Retry
            </button>
          )}
        </div>
      ) : (
        children
      )}
    </div>
  )
}

// ── Main Page ──

export default function SystemHealth() {
  // Agent status
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)
  const [statusLoading, setStatusLoading] = useState(true)
  const [statusError, setStatusError] = useState<string | null>(null)

  // Agent actions
  const [actions, setActions] = useState<AgentAction[]>([])
  const [actionsLoading, setActionsLoading] = useState(true)
  const [actionsError, setActionsError] = useState<string | null>(null)

  // Agent metrics
  const [metrics, setMetrics] = useState<AgentMetrics | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(true)
  const [metricsError, setMetricsError] = useState<string | null>(null)

  // Reflections
  const [reflections, setReflections] = useState<Reflection[]>([])
  const [reflectionsLoading, setReflectionsLoading] = useState(true)
  const [reflectionsError, setReflectionsError] = useState<string | null>(null)

  // Paper portfolio
  const [portfolio, setPortfolio] = useState<PaperPortfolioData | null>(null)
  const [portfolioLoading, setPortfolioLoading] = useState(true)
  const [portfolioError, setPortfolioError] = useState<string | null>(null)

  // ── Loaders ──

  const loadStatus = useCallback(() => {
    setStatusLoading(true)
    setStatusError(null)
    fetchAgentStatus()
      .then(setAgentStatus)
      .catch(() => setStatusError('Could not load agent status.'))
      .finally(() => setStatusLoading(false))
  }, [])

  const loadActions = useCallback(() => {
    setActionsLoading(true)
    setActionsError(null)
    fetchAgentActions(50)
      .then(setActions)
      .catch(() => setActionsError('Could not load agent actions.'))
      .finally(() => setActionsLoading(false))
  }, [])

  const loadMetrics = useCallback(() => {
    setMetricsLoading(true)
    setMetricsError(null)
    fetchAgentMetrics()
      .then(setMetrics)
      .catch(() => setMetricsError('Could not load accuracy metrics.'))
      .finally(() => setMetricsLoading(false))
  }, [])

  const loadReflections = useCallback(() => {
    setReflectionsLoading(true)
    setReflectionsError(null)
    fetchReflections(1)
      .then(setReflections)
      .catch(() => setReflectionsError('Could not load reflections.'))
      .finally(() => setReflectionsLoading(false))
  }, [])

  const loadPortfolio = useCallback(() => {
    setPortfolioLoading(true)
    setPortfolioError(null)
    fetchPaperPortfolio()
      .then(setPortfolio)
      .catch(() => setPortfolioError('Could not load paper portfolio.'))
      .finally(() => setPortfolioLoading(false))
  }, [])

  const loadAll = useCallback(() => {
    loadStatus()
    loadActions()
    loadMetrics()
    loadReflections()
    loadPortfolio()
  }, [loadStatus, loadActions, loadMetrics, loadReflections, loadPortfolio])

  useEffect(() => { loadAll() }, [loadAll])

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(loadAll, 60_000)
    return () => clearInterval(interval)
  }, [loadAll])

  // ── Derived data ──

  const retrainActions = actions.filter(a => a.action_type === 'retrain')

  const accuracyChartData = metrics
    ? metrics.dates.map((date, i) => ({
        date,
        buy: metrics.buy_accuracy[i],
        sell: metrics.sell_accuracy[i],
        overall: metrics.overall_accuracy[i],
      }))
    : []

  const portfolioChartData = portfolio
    ? portfolio.dates.map((date, i) => ({
        date,
        value: portfolio.values[i],
        return: portfolio.daily_returns[i],
      }))
    : []

  // ── Regime colour ──

  function regimeColor(regime: string): string {
    const r = regime.toUpperCase()
    if (r === 'BULL') return 'text-emerald-400'
    if (r === 'BEAR') return 'text-red-400'
    return 'text-amber-400'
  }

  function regimeBg(regime: string): string {
    const r = regime.toUpperCase()
    if (r === 'BULL') return 'bg-emerald-500/10'
    if (r === 'BEAR') return 'bg-red-500/10'
    return 'bg-amber-500/10'
  }

  // ── Action status colour ──

  function actionStatusColor(status: string): string {
    if (status === 'executed') return 'text-emerald-400'
    if (status === 'proposed') return 'text-amber-400'
    if (status === 'suspended') return 'text-red-400'
    return 'text-gray-400'
  }

  function actionDotColor(status: string): string {
    if (status === 'executed') return 'bg-emerald-400'
    if (status === 'proposed') return 'bg-amber-400'
    if (status === 'suspended') return 'bg-red-400'
    return 'bg-gray-400'
  }

  // ── Format helpers ──

  function fmtTime(iso: string): string {
    const d = new Date(iso)
    return d.toLocaleString('en-GB', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  function fmtDate(iso: string): string {
    const d = new Date(iso)
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  }

  function fmtCurrency(val: number): string {
    return val.toLocaleString(undefined, {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    })
  }

  // ── Render ──

  return (
    <div>
      {/* Page header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-white text-xl font-semibold">System Health</h2>
          <p className="text-gray-500 text-sm">Autonomous agent monitoring and diagnostics</p>
        </div>
        <button
          onClick={loadAll}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/15 text-cyan-400 rounded-lg text-sm hover:bg-cyan-500/25 transition-colors cursor-pointer"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* 1. Agent Status Banner (full width) */}
      <Section
        title="Agent Status"
        loading={statusLoading}
        error={statusError}
        onRetry={loadStatus}
        className="mb-6"
      >
        {agentStatus && (
          <div className="flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-8">
            {/* Live/Offline indicator */}
            <div className="flex items-center gap-3">
              <span
                className={`w-3 h-3 rounded-full flex-shrink-0 ${
                  agentStatus.status === 'online'
                    ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                    : 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]'
                }`}
              />
              <span className={`text-lg font-bold ${
                agentStatus.status === 'online' ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {agentStatus.status === 'online' ? 'Live' : 'Offline'}
              </span>
            </div>

            {/* Stats row */}
            <div className="flex flex-wrap gap-6 text-sm">
              <div>
                <span className="text-gray-500">Last Cycle</span>
                <span className="text-white ml-2 font-mono">
                  {agentStatus.last_cycle ? fmtTime(agentStatus.last_cycle) : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Decisions Today</span>
                <span className="text-white ml-2 font-mono">{agentStatus.decisions_today}</span>
              </div>
              <div>
                <span className="text-gray-500">Market Regime</span>
                <span className={`ml-2 font-bold px-2 py-0.5 rounded text-xs ${regimeColor(agentStatus.current_regime)} ${regimeBg(agentStatus.current_regime)}`}>
                  {agentStatus.current_regime.toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Uptime</span>
                <span className="text-white ml-2 font-mono">{agentStatus.uptime_hours.toFixed(1)}h</span>
              </div>
            </div>
          </div>
        )}
      </Section>

      {/* 2-column grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 2. Action Timeline */}
        <Section
          title="Action Timeline"
          loading={actionsLoading}
          error={actionsError}
          onRetry={loadActions}
        >
          {actions.length === 0 ? (
            <p className="text-gray-500 text-sm">No actions recorded yet.</p>
          ) : (
            <div className="space-y-0 max-h-[420px] overflow-y-auto pr-1 scrollbar-thin">
              {actions.map((action) => (
                <div
                  key={action.id}
                  className="flex gap-3 py-3 border-b border-[#1f2937]/50 last:border-b-0"
                >
                  {/* Timeline dot */}
                  <div className="flex flex-col items-center pt-1.5">
                    <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${actionDotColor(action.status)}`} />
                    <span className="w-px flex-1 bg-[#1f2937] mt-1" />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className={`text-xs font-bold uppercase ${actionStatusColor(action.status)}`}>
                        {action.action_type}
                      </span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${actionStatusColor(action.status)} bg-white/5`}>
                        {action.status}
                      </span>
                      {action.asset && (
                        <span className="text-white text-xs font-mono">{action.asset}</span>
                      )}
                    </div>
                    <p className="text-gray-400 text-xs leading-relaxed truncate">{action.reason}</p>
                    {/* Accuracy delta for retrains */}
                    {action.action_type === 'retrain' &&
                      action.accuracy_before != null &&
                      action.accuracy_after != null && (
                        <div className="flex items-center gap-1 mt-1 text-xs">
                          <span className="text-gray-500">{action.accuracy_before.toFixed(1)}%</span>
                          <span className="text-gray-600">&rarr;</span>
                          <span
                            className={
                              action.accuracy_after > action.accuracy_before
                                ? 'text-emerald-400'
                                : action.accuracy_after < action.accuracy_before
                                ? 'text-red-400'
                                : 'text-gray-400'
                            }
                          >
                            {action.accuracy_after.toFixed(1)}%
                          </span>
                          <span className="text-gray-600 ml-1">
                            ({action.accuracy_after > action.accuracy_before ? '+' : ''}
                            {(action.accuracy_after - action.accuracy_before).toFixed(1)}pp)
                          </span>
                        </div>
                      )}
                    <span className="text-gray-600 text-[10px]">{fmtTime(action.created_at)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Section>

        {/* 3. Accuracy Trend */}
        <Section
          title="Accuracy Trend"
          loading={metricsLoading}
          error={metricsError}
          onRetry={loadMetrics}
        >
          {accuracyChartData.length === 0 ? (
            <p className="text-gray-500 text-sm">No accuracy data available yet.</p>
          ) : (
            <div className="h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={accuracyChartData}>
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#1f2937' }}
                    tickFormatter={(v: string) => fmtDate(v)}
                  />
                  <YAxis
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#1f2937' }}
                    domain={[30, 80]}
                    tickFormatter={(v: number) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelFormatter={(label: string) => fmtDate(label)}
                    formatter={(value: number, name: string) => [
                      `${value.toFixed(1)}%`,
                      name === 'buy' ? 'BUY Accuracy' : name === 'sell' ? 'SELL Accuracy' : 'Overall',
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="overall"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={false}
                    name="overall"
                  />
                  <Line
                    type="monotone"
                    dataKey="buy"
                    stroke="#10b981"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    name="buy"
                  />
                  <Line
                    type="monotone"
                    dataKey="sell"
                    stroke="#ef4444"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    name="sell"
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-6 mt-2 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="w-4 h-0.5 bg-cyan-400 rounded" />
                  <span className="text-gray-400">Overall</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-4 h-0.5 bg-emerald-400 rounded border-dashed" />
                  <span className="text-gray-400">BUY</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-4 h-0.5 bg-red-400 rounded" />
                  <span className="text-gray-400">SELL</span>
                </div>
              </div>
            </div>
          )}
        </Section>

        {/* 4. Retrain History */}
        <Section
          title="Retrain History"
          loading={actionsLoading}
          error={actionsError}
          onRetry={loadActions}
        >
          {retrainActions.length === 0 ? (
            <p className="text-gray-500 text-sm">No retrains recorded yet.</p>
          ) : (
            <div className="overflow-x-auto max-h-[380px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 text-xs uppercase border-b border-[#1f2937]">
                    <th className="text-left px-3 py-2">Timestamp</th>
                    <th className="text-left px-3 py-2">Asset</th>
                    <th className="text-right px-3 py-2">Before</th>
                    <th className="text-center px-2 py-2" />
                    <th className="text-right px-3 py-2">After</th>
                    <th className="text-left px-3 py-2">Trigger</th>
                  </tr>
                </thead>
                <tbody>
                  {retrainActions.map((a) => {
                    const improved =
                      a.accuracy_before != null &&
                      a.accuracy_after != null &&
                      a.accuracy_after > a.accuracy_before
                    const declined =
                      a.accuracy_before != null &&
                      a.accuracy_after != null &&
                      a.accuracy_after < a.accuracy_before

                    return (
                      <tr
                        key={a.id}
                        className="border-b border-[#1f2937]/50 hover:bg-white/[0.02]"
                      >
                        <td className="px-3 py-2 text-gray-400 text-xs font-mono whitespace-nowrap">
                          {fmtTime(a.created_at)}
                        </td>
                        <td className="px-3 py-2 text-white font-mono text-xs">
                          {a.asset || '-'}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-xs text-gray-400">
                          {a.accuracy_before != null ? `${a.accuracy_before.toFixed(1)}%` : '-'}
                        </td>
                        <td className="px-2 py-2 text-center text-gray-600 text-xs">&rarr;</td>
                        <td
                          className={`px-3 py-2 text-right font-mono text-xs ${
                            improved
                              ? 'text-emerald-400'
                              : declined
                              ? 'text-red-400'
                              : 'text-gray-400'
                          }`}
                        >
                          {a.accuracy_after != null ? `${a.accuracy_after.toFixed(1)}%` : '-'}
                        </td>
                        <td className="px-3 py-2 text-gray-500 text-xs max-w-[200px] truncate">
                          {a.reason}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        {/* 5. Weekly Reflection */}
        <Section
          title="Weekly Reflection"
          loading={reflectionsLoading}
          error={reflectionsError}
          onRetry={loadReflections}
        >
          {reflections.length === 0 ? (
            <p className="text-gray-500 text-sm">No reflections available yet.</p>
          ) : (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] uppercase tracking-wider text-cyan-400 font-medium px-2 py-0.5 rounded bg-cyan-500/10">
                  {reflections[0].reflection_type}
                </span>
                <span className="text-gray-600 text-xs">
                  {fmtTime(reflections[0].created_at)}
                </span>
              </div>
              <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">
                {reflections[0].content}
              </p>
            </div>
          )}
        </Section>

        {/* 6. Paper Trading P&L (full width) */}
        <Section
          title="Paper Trading P&L"
          loading={portfolioLoading}
          error={portfolioError}
          onRetry={loadPortfolio}
          className="lg:col-span-2"
        >
          {!portfolio || portfolioChartData.length === 0 ? (
            <p className="text-gray-500 text-sm">No paper trading data available yet.</p>
          ) : (
            <>
              {/* Summary stats */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
                <div className="bg-[#0a0e17] rounded-lg p-3">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">Current Value</div>
                  <div className="text-white text-lg font-bold font-mono">
                    {fmtCurrency(portfolio.values[portfolio.values.length - 1])}
                  </div>
                </div>
                <div className="bg-[#0a0e17] rounded-lg p-3">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">Starting Value</div>
                  <div className="text-white text-lg font-bold font-mono">
                    {fmtCurrency(portfolio.values[0])}
                  </div>
                </div>
                <div className="bg-[#0a0e17] rounded-lg p-3">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">Total Return</div>
                  {(() => {
                    const start = portfolio.values[0]
                    const end = portfolio.values[portfolio.values.length - 1]
                    const pct = start > 0 ? ((end - start) / start) * 100 : 0
                    return (
                      <div className={`text-lg font-bold font-mono ${pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
                      </div>
                    )
                  })()}
                </div>
                <div className="bg-[#0a0e17] rounded-lg p-3">
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">Days Tracked</div>
                  <div className="text-white text-lg font-bold font-mono">
                    {portfolio.dates.length}
                  </div>
                </div>
              </div>

              {/* Portfolio value chart */}
              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={portfolioChartData}>
                    <defs>
                      <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="date"
                      tick={{ fill: '#6b7280', fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: '#1f2937' }}
                      tickFormatter={(v: string) => fmtDate(v)}
                    />
                    <YAxis
                      tick={{ fill: '#6b7280', fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: '#1f2937' }}
                      tickFormatter={(v: number) => fmtCurrency(v)}
                      width={80}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      labelFormatter={(label: string) => fmtDate(label)}
                      formatter={(value: number, name: string) => {
                        if (name === 'value') return [fmtCurrency(value), 'Portfolio Value']
                        return [`${value.toFixed(2)}%`, 'Daily Return']
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#22d3ee"
                      strokeWidth={2}
                      fill="url(#portfolioGradient)"
                      name="value"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </Section>
      </div>
    </div>
  )
}
