import { useState, useEffect, useRef } from 'react'
import { Send, Copy, Check, ChevronDown } from 'lucide-react'
import { sendChat, fetchDailyTracker, fetchSignals } from '../lib/api'
import type { ChatMessage, EnrichedSignal, DailyTrackerResult } from '../lib/types'

const QUICK_QUESTIONS = [
  'What should I do with my portfolio today?',
  'Am I taking too much risk?',
  'Which signals are strongest right now?',
  'Should I buy more of anything?',
  'Explain what the Sharpe ratio means',
  'Is now a good time to start investing?',
]

function fmt(n: number, dp = 2): string { return n.toFixed(dp) }

export default function Advisor() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [tracker, setTracker] = useState<DailyTrackerResult | null>(null)
  const [signals, setSignals] = useState<EnrichedSignal[]>([])
  const [contextExpanded, setContextExpanded] = useState(false)
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchDailyTracker().then(setTracker).catch(() => {})
    fetchSignals().then(setSignals).catch(() => {})
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const buys = signals.filter(s => s.signal === 'BUY').length
  const sells = signals.filter(s => s.signal === 'SELL').length

  const mood = buys > sells + 2
    ? { icon: 'positive', text: 'Generally positive signals today', color: 'text-emerald-400' }
    : sells > buys + 2
    ? { icon: 'caution', text: 'More caution signals than usual', color: 'text-red-400' }
    : { icon: 'mixed', text: 'Mixed signals — be selective', color: 'text-amber-400' }

  const moodExplain = buys > sells + 2
    ? 'Most of our models are seeing buying opportunities today. This can be a good time to review your watchlist.'
    : sells > buys + 2
    ? 'Our models are flagging more caution than usual. Consider holding off on new positions until signals improve.'
    : 'Some assets look attractive while others show caution. Focus on the highest-quality signals if acting today.'

  async function handleSend(text?: string) {
    const msg = (text || input).trim()
    if (!msg || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: msg }])
    setLoading(true)
    try {
      const response = await sendChat(msg, 'advisor')
      setMessages(prev => [...prev, { role: 'assistant', content: response }])
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Failed to get response. Is the server running?' }])
    } finally {
      setLoading(false)
    }
  }

  function copyMessage(idx: number, content: string) {
    navigator.clipboard.writeText(content)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 2000)
  }

  return (
    <div className="flex gap-6 h-[calc(100vh-80px)]">
      {/* Left Sidebar — hidden on mobile */}
      <div className="hidden md:flex flex-col gap-4 w-[280px] flex-shrink-0 overflow-y-auto">
        {/* Portfolio Snapshot */}
        {tracker?.has_data && (
          <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
            <h4 className="text-gray-400 text-xs uppercase tracking-wider mb-3">Portfolio Snapshot</h4>
            <div className="space-y-2">
              <MetricRow label="Current Value" value={`£${Math.round(tracker.current_value ?? 0).toLocaleString()}`} color="text-cyan-400" />
              <MetricRow
                label="Today"
                value={`${(tracker.daily_return ?? 0) >= 0 ? '+' : ''}${fmt(tracker.daily_return ?? 0)}%`}
                color={(tracker.daily_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}
              />
              <MetricRow
                label="Since Inception"
                value={`${(tracker.cumulative_return ?? 0) >= 0 ? '+' : ''}${fmt(tracker.cumulative_return ?? 0)}%`}
                color={(tracker.cumulative_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}
              />
              <MetricRow
                label="Accuracy"
                value={`${fmt(tracker.model_accuracy_pct ?? 0, 1)}%`}
                color={(tracker.model_accuracy_pct ?? 50) >= 55 ? 'text-emerald-400' : 'text-amber-400'}
              />
            </div>
          </div>
        )}

        {/* Market Mood */}
        <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
          <h4 className="text-gray-400 text-xs uppercase tracking-wider mb-3">Market Mood</h4>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">{mood.icon === 'positive' ? '\u{1F7E2}' : mood.icon === 'caution' ? '\u{1F534}' : '\u{1F7E1}'}</span>
            <span className={`text-sm font-medium ${mood.color}`}>{mood.text}</span>
          </div>
          <p className="text-gray-500 text-xs leading-relaxed">{moodExplain}</p>
          <div className="flex gap-3 mt-2 text-xs text-gray-500">
            <span className="text-emerald-400">{buys} BUY</span>
            <span className="text-red-400">{sells} SELL</span>
            <span className="text-amber-400">{signals.length - buys - sells} HOLD</span>
          </div>
        </div>

        {/* Quick Questions */}
        <div className="bg-[#111827] border border-[#1f2937] rounded-lg p-4">
          <h4 className="text-gray-400 text-xs uppercase tracking-wider mb-3">Quick Questions</h4>
          <div className="flex flex-wrap gap-2">
            {QUICK_QUESTIONS.map((q, i) => (
              <button
                key={i}
                onClick={() => handleSend(q)}
                className="text-xs px-3 py-1.5 rounded-full bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors text-left cursor-pointer"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Right Panel — Chat */}
      <div className="flex-1 flex flex-col bg-[#111827] border border-[#1f2937] rounded-lg overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 border-b border-[#1f2937]">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-semibold">AI Advisor</h3>
          </div>
          <p className="text-gray-600 text-xs">Powered by local AI — your data stays on your network</p>
        </div>

        {/* Context pill */}
        {signals.length > 0 && (
          <div className="px-4 py-2 border-b border-[#1f2937]/50">
            <button
              onClick={() => setContextExpanded(!contextExpanded)}
              className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-400 cursor-pointer"
            >
              <span>Portfolio context loaded · {signals.length} assets{tracker?.has_data ? ` · Today ${(tracker.daily_return ?? 0) >= 0 ? '+' : ''}${fmt(tracker.daily_return ?? 0)}%` : ''}</span>
              <ChevronDown className={`w-3 h-3 transition-transform ${contextExpanded ? 'rotate-180' : ''}`} />
            </button>
            {contextExpanded && (
              <div className="mt-2 text-xs text-gray-600 grid grid-cols-3 gap-1">
                {signals.slice(0, 15).map(s => (
                  <span key={s.asset}>
                    <span className={s.signal === 'BUY' ? 'text-emerald-400' : s.signal === 'SELL' ? 'text-red-400' : 'text-amber-400'}>
                      {s.signal}
                    </span>{' '}{s.asset}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.length === 0 && (
            <div className="text-gray-500 text-sm text-center mt-8">
              <p>Ask anything about your portfolio, signals, or investing concepts.</p>
              {/* Mobile quick questions */}
              <div className="md:hidden flex flex-wrap gap-2 mt-4 justify-center">
                {QUICK_QUESTIONS.slice(0, 3).map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSend(q)}
                    className="text-xs px-3 py-1.5 rounded-full bg-cyan-500/10 text-cyan-400 cursor-pointer"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm relative group ${
                msg.role === 'user'
                  ? 'bg-cyan-500/15 text-cyan-100'
                  : 'bg-[#0a0e17] text-gray-300 border border-[#1f2937]'
              }`}>
                {msg.role === 'assistant' && (
                  <button
                    onClick={() => copyMessage(i, msg.content)}
                    className="absolute top-1 right-1 p-1 rounded opacity-0 group-hover:opacity-100 text-gray-600 hover:text-gray-400 transition-opacity cursor-pointer"
                    title="Copy"
                  >
                    {copiedIdx === i ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  </button>
                )}
                {msg.content.split('\n').map((line, j) => (
                  <span key={j}>
                    {line.split(/(\*\*[^*]+\*\*)/).map((part, k) =>
                      part.startsWith('**') && part.endsWith('**')
                        ? <strong key={k} className="text-white">{part.slice(2, -2)}</strong>
                        : <span key={k}>{part}</span>
                    )}
                    {j < msg.content.split('\n').length - 1 && <br />}
                  </span>
                ))}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-[#0a0e17] border border-[#1f2937] rounded-lg px-4 py-2 flex gap-1">
                <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-3 border-t border-[#1f2937]">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSend()
              }}
              placeholder="Ask about signals, risk, positions... (Ctrl+Enter to send)"
              rows={2}
              className="flex-1 bg-[#0a0e17] border border-[#1f2937] rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 outline-none focus:border-cyan-500/50 resize-none"
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              className="bg-cyan-500/15 text-cyan-400 rounded-lg px-3 py-2 hover:bg-cyan-500/25 disabled:opacity-30 transition-colors cursor-pointer self-end"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500 text-xs">{label}</span>
      <span className={`text-sm font-medium ${color}`}>{value}</span>
    </div>
  )
}
