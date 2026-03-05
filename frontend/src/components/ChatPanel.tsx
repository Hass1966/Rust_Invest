import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { sendChat } from '../lib/api'
import type { ChatMessage } from '../lib/types'

export default function ChatPanel({ tabContext }: { tabContext: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  async function handleSend() {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const response = await sendChat(text, tabContext)
      setMessages(prev => [...prev, { role: 'assistant', content: response }])
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Failed to get response. Is the server running?' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[#1f2937]">
        <h3 className="text-sm font-semibold text-white">AI Analyst</h3>
        <p className="text-xs text-gray-500">Context: {tabContext}</p>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-gray-500 text-sm text-center mt-8">
            <p>Ask questions about your portfolio, signals, or market conditions.</p>
            <p className="mt-2 text-xs text-gray-600">Examples:</p>
            <p className="text-xs text-gray-600">"Should I increase my AAPL position?"</p>
            <p className="text-xs text-gray-600">"What does the RSI say about TSLA?"</p>
            <p className="text-xs text-gray-600">"Summarize today's signals"</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
              msg.role === 'user'
                ? 'bg-cyan-500/15 text-cyan-100'
                : 'bg-[#0a0e17] text-gray-300 border border-[#1f2937]'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-[#0a0e17] border border-[#1f2937] rounded-lg px-3 py-2">
              <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
            </div>
          </div>
        )}
      </div>

      <div className="p-3 border-t border-[#1f2937]">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="Ask about signals, risk, positions..."
            className="flex-1 bg-[#0a0e17] border border-[#1f2937] rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 outline-none focus:border-cyan-500/50"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="bg-cyan-500/15 text-cyan-400 rounded-lg px-3 py-2 hover:bg-cyan-500/25 disabled:opacity-30 transition-colors cursor-pointer"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
