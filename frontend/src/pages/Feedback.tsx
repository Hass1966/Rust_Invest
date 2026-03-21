import { useState } from 'react'
import { submitSurveyFeedback } from '../lib/api'

export default function Feedback() {
  const [understand, setUnderstand] = useState('')
  const [checkDaily, setCheckDaily] = useState('')
  const [trustMore, setTrustMore] = useState('')
  const [missing, setMissing] = useState('')
  const [wouldPay, setWouldPay] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async () => {
    setSubmitting(true)
    try {
      await submitSurveyFeedback({
        q_understand: understand || undefined,
        q_check_daily: checkDaily || undefined,
        q_trust_more: trustMore || undefined,
        q_missing: missing || undefined,
        q_would_pay: wouldPay || undefined,
      })
      setSubmitted(true)
    } catch {
      alert('Failed to submit feedback. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  if (submitted) {
    return (
      <div className="max-w-2xl mx-auto text-center py-20">
        <div className="text-4xl mb-4">&#10003;</div>
        <h2 className="text-xl font-semibold text-white mb-2">Thank you for your feedback</h2>
        <p className="text-gray-400">Your responses help us improve Alpha Signal.</p>
        <button
          onClick={() => { setSubmitted(false); setUnderstand(''); setCheckDaily(''); setTrustMore(''); setMissing(''); setWouldPay('') }}
          className="mt-6 text-cyan-400 hover:text-cyan-300 text-sm cursor-pointer"
        >
          Submit another response
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white">Feedback</h2>
        <p className="text-sm text-gray-500 mt-1">
          Help us improve Alpha Signal. All responses are anonymous.
        </p>
      </div>

      {/* Q1 */}
      <QuestionCard
        number={1}
        question="Did you understand what the signal was telling you?"
        options={['Yes', 'Mostly', 'No']}
        value={understand}
        onChange={setUnderstand}
      />

      {/* Q2 */}
      <QuestionCard
        number={2}
        question="Would you check this app daily?"
        options={['Yes', 'Maybe', 'No']}
        value={checkDaily}
        onChange={setCheckDaily}
      />

      {/* Q3 */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
        <div className="flex items-baseline gap-3 mb-3">
          <span className="text-cyan-400 font-mono text-sm">3.</span>
          <label className="text-gray-300 text-sm font-medium">
            What would make you trust the signals more?
          </label>
        </div>
        <textarea
          value={trustMore}
          onChange={e => setTrustMore(e.target.value)}
          placeholder="e.g. longer track record, more explanation, backtested proof..."
          className="w-full bg-[#0a0e17] border border-[#1f2937] rounded-lg px-4 py-3 text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-cyan-800 resize-none"
          rows={3}
        />
      </div>

      {/* Q4 */}
      <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
        <div className="flex items-baseline gap-3 mb-3">
          <span className="text-cyan-400 font-mono text-sm">4.</span>
          <label className="text-gray-300 text-sm font-medium">
            What is missing?
          </label>
        </div>
        <textarea
          value={missing}
          onChange={e => setMissing(e.target.value)}
          placeholder="e.g. alerts, more assets, risk levels, mobile app..."
          className="w-full bg-[#0a0e17] border border-[#1f2937] rounded-lg px-4 py-3 text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-cyan-800 resize-none"
          rows={3}
        />
      </div>

      {/* Q5 */}
      <QuestionCard
        number={5}
        question="Would you pay for full access?"
        options={['\u00A35/mo', '\u00A310/mo', '\u00A320/mo', '\u00A350/mo', 'No']}
        value={wouldPay}
        onChange={setWouldPay}
      />

      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 text-white font-medium py-3 rounded-xl transition-colors cursor-pointer"
      >
        {submitting ? 'Submitting...' : 'Submit Feedback'}
      </button>
    </div>
  )
}

function QuestionCard({
  number, question, options, value, onChange,
}: {
  number: number
  question: string
  options: string[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
      <div className="flex items-baseline gap-3 mb-3">
        <span className="text-cyan-400 font-mono text-sm">{number}.</span>
        <span className="text-gray-300 text-sm font-medium">{question}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {options.map(opt => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={`px-4 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
              value === opt
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                : 'bg-[#0a0e17] text-gray-400 border border-[#1f2937] hover:border-gray-600'
            }`}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  )
}
