import { Link } from 'react-router-dom'

export default function Terms() {
  return (
    <div className="max-w-3xl mx-auto space-y-8 py-4">
      <div>
        <h2 className="text-2xl font-bold text-white">Terms of Service</h2>
        <p className="text-gray-500 text-sm mt-1">Last updated: March 2026</p>
      </div>

      <Section title="1. Service description">
        <p className="text-gray-400 text-sm leading-relaxed">
          Alpha Signal is an AI-powered market signal platform that provides algorithmic trading signals
          for stocks, forex, and cryptocurrency. The service analyses market data using machine learning
          models and presents buy, hold, and sell signals with confidence scores.
        </p>
      </Section>

      <Section title="2. Not financial advice">
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4 mb-3">
          <p className="text-amber-400 text-sm font-medium leading-relaxed">
            Alpha Signal provides information and analysis only. It does not constitute financial advice,
            investment advice, trading advice, or any other form of professional advice.
          </p>
        </div>
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed">
          <li>Past performance does not guarantee future results</li>
          <li>Model accuracy can vary significantly across market conditions</li>
          <li>Users make their own investment decisions and bear full responsibility for those decisions</li>
          <li>You should consult a qualified financial advisor before making investment decisions</li>
        </ul>
      </Section>

      <Section title="3. No guarantees">
        <p className="text-gray-400 text-sm leading-relaxed">
          The service is provided &ldquo;as-is&rdquo; during the beta period. We make no guarantees regarding:
        </p>
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed mt-2">
          <li>Signal accuracy or profitability</li>
          <li>Uptime or availability of the service</li>
          <li>Completeness or timeliness of market data</li>
          <li>Suitability of signals for your specific financial situation</li>
        </ul>
      </Section>

      <Section title="4. User responsibilities">
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed">
          <li>You must be at least 18 years old to use this service</li>
          <li>You are responsible for keeping your account secure</li>
          <li>You agree not to use the service for any unlawful purpose</li>
          <li>You agree not to attempt to reverse-engineer the signal generation algorithms</li>
        </ul>
      </Section>

      <Section title="5. Beta service">
        <p className="text-gray-400 text-sm leading-relaxed">
          Alpha Signal is currently in beta. During the beta period:
        </p>
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed mt-2">
          <li>The service is provided free of charge</li>
          <li>Features may change, be added, or be removed without notice</li>
          <li>Data may be reset during major updates (we will provide advance notice)</li>
        </ul>
      </Section>

      <Section title="6. Limitation of liability">
        <p className="text-gray-400 text-sm leading-relaxed">
          To the maximum extent permitted by law, Alpha Signal and its operators shall not be liable
          for any losses, damages, or costs arising from the use of this service, including but not
          limited to trading losses, lost profits, or consequential damages.
        </p>
      </Section>

      <Section title="7. Governing law">
        <p className="text-gray-400 text-sm leading-relaxed">
          These terms are governed by and construed in accordance with the laws of England and Wales.
          Any disputes shall be subject to the exclusive jurisdiction of the courts of England and Wales.
        </p>
      </Section>

      <Section title="8. Contact">
        <p className="text-gray-400 text-sm leading-relaxed">
          For questions about these terms, contact us at{' '}
          <a href="mailto:hassan@hassanshuman.co.uk" className="text-cyan-400 hover:underline">
            hassan@hassanshuman.co.uk
          </a>
        </p>
      </Section>

      <div className="pt-4 border-t border-[#1f2937]">
        <Link to="/" className="text-cyan-400 text-sm hover:underline">&larr; Back to Dashboard</Link>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
      <h3 className="text-lg font-semibold text-white mb-3">{title}</h3>
      {children}
    </div>
  )
}
