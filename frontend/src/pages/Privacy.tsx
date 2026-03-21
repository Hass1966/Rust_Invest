import { Link } from 'react-router-dom'

export default function Privacy() {
  return (
    <div className="max-w-3xl mx-auto space-y-8 py-4">
      <div>
        <h2 className="text-2xl font-bold text-white">Privacy Policy</h2>
        <p className="text-gray-500 text-sm mt-1">Last updated: March 2026</p>
      </div>

      <Section title="What data we collect">
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed">
          <li><strong className="text-gray-300">Email address</strong> &mdash; collected during sign-in via Google or Microsoft OAuth. Used to identify your account.</li>
          <li><strong className="text-gray-300">Portfolio holdings</strong> &mdash; the assets, quantities, and start dates you add to your portfolio. Stored to generate personalised signals.</li>
          <li><strong className="text-gray-300">Usage data</strong> &mdash; pages visited and features used, collected to improve the product. No tracking pixels or third-party analytics.</li>
        </ul>
      </Section>

      <Section title="How we use your data">
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed">
          <li>To generate personalised trading signals for your portfolio</li>
          <li>To send daily email alerts when your signals change (you can unsubscribe at any time)</li>
          <li>To improve signal accuracy and user experience</li>
        </ul>
      </Section>

      <Section title="What we never do">
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed">
          <li>We <strong className="text-gray-300">never sell</strong> your data to third parties</li>
          <li>We <strong className="text-gray-300">never store</strong> financial credentials, bank details, or brokerage logins</li>
          <li>We <strong className="text-gray-300">never share</strong> your portfolio with other users</li>
          <li>We do not use third-party advertising or tracking services</li>
        </ul>
      </Section>

      <Section title="Data storage and security">
        <p className="text-gray-400 text-sm leading-relaxed">
          Your data is stored in an encrypted database on secure servers. Authentication is handled
          via OAuth 2.0 (Google and Microsoft) &mdash; we never see or store your password. Session tokens
          are stored in memory only and expire after 24 hours.
        </p>
      </Section>

      <Section title="Your rights (GDPR)">
        <ul className="list-disc list-inside space-y-2 text-gray-400 text-sm leading-relaxed">
          <li><strong className="text-gray-300">Access</strong> &mdash; you can request a copy of all data we hold about you</li>
          <li><strong className="text-gray-300">Deletion</strong> &mdash; you can request complete deletion of your account and all associated data</li>
          <li><strong className="text-gray-300">Portability</strong> &mdash; you can export your data in a standard format</li>
          <li><strong className="text-gray-300">Rectification</strong> &mdash; you can update or correct your data at any time via the Settings page</li>
        </ul>
      </Section>

      <Section title="Contact">
        <p className="text-gray-400 text-sm leading-relaxed">
          For any privacy-related questions or requests, contact us at{' '}
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
