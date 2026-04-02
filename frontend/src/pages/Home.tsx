import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  TrendingUp, BarChart3, Target, Compass, Wallet, Brain,
  ArrowRight, Activity, Shield, Zap, GitMerge,
  LineChart, Binary, TreeDeciduous, Network, Layers, Eye,
  AlertTriangle, BookOpen, Newspaper
} from 'lucide-react'
import { fetchSignalTruth, type SignalTruthData } from '../lib/api'

/* ── tiny helpers ── */
const Stat = ({ label, value, sub }: { label: string; value: string; sub?: string }) => (
  <div className="text-center">
    <div className="text-3xl sm:text-4xl font-bold text-cyan-400">{value}</div>
    <div className="text-sm text-gray-400 mt-1">{label}</div>
    {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
  </div>
)

const Card = ({ icon: Icon, title, children, accent = 'cyan' }: {
  icon: React.ElementType; title: string; children: React.ReactNode; accent?: string
}) => {
  const colors: Record<string, string> = {
    cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
    green: 'text-green-400 bg-green-500/10 border-green-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
    red: 'text-red-400 bg-red-500/10 border-red-500/20',
  }
  const c = colors[accent] || colors.cyan
  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-6 hover:border-[#374151] transition-colors">
      <div className={`w-10 h-10 rounded-lg ${c} border flex items-center justify-center mb-4`}>
        <Icon className="w-5 h-5" />
      </div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <div className="text-gray-400 text-sm leading-relaxed">{children}</div>
    </div>
  )
}

const ModelCard = ({ icon: Icon, name, what, why, accent }: {
  icon: React.ElementType; name: string; what: string; why: string; accent: string
}) => (
  <div className="bg-[#0d1117] border border-[#1f2937] rounded-xl p-5 hover:border-[#374151] transition-colors">
    <div className="flex items-center gap-3 mb-3">
      <div className={`w-8 h-8 rounded-lg ${accent} border flex items-center justify-center`}>
        <Icon className="w-4 h-4" />
      </div>
      <h4 className="font-semibold text-white">{name}</h4>
    </div>
    <p className="text-gray-400 text-sm mb-2"><span className="text-gray-300 font-medium">What it does:</span> {what}</p>
    <p className="text-gray-500 text-sm"><span className="text-gray-400 font-medium">Why we use it:</span> {why}</p>
  </div>
)

export default function Home() {
  const [truth, setTruth] = useState<SignalTruthData | null>(null)

  useEffect(() => {
    fetchSignalTruth().then(setTruth).catch(() => {})
  }, [])

  const accuracy = truth ? `${truth.overall_accuracy.toFixed(1)}%` : '...'
  const totalSignals = truth ? truth.total_signals.toLocaleString() : '...'
  const resolved = truth ? truth.total_resolved.toLocaleString() : '...'

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-16">

      {/* ─── HERO ─── */}
      <section className="text-center pt-8 sm:pt-14">
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 rounded-2xl bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center">
            <TrendingUp className="w-9 h-9 text-cyan-400" />
          </div>
        </div>

        <h1 className="text-4xl sm:text-5xl font-bold text-white mb-4 leading-tight">
          AI-Powered Trading Signals
        </h1>

        <p className="text-lg sm:text-xl text-gray-400 max-w-2xl mx-auto mb-8 leading-relaxed">
          Alpha Signal watches <span className="text-cyan-400 font-semibold">196 assets</span> around the clock
          — stocks, currencies and crypto — and tells you in plain English
          whether to <span className="text-green-400 font-medium">buy</span>,{' '}
          <span className="text-red-400 font-medium">sell</span> or{' '}
          <span className="text-amber-400 font-medium">hold</span>.
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-semibold transition-colors"
          >
            Open Dashboard <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/track-record"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-[#1f2937] text-gray-300 hover:text-white hover:border-[#374151] transition-colors"
          >
            <Target className="w-4 h-4" /> View Track Record
          </Link>
        </div>
      </section>

      {/* ─── WHAT IS ALPHA SIGNAL? ─── */}
      <section>
        <h2 className="text-2xl font-bold text-white text-center mb-2">What is Alpha Signal?</h2>
        <p className="text-gray-400 text-center max-w-3xl mx-auto mb-8">
          Think of it as a second opinion for your investments. Our AI analyses price data,
          trends, momentum, sentiment from the news, and dozens of other factors every day,
          then gives you a simple recommendation for each asset — no jargon, no confusing charts.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Card icon={Eye} title="Watches 196 Assets" accent="cyan">
            Tracks 150 stocks &amp; ETFs (US, UK, European), 30 FX pairs, and 16 cryptocurrencies
            across bonds, commodities, defensive sectors and growth. New assets added regularly.
          </Card>
          <Card icon={Brain} title="6 AI Models Vote" accent="purple">
            Every signal comes from six different AI models that each cast a vote.
            When most agree, you get a high-confidence signal. When they disagree,
            the signal is marked lower confidence so you know to be careful.
          </Card>
          <Card icon={Newspaper} title="Reads the News" accent="green">
            Claude AI reads headlines from Google, NewsAPI, and Reddit to understand
            what's happening in the world — wars, interest rates, earnings — and adjusts
            signals accordingly.
          </Card>
          <Card icon={Shield} title="Honest Track Record" accent="amber">
            Every signal is recorded and checked against what actually happened.
            The Track Record page shows the real numbers — wins <em>and</em> losses.
            Nothing is hidden.
          </Card>
          <Card icon={Zap} title="Daily Updates" accent="cyan">
            New signals are generated every day so you always have the latest view.
            A morning briefing (powered by Claude AI) summarises the key opportunities.
          </Card>
          <Card icon={Activity} title="Plain English" accent="green">
            Each signal comes with a reason written in plain language, not financial
            jargon. You'll see exactly <em>why</em> the AI thinks a stock will go up or down.
          </Card>
        </div>
      </section>

      {/* ─── LIVE STATS ─── */}
      <section className="bg-[#111827] border border-[#1f2937] rounded-2xl p-8">
        <h2 className="text-2xl font-bold text-white text-center mb-2">Live Track Record</h2>
        <p className="text-gray-400 text-center mb-8 text-sm">
          Real results since 15 March 2026. Updated after every signal resolves.
        </p>

        <div className="grid grid-cols-3 gap-6 mb-6">
          <Stat label="Model Accuracy" value={accuracy} sub="overall correct %" />
          <Stat label="Signals Generated" value={totalSignals} sub="total to date" />
          <Stat label="Signals Resolved" value={resolved} sub="checked against market" />
        </div>

        {truth && truth.by_signal_type.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-6">
            {truth.by_signal_type.map(s => (
              <div key={s.signal_type} className="bg-[#0d1117] rounded-lg p-4 text-center">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                  s.signal_type === 'BUY' ? 'bg-green-500/15 text-green-400'
                    : s.signal_type === 'SELL' ? 'bg-red-500/15 text-red-400'
                    : 'bg-amber-500/15 text-amber-400'
                }`}>{s.signal_type}</span>
                <div className="text-2xl font-bold text-white mt-2">{s.accuracy.toFixed(1)}%</div>
                <div className="text-xs text-gray-500">{s.correct}/{s.total} correct</div>
              </div>
            ))}
          </div>
        )}

        {truth && truth.by_asset_class.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
            {truth.by_asset_class.map(a => (
              <div key={a.asset_class} className="bg-[#0d1117] rounded-lg p-4 flex items-center justify-between">
                <span className="text-sm text-gray-300 capitalize">{a.asset_class}</span>
                <div className="text-right">
                  <span className="text-lg font-bold text-white">{a.accuracy.toFixed(1)}%</span>
                  <span className="text-xs text-gray-500 ml-2">{a.correct}/{a.total}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="text-center mt-6">
          <Link
            to="/track-record"
            className="text-cyan-400 text-sm hover:text-cyan-300 transition-colors inline-flex items-center gap-1"
          >
            See full track record <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </section>

      {/* ─── HOW IT WORKS (simple) ─── */}
      <section>
        <h2 className="text-2xl font-bold text-white text-center mb-2">How It Works</h2>
        <p className="text-gray-400 text-center max-w-2xl mx-auto mb-8">
          Three steps, every single day.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { step: '1', title: 'Gather Data', desc: 'We pull the latest prices, trading volumes, technical indicators, news headlines and social media sentiment for all 196 assets.', icon: Activity },
            { step: '2', title: 'Run the AI Models', desc: 'Six different machine-learning models independently analyse the data and each casts a vote: BUY, SELL or HOLD. We combine them into one consensus signal.', icon: Brain },
            { step: '3', title: 'Deliver Your Signals', desc: 'You open the Dashboard and see a clear card for every asset with the recommendation, the confidence level, and a plain-English explanation of why.', icon: BarChart3 },
          ].map(({ step, title, desc, icon: Icon }) => (
            <div key={step} className="relative bg-[#111827] border border-[#1f2937] rounded-xl p-6">
              <div className="absolute -top-3 left-6 bg-cyan-500 text-black text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center">
                {step}
              </div>
              <div className="mt-2 mb-3">
                <Icon className="w-6 h-6 text-cyan-400" />
              </div>
              <h3 className="text-white font-semibold mb-2">{title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ─── PAGE GUIDE ─── */}
      <section>
        <h2 className="text-2xl font-bold text-white text-center mb-2">What's On Each Page</h2>
        <p className="text-gray-400 text-center max-w-2xl mx-auto mb-8">
          A quick tour of the main tabs so you know where everything lives.
        </p>

        <div className="space-y-3">
          {[
            {
              icon: BarChart3, label: 'Dashboard', path: '/dashboard', accent: 'cyan',
              desc: 'Your home base. See today\'s BUY / SELL / HOLD signals for all 196 assets at a glance, an AI-generated morning briefing that summarises the day\'s top opportunities, and expandable cards showing which models agree and why.'
            },
            {
              icon: Target, label: 'Track Record', path: '/track-record', accent: 'amber',
              desc: 'Full transparency. Every signal we\'ve ever generated is logged here with its outcome — did the price actually go up or down? See overall accuracy, breakdowns by asset class and signal type, rolling accuracy over time, and the best and worst performing assets.'
            },
            {
              icon: Wallet, label: 'My Portfolio', path: '/my-portfolio', accent: 'green',
              desc: 'Add your own holdings and see how Alpha Signal\'s recommendations compare to a simple buy-and-hold strategy. It calculates your Sharpe ratio, win rate, and total return — and shows an equity curve chart so you can see the difference over time. Requires sign-in.'
            },
            {
              icon: Compass, label: 'Explore', path: '/explore', accent: 'purple',
              desc: 'The playground. Run a "what-if" simulation with virtual money to test the strategy risk-free, browse historical signal data, or ask the AI advisor any question about a specific stock or market trend.'
            },
          ].map(({ icon: Icon, label, path, accent, desc }) => {
            const accentClasses: Record<string, string> = {
              cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
              amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
              green: 'text-green-400 bg-green-500/10 border-green-500/20',
              purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
            }
            return (
              <Link
                key={path}
                to={path}
                className="flex items-start gap-4 bg-[#111827] border border-[#1f2937] rounded-xl p-5 hover:border-[#374151] transition-colors group"
              >
                <div className={`w-10 h-10 rounded-lg ${accentClasses[accent]} border flex items-center justify-center shrink-0 mt-0.5`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-white font-semibold group-hover:text-cyan-400 transition-colors">
                    {label} <ArrowRight className="w-3.5 h-3.5 inline opacity-0 group-hover:opacity-100 transition-opacity" />
                  </h3>
                  <p className="text-gray-400 text-sm mt-1 leading-relaxed">{desc}</p>
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* ─── UNDER THE HOOD (for the mathematicians) ─── */}
      <section>
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-medium mb-4">
            <BookOpen className="w-3 h-3" /> For the Technically Curious
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Under the Hood: The Ensemble</h2>
          <p className="text-gray-400 max-w-3xl mx-auto">
            Alpha Signal doesn't rely on a single model. It runs <strong className="text-white">six independent algorithms</strong> and
            combines their votes using a weighted ensemble. Each model has different strengths — by blending
            them together, we reduce the chance of any single model's blind spots causing a bad signal.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <ModelCard
            icon={LineChart}
            name="Linear Regression"
            accent="text-blue-400 bg-blue-500/10 border-blue-500/20"
            what="Fits a straight line through historical price-change data to predict tomorrow's move. The simplest model in the ensemble."
            why="It's fast, interpretable, and provides a stable baseline. If the market is trending clearly, a straight line often captures it well."
          />
          <ModelCard
            icon={Binary}
            name="Logistic Regression"
            accent="text-indigo-400 bg-indigo-500/10 border-indigo-500/20"
            what="Instead of predicting a price, it predicts the probability that the price will go UP. Outputs a percentage between 0% and 100%."
            why="It reframes the problem as a classification task (up vs. down), which avoids the noise of trying to guess exact price moves."
          />
          <ModelCard
            icon={TreeDeciduous}
            name="Gradient Boosted Trees (GBT)"
            accent="text-green-400 bg-green-500/10 border-green-500/20"
            what="Builds hundreds of small decision trees, each one learning from the mistakes of the previous. The final prediction is the combined vote of all trees."
            why="GBT is the workhorse of quantitative finance. It handles non-linear patterns, missing data, and feature interactions that linear models can't see."
          />
          <ModelCard
            icon={Network}
            name="LSTM Neural Network"
            accent="text-pink-400 bg-pink-500/10 border-pink-500/20"
            what="A deep-learning model designed for sequences. It reads the last 60 days of data in order and remembers patterns that unfold over time."
            why="Markets have memory — a crash doesn't just affect today, it echoes for weeks. LSTMs are specifically built to capture these time-dependent patterns."
          />
          <ModelCard
            icon={Layers}
            name="Regime Ensemble"
            accent="text-amber-400 bg-amber-500/10 border-amber-500/20"
            what="First detects what 'regime' the market is in (trending up, trending down, or choppy sideways), then applies the model that works best in that regime."
            why="No single model works everywhere. Bull markets and crashes behave very differently, so we switch strategies depending on market conditions."
          />
          <ModelCard
            icon={GitMerge}
            name="Temporal Fusion Transformer"
            accent="text-cyan-400 bg-cyan-500/10 border-cyan-500/20"
            what="A state-of-the-art deep learning architecture that combines attention mechanisms with temporal processing. It learns which features matter most at each point in time."
            why="TFT can weigh recent vs. older data adaptively and handles the full 154-feature set, including news sentiment, sector data, BOE/ECB rates, macro indicators, and cross-asset signals."
          />
        </div>

        {/* Ensemble explanation */}
        <div className="mt-8 bg-[#111827] border border-[#1f2937] rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <GitMerge className="w-5 h-5 text-cyan-400" /> How the Ensemble Combines Votes
          </h3>
          <div className="text-gray-400 text-sm leading-relaxed space-y-3">
            <p>
              Each model independently scores every asset and casts a vote: <span className="text-green-400">BUY</span>,{' '}
              <span className="text-red-400">SELL</span>, or <span className="text-amber-400">HOLD</span>.
              But not all models are treated equally — each model's vote is weighted by how well it has
              performed on recent out-of-sample data (called <em className="text-gray-300">walk-forward validation</em>).
            </p>
            <p>
              The weighted votes are combined into a single probability. If most models agree strongly,
              the signal is marked <strong className="text-white">High Confidence</strong>. If they're split,
              it's marked <strong className="text-white">Low Confidence</strong> — a sign to be cautious.
            </p>
            <p>
              On top of this, a <span className="text-purple-400">news sentiment layer</span> powered by Claude AI
              reads the latest headlines and adjusts the final signal. For example, if the models say BUY
              but there's breaking news about an SEC investigation, the sentiment layer can dampen the signal.
            </p>
          </div>
        </div>

        {/* Feature engineering note */}
        <div className="mt-4 bg-[#0d1117] border border-[#1f2937] rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" /> 143 Features Per Asset
          </h3>
          <p className="text-gray-400 text-sm leading-relaxed mb-4">
            Each model receives a rich feature vector covering 15 categories of market data. Here are the main groups:
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            {[
              'Price returns (1d–60d)',
              'Moving averages',
              'RSI & momentum',
              'MACD signals',
              'Bollinger Bands',
              'Volume changes',
              'Volatility (ATR)',
              'Drawdown metrics',
              'Trend strength',
              'Sector ETF data',
              'Mean reversion',
              'FX carry & rates',
              'News sentiment',
              'Reddit mentions',
              'Sentiment momentum',
            ].map(f => (
              <div key={f} className="bg-[#111827] rounded-lg px-3 py-2 text-gray-400 border border-[#1f2937]">{f}</div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── DISCLAIMER ─── */}
      <section className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white mb-2">Important Disclaimer</h2>
            <div className="text-gray-400 text-sm leading-relaxed space-y-2">
              <p>
                <strong className="text-amber-400">Alpha Signal is not financial advice.</strong>{' '}
                This is an educational project and a technology demonstration built by a solo developer.
                The signals are generated by machine-learning models that can and do get things wrong.
              </p>
              <p>
                Never invest money you can't afford to lose. Always do your own research and consider
                consulting a qualified financial advisor before making investment decisions. Past performance
                — including any accuracy figures shown on this site — does not guarantee future results.
              </p>
              <p className="text-gray-500">
                Markets are inherently unpredictable. Our models work with probabilities, not certainties.
                The Track Record page exists specifically so you can judge the system's reliability for yourself.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ─── CTA ─── */}
      <section className="text-center">
        <h2 className="text-2xl font-bold text-white mb-3">Ready to explore?</h2>
        <p className="text-gray-400 mb-6">Jump into the Dashboard and see today's signals.</p>
        <Link
          to="/dashboard"
          className="inline-flex items-center gap-2 px-8 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-semibold transition-colors text-lg"
        >
          Open Dashboard <ArrowRight className="w-5 h-5" />
        </Link>
      </section>
    </div>
  )
}
