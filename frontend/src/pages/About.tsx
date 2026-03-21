export default function About() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Hero */}
      <div>
        <h2 className="text-2xl font-bold text-white">About Alpha Signal</h2>
        <p className="text-gray-400 mt-2 leading-relaxed">
          An ML-powered market signal engine built entirely in Rust. It trains a 6-model ensemble
          on 7 years of historical data, generates daily BUY / SELL / HOLD signals for stocks,
          forex, and crypto, and tracks every prediction against real outcomes for full transparency.
        </p>
      </div>

      {/* The 6-Model Ensemble */}
      <Section title="The 6-Model Ensemble">
        <p className="text-gray-400 text-sm mb-4">
          Each asset is evaluated by six independent models. Their probabilities are combined
          via a stacking meta-learner (logistic regression trained on out-of-fold predictions)
          or, as a fallback, an accuracy-squared weighted average. Models that underperform
          (&lt;54% accuracy) are automatically gated out per asset.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          <ModelCard
            name="Linear Regression"
            tag="LinReg"
            accuracy="~69%"
            description="OLS with gradient descent. Predicts return magnitude, passed through sigmoid for P(up). Fast, interpretable baseline — feature weights show what matters most."
            strength="Interpretability"
          />
          <ModelCard
            name="Logistic Regression"
            tag="LogReg"
            accuracy="~69%"
            description="Binary classifier with sigmoid activation and cross-entropy loss. Outputs calibrated P(up) directly. Better probability estimates than LinReg for confidence weighting."
            strength="Calibrated probabilities"
          />
          <ModelCard
            name="Gradient Boosted Trees"
            tag="GBT"
            accuracy="~71%"
            description="80 boosted CART trees fitting log-loss pseudo-residuals. Max depth 4, learning rate 0.08, subsample 80%. Finds non-linear interactions like 'RSI < 30 AND volatility > 2%'."
            strength="Best single model"
            highlight
          />
          <ModelCard
            name="LSTM"
            tag="LSTM"
            accuracy="~54%"
            description="Long Short-Term Memory recurrent network built with Candle. Processes sequences of 10 days using the top 30 GBT-ranked features. Hidden size 64, AdamW optimizer."
            strength="Temporal patterns"
          />
          <ModelCard
            name="GRU"
            tag="GRU"
            accuracy="~54%"
            description="Gated Recurrent Unit — lighter alternative to LSTM with fewer parameters (reset/update gates instead of input/forget/output). Same hyperparameters and feature selection."
            strength="Efficient sequences"
          />
          <ModelCard
            name="Random Forest"
            tag="RF"
            accuracy="~55%"
            description="100 decision trees with bootstrap sampling and random feature subsets (50% per tree). Majority vote classification. Reuses GBT's CART tree builder."
            strength="Robustness"
          />
        </div>
      </Section>

      {/* Signal Generation */}
      <Section title="How Signals Are Generated">
        <div className="space-y-4 text-sm text-gray-400 leading-relaxed">
          <Step n={1} title="Feature Extraction">
            104 features are computed from the latest market data — price technicals (RSI, MACD, Bollinger Bands, SMAs),
            volume metrics, volatility measures, macro indicators (VIX, Treasury yields, sector ETFs),
            and asset-class-specific features (crypto on-chain data, FX carry scores).
          </Step>
          <Step n={2} title="Model Forward Pass">
            All six models produce independent P(up) probabilities. Models with walk-forward
            accuracy below 54% for that specific asset are gated out (weight set to zero).
            Models above 58% accuracy get double weight.
          </Step>
          <Step n={3} title="Ensemble Combination">
            If a stacking meta-learner is available (trained on out-of-fold predictions),
            it combines the six probabilities via logistic regression. Otherwise, an
            accuracy-squared weighted average is used, with GBT receiving a 1.2x bonus.
          </Step>
          <Step n={4} title="Signal Decision">
            The ensemble probability is compared against per-asset thresholds (e.g., TSLA needs
            &gt;0.62 for BUY, &lt;0.38 for SELL; FX pairs use 0.55/0.45). When all active models
            agree, thresholds are relaxed by 0.02. If the best model's accuracy is below 50.5%,
            the system outputs HOLD ("no edge").
          </Step>
          <Step n={5} title="Confidence Scoring">
            Confidence (0–10) is the minimum of: distance from 50/50 (×200), accuracy margin
            above 50% (×2), and the best single model's accuracy margin. This prevents
            overconfidence when model accuracy is modest.
          </Step>
        </div>
      </Section>

      {/* Feature Engineering */}
      <Section title="104 Active Features">
        <p className="text-gray-400 text-sm mb-4">
          119 features are computed in total; 15 identified as noise are pruned. The remaining
          104 span 13 categories. Features are z-score normalized per training fold to prevent
          data leakage.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <FeatureGroup name="Price Technicals" count={26}
            examples="RSI (7/14-day), MACD histogram & delta, SMAs (7/30/50/200), Bollinger Bands, EMAs, price vs 52-week range" />
          <FeatureGroup name="Volume" count={6}
            examples="Volume ratios (5d/20d), OBV slope, price-volume correlation" />
          <FeatureGroup name="Volatility" count={8}
            examples="Rolling vol (5/20/60-day), ATR, Garman-Klass vol, max drawdown 20d" />
          <FeatureGroup name="Momentum" count={10}
            examples="Multi-timeframe returns (1/3/5/10/20-day), up-days ratio" />
          <FeatureGroup name="Calendar" count={5}
            examples="Day-of-week (sin/cos encoded), quarter, Monday/Friday flags" />
          <FeatureGroup name="Market Context" count={20}
            examples="VIX level & delta, Treasury yields & spread, SPY returns, 8 sector ETF returns, gold, dollar" />
          <FeatureGroup name="Lagged" count={8}
            examples="1/2/3-day return lags, lagged RSI, MACD, volatility, volume ratio" />
          <FeatureGroup name="Statistical" count={6}
            examples="Skewness, kurtosis, autocorrelation (1/5-day), Hurst exponent, mean-reversion score" />
          <FeatureGroup name="Cross-Asset" count={12}
            examples="Stock vs sector (5/10/20-day), vs SPY, breadth score, vol-adjusted momentum" />
          <FeatureGroup name="Event & Sentiment" count={8}
            examples="Days to/since earnings, Fear & Greed index, volume surge, smart money flow" />
          <FeatureGroup name="Macro" count={5}
            examples="DXY level & delta, yield curve (10y-2y spread), Fed funds rate" />
          <FeatureGroup name="Crypto On-Chain" count={8}
            examples="BTC/ETH funding rates, active addresses, social scores (LunarCrush)" />
          <FeatureGroup name="FX-Specific" count={3}
            examples="Interest rate differential, carry score, days to central bank meeting" />
        </div>
      </Section>

      {/* Walk-Forward Validation */}
      <Section title="Walk-Forward Validation">
        <div className="text-sm text-gray-400 leading-relaxed space-y-3">
          <p>
            All models are trained using <span className="text-gray-200">chronological walk-forward validation</span> to
            prevent look-ahead bias — the most common source of inflated backtesting results.
          </p>
          <div className="bg-[#0a0e17] rounded-lg p-4 font-mono text-xs text-gray-500 leading-6 overflow-x-auto">
            <div>{'Data: |████████████████████████████████████████████|'}</div>
            <div>{' '}</div>
            <div>{'Fold 1: |■■■■■■■ TRAIN ■■■■■■■|▒▒ TEST ▒▒|'}</div>
            <div>{'Fold 2:    |■■■■■■■ TRAIN ■■■■■■■|▒▒ TEST ▒▒|'}</div>
            <div>{'Fold 3:       |■■■■■■■ TRAIN ■■■■■■■|▒▒ TEST ▒▒|'}</div>
            <div>{'  ...              ──── slides forward ────►'}</div>
          </div>
          <ul className="list-disc list-inside space-y-1 text-gray-500">
            <li><span className="text-gray-300">60% train / 30-day test</span> windows, stepping forward by one test window each fold</li>
            <li>Normalization (z-score) computed on the train set only, then applied to the test set</li>
            <li>Out-of-fold predictions are collected across all folds to train the stacking meta-learner</li>
            <li>Final production models are trained on the last train window before deployment</li>
          </ul>
        </div>
      </Section>

      {/* Signal Truth */}
      <Section title="Signal Truth & Outcome Resolution">
        <div className="text-sm text-gray-400 leading-relaxed space-y-3">
          <p>
            Every signal generated is recorded in a <span className="text-gray-200">signal_history</span> table
            with full model probabilities. On the next cycle, the previous signal is resolved against
            the actual price movement:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-[#0a0e17] rounded-lg p-3 border border-green-500/20">
              <div className="text-green-400 font-medium text-xs mb-1">BUY correct if</div>
              <div className="text-gray-300 text-sm">Price went up</div>
            </div>
            <div className="bg-[#0a0e17] rounded-lg p-3 border border-red-500/20">
              <div className="text-red-400 font-medium text-xs mb-1">SELL correct if</div>
              <div className="text-gray-300 text-sm">Price went down</div>
            </div>
            <div className="bg-[#0a0e17] rounded-lg p-3 border border-amber-500/20">
              <div className="text-amber-400 font-medium text-xs mb-1">HOLD correct if</div>
              <div className="text-gray-300 text-sm">Price moved &lt;1%</div>
            </div>
          </div>
          <p className="text-gray-500">
            Results are shown on the <span className="text-cyan-400">Signal Truth</span> tab with
            rolling accuracy (today, this week, all time), breakdowns by signal type and asset class,
            and every individual signal with its outcome.
          </p>
        </div>
      </Section>

      {/* Tech Stack */}
      <Section title="Tech Stack">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <TechCard label="Backend" value="Rust" detail="Axum web framework, tokio async runtime" />
          <TechCard label="ML Framework" value="Candle" detail="Hugging Face candle-core/candle-nn 0.9 for LSTM/GRU" />
          <TechCard label="Database" value="SQLite" detail="rusqlite — prices, signals, holdings, history" />
          <TechCard label="Frontend" value="React 19" detail="Vite + TailwindCSS 4 + Recharts" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
          <TechCard label="Data Sources" value="Yahoo Finance" detail="Stocks, FX, market indicators (7-year history)" />
          <TechCard label="Crypto Data" value="CoinGecko" detail="Top 15 coins, on-chain via LunarCrush" />
          <TechCard label="Parallelism" value="Rayon" detail="GBT split search across 104 features" />
          <TechCard label="Scheduling" value="tokio cron" detail="Hourly signals with market-hours awareness" />
        </div>
      </Section>

      {/* Disclaimer */}
      <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 text-sm text-amber-400/80 leading-relaxed">
        <span className="font-medium text-amber-400">Disclaimer:</span> This is an experimental
        research project. Past model accuracy does not guarantee future performance. Signals are
        not financial advice. Always do your own research before making investment decisions.
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[#111827] rounded-xl border border-[#1f2937] p-5">
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      {children}
    </div>
  )
}

function ModelCard({ name, tag, accuracy, description, strength, highlight }: {
  name: string; tag: string; accuracy: string; description: string; strength: string; highlight?: boolean
}) {
  return (
    <div className={`bg-[#0a0e17] rounded-lg p-4 border ${highlight ? 'border-cyan-500/30' : 'border-[#1f2937]'}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-white">{name}</span>
        <span className={`text-xs px-2 py-0.5 rounded-full ${
          highlight ? 'bg-cyan-500/15 text-cyan-400' : 'bg-white/5 text-gray-400'
        }`}>{tag}</span>
      </div>
      <div className={`text-xl font-bold mb-2 ${highlight ? 'text-cyan-400' : 'text-gray-300'}`}>{accuracy}</div>
      <p className="text-xs text-gray-500 leading-relaxed mb-2">{description}</p>
      <div className="text-xs text-gray-600">Key strength: <span className="text-gray-400">{strength}</span></div>
    </div>
  )
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-xs font-bold mt-0.5">
        {n}
      </div>
      <div>
        <div className="text-gray-200 font-medium mb-1">{title}</div>
        <div className="text-gray-500">{children}</div>
      </div>
    </div>
  )
}

function FeatureGroup({ name, count, examples }: { name: string; count: number; examples: string }) {
  return (
    <div className="bg-[#0a0e17] rounded-lg px-3 py-2 border border-[#1f2937]">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-gray-300">{name}</span>
        <span className="text-xs text-gray-600">{count}</span>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed">{examples}</p>
    </div>
  )
}

function TechCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="bg-[#0a0e17] rounded-lg px-3 py-2 border border-[#1f2937]">
      <div className="text-xs text-gray-600 mb-0.5">{label}</div>
      <div className="text-sm font-medium text-gray-200">{value}</div>
      <div className="text-xs text-gray-600 mt-0.5">{detail}</div>
    </div>
  )
}
