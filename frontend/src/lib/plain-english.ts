interface PhraseMatch {
  patterns: RegExp[]
  plain: string
}

const PHRASE_MAP: PhraseMatch[] = [
  {
    patterns: [/lower bollinger band/i, /potentially oversold/i],
    plain: 'Price has dropped to an unusually low level recently',
  },
  {
    patterns: [/upper bollinger band/i],
    plain: 'Price has climbed to an unusually high level recently',
  },
  {
    patterns: [/overbought/i, /RSI in overbought/i],
    plain: 'Price has risen quickly and may be due a breather',
  },
  {
    patterns: [/oversold territory/i],
    plain: 'Price has fallen sharply and may bounce back',
  },
  {
    patterns: [/SMA crossover negative/i, /trend is bearish/i],
    plain: 'The medium-term trend is pointing downward',
  },
  {
    patterns: [/SMA crossover positive/i, /trend is bullish/i],
    plain: 'The medium-term trend is pointing upward',
  },
  {
    patterns: [/strong model consensus/i],
    plain: 'All our models agree on this signal',
  },
  {
    patterns: [/volatility rising sharply/i],
    plain: 'The price has been moving more erratically than usual',
  },
  {
    patterns: [/volatility contracting/i],
    plain: 'The price has been unusually stable recently',
  },
  {
    patterns: [/volatility rising(?! sharply)/i],
    plain: 'Price movements have been getting a bit choppier',
  },
  {
    patterns: [/momentum weakening/i],
    plain: 'The recent price momentum is slowing down',
  },
  {
    patterns: [/momentum improving/i],
    plain: 'Price momentum is picking up',
  },
  {
    patterns: [/models disagree/i],
    plain: 'Our models are giving mixed readings',
  },
]

export function translateSignalSummary(summary: string, _signal: string, _asset: string): string {
  const matched: string[] = []

  for (const entry of PHRASE_MAP) {
    for (const pattern of entry.patterns) {
      if (pattern.test(summary)) {
        if (!matched.includes(entry.plain)) {
          matched.push(entry.plain)
        }
        break
      }
    }
  }

  if (matched.length === 0) return summary

  if (matched.length === 1) return matched[0] + '.'
  return matched[0] + ', and ' + matched.slice(1).join('. Also, ').toLowerCase() + '.'
}

export function confidenceLabel(confidence: number): { text: string; color: string } {
  if (confidence > 15) return { text: 'High confidence', color: 'text-emerald-400' }
  if (confidence >= 8) return { text: 'Moderate confidence', color: 'text-amber-400' }
  return { text: 'Low confidence', color: 'text-gray-500' }
}
