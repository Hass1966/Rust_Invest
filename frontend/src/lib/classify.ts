// Definitive asset classification — single source of truth across the app.

const CRYPTO_IDS = new Set([
  // CoinGecko IDs (lowercase)
  'bitcoin', 'ethereum', 'solana', 'ripple', 'dogecoin', 'cardano',
  'tron', 'avalanche-2', 'chainlink', 'polkadot', 'near', 'sui',
  'aptos', 'arbitrum', 'the-open-network', 'uniswap', 'litecoin',
  'shiba-inu', 'stellar', 'matic-network',
  // Common short tickers (uppercase)
  'BTC', 'ETH', 'DOGE', 'ADA', 'XRP', 'SOL', 'TRX',
  'AVAX', 'LINK', 'DOT', 'NEAR', 'SUI', 'APT', 'ARB',
  'TON', 'UNI', 'LTC', 'SHIB', 'XLM', 'MATIC',
])

/**
 * Classify an asset symbol into its asset class.
 *   FX:     any symbol ending in =X
 *   Crypto: any CoinGecko ID or known short ticker
 *   Stock:  everything else
 */
export function classifyAsset(symbol: string): 'stock' | 'fx' | 'crypto' {
  if (symbol.endsWith('=X')) return 'fx'
  if (CRYPTO_IDS.has(symbol) || CRYPTO_IDS.has(symbol.toLowerCase())) return 'crypto'
  return 'stock'
}

/** Human-readable quantity label based on asset class */
export function quantityLabel(assetClass: string): string {
  switch (assetClass) {
    case 'crypto': return 'Units'
    case 'fx': return 'Notional £'
    default: return 'Shares'
  }
}
