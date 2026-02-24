use rusqlite::{Connection, Result, params};
use crate::models::CoinData;

pub struct Database {
    conn: Connection,
}

impl Database {
    // Open (or create) the database file
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Database { conn };
        db.create_tables()?;
        Ok(db)
    }

    // Create tables if they don't exist
    fn create_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS crypto_prices (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id     TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                name        TEXT NOT NULL,
                price_usd   REAL NOT NULL,
                change_24h  REAL,
                market_cap  REAL,
                volume      REAL,
                high_24h    REAL,
                low_24h     REAL,
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crypto_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                coin_id     TEXT NOT NULL,
                price_usd   REAL NOT NULL,
                volume      REAL,
                timestamp   TEXT NOT NULL,
                UNIQUE(coin_id, timestamp)
            );

            CREATE TABLE IF NOT EXISTS stock_quotes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                name            TEXT NOT NULL,
                price           REAL NOT NULL,
                change_value    REAL,
                change_percent  TEXT,
                high            REAL,
                low             REAL,
                volume          TEXT,
                timestamp       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_crypto_coin
                ON crypto_prices(coin_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_history_coin
                ON crypto_history(coin_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_stock_symbol
                ON stock_quotes(symbol, timestamp);"
        )?;
        Ok(())
    }

    // Store a live crypto snapshot
    pub fn insert_crypto(&self, coin: &CoinData, timestamp: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO crypto_prices
                (coin_id, symbol, name, price_usd, change_24h,
                 market_cap, volume, high_24h, low_24h, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                coin.id,
                coin.symbol,
                coin.name,
                coin.current_price,
                coin.price_change_percentage_24h,
                coin.market_cap,
                coin.total_volume,
                coin.high_24h,
                coin.low_24h,
                timestamp,
            ],
        )?;
        Ok(())
    }

    // Store historical price point (skips duplicates)
    pub fn insert_history(
        &self,
        coin_id: &str,
        price: f64,
        volume: Option<f64>,
        timestamp: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO crypto_history
                (coin_id, price_usd, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![coin_id, price, volume, timestamp],
        )?;
        Ok(())
    }

    // Store a stock quote
    pub fn insert_stock(
        &self,
        symbol: &str,
        name: &str,
        price: f64,
        change: f64,
        change_pct: &str,
        high: f64,
        low: f64,
        volume: &str,
        timestamp: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO stock_quotes
                (symbol, name, price, change_value, change_percent,
                 high, low, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![symbol, name, price, change, change_pct,
                    high, low, volume, timestamp],
        )?;
        Ok(())
    }

    // Count records (useful for checking what we have)
    pub fn count_crypto_history(&self, coin_id: &str) -> Result<i64> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM crypto_history WHERE coin_id = ?1",
            params![coin_id],
            |row| row.get(0),
        )
    }

    pub fn count_all_history(&self) -> Result<i64> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM crypto_history",
            [],
            |row| row.get(0),
        )
    }
}