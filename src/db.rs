use rusqlite::{Connection, Result, params};
use crate::models::CoinData;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Database { conn };
        db.create_tables()?;
        Ok(db)
    }

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

            CREATE TABLE IF NOT EXISTS stock_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                price       REAL NOT NULL,
                volume      REAL,
                timestamp   TEXT NOT NULL,
                UNIQUE(symbol, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_stock_history
                ON stock_history(symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_crypto_coin
                ON crypto_prices(coin_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_history_coin
                ON crypto_history(coin_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_stock_symbol
                ON stock_quotes(symbol, timestamp);"
        )?;
        Ok(())
    }

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

    pub fn insert_stock_history(
        &self,
        symbol: &str,
        price: f64,
        volume: Option<f64>,
        timestamp: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO stock_history
                (symbol, price, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![symbol, price, volume, timestamp],
        )?;
        Ok(())
    }

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

    pub fn count_stock_history(&self, symbol: &str) -> Result<i64> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM stock_history WHERE symbol = ?1",
            params![symbol],
            |row| row.get(0),
        )
    }

    pub fn get_coin_history(&self, coin_id: &str) -> Result<Vec<crate::analysis::PricePoint>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price_usd, volume
             FROM crypto_history
             WHERE coin_id = ?1
             ORDER BY timestamp ASC"
        )?;

        let points = stmt.query_map(params![coin_id], |row| {
            Ok(crate::analysis::PricePoint {
                timestamp: row.get(0)?,
                price: row.get(1)?,
                volume: row.get(2)?,
            })
        })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(points)
    }

    pub fn get_stock_history(&self, symbol: &str) -> Result<Vec<crate::analysis::PricePoint>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price, volume
             FROM stock_history
             WHERE symbol = ?1
             ORDER BY timestamp ASC"
        )?;

        let points = stmt.query_map(params![symbol], |row| {
            Ok(crate::analysis::PricePoint {
                timestamp: row.get(0)?,
                price: row.get(1)?,
                volume: row.get(2)?,
            })
        })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(points)
    }

    pub fn get_all_coin_ids(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT coin_id FROM crypto_history ORDER BY coin_id"
        )?;

        let ids = stmt.query_map([], |row| {
            row.get(0)
        })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(ids)
    }

    pub fn get_price_range(&self, coin_id: &str) -> Result<(String, String)> {
        self.conn.query_row(
            "SELECT MIN(timestamp), MAX(timestamp)
             FROM crypto_history
             WHERE coin_id = ?1",
            params![coin_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
    }
}