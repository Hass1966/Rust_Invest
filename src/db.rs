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
                ON stock_quotes(symbol, timestamp);

            CREATE TABLE IF NOT EXISTS market_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                price       REAL NOT NULL,
                volume      REAL,
                timestamp   TEXT NOT NULL,
                UNIQUE(symbol, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_market_history
                ON market_history(symbol, timestamp);

            CREATE TABLE IF NOT EXISTS fx_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                price       REAL NOT NULL,
                volume      REAL,
                timestamp   TEXT NOT NULL,
                UNIQUE(symbol, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_fx_history
                ON fx_history(symbol, timestamp);

            CREATE TABLE IF NOT EXISTS signal_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                asset           TEXT NOT NULL,
                asset_class     TEXT NOT NULL,
                signal          TEXT NOT NULL,
                confidence      REAL,
                probability_up  REAL,
                model_agreement TEXT,
                rsi             REAL,
                trend           TEXT,
                price           REAL,
                model_version   INTEGER NOT NULL,
                quality         TEXT,
                reason          TEXT,
                suggested_action TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_asset_time
                ON signal_snapshots(asset, timestamp);

            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version INTEGER NOT NULL,
                asset TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                total_return REAL,
                buy_hold_return REAL,
                excess_return REAL,
                annualised_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                win_rate REAL,
                profit_factor REAL,
                expectancy REAL,
                days_in_market INTEGER,
                total_days INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(model_version, asset)
            );

            CREATE TABLE IF NOT EXISTS portfolio_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                starting_capital REAL NOT NULL,
                final_value REAL,
                total_return REAL,
                annualised_return REAL,
                benchmark_return REAL,
                excess_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                n_assets INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(model_version, strategy)
            );

            CREATE TABLE IF NOT EXISTS portfolio_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                asset TEXT NOT NULL,
                weight REAL,
                allocated_amount REAL,
                asset_return REAL,
                contribution REAL,
                sharpe REAL,
                UNIQUE(model_version, strategy, asset)
            );

            CREATE TABLE IF NOT EXISTS daily_portfolio (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL UNIQUE,
                seed_value      REAL NOT NULL,
                portfolio_value REAL NOT NULL,
                daily_return    REAL NOT NULL,
                cumulative_return REAL NOT NULL,
                signals_json    TEXT,
                model_version   INTEGER NOT NULL,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_daily_portfolio_date
                ON daily_portfolio(date);

            CREATE TABLE IF NOT EXISTS earnings_dates (
                symbol        TEXT NOT NULL,
                earnings_date TEXT NOT NULL,
                PRIMARY KEY (symbol, earnings_date)
            );

            CREATE TABLE IF NOT EXISTS fear_greed (
                date  TEXT PRIMARY KEY,
                value REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                asset           TEXT NOT NULL,
                asset_class     TEXT NOT NULL,
                signal_type     TEXT NOT NULL,
                price_at_signal REAL NOT NULL,
                confidence      REAL NOT NULL,
                linreg_prob     REAL,
                logreg_prob     REAL,
                gbt_prob        REAL,
                outcome_price   REAL,
                pct_change      REAL,
                was_correct     INTEGER,
                resolution_ts   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_signal_history_asset
                ON signal_history(asset, timestamp);
            CREATE INDEX IF NOT EXISTS idx_signal_history_pending
                ON signal_history(resolution_ts);
            CREATE INDEX IF NOT EXISTS idx_signal_history_ts
                ON signal_history(timestamp);

            CREATE TABLE IF NOT EXISTS user_holdings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                quantity    REAL NOT NULL,
                start_date  TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                asset           TEXT NOT NULL,
                signal          TEXT NOT NULL,
                confidence      REAL NOT NULL,
                price_at_prediction REAL NOT NULL,
                actual_direction TEXT,
                was_correct     INTEGER,
                price_at_outcome REAL,
                outcome_timestamp TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_asset_time
                ON predictions(asset, timestamp);
            CREATE INDEX IF NOT EXISTS idx_predictions_pending
                ON predictions(outcome_timestamp);"
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

    // ── Market indicator history (VIX, treasuries, sector ETFs) ──

    pub fn insert_market_history(
        &self,
        symbol: &str,
        price: f64,
        volume: Option<f64>,
        timestamp: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO market_history
                (symbol, price, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![symbol, price, volume, timestamp],
        )?;
        Ok(())
    }

    pub fn count_market_history(&self, symbol: &str) -> Result<i64> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM market_history WHERE symbol = ?1",
            params![symbol],
            |row| row.get(0),
        )
    }

    pub fn get_market_history(&self, symbol: &str) -> Result<Vec<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price
             FROM market_history
             WHERE symbol = ?1
             ORDER BY timestamp ASC"
        )?;

        let points = stmt.query_map(params![symbol], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(points)
    }

    /// Get market history as just price vector (for feature building)
    pub fn get_market_prices(&self, symbol: &str) -> Result<Vec<f64>> {
        let mut stmt = self.conn.prepare(
            "SELECT price FROM market_history
             WHERE symbol = ?1
             ORDER BY timestamp ASC"
        )?;

        let prices = stmt.query_map(params![symbol], |row| {
            row.get::<_, f64>(0)
        })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(prices)
    }

    // ── FX currency pair history ──

    pub fn insert_fx_history(
        &self,
        symbol: &str,
        price: f64,
        volume: Option<f64>,
        timestamp: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO fx_history
                (symbol, price, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![symbol, price, volume, timestamp],
        )?;
        Ok(())
    }

    pub fn count_fx_history(&self, symbol: &str) -> Result<i64> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM fx_history WHERE symbol = ?1",
            params![symbol],
            |row| row.get(0),
        )
    }

    pub fn get_fx_history(&self, symbol: &str) -> Result<Vec<crate::analysis::PricePoint>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price, volume
             FROM fx_history
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

    // ── Signal snapshots ──

    pub fn insert_signal_snapshot(
        &self,
        signal: &crate::enriched_signals::EnrichedSignal,
        model_version: u32,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO signal_snapshots
                (timestamp, asset, asset_class, signal, confidence,
                 probability_up, model_agreement, rsi, trend, price,
                 model_version, quality, reason, suggested_action)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                signal.timestamp,
                signal.asset,
                signal.asset_class,
                signal.signal,
                signal.technical.confidence,
                signal.technical.probability_up,
                signal.technical.model_agreement,
                signal.technical.rsi,
                signal.technical.trend,
                signal.price,
                model_version,
                signal.technical.quality,
                signal.reason,
                signal.suggested_action,
            ],
        )?;
        Ok(())
    }

    /// Get the most recent signal for each asset within the last N days
    /// Used for the signal history heatmap
    pub fn get_recent_signals_all_assets(&self, days: usize) -> Result<Vec<SignalSnapshotRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, asset, asset_class, signal, confidence,
                    probability_up, model_agreement, rsi, trend, price,
                    quality, reason, suggested_action
             FROM signal_snapshots
             WHERE timestamp >= datetime('now', ?1)
             ORDER BY asset, timestamp DESC"
        )?;
        let days_param = format!("-{} days", days);
        let rows = stmt.query_map(params![days_param], |row| {
            Ok(SignalSnapshotRow {
                timestamp: row.get(0)?,
                asset: row.get(1)?,
                asset_class: row.get(2)?,
                signal: row.get(3)?,
                confidence: row.get(4)?,
                probability_up: row.get(5)?,
                model_agreement: row.get(6)?,
                rsi: row.get(7)?,
                trend: row.get(8)?,
                price: row.get(9)?,
                quality: row.get(10)?,
                reason: row.get(11)?,
                suggested_action: row.get(12)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn get_signal_history(&self, asset: &str, limit: usize) -> Result<Vec<SignalSnapshotRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, asset, asset_class, signal, confidence,
                    probability_up, model_agreement, rsi, trend, price,
                    quality, reason, suggested_action
             FROM signal_snapshots
             WHERE asset = ?1
             ORDER BY timestamp DESC
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![asset, limit as i64], |row| {
            Ok(SignalSnapshotRow {
                timestamp: row.get(0)?,
                asset: row.get(1)?,
                asset_class: row.get(2)?,
                signal: row.get(3)?,
                confidence: row.get(4)?,
                probability_up: row.get(5)?,
                model_agreement: row.get(6)?,
                rsi: row.get(7)?,
                trend: row.get(8)?,
                price: row.get(9)?,
                quality: row.get(10)?,
                reason: row.get(11)?,
                suggested_action: row.get(12)?,
            })
        })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(rows)
    }

    // ── Backtest & portfolio persistence ──

    pub fn insert_backtest_result(
        &self,
        model_version: u32,
        asset: &str,
        asset_class: &str,
        r: &crate::backtester::BacktestResult,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO backtest_results
                (model_version, asset, asset_class, total_return, buy_hold_return,
                 excess_return, annualised_return, sharpe_ratio, max_drawdown,
                 volatility, win_rate, profit_factor, expectancy,
                 days_in_market, total_days)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                model_version, asset, asset_class,
                r.total_return_pct, r.benchmark_return_pct,
                r.excess_return_pct, r.annualised_return_pct,
                r.sharpe_ratio, r.max_drawdown_pct, r.volatility_pct,
                r.win_rate, r.profit_factor, r.expectancy,
                r.days_in_market as i64, r.total_days as i64,
            ],
        )?;
        Ok(())
    }

    pub fn insert_portfolio_result(
        &self,
        model_version: u32,
        strategy: &str,
        starting_capital: f64,
        r: &crate::portfolio::PortfolioResult,
    ) -> Result<()> {
        let final_value = starting_capital * (1.0 + r.total_return_pct / 100.0);
        self.conn.execute(
            "INSERT OR REPLACE INTO portfolio_results
                (model_version, strategy, starting_capital, final_value,
                 total_return, annualised_return, benchmark_return, excess_return,
                 sharpe_ratio, max_drawdown, volatility, n_assets)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                model_version, strategy, starting_capital, final_value,
                r.total_return_pct, r.annualised_return_pct,
                r.benchmark_return_pct, r.excess_return_pct,
                r.sharpe_ratio, r.max_drawdown_pct, r.volatility_pct,
                r.n_assets as i64,
            ],
        )?;

        // Insert allocations
        for a in &r.allocations {
            self.conn.execute(
                "INSERT OR REPLACE INTO portfolio_allocations
                    (model_version, strategy, asset, weight, allocated_amount,
                     asset_return, contribution, sharpe)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    model_version, strategy, a.symbol,
                    a.weight, a.capital, a.asset_return, a.contribution, a.sharpe,
                ],
            )?;
        }
        Ok(())
    }

    pub fn get_backtest_results(&self, model_version: u32) -> Result<Vec<BacktestRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT asset, asset_class, total_return, buy_hold_return, excess_return,
                    annualised_return, sharpe_ratio, max_drawdown, volatility,
                    win_rate, profit_factor, expectancy, days_in_market, total_days
             FROM backtest_results
             WHERE model_version = ?1
             ORDER BY asset"
        )?;
        let rows = stmt.query_map(params![model_version], |row| {
            Ok(BacktestRow {
                asset: row.get(0)?,
                asset_class: row.get(1)?,
                total_return: row.get(2)?,
                buy_hold_return: row.get(3)?,
                excess_return: row.get(4)?,
                annualised_return: row.get(5)?,
                sharpe_ratio: row.get(6)?,
                max_drawdown: row.get(7)?,
                volatility: row.get(8)?,
                win_rate: row.get(9)?,
                profit_factor: row.get(10)?,
                expectancy: row.get(11)?,
                days_in_market: row.get(12)?,
                total_days: row.get(13)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn get_portfolio_results(&self, model_version: u32) -> Result<Vec<PortfolioRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT strategy, starting_capital, final_value, total_return,
                    annualised_return, benchmark_return, excess_return,
                    sharpe_ratio, max_drawdown, volatility, n_assets
             FROM portfolio_results
             WHERE model_version = ?1
             ORDER BY strategy"
        )?;
        let rows = stmt.query_map(params![model_version], |row| {
            Ok(PortfolioRow {
                strategy: row.get(0)?,
                starting_capital: row.get(1)?,
                final_value: row.get(2)?,
                total_return: row.get(3)?,
                annualised_return: row.get(4)?,
                benchmark_return: row.get(5)?,
                excess_return: row.get(6)?,
                sharpe_ratio: row.get(7)?,
                max_drawdown: row.get(8)?,
                volatility: row.get(9)?,
                n_assets: row.get(10)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn get_portfolio_allocations(&self, model_version: u32, strategy: &str) -> Result<Vec<AllocationRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT asset, weight, allocated_amount, asset_return, contribution, sharpe
             FROM portfolio_allocations
             WHERE model_version = ?1 AND strategy = ?2
             ORDER BY contribution DESC"
        )?;
        let rows = stmt.query_map(params![model_version, strategy], |row| {
            Ok(AllocationRow {
                asset: row.get(0)?,
                weight: row.get(1)?,
                allocated_amount: row.get(2)?,
                asset_return: row.get(3)?,
                contribution: row.get(4)?,
                sharpe: row.get(5)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn has_backtest_data(&self, model_version: u32) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM backtest_results WHERE model_version = ?1",
            params![model_version],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    // ── Daily portfolio tracker ──

    pub fn upsert_daily_portfolio(&self, row: &DailyPortfolioRow) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO daily_portfolio
                (date, seed_value, portfolio_value, daily_return,
                 cumulative_return, signals_json, model_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                row.date, row.seed_value, row.portfolio_value,
                row.daily_return, row.cumulative_return,
                row.signals_json, row.model_version,
            ],
        )?;
        Ok(())
    }

    pub fn get_daily_portfolio(&self, limit: usize) -> Result<Vec<DailyPortfolioRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, seed_value, portfolio_value, daily_return,
                    cumulative_return, signals_json, model_version
             FROM daily_portfolio
             ORDER BY date DESC
             LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(DailyPortfolioRow {
                date: row.get(0)?,
                seed_value: row.get(1)?,
                portfolio_value: row.get(2)?,
                daily_return: row.get(3)?,
                cumulative_return: row.get(4)?,
                signals_json: row.get(5)?,
                model_version: row.get(6)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn get_latest_daily_portfolio(&self) -> Result<Option<DailyPortfolioRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, seed_value, portfolio_value, daily_return,
                    cumulative_return, signals_json, model_version
             FROM daily_portfolio
             ORDER BY date DESC
             LIMIT 1"
        )?;
        let mut rows = stmt.query_map([], |row| {
            Ok(DailyPortfolioRow {
                date: row.get(0)?,
                seed_value: row.get(1)?,
                portfolio_value: row.get(2)?,
                daily_return: row.get(3)?,
                cumulative_return: row.get(4)?,
                signals_json: row.get(5)?,
                model_version: row.get(6)?,
            })
        })?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    pub fn get_backtest_seed_value(&self, model_version: u32) -> Result<Option<f64>> {
        let result = self.conn.query_row(
            "SELECT final_value FROM portfolio_results
             WHERE model_version = ?1 AND strategy = 'sharpe'
             LIMIT 1",
            params![model_version],
            |row| row.get::<_, f64>(0),
        );
        match result {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn has_daily_portfolio_for_date(&self, date: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM daily_portfolio WHERE date = ?1",
            params![date],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    // ── Live price upserts (INSERT OR REPLACE for latest price) ──

    pub fn upsert_stock_price(&self, symbol: &str, price: f64, volume: Option<f64>, timestamp: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO stock_history (symbol, price, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![symbol, price, volume, timestamp],
        )?;
        Ok(())
    }

    pub fn upsert_fx_price(&self, symbol: &str, price: f64, volume: Option<f64>, timestamp: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO fx_history (symbol, price, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![symbol, price, volume, timestamp],
        )?;
        Ok(())
    }

    pub fn upsert_crypto_price(&self, coin_id: &str, price: f64, volume: Option<f64>, timestamp: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO crypto_history (coin_id, price_usd, volume, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![coin_id, price, volume, timestamp],
        )?;
        Ok(())
    }

    pub fn upsert_market_price(&self, symbol: &str, price: f64, timestamp: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO market_history (symbol, price, volume, timestamp)
             VALUES (?1, ?2, NULL, ?3)",
            params![symbol, price, timestamp],
        )?;
        Ok(())
    }

    // ── Earnings dates ──

    pub fn insert_earnings_date(&self, symbol: &str, earnings_date: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO earnings_dates (symbol, earnings_date) VALUES (?1, ?2)",
            params![symbol, earnings_date],
        )?;
        Ok(())
    }

    pub fn get_earnings_dates(&self, symbol: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT earnings_date FROM earnings_dates WHERE symbol = ?1 ORDER BY earnings_date ASC"
        )?;
        let dates = stmt.query_map(params![symbol], |row| {
            row.get::<_, String>(0)
        })?.filter_map(|r| r.ok()).collect();
        Ok(dates)
    }

    // ── Fear & Greed index ──

    pub fn insert_fear_greed(&self, date: &str, value: f64) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO fear_greed (date, value) VALUES (?1, ?2)",
            params![date, value],
        )?;
        Ok(())
    }

    pub fn get_fear_greed_history(&self) -> Result<Vec<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, value FROM fear_greed ORDER BY date ASC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    /// Execute a raw SQL statement and return the number of rows affected
    pub fn execute_raw(&self, sql: &str) -> Result<usize> {
        Ok(self.conn.execute(sql, [])?)
    }

    // ── User holdings (personal portfolio tracker) ──

    pub fn insert_user_holding(&self, symbol: &str, quantity: f64, start_date: &str, asset_class: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO user_holdings (symbol, quantity, start_date, asset_class)
             VALUES (?1, ?2, ?3, ?4)",
            params![symbol, quantity, start_date, asset_class],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_user_holdings(&self) -> Result<Vec<UserHolding>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, symbol, quantity, start_date, asset_class, created_at
             FROM user_holdings ORDER BY created_at ASC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(UserHolding {
                id: row.get(0)?,
                symbol: row.get(1)?,
                quantity: row.get(2)?,
                start_date: row.get(3)?,
                asset_class: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn update_user_holding(&self, id: i64, quantity: f64, start_date: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE user_holdings SET quantity = ?1, start_date = ?2 WHERE id = ?3",
            params![quantity, start_date, id],
        )?;
        Ok(())
    }

    pub fn delete_user_holding(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM user_holdings WHERE id = ?1", params![id])?;
        Ok(())
    }

    // ── User-scoped holdings (with user_id) ──

    pub fn get_user_holdings_for(&self, user_id: i64) -> Result<Vec<UserHolding>> {
        // If user_id is 0, return all (backwards compat for unauthenticated)
        if user_id == 0 {
            return self.get_user_holdings();
        }
        let mut stmt = self.conn.prepare(
            "SELECT id, symbol, quantity, start_date, asset_class, created_at
             FROM user_holdings WHERE user_id = ?1 ORDER BY created_at ASC"
        )?;
        let rows = stmt.query_map(params![user_id], |row| {
            Ok(UserHolding {
                id: row.get(0)?,
                symbol: row.get(1)?,
                quantity: row.get(2)?,
                start_date: row.get(3)?,
                asset_class: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    pub fn insert_user_holding_for(&self, user_id: i64, symbol: &str, quantity: f64, start_date: &str, asset_class: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO user_holdings (symbol, quantity, start_date, asset_class, user_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![symbol, quantity, start_date, asset_class, user_id],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get the price at or before a given date from stock_history
    pub fn get_stock_price_at_date(&self, symbol: &str, date: &str) -> Result<Option<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price FROM stock_history
             WHERE symbol = ?1 AND timestamp <= ?2
             ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![symbol, date], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Get the price at or before a given date from fx_history
    pub fn get_fx_price_at_date(&self, symbol: &str, date: &str) -> Result<Option<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price FROM fx_history
             WHERE symbol = ?1 AND timestamp <= ?2
             ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![symbol, date], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Get the price at or before a given date from crypto_history
    pub fn get_crypto_price_at_date(&self, coin_id: &str, date: &str) -> Result<Option<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, price_usd FROM crypto_history
             WHERE coin_id = ?1 AND timestamp <= ?2
             ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![coin_id, date], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Get the latest price for any asset class
    pub fn get_latest_stock_price(&self, symbol: &str) -> Result<Option<f64>> {
        let mut stmt = self.conn.prepare(
            "SELECT price FROM stock_history WHERE symbol = ?1 ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![symbol], |row| row.get::<_, f64>(0))?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    pub fn get_latest_fx_price(&self, symbol: &str) -> Result<Option<f64>> {
        let mut stmt = self.conn.prepare(
            "SELECT price FROM fx_history WHERE symbol = ?1 ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![symbol], |row| row.get::<_, f64>(0))?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    pub fn get_latest_crypto_price(&self, coin_id: &str) -> Result<Option<f64>> {
        let mut stmt = self.conn.prepare(
            "SELECT price_usd FROM crypto_history WHERE coin_id = ?1 ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![coin_id], |row| row.get::<_, f64>(0))?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Get signal history for an asset from a specific date onwards
    pub fn get_signal_history_for_asset_from(&self, asset: &str, from_date: &str) -> Result<Vec<SignalHistoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, asset, asset_class, signal_type, price_at_signal,
                    confidence, linreg_prob, logreg_prob, gbt_prob,
                    outcome_price, pct_change, was_correct, resolution_ts
             FROM signal_history
             WHERE asset = ?1 AND timestamp >= ?2
             ORDER BY timestamp ASC"
        )?;
        let rows = stmt.query_map(params![asset, from_date], |row| {
            Ok(SignalHistoryRow {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                asset: row.get(2)?,
                asset_class: row.get(3)?,
                signal_type: row.get(4)?,
                price_at_signal: row.get(5)?,
                confidence: row.get(6)?,
                linreg_prob: row.get(7)?,
                logreg_prob: row.get(8)?,
                gbt_prob: row.get(9)?,
                outcome_price: row.get(10)?,
                pct_change: row.get(11)?,
                was_correct: row.get::<_, Option<i32>>(12)?.map(|v| v != 0),
                resolution_ts: row.get(13)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    /// Get the earliest signal_history timestamp for a given asset
    pub fn get_signal_tracking_start(&self, asset: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT MIN(timestamp) FROM signal_history WHERE asset = ?1"
        )?;
        let mut rows = stmt.query_map(params![asset], |row| row.get::<_, Option<String>>(0))?;
        Ok(rows.next().and_then(|r| r.ok()).flatten())
    }

    // ── Signal history tracking (every signal, every cycle) ──

    pub fn insert_signal_history(
        &self,
        timestamp: &str,
        asset: &str,
        asset_class: &str,
        signal_type: &str,
        price_at_signal: f64,
        confidence: f64,
        linreg_prob: f64,
        logreg_prob: f64,
        gbt_prob: f64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO signal_history
                (timestamp, asset, asset_class, signal_type, price_at_signal,
                 confidence, linreg_prob, logreg_prob, gbt_prob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![timestamp, asset, asset_class, signal_type, price_at_signal,
                    confidence, linreg_prob, logreg_prob, gbt_prob],
        )?;
        Ok(())
    }

    /// Get the most recent unresolved signal for a given asset
    pub fn get_last_unresolved_signal(&self, asset: &str) -> Result<Option<SignalHistoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, asset, asset_class, signal_type, price_at_signal,
                    confidence, linreg_prob, logreg_prob, gbt_prob,
                    outcome_price, pct_change, was_correct, resolution_ts
             FROM signal_history
             WHERE asset = ?1 AND resolution_ts IS NULL
             ORDER BY timestamp DESC LIMIT 1"
        )?;
        let mut rows = stmt.query_map(params![asset], |row| {
            Ok(SignalHistoryRow {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                asset: row.get(2)?,
                asset_class: row.get(3)?,
                signal_type: row.get(4)?,
                price_at_signal: row.get(5)?,
                confidence: row.get(6)?,
                linreg_prob: row.get(7)?,
                logreg_prob: row.get(8)?,
                gbt_prob: row.get(9)?,
                outcome_price: row.get(10)?,
                pct_change: row.get(11)?,
                was_correct: row.get::<_, Option<i32>>(12)?.map(|v| v != 0),
                resolution_ts: row.get(13)?,
            })
        })?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Resolve a signal_history entry with the outcome
    pub fn resolve_signal_history(
        &self,
        id: i64,
        outcome_price: f64,
        pct_change: f64,
        was_correct: bool,
        resolution_ts: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE signal_history
             SET outcome_price = ?1, pct_change = ?2, was_correct = ?3, resolution_ts = ?4
             WHERE id = ?5",
            params![outcome_price, pct_change, was_correct as i32, resolution_ts, id],
        )?;
        Ok(())
    }

    /// Get ALL unresolved signals (for batch resolution)
    pub fn get_all_unresolved_signals(&self) -> Result<Vec<SignalHistoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, asset, asset_class, signal_type, price_at_signal,
                    confidence, linreg_prob, logreg_prob, gbt_prob,
                    outcome_price, pct_change, was_correct, resolution_ts
             FROM signal_history
             WHERE resolution_ts IS NULL
             ORDER BY timestamp ASC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(SignalHistoryRow {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                asset: row.get(2)?,
                asset_class: row.get(3)?,
                signal_type: row.get(4)?,
                price_at_signal: row.get(5)?,
                confidence: row.get(6)?,
                linreg_prob: row.get(7)?,
                logreg_prob: row.get(8)?,
                gbt_prob: row.get(9)?,
                outcome_price: row.get(10)?,
                pct_change: row.get(11)?,
                was_correct: row.get::<_, Option<i32>>(12)?.map(|v| v != 0),
                resolution_ts: row.get(13)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    /// Get signal history for the truth page with optional limit
    pub fn get_signal_history_all(&self, limit: usize) -> Result<Vec<SignalHistoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, asset, asset_class, signal_type, price_at_signal,
                    confidence, linreg_prob, logreg_prob, gbt_prob,
                    outcome_price, pct_change, was_correct, resolution_ts
             FROM signal_history
             ORDER BY timestamp DESC
             LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(SignalHistoryRow {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                asset: row.get(2)?,
                asset_class: row.get(3)?,
                signal_type: row.get(4)?,
                price_at_signal: row.get(5)?,
                confidence: row.get(6)?,
                linreg_prob: row.get(7)?,
                logreg_prob: row.get(8)?,
                gbt_prob: row.get(9)?,
                outcome_price: row.get(10)?,
                pct_change: row.get(11)?,
                was_correct: row.get::<_, Option<i32>>(12)?.map(|v| v != 0),
                resolution_ts: row.get(13)?,
            })
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    // ── Predictions tracking ──

    pub fn insert_prediction(&self, timestamp: &str, asset: &str, signal: &str, confidence: f64, price_at_prediction: f64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO predictions (timestamp, asset, signal, confidence, price_at_prediction)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![timestamp, asset, signal, confidence, price_at_prediction],
        )?;
        Ok(())
    }

    pub fn get_pending_predictions(&self) -> Result<Vec<PendingPrediction>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, asset, signal, price_at_prediction
             FROM predictions WHERE outcome_timestamp IS NULL"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(PendingPrediction {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                asset: row.get(2)?,
                signal: row.get(3)?,
                price_at_prediction: row.get(4)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    pub fn update_prediction_outcome(&self, id: i64, actual_direction: &str, was_correct: bool, price_at_outcome: f64, outcome_timestamp: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE predictions SET actual_direction = ?1, was_correct = ?2, price_at_outcome = ?3, outcome_timestamp = ?4 WHERE id = ?5",
            rusqlite::params![actual_direction, was_correct as i32, price_at_outcome, outcome_timestamp, id],
        )?;
        Ok(())
    }

    pub fn get_predictions_history(&self, limit: usize) -> Result<Vec<PredictionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, asset, signal, confidence, price_at_prediction,
                    actual_direction, was_correct, price_at_outcome, outcome_timestamp
             FROM predictions ORDER BY timestamp DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |row| {
            Ok(PredictionRecord {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                asset: row.get(2)?,
                signal: row.get(3)?,
                confidence: row.get(4)?,
                price_at_prediction: row.get(5)?,
                actual_direction: row.get(6)?,
                was_correct: row.get::<_, Option<i32>>(7)?.map(|v| v != 0),
                price_at_outcome: row.get(8)?,
                outcome_timestamp: row.get(9)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BacktestRow {
    pub asset: String,
    pub asset_class: String,
    pub total_return: f64,
    pub buy_hold_return: f64,
    pub excess_return: f64,
    pub annualised_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub days_in_market: i64,
    pub total_days: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PortfolioRow {
    pub strategy: String,
    pub starting_capital: f64,
    pub final_value: f64,
    pub total_return: f64,
    pub annualised_return: f64,
    pub benchmark_return: f64,
    pub excess_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub n_assets: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AllocationRow {
    pub asset: String,
    pub weight: f64,
    pub allocated_amount: f64,
    pub asset_return: f64,
    pub contribution: f64,
    pub sharpe: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SignalSnapshotRow {
    pub timestamp: String,
    pub asset: String,
    pub asset_class: String,
    pub signal: String,
    pub confidence: Option<f64>,
    pub probability_up: Option<f64>,
    pub model_agreement: Option<String>,
    pub rsi: Option<f64>,
    pub trend: Option<String>,
    pub price: Option<f64>,
    pub quality: Option<String>,
    pub reason: Option<String>,
    pub suggested_action: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DailyPortfolioRow {
    pub date: String,
    pub seed_value: f64,
    pub portfolio_value: f64,
    pub daily_return: f64,
    pub cumulative_return: f64,
    pub signals_json: Option<String>,
    pub model_version: i64,
}

#[derive(Debug, Clone)]
pub struct PendingPrediction {
    pub id: i64,
    pub timestamp: String,
    pub asset: String,
    pub signal: String,
    pub price_at_prediction: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PredictionRecord {
    pub id: i64,
    pub timestamp: String,
    pub asset: String,
    pub signal: String,
    pub confidence: f64,
    pub price_at_prediction: f64,
    pub actual_direction: Option<String>,
    pub was_correct: Option<bool>,
    pub price_at_outcome: Option<f64>,
    pub outcome_timestamp: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UserHolding {
    pub id: i64,
    pub symbol: String,
    pub quantity: f64,
    pub start_date: String,
    pub asset_class: String,
    pub created_at: Option<String>,
}

impl Database {
    /// Get users with email alerts enabled
    pub fn get_alert_users(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, email FROM users WHERE email_alerts = 1 AND is_active = 1"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?.filter_map(|r| r.ok()).collect();
        Ok(rows)
    }

    /// Get user's last signal hash
    pub fn get_user_signal_hash(&self, user_id: i64) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT last_signal_hash FROM users WHERE id = ?1"
        )?;
        let result = stmt.query_row(params![user_id], |row| {
            row.get::<_, Option<String>>(0)
        }).ok().flatten();
        Ok(result)
    }

    /// Set user's last signal hash
    pub fn set_user_signal_hash(&self, user_id: i64, hash: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE users SET last_signal_hash = ?1 WHERE id = ?2",
            params![hash, user_id],
        )?;
        Ok(())
    }

    /// Disable email alerts for user by email
    pub fn disable_email_alerts_by_email(&self, email: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE users SET email_alerts = 0 WHERE email = ?1",
            params![email],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SignalHistoryRow {
    pub id: i64,
    pub timestamp: String,
    pub asset: String,
    pub asset_class: String,
    pub signal_type: String,
    pub price_at_signal: f64,
    pub confidence: f64,
    pub linreg_prob: Option<f64>,
    pub logreg_prob: Option<f64>,
    pub gbt_prob: Option<f64>,
    pub outcome_price: Option<f64>,
    pub pct_change: Option<f64>,
    pub was_correct: Option<bool>,
    pub resolution_ts: Option<String>,
}