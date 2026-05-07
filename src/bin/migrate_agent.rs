/// One-time migration: SQLite agent data → PostgreSQL
use rust_invest::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Migrating agent data from SQLite to PostgreSQL...");

    let database = db::Database::new("rust_invest.db")?;
    database.set_wal_mode();

    let pool = pg::create_pool()?;

    pg::migrate_agent_data_to_pg(&pool, &database).await?;

    println!("Migration complete!");
    Ok(())
}
