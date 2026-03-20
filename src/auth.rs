/// Authentication & Authorization Module
/// ======================================
/// JWT-based auth with bcrypt password hashing.
/// Users table, registration, login, token verification.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use chrono::{Utc, Duration};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

// ════════════════════════════════════════
// Config
// ════════════════════════════════════════

const BCRYPT_COST: u32 = 12;
const JWT_EXPIRY_HOURS: i64 = 24;
const RATE_LIMIT_WINDOW_SECS: i64 = 60;
const RATE_LIMIT_MAX_ATTEMPTS: usize = 5;

// ════════════════════════════════════════
// Types
// ════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct User {
    pub id: i64,
    pub email: String,
    pub created_at: String,
    pub last_login: Option<String>,
    pub is_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: i64,        // user_id
    pub email: String,
    pub exp: usize,      // expiry timestamp
    pub iat: usize,      // issued at
}

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub token: String,
    pub user: User,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ════════════════════════════════════════
// Rate Limiter
// ════════════════════════════════════════

#[derive(Clone)]
pub struct RateLimiter {
    attempts: Arc<RwLock<HashMap<String, Vec<i64>>>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self { attempts: Arc::new(RwLock::new(HashMap::new())) }
    }

    pub async fn check(&self, key: &str) -> bool {
        let now = Utc::now().timestamp();
        let cutoff = now - RATE_LIMIT_WINDOW_SECS;

        let mut map = self.attempts.write().await;
        let entry = map.entry(key.to_string()).or_default();
        entry.retain(|&t| t > cutoff);

        if entry.len() >= RATE_LIMIT_MAX_ATTEMPTS {
            return false; // rate limited
        }

        entry.push(now);
        true
    }
}

// ════════════════════════════════════════
// JWT Secret
// ════════════════════════════════════════

pub fn get_jwt_secret() -> String {
    std::env::var("JWT_SECRET").unwrap_or_else(|_| {
        // Generate a stable secret from a fixed seed for this instance
        "rust-invest-jwt-secret-change-in-production-2026".to_string()
    })
}

// ════════════════════════════════════════
// Database: Users table
// ════════════════════════════════════════

pub fn create_auth_tables(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            last_login    TEXT,
            is_active     INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

        -- Add user_id column to user_holdings if it doesn't exist
        -- SQLite doesn't support ALTER TABLE ADD COLUMN IF NOT EXISTS,
        -- so we use a pragma check
        "
    )?;

    // Check if user_id column exists on user_holdings
    let has_user_id: bool = conn
        .prepare("PRAGMA table_info(user_holdings)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|name| name == "user_id");

    if !has_user_id {
        // Add column with default 0 (will be migrated to admin user)
        conn.execute("ALTER TABLE user_holdings ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0", [])?;
    }

    Ok(())
}

/// Create admin user and migrate existing holdings
pub fn ensure_admin_user(conn: &Connection, email: &str, password: &str) -> rusqlite::Result<i64> {
    // Check if admin exists
    let existing: Option<i64> = conn
        .query_row("SELECT id FROM users WHERE email = ?1", params![email], |row| row.get(0))
        .ok();

    if let Some(id) = existing {
        return Ok(id);
    }

    // Create admin user
    let hash = bcrypt::hash(password, BCRYPT_COST).unwrap_or_default();
    conn.execute(
        "INSERT INTO users (email, password_hash, is_active) VALUES (?1, ?2, 1)",
        params![email, hash],
    )?;
    let admin_id = conn.last_insert_rowid();

    // Migrate existing holdings (user_id = 0) to admin
    conn.execute(
        "UPDATE user_holdings SET user_id = ?1 WHERE user_id = 0",
        params![admin_id],
    )?;

    Ok(admin_id)
}

// ════════════════════════════════════════
// Auth operations
// ════════════════════════════════════════

pub fn register(conn: &Connection, email: &str, password: &str) -> Result<AuthResponse, String> {
    if email.is_empty() || !email.contains('@') {
        return Err("Invalid email address".to_string());
    }
    if password.len() < 8 {
        return Err("Password must be at least 8 characters".to_string());
    }

    // Check if email already exists
    let exists: bool = conn
        .query_row("SELECT COUNT(*) FROM users WHERE email = ?1", params![email], |row| row.get::<_, i64>(0))
        .unwrap_or(0) > 0;
    if exists {
        return Err("Email already registered".to_string());
    }

    let hash = bcrypt::hash(password, BCRYPT_COST).map_err(|e| format!("Hash error: {}", e))?;
    conn.execute(
        "INSERT INTO users (email, password_hash, is_active) VALUES (?1, ?2, 1)",
        params![email, hash],
    ).map_err(|e| format!("DB error: {}", e))?;

    let user_id = conn.last_insert_rowid();
    let user = User {
        id: user_id,
        email: email.to_string(),
        created_at: Utc::now().to_rfc3339(),
        last_login: Some(Utc::now().to_rfc3339()),
        is_active: true,
    };

    let token = generate_token(user_id, email)?;
    Ok(AuthResponse { token, user })
}

pub fn login(conn: &Connection, email: &str, password: &str) -> Result<AuthResponse, String> {
    let (user_id, hash, is_active): (i64, String, bool) = conn
        .query_row(
            "SELECT id, password_hash, is_active FROM users WHERE email = ?1",
            params![email],
            |row| Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? == 1)),
        )
        .map_err(|_| "Invalid email or password".to_string())?;

    if !is_active {
        return Err("Account is deactivated".to_string());
    }

    let valid = bcrypt::verify(password, &hash).unwrap_or(false);
    if !valid {
        return Err("Invalid email or password".to_string());
    }

    // Update last_login
    let now = Utc::now().to_rfc3339();
    let _ = conn.execute("UPDATE users SET last_login = ?1 WHERE id = ?2", params![now, user_id]);

    let user = User {
        id: user_id,
        email: email.to_string(),
        created_at: conn.query_row("SELECT created_at FROM users WHERE id = ?1", params![user_id], |row| row.get(0)).unwrap_or_default(),
        last_login: Some(now),
        is_active: true,
    };

    let token = generate_token(user_id, email)?;
    Ok(AuthResponse { token, user })
}

pub fn generate_token(user_id: i64, email: &str) -> Result<String, String> {
    let secret = get_jwt_secret();
    let now = Utc::now();
    let exp = now + Duration::hours(JWT_EXPIRY_HOURS);

    let claims = Claims {
        sub: user_id,
        email: email.to_string(),
        exp: exp.timestamp() as usize,
        iat: now.timestamp() as usize,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    ).map_err(|e| format!("Token error: {}", e))
}

pub fn verify_token(token: &str) -> Result<Claims, String> {
    let secret = get_jwt_secret();
    let validation = Validation::default();

    decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .map(|data| data.claims)
    .map_err(|e| format!("Invalid token: {}", e))
}

/// Extract token from Authorization header
pub fn extract_token(auth_header: Option<&str>) -> Result<String, String> {
    let header = auth_header.ok_or("Missing Authorization header")?;
    if header.starts_with("Bearer ") {
        Ok(header[7..].to_string())
    } else {
        Err("Invalid Authorization header format".to_string())
    }
}
