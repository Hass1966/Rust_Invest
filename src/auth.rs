/// Authentication & Authorization Module
/// ======================================
/// OAuth 2.0 (Google + Microsoft) + JWT session tokens.
/// Users table with optional password_hash (OAuth users have none).

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_provider: Option<String>,
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
// OAuth Config
// ════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct OAuthConfig {
    pub google_client_id: String,
    pub google_client_secret: String,
    pub google_redirect_uri: String,
    pub microsoft_client_id: String,
    pub microsoft_client_secret: String,
    pub microsoft_redirect_uri: String,
}

impl OAuthConfig {
    pub fn from_env() -> Option<Self> {
        let google_id = std::env::var("GOOGLE_CLIENT_ID").ok()?;
        if google_id == "placeholder" || google_id.is_empty() {
            return None;
        }
        Some(Self {
            google_client_id: google_id,
            google_client_secret: std::env::var("GOOGLE_CLIENT_SECRET").unwrap_or_default(),
            google_redirect_uri: std::env::var("GOOGLE_REDIRECT_URI")
                .unwrap_or_else(|_| "http://localhost:8081/api/v1/auth/google/callback".to_string()),
            microsoft_client_id: std::env::var("MICROSOFT_CLIENT_ID").unwrap_or_default(),
            microsoft_client_secret: std::env::var("MICROSOFT_CLIENT_SECRET").unwrap_or_default(),
            microsoft_redirect_uri: std::env::var("MICROSOFT_REDIRECT_URI")
                .unwrap_or_else(|_| "http://localhost:8081/api/v1/auth/microsoft/callback".to_string()),
        })
    }
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
            password_hash TEXT,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            last_login    TEXT,
            is_active     INTEGER NOT NULL DEFAULT 1,
            oauth_provider TEXT,
            oauth_id      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        "
    )?;

    // Migrate: add oauth columns if they don't exist
    let columns: Vec<String> = conn
        .prepare("PRAGMA table_info(users)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .collect();

    if !columns.contains(&"oauth_provider".to_string()) {
        let _ = conn.execute("ALTER TABLE users ADD COLUMN oauth_provider TEXT", []);
    }
    if !columns.contains(&"oauth_id".to_string()) {
        let _ = conn.execute("ALTER TABLE users ADD COLUMN oauth_id TEXT", []);
    }

    // Make password_hash nullable — SQLite doesn't enforce NOT NULL on ALTER,
    // but the original schema had NOT NULL. We just allow NULL inserts for OAuth users.

    // Check if user_id column exists on user_holdings
    let has_user_id: bool = conn
        .prepare("PRAGMA table_info(user_holdings)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|name| name == "user_id");

    if !has_user_id {
        conn.execute("ALTER TABLE user_holdings ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0", [])?;
    }

    Ok(())
}

/// Create admin user and migrate existing holdings
pub fn ensure_admin_user(conn: &Connection, email: &str, password: &str) -> rusqlite::Result<i64> {
    let existing: Option<i64> = conn
        .query_row("SELECT id FROM users WHERE email = ?1", params![email], |row| row.get(0))
        .ok();

    if let Some(id) = existing {
        return Ok(id);
    }

    let hash = bcrypt::hash(password, 12).unwrap_or_default();
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
// OAuth: Find or create user
// ════════════════════════════════════════

pub fn find_or_create_oauth_user(
    conn: &Connection,
    email: &str,
    provider: &str,
    oauth_id: &str,
) -> Result<AuthResponse, String> {
    // Try to find existing user by email
    let existing: Option<(i64, bool)> = conn
        .query_row(
            "SELECT id, is_active FROM users WHERE email = ?1",
            params![email],
            |row| Ok((row.get(0)?, row.get::<_, i64>(1)? == 1)),
        )
        .ok();

    let user_id = if let Some((id, is_active)) = existing {
        if !is_active {
            return Err("Account is deactivated".to_string());
        }
        // Update OAuth info and last_login
        let now = Utc::now().to_rfc3339();
        let _ = conn.execute(
            "UPDATE users SET oauth_provider = ?1, oauth_id = ?2, last_login = ?3 WHERE id = ?4",
            params![provider, oauth_id, now, id],
        );
        id
    } else {
        // Create new user (no password for OAuth users)
        conn.execute(
            "INSERT INTO users (email, password_hash, is_active, oauth_provider, oauth_id, last_login)
             VALUES (?1, NULL, 1, ?2, ?3, datetime('now'))",
            params![email, provider, oauth_id],
        ).map_err(|e| format!("DB error: {}", e))?;
        conn.last_insert_rowid()
    };

    let user = User {
        id: user_id,
        email: email.to_string(),
        created_at: conn.query_row("SELECT created_at FROM users WHERE id = ?1", params![user_id], |row| row.get(0)).unwrap_or_default(),
        last_login: Some(Utc::now().to_rfc3339()),
        is_active: true,
        oauth_provider: Some(provider.to_string()),
    };

    let token = generate_token(user_id, email)?;
    Ok(AuthResponse { token, user })
}

// ════════════════════════════════════════
// OAuth URL builders
// ════════════════════════════════════════

pub fn google_auth_url(config: &OAuthConfig) -> String {
    format!(
        "https://accounts.google.com/o/oauth2/v2/auth?client_id={}&redirect_uri={}&response_type=code&scope=openid%20email%20profile&access_type=offline&prompt=consent",
        urlencoding(&config.google_client_id),
        urlencoding(&config.google_redirect_uri),
    )
}

pub fn microsoft_auth_url(config: &OAuthConfig) -> String {
    format!(
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id={}&redirect_uri={}&response_type=code&scope=openid%20email%20profile&response_mode=query",
        urlencoding(&config.microsoft_client_id),
        urlencoding(&config.microsoft_redirect_uri),
    )
}

fn urlencoding(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

// ════════════════════════════════════════
// OAuth token exchange
// ════════════════════════════════════════

#[derive(Deserialize)]
pub struct OAuthTokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub id_token: Option<String>,
}

#[derive(Deserialize)]
pub struct GoogleUserInfo {
    pub sub: String,
    pub email: String,
}

#[derive(Deserialize)]
pub struct MicrosoftUserInfo {
    pub id: String,
    #[serde(alias = "mail", alias = "userPrincipalName")]
    pub email: Option<String>,
}

pub async fn exchange_google_code(
    client: &reqwest::Client,
    config: &OAuthConfig,
    code: &str,
) -> Result<(String, String), String> {
    // Exchange code for token
    let token_res = client
        .post("https://oauth2.googleapis.com/token")
        .form(&[
            ("code", code),
            ("client_id", &config.google_client_id),
            ("client_secret", &config.google_client_secret),
            ("redirect_uri", &config.google_redirect_uri),
            ("grant_type", "authorization_code"),
        ])
        .send()
        .await
        .map_err(|e| format!("Token exchange failed: {}", e))?;

    if !token_res.status().is_success() {
        let body = token_res.text().await.unwrap_or_default();
        return Err(format!("Google token error: {}", body));
    }

    let token_data: OAuthTokenResponse = token_res.json().await
        .map_err(|e| format!("Parse token: {}", e))?;

    // Fetch user info
    let user_res = client
        .get("https://www.googleapis.com/oauth2/v3/userinfo")
        .bearer_auth(&token_data.access_token)
        .send()
        .await
        .map_err(|e| format!("Userinfo failed: {}", e))?;

    let user_info: GoogleUserInfo = user_res.json().await
        .map_err(|e| format!("Parse userinfo: {}", e))?;

    Ok((user_info.email, user_info.sub))
}

pub async fn exchange_microsoft_code(
    client: &reqwest::Client,
    config: &OAuthConfig,
    code: &str,
) -> Result<(String, String), String> {
    // Exchange code for token
    let token_res = client
        .post("https://login.microsoftonline.com/common/oauth2/v2.0/token")
        .form(&[
            ("code", code),
            ("client_id", &config.microsoft_client_id),
            ("client_secret", &config.microsoft_client_secret),
            ("redirect_uri", &config.microsoft_redirect_uri),
            ("grant_type", "authorization_code"),
            ("scope", "openid email profile"),
        ])
        .send()
        .await
        .map_err(|e| format!("Token exchange failed: {}", e))?;

    if !token_res.status().is_success() {
        let body = token_res.text().await.unwrap_or_default();
        return Err(format!("Microsoft token error: {}", body));
    }

    let token_data: OAuthTokenResponse = token_res.json().await
        .map_err(|e| format!("Parse token: {}", e))?;

    // Fetch user info from Microsoft Graph
    let user_res = client
        .get("https://graph.microsoft.com/v1.0/me")
        .bearer_auth(&token_data.access_token)
        .send()
        .await
        .map_err(|e| format!("Graph API failed: {}", e))?;

    let user_info: MicrosoftUserInfo = user_res.json().await
        .map_err(|e| format!("Parse user: {}", e))?;

    let email = user_info.email.unwrap_or_default();
    if email.is_empty() {
        return Err("Microsoft account has no email".to_string());
    }

    Ok((email, user_info.id))
}

// ════════════════════════════════════════
// Legacy auth operations (kept for admin)
// ════════════════════════════════════════

pub fn register(conn: &Connection, email: &str, password: &str) -> Result<AuthResponse, String> {
    if email.is_empty() || !email.contains('@') {
        return Err("Invalid email address".to_string());
    }
    if password.len() < 8 {
        return Err("Password must be at least 8 characters".to_string());
    }

    let exists: bool = conn
        .query_row("SELECT COUNT(*) FROM users WHERE email = ?1", params![email], |row| row.get::<_, i64>(0))
        .unwrap_or(0) > 0;
    if exists {
        return Err("Email already registered".to_string());
    }

    let hash = bcrypt::hash(password, 12).map_err(|e| format!("Hash error: {}", e))?;
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
        oauth_provider: None,
    };

    let token = generate_token(user_id, email)?;
    Ok(AuthResponse { token, user })
}

pub fn login(conn: &Connection, email: &str, password: &str) -> Result<AuthResponse, String> {
    let row: (i64, Option<String>, bool) = conn
        .query_row(
            "SELECT id, password_hash, is_active FROM users WHERE email = ?1",
            params![email],
            |row| Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? == 1)),
        )
        .map_err(|_| "Invalid email or password".to_string())?;

    let (user_id, hash_opt, is_active) = row;

    if !is_active {
        return Err("Account is deactivated".to_string());
    }

    let hash = hash_opt.ok_or_else(|| "This account uses OAuth sign-in".to_string())?;
    let valid = bcrypt::verify(password, &hash).unwrap_or(false);
    if !valid {
        return Err("Invalid email or password".to_string());
    }

    let now = Utc::now().to_rfc3339();
    let _ = conn.execute("UPDATE users SET last_login = ?1 WHERE id = ?2", params![now, user_id]);

    let user = User {
        id: user_id,
        email: email.to_string(),
        created_at: conn.query_row("SELECT created_at FROM users WHERE id = ?1", params![user_id], |row| row.get(0)).unwrap_or_default(),
        last_login: Some(now),
        is_active: true,
        oauth_provider: None,
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
