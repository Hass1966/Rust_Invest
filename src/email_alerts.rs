/// Email Alerts — Daily signal change notifications
/// =================================================
/// Sends HTML emails to users when their portfolio signals change.

use crate::db::Database;
use crate::enriched_signals::EnrichedSignal;
use std::collections::HashMap;

pub struct EmailConfig {
    pub smtp_from: String,
    pub smtp_password: String,
    pub app_url: String,
}

impl EmailConfig {
    pub fn from_env() -> Option<Self> {
        let from = std::env::var("SMTP_FROM").ok()?;
        let password = std::env::var("SMTP_PASSWORD").ok()?;
        let app_url = std::env::var("APP_URL").unwrap_or_else(|_| "http://localhost:8081".to_string());
        if from.is_empty() || password.is_empty() || password == "placeholder" {
            return None;
        }
        Some(Self { smtp_from: from, smtp_password: password, app_url })
    }
}

#[derive(Debug)]
pub struct SignalChange {
    pub asset: String,
    pub previous: String,
    pub new: String,
    pub confidence: f64,
}

/// Check for signal changes and send email alerts
pub async fn send_daily_alerts(
    db: &Database,
    signals: &HashMap<String, EnrichedSignal>,
    config: &EmailConfig,
) {
    // Get all users with email_alerts enabled
    let users = match db.get_alert_users() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("  [Email] Failed to get alert users: {}", e);
            return;
        }
    };

    if users.is_empty() {
        println!("  [Email] No users with alerts enabled");
        return;
    }

    println!("  [Email] Checking {} users for signal changes", users.len());

    for (user_id, email) in &users {
        // Get user's holdings
        let holdings = match db.get_user_holdings_for(*user_id) {
            Ok(h) => h,
            Err(_) => continue,
        };

        if holdings.is_empty() {
            continue;
        }

        // Compute current signal hash for user's holdings
        let mut current_signals: Vec<(String, String, f64)> = Vec::new();
        for h in &holdings {
            if let Some(sig) = signals.get(&h.symbol) {
                current_signals.push((h.symbol.clone(), sig.signal.clone(), sig.technical.confidence));
            }
        }

        let current_hash = compute_signal_hash(&current_signals);

        // Check if hash changed
        let last_hash = db.get_user_signal_hash(*user_id).unwrap_or(None);
        if last_hash.as_deref() == Some(&current_hash) {
            continue; // No changes
        }

        // Determine what changed
        let changes = detect_changes(db, *user_id, &current_signals);
        if changes.is_empty() {
            // First time — just store hash
            let _ = db.set_user_signal_hash(*user_id, &current_hash);
            continue;
        }

        // Build and send email
        let first_name = email.split('@').next().unwrap_or("there");
        let html = build_email_html(first_name, &changes, &config.app_url);
        let subject = format!(
            "Alpha Signal — Your portfolio update {}",
            chrono::Utc::now().format("%d %b %Y")
        );

        match send_email(config, email, &subject, &html).await {
            Ok(_) => {
                println!("  [Email] Sent alert to {}", email);
                let _ = db.set_user_signal_hash(*user_id, &current_hash);
            }
            Err(e) => {
                eprintln!("  [Email] Failed to send to {}: {}", email, e);
            }
        }
    }
}

fn compute_signal_hash(signals: &[(String, String, f64)]) -> String {
    let mut sorted = signals.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let parts: Vec<String> = sorted.iter().map(|(s, sig, _)| format!("{}:{}", s, sig)).collect();
    parts.join(",")
}

fn detect_changes(
    db: &Database,
    user_id: i64,
    current: &[(String, String, f64)],
) -> Vec<SignalChange> {
    let last_hash = db.get_user_signal_hash(user_id).unwrap_or(None);
    let last_hash = match last_hash {
        Some(h) => h,
        None => return Vec::new(),
    };

    let previous: HashMap<String, String> = last_hash
        .split(',')
        .filter_map(|part| {
            let mut iter = part.splitn(2, ':');
            let sym = iter.next()?;
            let sig = iter.next()?;
            Some((sym.to_string(), sig.to_string()))
        })
        .collect();

    let mut changes = Vec::new();
    for (symbol, signal, confidence) in current {
        if let Some(prev) = previous.get(symbol) {
            if prev != signal {
                changes.push(SignalChange {
                    asset: symbol.clone(),
                    previous: prev.clone(),
                    new: signal.clone(),
                    confidence: *confidence,
                });
            }
        }
    }
    changes
}

fn build_email_html(first_name: &str, changes: &[SignalChange], app_url: &str) -> String {
    let mut rows = String::new();
    for c in changes {
        let prev_color = match c.previous.as_str() {
            "BUY" => "#10b981",
            "SELL" => "#ef4444",
            _ => "#f59e0b",
        };
        let new_color = match c.new.as_str() {
            "BUY" => "#10b981",
            "SELL" => "#ef4444",
            _ => "#f59e0b",
        };
        rows.push_str(&format!(
            r#"<tr>
                <td style="padding:8px 12px;border-bottom:1px solid #1f2937;color:#e5e7eb">{}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #1f2937;color:{}">{}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #1f2937;color:{}">{}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #1f2937;color:#9ca3af">{:.0}%</td>
            </tr>"#,
            c.asset, prev_color, c.previous, new_color, c.new, c.confidence * 10.0
        ));
    }

    format!(
        r#"<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0a0e17;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif">
<div style="max-width:600px;margin:0 auto;padding:20px">
    <div style="background:#111827;border-radius:12px;overflow:hidden;border:1px solid #1f2937">
        <div style="background:linear-gradient(135deg,#0891b2,#06b6d4);padding:20px 24px">
            <h1 style="margin:0;color:#fff;font-size:20px">Alpha Signal</h1>
        </div>
        <div style="padding:24px">
            <p style="color:#e5e7eb;margin:0 0 16px">Good morning {}</p>
            <p style="color:#9ca3af;margin:0 0 20px;font-size:14px">Your portfolio signals have changed:</p>
            <table style="width:100%;border-collapse:collapse;font-size:14px">
                <thead>
                    <tr style="border-bottom:2px solid #1f2937">
                        <th style="text-align:left;padding:8px 12px;color:#6b7280;font-weight:500">Asset</th>
                        <th style="text-align:left;padding:8px 12px;color:#6b7280;font-weight:500">Previous</th>
                        <th style="text-align:left;padding:8px 12px;color:#6b7280;font-weight:500">New</th>
                        <th style="text-align:left;padding:8px 12px;color:#6b7280;font-weight:500">Confidence</th>
                    </tr>
                </thead>
                <tbody>{}</tbody>
            </table>
            <p style="color:#6b7280;font-size:12px;margin:20px 0 0;font-style:italic">Not financial advice.</p>
            <div style="margin-top:20px">
                <a href="{}/my-portfolio" style="display:inline-block;background:#06b6d4;color:#000;padding:10px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px">
                    View your portfolio
                </a>
            </div>
            <div style="margin-top:16px">
                <a href="{}/api/v1/email/unsubscribe?email={}" style="color:#6b7280;font-size:11px;text-decoration:underline">
                    Unsubscribe from alerts
                </a>
            </div>
        </div>
    </div>
</div>
</body>
</html>"#,
        first_name, rows, app_url, app_url, first_name
    )
}

async fn send_email(
    config: &EmailConfig,
    to: &str,
    subject: &str,
    html_body: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use lettre::message::header::ContentType;
    use lettre::transport::smtp::authentication::Credentials;
    use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

    let email = Message::builder()
        .from(config.smtp_from.parse()?)
        .to(to.parse()?)
        .subject(subject)
        .header(ContentType::TEXT_HTML)
        .body(html_body.to_string())?;

    let creds = Credentials::new(config.smtp_from.clone(), config.smtp_password.clone());

    let mailer = AsyncSmtpTransport::<Tokio1Executor>::relay("smtp.gmail.com")?
        .credentials(creds)
        .build();

    mailer.send(email).await?;
    Ok(())
}
