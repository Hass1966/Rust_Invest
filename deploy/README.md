# Alpha Signal — Deployment Guide

## First Time Setup

1. Provision a Hetzner server (Ubuntu 24.04, CX21 or higher)
2. SSH into the server: `ssh root@YOUR_IP`
3. Run the setup script:

```bash
curl -sSL https://raw.githubusercontent.com/Hass1966/Rust_Invest/main/deploy/setup-hetzner.sh | bash
```

Or manually:
```bash
git clone https://github.com/Hass1966/Rust_Invest.git
cd Rust_Invest/deploy
bash setup-hetzner.sh
```

## Updating

SSH into the server and run:
```bash
cd ~/Rust_Invest/deploy
bash update.sh
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PORT` | Server port (default: 8081) | No |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | Yes |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | Yes |
| `GOOGLE_REDIRECT_URI` | Google OAuth callback URL | Yes |
| `MICROSOFT_CLIENT_ID` | Microsoft OAuth client ID | Yes |
| `MICROSOFT_CLIENT_SECRET` | Microsoft OAuth client secret | Yes |
| `MICROSOFT_REDIRECT_URI` | Microsoft OAuth callback URL | Yes |
| `NEWSAPI_KEY` | NewsAPI.org API key | Yes |
| `JWT_SECRET` | JWT signing secret | Yes |
| `LLM_PROVIDER` | LLM provider (anthropic/ollama) | No |
| `LLM_API_KEY` | LLM API key | No |

## DNS Records

| Type | Name | Value |
|------|------|-------|
| A | app.alphasignal.co.uk | YOUR_HETZNER_IP |
| CNAME | alphasignal.co.uk | AWS Amplify domain (for marketing site) |

## OAuth Setup

### Google
1. Go to https://console.cloud.google.com/apis/credentials
2. Create OAuth 2.0 Client ID (Web application)
3. Add authorized redirect URI: `https://app.alphasignal.co.uk/api/v1/auth/google/callback`
4. Copy Client ID and Client Secret

### Microsoft
1. Go to https://portal.azure.com → App registrations
2. Register new application
3. Add redirect URI: `https://app.alphasignal.co.uk/api/v1/auth/microsoft/callback`
4. Create client secret under Certificates & secrets
5. Copy Application (client) ID and client secret value

## Checking Logs

```bash
# Service logs (last 50 lines)
sudo journalctl -u rustinvest -n 50

# Follow logs in real time
sudo journalctl -u rustinvest -f

# Service status
sudo systemctl status rustinvest
```

## Database Backup

```bash
# Backup
cp ~/Rust_Invest/rust_invest.db ~/backups/rust_invest_$(date +%Y%m%d).db

# Restore
cp ~/backups/rust_invest_YYYYMMDD.db ~/Rust_Invest/rust_invest.db
sudo systemctl restart rustinvest
```

## Architecture

```
Internet → Nginx (port 443, SSL) → Rust Invest (port 8081)
                                     ├── Frontend (SPA from frontend/dist/)
                                     ├── REST API (/api/v1/...)
                                     └── SQLite (rust_invest.db)
```
