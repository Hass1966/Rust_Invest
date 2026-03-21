#!/bin/bash
# ═══════════════════════════════════════════════
# Alpha Signal — Hetzner First-Time Setup
# ═══════════════════════════════════════════════
# Run this ONCE on a fresh Ubuntu 24.04 server.
# Usage: bash setup-hetzner.sh
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        ALPHA SIGNAL — Hetzner Server Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Gather parameters ──
read -rp "Hetzner IP (e.g. 65.21.xxx.xxx): " HETZNER_IP
read -rp "Domain (e.g. app.alphasignal.co.uk): " DOMAIN
read -rp "Google Client ID: " GOOGLE_CLIENT_ID
read -rp "Google Client Secret: " GOOGLE_CLIENT_SECRET
read -rp "Microsoft Client ID: " MICROSOFT_CLIENT_ID
read -rp "Microsoft Client Secret: " MICROSOFT_CLIENT_SECRET
read -rp "NewsAPI Key: " NEWSAPI_KEY
read -rp "JWT Secret (leave blank for random): " JWT_SECRET
read -rp "Admin Email [hassan@hassanshuman.co.uk]: " ADMIN_EMAIL
ADMIN_EMAIL="${ADMIN_EMAIL:-hassan@hassanshuman.co.uk}"

if [ -z "$JWT_SECRET" ]; then
    JWT_SECRET=$(openssl rand -hex 32)
    echo "  Generated JWT secret: ${JWT_SECRET:0:8}..."
fi

GOOGLE_REDIRECT_URI="https://${DOMAIN}/api/v1/auth/google/callback"
MICROSOFT_REDIRECT_URI="https://${DOMAIN}/api/v1/auth/microsoft/callback"

echo ""
echo "━━━ Installing dependencies ━━━"

# ── 2. Install dependencies ──
apt-get update -y
apt-get install -y curl git nginx certbot python3-certbot-nginx sqlite3 build-essential pkg-config libssl-dev

# Install Rust
if ! command -v rustup &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Install Node.js 20
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

echo ""
echo "━━━ Cloning repository ━━━"

# ── 3. Clone repo ──
cd "$HOME"
if [ -d "Rust_Invest" ]; then
    echo "  Rust_Invest already exists, pulling latest..."
    cd Rust_Invest && git pull origin main
else
    git clone https://github.com/Hass1966/Rust_Invest.git
    cd Rust_Invest
fi

echo ""
echo "━━━ Building frontend ━━━"

# ── 4. Build frontend ──
cd frontend
npm install
npm run build
cd ..

echo ""
echo "━━━ Building backend (this may take a while) ━━━"

# ── 5. Build backend ──
source "$HOME/.cargo/env"
cargo build --release

echo ""
echo "━━━ Creating systemd service ━━━"

# ── 6. Create systemd service ──
cat > /etc/systemd/system/rustinvest.service << SVCEOF
[Unit]
Description=Alpha Signal - AI Trading Signal Server
After=network.target

[Service]
WorkingDirectory=$HOME/Rust_Invest
ExecStart=$HOME/Rust_Invest/target/release/serve
Restart=always
RestartSec=10
Environment="PORT=8081"
Environment="GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}"
Environment="GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}"
Environment="GOOGLE_REDIRECT_URI=${GOOGLE_REDIRECT_URI}"
Environment="MICROSOFT_CLIENT_ID=${MICROSOFT_CLIENT_ID}"
Environment="MICROSOFT_CLIENT_SECRET=${MICROSOFT_CLIENT_SECRET}"
Environment="MICROSOFT_REDIRECT_URI=${MICROSOFT_REDIRECT_URI}"
Environment="NEWSAPI_KEY=${NEWSAPI_KEY}"
Environment="JWT_SECRET=${JWT_SECRET}"
Environment="SMTP_FROM=alerts@alphasignal.co.uk"
Environment="SMTP_PASSWORD=placeholder"
Environment="APP_URL=https://${DOMAIN}"

[Install]
WantedBy=multi-user.target
SVCEOF

echo ""
echo "━━━ Configuring nginx ━━━"

# ── 7. Create nginx config ──
cat > /etc/nginx/sites-available/rustinvest << NGXEOF
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
        proxy_read_timeout 180s;
    }
}
NGXEOF

ln -sf /etc/nginx/sites-available/rustinvest /etc/nginx/sites-enabled/rustinvest
rm -f /etc/nginx/sites-enabled/default
nginx -t

echo ""
echo "━━━ Starting services ━━━"

# ── 9. Start services ──
systemctl daemon-reload
systemctl enable rustinvest
systemctl start rustinvest
systemctl reload nginx

# Wait for service to start
sleep 3

echo ""
echo "━━━ Installing SSL certificate ━━━"

# ── 8. SSL with certbot ──
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$ADMIN_EMAIL" || {
    echo "  ⚠ Certbot failed — ensure DNS A record points to this server first"
    echo "  Run manually later: certbot --nginx -d ${DOMAIN}"
}

echo ""
echo "━━━ Running smoke tests ━━━"

# ── 10. Smoke tests ──
PASS=0
FAIL=0

if curl -sf "http://localhost:8081/api/v1/signals/current" | head -c 50 > /dev/null 2>&1; then
    echo "  PASS: /api/v1/signals/current"
    ((PASS++))
else
    echo "  FAIL: /api/v1/signals/current"
    ((FAIL++))
fi

if curl -sf "http://localhost:8081/api/v1/auth/google" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -q "302\|307"; then
    echo "  PASS: /api/v1/auth/google (redirect)"
    ((PASS++))
else
    echo "  FAIL: /api/v1/auth/google"
    ((FAIL++))
fi

if curl -sf "http://localhost:8081/health" > /dev/null 2>&1; then
    echo "  PASS: /health"
    ((PASS++))
else
    echo "  FAIL: /health"
    ((FAIL++))
fi

echo ""
echo "  Tests: ${PASS} passed, ${FAIL} failed"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Alpha Signal is live at https://${DOMAIN}                  "
echo "║  SSL certificate installed (if DNS was ready)               "
echo "║  Service running as systemd unit 'rustinvest'               "
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Next steps:                                                "
echo "║  1. Add DNS A record: ${DOMAIN} → ${HETZNER_IP}            "
echo "║  2. Add OAuth redirect URI in Google Console:               "
echo "║     https://${DOMAIN}/api/v1/auth/google/callback           "
echo "║  3. Add OAuth redirect URI in Microsoft Portal:             "
echo "║     https://${DOMAIN}/api/v1/auth/microsoft/callback        "
echo "║  4. Set NEWSAPI_KEY at newsapi.org (free tier)              "
echo "╚══════════════════════════════════════════════════════════════╝"
