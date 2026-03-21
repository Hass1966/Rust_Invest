#!/bin/bash
# Install nginx configs — run with sudo
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Install rustinvest config
cp "$SCRIPT_DIR/rustinvest" /etc/nginx/sites-available/rustinvest
ln -sf /etc/nginx/sites-available/rustinvest /etc/nginx/sites-enabled/rustinvest
echo "✓ Installed rustinvest nginx config (port 8082 → 8081)"

# 2. Fix finopsmind config (proxy to 8080 not 8081)
cp "$SCRIPT_DIR/finopsmind-fixed" /etc/nginx/sites-available/finopsmind
echo "✓ Fixed finopsmind nginx config (port 3000 → 8080)"

# 3. Test and reload
nginx -t && nginx -s reload
echo "✓ Nginx reloaded successfully"

# 4. Quick verification
echo ""
echo "Verifying services..."
curl -sf http://localhost:8081/api/v1/signals/current | head -c 100 && echo " ← Rust Invest direct (8081) OK" || echo "⚠ Rust Invest (8081) not responding"
curl -sf http://localhost:8080/api/health | head -c 100 && echo " ← FinOpsMind direct (8080) OK" || echo "⚠ FinOpsMind (8080) not responding"
