#!/bin/bash
# ═══════════════════════════════════════════════
# Alpha Signal — Deploy Latest Code
# ═══════════════════════════════════════════════
# Run this on Hetzner to deploy latest code.
set -euo pipefail

echo "━━━ Updating Alpha Signal ━━━"

cd ~/Rust_Invest
git pull origin main

echo "  Building frontend..."
cd frontend && npm install && npm run build && cd ..

echo "  Building backend..."
source "$HOME/.cargo/env"
cargo build --release

echo "  Restarting service..."
sudo systemctl restart rustinvest

# Wait and verify
sleep 3
if curl -sf http://localhost:8081/health > /dev/null 2>&1; then
    echo ""
    echo "  Alpha Signal updated successfully"
else
    echo ""
    echo "  WARNING: Service may not have started cleanly"
    echo "  Check logs: sudo journalctl -u rustinvest -n 50"
fi
