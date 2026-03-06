#!/bin/bash
cd ~/projects/Rust_Invest
echo "[$(date)] Starting weekly retrain..." >> logs/retrain.log
./target/release/train >> logs/retrain.log 2>&1
echo "[$(date)] Retrain complete. Restarting serve..." >> logs/retrain.log
sudo systemctl restart rustinvest
echo "[$(date)] Done." >> logs/retrain.log
