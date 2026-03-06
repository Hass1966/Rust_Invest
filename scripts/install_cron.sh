#!/bin/bash
# Install weekly retrain cron job for Sunday 02:00
(crontab -l 2>/dev/null; echo "0 2 * * 0 /home/hassan/projects/Rust_Invest/scripts/weekly_retrain.sh") | crontab -
echo "Cron job installed. Verify with: crontab -l"
