#!/usr/bin/env bash
# Wait for v21_bird_4gpu to finish, then kill all Ollama daemons (free GPUs),
# then launch Arctic-R1 transformers run.
#
# Run: nohup bash scripts/watch_and_run_arctic.sh > logs/arctic_watcher.log 2>&1 &
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG="logs/arctic_watcher.log"
echo "[watcher] started at $(date)" | tee -a "$LOG"

# Wait until no run_v21_bird_4gpu Python remains
while pgrep -f run_v21_bird_4gpu > /dev/null; do
    PIDS=$(pgrep -f run_v21_bird_4gpu | tr '\n' ' ')
    echo "[watcher] $(date '+%H:%M:%S') BIRD still running (PIDs: $PIDS)" | tee -a "$LOG"
    sleep 60
done
echo "[watcher] $(date '+%H:%M:%S') BIRD finished. Stopping Ollama..." | tee -a "$LOG"

# Stop all Ollama daemons to free GPUs
pkill -f "ollama serve" 2>/dev/null || true
sleep 5

# Verify GPUs free
nvidia-smi --query-gpu=index,memory.used --format=csv | tee -a "$LOG"

echo "[watcher] $(date '+%H:%M:%S') Launching Arctic-R1 transformers run..." | tee -a "$LOG"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
nohup .venv/bin/python -u -m scripts.run_v21_arctic_r1 > logs/v21_arctic_r1.log 2>&1 &
ARCTIC_PID=$!
echo "[watcher] Arctic-R1 PID=$ARCTIC_PID launched" | tee -a "$LOG"
echo "[watcher] Tail logs/v21_arctic_r1.log to monitor" | tee -a "$LOG"
