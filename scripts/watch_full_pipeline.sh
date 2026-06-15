#!/usr/bin/env bash
# Full v2.1 experiment pipeline watcher.
# Order:
#   1. wait for run_v21_bird_4gpu (BIRD) to finish
#   2. start run_v21_xlingual_4gpu (Spider EN + CSpider ZH-TW + MultiSpider JA)
#      -- Qwen + Gemma × 5 strategy × 3 langs × 4-GPU parallel
#   3. after cross-lingual finishes, kill ollama and start
#      run_v21_arctic_r1 (transformers, fp16, 4 benchmarks × S0)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG="logs/full_pipeline_watcher.log"
echo "[watcher] started at $(date)" | tee -a "$LOG"

wait_for() {
    local pattern=$1
    local label=$2
    while pgrep -f "$pattern" > /dev/null; do
        local pids
        pids=$(pgrep -f "$pattern" | tr '\n' ' ')
        echo "[watcher] $(date '+%H:%M:%S') $label still running (PIDs: $pids)" | tee -a "$LOG"
        sleep 60
    done
    echo "[watcher] $(date '+%H:%M:%S') $label finished." | tee -a "$LOG"
}

# Stage 1: wait for BIRD
wait_for "run_v21_bird_4gpu" "BIRD"

# Stage 2: cross-lingual (Spider + CSpider + MultiSpider JA)
echo "[watcher] $(date '+%H:%M:%S') Launching cross-lingual 4-GPU run..." | tee -a "$LOG"
nohup .venv/bin/python -u -m scripts.run_v21_xlingual_4gpu > logs/v21_xlingual.log 2>&1 &
XLINGUAL_PID=$!
echo "[watcher] cross-lingual PID=$XLINGUAL_PID launched" | tee -a "$LOG"
sleep 5
wait_for "run_v21_xlingual_4gpu" "cross-lingual"

# Stage 3: kill Ollama, free GPUs, then Arctic-R1
echo "[watcher] $(date '+%H:%M:%S') Stopping Ollama to free GPUs..." | tee -a "$LOG"
pkill -f "ollama serve" 2>/dev/null || true
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv | tee -a "$LOG"

echo "[watcher] $(date '+%H:%M:%S') Launching Arctic-R1 transformers run..." | tee -a "$LOG"
nohup .venv/bin/python -u -m scripts.run_v21_arctic_r1 > logs/v21_arctic_r1.log 2>&1 &
ARCTIC_PID=$!
echo "[watcher] Arctic-R1 PID=$ARCTIC_PID launched" | tee -a "$LOG"
wait_for "run_v21_arctic_r1" "Arctic-R1"

# Stage 4-5: seeds 43 and 44 (multi-seed for bootstrap CI per v2.1 plan)
# Note: requires updating SEED in scripts to support new seeds
echo "[watcher] $(date '+%H:%M:%S') Seed 42 pipeline complete." | tee -a "$LOG"
echo "[watcher] NOTE: seed 43 and 44 still required by v2.1 (3 seeds for bootstrap CI)." | tee -a "$LOG"
echo "[watcher] Seeds 43+44 not auto-launched here — manual restart needed after reviewing seed 42 results." | tee -a "$LOG"
echo "[watcher] All stages for SEED=42 submitted. Monitor logs/v21_arctic_r1.log" | tee -a "$LOG"
