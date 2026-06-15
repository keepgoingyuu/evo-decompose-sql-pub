#!/usr/bin/env bash
# Start 3 Ollama instances, each pinned to one V100.
# GPU 0: qwen2.5-coder:7b      port 11434
# GPU 1: gemma4:26b              port 11435
# GPU 2: Arctic-Text2SQL-R1 (TBD) port 11436
#
# Models load lazily on first request, so this script only starts daemons.
# Stop with: pkill -f "ollama serve"
set -euo pipefail

# Kill existing daemons
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# Logs
mkdir -p /tmp/ollama_multi
LOG_DIR=/tmp/ollama_multi

start_instance() {
  local gpu_id=$1
  local port=$2
  local label=$3
  CUDA_VISIBLE_DEVICES=$gpu_id \
  OLLAMA_HOST=127.0.0.1:$port \
  OLLAMA_MODELS=$HOME/.ollama/models \
  nohup ollama serve > "$LOG_DIR/${label}_gpu${gpu_id}_port${port}.log" 2>&1 &
  echo "  started ollama serve  GPU $gpu_id  port $port  ($label)  pid $!"
}

echo "Starting 3 Ollama instances (one per GPU)..."
start_instance 0 11434 qwen
start_instance 1 11435 gemma4
start_instance 2 11436 arctic
sleep 3

echo
echo "=== Health checks ==="
for port in 11434 11435 11436; do
  if curl -sS "http://127.0.0.1:$port/api/version" 2>&1; then
    echo "  port $port: OK"
  else
    echo "  port $port: FAIL"
  fi
done

echo
echo "=== Process state ==="
ps -ef | grep "ollama serve" | grep -v grep | awk '{print $2, $8, $9, $10}'
