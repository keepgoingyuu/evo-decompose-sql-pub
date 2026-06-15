#!/usr/bin/env bash
# Start 4 Ollama instances, one per GPU, for V2.1 cross-lingual experiments.
# GPU 0 port 11434: qwen2.5-coder:7b  (Qwen instance #1)
# GPU 1 port 11435: gemma3n:e4b       (Gemma instance #1)
# GPU 2 port 11437: qwen2.5-coder:7b  (Qwen instance #2 — splits Qwen workload)
# GPU 3 port 11438: gemma3n:e4b       (Gemma instance #2 — splits Gemma workload)
#
# Each instance pre-loads its model lazily on first request.
# Stop with: pkill -f "ollama serve"
set -euo pipefail

pkill -f "ollama serve" 2>/dev/null || true
sleep 2

mkdir -p /tmp/ollama_4gpu
LOG_DIR=/tmp/ollama_4gpu

start_instance() {
  local gpu_id=$1
  local port=$2
  local label=$3
  CUDA_VISIBLE_DEVICES=$gpu_id \
  OLLAMA_HOST=127.0.0.1:$port \
  OLLAMA_MODELS=$HOME/.ollama/models \
  nohup ollama serve > "$LOG_DIR/${label}_gpu${gpu_id}_port${port}.log" 2>&1 &
  echo "  GPU $gpu_id  port $port  ($label)  pid $!"
}

echo "Starting 4 Ollama instances (Qwen×2 + Gemma×2)..."
start_instance 0 11434 qwen-A
start_instance 1 11435 gemma-A
start_instance 2 11437 qwen-B
start_instance 3 11438 gemma-B
sleep 4

echo
echo "=== Health checks ==="
for port in 11434 11435 11437 11438; do
  if curl -sS "http://127.0.0.1:$port/api/version" 2>&1; then
    echo "  port $port: OK"
  else
    echo "  port $port: FAIL"
  fi
done

echo
echo "=== Process state ==="
ps -ef | grep "ollama serve" | grep -v grep | awk '{print $2, $8, $9}'
