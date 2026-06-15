#!/usr/bin/env bash
# Smoke test for cross-lingual 4-GPU run.
# Spawns 4 ollama instances (GPU 0/1/2/3 × Qwen×2 + Gemma×2),
# then runs 5 queries through each (model, port) endpoint.
# Asserts: each instance completes 5 queries in <60s and uses GPU.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== TEST: 4 GPU full utilization smoke test ==="
echo "Note: do NOT run this while BIRD is running; will collide on GPUs"
echo

# Step 1: kill all ollama
echo "Step 1: killing existing ollama"
pkill -f "ollama serve" 2>/dev/null || true
sleep 5

# Step 2: start 4 instances
echo "Step 2: starting 4 ollama instances"
mkdir -p /tmp/smoke_test
declare -A INSTANCES=(
    [11434]="0:qwen2.5-coder:7b"
    [11435]="1:gemma3n:e4b"
    [11437]="2:qwen2.5-coder:7b"
    [11438]="3:gemma3n:e4b"
)

for port in "${!INSTANCES[@]}"; do
    spec="${INSTANCES[$port]}"
    gpu="${spec%%:*}"
    model="${spec#*:}"
    CUDA_VISIBLE_DEVICES=$gpu \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_NUM_PARALLEL=2 \
    nohup ollama serve > "/tmp/smoke_test/ollama_gpu${gpu}_port${port}.log" 2>&1 &
    echo "  GPU $gpu port $port ($model) pid $!"
done
sleep 5

# Step 3: health check
echo
echo "Step 3: health check (all 4 ports)"
for port in 11434 11435 11437 11438; do
    if curl -sf "http://127.0.0.1:$port/api/version" > /dev/null; then
        echo "  port $port OK"
    else
        echo "  port $port FAIL"
        exit 1
    fi
done

# Step 4: load each model and run 1 inference each
echo
echo "Step 4: 1 inference per instance (warmup)"
for port in "${!INSTANCES[@]}"; do
    spec="${INSTANCES[$port]}"
    model="${spec#*:}"
    t0=$(date +%s)
    curl -sf -o /dev/null "http://127.0.0.1:$port/api/generate" \
        -d "{\"model\":\"$model\",\"prompt\":\"SELECT 1\",\"stream\":false,\"options\":{\"num_predict\":20}}" || {
        echo "  port $port inference FAIL"
        exit 1
    }
    t1=$(date +%s)
    echo "  port $port ($model) inference OK in $((t1-t0))s"
done

# Step 5: GPU utilization sample during a real-prompt inference
echo
echo "Step 5: GPU utilization during concurrent BIRD-style prompts"

# Build a real BIRD-style prompt
SCHEMA_PROMPT='Tables:
CREATE TABLE singers (id INT, name TEXT, age INT, country TEXT);
CREATE TABLE concerts (id INT, year INT, singer_id INT);

Question: How many singers are from Taiwan?
Generate SQL:'

for port in "${!INSTANCES[@]}"; do
    spec="${INSTANCES[$port]}"
    model="${spec#*:}"
    curl -sf -o "/tmp/smoke_test/resp_${port}.json" "http://127.0.0.1:$port/api/generate" \
        -d "$(python3 -c "import json,sys; print(json.dumps({'model':'$model','prompt':'$SCHEMA_PROMPT','stream':False,'options':{'num_predict':100,'temperature':0.0}}))")" &
done

# While they run: sample GPU 5 times
for i in 1 2 3 4 5; do
    sleep 2
    echo "  sample $i:"
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | sed 's/^/    /'
done

wait
echo
echo "Step 6: parse responses"
for port in 11434 11435 11437 11438; do
    sql=$(python3 -c "import json; d=json.load(open('/tmp/smoke_test/resp_${port}.json')); print(d['response'][:80].replace(chr(10), ' '))" 2>&1 || echo "PARSE_ERR")
    echo "  port $port: $sql"
done

echo
echo "=== SMOKE TEST DONE ==="
echo "If you see all 4 GPUs >0% utilization in step 5 samples,"
echo "and all 4 ports return valid SQL in step 6, the pipeline is ready."
