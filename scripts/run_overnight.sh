#!/usr/bin/env bash
# Overnight LOCAL-only pipeline (no API calls).
# Run forever until killed; safely resumable.
#
# Stages:
#   1. wait for gemma4:26b ollama pull
#   2. unzip spider_data.zip (idempotent)
#   3. start 2 ollama instances (GPU 0 = qwen, GPU 1 = gemma4)
#   4. smoke test 2 local models
#   5. pilot 2 models × 5 strategies × 50 query × seed=42 (BIRD subset)
#   6. main BIRD 609 mod+chal × 5 strategies × seed=42 (Qwen first then Gemma 4)
#
# All output: ~/evo-decompose-sql/logs/latest/main.log
# All result JSONs: ~/evo-decompose-sql/results/q1_local_*.json
#
# Local-only -> $0 API spend tonight. API path runs tomorrow separately.

set -uo pipefail

cd "$HOME/evo-decompose-sql"
source .venv/bin/activate

LOG_ROOT="$HOME/evo-decompose-sql/logs/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
ln -sfn "$LOG_ROOT" "$HOME/evo-decompose-sql/logs/latest"

MAIN_LOG="$LOG_ROOT/main.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

stage_done() {
    touch "$LOG_ROOT/$1.done"
}

stage_skip_if_done() {
    if [[ -f "$LOG_ROOT/$1.done" ]]; then
        log "SKIP $1 (already done)"
        return 0
    fi
    return 1
}

log "===== OVERNIGHT LOCAL-ONLY RUN STARTED ====="
log "Log root: $LOG_ROOT"
log "Models: qwen25c7b (GPU 0), gemma4 (GPU 1) -- LOCAL ONLY, no API"

# ============================================================
# STAGE 1: wait for gemma4:26b
# ============================================================
log "===== STAGE 1: waiting for gemma4:26b ====="
attempt=0
until ollama list 2>&1 | grep -q "^gemma4:26b"; do
    attempt=$((attempt+1))
    if [[ $((attempt % 10)) -eq 0 ]]; then
        progress=$(grep -oE 'pulling [a-f0-9]+: *[0-9]+%' /tmp/ollama_pulls2.log 2>/dev/null | tail -1 || echo "?")
        log "  [attempt $attempt] gemma4:26b not yet ready ($progress)"
    fi
    sleep 30
done
log "gemma4:26b ready"
ollama list | tee -a "$MAIN_LOG"

# ============================================================
# STAGE 2: unzip Spider (idempotent)
# ============================================================
if ! stage_skip_if_done stage2; then
    log "===== STAGE 2: unzip spider_data.zip ====="
    cd data/spider/spider_data
    if [[ ! -d "spider_data" ]] && [[ ! -f "dev.json" ]]; then
        unzip -q spider_data.zip
    else
        log "  already extracted"
    fi
    ls | tee -a "$MAIN_LOG"
    cd "$HOME/evo-decompose-sql"
    stage_done stage2
fi

# ============================================================
# STAGE 3: start 2 ollama instances (GPU 0 + GPU 1)
# ============================================================
log "===== STAGE 3: starting 2 Ollama instances (GPU 0/1) ====="
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=$HOME/.ollama/models \
    nohup ollama serve > "$LOG_ROOT/ollama_qwen_gpu0.log" 2>&1 &
log "  started qwen daemon  GPU 0  port 11434  pid $!"

CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=$HOME/.ollama/models \
    nohup ollama serve > "$LOG_ROOT/ollama_gemma4_gpu1.log" 2>&1 &
log "  started gemma4 daemon GPU 1  port 11435  pid $!"

sleep 8

log "Health checks:"
for port in 11434 11435; do
    v=$(curl -sS "http://127.0.0.1:$port/api/version" 2>&1 || echo "FAIL")
    log "  port $port: $v"
done

# ============================================================
# STAGE 4: smoke test 2 local models
# ============================================================
if ! stage_skip_if_done stage4; then
    log "===== STAGE 4: smoke test (qwen + gemma4) ====="
    python3 << 'PYEOF' 2>&1 | tee -a "$MAIN_LOG"
import sys, time
sys.path.insert(0, '.')
from src.utils.api_clients import call_llm

prompt = "Output only the SQL keyword for selecting all rows. No explanation. Just one word."

for m in ['qwen25c7b', 'gemma4']:
    t0 = time.time()
    try:
        r = call_llm(prompt, m, seed=42)
        elapsed = time.time() - t0
        ok = ('SELECT' in r.upper()) and ('ERROR' not in r.upper()[:20])
        status = 'OK' if ok else 'WEIRD'
        print(f'  {m:14s} {status:5s}  {elapsed:5.1f}s  {repr(r[:80])}')
    except Exception as e:
        print(f'  {m:14s} FAIL   {time.time()-t0:5.1f}s  {str(e)[:200]}')
        sys.exit(1)
print('SMOKE PASS')
PYEOF
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log "STAGE 4 FAILED. Stopping."
        exit 1
    fi
    stage_done stage4
fi

# ============================================================
# STAGE 5: pilot — 2 models × 5 strategies × 50 BIRD query × seed=42
# ============================================================
if ! stage_skip_if_done stage5; then
    log "===== STAGE 5: pilot (50 query × 2 model × 5 strategy) ====="
    python3 scripts/run_q1_pilot_local.py 2>&1 | tee -a "$MAIN_LOG"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log "STAGE 5 FAILED. Stopping."
        exit 1
    fi
    stage_done stage5
fi

# ============================================================
# STAGE 6: main BIRD 609 mod+chal × 5 strategies × seed=42 × 2 models
# (Qwen first because faster ~11h. Gemma 4 next ~33h, may need 2nd night.)
# ============================================================
log "===== STAGE 6: BIRD main exp (no time limit, runs until done) ====="
python3 scripts/run_q1_bird_local.py 2>&1 | tee -a "$MAIN_LOG"
log "===== STAGE 6 EXITED ====="

log "===== OVERNIGHT RUN COMPLETED ====="
log "Logs: $LOG_ROOT"
log "Results: ~/evo-decompose-sql/results/q1_local_*.json"
