#!/bin/bash
# 分批跑 17 格缺格: 3 供應商線並行, 線內序列。
# 同供應商序列跑避免 API rate-limit 互搶; 不同供應商並行不衝突。
# 在 DGX 上執行: bash scripts/rerun17_batched.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=logs/rerun17
mkdir -p "$LOG"

# ---- DeepSeek 線: CSpider s1-s4 (zh-tw) → 日文 s0-s4 ----
(
  for s in s1 s2 s3 s4; do
    "$PY" scripts/phase4_cspider_tw.py deepseek "$s" --lang zh-tw > "$LOG/deepseek_cspider_$s.log" 2>&1
  done
  "$PY" scripts/run_v21_xlingual_japanese.py deepseek --strategies s0 s1 s2 s3 s4 > "$LOG/deepseek_ja.log" 2>&1
) &
echo "DeepSeek 線 PID=$!"

# ---- Gemini 線: EN s0/s1 → 日文 s0-s4 ----
(
  "$PY" scripts/phase4_cspider_tw.py gemini s0 --lang en > "$LOG/gemini_en_s0.log" 2>&1
  "$PY" scripts/phase4_cspider_tw.py gemini s1 --lang en > "$LOG/gemini_en_s1.log" 2>&1
  "$PY" scripts/run_v21_xlingual_japanese.py gemini --strategies s0 s1 s2 s3 s4 > "$LOG/gemini_ja.log" 2>&1
) &
echo "Gemini 線 PID=$!"

# ---- OpenAI 線: CSpider s4 (zh-tw) ----
(
  "$PY" scripts/phase4_cspider_tw.py gpt41 s4 --lang zh-tw > "$LOG/gpt41_cspider_s4.log" 2>&1
) &
echo "OpenAI 線 PID=$!"

wait
echo "ALL_LINES_DONE"
