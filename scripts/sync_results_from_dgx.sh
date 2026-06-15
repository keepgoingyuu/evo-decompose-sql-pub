#!/usr/bin/env bash
# Sync experiment results from DGX to local repo.
# Usage: bash scripts/sync_results_from_dgx.sh [E1|E2|E3|all]

set -euo pipefail

DGX_HOST="dgx@123.193.150.180"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT="${1:-all}"

# Path mapping: DGX writes to results/q1_2026_04/main/, but local layout is data/results/E*/.
# Each experiment has (remote_pattern, local_destination).

sync_one() {
  local label="$1"; local remote_glob="$2"; local local_dest="$3"
  echo "=== syncing ${label} ==="
  mkdir -p "${local_dest}"
  rsync -avz --progress \
    --include="${remote_glob}" --exclude='*.tmp' --exclude='checkpoint_*' \
    "${DGX_HOST}:${remote_glob}" "${local_dest}/" 2>/dev/null || true
}

case "${EXPERIMENT}" in
  E1)
    sync_one "E1 BIRD"     "~/evo-decompose-sql/results/q1_2026_04/main/v21_bird_main_*"     "${LOCAL_ROOT}/data/results/E1_gemma_4_e4b/merged"
    sync_one "E1 xlingual" "~/evo-decompose-sql/results/q1_2026_04/main/v21_xlingual_*"      "${LOCAL_ROOT}/data/results/E1_gemma_4_e4b/merged"
    ;;
  E2)
    sync_one "E2 Arctic"   "~/evo-decompose-sql/results/q1_2026_04/main/v21_arctic_r1_*"     "${LOCAL_ROOT}/data/results/E2_arctic_r1"
    ;;
  E3)
    sync_one "E3 GPT-4.1 JA" "~/evo-decompose-sql/results/q1_2026_04/main/v21_gpt41_japanese_*" "${LOCAL_ROOT}/data/results/E3_gpt41_japanese"
    ;;
  analysis)
    sync_one "analysis"    "~/evo-decompose-sql/data/results/analysis/*"                    "${LOCAL_ROOT}/data/results/analysis"
    ;;
  all)
    "$0" E1 && "$0" E2 && "$0" E3
    ;;
  *)
    echo "Usage: $0 [E1|E2|E3|analysis|all]" >&2
    exit 1
    ;;
esac

echo "Done. Run sanity check before commit."
