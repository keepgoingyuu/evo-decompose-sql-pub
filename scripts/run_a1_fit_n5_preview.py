"""N=5 preview fit: original 4 models (BIRD hardcoded) + Gemma 4 E4B from v2.1.

Qwen25C7B v2.1 rerun (33.0%) replaces phase1 hardcoded (27.59%) — same model, fresher data.
Awaiting Arctic-R1 to upgrade to N=6.
"""
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inverse_correlation_fit import BIRD_HARDCODED, fit_per_strategy

E1_MERGED = REPO / 'data' / 'results' / 'E1_gemma_4_e4b' / 'merged'

def load_v21_bird(model: str) -> dict[str, float]:
    out = {}
    for s in ['s0', 's1', 's2', 's3', 's4']:
        f = E1_MERGED / f'v21_bird_main_{model}_{s}_seed42.json'
        d = json.load(open(f))
        out[s] = round(d['metadata']['accuracy'] * 100, 2)
    return out

data = dict(BIRD_HARDCODED)
data['Qwen2.5-Coder-7B'] = load_v21_bird('qwen25c7b')
data['Gemma-4-E4B'] = load_v21_bird('gemma4')

print("Input data:")
for k, v in data.items():
    print(f"  {k:<28} {v}")

out_dir = REPO / 'data' / 'results' / 'analysis'
fit_per_strategy(data, out_dir / 'n5_preview')
