# Strategy Selection for LLM-based Text-to-SQL

Reproducibility package for the paper:

> **Strategy Selection for LLM-based Text-to-SQL: A Cross-Model Cross-Lingual Empirical Study with Reinforcement Learning Boundary**
> Chih-Yu Lin, Huan-Yu Chen — National Taichung University of Science and Technology
> (under journal review)

This repository contains the per-query predictions, evaluation pipeline, and
statistical-analysis scripts that produce the tables and figure statistics in the paper.

---

## What's here

```
results/
├── thesis_2026_02/     Original 4-model EN/ZH-TW runs (phase1/2/4 per-query files)
├── q1_2026_04/
│   ├── main/           v2.1 runs: Gemma-4-E4B & Qwen2.5-Coder BIRD + cross-lingual;
│   │                   GPT-4.1 / DeepSeek-V3 / Gemini Flash Japanese runs
│   └── pilot/          Pilot runs (provenance only)
├── E2_arctic_r1/       RL anchor: Arctic-Text2SQL-R1 7B per-query predictions,
│                       4 benchmarks under the uniform protocol
└── analysis/           Derived metrics + statistical artifacts (rebuilt by scripts/)
    └── n5_preview/     a2–a5 statistical outputs used by the paper
src/         Text-to-SQL pipelines (S0–S4) + execution-accuracy evaluation utilities
scripts/     Experiment runners + statistical-analysis scripts
CITATION.cff How to cite
requirements.txt
.env.example Template for API keys (only needed to re-run inference)
```

**Canonical files.** Per-cell canonical predictions are the *merged* files
`v21_<bench>_<model>_<strategy>_seed42.json` and `phase*_<model>_<strategy>*.json`.
Files suffixed `_chunkN`, `_smoke`, or `_partial` are intermediate shards kept for
provenance — analysis scripts never read them.

### The experimental matrix

| Axis | Values |
|---|---|
| **Models (5)** | GPT-4.1, DeepSeek-V3, Qwen2.5-Coder-7B, Gemini Flash, Gemma-4-E4B |
| **Strategies (5)** | S0 direct generation · S1 schema filtering · S2 question decomposition · S3 skeleton-then-fill · S4 classify-then-generate |
| **Benchmarks (4)** | BIRD mod+chal (609) · Spider EN (1034) · CSpider ZH-TW (1034) · MultiSpider 2.0 JA (1034) |
| **RL anchor** | Snowflake Arctic-Text2SQL-R1 7B (`results/E2_arctic_r1/`) |

All experiments use deterministic decoding (temperature 0, seed 42) under a single
uniform prompting protocol.

---

## Evaluation metric

Execution accuracy (EX): a prediction is correct iff its execution result on the
database equals the gold query's result. Result rows are compared row-by-row in the
order returned by SQLite (a conservative lower bound on the official order-insensitive
set-equality EX). See `src/utils/sql_utils.py` (`compare_results`).

---

## Reproducing the paper's numbers

The `results/` predictions are the ground truth — the paper's accuracy tables are
recomputed directly from the `correct` field of each per-query record. No model
inference is needed to reproduce the tables.

```bash
pip install -r requirements.txt

# Example: recompute BIRD mod+chal accuracy for a (model, strategy)
python -c "
import json
d = json.load(open('results/thesis_2026_02/phase2_gpt41_s1.json'))
r = d['results']
acc = sum(1 for x in r if x['correct']) / len(r) * 100
print(f'GPT-4.1 BIRD S1 accuracy = {acc:.2f}%')
"   # → 41.05%
```

### Paper item → script → artifact

| Paper item | Script | Output artifact |
|---|---|---|
| Accuracy tables (per model × strategy × benchmark) | per-query `correct` fields (snippet above) | — |
| Unified metrics matrix (all models × 3 languages × 5 strategies) | `scripts/build_xlingual_metrics.py` | `results/analysis/xlingual_metrics_unified.json` |
| Cross-lingual ranking concordance (Kendall's W = 0.91, exact permutation p = 240/13,824 = 0.017, pairwise τ ≥ 0.67) | `scripts/a6_kendall_w.py` | `results/analysis/n5_preview/a6_kendall_w.json` |
| Cross-lingual slope-invariance test (per-strategy α CIs) | `scripts/a3_cross_lingual_invariance.py` | `results/analysis/n5_preview/a3_cross_lingual_invariance.json` |
| Paired t-tests + Cohen's d_z (decomposition harm, d_z > 1.0 in all 3 languages) | `scripts/a4_statistical_rigor.py` | `results/analysis/n5_preview/a4_statistical_rigor.json` |
| Capability–benefit fits (per-strategy α) | `scripts/a2_strategy_specific_alpha.py` | `results/analysis/n5_preview/a2_strategy_specific_alpha.json` |
| Adaptive-selector evaluation (oracle 50.20%, gap 16.71 pp; S0→S1 fallback 9.28 pp; 11 learned selectors, 5-fold CV) | `scripts/phase3_v3_n5_selector.py`¹ | `results/analysis/n5_final/n5_selector_results.json` |
| Per-model oracle preview (simplified alignment variant) | `scripts/a5_adaptive_selector_v21.py` | `results/analysis/n5_preview/a5_adaptive_selector.json` |
| RL-anchor accuracy under uniform protocol | per-query files | `results/E2_arctic_r1/*.json`, summary in `results/analysis/rl_baseline_summary.json` |
| Latency profile | `scripts/analyze_latency.py` | `results/analysis/latency_metrics.json` |

¹ Re-running the full selector cross-validation (28-feature set) additionally
requires the BIRD databases downloaded under `data/` (see below); the shipped
`n5_final` artifact contains all numbers reported in the paper. All other
scripts run from the shipped `results/` tree alone.

Quick verification run (needs only `numpy` + `scipy`; each script rebuilds its
output artifact in place):

```bash
python scripts/build_xlingual_metrics.py        # rebuild the 5-model x 3-language matrix
python scripts/a6_kendall_w.py                  # Kendall W = 0.91, exact p = 240/13824, tau >= 0.67
python scripts/a4_statistical_rigor.py          # paired t-tests + Cohen's d_z (> 1.0 in all 3 languages)
python scripts/a3_cross_lingual_invariance.py   # per-strategy slope-invariance verdicts
```

---

## Benchmark data (not included)

The benchmark databases and questions (BIRD, Spider, CSpider, MultiSpider 2.0) are
**not redistributed here** — they carry their own licences. Download them from the
official sources and place them under `data/`:

- BIRD: https://bird-bench.github.io/
- Spider: https://yale-lily.github.io/spider
- CSpider: https://taolusi.github.io/CSpider-explorer/
- MultiSpider: https://github.com/Longtao-Hu/MultiSpider

To re-run model inference (not required for reproducing the tables), copy
`.env.example` to `.env` and fill in your API keys.

---

## Licence

Code and analysis scripts: MIT (see `LICENSE`).
Prediction outputs under `results/` are released for research reproducibility.
Benchmark data are subject to their original licences.
