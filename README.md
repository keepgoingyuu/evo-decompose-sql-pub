# Strategy Selection for LLM-based Text-to-SQL

Reproducibility package for the paper:

> **Strategy Selection for LLM-based Text-to-SQL: A Cross-Model Cross-Lingual Empirical Study with Reinforcement Learning Boundary**
> Chih-Yu Lin, Huan-Yu Chen, Chun-Hung Lin — National Taichung University of Science and Technology
> Submitted to *Data & Knowledge Engineering* (Elsevier).

This repository contains the per-query predictions, evaluation pipeline, and
statistical-analysis scripts that produce the tables and figures in the paper.

---

## What's here

```
results/     Per-query predictions: 5 LLMs × 5 strategies × 4 benchmarks (221 JSON files)
src/         Text-to-SQL pipelines (S0–S4) + execution-accuracy evaluation utilities
scripts/     Experiment runners + statistical-analysis scripts that build the paper's tables
requirements.txt
.env.example Template for API keys (copy to .env and fill in)
```

### The experimental matrix

| Axis | Values |
|---|---|
| **Models (5)** | GPT-4.1, DeepSeek-V3, Qwen2.5-Coder-7B, Gemini Flash, Gemma-4-E4B |
| **Strategies (5)** | S0 direct generation · S1 schema filtering · S2 question decomposition · S3 skeleton-then-fill · S4 classify-then-generate |
| **Benchmarks (4)** | BIRD mod+chal (609) · Spider EN (1034) · CSpider ZH-TW (1034) · MultiSpider 2.0 JA (1034) |
| **RL anchor** | Snowflake Arctic-Text2SQL-R1 7B |

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
"
```

Statistical-analysis scripts (paired t-tests, Cohen's d_z, cross-lingual concordance,
capability-benefit fit, adaptive-selector evaluation) are in `scripts/` — e.g.
`a3_cross_lingual_invariance.py`, `a4_statistical_rigor.py`, `a5_adaptive_selector_v21.py`.

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
