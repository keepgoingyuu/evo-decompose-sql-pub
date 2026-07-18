"""A4: Statistical Rigor Pass.

Per 04_execution_plan_v2.md §3.2 — apply paired t-test + Bonferroni + Cohen's d
to each strategy comparison.

NOTE: With seed-42 only (no seed 1/2 reruns), pairing is over models, not seeds.
This is the "across-model paired comparison" version. Per-model multi-seed
analysis requires DGX to rerun E1 with seeds 1, 2 (extra ~40h, not done yet).

Method: For each (bench, strategy_pair=(s_i, s_j)), compute paired diff
(per_model accuracy_s_i - accuracy_s_j) across models, then:
  - paired t-test (H0: mean diff = 0)
  - Bonferroni correction across (4 strat) × (5 strat-pair compares per strat)
  - Cohen's d_z for paired samples
"""
import json
from itertools import combinations
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

import numpy as np
from scipy import stats

ANALYSIS_DIR = REPO / 'results' / 'analysis'
DATA = json.load(open(ANALYSIS_DIR / 'xlingual_metrics_unified.json'))


def cohen_d_z(diffs: np.ndarray) -> float:
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return float('nan')
    return float(np.mean(diffs) / sd)


def paired_test(per_model_acc: list[tuple[float, float]]):
    """Each tuple is (acc_i, acc_j) for one model. Return paired test stats."""
    acc_i = np.array([p[0] for p in per_model_acc])
    acc_j = np.array([p[1] for p in per_model_acc])
    diffs = acc_i - acc_j
    if len(diffs) < 2 or np.std(diffs) == 0:
        return None
    t_stat, p_val = stats.ttest_rel(acc_i, acc_j)
    return {
        'n_pairs': len(diffs),
        'mean_diff': float(np.mean(diffs)),
        'std_diff': float(np.std(diffs, ddof=1)),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'cohen_dz': cohen_d_z(diffs),
    }


print("=" * 70)
print("A4: Statistical Rigor Pass — Paired t-test + Bonferroni + Cohen's d")
print("=" * 70)

LANG_BENCH = {'EN': 'EN', 'ZH-TW': 'ZH-TW', 'JA': 'JA'}
STRATEGIES = ['s0', 's1', 's2', 's3', 's4']
PAIRS = list(combinations(STRATEGIES, 2))  # 10 pairs

results = {}
all_p_values = []

for lang in LANG_BENCH:
    print(f"\n=== {lang} ===")
    print(f"{'pair':<10}{'n':>4}{'Δmean':>10}{'t':>10}{'p':>12}{'p_bonf':>12}{'cohen_dz':>12}")
    print("-" * 70)
    lang_models = {m: langs[lang] for m, langs in DATA.items() if lang in langs}
    results[lang] = {}
    for s_a, s_b in PAIRS:
        rows = [(d[s_a], d[s_b]) for d in lang_models.values() if s_a in d and s_b in d]
        test = paired_test(rows)
        pair_label = f'{s_a.upper()}-{s_b.upper()}'
        if not test:
            results[lang][pair_label] = None
            continue
        # Bonferroni across 10 pairs × 3 langs = 30 tests
        p_bonf = min(test['p_value'] * 30, 1.0)
        test['p_bonferroni'] = p_bonf
        test['significant_after_bonf'] = p_bonf < 0.05
        results[lang][pair_label] = test
        all_p_values.append(test['p_value'])
        sig = '***' if p_bonf < 0.001 else ('**' if p_bonf < 0.01 else ('*' if p_bonf < 0.05 else ''))
        print(f"{pair_label:<10}{test['n_pairs']:>4}{test['mean_diff']:>10.2f}{test['t_stat']:>10.2f}{test['p_value']:>12.4f}{p_bonf:>12.4f}{test['cohen_dz']:>12.3f}{sig:>4}")

# Headline summary
print("\n" + "=" * 70)
print("Headline statistic checks")
print("=" * 70)

# S2 vs S0 (calibration of "decomposition harmful" hypothesis)
print("\nS2-S0 (decomposition vs direct):")
for lang in LANG_BENCH:
    t = results[lang].get('S0-S2')
    if t:
        print(f"  {lang:<8} Δmean={t['mean_diff']:+.2f}  p={t['p_value']:.4f}  p_bonf={t['p_bonferroni']:.4f}  d_z={t['cohen_dz']:.3f}")

print("\nS1-S0 (schema filter vs direct):")
for lang in LANG_BENCH:
    t = results[lang].get('S0-S1')
    if t:
        print(f"  {lang:<8} Δmean={t['mean_diff']:+.2f}  p={t['p_value']:.4f}  p_bonf={t['p_bonferroni']:.4f}  d_z={t['cohen_dz']:.3f}")

out = {
    'note': 'Paired t-test across models per language; seed-42 only (no within-model paired seed comparisons yet)',
    'bonferroni_n_tests': 30,
    'results': results,
}
out_path = ANALYSIS_DIR / 'n5_preview' / 'a4_statistical_rigor.json'
json.dump(out, open(out_path, 'w'), indent=2, default=str)
print(f"\n✓ Saved: {out_path}")
