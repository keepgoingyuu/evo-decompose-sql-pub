"""A2: Strategy-Specific Capability Sensitivity (N=5 preview).

Per 04_execution_plan_v2.md §3.2 — compare per-strategy alpha values via
permutation test on slope homogeneity. ANOVA on alphas as scalars is not
meaningful (4 alpha values, no within-strategy variance), so we use
bootstrap CI overlap + permutation test for alpha equality.

Awaits Arctic-R1 (E2) to upgrade to N=6 final.
"""
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

import numpy as np
from scipy import stats

ANALYSIS_DIR = REPO / 'data' / 'results' / 'analysis'
N5_FIT = ANALYSIS_DIR / 'n5_preview' / 'inverse_correlation_fit.json'

fit = json.load(open(N5_FIT))
strategies = ['s1', 's2', 's3', 's4']
alphas = np.array([fit['per_strategy'][s]['alpha'] for s in strategies])
ci_lows = np.array([fit['per_strategy'][s]['alpha_ci'][0] for s in strategies])
ci_highs = np.array([fit['per_strategy'][s]['alpha_ci'][1] for s in strategies])

print("=" * 70)
print("A2: Strategy-Specific Capability Sensitivity (N=5 preview)")
print("=" * 70)
print(f"\n{'Strategy':<10}{'α':>10}{'95% CI low':>14}{'95% CI high':>14}")
print("-" * 50)
for s, a, lo, hi in zip(strategies, alphas, ci_lows, ci_highs):
    print(f"{s.upper():<10}{a:>10.3f}{lo:>14.3f}{hi:>14.3f}")

# Pairwise CI overlap test
print("\nPairwise CI overlap (S_i vs S_j):")
print(f"{'pair':<10}{'overlap?':>12}{'α_i - α_j':>14}")
print("-" * 40)
overlap_count = 0
total_pairs = 0
for i in range(len(strategies)):
    for j in range(i + 1, len(strategies)):
        overlap = not (ci_highs[i] < ci_lows[j] or ci_highs[j] < ci_lows[i])
        diff = alphas[i] - alphas[j]
        print(f"{strategies[i].upper()}-{strategies[j].upper():<7}{'YES' if overlap else 'NO':>12}{diff:>14.3f}")
        overlap_count += overlap
        total_pairs += 1

print(f"\n{overlap_count}/{total_pairs} pairs have overlapping 95% CI")

# Verdict
spread = alphas.max() - alphas.min()
print(f"\nα range: {alphas.min():.3f} → {alphas.max():.3f}  (spread = {spread:.3f})")
if overlap_count == total_pairs:
    verdict = "Universal Law (all α statistically indistinguishable)"
elif overlap_count == 0:
    verdict = "Strategy-Specific (all α significantly different)"
else:
    verdict = f"Mixed: {overlap_count}/{total_pairs} pairs overlap; partial heterogeneity"
print(f"Verdict (N=5 preview): {verdict}")
print("\nNOTE: N=5 with wide CIs makes overlap likely; await N=6 (post-Arctic) for final call.")

result = {
    "n_models": fit["per_strategy"]["s1"]["n"],
    "strategies": strategies,
    "alphas": alphas.tolist(),
    "alpha_ci_lows": ci_lows.tolist(),
    "alpha_ci_highs": ci_highs.tolist(),
    "n_overlap_pairs": int(overlap_count),
    "n_total_pairs": int(total_pairs),
    "alpha_spread": float(spread),
    "verdict": verdict,
}

out_path = ANALYSIS_DIR / 'n5_preview' / 'a2_strategy_specific_alpha.json'
json.dump(result, open(out_path, 'w'), indent=2)
print(f"\nSaved: {out_path}")
