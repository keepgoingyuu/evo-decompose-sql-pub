"""A3: Cross-Lingual Law Invariance Test (N=5 preview).

Per 04_execution_plan_v2.md §3.2 — fit one law per language, check whether
α values for each strategy overlap across EN / ZH-TW / JA within 95% bootstrap CI.

Data source: results/analysis/xlingual_metrics_unified.json
(built by scripts/build_xlingual_metrics.py)
"""
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

import numpy as np
from scipy import stats

ANALYSIS_DIR = REPO / 'results' / 'analysis'
DATA = json.load(open(ANALYSIS_DIR / 'xlingual_metrics_unified.json'))


def fit_with_ci(c_values: np.ndarray, delta_values: np.ndarray, n_resamples: int = 5000):
    log_c = np.log(c_values)
    slope, intercept, r_val, p_val, std_err = stats.linregress(log_c, delta_values)
    alpha = -slope
    rng = np.random.default_rng(42)
    n = len(c_values)
    alphas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(c_values[idx])) < 2:
            continue
        try:
            slope_b, _, _, _, _ = stats.linregress(np.log(c_values[idx]), delta_values[idx])
            alphas.append(-slope_b)
        except Exception:
            continue
    if not alphas:
        return alpha, float(r_val ** 2), (float('nan'), float('nan')), n
    return alpha, float(r_val ** 2), (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))), n


def fit_lang_strategy(lang_data: dict, strategy: str):
    """Fit per-strategy law using all models with both s0 and target strategy."""
    pairs = [(d['s0'], d[strategy] - d['s0']) for d in lang_data.values()
             if 's0' in d and strategy in d]
    if len(pairs) < 3:
        return None
    c = np.array([p[0] for p in pairs])
    delta = np.array([p[1] for p in pairs])
    alpha, r2, ci, n = fit_with_ci(c, delta)
    return {'alpha': float(alpha), 'r_squared': r2, 'alpha_ci': ci, 'n': n}


print("=" * 70)
print("A3: Cross-Lingual Law Invariance Test (N=5 preview)")
print("=" * 70)

results = {}
for lang in ['EN', 'ZH-TW', 'JA']:
    lang_data = {m: langs[lang] for m, langs in DATA.items() if lang in langs}
    print(f"\n{lang}: {len(lang_data)} models — {list(lang_data.keys())}")
    results[lang] = {}
    for s in ['s1', 's2', 's3', 's4']:
        fit = fit_lang_strategy(lang_data, s)
        results[lang][s] = fit

# Comparison table
print(f"\n{'Strategy':<10}" + "".join(f"{lang+' α':>14}" for lang in ['EN', 'ZH-TW', 'JA']))
print("-" * 60)
for s in ['s1', 's2', 's3', 's4']:
    row = f"{s.upper():<10}"
    for lang in ['EN', 'ZH-TW', 'JA']:
        fit = results[lang].get(s)
        if fit:
            row += f"{fit['alpha']:>9.2f}(n={fit['n']})"
        else:
            row += f"{'N/A':>14}"
    print(row)

# Overlap test: for each strategy, do EN/ZH-TW/JA alphas have overlapping CIs?
print(f"\n{'Strategy':<10}{'EN ∩ ZH-TW':>14}{'EN ∩ JA':>14}{'ZH-TW ∩ JA':>14}{'Verdict':<22}")
print("-" * 75)
verdict_per_strategy = {}
for s in ['s1', 's2', 's3', 's4']:
    fits = {lang: results[lang].get(s) for lang in ['EN', 'ZH-TW', 'JA']}
    if not all(fits.values()):
        verdict_per_strategy[s] = 'incomplete'
        print(f"{s.upper():<10}{'INCOMPLETE':>56}")
        continue

    def overlap(f1, f2):
        if f1 is None or f2 is None:
            return None
        lo1, hi1 = f1['alpha_ci']
        lo2, hi2 = f2['alpha_ci']
        return not (hi1 < lo2 or hi2 < lo1)

    o_en_zh = overlap(fits['EN'], fits['ZH-TW'])
    o_en_ja = overlap(fits['EN'], fits['JA'])
    o_zh_ja = overlap(fits['ZH-TW'], fits['JA'])
    overlaps = [o_en_zh, o_en_ja, o_zh_ja]

    if all(overlaps):
        verdict = 'invariant ✓'
    elif not any(overlaps):
        verdict = 'lang-divergent ✗'
    else:
        verdict = f'{sum(overlaps)}/3 overlap'
    verdict_per_strategy[s] = verdict
    cells = ['YES' if o else 'NO' for o in overlaps]
    print(f"{s.upper():<10}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}  {verdict:<22}")

n_invariant = sum(1 for v in verdict_per_strategy.values() if 'invariant' in v)
print(f"\nOverall: {n_invariant}/4 strategies invariant across languages (N=5 preview)")

out = {
    'note': 'N=3-5 per language preview; awaiting Arctic-R1 + missing phase4 cells',
    'per_lang_per_strategy': {
        lang: {s: (
            {'alpha': v['alpha'], 'r_squared': v['r_squared'],
             'alpha_ci_low': v['alpha_ci'][0], 'alpha_ci_high': v['alpha_ci'][1], 'n': v['n']}
            if v else None
        ) for s, v in d.items()}
        for lang, d in results.items()
    },
    'invariance_verdict_per_strategy': verdict_per_strategy,
    'n_invariant_of_4': n_invariant,
}
out_path = ANALYSIS_DIR / 'n5_preview' / 'a3_cross_lingual_invariance.json'
json.dump(out, open(out_path, 'w'), indent=2)
print(f"\n✓ Saved: {out_path}")
