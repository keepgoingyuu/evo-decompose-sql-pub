"""A6: Cross-lingual strategy-ranking concordance (Kendall's W + exact permutation test).

Reproduces the paper's headline concordance statistics:
- Per-language rankings of the four non-trivial strategies (S1-S4), ranked by
  mean execution accuracy across the five prompting models.
- Kendall's coefficient of concordance W across the three languages.
- Exact permutation p-value: each language's ranking is permuted independently
  over all (4!)^3 = 13,824 arrangements; p = P(W_perm >= W_obs).
- Pairwise Kendall's tau between language rankings.

Data source: results/analysis/xlingual_metrics_unified.json
(built by scripts/build_xlingual_metrics.py)

Expected output (paper Section 5, cross-lingual invariance):
  W = 0.91, exact permutation p = 0.017 (240 / 13,824), pairwise tau >= 0.67
"""
import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO / 'results' / 'analysis'
DATA = json.load(open(ANALYSIS_DIR / 'xlingual_metrics_unified.json'))

LANGS = ['EN', 'ZH-TW', 'JA']
STRATS = ['s1', 's2', 's3', 's4']  # non-trivial strategies


def mean_acc_table():
    """mean accuracy per (language, strategy) across all models with full coverage."""
    table = {}
    for lang in LANGS:
        models = [m for m in DATA if lang in DATA[m] and all(s in DATA[m][lang] for s in STRATS)]
        table[lang] = {s: float(np.mean([DATA[m][lang][s] for m in models])) for s in STRATS}
        print(f"{lang}: {len(models)} models — " +
              ", ".join(f"{s.upper()}={table[lang][s]:.2f}" for s in STRATS))
    return table


def ranks_of(scores: dict) -> list:
    """rank strategies by descending mean accuracy: best = rank 1. Returns ranks aligned to STRATS order."""
    order = sorted(STRATS, key=lambda s: -scores[s])
    return [order.index(s) + 1 for s in STRATS]


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """rank_matrix: (m judges x n items) of ranks."""
    m, n = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)
    s = ((rank_sums - rank_sums.mean()) ** 2).sum()
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def main():
    print("=" * 70)
    print("A6: Cross-lingual strategy-ranking concordance (S1-S4)")
    print("=" * 70)

    table = mean_acc_table()
    rank_rows = []
    for lang in LANGS:
        r = ranks_of(table[lang])
        rank_rows.append(r)
        order = sorted(STRATS, key=lambda s: -table[lang][s])
        print(f"{lang:6s} ranking: {' > '.join(s.upper() for s in order)}   ranks={r}")

    R = np.array(rank_rows)
    w_obs = kendalls_w(R)

    # Exact permutation test: permute each language's ranking independently.
    n_items = len(STRATS)
    all_perms = list(permutations(range(1, n_items + 1)))
    count_ge = 0
    total = 0
    for combo in product(all_perms, repeat=len(LANGS)):
        total += 1
        if kendalls_w(np.array(combo)) >= w_obs - 1e-12:
            count_ge += 1
    p_exact = count_ge / total

    print(f"\nKendall's W = {w_obs:.4f}")
    print(f"Exact permutation p = {count_ge} / {total} = {p_exact:.4f}")

    taus = {}
    for i in range(len(LANGS)):
        for j in range(i + 1, len(LANGS)):
            tau, _ = stats.kendalltau(R[i], R[j])
            taus[f"{LANGS[i]}~{LANGS[j]}"] = round(float(tau), 4)
            print(f"pairwise tau {LANGS[i]:6s} ~ {LANGS[j]:6s} = {tau:+.2f}")

    out = {
        'note': 'Kendall W over per-language rankings of S1-S4 (mean accuracy across models); exact permutation test over (4!)^3 arrangements.',
        'mean_accuracy': table,
        'ranks': {lang: rank_rows[i] for i, lang in enumerate(LANGS)},
        'strategies': STRATS,
        'kendalls_w': round(w_obs, 4),
        'exact_permutation': {'count_ge': count_ge, 'total': total, 'p': round(p_exact, 5)},
        'pairwise_tau': taus,
    }
    out_path = ANALYSIS_DIR / 'n5_preview' / 'a6_kendall_w.json'
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f"\n✓ Saved: {out_path}")


if __name__ == '__main__':
    main()
