"""A7: Question-level statistics for the S0-vs-S2 contrast (per language).

Motivation (JIIS reviews, 2026-08-19): both reviewers challenged the model-level
paired t / Cohen's d_z evidence as under-powered at N=5 models. This script adds
question-level inference that does not depend on the 5-model sample size:

  1. Per (model, language): exact McNemar test on the paired per-question
     S0/S2 outcomes (two-sided binomial test on discordant pairs), with
     Holm-Bonferroni correction across the 15 (model, language) cells.
  2. Per language, pooled over models: cluster-respecting sign-flip
     permutation test. Each question q contributes one clustered paired
     difference d_q = mean_m [S0(q,m) - S2(q,m)] over the 5 models, so the
     dependence between the 5 outcomes of the same question is absorbed
     before resampling. Also reports a question-clustered bootstrap 95% CI
     for the pooled drop.

Data sources (same files as build_xlingual_metrics.py, per-question level):
  original-3 EN/ZH-TW : results/thesis_2026_02/phase4_<m>_{s0,s2}_{en,zh-tw}.json
  frontier JA         : results/q1_2026_04/main/v21_<m>_japanese_{s0,s2}_seed42.json
  Gemma4/Qwen25C7B    : results/q1_2026_04/main/v21_xlingual_<bench>_<m>_{s0,s2}_seed42.json

Join key: query_idx within each (model, language) file pair.

Output: results/analysis/question_level_s2_stats.json

Usage: venv/bin/python scripts/a7_question_level_s2_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
PHASE4_DIR = REPO / "results/thesis_2026_02"
V21_DIR = REPO / "results/q1_2026_04/main"
OUT = REPO / "results/analysis/question_level_s2_stats.json"

SEED = 42
N_PERM = 100_000
N_BOOT = 10_000

LANGS = ["EN", "ZH-TW", "JA"]
MODELS = ["GPT-4.1", "DeepSeek-V3", "Gemini-3-Flash-Preview",
          "Qwen2.5-Coder-7B", "Gemma-4-E4B"]

V21_BENCH = {"EN": "xlingual_spider_en", "ZH-TW": "xlingual_cspider_zh_tw",
             "JA": "xlingual_multispider_ja"}


def file_for(model: str, lang: str, s: str) -> Path:
    """Resolve the per-question result file for one (model, language, strategy)."""
    raw = {"GPT-4.1": "gpt41", "DeepSeek-V3": "deepseek",
           "Gemini-3-Flash-Preview": "gemini"}.get(model)
    if raw:
        if lang == "JA":
            return V21_DIR / f"v21_{raw}_japanese_{s}_seed42.json"
        return PHASE4_DIR / f"phase4_{raw}_{s}_{lang.lower()}.json"
    raw = {"Qwen2.5-Coder-7B": "qwen25c7b", "Gemma-4-E4B": "gemma4"}[model]
    return V21_DIR / f"v21_{V21_BENCH[lang]}_{raw}_{s}_seed42.json"


def load_correct(f: Path) -> dict:
    """{query_idx: bool} from one result file."""
    d = json.load(open(f))
    return {r["query_idx"]: bool(r["correct"]) for r in d["results"]}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return stats.binomtest(b, n, 0.5, alternative="two-sided").pvalue


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict {key: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=" * 74)
    print("A7: Question-level S0-vs-S2 statistics (McNemar + clustered permutation)")
    print("=" * 74)

    per_cell = {}
    lang_matrix = {}  # lang -> {qidx: {model: (s0, s2)}}
    for lang in LANGS:
        lang_matrix[lang] = {}
        for model in MODELS:
            f0, f2 = file_for(model, lang, "s0"), file_for(model, lang, "s2")
            if not f0.exists() or not f2.exists():
                print(f"  skip {model} {lang}: missing {f0.name if not f0.exists() else f2.name}")
                continue
            s0, s2 = load_correct(f0), load_correct(f2)
            common = sorted(set(s0) & set(s2))
            b = sum(1 for q in common if s0[q] and not s2[q])   # S2 loses
            c = sum(1 for q in common if not s0[q] and s2[q])   # S2 rescues
            n = len(common)
            drop_pp = (b - c) / n * 100
            per_cell[(model, lang)] = {
                "n_questions": n,
                "s0_acc_pct": sum(s0[q] for q in common) / n * 100,
                "s2_acc_pct": sum(s2[q] for q in common) / n * 100,
                "discordant_s2_loses": b,
                "discordant_s2_rescues": c,
                "drop_pp": drop_pp,
                "p_mcnemar_exact": mcnemar_exact(b, c),
            }
            for q in common:
                lang_matrix[lang].setdefault(q, {})[model] = (s0[q], s2[q])

    adj = holm({k: v["p_mcnemar_exact"] for k, v in per_cell.items()})
    for k in per_cell:
        per_cell[k]["p_holm"] = adj[k]

    print(f"\nPer-cell exact McNemar (Holm-corrected over {len(per_cell)} cells):")
    print(f"{'model':<26}{'lang':<7}{'n':>6}{'S0%':>8}{'S2%':>8}{'drop pp':>9}"
          f"{'b/c':>10}{'p_exact':>11}{'p_holm':>10}")
    for (model, lang), v in per_cell.items():
        print(f"{model:<26}{lang:<7}{v['n_questions']:>6}{v['s0_acc_pct']:>8.2f}"
              f"{v['s2_acc_pct']:>8.2f}{v['drop_pp']:>+9.2f}"
              f"{v['discordant_s2_loses']:>5}/{v['discordant_s2_rescues']:<4}"
              f"{v['p_mcnemar_exact']:>11.2e}{v['p_holm']:>10.2e}")

    pooled = {}
    print(f"\nPer-language pooled (question-clustered, {N_PERM:,} sign-flips):")
    for lang in LANGS:
        qs = [q for q, md in lang_matrix[lang].items() if len(md) == len(MODELS)]
        d_q = np.array([np.mean([md[m][0] - md[m][1] for m in MODELS])
                        for q, md in ((q, lang_matrix[lang][q]) for q in qs)])
        obs = d_q.mean() * 100
        # sign-flip permutation: under H0 each question's clustered difference
        # is symmetric around 0
        flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d_q)))
        null = (flips * d_q).mean(axis=1) * 100
        p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
        boot_idx = rng.integers(0, len(d_q), size=(N_BOOT, len(d_q)))
        boot = d_q[boot_idx].mean(axis=1) * 100
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        pooled[lang] = {
            "n_questions": len(d_q),
            "n_models": len(MODELS),
            "pooled_drop_pp": float(obs),
            "ci95_cluster_bootstrap": ci,
            "p_signflip_permutation": float(p_perm),
        }
        print(f"  {lang:<7} n={len(d_q):>5}  drop={obs:+.2f} pp  "
              f"CI95=[{ci[0]:+.2f}, {ci[1]:+.2f}]  p={p_perm:.2e}")

    OUT.write_text(json.dumps({
        "experiment_id": "a7",
        "name": "question_level_s0_vs_s2",
        "seed": SEED,
        "n_permutations": N_PERM,
        "n_bootstrap": N_BOOT,
        "per_cell": {f"{m}|{l}": v for (m, l), v in per_cell.items()},
        "pooled_per_language": pooled,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
