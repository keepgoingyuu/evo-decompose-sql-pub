"""A8: Direct evidence for the error-propagation mechanism of S2 harm.

JIIS Reviewer 1 (2026-08-19): "the paper mainly attributes this phenomenon to
error propagation, but no direct evidence is provided". The per-query S2 logs
do not store intermediate sub-questions, but two mechanism signals are
recoverable from the final outputs:

  1. FAILURE-MODE SPLIT. Among queries the model answers correctly under S0
     but loses under S2 ("broken" queries), classify the S2 output as
     (a) execution failure (SQL fails to run: syntax/binding errors, `error`
         field non-null), i.e. the assembled query is structurally invalid, or
     (b) executable-but-wrong result set.
     Error propagation through decompose -> solve -> assemble predicts an
     elevated execution-failure share relative to the model's own S0
     failure profile.

  2. DOSE-RESPONSE. S2's call_count = (#sub-questions + 2) (decompose +
     one call per sub-question + assemble). Within the stratum of queries the
     model already solves under S0 (controlling question difficulty), test
     whether P(S2 breaks the query) rises with the number of sub-questions.
     A monotone rise is direct dose-response evidence: more propagation
     steps -> more breakage. Cochran-Armitage-style trend test via the
     point-biserial correlation between #sub-questions and breakage, with a
     permutation p-value.

Cells: 5 models x {BIRD EN, Spider EN, CSpider ZH-TW, MultiSpider JA}.
Output: results/analysis/s2_error_propagation.json

Usage: venv/bin/python scripts/a8_s2_error_propagation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PHASE1 = REPO / "results/thesis_2026_02"
V21 = REPO / "results/q1_2026_04/main"
OUT = REPO / "results/analysis/s2_error_propagation.json"

SEED = 42
N_PERM = 100_000

MODELS = ["GPT-4.1", "DeepSeek-V3", "Gemini-3-Flash-Preview",
          "Qwen2.5-Coder-7B", "Gemma-4-E4B"]
RAW_FRONTIER = {"GPT-4.1": "gpt41", "DeepSeek-V3": "deepseek",
                "Gemini-3-Flash-Preview": "gemini"}
RAW_V21 = {"Qwen2.5-Coder-7B": "qwen25c7b", "Gemma-4-E4B": "gemma4"}
XBENCH = {"EN": "xlingual_spider_en", "ZH-TW": "xlingual_cspider_zh_tw",
          "JA": "xlingual_multispider_ja"}


def files_for(model: str, bench: str) -> tuple[Path, Path] | None:
    """(s0_file, s2_file) for one (model, benchmark)."""
    if bench == "BIRD":
        if model in RAW_FRONTIER:
            raw = RAW_FRONTIER[model]
            p1 = {"gpt41": "phase1_gpt41_baseline.json",
                  "deepseek": "phase1_deepseek_baseline.json",
                  "gemini": "phase1_gemini3flash_baseline.json"}[raw]
            return PHASE1 / p1, PHASE1 / f"phase2_{raw}_s2.json"
        raw = RAW_V21[model]
        return (V21 / f"v21_bird_main_{raw}_s0_seed42.json",
                V21 / f"v21_bird_main_{raw}_s2_seed42.json")
    if model in RAW_FRONTIER:
        raw = RAW_FRONTIER[model]
        if bench == "JA":
            return (V21 / f"v21_{raw}_japanese_s0_seed42.json",
                    V21 / f"v21_{raw}_japanese_s2_seed42.json")
        lang = bench.lower()
        return (PHASE1 / f"phase4_{raw}_s0_{lang}.json",
                PHASE1 / f"phase4_{raw}_s2_{lang}.json")
    raw = RAW_V21[model]
    b = XBENCH[bench]
    return (V21 / f"v21_{b}_{raw}_s0_seed42.json",
            V21 / f"v21_{b}_{raw}_s2_seed42.json")


def load_rows(f: Path, bird_frontier: bool) -> dict:
    """{qid: {correct, exec_error, n_sub}}; qid = query_id or query_idx."""
    d = json.load(open(f))
    out = {}
    for r in d["results"]:
        if bird_frontier and r.get("difficulty") not in ("moderate", "challenging"):
            continue
        qid = r.get("query_idx", r.get("query_id"))
        err = r.get("error")
        cc = r.get("call_count")
        out[qid] = {
            "correct": bool(r["correct"]),
            "exec_error": bool(err),
            "n_sub": (cc - 2) if (cc is not None and cc >= 3) else None,
        }
    return out


def trend_test(ks: np.ndarray, broken: np.ndarray, rng) -> float:
    """Permutation p for positive correlation between #sub-questions and breakage."""
    if len(set(ks.tolist())) < 2 or broken.std() == 0:
        return float("nan")
    obs = np.corrcoef(ks, broken)[0, 1]
    perm = np.empty(N_PERM)
    b = broken.copy()
    for i in range(N_PERM):
        rng.shuffle(b)
        perm[i] = np.corrcoef(ks, b)[0, 1]
    return float((np.sum(perm >= obs) + 1) / (N_PERM + 1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("A8: S2 error-propagation evidence (failure-mode split + dose-response)")
    print("=" * 78)

    cells = {}
    pooled_dose = {"ks": [], "broken": []}   # over all cells, S0-correct stratum
    pooled_modes = {"broken_exec": 0, "broken_total": 0,
                    "s0fail_exec": 0, "s0fail_total": 0}

    for bench in ["BIRD", "EN", "ZH-TW", "JA"]:
        for model in MODELS:
            fpair = files_for(model, bench)
            f0, f2 = fpair
            if not f0.exists() or not f2.exists():
                print(f"  skip {model} {bench}")
                continue
            bird_frontier = bench == "BIRD" and model in RAW_FRONTIER
            s0 = load_rows(f0, bird_frontier)
            s2 = load_rows(f2, bird_frontier)
            common = sorted(set(s0) & set(s2))

            # --- failure-mode split ---
            broken = [q for q in common if s0[q]["correct"] and not s2[q]["correct"]]
            broken_exec = sum(1 for q in broken if s2[q]["exec_error"])
            s0_fail = [q for q in common if not s0[q]["correct"]]
            s0_fail_exec = sum(1 for q in s0_fail if s0[q]["exec_error"])

            # --- dose-response within S0-correct stratum ---
            stratum = [q for q in common
                       if s0[q]["correct"] and s2[q]["n_sub"] is not None]
            ks = np.array([s2[q]["n_sub"] for q in stratum], dtype=float)
            br = np.array([0.0 if s2[q]["correct"] else 1.0 for q in stratum])
            by_k = {}
            for k in sorted(set(ks.tolist())):
                m = ks == k
                by_k[int(k)] = {"n": int(m.sum()),
                                "break_rate": float(br[m].mean())}
            p_trend = trend_test(ks.copy(), br.copy(), rng)

            cells[(model, bench)] = {
                "n_common": len(common),
                "n_broken": len(broken),
                "broken_exec_share": broken_exec / len(broken) if broken else None,
                "s0_fail_exec_share": s0_fail_exec / len(s0_fail) if s0_fail else None,
                "dose_response_by_k": by_k,
                "p_trend_perm": p_trend,
            }
            pooled_dose["ks"].extend(ks.tolist())
            pooled_dose["broken"].extend(br.tolist())
            pooled_modes["broken_exec"] += broken_exec
            pooled_modes["broken_total"] += len(broken)
            pooled_modes["s0fail_exec"] += s0_fail_exec
            pooled_modes["s0fail_total"] += len(s0_fail)

            bx = cells[(model, bench)]["broken_exec_share"]
            sx = cells[(model, bench)]["s0_fail_exec_share"]
            print(f"{model:<24}{bench:<7} broken={len(broken):>4} "
                  f"exec-err {bx*100 if bx is not None else float('nan'):5.1f}% "
                  f"(S0-fail baseline {sx*100 if sx is not None else float('nan'):5.1f}%)  "
                  f"trend p={p_trend:.4f}  by_k={ {k:round(v['break_rate'],2) for k,v in by_k.items()} }")

    # pooled dose-response
    ks = np.array(pooled_dose["ks"]); br = np.array(pooled_dose["broken"])
    pooled_by_k = {}
    for k in sorted(set(ks.tolist())):
        m = ks == k
        pooled_by_k[int(k)] = {"n": int(m.sum()), "break_rate": float(br[m].mean())}
    p_pooled = trend_test(ks.copy(), br.copy(), rng)
    print("\nPOOLED dose-response (S0-correct stratum, all 20 cells):")
    for k, v in pooled_by_k.items():
        print(f"  k={k}: n={v['n']:>5}  P(S2 breaks)={v['break_rate']*100:.1f}%")
    print(f"  trend permutation p = {p_pooled:.2e}")
    be = pooled_modes["broken_exec"] / pooled_modes["broken_total"]
    se = pooled_modes["s0fail_exec"] / pooled_modes["s0fail_total"]
    print(f"\nPOOLED failure-mode split: exec-error share of S2-broken = {be*100:.1f}% "
          f"({pooled_modes['broken_exec']}/{pooled_modes['broken_total']}) "
          f"vs S0-failure baseline {se*100:.1f}% "
          f"({pooled_modes['s0fail_exec']}/{pooled_modes['s0fail_total']})")

    OUT.write_text(json.dumps({
        "experiment_id": "a8",
        "name": "s2_error_propagation_evidence",
        "seed": SEED,
        "n_permutations": N_PERM,
        "per_cell": {f"{m}|{b}": v for (m, b), v in cells.items()},
        "pooled_dose_response": {"by_k": pooled_by_k, "p_trend_perm": p_pooled},
        "pooled_failure_modes": pooled_modes | {
            "broken_exec_share": be, "s0_fail_exec_share": se},
    }, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
