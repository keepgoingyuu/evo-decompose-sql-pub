"""All matrix-level statistics of the IS revision, computed from analysis/matrix.jsonl.

Sections (each written to analysis/stats.json and printed):
  main        accuracy per (benchmark, model, strategy) under both scoring rules
  contrast    every strategy vs S0 per (model, benchmark): drop, broken/rescued, exact McNemar, Holm
  pooled      per benchmark, pooled over the five panel models: question-clustered drop,
              bootstrap 95% CI, sign-flip permutation p; model-level paired t and Cohen's d_z
  ranking     per-benchmark ranking of S1-S4 by mean benefit; Kendall's W with exact permutation p
  fit         log-linear capability-benefit fit on BIRD
  depth       S2 breakage probability by number of generated sub-questions (S0-correct stratum)
  execfail    share of execution failures among S2-broken queries and among S0 failures
  cost        mean calls and tokens per query per strategy
  gate        selector reference points: always-S0, always-S1, execution gate S0->S1, oracles

Usage: python3 scripts/rerun/revision_stats.py [--rule strict|official]
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PANEL, REPO, STRATEGIES  # noqa: E402

AN = REPO / "results/rerun_2026_11/analysis"
BENCHES = ["bird", "en", "zh-tw", "ja"]
SPIDER = ["en", "zh-tw", "ja"]
RNG = np.random.default_rng(42)


def load() -> dict:
    """{(bench, model, strategy): {idx: row}}"""
    m = defaultdict(dict)
    for line in open(AN / "matrix.jsonl", encoding="utf-8"):
        r = json.loads(line)
        m[r["bench"], r["model"], r["strategy"]][r["idx"]] = r
    return m


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def vec(M, b, m, s, rule):
    d = M[b, m, s]
    return np.array([d[i][rule] for i in sorted(d)], dtype=float)


def kendall_w(ranks: np.ndarray) -> float:
    m, n = ranks.shape
    s = ((ranks.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return 12 * s / (m ** 2 * (n ** 3 - n))


def main() -> None:
    rule = sys.argv[sys.argv.index("--rule") + 1] if "--rule" in sys.argv else "strict"
    M = load()
    models = PANEL + ["gpt54mini"]
    out: dict = {"rule": rule}
    acc = lambda b, m, s, r=rule: 100 * vec(M, b, m, s, r).mean()  # noqa: E731

    # --- main -------------------------------------------------------------
    out["main"] = {f"{b}|{m}|{s}": {"strict": round(acc(b, m, s, "strict"), 2),
                                    "official": round(acc(b, m, s, "official"), 2),
                                    "n": len(M[b, m, s]),
                                    "n_new": sum(r["src"] == "new" for r in M[b, m, s].values())}
                   for (b, m, s) in M}
    print(f"== main ({rule}) ==")
    for b in BENCHES:
        for m in models:
            if (b, m, "s0") in M:
                print(f"  {b:6}{m:10}" + " ".join(f"{acc(b, m, s):6.2f}" for s in STRATEGIES))

    # --- contrast ---------------------------------------------------------
    rows = []
    for b in BENCHES:
        for m in models:
            if (b, m, "s0") not in M:
                continue
            v0 = vec(M, b, m, "s0", rule)
            for s in STRATEGIES[1:]:
                v = vec(M, b, m, s, rule)
                bb, cc = int(((v0 == 1) & (v == 0)).sum()), int(((v0 == 0) & (v == 1)).sum())
                rows.append({"bench": b, "model": m, "strategy": s, "drop": round(100 * (v0.mean() - v.mean()), 2),
                             "broken": bb, "rescued": cc, "p": mcnemar(bb, cc)})
    for s in STRATEGIES[1:]:
        sel = [r for r in rows if r["strategy"] == s and r["model"] in PANEL]
        for r, a in zip(sel, holm([r["p"] for r in sel])):
            r["p_holm"] = a
    out["contrast"] = rows
    print("\n== S0 vs S2 per cell (drop = S0 - S2; Holm over the 20 panel cells) ==")
    for r in rows:
        if r["strategy"] == "s2":
            print(f"  {r['bench']:6}{r['model']:10} {r['drop']:+6.2f}  b={r['broken']:3d} c={r['rescued']:3d} "
                  f"p={r['p']:.2g}  holm={r.get('p_holm', float('nan')):.2g}")
    print("\n== cells significant after Holm (panel, 20 cells per strategy): worse / better than S0 ==")
    for s in STRATEGIES[1:]:
        sel = [r for r in rows if r["strategy"] == s and r["model"] in PANEL]
        worse = sum(r["p_holm"] < .05 and r["drop"] > 0 for r in sel)
        better = sum(r["p_holm"] < .05 and r["drop"] < 0 for r in sel)
        neg = sum(r["drop"] > 0 for r in sel)
        print(f"  {s}: below S0 in {neg}/20 cells; significant worse {worse}, significant better {better}")

    # --- pooled -----------------------------------------------------------
    pooled = []
    for b in BENCHES:
        for s in STRATEGIES[1:]:
            d = np.mean([vec(M, b, m, "s0", rule) - vec(M, b, m, s, rule) for m in PANEL], axis=0)
            obs = d.mean()
            boot = RNG.choice(d, size=(10000, len(d)), replace=True).mean(1)
            flips = RNG.choice([-1.0, 1.0], size=(20000, len(d)))
            p_perm = (np.abs((flips * d).mean(1)) >= abs(obs) - 1e-12).mean()
            per_model = np.array([acc(b, m, "s0") - acc(b, m, s) for m in PANEL])
            t, p_t = stats.ttest_1samp(per_model, 0)
            pooled.append({"bench": b, "strategy": s, "drop": round(100 * obs, 2),
                           "ci": [round(100 * np.percentile(boot, 2.5), 2), round(100 * np.percentile(boot, 97.5), 2)],
                           "p_perm": max(float(p_perm), 1 / 20000), "t": round(float(t), 2), "p_t": float(p_t),
                           "dz": round(float(per_model.mean() / per_model.std(ddof=1)), 2),
                           "per_model": [round(float(x), 2) for x in per_model]})
    out["pooled"] = pooled
    print("\n== pooled over 5 panel models (drop = S0 - Si) ==")
    for r in pooled:
        print(f"  {r['bench']:6}{r['strategy']} {r['drop']:+6.2f} CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] "
              f"perm p={r['p_perm']:.2g} | t={r['t']:+.2f} p={r['p_t']:.3f} dz={r['dz']:+.2f} | {r['per_model']}")

    # --- ranking ----------------------------------------------------------
    benefit = {b: {s: float(np.mean([acc(b, m, s) - acc(b, m, "s0") for m in PANEL])) for s in STRATEGIES[1:]}
               for b in BENCHES}
    rank = {b: stats.rankdata([-benefit[b][s] for s in STRATEGIES[1:]]) for b in BENCHES}
    perms = [np.array(p) for p in itertools.permutations([1, 2, 3, 4])]

    def w_exact(bs):
        R = np.array([rank[b] for b in bs])
        w = kendall_w(R)
        cnt = tot = 0
        for combo in itertools.product(perms, repeat=len(bs)):
            tot += 1
            cnt += kendall_w(np.array(combo)) >= w - 1e-12
        return round(float(w), 3), cnt / tot

    out["ranking"] = {"benefit": {b: {s: round(v, 2) for s, v in benefit[b].items()} for b in BENCHES},
                      "rank": {b: [int(x) for x in rank[b]] for b in BENCHES},
                      "W_spider3": w_exact(SPIDER), "W_all4": w_exact(BENCHES),
                      "tau": {f"{a}|{b}": round(float(stats.kendalltau(rank[a], rank[b])[0]), 2)
                              for a, b in itertools.combinations(BENCHES, 2)}}
    print("\n== ranking of S1..S4 by mean benefit ==")
    for b in BENCHES:
        order = sorted(STRATEGIES[1:], key=lambda s: -benefit[b][s])
        print(f"  {b:6}" + " > ".join(f"{s}({benefit[b][s]:+.2f})" for s in order))
    print(f"  Kendall W, 3 Spider-family languages: {out['ranking']['W_spider3']}; all 4 benchmarks: {out['ranking']['W_all4']}")
    print(f"  tau: {out['ranking']['tau']}")

    # --- fit --------------------------------------------------------------
    fit = {}
    c = np.array([acc("bird", m, "s0") for m in PANEL])
    for s in STRATEGIES[1:]:
        y = np.array([acc("bird", m, s) for m in PANEL]) - c
        res = stats.linregress(-np.log(c), y)
        fit[s] = {"alpha": round(res.slope, 2), "beta": round(res.intercept, 2), "r2": round(res.rvalue ** 2, 3),
                  "p": round(res.pvalue, 4)}
    out["fit"] = fit
    print("\n== capability-benefit fit (BIRD, N=5) ==")
    for s, v in fit.items():
        print(f"  {s} {v}")
    # rank correlation between capability and S2 drop over all 20 panel cells
    caps = [acc(b, m, "s0") for b in BENCHES for m in PANEL]
    drops = [acc(b, m, "s0") - acc(b, m, "s2") for b in BENCHES for m in PANEL]
    out["cap_vs_s2drop"] = {b: [round(float(x), 3) for x in stats.spearmanr(
        [acc(b, m, "s0") for m in PANEL], [acc(b, m, "s0") - acc(b, m, "s2") for m in PANEL])] for b in BENCHES}
    print(f"  Spearman(S0 accuracy, S2 drop) per benchmark (rho, p): {out['cap_vs_s2drop']}")
    _ = caps, drops

    # --- depth ------------------------------------------------------------
    depth = defaultdict(lambda: [0, 0])
    for b in BENCHES:
        for m in PANEL:
            for i, r2 in M[b, m, "s2"].items():
                if r2["src"] == "new" and r2["n_sub"] is not None and M[b, m, "s0"][i][rule]:
                    k = min(r2["n_sub"], 5)
                    depth[k][0] += 1
                    depth[k][1] += not r2[rule]
    out["depth"] = {str(k): {"n": v[0], "broken_pct": round(100 * v[1] / v[0], 1)} for k, v in sorted(depth.items())}
    print(f"\n== S2 breakage by number of sub-questions (S0-correct, regenerated S2 records; 5 = 5+) ==\n  {out['depth']}")

    # --- execfail ---------------------------------------------------------
    ef = {}
    for m in PANEL:
        br = [M[b, m, "s2"][i] for b in BENCHES for i in M[b, m, "s2"]
              if M[b, m, "s0"][i][rule] and not M[b, m, "s2"][i][rule]]
        f0 = [r for b in BENCHES for r in M[b, m, "s0"].values() if not r[rule]]
        ef[m] = {"s2_broken": len(br), "s2_broken_exec_pct": round(100 * np.mean([r["exec_error"] for r in br]), 1),
                 "s0_fail": len(f0), "s0_fail_exec_pct": round(100 * np.mean([r["exec_error"] for r in f0]), 1)}
    out["execfail"] = ef
    print(f"\n== execution-failure share ==")
    for m, v in ef.items():
        print(f"  {m:10} {v}")

    # --- cost -------------------------------------------------------------
    cost = {}
    for b in BENCHES:
        for m in models:
            for s in STRATEGIES:
                if (b, m, s) not in M:
                    continue
                new = [r for r in M[b, m, s].values() if r["src"] == "new"]
                calls = [r["call_count"] for r in M[b, m, s].values() if r["call_count"]]
                tin = [r["in_tok"] for r in new if r["in_tok"]]
                tout = [r["out_tok"] for r in new if r["out_tok"]]
                gt = [r["gen_time"] for r in new if r["gen_time"]]
                cost[f"{b}|{m}|{s}"] = {"calls": round(float(np.mean(calls)), 2) if calls else None,
                                        "in_tok": round(float(np.mean(tin))) if tin else None,
                                        "out_tok": round(float(np.mean(tout))) if tout else None,
                                        "gen_time_p50_new": round(float(np.median(gt)), 2) if gt else None,
                                        "n_new": len(new)}
    out["cost"] = cost
    print("\n== mean calls per query (all records with a call count) ==")
    for b in BENCHES:
        for m in models:
            if (b, m, "s0") in M:
                print(f"  {b:6}{m:10}" + " ".join(f"{cost[f'{b}|{m}|{s}']['calls'] or float('nan'):5.2f}" for s in STRATEGIES))

    # --- gate -------------------------------------------------------------
    gate = {}
    for b in BENCHES:
        for scope in PANEL + ["pooled"]:
            ms = PANEL if scope == "pooled" else [scope]
            c0 = np.concatenate([vec(M, b, m, "s0", rule) for m in ms])
            c1 = np.concatenate([vec(M, b, m, "s1", rule) for m in ms])
            e0 = np.concatenate([np.array([M[b, m, "s0"][i]["exec_error"] for i in sorted(M[b, m, "s0"])], dtype=float)
                                 for m in ms])
            allc = np.max([np.concatenate([vec(M, b, m, s, rule) for m in ms]) for s in STRATEGIES], axis=0)
            g = np.where(e0 == 1, c1, c0)
            gate[f"{b}|{scope}"] = {"always_s0": round(100 * c0.mean(), 2), "always_s1": round(100 * c1.mean(), 2),
                                    "exec_gate": round(100 * g.mean(), 2), "trigger_pct": round(100 * e0.mean(), 2),
                                    "gate_calls": round(float(1 + 2 * e0.mean()), 2),
                                    "oracle_s0_or_s1": round(100 * np.maximum(c0, c1).mean(), 2),
                                    "oracle_all": round(100 * allc.mean(), 2), "n": len(c0)}
    out["gate"] = gate
    print("\n== selector reference points (pooled over 5 panel models) ==")
    for b in BENCHES:
        print(f"  {b:6} {gate[f'{b}|pooled']}")
    print("  per model, BIRD:")
    for m in PANEL:
        print(f"    {m:10} {gate[f'bird|{m}']}")

    json.dump(out, open(AN / f"stats_{rule}.json", "w"), indent=1)
    print(f"\n-> {AN / f'stats_{rule}.json'}")


if __name__ == "__main__":
    main()
