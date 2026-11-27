"""S1 template ablation for Gemma-3n-E4B, run on a second machine (IS revision: R1-W3).

The ablation of the other models takes S0 and the canonical template (V1) from the matrix.
For Gemma-3n-E4B the ablation was run on an Apple-silicon host (same weights, same Ollama
build and quantisation as the matrix, different hardware), so S0 and V1 were re-run there too
(scripts/rerun/ablation_cells.py ... s0 v1 / s1 v1) and every arm of each paired comparison
comes from that one machine. This script

  - re-scores every record from its stored SQL with the scoring function used for the matrix
    (common.score_sql: both rules, lenient text decoding on the Spider family, UNSCORABLE);
  - compares each S1 template with S0 on the same 400 queries (accuracy, drop, exact McNemar);
  - reports how far the second machine reproduces the matrix on the two arms they share
    (S0 and S1-V1): share of identical verdicts and of identical SQL strings.

Output: results/rerun_2026_11/analysis/ablation_mac.json
Usage: venv/bin/python scripts/rerun/revision_ablation_mac.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, gold_result, load_queries, score_sql  # noqa: E402

AB = REPO / "results/rerun_2026_11/ablation_mac"
OUT = REPO / "results/rerun_2026_11"
MODEL = "gemma4"
ARMS = [("s0", "v1"), ("s1", "v1"), ("s1", "v2"), ("s1", "v3")]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    out, tests = {}, []
    for bench in ["en", "zh-tw", "ja"]:
        queries = load_queries(bench)
        arms = {}
        for strat, ver in ARMS:
            recs = {}
            for line in open(AB / f"{bench}__{MODEL}__{strat}-{ver}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                q = queries[r["idx"]]
                err, strict, official = score_sql(bench, q, r["pred_sql"], gold_result(bench, q))
                recs[r["idx"]] = {"strict": strict, "official": official, "sql": r["pred_sql"],
                                  "calls": r["call_count"], "exec_error": err is not None}
            arms[strat, ver] = recs
        ids = sorted(set.intersection(*[set(v) for v in arms.values()]))
        n = len(ids)
        s0 = arms["s0", "v1"]
        cell = {"n": n, "s0": {rule: round(100 * sum(s0[i][rule] for i in ids) / n, 2)
                               for rule in ("strict", "official")}, "variants": {}}
        for ver in ("v1", "v2", "v3"):
            v = arms["s1", ver]
            row = {}
            for rule in ("strict", "official"):
                b = sum(s0[i][rule] and not v[i][rule] for i in ids)
                c = sum(v[i][rule] and not s0[i][rule] for i in ids)
                acc = 100 * sum(v[i][rule] for i in ids) / n
                row[rule] = {"acc": round(acc, 2), "drop": round(cell["s0"][rule] - acc, 2), "b": b, "c": c,
                             "p": mcnemar(b, c)}
                tests.append((bench, ver, rule))
            row["calls"] = round(sum(v[i]["calls"] for i in ids) / n, 2)
            cell["variants"][ver] = row
        # reproduction of the matrix (DGX) on the shared arms
        repro = {}
        for strat in ("s0", "s1"):
            dgx = {}
            for line in open(OUT / f"{bench}__{MODEL}__{strat}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                dgx[r["idx"]] = r
            mac = arms[strat, "v1"]
            q_ok = [i for i in ids if i in dgx]
            repro[strat] = {
                "n": len(q_ok),
                "same_verdict_strict": round(100 * sum(bool(dgx[i]["correct"]) == mac[i]["strict"] for i in q_ok) / len(q_ok), 1),
                "same_verdict_official": round(100 * sum(bool(dgx[i]["correct_official"]) == mac[i]["official"] for i in q_ok) / len(q_ok), 1),
                "same_sql": round(100 * sum(dgx[i]["pred_sql"] == mac[i]["sql"] for i in q_ok) / len(q_ok), 1),
                "acc_dgx_strict": round(100 * sum(bool(dgx[i]["correct"]) for i in q_ok) / len(q_ok), 2),
                "acc_mac_strict": round(100 * sum(mac[i]["strict"] for i in q_ok) / len(q_ok), 2),
                "acc_dgx_official": round(100 * sum(bool(dgx[i]["correct_official"]) for i in q_ok) / len(q_ok), 2),
                "acc_mac_official": round(100 * sum(mac[i]["official"] for i in q_ok) / len(q_ok), 2),
            }
        cell["reproduces_matrix"] = repro
        out[bench] = cell
    # Holm over the nine S1 cells of this model, per rule
    for rule in ("strict", "official"):
        keys = [(b, v) for b in out for v in ("v1", "v2", "v3")]
        adj = holm([out[b]["variants"][v][rule]["p"] for b, v in keys])
        for (b, v), a in zip(keys, adj):
            out[b]["variants"][v][rule]["p_holm9"] = a
    json.dump(out, open(OUT / "analysis/ablation_mac.json", "w"), indent=1)
    for bench, cell in out.items():
        print(f"{bench}  n={cell['n']}  S0 strict {cell['s0']['strict']:.2f}  official {cell['s0']['official']:.2f}")
        for ver, row in cell["variants"].items():
            s, o = row["strict"], row["official"]
            print(f"   {ver}: acc {s['acc']:.2f} drop {s['drop']:+.2f} b/c {s['b']}/{s['c']} p={s['p']:.3g} "
                  f"(holm {s['p_holm9']:.3g}) | official drop {o['drop']:+.2f} b/c {o['b']}/{o['c']} p={o['p']:.3g} "
                  f"(holm {o['p_holm9']:.3g}) | calls {row['calls']}")
        for strat, r in cell["reproduces_matrix"].items():
            print(f"   reproduces matrix {strat}: verdict {r['same_verdict_strict']}% / {r['same_verdict_official']}%  "
                  f"sql {r['same_sql']}%  acc dgx {r['acc_dgx_strict']} mac {r['acc_mac_strict']}")


if __name__ == "__main__":
    main()
