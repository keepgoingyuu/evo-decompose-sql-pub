"""Merge original and regenerated records for one model and report accuracy under both rules.

For every query the regenerated record is used when one exists; otherwise the original record
is kept (its prediction is fully stored) and re-scored here under the official rule.

Usage: python3 scripts/rerun/merge_report.py gpt41 [--json out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCHMARKS, REPO, STRATEGIES, legacy_records, load_queries  # noqa: E402

sys.path.insert(0, str(REPO))
from src.utils.sql_utils import (compare_results, compare_results_bird, compare_results_spider,  # noqa: E402
                                 execute_sql)

OUT = REPO / "results/rerun_2026_11"


def merged(model: str, bench: str, strat: str) -> dict[int, dict]:
    """{idx: {strict, official, legacy_strict, src}} for one cell."""
    qs = load_queries(bench)
    old = legacy_records(bench, model, strat)
    new = {}
    f = OUT / f"{bench}__{model}__{strat}.jsonl"
    if f.exists():
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            new[r["idx"]] = r
    res = {}
    gold_cache = {}
    for q in qs:
        i = q["idx"]
        o = old.get(i)
        if i in new:
            r = new[i]
            res[i] = {"strict": bool(r["correct"]), "official": bool(r["correct_official"]),
                      "old": bool(o["correct"]) if o else None, "src": "new", "error": r.get("error")}
            continue
        if o is None:
            res[i] = {"strict": None, "official": None, "old": None, "src": "missing"}
            continue
        strict = bool(o["correct"])
        official = strict
        pred = o.get("pred_sql") or ""
        if not o.get("error") and pred and len(pred) < 200:
            if i not in gold_cache:
                gold_cache[i] = execute_sql(q["db_path"], q["gold_sql"])[0]
            g = gold_cache[i]
            p, err = execute_sql(q["db_path"], pred)
            if g is not None and p is not None:
                official = (compare_results_bird(g, p) if BENCHMARKS[bench]["family"] == "bird"
                            else compare_results_spider(g, p, q["gold_sql"]))
        res[i] = {"strict": strict, "official": official, "old": strict, "src": "kept", "error": o.get("error")}
    return res


def mcnemar(b: int, c: int) -> float:
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def main() -> None:
    model = sys.argv[1]
    cells = {}
    for b in BENCHMARKS:
        for s in STRATEGIES:
            cells[b, s] = merged(model, b, s)
    acc = lambda d, k: 100 * sum(bool(v[k]) for v in d.values()) / len(d)  # noqa: E731
    print(f"{model}: accuracy (%)  reported -> corrected strict | corrected official     [regenerated / missing]")
    for b in BENCHMARKS:
        for s in STRATEGIES:
            d = cells[b, s]
            n_new = sum(v["src"] == "new" for v in d.values())
            n_miss = sum(v["src"] == "missing" for v in d.values())
            print(f"  {b:6} {s}  {acc(d, 'old'):6.2f} -> {acc(d, 'strict'):6.2f} | {acc(d, 'official'):6.2f}"
                  f"     [{n_new}/{n_miss}]")
    print("\nS0 vs S2 on the corrected data (strict rule): drop pp, broken b / rescued c, exact McNemar p ; official-rule drop")
    for b in BENCHMARKS:
        d0, d2 = cells[b, "s0"], cells[b, "s2"]
        bb = sum(d0[i]["strict"] and not d2[i]["strict"] for i in d0)
        cc = sum(d2[i]["strict"] and not d0[i]["strict"] for i in d0)
        old_drop = acc(d0, "old") - acc(d2, "old")
        print(f"  {b:6} reported {old_drop:+6.2f} -> corrected {acc(d0, 'strict') - acc(d2, 'strict'):+6.2f}  "
              f"(b={bb}, c={cc}, p={mcnemar(bb, cc):.2g})   official {acc(d0, 'official') - acc(d2, 'official'):+6.2f}")
    print("\nbest strategy per benchmark: reported -> corrected(strict) / corrected(official)")
    for b in BENCHMARKS:
        best = lambda k: max(STRATEGIES, key=lambda s: acc(cells[b, s], k))  # noqa: E731
        print(f"  {b:6} {best('old')} -> {best('strict')} / {best('official')}")
    if "--json" in sys.argv:
        out = {f"{b}|{s}": {k: round(acc(cells[b, s], k), 2) for k in ("old", "strict", "official")}
               for b in BENCHMARKS for s in STRATEGIES}
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
