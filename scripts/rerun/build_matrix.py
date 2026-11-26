"""Corrected per-question matrix for the IS revision: one row per (benchmark, model, strategy, query).

For every query the regenerated record is used when one exists; otherwise the original record
is kept (its prediction is fully stored and untouched by the known harness defects) and
re-scored under the official rule. Every later analysis reads this one file.

Row: {bench, model, strategy, idx, strict, official, src, exec_error, call_count, gen_time,
      in_tok, out_tok, n_sub}
  src         "new" (regenerated 2026-11) | "kept" (original run)
  exec_error  the final SQL did not execute (or no SQL was produced)
  n_sub       S2 only: number of sub-questions the model generated (new records only)

Output: results/rerun_2026_11/analysis/matrix.jsonl
Usage: python3 scripts/rerun/build_matrix.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCHMARKS, PANEL, REPO, STRATEGIES, legacy_records, load_queries  # noqa: E402

sys.path.insert(0, str(REPO))
from src.utils.sql_utils import compare_results_bird, compare_results_spider, execute_sql  # noqa: E402

OUT = REPO / "results/rerun_2026_11"
CELLS = [(b, m) for m in PANEL for b in BENCHMARKS] + [("bird", "gpt54mini")]


def cell_rows(args: tuple[str, str]) -> list[dict]:
    bench, model = args
    qs = load_queries(bench)
    gold_cache: dict[int, object] = {}
    rows = []
    for strat in STRATEGIES:
        old = legacy_records(bench, model, strat)
        new = {}
        f = OUT / f"{bench}__{model}__{strat}.jsonl"
        if f.exists():
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    new[r["idx"]] = r
        for q in qs:
            i = q["idx"]
            base = {"bench": bench, "model": model, "strategy": strat, "idx": i}
            if i in new:
                r = new[i]
                steps = r.get("steps") or []
                tok = lambda k: sum(s.get(k) or 0 for s in steps) or None  # noqa: E731
                rows.append({**base, "strict": bool(r["correct"]), "official": bool(r["correct_official"]),
                             "src": "new", "exec_error": bool(r.get("error")),
                             "call_count": r.get("call_count"), "gen_time": r.get("gen_time"),
                             "in_tok": tok("in_tok"), "out_tok": tok("out_tok"),
                             "n_sub": sum(str(s.get("step", "")).startswith("sub_sql") for s in steps)
                             if strat == "s2" else None})
                continue
            o = old.get(i)
            if o is None:
                raise RuntimeError(f"no record for {bench}/{model}/{strat}/{i}")
            strict = bool(o["correct"])
            official = strict
            pred = o.get("pred_sql") or ""
            if not o.get("error") and pred:
                if i not in gold_cache:
                    gold_cache[i] = execute_sql(q["db_path"], q["gold_sql"])[0]
                g = gold_cache[i]
                p, _ = execute_sql(q["db_path"], pred)
                if g is not None and p is not None:
                    official = (compare_results_bird(g, p) if BENCHMARKS[bench]["family"] == "bird"
                                else compare_results_spider(g, p, q["gold_sql"]))
            rows.append({**base, "strict": strict, "official": bool(official), "src": "kept",
                         "exec_error": bool(o.get("error")), "call_count": o.get("call_count"),
                         "gen_time": o.get("gen_time"), "in_tok": None, "out_tok": None, "n_sub": None})
    return rows


def main() -> None:
    out = OUT / "analysis"
    out.mkdir(exist_ok=True)
    with Pool(6) as pool:
        parts = pool.map(cell_rows, CELLS)
    n = 0
    with open(out / "matrix.jsonl", "w", encoding="utf-8") as fh:
        for rows in parts:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                n += 1
    print(f"{n} rows -> {out / 'matrix.jsonl'}")


if __name__ == "__main__":
    main()
