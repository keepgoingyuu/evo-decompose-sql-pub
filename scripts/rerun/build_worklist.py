"""Decide which stored records of the original matrix must be regenerated (IS revision, 2026-11).

A record is kept only if its stored prediction is complete and it was not touched by a
known harness defect. It is regenerated when any of these holds:

  strip     stored error is the signature of the WITH-stripping extraction defect
  trunc     stored pred_sql is cut at 200 characters (cannot be re-scored)
  budget    stored error is `incomplete input` / `no SQL` (possible output-budget truncation)
  missing   the cell has no stored record for the query
  protocol  whole cell: the cell was produced by a different harness than the rest of
            the matrix (API models: BIRD S0 via phase1_*_baseline.py with its own
            extractor and max_tokens=1024; ZH-TW S0 via a Chinese-specific prompt)

Output: results/rerun_2026_11/worklist.json  {"bench|model|strategy": {"idx": [...], "why": {...}}}

Usage: venv/bin/python scripts/rerun/build_worklist.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (API_MODELS, BENCHMARKS, BUDGET_ERRORS, PANEL, REPO, STRATEGIES, STRIP_SIGNATURE,  # noqa: E402
                    legacy_records, load_queries)

OUT = REPO / "results/rerun_2026_11/worklist.json"


def reason(rec: dict | None) -> str | None:
    if rec is None:
        return "missing"
    err = rec.get("error") or ""
    if STRIP_SIGNATURE in err:
        return "strip"
    if len(rec.get("pred_sql") or "") >= 200:
        return "trunc"
    if any(e in err for e in BUDGET_ERRORS):
        return "budget"
    return None


def main() -> None:
    work, total = {}, Counter()
    cells = [(b, m, s) for b in BENCHMARKS for m in PANEL for s in STRATEGIES]
    cells += [("bird", "gpt54mini", s) for s in STRATEGIES]
    for bench, model, strat in cells:
        n = len(load_queries(bench))
        whole = model in API_MODELS and strat == "s0" and bench in ("bird", "zh-tw")
        try:
            legacy = legacy_records(bench, model, strat)
        except FileNotFoundError:
            legacy = {}
        why = Counter()
        idxs = []
        for i in range(n):
            r = "protocol" if whole else reason(legacy.get(i))
            if r:
                idxs.append(i)
                why[r] += 1
        work[f"{bench}|{model}|{strat}"] = {"idx": idxs, "n": n, "why": dict(why)}
        total.update(why)
        total["_records"] += n
        total["_rerun"] += len(idxs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(work, open(OUT, "w"), indent=1)

    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"records {total['_records']}  to regenerate {total['_rerun']} "
          f"({100 * total['_rerun'] / total['_records']:.1f}%)  by reason: "
          + ", ".join(f"{k}={v}" for k, v in sorted(total.items()) if not k.startswith("_")))
    print(f"\n{'model':10} {'bench':6} " + " ".join(f"{s:>6}" for s in STRATEGIES) + "   total")
    for model in PANEL + ["gpt54mini"]:
        msum = 0
        for bench in BENCHMARKS:
            if model == "gpt54mini" and bench != "bird":
                continue
            row = [len(work[f"{bench}|{model}|{s}"]["idx"]) for s in STRATEGIES]
            msum += sum(row)
            print(f"{model:10} {bench:6} " + " ".join(f"{x:6d}" for x in row) + f"   {sum(row):5d}")
        print(f"{model:10} {'all':6} {'':34}   {msum:5d}")


if __name__ == "__main__":
    main()
