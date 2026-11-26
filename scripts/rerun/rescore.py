"""Re-score every record in one environment (IS revision).

Verdicts stored at generation time were computed on two machines with different SQLite
versions, which changes the last digits of floating-point aggregates and the speed of slow
queries. This script recomputes, from the stored SQL, the fields `error`, `correct`,
`correct_official`, `legacy_error`, `legacy_correct` of every regenerated record (main matrix,
template ablation, Arctic-R1) and rewrites them in place; the values computed at generation
time are preserved under `at_generation`. For records kept from the original runs the verdicts
are written to results/rerun_2026_11/kept_verdicts.jsonl, which build_matrix.py reads.

Queries listed in common.UNSCORABLE are scored as failures for all systems.
Predictions that hit the 30 s execution limit under parallel load are re-executed serially.

Usage: python3 scripts/rerun/rescore.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BENCHMARKS, PANEL, REPO, STRATEGIES, UNSCORABLE, UNSCORABLE_ERROR, gold_result,  # noqa: E402
                    legacy_records, load_queries, score_sql)

OUT = REPO / "results/rerun_2026_11"
ENV = {"sqlite": sqlite3.sqlite_version, "date": dt.date.today().isoformat()}
KEEP = ("error", "correct", "correct_official", "legacy_error", "legacy_correct")
_Q: dict[str, list[dict]] = {}
_G: dict[tuple[str, int], object] = {}


def _q(bench: str, idx: int) -> dict:
    if bench not in _Q:
        _Q[bench] = load_queries(bench)
    return _Q[bench][idx]


def _gold(bench: str, idx: int):
    if (bench, idx) not in _G:
        _G[bench, idx] = gold_result(bench, _q(bench, idx))
    return _G[bench, idx]


def verdict(bench: str, idx: int, sql: str, never_called: bool = False):
    if never_called:
        return UNSCORABLE_ERROR, False, False
    return score_sql(bench, _q(bench, idx), sql, _gold(bench, idx))


def rescore_file(path: str) -> list:
    """Rewrite one jsonl file; return [(path, idx)] of predictions that timed out."""
    rows, slow = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        b, i = r["bench"], r["idx"]
        r.setdefault("at_generation", {k: r.get(k) for k in KEEP if k in r})
        pred = r.get("pred_sql") or ""
        never = (not pred) and str(r["at_generation"].get("error") or "").startswith("gold_error")
        if never and i not in UNSCORABLE[b]:
            raise RuntimeError(f"{path}: idx {i} was never generated and is not listed as unscorable")
        err, strict, official = verdict(b, i, pred, never)
        r.update(error=err, correct=strict, correct_official=official)
        if "legacy_correct" in r:
            lp = r.get("legacy_pred_sql")
            l_err, l_strict, _ = (err, strict, official) if lp is None else verdict(b, i, lp, never)
            r.update(legacy_error=l_err, legacy_correct=l_strict)
            if l_err == "execution_timeout":
                slow.append((path, i))
        if err == "execution_timeout":
            slow.append((path, i))
        r["scored"] = ENV
        rows.append(r)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return slow


def kept_cell(args) -> list[dict]:
    bench, model, strat = args
    new = set()
    f = OUT / f"{bench}__{model}__{strat}.jsonl"
    if f.exists():
        new = {json.loads(line)["idx"] for line in open(f, encoding="utf-8") if line.strip()}
    rows = []
    for i, o in sorted(legacy_records(bench, model, strat).items()):
        if i in new:
            continue
        pred = o.get("pred_sql") or ""
        never = (not pred) and str(o.get("error") or "").startswith("gold_error")
        if never and i not in UNSCORABLE[bench]:
            raise RuntimeError(f"kept {bench}/{model}/{strat}/{i} was never generated and is not listed as unscorable")
        err, strict, official = verdict(bench, i, pred, never)
        rows.append({"bench": bench, "model": model, "strategy": strat, "idx": i, "error": err, "strict": strict,
                     "official": official, "stored_correct": bool(o.get("correct")), "stored_error": o.get("error")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    files = sorted(glob.glob(str(OUT / "*.jsonl"))) + sorted(glob.glob(str(OUT / "ablation/*.jsonl"))) \
        + sorted(glob.glob(str(OUT / "arctic/*.jsonl")))
    files = [f for f in files if not f.endswith("kept_verdicts.jsonl")]
    cells = [(b, m, s) for b in BENCHMARKS for m in PANEL + (["gpt54mini"] if b == "bird" else []) for s in STRATEGIES]
    with Pool(a.jobs) as pool:
        slow = [x for part in pool.map(rescore_file, files, chunksize=1) for x in part]
        kept = [r for part in pool.map(kept_cell, cells, chunksize=1) for r in part]
    # serial second pass for predictions that hit the limit under parallel load
    n_fix = 0
    for path in sorted({p for p, _ in slow}):
        idxs = {i for p, i in slow if p == path}
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        for r in rows:
            if r["idx"] in idxs:
                err, strict, official = verdict(r["bench"], r["idx"], r.get("pred_sql") or "")
                n_fix += err != r["error"]
                r.update(error=err, correct=strict, correct_official=official)
                if "legacy_correct" in r:
                    lp = r.get("legacy_pred_sql")
                    l = (err, strict, official) if lp is None else verdict(r["bench"], r["idx"], lp)
                    r.update(legacy_error=l[0], legacy_correct=l[1])
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in kept:
        if r["error"] == "execution_timeout":
            leg = legacy_records(r["bench"], r["model"], r["strategy"])[r["idx"]]
            err, strict, official = verdict(r["bench"], r["idx"], leg.get("pred_sql") or "")
            n_fix += err != r["error"]
            r.update(error=err, strict=strict, official=official)
    with open(OUT / "kept_verdicts.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(files)} files rescored; {len(slow)} timeouts re-run serially ({n_fix} changed); "
          f"{len(kept)} kept records -> kept_verdicts.jsonl; env {ENV}")


if __name__ == "__main__":
    main()
