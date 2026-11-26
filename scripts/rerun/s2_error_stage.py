"""Where does question decomposition (S2) fail? (IS revision, R3 point 3)

For every query that a model answers correctly with direct generation (S0) but wrongly with
S2, the stored S2 trace (decompose -> sub-SQL_1..k -> merge) is replayed offline:
each sub-question's SQL is re-extracted from its raw response and executed.

Stage assigned to each broken query (first applicable):
  no_decomposition  the model produced <=1 sub-question, S2 fell back to direct generation
  sub_sql           at least one sub-question's SQL fails to execute
  merge_invalid     every sub-SQL executes, but the merged final SQL does not
  merge_wrong       every sub-SQL executes and the merged SQL executes, but returns the wrong rows
Only regenerated records (full traces) are used; S0 verdicts come from the corrected matrix.

Usage: python3 scripts/rerun/s2_error_stage.py [models...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCHMARKS, REPO, legacy_records, load_queries  # noqa: E402

sys.path.insert(0, str(REPO))
from src.utils.sql_utils import clean_sql, execute_sql  # noqa: E402

OUT = REPO / "results/rerun_2026_11"


_M: dict = {}


def s0_verdicts(model, bench):
    """S0 strict verdicts from the corrected matrix."""
    if not _M:
        for line in open(OUT / "analysis/matrix.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["strategy"] == "s0":
                _M.setdefault((r["model"], r["bench"]), {})[r["idx"]] = bool(r["strict"])
    return _M[model, bench]


def main() -> None:
    models = sys.argv[1:] or ["gpt41", "deepseek", "gemini", "qwen25c7b", "gemma4"]
    total = Counter()
    print(f"{'model':10} {'bench':6} {'broken':>6}  no_decomp  sub_sql  merge_invalid  merge_wrong   (traced/broken)")
    for model in models:
        for bench in BENCHMARKS:
            f = OUT / f"{bench}__{model}__s2.jsonl"
            if not f.exists():
                continue
            qs = load_queries(bench)
            s0 = s0_verdicts(model, bench)
            c = Counter()
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("correct") or not s0.get(r["idx"]):
                    continue
                c["broken"] += 1
                steps = r.get("steps", [])
                if any(s.get("step") == "decompose_fallback" for s in steps):
                    c["no_decomposition"] += 1
                    continue
                db = qs[r["idx"]]["db_path"]
                subs = [s for s in steps if str(s.get("step", "")).startswith("sub_sql")]
                if any(execute_sql(db, clean_sql(s.get("raw", "")), lenient_text=BENCHMARKS[bench]["family"] == "spider")[1]
                       for s in subs):
                    c["sub_sql"] += 1
                elif r.get("error"):
                    c["merge_invalid"] += 1
                else:
                    c["merge_wrong"] += 1
            if c["broken"]:
                b = c["broken"]
                print(f"{model:10} {bench:6} {b:6d}  " + "  ".join(
                    f"{100 * c[k] / b:8.1f}%" for k in ("no_decomposition", "sub_sql", "merge_invalid", "merge_wrong")))
                total.update(c)
    b = total["broken"]
    print(f"\n{'ALL':17} {b:6d}  " + "  ".join(f"{100 * total[k] / b:8.1f}%"
                                         for k in ("no_decomposition", "sub_sql", "merge_invalid", "merge_wrong")))
    print("note: only queries whose S2 run was regenerated (full trace) are included")


if __name__ == "__main__":
    main()
