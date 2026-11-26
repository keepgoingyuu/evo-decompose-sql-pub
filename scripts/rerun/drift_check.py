"""Reproducibility check: regenerate queries whose original prediction is fully stored and
compare old vs new. Tells whether patching a cell (mixing original and regenerated records)
is defensible for that model.

Usage: python3 scripts/rerun/drift_check.py <dir with *.jsonl from rerun_cells.py --all>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import legacy_records  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().rstrip(";")).lower()


def main() -> None:
    for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
        bench, model, strat = f.stem.split("__")
        old = legacy_records(bench, model, strat)
        n = same_sql = same_out = flip_up = flip_down = 0
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            o = old.get(r["idx"])
            if not o or len(o.get("pred_sql") or "") >= 200 or not o.get("pred_sql"):
                continue  # original prediction not fully stored: nothing to compare
            n += 1
            same_sql += norm(o["pred_sql"]) == norm(r["pred_sql"])
            same_out += bool(o["correct"]) == bool(r["correct"])
            flip_up += (not o["correct"]) and r["correct"]
            flip_down += bool(o["correct"]) and not r["correct"]
        if n:
            print(f"{model:10} {bench:6} {strat}  comparable {n:4d}  identical SQL {100 * same_sql / n:5.1f}%  "
                  f"same verdict {100 * same_out / n:5.1f}%  wrong->right {flip_up}  right->wrong {flip_down}")


if __name__ == "__main__":
    main()
