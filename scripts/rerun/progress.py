"""Progress of the rerun against the worklist. Exit code 1 while anything is pending.

Usage: python3 scripts/rerun/progress.py [--models qwen25c7b gemma4] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

OUT = REPO / "results/rerun_2026_11"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    work = json.load(open(OUT / "worklist.json"))
    pending_total = 0
    for key, cell in work.items():
        bench, model, strat = key.split("|")
        if args.models and model not in args.models:
            continue
        f = OUT / f"{bench}__{model}__{strat}.jsonl"
        done = set()
        if f.exists():
            done = {json.loads(line)["idx"] for line in open(f, encoding="utf-8") if line.strip()}
        pending = len(set(cell["idx"]) - done)
        pending_total += pending
        if not args.quiet and cell["idx"]:
            print(f"{model:10} {bench:6} {strat}  {len(cell['idx']) - pending:5d}/{len(cell['idx']):5d}")
    if not args.quiet:
        print(f"pending: {pending_total}")
    sys.exit(1 if pending_total else 0)


if __name__ == "__main__":
    main()
