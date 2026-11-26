"""Cross-check the BIRD official-rule verdicts against the official BIRD evaluation script (IS revision, R3-5).

The official script is `bird/llm/src/evaluation.py` of the BIRD release
(https://github.com/AlibabaResearch/DAMO-ConvAI). Its `execute_model` function (execute the
predicted and the gold query, compare the two results as sets of rows, 30 s limit) is imported
unmodified and applied to every BIRD prediction of the matrix: regenerated records, records kept
from the original runs, and the two Arctic-R1 runs. The verdict is compared with the
official-rule verdict used in the paper (`official` in analysis/matrix.jsonl, `correct_official`
in the Arctic files).

Output: results/rerun_2026_11/bird_official_check.json
Usage:  python3 scripts/rerun/bird_official_check.py --official /path/to/evaluation.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PANEL, REPO, STRATEGIES, legacy_records, load_queries  # noqa: E402

OUT = REPO / "results/rerun_2026_11"
_EV = None
_PATH = None


def _init(path: str) -> None:
    global _EV, _PATH
    _PATH = path
    spec = importlib.util.spec_from_file_location("bird_official_evaluation", path)
    _EV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_EV)


def _official(task):
    key, pred, gold, db = task
    return key, int(_EV.execute_model(pred, gold, db, 0, 30.0)["res"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="path to the official BIRD evaluation.py")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    qs = {q["idx"]: q for q in load_queries("bird")}
    ours, src = {}, {}
    for line in open(OUT / "analysis/matrix.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["bench"] == "bird":
            ours[r["model"], r["strategy"], r["idx"]] = bool(r["official"])
            src[r["model"], r["strategy"], r["idx"]] = r["src"]
    tasks = []
    for m in PANEL + ["gpt54mini"]:
        for s in STRATEGIES:
            new = {}
            f = OUT / f"bird__{m}__{s}.jsonl"
            if f.exists():
                new = {(r := json.loads(line))["idx"]: r for line in open(f, encoding="utf-8") if line.strip()}
            old = legacy_records("bird", m, s)
            for i, q in qs.items():
                pred = (new[i] if i in new else old[i]).get("pred_sql") or ""
                tasks.append(((m, s, i), pred, q["gold_sql"], q["db_path"]))
    for tag in ("native", "uniform"):
        for line in open(OUT / f"arctic/bird__arctic__{tag}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                k = ("arctic", tag, r["idx"])
                ours[k], src[k] = bool(r["correct_official"]), "new"
                tasks.append((k, r.get("pred_sql") or "", qs[r["idx"]]["gold_sql"], qs[r["idx"]]["db_path"]))
    with Pool(a.jobs, initializer=_init, initargs=(a.official,)) as pool:
        off = dict(pool.imap_unordered(_official, tasks, chunksize=50))
    n = len(off)
    agree = sum(bool(off[k]) == ours[k] for k in off)
    dis = [{"model": k[0], "strategy": k[1], "idx": k[2], "src": src[k], "official_script": bool(off[k]), "ours": ours[k]}
           for k in sorted(off) if bool(off[k]) != ours[k]]
    by = defaultdict(Counter)
    for d in dis:
        by[d["src"]]["official_only" if d["official_script"] else "ours_only"] += 1
    res = {"official_script": "bird/llm/src/evaluation.py (AlibabaResearch/DAMO-ConvAI), execute_model, 30 s limit",
           "n": n, "agree": agree, "agreement_pct": round(100 * agree / n, 3),
           "official_script_correct": sum(off.values()), "ours_correct": sum(ours[k] for k in off),
           "disagreements_by_source": {k: dict(v) for k, v in by.items()}, "disagreements": dis}
    json.dump(res, open(OUT / "bird_official_check.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "disagreements"})


if __name__ == "__main__":
    main()
