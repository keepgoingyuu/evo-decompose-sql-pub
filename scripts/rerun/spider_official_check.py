"""Cross-check `compare_results_spider` against the official Spider execution evaluator (IS revision, R3-5).

The official rule is `eval_exec_match` from test-suite-sql-eval (Zhong et al., 2020), vendored
unchanged in baselines/MAC-SQL/evaluation/exec_eval.py. For every executable Spider-family
prediction of the five panel models, regenerated or kept from the original runs (full SQL), this script recomputes the verdict with the official function
and compares it with the stored `correct_official`.

Two official settings are reported:
  keep_distinct=True   the setting our re-implementation mirrors (DISTINCT is part of the query)
  keep_distinct=False  the script's command-line default (DISTINCT removed from gold and pred)
plug_value=False in both (the predicted values are evaluated as written). Each Spider directory
holds a single database, so the check runs on the original dev databases, not on the distilled
test-suite databases.

Output: results/rerun_2026_11/spider_official_check.json (+ a disagreement list for inspection)
Usage:  python3 scripts/rerun/spider_official_check.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PANEL, REPO, STRATEGIES, UNSCORABLE, legacy_records, load_queries  # noqa: E402

EVAL_DIR = REPO / "baselines/MAC-SQL/evaluation"
OUT = REPO / "results/rerun_2026_11"
BENCHES = ["en", "zh-tw", "ja"]
_DB: dict[str, list[str]] = {}


def _init() -> None:
    sys.path.insert(0, str(EVAL_DIR))
    for b in BENCHES:
        _DB[b] = [q["db_path"] for q in load_queries(b)]


def _official(task):
    from exec_eval import eval_exec_match  # noqa: E402  (official, unmodified)
    bench, idx, pred, gold = task
    out = {}
    for kd in (True, False):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out[kd] = bool(eval_exec_match(db=_DB[bench][idx], p_str=pred, g_str=gold, plug_value=False,
                                               keep_distinct=kd, progress_bar_for_each_datapoint=False))
        except Exception as e:  # noqa: BLE001  (gold failure or parser crash inside remove_distinct)
            out[kd] = "error: " + type(e).__name__
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    tasks, meta = [], []
    kept = defaultdict(dict)
    for line in open(OUT / "kept_verdicts.jsonl", encoding="utf-8"):
        k = json.loads(line)
        if k["bench"] in BENCHES:
            kept[k["bench"], k["model"], k["strategy"]][k["idx"]] = k
    for bench in BENCHES:
        qs = load_queries(bench)
        for model in PANEL:
            for strat in STRATEGIES:
                new = {}
                f = OUT / f"{bench}__{model}__{strat}.jsonl"
                if f.exists():
                    new = {(r := json.loads(line))["idx"]: r for line in open(f, encoding="utf-8") if line.strip()}
                old = legacy_records(bench, model, strat)
                for q in qs:
                    i = q["idx"]
                    if i in UNSCORABLE[bench]:
                        continue  # reference query not evaluable; failure for all systems by definition
                    if i in new:
                        r = new[i]
                        pred, err, ours, strict, src = (r.get("pred_sql") or "", r.get("error"),
                                                        bool(r["correct_official"]), bool(r["correct"]), "new")
                    else:
                        k = kept[bench, model, strat][i]
                        pred, err, ours, strict, src = (old[i].get("pred_sql") or "", k["error"],
                                                        bool(k["official"]), bool(k["strict"]), "kept")
                    if not ours and (not pred or err):
                        continue  # not executable: wrong under every rule, nothing to cross-check
                    tasks.append((bench, i, pred, q["gold_sql"]))
                    meta.append((bench, model, strat, i, ours, strict, src))

    print(f"checking {len(tasks)} executable Spider-family predictions", flush=True)
    with ProcessPoolExecutor(args.jobs, initializer=_init) as ex:
        res = list(ex.map(_official, tasks, chunksize=64))

    summ = {}
    for kd in (True, False):
        c = Counter()
        per = defaultdict(Counter)
        for (bench, model, strat, idx, ours, strict, src), o in zip(meta, res):
            off = o[kd]
            if isinstance(off, str):
                c["official_error"] += 1
                continue
            key = "agree" if off == ours else ("ours_only" if ours else "official_only")
            c[key] += 1
            c[f"{src}_{key}"] += 1
            per[f"{bench}|{model}|{strat}"][key] += 1
            c["official_correct"] += off
            c["ours_correct"] += ours
            c["strict_correct"] += strict
        n = c["agree"] + c["ours_only"] + c["official_only"]
        summ["keep_distinct" if kd else "remove_distinct"] = {
            "n": n, **c, "agreement_pct": round(100 * c["agree"] / max(1, n), 3),
            "net_pp_vs_ours": round(100 * (c["official_correct"] - c["ours_correct"]) / max(1, n), 3),
            "cells_with_disagreement": {k: dict(v) for k, v in per.items() if v["ours_only"] or v["official_only"]}}

    # disagreement examples with their SQL (keep_distinct setting)
    examples = []
    for t, (bench, model, strat, idx, ours, _, src), o in zip(tasks, meta, res):
        if not isinstance(o[True], str) and o[True] != ours:
            examples.append({"bench": bench, "model": model, "strategy": strat, "idx": idx, "ours": ours, "src": src,
                             "official": o[True], "gold_sql": t[3], "pred_sql": t[2]})
    summ["keep_distinct"]["disagreements"] = examples[:200]
    json.dump(summ, open(OUT / "spider_official_check.json", "w"), ensure_ascii=False, indent=1)
    for k, v in summ.items():
        print(f"{k:16} n={v['n']}  agree={v['agree']} ({v['agreement_pct']}%)  ours_only={v.get('ours_only', 0)}  "
              f"official_only={v.get('official_only', 0)}  official_err={v.get('official_error', 0)}  "
              f"net {v['net_pp_vs_ours']:+} pp")


if __name__ == "__main__":
    main()
