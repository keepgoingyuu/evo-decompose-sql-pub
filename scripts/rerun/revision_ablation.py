"""Template ablation for S1 and S2 on the corrected harness (IS revision: R1-W3, R3-2).

For each (model, language) with ablation runs, compares every template variant with direct
generation (S0) on the same 400-query subset: accuracy, paired drop, exact McNemar test,
mean calls, and for S2 the share of queries on which the model produced fewer than two
sub-questions (the pipeline then falls back to one direct-generation call).
V1 (canonical template) and S0 come from the corrected matrix.

Output: results/rerun_2026_11/analysis/ablation.json
Usage: python3 scripts/rerun/revision_ablation.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

AN = REPO / "results/rerun_2026_11/analysis"
AB = REPO / "results/rerun_2026_11/ablation"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def main() -> None:
    M = defaultdict(dict)
    for line in open(AN / "matrix.jsonl", encoding="utf-8"):
        r = json.loads(line)
        M[r["bench"], r["model"], r["strategy"]][r["idx"]] = r
    runs = defaultdict(dict)  # (model, bench, strat) -> {variant: {idx: rec}}
    for f in sorted(glob.glob(str(AB / "*.jsonl"))):
        bench, model, tag = os.path.basename(f)[:-6].split("__")
        strat, ver = tag.split("-")
        runs[model, bench, strat][ver] = {(r := json.loads(line))["idx"]: r for line in open(f, encoding="utf-8")
                                          if line.strip()}
    out = {}
    for (model, bench, strat), vs in sorted(runs.items()):
        ids = sorted(set.intersection(*[set(v) for v in vs.values()]))
        s0 = M[bench, model, "s0"]
        v1 = M[bench, model, strat]
        for rule, key in (("strict", "correct"), ("official", "correct_official")):
            a0 = 100 * sum(s0[i][rule] for i in ids) / len(ids)
            rows = {}
            for ver in ["v1"] + sorted(vs):
                if ver == "v1":
                    ok = {i: bool(v1[i][rule]) for i in ids}
                    calls = [v1[i]["call_count"] for i in ids if v1[i]["call_count"]]
                    traced = [v1[i]["n_sub"] for i in ids if v1[i]["n_sub"] is not None]
                    nodec = (round(100 * sum(k < 2 for k in traced) / len(traced), 1), len(traced)) if traced else None
                else:
                    ok = {i: bool(vs[ver][i][key]) for i in ids}
                    calls = [vs[ver][i]["call_count"] for i in ids]
                    nsub = [sum(str(s.get("step", "")).startswith("sub_sql") for s in vs[ver][i]["steps"]) for i in ids]
                    nodec = (round(100 * sum(k < 2 for k in nsub) / len(ids), 1), len(ids)) if strat == "s2" else None
                b = sum(s0[i][rule] and not ok[i] for i in ids)
                c = sum(not s0[i][rule] and ok[i] for i in ids)
                acc = 100 * sum(ok.values()) / len(ids)
                rows[ver] = {"acc": round(acc, 2), "drop": round(a0 - acc, 2), "broken": b, "rescued": c,
                             "p": mcnemar(b, c), "calls": round(sum(calls) / len(calls), 2) if calls else None,
                             "no_decomposition": nodec}
            out[f"{model}|{bench}|{strat}|{rule}"] = {"n": len(ids), "s0": round(a0, 2), "variants": rows}
    json.dump(out, open(AN / "ablation.json", "w"), indent=1)
    for k, v in out.items():
        if k.endswith("strict"):
            print(k, f"n={v['n']} S0={v['s0']}")
            for ver, r in v["variants"].items():
                print(f"   {ver}: acc {r['acc']:.2f} drop {r['drop']:+.2f} b/c {r['broken']}/{r['rescued']} p={r['p']:.2g} "
                      f"calls {r['calls']} no-decomp {r['no_decomposition']}")


if __name__ == "__main__":
    main()
