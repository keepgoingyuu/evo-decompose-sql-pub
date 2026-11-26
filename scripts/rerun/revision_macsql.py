"""MAC-SQL (GPT-4.1 backbone, 609 BIRD mod+chal queries) against the corrected GPT-4.1 cells.

Scores the stored MAC-SQL predictions under both rules, pairs them per query with the
GPT-4.1 single-strategy cells of the corrected matrix, and reads the API trace for calls and
tokens per query (R3 point 5).

Output: results/rerun_2026_11/analysis/macsql.json
Usage: python3 scripts/rerun/revision_macsql.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, STRATEGIES, load_queries  # noqa: E402

sys.path.insert(0, str(REPO))
from src.utils.sql_utils import compare_results, compare_results_bird, execute_sql  # noqa: E402

D = REPO / "baselines/MAC-SQL/outputs/modchal609"
AN = REPO / "results/rerun_2026_11/analysis"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def main() -> None:
    qs = load_queries("bird")
    qid_of = json.load(open(D / "idx_to_bird_question_id.json"))  # position in the MAC-SQL input file -> BIRD question_id
    by_qid = {q["question_id"]: q for q in qs}
    outs = [json.loads(line) for line in open(D / "output.json", encoding="utf-8") if line.strip()]
    assert len(outs) == len(qid_of) == 609
    mac = {}
    for o in outs:
        q = by_qid[qid_of[o["idx"]]]
        gold, _ = execute_sql(q["db_path"], q["gold_sql"])
        pred, err = execute_sql(q["db_path"], o.get("pred") or "")
        ok = gold is not None and pred is not None
        mac[q["idx"]] = {"strict": bool(ok and compare_results(gold, pred)),
                         "official": bool(ok and compare_results_bird(gold, pred)), "exec_error": bool(err)}
    M = defaultdict(dict)
    for line in open(AN / "matrix.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["bench"] == "bird" and r["model"] == "gpt41":
            M[r["strategy"]][r["idx"]] = r
    res = {"n": len(mac), "exec_errors": sum(v["exec_error"] for v in mac.values())}
    for rule in ("strict", "official"):
        acc = 100 * sum(v[rule] for v in mac.values()) / len(mac)
        res[rule] = {"macsql": round(acc, 2), "vs": {}}
        for s in STRATEGIES:
            a = 100 * sum(r[rule] for r in M[s].values()) / len(M[s])
            b = sum(mac[i][rule] and not M[s][i][rule] for i in mac)
            c = sum(M[s][i][rule] and not mac[i][rule] for i in mac)
            res[rule]["vs"][s] = {"gpt41": round(a, 2), "gain": round(acc - a, 2), "mac_only": b, "single_only": c,
                                  "p": mcnemar(b, c)}
    calls, ptok, rtok, agents = Counter(), Counter(), Counter(), Counter()
    for line in open(D / "api_trace.json", encoding="utf-8"):
        if line.strip():
            t = json.loads(line)
            calls[t["idx"]] += 1
            ptok[t["idx"]] += t.get("prompt_token") or 0
            rtok[t["idx"]] += t.get("response_token") or 0
            agents[t["send_to"]] += 1
    n = len(mac)
    s0 = [r for r in M["s0"].values() if r["in_tok"]]
    res["cost"] = {"calls_per_query": round(sum(calls.values()) / n, 2), "calls_by_agent": dict(agents),
                   "prompt_tok_per_query": round(sum(ptok.values()) / n), "response_tok_per_query": round(sum(rtok.values()) / n),
                   "gpt41_s0_prompt_tok_per_query_regenerated": round(sum(r["in_tok"] for r in s0) / len(s0)) if s0 else None,
                   "gpt41_s0_n_regenerated": len(s0)}
    json.dump(res, open(AN / "macsql.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
