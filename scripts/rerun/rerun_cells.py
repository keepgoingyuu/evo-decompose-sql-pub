"""Unified runner for the 2026-11 IS-revision rerun.

One harness for every model and benchmark, replacing the five per-phase runners:
  - S0 uses S0_DIRECT for every model and language; S1-S4 use the shared pipelines
  - SQL extraction with the fixed `clean_sql` (WITH prefix preserved)
  - every LLM call's full raw response, finish reason and token counts are stored
  - full predicted and gold SQL are stored (nothing is cut)
  - each prediction is scored under the paper's strict rule and the benchmark's official rule,
    and the pre-fix extractor is applied to the same final response for comparison

Output is append-only JSONL, one line per query, so a run can stop and resume at any point:
  results/rerun_2026_11/<bench>__<model>__<strategy>.jsonl

Usage:
  venv/bin/python scripts/rerun/rerun_cells.py gpt41 bird s2            # worklist entries only
  venv/bin/python scripts/rerun/rerun_cells.py qwen25c7b en s0 --all    # whole cell
  ... --limit 20          only the first 20 queries of the list (smoke / reproducibility check)
  ... --window 18:00-08:00  only run inside this local-time window; exit cleanly outside it
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCHMARKS, REPO, STRATEGIES, load_queries  # noqa: E402

sys.path.insert(0, str(REPO))
from src.pipelines.base import BasePipeline  # noqa: E402
from src.pipelines.s1_schema_filter import S1SchemaFilter  # noqa: E402
from src.pipelines.s2_decompose import S2Decompose  # noqa: E402
from src.pipelines.s3_skeleton import S3Skeleton  # noqa: E402
from src.pipelines.s4_din_sql import S4DinSQL  # noqa: E402
from src.utils.api_clients import MODEL_CONFIG  # noqa: E402
from src.utils.prompt_templates import S0_DIRECT  # noqa: E402
from src.utils.schema_utils import get_schema  # noqa: E402
from src.utils.sql_utils import (clean_sql, clean_sql_legacy, compare_results, compare_results_bird,  # noqa: E402
                                 compare_results_spider, execute_sql)

SEED = 42
OUT_DIR = REPO / "results/rerun_2026_11"
MAX_CONSECUTIVE_FAILURES = 5
PIPELINES = {"s1": S1SchemaFilter, "s2": S2Decompose, "s3": S3Skeleton, "s4": S4DinSQL}


class S0Direct(BasePipeline):
    name = "s0_direct"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        raw = self._call_llm(S0_DIRECT.format(schema=schema, question=question,
                                              evidence_section=evidence_section), "direct_generate")
        return self._make_result(clean_sql(raw))


def harness_id() -> str:
    """Content hash of the code that determines a prediction, identical on every machine
    (the DGX copy is not a git checkout)."""
    import hashlib
    h = hashlib.sha1()
    files = sorted((REPO / "src").rglob("*.py")) + [Path(__file__), Path(__file__).with_name("common.py")]
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def in_window(window: str | None) -> bool:
    if not window:
        return True
    a, b = (dt.time.fromisoformat(x) for x in window.split("-"))
    now = dt.datetime.now().time()
    return (a <= now or now < b) if a > b else (a <= now < b)


def score(bench: str, q: dict, sql: str, gold_result):
    """(error, strict, official) for one predicted SQL string."""
    if not sql or "SELECT" not in sql.upper():
        return "no SQL generated", False, False
    pred_result, err = execute_sql(q["db_path"], sql)
    if err:
        return err[:300], False, False
    strict = compare_results(gold_result, pred_result)
    if BENCHMARKS[bench]["family"] == "bird":
        official = compare_results_bird(gold_result, pred_result)
    else:
        official = compare_results_spider(gold_result, pred_result, q["gold_sql"])
    return None, strict, official


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODEL_CONFIG))
    ap.add_argument("bench", choices=sorted(BENCHMARKS))
    ap.add_argument("strategy", choices=STRATEGIES)
    ap.add_argument("--all", action="store_true", help="run the whole cell instead of the worklist entries")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", default=None, help="HH:MM-HH:MM local time; stop cleanly outside it")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bench}__{args.model}__{args.strategy}.jsonl"
    tag = f"[{args.model}/{args.bench}/{args.strategy}]"

    queries = load_queries(args.bench)
    if args.all:
        todo = [q["idx"] for q in queries]
    else:
        wl = json.load(open(OUT_DIR / "worklist.json"))
        todo = wl[f"{args.bench}|{args.model}|{args.strategy}"]["idx"]
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["idx"])
    if args.limit:
        todo = todo[:args.limit]   # first N of the cell's list, so a limited run is idempotent
    pending = [i for i in todo if i not in done]
    print(f"{tag} todo {len(todo)}  done {len(done & set(todo))}  running {len(pending)}", flush=True)
    if not pending:
        return

    git_sha = harness_id()
    pipeline = S0Direct(args.model) if args.strategy == "s0" else PIPELINES[args.strategy](args.model)
    pipeline._seed = SEED
    schema_cache: dict[str, str] = {}
    t_start, n_ok, n_fail = time.time(), 0, 0

    with open(out, "a", encoding="utf-8") as fh:
        for n, idx in enumerate(pending, 1):
            if not in_window(args.window):
                print(f"{tag} outside window {args.window}; stopping after {n - 1} queries", flush=True)
                break
            q = queries[idx]
            rec = {"bench": args.bench, "model": args.model, "model_id": MODEL_CONFIG[args.model]["model"],
                   "strategy": args.strategy, "idx": idx, "question_id": q["question_id"], "db_id": q["db_id"],
                   "question": q["question"], "gold_sql": q["gold_sql"], "seed": SEED, "harness": git_sha,
                   "ts": dt.datetime.now().isoformat(timespec="seconds")}
            gold_result, gold_err = execute_sql(q["db_path"], q["gold_sql"])
            if gold_err:
                rec.update(error="gold_error: " + gold_err[:200], pred_sql="", steps=[], call_count=0,
                           correct=False, correct_official=False)
            else:
                if q["db_path"] not in schema_cache:
                    schema_cache[q["db_path"]] = get_schema(q["db_path"])
                t0 = time.time()
                try:
                    res = pipeline.run(q["question"], schema_cache[q["db_path"]],
                                       evidence=q["evidence"], db_path=q["db_path"])
                except Exception as e:  # noqa: BLE001 - recorded, never silently dropped
                    res = {"pred_sql": "", "steps": list(pipeline.steps) + [{"step": "pipeline_error",
                                                                               "raw": repr(e)[:500]}],
                           "call_count": pipeline.call_count}
                rec["gen_time"] = round(time.time() - t0, 2)
                pred = res.get("pred_sql", "") or ""
                steps = res.get("steps", [])
                # A failed API call is not a model answer: leave the query pending so a later
                # run retries it, and stop if the endpoint keeps failing.
                failed = [st for st in steps if str(st.get("raw", "")).startswith("ERROR:")
                          or st.get("step") == "pipeline_error"]
                if failed:
                    n_fail += 1
                    print(f"{tag} idx {idx} call failed, left pending: {str(failed[0].get('raw'))[:160]}", flush=True)
                    if n_fail >= MAX_CONSECUTIVE_FAILURES:
                        print(f"{tag} {n_fail} consecutive failures; aborting", flush=True)
                        sys.exit(2)
                    continue
                n_fail = 0
                err, strict, official = score(args.bench, q, pred, gold_result)
                final_raw = steps[-1].get("raw", "") if steps else ""
                legacy = clean_sql_legacy(final_raw) if final_raw else ""
                l_err, l_strict, _ = (err, strict, official) if legacy == pred else score(args.bench, q, legacy,
                                                                                         gold_result)
                rec.update(pred_sql=pred, error=err, correct=strict, correct_official=official,
                           legacy_pred_sql=None if legacy == pred else legacy, legacy_error=l_err,
                           legacy_correct=l_strict, call_count=res.get("call_count", 0), steps=steps)
                n_ok += strict
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(pending):
                el = time.time() - t_start
                print(f"{tag} {n}/{len(pending)}  strict-acc {100 * n_ok / n:.1f}%  "
                      f"{el / n:.1f}s/q  eta {el / n * (len(pending) - n) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
