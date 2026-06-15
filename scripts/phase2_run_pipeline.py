"""Phase 2: Run multi-step pipelines (S1-S4) on BIRD dev set.

Usage:
    python scripts/phase2_run_pipeline.py <model> <pipeline> [options]

Models: qwen, gemini, deepseek, gpt41
Pipelines: s0, s1, s2, s3, s4

Options:
    --self-correct     Enable self-correction (max 2 retries on exec errors)
    --subset mod       Only moderate+challenging queries (disagreement zone)
    --limit N          Process only first N queries
    --enriched         Use enriched schema (with column descriptions + samples)
    --workers N        Parallel workers for API calls (default 1)

Examples:
    python scripts/phase2_run_pipeline.py deepseek s1 --limit 30
    python scripts/phase2_run_pipeline.py gemini s2 --subset mod --workers 5
    python scripts/phase2_run_pipeline.py gpt41 s3 --self-correct
    python scripts/phase2_run_pipeline.py deepseek s4 --limit 5
"""
import argparse
import json
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.api_clients import estimate_cost
from src.utils.sql_utils import clean_sql, execute_sql, compare_results
from src.utils.schema_utils import get_schema, get_enriched_schema, DB_DIR, BIRD_DIR
from src.utils.prompt_templates import S0_DIRECT
from src.pipelines.base import BasePipeline
from src.pipelines.s1_schema_filter import S1SchemaFilter
from src.pipelines.s2_decompose import S2Decompose
from src.pipelines.s3_skeleton import S3Skeleton
from src.pipelines.s4_din_sql import S4DinSQL
from src.self_correction import SelfCorrectionWrapper

PIPELINE_MAP = {
    "s1": S1SchemaFilter,
    "s2": S2Decompose,
    "s3": S3Skeleton,
    "s4": S4DinSQL,
}


class S0Direct(BasePipeline):
    """S0 baseline using shared utils (for comparison within Phase 2 framework)."""
    name = "s0_direct"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = S0_DIRECT.format(schema=schema, question=question, evidence_section=evidence_section)
        raw = self._call_llm(prompt, "direct_generate")
        return self._make_result(clean_sql(raw))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 pipeline runner")
    parser.add_argument("model", choices=["qwen", "gemini", "deepseek", "gpt41"])
    parser.add_argument("pipeline", choices=["s0", "s1", "s2", "s3", "s4"])
    parser.add_argument("--self-correct", action="store_true", help="Enable self-correction")
    parser.add_argument("--subset", choices=["mod", "all"], default="all",
                        help="'mod' = moderate+challenging only, 'all' = full dev set")
    parser.add_argument("--limit", type=int, default=0, help="Max queries to process (0=all)")
    parser.add_argument("--enriched", action="store_true", help="Use enriched schema")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1)")
    return parser.parse_args()


def load_bird_data(subset="all"):
    """Load BIRD dev set, optionally filtering to moderate+challenging."""
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)
    if subset == "mod":
        bird = [q for q in bird if q["difficulty"] in ("moderate", "challenging")]
    return bird


def run_experiment(args):
    bird = load_bird_data(args.subset)
    if args.limit:
        bird = bird[:args.limit]

    # Build pipeline
    if args.pipeline == "s0":
        pipeline = S0Direct(args.model)
    else:
        pipeline = PIPELINE_MAP[args.pipeline](args.model)

    if args.self_correct:
        pipeline = SelfCorrectionWrapper(pipeline, max_retries=2)

    sc_tag = "+sc" if args.self_correct else ""
    enriched_tag = "+enriched" if args.enriched else ""
    run_name = "phase2_%s_%s%s%s" % (args.model, args.pipeline, sc_tag, enriched_tag)
    result_file = PROJECT_ROOT / "results" / ("%s.json" % run_name)
    partial_file = PROJECT_ROOT / "results" / ("%s_partial.json" % run_name)

    print("=" * 70)
    print("PHASE 2: %s on %s (%s)" % (args.pipeline.upper() + sc_tag, args.model, args.subset))
    print("Queries: %d | Schema: %s" % (len(bird), "enriched" if args.enriched else "raw"))
    print("Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Resume from partial results
    results = []
    done_ids = set()
    if partial_file.exists():
        with open(partial_file) as f:
            partial = json.load(f)
            results = partial.get("results", [])
            done_ids = {r["query_id"] for r in results}
            print("  Resuming from %d completed queries" % len(done_ids))

    stats = defaultdict(lambda: {"total": 0, "correct": 0, "exec_err": 0, "no_sql": 0})
    for r in results:
        d = r["difficulty"]
        stats[d]["total"] += 1
        if r.get("correct"):
            stats[d]["correct"] += 1
        if r.get("error") and r["error"] != "no SQL generated" and "gold_error" not in r.get("error", ""):
            stats[d]["exec_err"] += 1

    start_time = time.time()
    api_cost = 0.0

    # Build work items (skip already done)
    work_items = []
    for i, q in enumerate(bird):
        query_id = q.get("question_id", i)
        if query_id in done_ids:
            continue
        db_id = q["db_id"]
        db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))
        if not Path(db_path).exists():
            continue
        work_items.append((i, q, query_id, db_path))

    def _process_query(item, worker_pipeline):
        """Process a single query. Thread-safe: each worker has its own pipeline."""
        i, q, query_id, db_path = item
        db_id = q["db_id"]
        diff = q["difficulty"]
        question = q["question"]
        gold_sql = q["SQL"]
        evidence = q.get("evidence", "")

        # Execute gold SQL
        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            return {
                "query_id": query_id, "db_id": db_id, "difficulty": diff,
                "correct": False, "error": "gold_error: " + gold_err[:100],
            }

        # Get schema
        if args.enriched:
            schema = get_enriched_schema(db_path, db_id)
        else:
            schema = get_schema(db_path)

        # Run pipeline
        t0 = time.time()
        try:
            result = worker_pipeline.run(question, schema, evidence, db_path)
        except Exception as e:
            result = {"pred_sql": "", "steps": [{"step": "error", "raw": str(e)[:200]}], "call_count": 0}
        gen_time = time.time() - t0

        pred_sql = result.get("pred_sql", "")
        call_count = result.get("call_count", 0)

        # Execute and compare
        has_sql = pred_sql and "SELECT" in pred_sql.upper()
        if has_sql:
            pred_result, pred_err = execute_sql(db_path, pred_sql)
        else:
            pred_result, pred_err = None, "no SQL generated"

        correct = compare_results(gold_result, pred_result)

        return {
            "query_id": query_id, "db_id": db_id, "difficulty": diff,
            "question": question[:100],
            "gold_sql": gold_sql[:200], "pred_sql": pred_sql[:200],
            "correct": correct, "gen_time": round(gen_time, 2),
            "call_count": call_count,
            "correction_attempts": result.get("correction_attempts", 0),
            "error": (pred_err[:100] if pred_err else None) if has_sql else "no SQL generated",
            "_cost": call_count * estimate_cost(args.model, 800, 150),
            "_has_sql": has_sql, "_pred_err": bool(pred_err) if has_sql else False,
        }

    workers = getattr(args, "workers", 1) or 1
    lock = threading.Lock()
    new_processed = 0

    def _collect_result(r):
        """Collect a processed result into shared state. Called under lock."""
        nonlocal api_cost, new_processed
        diff = r.get("difficulty", "unknown")
        api_cost += r.pop("_cost", 0)
        has_sql = r.pop("_has_sql", False)
        pred_err = r.pop("_pred_err", False)

        stats[diff]["total"] += 1
        if r.get("correct"):
            stats[diff]["correct"] += 1
        if pred_err and has_sql:
            stats[diff]["exec_err"] += 1

        results.append(r)
        new_processed += 1

        # Progress report every 50 queries
        if new_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = new_processed / elapsed * 60 if elapsed > 0 else 0
            remaining = (len(work_items) - new_processed) / rate if rate > 0 else 0
            total_correct = sum(s["correct"] for s in stats.values())
            total_done = sum(s["total"] for s in stats.values())
            acc = total_correct / total_done * 100 if total_done else 0
            print("  [%d/%d] %.1f%% acc | %.0f q/min | ~%.0f min left | ~$%.3f" % (
                len(results), len(bird), acc, rate, remaining, api_cost), flush=True)

        # Save partial every 100 queries
        if new_processed % 100 == 0:
            _save_partial(partial_file, stats, results, api_cost)

    if workers <= 1:
        # Sequential mode (original behavior)
        for item in work_items:
            r = _process_query(item, pipeline)
            _collect_result(r)
    else:
        # Parallel mode: each worker gets its own pipeline instance
        def _make_pipeline():
            if args.pipeline == "s0":
                p = S0Direct(args.model)
            else:
                p = PIPELINE_MAP[args.pipeline](args.model)
            if args.self_correct:
                p = SelfCorrectionWrapper(p, max_retries=2)
            return p

        worker_pipelines = [_make_pipeline() for _ in range(workers)]
        print("  Using %d parallel workers" % workers, flush=True)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, item in enumerate(work_items):
                wp = worker_pipelines[idx % workers]
                f = executor.submit(_process_query, item, wp)
                futures[f] = item

            for f in as_completed(futures):
                r = f.result()
                with lock:
                    _collect_result(r)

    # Final save
    _save_partial(partial_file, stats, results, api_cost)

    elapsed = time.time() - start_time
    _print_summary(run_name, args, stats, elapsed, api_cost, result_file, results)


def _save_partial(path, stats, results, api_cost):
    with open(path, "w") as f:
        json.dump({"stats": dict(stats), "results": results, "api_cost_estimate": api_cost},
                  f, ensure_ascii=False)


def _print_summary(run_name, args, stats, elapsed, api_cost, result_file, results):
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS: %s" % run_name)
    print("=" * 70)
    print("Model: %s | Pipeline: %s | Self-correct: %s" % (
        args.model, args.pipeline, args.self_correct))
    print("Total time: %.1f minutes" % (elapsed / 60))
    print("Estimated API cost: $%.3f" % api_cost)
    print()

    print("%-15s %8s %8s %8s %8s" % ("Difficulty", "Total", "Correct", "Acc%", "Exec Err"))
    print("-" * 55)
    overall_total = 0
    overall_correct = 0
    for d in ["simple", "moderate", "challenging"]:
        s = stats.get(d, {"total": 0, "correct": 0, "exec_err": 0})
        if s["total"] == 0:
            continue
        acc = s["correct"] / s["total"] * 100
        print("%-15s %8d %8d %7.1f%% %8d" % (d, s["total"], s["correct"], acc, s["exec_err"]))
        overall_total += s["total"]
        overall_correct += s["correct"]

    if overall_total:
        overall_acc = overall_correct / overall_total * 100
        print("-" * 55)
        print("%-15s %8d %8d %7.1f%%" % ("OVERALL", overall_total, overall_correct, overall_acc))

    # Avg call count
    call_counts = [r.get("call_count", 0) for r in results if r.get("call_count")]
    if call_counts:
        print("Avg LLM calls/query: %.1f" % (sum(call_counts) / len(call_counts)))

    # Self-correction stats
    corrections = [r.get("correction_attempts", 0) for r in results]
    if any(c > 0 for c in corrections):
        corrected = sum(1 for c in corrections if c > 0)
        print("Self-correction applied: %d/%d queries" % (corrected, len(corrections)))

    # Save final results
    output = {
        "run_name": run_name,
        "model": args.model,
        "pipeline": args.pipeline,
        "self_correct": args.self_correct,
        "enriched_schema": args.enriched,
        "subset": args.subset,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_minutes": round(elapsed / 60, 2),
        "api_cost_estimate": round(api_cost, 4),
        "stats": {},
        "results": results,
    }
    for d in ["simple", "moderate", "challenging"]:
        s = stats.get(d, {"total": 0, "correct": 0, "exec_err": 0})
        if s["total"]:
            output["stats"][d] = {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": round(s["correct"] / s["total"], 4),
                "exec_errors": s["exec_err"],
            }
    if overall_total:
        output["stats"]["overall"] = {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": round(overall_correct / overall_total, 4),
        }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved to %s" % result_file)
    print("Finished: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
