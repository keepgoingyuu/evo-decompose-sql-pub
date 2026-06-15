"""Phase 4: Cross-lingual strategy evaluation (RQ4).

Compare decomposition strategy effectiveness across English and Chinese:
  - English questions from Spider dev set (1,034 queries)
  - Traditional Chinese questions from CSpider-TW (same 1,034 queries)
  - (Supplementary) Simplified Chinese from CSpider-TW

All use the same Spider databases, enabling direct cross-lingual comparison
of strategy rankings (S0-S4).

Usage:
    python scripts/phase4_cspider_tw.py <model> <strategy> [options]

Models: qwen, gemini, deepseek, gpt41
Strategies: s0, s1, s2, s3, s4

Options:
    --lang en|zh-tw|zh-cn|both   Language (default: zh-tw)
    --limit N                     Process only first N queries (0=all)
    --self-correct                Enable self-correction (max 2 retries)

Examples:
    python scripts/phase4_cspider_tw.py deepseek s0 --lang en --limit 50
    python scripts/phase4_cspider_tw.py gpt41 s1 --lang zh-tw
    python scripts/phase4_cspider_tw.py deepseek s2 --lang en
    python scripts/phase4_cspider_tw.py gpt41 s4 --self-correct --limit 100
"""
import argparse
import json
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.api_clients import call_llm, estimate_cost
from src.utils.sql_utils import clean_sql, execute_sql, compare_results
from src.utils.schema_utils import get_schema
from src.utils.prompt_templates import S0_DIRECT
from src.pipelines.base import BasePipeline
from src.pipelines.s1_schema_filter import S1SchemaFilter
from src.pipelines.s2_decompose import S2Decompose
from src.pipelines.s3_skeleton import S3Skeleton
from src.pipelines.s4_din_sql import S4DinSQL
from src.self_correction import SelfCorrectionWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSPIDER_DATA = PROJECT_ROOT / "data" / "cspider" / "dev_tw.json"
SPIDER_EN_DATA = PROJECT_ROOT / "data" / "spider" / "spider_data" / "dev.json"
SPIDER_DB_DIR = PROJECT_ROOT / "data" / "spider" / "spider_data" / "database"

PIPELINE_MAP = {
    "s1": S1SchemaFilter,
    "s2": S2Decompose,
    "s3": S3Skeleton,
    "s4": S4DinSQL,
}

LANG_CONFIG = {
    "en":    {"field": "question", "label": "English"},
    "zh-tw": {"field": "question", "label": "Traditional Chinese"},
    "zh-cn": {"field": "question_simplified", "label": "Simplified Chinese"},
}

# ---------------------------------------------------------------------------
# S0 direct prompt for Chinese questions
# ---------------------------------------------------------------------------
S0_DIRECT_ZH = """You are a SQL expert. Given the following database schema and a question in Chinese, generate a SQLite-compatible SQL query.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Rules:
- SQLite syntax ONLY (no ILIKE, no ::type, no date_part, no FILTER)
- Only use tables and columns from the schema above
- The question is in Chinese but the schema (table/column names) is in English
- Output ONLY the raw SQL query, no explanations, no markdown"""


class S0DirectZh(BasePipeline):
    """S0 baseline adapted for Chinese questions (CSpider-TW)."""
    name = "s0_direct_zh"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = S0_DIRECT_ZH.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        raw = self._call_llm(prompt, "direct_generate")
        return self._make_result(clean_sql(raw))


class S0DirectEn(BasePipeline):
    """S0 baseline for English questions (Spider dev set)."""
    name = "s0_direct_en"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = S0_DIRECT.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        raw = self._call_llm(prompt, "direct_generate")
        return self._make_result(clean_sql(raw))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: CSpider-TW cross-lingual experiments (RQ4)"
    )
    parser.add_argument("model", choices=["qwen", "gemini", "deepseek", "gpt41"])
    parser.add_argument("strategy", choices=["s0", "s1", "s2", "s3", "s4"])
    parser.add_argument(
        "--lang", choices=["en", "zh-tw", "zh-cn", "both"], default="zh-tw",
        help="Language: en (English Spider), zh-tw (Traditional, default), zh-cn (Simplified), both (zh-tw+zh-cn)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max queries to process (0=all)"
    )
    parser.add_argument(
        "--self-correct", action="store_true", help="Enable self-correction"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel workers (default 1)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(lang, limit=0):
    """Load dataset based on language.

    For 'en': loads Spider English dev.json (keys: db_id, question, query).
    For 'zh-tw'/'zh-cn'/'both': loads CSpider-TW dev_tw.json
        (keys: db_id, question, question_simplified, query).

    Returns list of dicts.
    """
    if lang == "en":
        if not SPIDER_EN_DATA.exists():
            print("ERROR: Spider English data not found at %s" % SPIDER_EN_DATA)
            sys.exit(1)
        with open(SPIDER_EN_DATA, encoding="utf-8") as f:
            data = json.load(f)
    else:
        if not CSPIDER_DATA.exists():
            print("ERROR: CSpider data not found at %s" % CSPIDER_DATA)
            print("Please place dev_tw.json in data/cspider/")
            sys.exit(1)
        with open(CSPIDER_DATA, encoding="utf-8") as f:
            data = json.load(f)

    if limit:
        data = data[:limit]

    return data


# ---------------------------------------------------------------------------
# Single-language experiment runner
# ---------------------------------------------------------------------------
def run_single_lang(args, data, lang):
    """Run experiment for a single language variant (zh-tw or zh-cn).

    Returns (metadata_dict, results_list).
    """
    lang_field = LANG_CONFIG[lang]["field"]
    lang_label = LANG_CONFIG[lang]["label"]

    # Build pipeline
    if args.strategy == "s0":
        pipeline = S0DirectEn(args.model) if lang == "en" else S0DirectZh(args.model)
    else:
        pipeline = PIPELINE_MAP[args.strategy](args.model)

    if args.self_correct:
        pipeline = SelfCorrectionWrapper(pipeline, max_retries=2)

    sc_tag = "+sc" if args.self_correct else ""
    run_name = "phase4_%s_%s_%s%s" % (args.model, args.strategy, lang, sc_tag)
    result_file = PROJECT_ROOT / "results" / ("%s.json" % run_name)
    partial_file = PROJECT_ROOT / "results" / ("%s_partial.json" % run_name)

    ds_name = "Spider English" if lang == "en" else "Spider (CSpider-TW)"
    print("=" * 70)
    print("PHASE 4 (RQ4): %s%s on %s | %s (%s)" % (
        args.strategy.upper(), sc_tag, args.model, lang_label, lang
    ))
    print("Queries: %d | Dataset: %s" % (len(data), ds_name))
    print("Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Resume from partial results if available
    results = []
    done_indices = set()
    if partial_file.exists():
        with open(partial_file, encoding="utf-8") as f:
            partial = json.load(f)
            results = partial.get("results", [])
            done_indices = {r["query_idx"] for r in results}
            print("  Resuming from %d completed queries" % len(done_indices))

    # Rebuild stats from partial results
    correct_count = sum(1 for r in results if r.get("correct"))
    exec_err_count = sum(
        1 for r in results
        if r.get("error") and r["error"] != "no SQL generated"
        and "gold_error" not in r.get("error", "")
    )

    start_time = time.time()
    api_cost = 0.0
    new_processed = 0

    # Build work items (skip already done)
    work_items = []
    for idx, q in enumerate(data):
        if idx in done_indices:
            continue
        db_id = q["db_id"]
        db_path = str(SPIDER_DB_DIR / db_id / (db_id + ".sqlite"))
        work_items.append((idx, q, db_path))

    def _process_query(item, worker_pipeline):
        """Process a single query. Thread-safe: each worker has its own pipeline."""
        idx, q, db_path = item
        db_id = q["db_id"]
        question = q[lang_field]
        gold_sql = q["query"]

        if not Path(db_path).exists():
            return {
                "query_idx": idx, "db_id": db_id,
                "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": "",
                "correct": False, "error": "db_not_found: %s" % db_id,
                "_cost": 0, "_has_sql": False, "_pred_err": False,
            }

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            return {
                "query_idx": idx, "db_id": db_id,
                "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": "",
                "correct": False, "error": "gold_error: %s" % gold_err[:100],
                "_cost": 0, "_has_sql": False, "_pred_err": False,
            }

        schema = get_schema(db_path)

        t0 = time.time()
        try:
            result = worker_pipeline.run(question, schema, evidence="", db_path=db_path)
        except Exception as e:
            result = {
                "pred_sql": "",
                "steps": [{"step": "error", "raw": str(e)[:200]}],
                "call_count": 0,
            }
        gen_time = time.time() - t0

        pred_sql = result.get("pred_sql", "")
        call_count = result.get("call_count", 0)

        has_sql = pred_sql and "SELECT" in pred_sql.upper()
        if has_sql:
            pred_result, pred_err = execute_sql(db_path, pred_sql)
        else:
            pred_result, pred_err = None, "no SQL generated"

        correct = compare_results(gold_result, pred_result)

        return {
            "query_idx": idx, "db_id": db_id,
            "question": question[:120],
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

    def _collect_result(r):
        """Collect a processed result into shared state. Called under lock."""
        nonlocal api_cost, new_processed, correct_count, exec_err_count
        api_cost += r.pop("_cost", 0)
        has_sql = r.pop("_has_sql", False)
        pred_err = r.pop("_pred_err", False)

        if r.get("correct"):
            correct_count += 1
        if pred_err and has_sql:
            exec_err_count += 1

        results.append(r)
        new_processed += 1

        if new_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = new_processed / elapsed * 60 if elapsed > 0 else 0
            remaining_queries = len(data) - len(results)
            eta_min = remaining_queries / rate if rate > 0 else 0
            acc = correct_count / len(results) * 100 if results else 0
            print("  [%d/%d] %.1f%% acc | %.0f q/min | ~%.0f min left | ~$%.3f" % (
                len(results), len(data), acc, rate, eta_min, api_cost
            ), flush=True)

        if new_processed % 100 == 0:
            _save_partial(partial_file, results, api_cost)

    if workers <= 1:
        for item in work_items:
            r = _process_query(item, pipeline)
            _collect_result(r)
    else:
        def _make_pipeline():
            if args.strategy == "s0":
                p = S0DirectEn(args.model) if lang == "en" else S0DirectZh(args.model)
            else:
                p = PIPELINE_MAP[args.strategy](args.model)
            if args.self_correct:
                p = SelfCorrectionWrapper(p, max_retries=2)
            return p

        worker_pipelines = [_make_pipeline() for _ in range(workers)]
        print("  Using %d parallel workers" % workers, flush=True)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for wi, item in enumerate(work_items):
                wp = worker_pipelines[wi % workers]
                f = executor.submit(_process_query, item, wp)
                futures[f] = item

            for f in as_completed(futures):
                r = f.result()
                with lock:
                    _collect_result(r)

    # Final partial save (for safety before computing summary)
    _save_partial(partial_file, results, api_cost)

    elapsed = time.time() - start_time
    total = len(results)
    accuracy = correct_count / total if total else 0

    # Per-database breakdown
    db_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        db_stats[r["db_id"]]["total"] += 1
        if r.get("correct"):
            db_stats[r["db_id"]]["correct"] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 4 RESULTS: %s" % run_name)
    print("=" * 70)
    print("Model: %s | Strategy: %s%s | Lang: %s" % (
        args.model, args.strategy, sc_tag, lang
    ))
    print("Total time: %.1f minutes" % (elapsed / 60))
    print("Estimated API cost: $%.3f" % api_cost)
    print()
    print("%-10s %8s %8s %8s" % ("Metric", "Value", "", ""))
    print("-" * 40)
    print("%-10s %8d" % ("Total", total))
    print("%-10s %8d" % ("Correct", correct_count))
    print("%-10s %7.1f%%" % ("Accuracy", accuracy * 100))
    print("%-10s %8d" % ("Exec Err", exec_err_count))

    # Top-5 / Bottom-5 databases by accuracy
    db_sorted = sorted(
        db_stats.items(),
        key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] else 0,
        reverse=True
    )
    print("\nPer-database accuracy:")
    print("%-30s %6s %6s %7s" % ("Database", "Total", "Corr", "Acc%"))
    print("-" * 55)
    for db, s in db_sorted:
        db_acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print("%-30s %6d %6d %6.1f%%" % (db, s["total"], s["correct"], db_acc))

    # Avg call count
    call_counts = [r.get("call_count", 0) for r in results if r.get("call_count")]
    if call_counts:
        print("\nAvg LLM calls/query: %.1f" % (sum(call_counts) / len(call_counts)))

    # Self-correction stats
    corrections = [r.get("correction_attempts", 0) for r in results]
    if any(c > 0 for c in corrections):
        corrected = sum(1 for c in corrections if c > 0)
        print("Self-correction applied: %d/%d queries" % (corrected, len(corrections)))

    # Build metadata
    metadata = {
        "model": args.model,
        "strategy": args.strategy,
        "self_correct": args.self_correct,
        "lang": lang,
        "lang_label": lang_label,
        "total": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "exec_errors": exec_err_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_minutes": round(elapsed / 60, 2),
        "api_cost_estimate": round(api_cost, 4),
    }

    # Save final results
    output = {
        "metadata": metadata,
        "db_stats": {
            db: {"total": s["total"], "correct": s["correct"],
                 "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0}
            for db, s in db_sorted
        },
        "results": results,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved to %s" % result_file)
    print("Finished: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))

    return metadata, results


# ---------------------------------------------------------------------------
# Comparison report for --lang both
# ---------------------------------------------------------------------------
def print_comparison(meta_tw, meta_cn):
    """Print side-by-side comparison of zh-tw vs zh-cn results."""
    print("\n" + "=" * 70)
    print("CROSS-LINGUAL COMPARISON: zh-tw vs zh-cn")
    print("=" * 70)
    print("%-25s %12s %12s %10s" % ("Metric", "zh-tw", "zh-cn", "Delta"))
    print("-" * 60)

    tw_acc = meta_tw["accuracy"] * 100
    cn_acc = meta_cn["accuracy"] * 100
    delta_acc = tw_acc - cn_acc

    print("%-25s %11.1f%% %11.1f%% %+9.1f%%" % ("Accuracy", tw_acc, cn_acc, delta_acc))
    print("%-25s %12d %12d %10d" % ("Correct", meta_tw["correct"], meta_cn["correct"],
                                     meta_tw["correct"] - meta_cn["correct"]))
    print("%-25s %12d %12d %10d" % ("Exec Errors", meta_tw["exec_errors"], meta_cn["exec_errors"],
                                     meta_tw["exec_errors"] - meta_cn["exec_errors"]))
    print("%-25s %11.2fm %11.2fm" % ("Time",
                                       meta_tw["total_time_minutes"],
                                       meta_cn["total_time_minutes"]))
    print("%-25s %11s %11s" % ("Cost",
                                "$%.3f" % meta_tw["api_cost_estimate"],
                                "$%.3f" % meta_cn["api_cost_estimate"]))
    print()

    if abs(delta_acc) < 1.0:
        verdict = "Minimal difference (<1%%) between Traditional and Simplified Chinese."
    elif delta_acc > 0:
        verdict = "Traditional Chinese outperforms Simplified by %.1f%%." % delta_acc
    else:
        verdict = "Simplified Chinese outperforms Traditional by %.1f%%." % abs(delta_acc)
    print("Verdict: %s" % verdict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_partial(path, results, api_cost):
    """Save partial results for resume capability."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "api_cost_estimate": api_cost},
            f, ensure_ascii=False
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load data based on language
    data = load_data(args.lang if args.lang != "both" else "zh-tw", args.limit)

    # Print dataset overview
    db_ids = sorted(set(q["db_id"] for q in data))
    ds_name = "Spider English" if args.lang == "en" else "CSpider-TW"
    print("%s dataset: %d queries across %d databases" % (ds_name, len(data), len(db_ids)))

    # Verify at least one database exists
    missing_dbs = [db for db in db_ids if not (SPIDER_DB_DIR / db / (db + ".sqlite")).exists()]
    if missing_dbs:
        print("WARNING: %d/%d databases not found: %s" % (
            len(missing_dbs), len(db_ids), ", ".join(missing_dbs[:5])
        ))
        if len(missing_dbs) == len(db_ids):
            print("ERROR: No Spider databases found at %s" % SPIDER_DB_DIR)
            print("Please download Spider and place databases there.")
            sys.exit(1)

    if args.lang == "both":
        # Run both zh-tw and zh-cn, then compare
        print("\n>>> Running zh-tw (Traditional Chinese) <<<")
        meta_tw, _ = run_single_lang(args, data, "zh-tw")

        print("\n>>> Running zh-cn (Simplified Chinese) <<<")
        meta_cn, _ = run_single_lang(args, data, "zh-cn")

        print_comparison(meta_tw, meta_cn)

        # Save combined comparison file
        comparison_file = PROJECT_ROOT / "results" / (
            "phase4_%s_%s_comparison.json" % (args.model, args.strategy)
        )
        comparison = {
            "zh-tw": meta_tw,
            "zh-cn": meta_cn,
            "delta_accuracy": round(meta_tw["accuracy"] - meta_cn["accuracy"], 4),
        }
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print("\nComparison saved to %s" % comparison_file)
    else:
        run_single_lang(args, data, args.lang)


if __name__ == "__main__":
    main()
