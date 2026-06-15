"""V2.1 cross-lingual run, 4 GPU PARALLEL.

For each benchmark in {spider_en, cspider_zh-tw, multispider_ja}:
  Run 2 models × 5 strategies × 1034 queries
  Split per-model query list into 2 halves, each half on a different GPU.

Workers (4 total per benchmark, all running simultaneously):
  Qwen-A on port 11434 (GPU 0)  — Qwen first half (queries 0..516)
  Qwen-B on port 11437 (GPU 2)  — Qwen second half (queries 517..1033)
  Gemma-A on port 11435 (GPU 1) — Gemma first half
  Gemma-B on port 11438 (GPU 3) — Gemma second half

Requires: scripts/start_ollama_4gpu.sh already started.

Output: results/q1_2026_04/main/v21_xlingual_<benchmark>_<model>_<strategy>_seed42.json
"""
import sys
import json
import os
import time
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Two endpoints per model — splits the 1034-query workload across 2 GPUs.
MODEL_ENDPOINTS = {
    "qwen25c7b": [11434, 11437],
    "gemma4":    [11435, 11438],
}
SEED = 42
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]

RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_04" / "main"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark configs
BENCHMARKS = {
    "spider_en": {
        "queries_path": "data/spider/spider_data/spider_data/dev.json",
        "db_dir": "data/spider/spider_data/spider_data/database",
        "lang_code": "en",
    },
    "cspider_zh_tw": {
        "queries_path": "data/cspider/dev_tw.json",
        "db_dir": "data/spider/spider_data/spider_data/database",
        "lang_code": "zh-tw",
    },
    "multispider_ja": {
        "queries_path": "data/multispider/dataset/multispider/with_english_value/dev_ja.json",
        "db_dir": "data/spider/spider_data/spider_data/database",
        "lang_code": "ja",
    },
}


def run_chunk(model: str, strategy: str, benchmark: str, port: int,
              query_subset: list, chunk_id: int) -> dict:
    """Worker: run one (model, strategy, benchmark, chunk) on a specific port.

    Pins the Ollama HTTP request to a specific port via OLLAMA_HOST env var.
    api_clients.py reads this env var if set; otherwise falls back to
    MODEL_CONFIG default.
    """
    os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"

    from src.utils.sql_utils import clean_sql, execute_sql, compare_results
    from src.utils.schema_utils import get_schema
    from src.utils.prompt_templates import S0_DIRECT
    from src.pipelines.s1_schema_filter import S1SchemaFilter
    from src.pipelines.s2_decompose import S2Decompose
    from src.pipelines.s3_skeleton import S3Skeleton
    from src.pipelines.s4_din_sql import S4DinSQL
    from src.pipelines.base import BasePipeline
    from src.utils.api_clients import call_llm  # noqa

    PIPELINES = {
        "s1": S1SchemaFilter, "s2": S2Decompose,
        "s3": S3Skeleton, "s4": S4DinSQL,
    }

    cfg = BENCHMARKS[benchmark]
    db_dir = PROJECT_ROOT / cfg["db_dir"]

    class S0Direct(BasePipeline):
        name = "s0_direct"
        def __init__(self, model_name, seed=None):
            super().__init__(model_name)
            self._seed = seed
        def run(self, question, schema, evidence="", db_path=""):
            self._reset()
            evidence_section = "Evidence: " + evidence if evidence else ""
            prompt = S0_DIRECT.format(
                schema=schema, question=question, evidence_section=evidence_section,
            )
            raw = self._call_llm(prompt, "direct_generate", seed=self._seed)
            return self._make_result(clean_sql(raw))

    if strategy == "s0":
        pipeline = S0Direct(model, seed=SEED)
    else:
        pipeline = PIPELINES[strategy](model)
        pipeline._seed = SEED

    out_path = RESULTS_DIR / (
        f"v21_xlingual_{benchmark}_{model}_{strategy}_chunk{chunk_id}_seed{SEED}.json"
    )
    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(query_subset):
            print(f"[{model}/{strategy}/{benchmark}/c{chunk_id}] SKIP (done)", flush=True)
            return {"path": str(out_path), "skipped": True}

    results = []
    correct = 0
    t0 = time.time()
    n = len(query_subset)
    for i, q in enumerate(query_subset):
        db_id = q.get("db_id", "")
        question = q.get("question", "")
        gold_sql = q.get("query", q.get("SQL", ""))
        evidence = q.get("evidence", "") or ""
        db_path = str(db_dir / db_id / f"{db_id}.sqlite")

        if not Path(db_path).exists():
            results.append({
                "query_idx": q.get("query_idx", i), "db_id": db_id,
                "correct": False, "error": "db_not_found",
            })
            continue

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            results.append({
                "query_idx": q.get("query_idx", i), "db_id": db_id,
                "correct": False, "error": f"gold_err: {gold_err[:100]}",
            })
            continue

        schema = get_schema(db_path)
        try:
            r = pipeline.run(question, schema, evidence=evidence, db_path=db_path)
        except Exception as e:
            results.append({
                "query_idx": q.get("query_idx", i), "db_id": db_id,
                "correct": False, "error": f"pipe_err: {str(e)[:120]}",
            })
            continue

        pred_sql = r.get("pred_sql", "")
        if pred_sql and "SELECT" in pred_sql.upper():
            pred_result, pred_err = execute_sql(db_path, pred_sql)
        else:
            pred_result, pred_err = None, "no SQL"

        is_correct = compare_results(gold_result, pred_result)
        if is_correct:
            correct += 1

        results.append({
            "query_idx": q.get("query_idx", i), "db_id": db_id,
            "question": question[:120], "gold_sql": gold_sql[:200],
            "pred_sql": pred_sql[:200], "correct": is_correct,
            "error": pred_err[:100] if pred_err else None,
            "call_count": r.get("call_count", 0),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"[{model}/{strategy}/{benchmark}/c{chunk_id}] "
                  f"{i+1}/{n} acc={acc:.1f}% {rate:.0f} q/min", flush=True)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0
    output = {
        "metadata": {
            "model": model, "strategy": strategy, "benchmark": benchmark,
            "chunk_id": chunk_id, "port": port, "seed": SEED,
            "n_queries": len(results), "accuracy": round(accuracy, 4),
            "correct": correct, "elapsed_min": round(elapsed / 60, 2),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[{model}/{strategy}/{benchmark}/c{chunk_id}] DONE "
          f"acc={accuracy*100:.2f}% in {elapsed/60:.1f}min", flush=True)
    return {"path": str(out_path), "accuracy": accuracy, "n": len(results)}


def merge_chunks(model: str, strategy: str, benchmark: str) -> dict:
    """After all chunks done, merge into a single result file."""
    chunks = sorted(RESULTS_DIR.glob(
        f"v21_xlingual_{benchmark}_{model}_{strategy}_chunk*_seed{SEED}.json"
    ))
    if not chunks:
        return {}
    all_results = []
    total_correct = 0
    total_elapsed = 0.0
    for c in chunks:
        d = json.load(open(c))
        all_results.extend(d["results"])
        total_correct += d["metadata"]["correct"]
        total_elapsed += d["metadata"]["elapsed_min"]
    accuracy = total_correct / len(all_results) if all_results else 0
    merged = {
        "metadata": {
            "model": model, "strategy": strategy, "benchmark": benchmark,
            "seed": SEED, "n_queries": len(all_results),
            "accuracy": round(accuracy, 4), "correct": total_correct,
            "elapsed_min_total": round(total_elapsed, 2),
            "n_chunks": len(chunks),
        },
        "results": all_results,
    }
    out_path = RESULTS_DIR / (
        f"v21_xlingual_{benchmark}_{model}_{strategy}_seed{SEED}.json"
    )
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  MERGED {model}/{strategy}/{benchmark}: acc={accuracy*100:.2f}%", flush=True)
    return merged


def load_benchmark(benchmark: str) -> list:
    cfg = BENCHMARKS[benchmark]
    p = PROJECT_ROOT / cfg["queries_path"]
    return json.load(open(p))


def main():
    print(f"=== V2.1 CROSS-LINGUAL (4 GPU PARALLEL) ===")

    grand_t0 = time.time()
    benchmarks_to_run = list(BENCHMARKS.keys())

    for benchmark in benchmarks_to_run:
        print(f"\n{'#'*70}\n# BENCHMARK: {benchmark}\n{'#'*70}")
        queries = load_benchmark(benchmark)
        for i, q in enumerate(queries):
            q["query_idx"] = i
        n = len(queries)
        midpoint = n // 2
        chunks = {0: queries[:midpoint], 1: queries[midpoint:]}
        print(f"  loaded {n} queries, split: chunk0={len(chunks[0])}, "
              f"chunk1={len(chunks[1])}")

        # Build job list: (model, strategy, chunk_id, port, queries)
        jobs = []
        for model, ports in MODEL_ENDPOINTS.items():
            for strategy in STRATEGIES:
                for chunk_id, port in enumerate(ports):
                    jobs.append((model, strategy, chunk_id, port, chunks[chunk_id]))

        # Run all jobs but cap concurrency to 4 (one per GPU)
        with ProcessPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(run_chunk, model, strategy, benchmark, port,
                          qs, chunk_id): (model, strategy, chunk_id)
                for (model, strategy, chunk_id, port, qs) in jobs
            }
            for fut in as_completed(futures):
                model, strategy, chunk_id = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  CRASH {model}/{strategy}/c{chunk_id}: {e}", flush=True)

        # Merge chunks
        print(f"\n  --- merging chunks for {benchmark} ---")
        for model in MODEL_ENDPOINTS:
            for strategy in STRATEGIES:
                merge_chunks(model, strategy, benchmark)

    elapsed = time.time() - grand_t0
    print(f"\n=== ALL BENCHMARKS DONE in {elapsed/3600:.2f} hours ===")


if __name__ == "__main__":
    main()
