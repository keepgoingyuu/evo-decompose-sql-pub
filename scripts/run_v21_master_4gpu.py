"""V2.1 master pipeline runner: 4-GPU fully utilized.

Pipeline:
  All remaining work × 4 GPU parallel (Qwen×2 + Gemma×2 instances).
  - Skip already-completed chunks (resumable via existing files).
  - Run BIRD remainder + Spider EN + CSpider ZH-TW + MultiSpider JA
    × 5 strategy × Qwen + Gemma all interleaved.
  - 4 worker pool, each pinned to one Ollama port (one GPU).

Required: scripts/start_ollama_4gpu.sh already started (4 instances).

Output: results/q1_2026_04/main/v21_<benchmark>_<model>_<strategy>_chunk<N>_seed42.json
        merged into v21_<benchmark>_<model>_<strategy>_seed42.json
"""
import sys
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Each model has 2 ollama endpoints (one per GPU)
MODEL_ENDPOINTS = {
    "qwen25c7b": [11434, 11437],   # GPU 0 + GPU 2
    "gemma4":    [11435, 11438],   # GPU 1 + GPU 3
}
SEED = 42
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]

RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_04" / "main"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark configs
BENCHMARKS = {
    "bird_modchal": {
        "queries_path": "data/bird/dev_20240627/dev.json",
        "filter": lambda q: q.get("difficulty") in ("moderate", "challenging"),
        "db_dir": "data/bird/dev_20240627/dev_databases",
        "gold_key": "SQL",
        "evidence_key": "evidence",
        "id_prefix": "v21_bird_main",  # use existing namespace
    },
    "spider_en": {
        "queries_path": "data/spider/spider_data/spider_data/dev.json",
        "filter": None,
        "db_dir": "data/spider/spider_data/spider_data/database",
        "gold_key": "query",
        "evidence_key": None,
        "id_prefix": "v21_xlingual_spider_en",
    },
    "cspider_zh_tw": {
        "queries_path": "data/cspider/dev_tw.json",
        "filter": None,
        "db_dir": "data/spider/spider_data/spider_data/database",
        "gold_key": "query",
        "evidence_key": None,
        "id_prefix": "v21_xlingual_cspider_zh_tw",
    },
    "multispider_ja": {
        "queries_path": "data/multispider/dataset/multispider/with_english_value/dev_ja.json",
        "filter": None,
        "db_dir": "data/spider/spider_data/spider_data/database",
        "gold_key": "query",
        "evidence_key": None,
        "id_prefix": "v21_xlingual_multispider_ja",
    },
}


def chunk_filename(benchmark: str, model: str, strategy: str, chunk_id: int) -> Path:
    cfg = BENCHMARKS[benchmark]
    return RESULTS_DIR / f"{cfg['id_prefix']}_{model}_{strategy}_chunk{chunk_id}_seed{SEED}.json"


def merged_filename(benchmark: str, model: str, strategy: str) -> Path:
    cfg = BENCHMARKS[benchmark]
    return RESULTS_DIR / f"{cfg['id_prefix']}_{model}_{strategy}_seed{SEED}.json"


def load_queries(benchmark: str) -> list:
    cfg = BENCHMARKS[benchmark]
    p = PROJECT_ROOT / cfg["queries_path"]
    qs = json.load(open(p))
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    return qs


def is_chunk_done(benchmark: str, model: str, strategy: str, chunk_id: int, expected_n: int) -> bool:
    """Chunk is done if file exists with matching n_queries."""
    f = chunk_filename(benchmark, model, strategy, chunk_id)
    if not f.exists():
        return False
    try:
        d = json.load(open(f))
        return d.get("metadata", {}).get("n_queries") == expected_n
    except Exception:
        return False


def is_merged_done(benchmark: str, model: str, strategy: str, expected_n: int) -> bool:
    f = merged_filename(benchmark, model, strategy)
    if not f.exists():
        return False
    try:
        d = json.load(open(f))
        return d.get("metadata", {}).get("n_queries") == expected_n
    except Exception:
        return False


def run_chunk(model: str, strategy: str, benchmark: str, port: int,
              query_subset: list, chunk_id: int) -> dict:
    """Worker: run one (model, strategy, benchmark, chunk) on a specific Ollama port."""
    from src.utils.sql_utils import clean_sql, execute_sql, compare_results
    from src.utils.schema_utils import get_schema
    from src.utils.prompt_templates import S0_DIRECT
    from src.pipelines.s1_schema_filter import S1SchemaFilter
    from src.pipelines.s2_decompose import S2Decompose
    from src.pipelines.s3_skeleton import S3Skeleton
    from src.pipelines.s4_din_sql import S4DinSQL
    from src.pipelines.base import BasePipeline
    from src.utils import api_clients

    # Monkey-patch model URL to specific port
    if model in api_clients.MODEL_CONFIG:
        old_url = api_clients.MODEL_CONFIG[model].get("url", "")
        api_path = "/api/" + old_url.rsplit("/api/", 1)[1] if "/api/" in old_url else "/api/chat"
        api_clients.MODEL_CONFIG[model]["url"] = f"http://127.0.0.1:{port}{api_path}"

    PIPELINES = {"s1": S1SchemaFilter, "s2": S2Decompose,
                 "s3": S3Skeleton, "s4": S4DinSQL}

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

    cfg = BENCHMARKS[benchmark]
    db_dir = PROJECT_ROOT / cfg["db_dir"]
    out_path = chunk_filename(benchmark, model, strategy, chunk_id)
    partial_path = out_path.with_suffix(".partial.json")

    # Skip if done
    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(query_subset):
            print(f"[{benchmark}/{model}/{strategy}/c{chunk_id}@{port}] SKIP", flush=True)
            return {"skipped": True}

    # Resume from partial
    results = []
    done_idx: set = set()
    if partial_path.exists():
        try:
            partial = json.load(open(partial_path))
            results = partial.get("results", [])
            done_idx = {r["query_idx"] for r in results}
        except Exception:
            results = []
            done_idx = set()

    correct = sum(1 for r in results if r.get("correct"))
    t0 = time.time()
    n = len(query_subset)
    for i, q in enumerate(query_subset):
        idx = q.get("query_idx", i)
        if idx in done_idx:
            continue

        db_id = q.get("db_id", "")
        question = q.get("question", "")
        gold_sql = q.get(cfg["gold_key"], "")
        evidence = (q.get(cfg["evidence_key"], "") if cfg["evidence_key"] else "") or ""
        db_path = str(db_dir / db_id / f"{db_id}.sqlite")

        if not Path(db_path).exists():
            results.append({"query_idx": idx, "db_id": db_id,
                            "correct": False, "error": "db_not_found"})
            continue

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            results.append({"query_idx": idx, "db_id": db_id,
                            "correct": False, "error": f"gold_err: {gold_err[:100]}"})
            continue

        schema = get_schema(db_path)
        try:
            r = pipeline.run(question, schema, evidence=evidence, db_path=db_path)
        except Exception as e:
            results.append({"query_idx": idx, "db_id": db_id,
                            "correct": False, "error": f"pipe_err: {str(e)[:120]}"})
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
            "query_idx": idx, "db_id": db_id,
            "question": question[:120], "gold_sql": gold_sql[:200],
            "pred_sql": pred_sql[:200], "correct": is_correct,
            "error": pred_err[:100] if pred_err else None,
            "call_count": r.get("call_count", 0),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"[{benchmark}/{model}/{strategy}/c{chunk_id}@{port}] "
                  f"{i+1}/{n} acc={acc:.1f}% {rate:.1f} q/min", flush=True)
            with open(partial_path, "w") as f:
                json.dump({"results": results}, f, ensure_ascii=False)

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
    if partial_path.exists():
        partial_path.unlink()
    print(f"[{benchmark}/{model}/{strategy}/c{chunk_id}@{port}] DONE "
          f"acc={accuracy*100:.2f}% in {elapsed/60:.1f}min", flush=True)
    return {"path": str(out_path), "accuracy": accuracy, "n": len(results)}


def merge_chunks(benchmark: str, model: str, strategy: str) -> None:
    chunks = sorted(RESULTS_DIR.glob(
        f"{BENCHMARKS[benchmark]['id_prefix']}_{model}_{strategy}_chunk*_seed{SEED}.json"
    ))
    if not chunks:
        return
    all_results = []
    total_correct = 0
    for c in chunks:
        d = json.load(open(c))
        all_results.extend(d["results"])
        total_correct += d["metadata"]["correct"]
    accuracy = total_correct / len(all_results) if all_results else 0
    merged = {
        "metadata": {
            "model": model, "strategy": strategy, "benchmark": benchmark,
            "seed": SEED, "n_queries": len(all_results),
            "accuracy": round(accuracy, 4), "correct": total_correct,
            "n_chunks": len(chunks),
        },
        "results": all_results,
    }
    out_path = merged_filename(benchmark, model, strategy)
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  MERGED {benchmark}/{model}/{strategy}: acc={accuracy*100:.2f}%", flush=True)


def main():
    print(f"=== V2.1 MASTER 4-GPU PIPELINE (seed={SEED}) ===")
    print(f"Models: Qwen on ports 11434+11437 (GPU 0+2)")
    print(f"        Gemma on ports 11435+11438 (GPU 1+3)")
    print(f"Skip-if-done: yes (any chunk file with matching n_queries)")

    grand_t0 = time.time()
    benchmarks = list(BENCHMARKS.keys())
    print(f"\nBenchmarks: {benchmarks}")

    for benchmark in benchmarks:
        print(f"\n{'#'*70}\n# BENCHMARK: {benchmark}\n{'#'*70}")
        queries = load_queries(benchmark)
        for i, q in enumerate(queries):
            q["query_idx"] = i
        n = len(queries)
        midpoint = n // 2
        chunks = {0: queries[:midpoint], 1: queries[midpoint:]}
        print(f"  loaded {n} queries: chunk0={len(chunks[0])}, chunk1={len(chunks[1])}")

        # Build interleaved job list
        jobs = []
        for strategy in STRATEGIES:
            for chunk_id in (0, 1):
                for model, ports in MODEL_ENDPOINTS.items():
                    port = ports[chunk_id]
                    # Skip if chunk already done
                    if is_chunk_done(benchmark, model, strategy, chunk_id, len(chunks[chunk_id])):
                        continue
                    jobs.append((model, strategy, chunk_id, port, chunks[chunk_id]))

        if not jobs:
            print(f"  ALL CHUNKS DONE for {benchmark}, merging...")
        else:
            print(f"  {len(jobs)} chunks remaining (skipped {2*5*2 - len(jobs)} done)")
            with ProcessPoolExecutor(max_workers=4) as ex:
                futures = {
                    ex.submit(run_chunk, m, s, benchmark, p, qs, c): (m, s, c)
                    for (m, s, c, p, qs) in jobs
                }
                for fut in as_completed(futures):
                    m, s, c = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"  CRASH {m}/{s}/c{c}: {e}", flush=True)

        # Merge chunks
        print(f"\n  --- merging {benchmark} ---")
        for model in MODEL_ENDPOINTS:
            for strategy in STRATEGIES:
                merge_chunks(benchmark, model, strategy)

    elapsed = time.time() - grand_t0
    print(f"\n=== ALL BENCHMARKS DONE in {elapsed/3600:.2f}h ===")


if __name__ == "__main__":
    main()
