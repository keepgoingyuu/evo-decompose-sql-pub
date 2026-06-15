"""V2.1 BIRD main, 4-GPU parallel.

For BIRD mod+chal 609 query × 5 strategy × seed 42:
  Qwen-A on GPU 0 (port 11434) — Qwen first half (queries 0..304)
  Qwen-B on GPU 2 (port 11437) — Qwen second half (queries 305..608)
  Gemma-A on GPU 1 (port 11435) — Gemma first half
  Gemma-B on GPU 3 (port 11438) — Gemma second half

Requires: scripts/start_ollama_4gpu.sh started.

Output: results/q1_2026_04/main/v21_bird_main_<model>_<strategy>_chunk<N>_seed42.json
        merged into v21_bird_main_<model>_<strategy>_seed42.json
"""
import sys
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Single-port concurrent Ollama: OLLAMA_NUM_PARALLEL=4 + MAX_LOADED_MODELS=2
# All workers point to the same port; Ollama internally batches and routes
# to the loaded model on the appropriate GPU.
MODEL_ENDPOINTS = {
    "qwen25c7b": [11434, 11434],  # 4 workers / 2 chunks all hit same port
    "gemma4":    [11434, 11434],
}
SEED = 42
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]

BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
DB_DIR = BIRD_DIR / "dev_databases"
RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_04" / "main"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_bird_modchal() -> list:
    full = json.load(open(BIRD_DIR / "dev.json"))
    return [q for q in full if q.get("difficulty") in ("moderate", "challenging")]


def run_chunk(model: str, strategy: str, port: int, query_subset: list, chunk_id: int) -> dict:
    """Worker: run one (model, strategy, chunk) on a specific Ollama port.

    We monkey-patch MODEL_CONFIG so the worker's call_llm targets the
    intended port without modifying api_clients.py.
    """
    from src.utils.sql_utils import clean_sql, execute_sql, compare_results
    from src.utils.schema_utils import get_schema
    from src.utils.prompt_templates import S0_DIRECT
    from src.pipelines.s1_schema_filter import S1SchemaFilter
    from src.pipelines.s2_decompose import S2Decompose
    from src.pipelines.s3_skeleton import S3Skeleton
    from src.pipelines.s4_din_sql import S4DinSQL
    from src.pipelines.base import BasePipeline
    from src.utils import api_clients

    # Monkey-patch the URL for this worker's model to point at the correct port
    if model in api_clients.MODEL_CONFIG:
        old_url = api_clients.MODEL_CONFIG[model].get("url", "")
        # Extract /api/... path from existing URL
        if "/api/" in old_url:
            api_path = "/api/" + old_url.rsplit("/api/", 1)[1]
        else:
            api_path = "/api/chat"
        api_clients.MODEL_CONFIG[model]["url"] = f"http://127.0.0.1:{port}{api_path}"
        print(f"  [worker pid={os.getpid()}] {model} -> http://127.0.0.1:{port}{api_path}",
              flush=True)

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

    out_path = RESULTS_DIR / (
        f"v21_bird_main_{model}_{strategy}_chunk{chunk_id}_seed{SEED}.json"
    )
    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(query_subset):
            print(f"[{model}/{strategy}/c{chunk_id}] SKIP (done)", flush=True)
            return {"skipped": True}

    results = []
    correct = 0
    t0 = time.time()
    n = len(query_subset)
    for i, q in enumerate(query_subset):
        db_id = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "") or ""
        gold_sql = q["SQL"]
        db_path = str(DB_DIR / db_id / f"{db_id}.sqlite")

        if not Path(db_path).exists():
            results.append({"query_idx": q.get("query_idx", i), "db_id": db_id,
                            "correct": False, "error": "db_not_found"})
            continue

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            results.append({"query_idx": q.get("query_idx", i), "db_id": db_id,
                            "correct": False, "error": f"gold_err: {gold_err[:100]}"})
            continue

        schema = get_schema(db_path)
        try:
            r = pipeline.run(question, schema, evidence=evidence, db_path=db_path)
        except Exception as e:
            results.append({"query_idx": q.get("query_idx", i), "db_id": db_id,
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
            "query_idx": q.get("query_idx", i), "db_id": db_id,
            "question": question[:120], "gold_sql": gold_sql[:200],
            "pred_sql": pred_sql[:200], "correct": is_correct,
            "error": pred_err[:100] if pred_err else None,
            "call_count": r.get("call_count", 0),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"[{model}/{strategy}/c{chunk_id}] "
                  f"{i+1}/{n} acc={acc:.1f}% {rate:.1f} q/min", flush=True)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0
    output = {
        "metadata": {
            "model": model, "strategy": strategy, "chunk_id": chunk_id, "port": port,
            "seed": SEED, "n_queries": len(results),
            "accuracy": round(accuracy, 4), "correct": correct,
            "elapsed_min": round(elapsed / 60, 2),
            "benchmark": "BIRD mod+chal",
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[{model}/{strategy}/c{chunk_id}] DONE "
          f"acc={accuracy*100:.2f}% in {elapsed/60:.1f}min", flush=True)
    return {"path": str(out_path), "accuracy": accuracy, "n": len(results)}


def merge_chunks(model: str, strategy: str) -> dict:
    chunks = sorted(RESULTS_DIR.glob(
        f"v21_bird_main_{model}_{strategy}_chunk*_seed{SEED}.json"
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
            "model": model, "strategy": strategy, "seed": SEED,
            "n_queries": len(all_results),
            "accuracy": round(accuracy, 4), "correct": total_correct,
            "elapsed_min_total": round(total_elapsed, 2),
            "n_chunks": len(chunks),
            "benchmark": "BIRD mod+chal",
        },
        "results": all_results,
    }
    out_path = RESULTS_DIR / f"v21_bird_main_{model}_{strategy}_seed{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  MERGED {model}/{strategy}: acc={accuracy*100:.2f}%", flush=True)
    return merged


def main():
    print(f"=== V2.1 BIRD MAIN (4-GPU PARALLEL) ===")
    queries = load_bird_modchal()
    for i, q in enumerate(queries):
        q["query_idx"] = i
    n = len(queries)
    midpoint = n // 2
    chunks = {0: queries[:midpoint], 1: queries[midpoint:]}
    print(f"  loaded {n} queries, split: chunk0={len(chunks[0])}, chunk1={len(chunks[1])}")

    # Build job list interleaved by model so all 4 GPUs are utilized from start.
    # Order:  qwen-c0, gemma-c0, qwen-c1, gemma-c1   (per strategy)
    jobs = []
    for strategy in STRATEGIES:
        for chunk_id in (0, 1):
            for model, ports in MODEL_ENDPOINTS.items():
                port = ports[chunk_id]
                jobs.append((model, strategy, chunk_id, port, chunks[chunk_id]))

    print(f"  total jobs: {len(jobs)} (4 workers parallel, interleaved)")
    print(f"  first 4 jobs (initial pool): " +
          ", ".join(f"{m}/{s}/c{c}@{p}" for (m, s, c, p, _) in jobs[:4]))

    grand_t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(run_chunk, model, strategy, port, qs, chunk_id):
                (model, strategy, chunk_id)
            for (model, strategy, chunk_id, port, qs) in jobs
        }
        for fut in as_completed(futures):
            model, strategy, chunk_id = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  CRASH {model}/{strategy}/c{chunk_id}: {e}", flush=True)

    print(f"\n  --- merging chunks ---")
    for model in MODEL_ENDPOINTS:
        for strategy in STRATEGIES:
            merge_chunks(model, strategy)

    elapsed = time.time() - grand_t0
    print(f"\n=== ALL DONE in {elapsed/3600:.2f}h ===")


if __name__ == "__main__":
    main()
