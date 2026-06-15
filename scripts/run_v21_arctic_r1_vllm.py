"""V2.1 Arctic-Text2SQL-R1 7B RL baseline runner — vLLM batched version.

50x+ faster than scripts/run_v21_arctic_r1.py (transformers single-batch).

Single GPU (V100 16GB fp16 fits 7B model + 4GB KV cache).
Batches all prompts per benchmark, runs vLLM offline inference, then
executes SQL serially (DB IO is the cheap part).

Usage on DGX (single-bench mode for 4-GPU data-parallel):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_v21_arctic_r1_vllm.py bird_modchal
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/run_v21_arctic_r1_vllm.py spider_en
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/run_v21_arctic_r1_vllm.py cspider_zh_tw
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u scripts/run_v21_arctic_r1_vllm.py multispider_ja

Or all 4 in one process (sequential): no arg.
"""
import sys
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# V100 / Volta: disable torch.compile (inductor backend crashes on Volta + vLLM 0.6.6)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

from vllm import LLM, SamplingParams

from src.utils.sql_utils import clean_sql, execute_sql, compare_results
from src.utils.schema_utils import get_schema
from src.utils.prompt_templates import S0_DIRECT

MODEL_ID = "Snowflake/Arctic-Text2SQL-R1-7B"
SEED = 42

RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_04" / "main"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
BIRD_DB_DIR = BIRD_DIR / "dev_databases"
SPIDER_DB_DIR = PROJECT_ROOT / "data" / "spider" / "spider_data" / "spider_data" / "database"

BENCHMARKS = {
    "bird_modchal": {
        "queries_path": str(BIRD_DIR / "dev.json"),
        "filter": lambda q: q.get("difficulty") in ("moderate", "challenging"),
        "db_dir": str(BIRD_DB_DIR),
        "gold_key": "SQL",
        "evidence_key": "evidence",
    },
    "spider_en": {
        "queries_path": str(PROJECT_ROOT / "data" / "spider" / "spider_data" / "spider_data" / "dev.json"),
        "filter": None,
        "db_dir": str(SPIDER_DB_DIR),
        "gold_key": "query",
        "evidence_key": None,
    },
    "cspider_zh_tw": {
        "queries_path": str(PROJECT_ROOT / "data" / "cspider" / "dev_tw.json"),
        "filter": None,
        "db_dir": str(SPIDER_DB_DIR),
        "gold_key": "query",
        "evidence_key": None,
    },
    "multispider_ja": {
        "queries_path": str(PROJECT_ROOT / "data" / "multispider" / "dataset" / "multispider" / "with_english_value" / "dev_ja.json"),
        "filter": None,
        "db_dir": str(SPIDER_DB_DIR),
        "gold_key": "query",
        "evidence_key": None,
    },
}


def load_queries(benchmark: str) -> list:
    cfg = BENCHMARKS[benchmark]
    qs = json.load(open(cfg["queries_path"]))
    if cfg["filter"]:
        qs = [q for q in qs if cfg["filter"](q)]
    return qs


def build_prompts(queries, cfg, tokenizer_chat_apply):
    """Pre-build all prompts. Returns list of (idx, prompt_text, q_meta)."""
    out = []
    for idx, q in enumerate(queries):
        db_id = q.get("db_id", "")
        question = q.get("question", "")
        gold_sql = q.get(cfg["gold_key"], "")
        evidence = (q.get(cfg["evidence_key"], "") if cfg["evidence_key"] else "") or ""
        db_path = str(Path(cfg["db_dir"]) / db_id / f"{db_id}.sqlite")
        if not Path(db_path).exists():
            out.append((idx, None, {
                "db_id": db_id, "question": question, "gold_sql": gold_sql,
                "db_path": db_path, "skip_reason": "db_not_found",
            }))
            continue
        try:
            schema = get_schema(db_path)
        except Exception as e:
            out.append((idx, None, {
                "db_id": db_id, "question": question, "gold_sql": gold_sql,
                "db_path": db_path, "skip_reason": f"schema_err:{str(e)[:60]}",
            }))
            continue
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = S0_DIRECT.format(
            schema=schema, question=question, evidence_section=evidence_section,
        )
        prompt_chat = tokenizer_chat_apply(prompt)
        out.append((idx, prompt_chat, {
            "db_id": db_id, "question": question, "gold_sql": gold_sql,
            "db_path": db_path,
        }))
    return out


def run_benchmark(llm, tokenizer_chat_apply, benchmark: str) -> dict:
    cfg = BENCHMARKS[benchmark]
    queries = load_queries(benchmark)
    out_path = RESULTS_DIR / f"v21_arctic_r1_{benchmark}_s0_seed{SEED}.json"
    partial_path = RESULTS_DIR / f"v21_arctic_r1_{benchmark}_s0_seed{SEED}_partial.json"

    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(queries):
            print(f"[arctic_r1/{benchmark}] SKIP (done, n={len(queries)})", flush=True)
            return existing

    print(f"[arctic_r1/{benchmark}] building {len(queries)} prompts...", flush=True)
    t0 = time.time()
    prompt_records = build_prompts(queries, cfg, tokenizer_chat_apply)
    valid_prompts = [(idx, p) for idx, p, _ in prompt_records if p is not None]
    print(f"  {len(valid_prompts)}/{len(queries)} valid prompts in {time.time()-t0:.1f}s", flush=True)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        seed=SEED,
    )
    print(f"[arctic_r1/{benchmark}] vLLM batch generating...", flush=True)
    t_gen = time.time()
    vllm_out = llm.generate([p for _, p in valid_prompts], sampling)
    gen_elapsed = time.time() - t_gen
    print(f"  generated {len(vllm_out)} in {gen_elapsed:.1f}s "
          f"({len(vllm_out)/gen_elapsed:.1f} q/s)", flush=True)

    pred_by_idx = {idx: out.outputs[0].text for (idx, _), out in zip(valid_prompts, vllm_out)}

    print(f"[arctic_r1/{benchmark}] executing SQL...", flush=True)
    t_exec = time.time()
    results = []
    correct = 0
    for idx, _, meta in prompt_records:
        if "skip_reason" in meta:
            results.append({"query_idx": idx, "db_id": meta["db_id"],
                            "correct": False, "error": meta["skip_reason"]})
            continue
        gold_result, gold_err = execute_sql(meta["db_path"], meta["gold_sql"])
        if gold_err:
            results.append({"query_idx": idx, "db_id": meta["db_id"],
                            "correct": False, "error": f"gold_err: {gold_err[:100]}"})
            continue
        raw = pred_by_idx.get(idx, "")
        pred_sql = clean_sql(raw)
        if pred_sql and "SELECT" in pred_sql.upper():
            pred_result, pred_err = execute_sql(meta["db_path"], pred_sql)
        else:
            pred_result, pred_err = None, "no SQL"
        is_correct = compare_results(gold_result, pred_result)
        if is_correct:
            correct += 1
        results.append({
            "query_idx": idx, "db_id": meta["db_id"],
            "question": meta["question"][:120], "gold_sql": meta["gold_sql"][:200],
            "pred_sql": pred_sql[:200], "correct": is_correct,
            "error": pred_err[:100] if pred_err else None,
        })

    exec_elapsed = time.time() - t_exec
    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0
    print(f"[arctic_r1/{benchmark}] DONE acc={accuracy*100:.2f}% "
          f"(gen={gen_elapsed:.0f}s exec={exec_elapsed:.0f}s total={elapsed:.0f}s)", flush=True)

    output = {
        "metadata": {
            "model": "arctic-text2sql-r1-7b",
            "model_id": MODEL_ID,
            "strategy": "s0",
            "benchmark": benchmark,
            "seed": SEED,
            "n_queries": len(results),
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "elapsed_min": round(elapsed / 60, 2),
            "gen_elapsed_s": round(gen_elapsed, 1),
            "engine": "vllm-0.6.6",
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if partial_path.exists():
        partial_path.unlink()
    return output


def main():
    print(f"=== V2.1 Arctic-Text2SQL-R1 7B (vLLM batched, RQ5) ===", flush=True)
    print(f"Loading {MODEL_ID} via vLLM (single GPU, fp16)...", flush=True)
    t_load = time.time()
    # V100 16GB single-GPU can't fit 7B fp16 (14GB) + KV cache.
    # Use tensor_parallel_size=2 → split model across 2 cards, 7GB each + KV cache room.
    # Setting via env: VLLM_TP_SIZE (default 2)
    tp_size = int(os.environ.get("VLLM_TP_SIZE", "2"))
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=True,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()

    def chat_apply(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    print(f"  vLLM loaded in {time.time()-t_load:.1f}s", flush=True)

    # Single-benchmark mode (for 4-GPU parallel) or all-bench sequential
    if len(sys.argv) >= 2:
        target = sys.argv[1]
        if target not in BENCHMARKS:
            print(f"Unknown benchmark: {target}. Choices: {list(BENCHMARKS)}")
            sys.exit(1)
        targets = [target]
    else:
        targets = list(BENCHMARKS.keys())

    summary = {}
    grand_t0 = time.time()
    for benchmark in targets:
        try:
            r = run_benchmark(llm, chat_apply, benchmark)
            summary[benchmark] = r["metadata"]["accuracy"]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[arctic_r1/{benchmark}] CRASH: {type(e).__name__}: {e}", flush=True)
            summary[benchmark] = None

    elapsed = time.time() - grand_t0
    print(f"\n=== DONE ({len(targets)} bench) in {elapsed/60:.1f}min ===", flush=True)
    for b, a in summary.items():
        print(f"  {b}: " + (f"{a*100:.2f}%" if a is not None else "X"), flush=True)


if __name__ == "__main__":
    main()
