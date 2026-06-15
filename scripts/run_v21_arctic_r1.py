"""V2.1 Arctic-Text2SQL-R1 7B RL baseline runner (RQ5).

Loads Snowflake/Arctic-Text2SQL-R1-7B (fp16 safetensors) via transformers
with device_map="auto" so it spreads across the available GPUs on DGX.

Runs S0 (single-pass) on 4 benchmarks:
  - BIRD mod+chal (609)
  - Spider EN (1034)
  - CSpider ZH-TW (1034)
  - MultiSpider JA (1034)

Total ~3711 queries × ~7s each on 2× V100 fp16 ≈ 7-8 hours.
Resumable per-benchmark.

Output: results/q1_2026_04/main/v21_arctic_r1_<benchmark>_s0_seed42.json
"""
import sys
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Block transformers from auto-loading datasets / from updating
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def run_benchmark(model, tokenizer, benchmark: str) -> dict:
    cfg = BENCHMARKS[benchmark]
    queries = load_queries(benchmark)
    out_path = RESULTS_DIR / f"v21_arctic_r1_{benchmark}_s0_seed{SEED}.json"
    partial_path = RESULTS_DIR / f"v21_arctic_r1_{benchmark}_s0_seed{SEED}_partial.json"

    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(queries):
            print(f"[arctic_r1/{benchmark}] SKIP (done, n={len(queries)})", flush=True)
            return existing

    results = []
    done_idx: set = set()
    if partial_path.exists():
        try:
            partial = json.load(open(partial_path))
            results = partial.get("results", [])
            done_idx = {r["query_idx"] for r in results}
            print(f"[arctic_r1/{benchmark}] resuming from {len(done_idx)}/{len(queries)}", flush=True)
        except Exception:
            results = []
            done_idx = set()

    correct = sum(1 for r in results if r.get("correct"))
    t0 = time.time()
    for idx, q in enumerate(queries):
        if idx in done_idx:
            continue
        db_id = q.get("db_id", "")
        question = q.get("question", "")
        gold_sql = q.get(cfg["gold_key"], "")
        evidence = (q.get(cfg["evidence_key"], "") if cfg["evidence_key"] else "") or ""
        db_path = str(Path(cfg["db_dir"]) / db_id / f"{db_id}.sqlite")

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
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = S0_DIRECT.format(
            schema=schema, question=question, evidence_section=evidence_section,
        )

        try:
            raw = generate_sql(model, tokenizer, prompt, max_new_tokens=256)
            pred_sql = clean_sql(raw)
        except Exception as e:
            results.append({"query_idx": idx, "db_id": db_id,
                            "correct": False, "error": f"gen_err: {str(e)[:120]}"})
            continue

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
        })

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100
            done_now = len(results) - len(done_idx)
            rate = done_now / elapsed * 60 if elapsed > 0 else 0
            print(f"[arctic_r1/{benchmark}] {idx+1}/{len(queries)} "
                  f"acc={acc:.1f}% {rate:.1f} q/min", flush=True)
            with open(partial_path, "w") as f:
                json.dump({"results": results}, f, ensure_ascii=False)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0
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
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if partial_path.exists():
        partial_path.unlink()
    print(f"[arctic_r1/{benchmark}] DONE acc={accuracy*100:.2f}% in {elapsed/60:.1f}min", flush=True)
    return output


def main():
    print(f"=== V2.1 Arctic-Text2SQL-R1 7B (RL baseline, RQ5) ===")
    print(f"Loading {MODEL_ID} (fp16, device_map=auto)...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  loaded in {time.time()-t_load:.1f}s")
    print(f"  device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    summary = {}
    grand_t0 = time.time()
    for benchmark in BENCHMARKS:
        try:
            r = run_benchmark(model, tokenizer, benchmark)
            summary[benchmark] = r["metadata"]["accuracy"]
        except Exception as e:
            print(f"[arctic_r1/{benchmark}] CRASH: {type(e).__name__}: {e}", flush=True)
            summary[benchmark] = None

    elapsed = time.time() - grand_t0
    print(f"\n=== ALL DONE in {elapsed/3600:.2f}h ===")
    print("=== ACCURACY ===")
    for b, a in summary.items():
        print(f"  {b}: " + (f"{a*100:.2f}%" if a is not None else "X"))


if __name__ == "__main__":
    main()
