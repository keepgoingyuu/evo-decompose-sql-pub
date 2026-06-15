"""V2.1 GPT-4.1 補日文：cross-lingual 補強。

Run: GPT-4.1 × 5 strategies × MultiSpider JA dev (1034 queries) × seed 42
Cost: ~$50 (estimated)
Time: ~3 hours
Endpoint: OpenAI API (api_clients gpt41 config)
Output: results/q1_2026_04/main/v21_gpt41_japanese_<strategy>_seed42.json
"""
import sys
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

assert os.environ.get("OPENAI_API_KEY", "").startswith("sk-"), \
    "OPENAI_API_KEY missing or invalid in .env"

from src.utils.sql_utils import clean_sql, execute_sql, compare_results
from src.utils.schema_utils import get_schema
from src.utils.prompt_templates import S0_DIRECT
from src.pipelines.s1_schema_filter import S1SchemaFilter
from src.pipelines.s2_decompose import S2Decompose
from src.pipelines.s3_skeleton import S3Skeleton
from src.pipelines.s4_din_sql import S4DinSQL
from src.pipelines.base import BasePipeline
from src.utils.api_clients import call_llm  # noqa

MODEL = "gpt41"
LANG = "ja"
SEED = 42
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]

QUERIES_PATH = PROJECT_ROOT / "data" / "multispider" / "dataset" / "multispider" / "with_english_value" / "dev_ja.json"
DB_DIR = PROJECT_ROOT / "data" / "spider" / "spider_data" / "spider_data" / "database"
RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_04" / "main"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINES = {
    "s1": S1SchemaFilter,
    "s2": S2Decompose,
    "s3": S3Skeleton,
    "s4": S4DinSQL,
}


class S0Direct(BasePipeline):
    name = "s0_direct"

    def __init__(self, model_name, seed=None):
        super().__init__(model_name)
        self._seed = seed

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = S0_DIRECT.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        raw = self._call_llm(prompt, "direct_generate", seed=self._seed)
        return self._make_result(clean_sql(raw))


def load_queries() -> list:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"MultiSpider JA dev not found at {QUERIES_PATH}")
    queries = json.load(open(QUERIES_PATH))
    return queries


def run_one_strategy(strategy: str) -> dict:
    queries = load_queries()
    out_path = RESULTS_DIR / f"v21_gpt41_japanese_{strategy}_seed{SEED}.json"
    partial_path = RESULTS_DIR / f"v21_gpt41_japanese_{strategy}_seed{SEED}_partial.json"

    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(queries):
            print(f"[gpt41/{strategy}] SKIP (done)", flush=True)
            return existing

    if strategy == "s0":
        pipeline = S0Direct(MODEL, seed=SEED)
    else:
        pipeline = PIPELINES[strategy](MODEL)
        pipeline._seed = SEED

    results = []
    done_idx: set = set()
    if partial_path.exists():
        try:
            partial = json.load(open(partial_path))
            results = partial.get("results", [])
            done_idx = {r["query_idx"] for r in results}
            print(f"[gpt41/{strategy}] resuming from {len(done_idx)} done", flush=True)
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
        gold_sql = q.get("query", q.get("SQL", ""))
        evidence = q.get("evidence", "") or ""
        db_path = str(DB_DIR / db_id / f"{db_id}.sqlite")

        if not Path(db_path).exists():
            results.append({
                "query_idx": idx, "db_id": db_id,
                "correct": False, "error": "db_not_found",
            })
            continue

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            results.append({
                "query_idx": idx, "db_id": db_id,
                "correct": False, "error": f"gold_err: {gold_err[:100]}",
            })
            continue

        schema = get_schema(db_path)
        try:
            r = pipeline.run(question, schema, evidence=evidence, db_path=db_path)
        except Exception as e:
            results.append({
                "query_idx": idx, "db_id": db_id,
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
            "query_idx": idx, "db_id": db_id,
            "question": question[:120], "gold_sql": gold_sql[:200],
            "pred_sql": pred_sql[:200], "correct": is_correct,
            "error": pred_err[:100] if pred_err else None,
            "call_count": r.get("call_count", 0),
        })

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100 if results else 0
            done_now = len(results) - len(done_idx)
            rate = done_now / elapsed * 60 if elapsed > 0 else 0
            print(f"[gpt41/{strategy}] {idx+1}/{len(queries)} acc={acc:.1f}% {rate:.1f} q/min", flush=True)
            with open(partial_path, "w") as f:
                json.dump({"results": results}, f, ensure_ascii=False)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0
    output = {
        "metadata": {
            "model": MODEL, "strategy": strategy, "lang": LANG, "seed": SEED,
            "n_queries": len(results), "accuracy": round(accuracy, 4),
            "correct": correct, "elapsed_min": round(elapsed / 60, 2),
            "benchmark": "MultiSpider JA dev",
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if partial_path.exists():
        partial_path.unlink()
    print(f"[gpt41/{strategy}] DONE acc={accuracy*100:.2f}% in {elapsed/60:.1f}min", flush=True)
    return output


def main():
    print(f"=== V2.1 GPT-4.1 JAPANESE (seed={SEED}) ===")
    print(f"Queries: {QUERIES_PATH}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Output dir: {RESULTS_DIR}")

    summary = {}
    grand_t0 = time.time()
    # Run strategies sequentially (OpenAI API is rate-limited; parallel may not help)
    for strategy in STRATEGIES:
        try:
            r = run_one_strategy(strategy)
            summary[strategy] = r["metadata"]["accuracy"]
        except Exception as e:
            print(f"[gpt41/{strategy}] CRASH: {e}", flush=True)
            summary[strategy] = None

    elapsed = time.time() - grand_t0
    print(f"\n=== DONE in {elapsed/3600:.2f}h ===")
    print("=== ACCURACY ===")
    for s in STRATEGIES:
        a = summary.get(s)
        print(f"  {s}: " + (f"{a*100:.2f}%" if a is not None else "X"))


if __name__ == "__main__":
    main()
