"""V2.1 cross-lingual 日文補強 (參數化版, 不限 gpt41)。

由 run_v21_gpt41_japanese.py 複製改造: 模型/策略改為命令列參數,
讓 DeepSeek / Gemini 也能跑 MultiSpider JA (補 17 格缺格中的日文部分)。
原 run_v21_gpt41_japanese.py 不動。

Run: <model> × <strategies> × MultiSpider JA dev (1034 queries) × seed 42
Output: results/q1_2026_04/main/v21_<model>_japanese_<strategy>_seed42.json

用法:
    python scripts/run_v21_xlingual_japanese.py deepseek
    python scripts/run_v21_xlingual_japanese.py gemini --strategies s0 s1 s2 s3 s4
    python scripts/run_v21_xlingual_japanese.py deepseek --strategies s2   # 只跑單一策略
"""
import sys
import json
import os
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

from src.utils.sql_utils import clean_sql, execute_sql, compare_results
from src.utils.schema_utils import get_schema
from src.utils.prompt_templates import S0_DIRECT
from src.pipelines.s1_schema_filter import S1SchemaFilter
from src.pipelines.s2_decompose import S2Decompose
from src.pipelines.s3_skeleton import S3Skeleton
from src.pipelines.s4_din_sql import S4DinSQL
from src.pipelines.base import BasePipeline
from src.utils.api_clients import call_llm  # noqa

# --- 各模型所需金鑰 (跑前斷言, 避免跑到一半才發現缺 key) ---
KEY_REQUIRED = {
    "gpt41": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

LANG = "ja"
SEED = 42

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
    return json.load(open(QUERIES_PATH))


def _make_pipeline(model: str, strategy: str):
    if strategy == "s0":
        p = S0Direct(model, seed=SEED)
    else:
        p = PIPELINES[strategy](model)
        p._seed = SEED
    return p


def _process_query(idx, q, pipeline):
    """處理單題 (供並發 worker 呼叫). 回傳 result dict。"""
    db_id = q.get("db_id", "")
    question = q.get("question", "")
    gold_sql = q.get("query", q.get("SQL", ""))
    evidence = q.get("evidence", "") or ""
    db_path = str(DB_DIR / db_id / f"{db_id}.sqlite")

    if not Path(db_path).exists():
        return {"query_idx": idx, "db_id": db_id, "correct": False, "error": "db_not_found"}

    gold_result, gold_err = execute_sql(db_path, gold_sql)
    if gold_err:
        return {"query_idx": idx, "db_id": db_id, "correct": False, "error": f"gold_err: {gold_err[:100]}"}

    schema = get_schema(db_path)
    try:
        r = pipeline.run(question, schema, evidence=evidence, db_path=db_path)
    except Exception as e:
        return {"query_idx": idx, "db_id": db_id, "correct": False, "error": f"pipe_err: {str(e)[:120]}"}

    pred_sql = r.get("pred_sql", "")
    if pred_sql and "SELECT" in pred_sql.upper():
        pred_result, pred_err = execute_sql(db_path, pred_sql)
    else:
        pred_result, pred_err = None, "no SQL"

    is_correct = compare_results(gold_result, pred_result)
    return {
        "query_idx": idx, "db_id": db_id,
        "question": question[:120], "gold_sql": gold_sql[:200],
        "pred_sql": pred_sql[:200], "correct": is_correct,
        "error": pred_err[:100] if pred_err else None,
        "call_count": r.get("call_count", 0),
        "gen_time": r.get("gen_time"),
    }


def run_one_strategy(model: str, strategy: str, limit: int = 0, workers: int = 8) -> dict:
    queries = load_queries()
    smoke = bool(limit)
    if limit:
        queries = queries[:limit]
    sfx = "_smoke" if smoke else ""
    out_path = RESULTS_DIR / f"v21_{model}_japanese_{strategy}_seed{SEED}{sfx}.json"
    partial_path = RESULTS_DIR / f"v21_{model}_japanese_{strategy}_seed{SEED}{sfx}_partial.json"

    if out_path.exists():
        existing = json.load(open(out_path))
        if existing.get("metadata", {}).get("n_queries") == len(queries):
            print(f"[{model}/{strategy}] SKIP (done)", flush=True)
            return existing

    results = []
    done_idx: set = set()
    if partial_path.exists():
        try:
            partial = json.load(open(partial_path))
            results = partial.get("results", [])
            done_idx = {r["query_idx"] for r in results}
            print(f"[{model}/{strategy}] resuming from {len(done_idx)} done", flush=True)
        except Exception:
            results, done_idx = [], set()

    correct = sum(1 for r in results if r.get("correct"))
    t0 = time.time()
    lock = threading.Lock()
    n_total = len(queries)

    # 每個 worker 一個 pipeline 實例 (避免 thread 間共享狀態衝突)
    worker_pipelines = [_make_pipeline(model, strategy) for _ in range(workers)]
    todo = [(idx, q) for idx, q in enumerate(queries) if idx not in done_idx]
    print(f"[{model}/{strategy}] {len(todo)} 題待跑, {workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for wi, (idx, q) in enumerate(todo):
            wp = worker_pipelines[wi % workers]
            futures[executor.submit(_process_query, idx, q, wp)] = idx
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                if r.get("correct"):
                    correct += 1
                done_count += 1
                if done_count % 50 == 0:
                    elapsed = time.time() - t0
                    acc = correct / len(results) * 100 if results else 0
                    rate = done_count / elapsed * 60 if elapsed > 0 else 0
                    print(f"[{model}/{strategy}] {len(results)}/{n_total} acc={acc:.1f}% {rate:.1f} q/min", flush=True)
                    with open(partial_path, "w") as f:
                        json.dump({"results": results}, f, ensure_ascii=False)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0
    output = {
        "metadata": {
            "model": model, "strategy": strategy, "lang": LANG, "seed": SEED,
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
    print(f"[{model}/{strategy}] DONE acc={accuracy*100:.2f}% in {elapsed/60:.1f}min", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="MultiSpider JA cross-lingual runner (參數化模型)")
    parser.add_argument("model", choices=["gpt41", "deepseek", "gemini"])
    parser.add_argument("--strategies", nargs="+", default=["s0", "s1", "s2", "s3", "s4"],
                        choices=["s0", "s1", "s2", "s3", "s4"])
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 題的 smoke test (輸出加 _smoke 後綴, 不覆蓋正式檔)")
    parser.add_argument("--workers", type=int, default=8,
                        help="並發 worker 數 (default 8; API 是 I/O 等待, 並發大幅加速)")
    args = parser.parse_args()

    # 跑前斷言金鑰存在 (對應模型)
    key = KEY_REQUIRED[args.model]
    assert os.environ.get(key), f"{key} missing in .env (model={args.model} 需要)"

    print(f"=== V2.1 {args.model.upper()} JAPANESE (seed={SEED}) ===")
    print(f"Queries: {QUERIES_PATH}")
    print(f"Strategies: {args.strategies}")

    summary = {}
    grand_t0 = time.time()
    for strategy in args.strategies:
        try:
            r = run_one_strategy(args.model, strategy, limit=args.limit, workers=args.workers)
            summary[strategy] = r["metadata"]["accuracy"]
        except Exception as e:
            print(f"[{args.model}/{strategy}] CRASH: {e}", flush=True)
            summary[strategy] = None

    elapsed = time.time() - grand_t0
    print(f"\n=== DONE in {elapsed/3600:.2f}h ===")
    for s in args.strategies:
        a = summary.get(s)
        print(f"  {s}: " + (f"{a*100:.2f}%" if a is not None else "X"))


if __name__ == "__main__":
    main()
