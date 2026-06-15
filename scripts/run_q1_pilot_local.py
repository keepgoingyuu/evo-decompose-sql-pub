"""Q1-path pilot: 2 local models × 5 strategies × 50 BIRD query × seed=42.

Purpose: verify (a) all 5 strategies run end-to-end on new model versions,
(b) inverse correlation trend still appears (qwen25c7b weaker than gemma4
expected? we want to see).

Output: results/q1_pilot_local_<model>_<strategy>.json (resumable).
"""
import sys, json, time, os
import sqlite3
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_clients import call_llm, estimate_cost
from src.utils.sql_utils import clean_sql, execute_sql, compare_results
from src.utils.schema_utils import get_schema
from src.utils.prompt_templates import S0_DIRECT
from src.pipelines.s1_schema_filter import S1SchemaFilter
from src.pipelines.s2_decompose import S2Decompose
from src.pipelines.s3_skeleton import S3Skeleton
from src.pipelines.s4_din_sql import S4DinSQL
from src.pipelines.base import BasePipeline

PROJECT_ROOT = Path(__file__).parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
DB_DIR = BIRD_DIR / "dev_databases"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = ["qwen25c7b", "gemma4"]
PIPELINES = {
    "s1": S1SchemaFilter,
    "s2": S2Decompose,
    "s3": S3Skeleton,
    "s4": S4DinSQL,
}
SEED = 42
N_QUERIES = 50  # pilot subset


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


def load_bird_subset(n):
    """Load first N moderate+challenging BIRD queries deterministically."""
    full = json.load(open(BIRD_DIR / "dev.json"))
    subset = [q for q in full if q.get("difficulty") in ("moderate", "challenging")]
    return subset[:n]


def run_one(model, strategy, queries):
    """Run one (model, strategy) over queries. Resumable."""
    out_path = RESULTS_DIR / f"q1_pilot_local_{model}_{strategy}_seed{SEED}.json"
    partial_path = RESULTS_DIR / f"q1_pilot_local_{model}_{strategy}_seed{SEED}_partial.json"

    if out_path.exists():
        print(f"  [skip] {out_path.name} already done")
        return json.load(open(out_path))

    # Build pipeline
    if strategy == "s0":
        pipeline = S0Direct(model, seed=SEED)
    else:
        pipeline = PIPELINES[strategy](model)
        # We add seed via patching call_llm at runtime if pipeline doesn't accept it
        pipeline._seed = SEED

    # Resume from partial
    results = []
    done_idx = set()
    if partial_path.exists():
        try:
            partial = json.load(open(partial_path))
            results = partial.get("results", [])
            done_idx = {r["query_idx"] for r in results}
            print(f"  resuming from {len(done_idx)} done")
        except Exception:
            pass

    correct = sum(1 for r in results if r.get("correct"))
    t0 = time.time()
    for idx, q in enumerate(queries):
        if idx in done_idx:
            continue
        db_id = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "") or ""
        gold_sql = q["SQL"]
        db_path = str(DB_DIR / db_id / f"{db_id}.sqlite")

        if not Path(db_path).exists():
            results.append({
                "query_idx": idx, "db_id": db_id, "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": "",
                "correct": False, "error": "db_not_found",
            })
            continue

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            results.append({
                "query_idx": idx, "db_id": db_id, "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": "",
                "correct": False, "error": f"gold_error: {gold_err[:100]}",
            })
            continue

        schema = get_schema(db_path)
        try:
            r = pipeline.run(question, schema, evidence=evidence, db_path=db_path)
        except Exception as e:
            results.append({
                "query_idx": idx, "db_id": db_id, "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": "",
                "correct": False, "error": f"pipeline_error: {str(e)[:200]}",
            })
            continue

        pred_sql = r.get("pred_sql", "")
        if pred_sql and "SELECT" in pred_sql.upper():
            pred_result, pred_err = execute_sql(db_path, pred_sql)
        else:
            pred_result, pred_err = None, "no SQL generated"

        is_correct = compare_results(gold_result, pred_result)
        if is_correct:
            correct += 1

        results.append({
            "query_idx": idx, "db_id": db_id, "question": question[:120],
            "gold_sql": gold_sql[:200], "pred_sql": pred_sql[:200],
            "correct": is_correct, "error": pred_err[:100] if pred_err else None,
            "call_count": r.get("call_count", 0),
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100 if results else 0
            rate = (idx + 1 - len(done_idx)) / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(queries)}] acc {acc:.1f}%  {rate:.0f} q/min", flush=True)
            with open(partial_path, "w") as f:
                json.dump({"results": results}, f, ensure_ascii=False)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0

    output = {
        "metadata": {
            "model": model, "strategy": strategy,
            "seed": SEED, "n_queries": len(results),
            "correct": correct, "accuracy": round(accuracy, 4),
            "elapsed_min": round(elapsed / 60, 2),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial_path.exists():
        partial_path.unlink()

    print(f"  DONE {model} {strategy}  acc={accuracy*100:.1f}%  ({elapsed/60:.1f} min)")
    return output


def main():
    print(f"=== Q1 PILOT (LOCAL ONLY, seed={SEED}) ===")
    print(f"Models: {MODELS}  Strategies: s0,s1,s2,s3,s4  Queries: {N_QUERIES}")

    queries = load_bird_subset(N_QUERIES)
    print(f"Loaded {len(queries)} BIRD mod+chal queries")

    summary = {}
    for model in MODELS:
        for strategy in ["s0", "s1", "s2", "s3", "s4"]:
            print(f"\n>>> {model} × {strategy} <<<")
            r = run_one(model, strategy, queries)
            summary[f"{model}_{strategy}"] = r["metadata"]["accuracy"]

    print("\n=== PILOT SUMMARY (accuracy %) ===")
    for k, v in summary.items():
        print(f"  {k:30s}  {v*100:.1f}%")

    summary_path = RESULTS_DIR / f"q1_pilot_local_summary_seed{SEED}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
