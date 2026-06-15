"""V2.1 BIRD main run: 2 local models × 5 strategies × 609 BIRD mod+chal × seed 42.

This script:
- Reuses run_one() logic from run_q1_pilot_local.py (proven-correct pipeline)
- BUT writes to a separate filename namespace (v21_bird_main) so existing pilot files don't cause skips
- Resumable per (model, strategy)
- Validates n_queries on resume (reject incompatible files)

Output: results/q1_2026_04/main/v21_bird_main_<model>_<strategy>_seed42.json

Run order: qwen25c7b first (fast), then gemma4 (slower).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_clients import call_llm  # noqa
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
RESULTS_DIR = PROJECT_ROOT / "results" / "q1_2026_04" / "main"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen25c7b", "gemma4"]
PIPELINES = {
    "s1": S1SchemaFilter,
    "s2": S2Decompose,
    "s3": S3Skeleton,
    "s4": S4DinSQL,
}
SEED = 42

EXPECTED_N = 609  # full BIRD mod+chal


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


def load_bird_full_modchal():
    full = json.load(open(BIRD_DIR / "dev.json"))
    return [q for q in full if q.get("difficulty") in ("moderate", "challenging")]


def run_one(model: str, strategy: str, queries: list) -> dict:
    out_path = RESULTS_DIR / f"v21_bird_main_{model}_{strategy}_seed{SEED}.json"
    partial_path = RESULTS_DIR / f"v21_bird_main_{model}_{strategy}_seed{SEED}_partial.json"

    if out_path.exists():
        existing = json.load(open(out_path))
        existing_n = existing.get("metadata", {}).get("n_queries", 0)
        if existing_n == len(queries):
            print(f"  [skip-complete] {out_path.name} (n={existing_n})")
            return existing
        else:
            print(
                f"  [reject-incompatible] {out_path.name} has n={existing_n}, "
                f"expected {len(queries)} -> renaming and rerunning"
            )
            out_path.rename(out_path.with_suffix(f".n{existing_n}.json"))

    if strategy == "s0":
        pipeline = S0Direct(model, seed=SEED)
    else:
        pipeline = PIPELINES[strategy](model)
        pipeline._seed = SEED

    results: list = []
    done_idx: set = set()
    if partial_path.exists():
        try:
            partial = json.load(open(partial_path))
            results = partial.get("results", [])
            done_idx = {r["query_idx"] for r in results}
            print(f"  resuming from {len(done_idx)}/{len(queries)} done")
        except Exception:
            results = []
            done_idx = set()

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

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = correct / len(results) * 100 if results else 0
            done_now = len(results) - len(done_idx)
            rate = done_now / elapsed * 60 if elapsed > 0 else 0
            print(
                f"  [{idx+1}/{len(queries)}] acc {acc:.1f}%  {rate:.1f} q/min",
                flush=True,
            )
            with open(partial_path, "w") as f:
                json.dump({"results": results}, f, ensure_ascii=False)

    elapsed = time.time() - t0
    accuracy = correct / len(results) if results else 0

    output = {
        "metadata": {
            "model": model,
            "strategy": strategy,
            "seed": SEED,
            "n_queries": len(results),
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "elapsed_min": round(elapsed / 60, 2),
            "benchmark": "BIRD mod+chal",
            "expected_n": EXPECTED_N,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if partial_path.exists():
        partial_path.unlink()

    print(
        f"  DONE {out_path.name}: acc={accuracy*100:.2f}% "
        f"({correct}/{len(results)}) in {elapsed/60:.1f}min"
    )
    return output


def main():
    print(f"=== V2.1 BIRD MAIN (LOCAL ONLY, seed={SEED}) ===")
    queries = load_bird_full_modchal()
    print(f"Loaded {len(queries)} BIRD moderate+challenging queries")

    models = MODELS
    strategies = ["s0", "s1", "s2", "s3", "s4"]

    summary = {}
    grand_t0 = time.time()
    for model in models:
        print(f"\n========== MODEL: {model} ==========")
        for strategy in strategies:
            print(f"\n>>> {model} × {strategy}  ({len(queries)} queries) <<<")
            try:
                r = run_one(model, strategy, queries)
                summary[f"{model}_{strategy}"] = r["metadata"]["accuracy"]
            except Exception as e:
                print(f"  ERROR in {model}/{strategy}: {e}")
                summary[f"{model}_{strategy}"] = None

            running_summary_path = (
                RESULTS_DIR / f"v21_bird_main_running_summary_seed{SEED}.json"
            )
            with open(running_summary_path, "w") as f:
                json.dump(
                    {
                        "completed": summary,
                        "current_model": model,
                        "current_strategy": strategy,
                        "elapsed_total_min": round((time.time() - grand_t0) / 60, 2),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    grand_elapsed = time.time() - grand_t0
    print(f"\n=== BIRD MAIN COMPLETED in {grand_elapsed/3600:.1f} hours ===")
    print("=== ACCURACY MATRIX ===")
    for model in models:
        cells = []
        for strategy in strategies:
            acc = summary.get(f"{model}_{strategy}")
            cells.append(f"{strategy}={acc*100:.1f}" if acc is not None else f"{strategy}=X")
        print(f"  {model}: " + " ".join(cells))


if __name__ == "__main__":
    main()
