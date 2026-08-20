"""A10: Multi-sample temperature variance for S0 and S2 (JIIS R2-4 / R2-6).

Reviewer 2 asked for multi-temperature sampling to estimate the model's
intrinsic variance, since all headline numbers use deterministic decoding
(temperature 0, seed 42). Design:

  For S0 and S2 (original V1 template), draw K=3 independent samples at
  temperature 0.7 (seeds 1, 2, 3) on the first SUBSET_K queries of the
  A9 subset (same seed-42 sample of Spider EN). Report per-run accuracy,
  mean +- sd, and the S2-S0 drop at temperature 0.7 versus the
  deterministic drop -- testing whether the harm is a t=0 artifact.

Protocol otherwise identical to A9 / phase4 (same schema serialisation,
execution comparison, record schema).

Usage:
    venv/bin/python scripts/a10_temperature_variance.py gpt41 s0 --sample 1
    venv/bin/python scripts/a10_temperature_variance.py gpt41 s2 --sample 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "a9", PROJECT_ROOT / "scripts/a9_s2_template_ablation.py")
a9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a9)

from src.pipelines.base import BasePipeline                       # noqa: E402
from src.utils.prompt_templates import S0_DIRECT                  # noqa: E402
from src.utils.schema_utils import get_schema                     # noqa: E402
from src.utils.sql_utils import clean_sql, execute_sql, compare_results  # noqa: E402

TEMPERATURE = 0.7
SUBSET_K = 200
OUT_DIR = PROJECT_ROOT / "results/ablation_2026_08"


class S0Direct(BasePipeline):
    name = "s0"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        raw = self._call_llm(S0_DIRECT.format(
            schema=schema, question=question,
            evidence_section=evidence_section), "direct")
        return self._make_result(clean_sql(raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gpt41", "qwen", "deepseek", "gemini"])
    ap.add_argument("strategy", choices=["s0", "s2"])
    ap.add_argument("--sample", type=int, required=True, choices=[1, 2, 3],
                    help="independent sample id; also used as the seed")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subset", type=int, default=SUBSET_K)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.load(open(a9.SPIDER_EN_DATA))
    idxs = a9.subset_indices(len(data))[: args.subset]
    run_name = f"a10_{args.model}_{args.strategy}_t07_sample{args.sample}_en"
    out_file = OUT_DIR / f"{run_name}.json"
    partial_file = OUT_DIR / f"{run_name}_partial.json"

    results, done = [], set()
    if partial_file.exists():
        results = json.load(open(partial_file)).get("results", [])
        done = {r["query_idx"] for r in results}
        print(f"resuming: {len(done)} done")

    work = []
    for idx in idxs:
        if idx in done:
            continue
        q = data[idx]
        db_path = str(a9.SPIDER_DB_DIR / q["db_id"] / (q["db_id"] + ".sqlite"))
        work.append((idx, q, db_path))
    print(f"{run_name}: {len(work)} queries at t={TEMPERATURE}, seed={args.sample}")

    t_start = time.time()

    def process(item):
        idx, q, db_path = item
        if args.strategy == "s0":
            pipeline = S0Direct(args.model)
        else:
            pipeline = a9.S2Variant(args.model, a9.VARIANTS["v1"], "v1")
        pipeline._seed = args.sample
        pipeline._temperature = TEMPERATURE
        question, gold_sql, db_id = q["question"], q["query"], q["db_id"]
        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            return {"query_idx": idx, "db_id": db_id, "question": question[:120],
                    "gold_sql": gold_sql[:200], "pred_sql": "", "correct": False,
                    "error": "gold_error: %s" % gold_err[:100], "call_count": 0}
        schema = get_schema(db_path)
        t0 = time.time()
        try:
            r = pipeline.run(question, schema, evidence="", db_path=db_path)
        except Exception as e:
            r = {"pred_sql": "", "call_count": pipeline.call_count,
                 "steps": [{"step": "error", "raw": str(e)[:200]}]}
        gen_time = time.time() - t0
        pred_sql = r.get("pred_sql", "")
        has_sql = bool(pred_sql) and "SELECT" in pred_sql.upper()
        if has_sql:
            pred_result, pred_err = execute_sql(db_path, pred_sql)
        else:
            pred_result, pred_err = None, "no SQL generated"
        return {"query_idx": idx, "db_id": db_id, "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": pred_sql[:200],
                "correct": compare_results(gold_result, pred_result),
                "gen_time": round(gen_time, 2),
                "call_count": r.get("call_count", 0),
                "error": (pred_err[:100] if pred_err else None) if has_sql
                         else "no SQL generated"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, w): w for w in work}
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 10 == 0:
                acc = sum(1 for r in results if r["correct"]) / len(results)
                json.dump({"results": results}, open(partial_file, "w"), indent=1)
                print(f"  {n}/{len(work)}  acc={acc*100:.2f}%", flush=True)

    acc = sum(1 for r in results if r["correct"]) / len(results) * 100
    meta = {"run": run_name, "model": args.model, "strategy": args.strategy,
            "language": "en", "dataset": "spider_dev",
            "subset_seed": a9.SUBSET_SEED, "subset_n": len(idxs),
            "temperature": TEMPERATURE, "seed": args.sample,
            "accuracy": acc / 100, "n": len(results),
            "elapsed_sec": round(time.time() - t_start, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump({"metadata": meta,
               "results": sorted(results, key=lambda r: r["query_idx"])},
              open(out_file, "w"), indent=1)
    if partial_file.exists():
        partial_file.unlink()
    print(f"DONE {run_name}: acc={acc:.2f}% n={len(results)} -> {out_file}")


if __name__ == "__main__":
    main()
