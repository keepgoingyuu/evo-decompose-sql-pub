"""A9: S2 prompt-template ablation (JIIS reviewers R1-2 / R2-1).

Both JIIS reviewers argued the "decomposition harms strong models" finding
could be an artifact of the single S2 template. This script evaluates two
additional S2 template variants -- same decompose -> sub-SQL -> merge
structure, different wording/format -- on a fixed random subset of Spider EN,
so the observed harm can be tested for stability across phrasings.

  V1  original templates (results reused from phase4_<model>_s2_en.json)
  V2  "planner" rewording: analyst persona, minimal-plan framing,
      stricter output-format instructions in all three stages
  V3  "few-shot" variant: decompose stage carries one worked toy example
      (toy schema, not from Spider), sub/merge stages lightly reworded

Protocol matches phase4_cspider_tw.py exactly: deterministic decoding
(temperature 0, seed 42), same schema serialisation (get_schema), same
execution comparison (execute_sql + compare_results), same record schema.

Subset: SUBSET_N indices sampled without replacement from Spider dev
(seed 42); indices are stored in the output for exact reuse by A10.

Usage:
    venv/bin/python scripts/a9_s2_template_ablation.py gpt41 v2
    venv/bin/python scripts/a9_s2_template_ablation.py gpt41 v3 --workers 6
    venv/bin/python scripts/a9_s2_template_ablation.py qwen v1 --subset 200
      (qwen = local Ollama rerun incl. v1, since canonical qwen ran on vLLM)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.base import BasePipeline               # noqa: E402
from src.utils.prompt_templates import (                   # noqa: E402
    S0_DIRECT, S2_DECOMPOSE, S2_SUB_SQL, S2_MERGE)
from src.utils.schema_utils import get_schema              # noqa: E402
from src.utils.sql_utils import clean_sql, execute_sql, compare_results  # noqa: E402

SPIDER_EN_DATA = PROJECT_ROOT / "data/spider/spider_data/dev.json"
_DB_CANDIDATES = [PROJECT_ROOT / "data/spider/spider_data/database",
                  PROJECT_ROOT / "data/spider/spider_data/spider_data/database"]
SPIDER_DB_DIR = next(p for p in _DB_CANDIDATES if p.exists())
OUT_DIR = PROJECT_ROOT / "results/ablation_2026_08"

SUBSET_SEED = 42
SUBSET_N = 400

# ============================================================
# Template variants (structure identical to V1: decompose/sub/merge)
# ============================================================

V2_DECOMPOSE = """You are a senior data analyst planning a SQL solution.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Plan the solution as an ordered list of the SMALLEST number of sub-questions
(between 2 and 4) such that each one can be answered by a single SELECT
statement over the schema above. Do not introduce sub-questions whose answer
is not needed for the final result.

Respond with the numbered list only, one sub-question per line:
1. ...
2. ...

If the question needs no decomposition, respond with a single line:
1. [the original question restated]"""

V2_SUB_SQL = """You are a senior data analyst. Answer the sub-question below
with exactly one SQLite SELECT statement.

Database Schema:
{schema}

Sub-question: {sub_question}
(It is part of solving: {question})
{evidence_section}

Constraints: SQLite dialect; only tables/columns present in the schema;
no comments, no explanation, no markdown fences -- output the SQL only."""

V2_MERGE = """You are a senior data analyst. Assemble the final query.

Database Schema:
{schema}

Original question: {question}
{evidence_section}

Solved sub-questions:
{sub_queries_section}

Write ONE final SQLite query that answers the original question, reusing the
logic of the sub-queries (CTEs, JOINs, or nested queries are all acceptable).
Output the SQL only -- no comments, no explanation, no markdown fences."""

V3_EXAMPLE = """Example (toy schema: employee(id, name, dept_id, salary); department(id, name)):
Question: Which department has the highest average salary, and how many employees does it have?
1. What is the average salary of each department?
2. Which department has the highest average salary?
3. How many employees work in that department?
"""

V3_DECOMPOSE = """Break the database question below into 2-4 self-contained
sub-questions, each answerable with one SQL query, following the style of
this example.

""" + V3_EXAMPLE + """
Database Schema:
{schema}

Question: {question}
{evidence_section}

Output only the numbered sub-question list for the question above.
If it is already simple, output one sub-question identical to the original."""

V3_SUB_SQL = """Generate one SQLite query for the sub-question, as a step
towards the original question.

Database Schema:
{schema}

Sub-question: {sub_question}
Original question (context): {question}
{evidence_section}

Use only schema tables and columns. Output only the raw SQL."""

V3_MERGE = """Combine the partial results into the final answer.

Database Schema:
{schema}

Original question: {question}
{evidence_section}

Sub-questions already solved:
{sub_queries_section}

Produce one final SQLite query answering the original question (CTEs,
subqueries, or JOINs as needed). Output only the raw SQL."""

VARIANTS = {
    "v1": {"dec": S2_DECOMPOSE, "sub": S2_SUB_SQL, "mer": S2_MERGE},
    "v2": {"dec": V2_DECOMPOSE, "sub": V2_SUB_SQL, "mer": V2_MERGE},
    "v3": {"dec": V3_DECOMPOSE, "sub": V3_SUB_SQL, "mer": V3_MERGE},
}


class S2Variant(BasePipeline):
    """S2 decompose->sub-SQL->merge with injectable templates.

    Mirrors src/pipelines/s2_decompose.py exactly, including the <=1
    sub-question fallback to S0 direct generation.
    """

    def __init__(self, model_name: str, templates: dict, variant: str):
        super().__init__(model_name)
        self.templates = templates
        self.name = f"s2_{variant}"

    def run(self, question, schema, evidence="", db_path=""):
        import re
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        dec_raw = self._call_llm(self.templates["dec"].format(
            schema=schema, question=question, evidence_section=evidence_section),
            "decompose")
        subs = []
        for line in dec_raw.strip().split("\n"):
            m = re.match(r'^\s*(?:\d+[\.\)]\s*|-\s*)(.*)', line)
            if m and m.group(1).strip():
                subs.append(m.group(1).strip())
        if len(subs) <= 1:
            self.steps.append({"step": "decompose_fallback",
                               "raw": "simple question, using S0 direct"})
            gen = self._call_llm(S0_DIRECT.format(
                schema=schema, question=question,
                evidence_section=evidence_section), "direct_generate")
            return self._make_result(clean_sql(gen))
        sub_sqls = []
        for i, sq in enumerate(subs):
            raw = self._call_llm(self.templates["sub"].format(
                schema=schema, sub_question=sq, question=question,
                evidence_section=evidence_section), "sub_sql_%d" % (i + 1))
            sub_sqls.append({"sub_question": sq, "sql": clean_sql(raw)})
        section = "\n".join("Sub-question %d: %s\nSQL: %s\n" %
                            (i + 1, s["sub_question"], s["sql"])
                            for i, s in enumerate(sub_sqls))
        mer = self._call_llm(self.templates["mer"].format(
            schema=schema, question=question, evidence_section=evidence_section,
            sub_queries_section=section), "merge")
        return self._make_result(clean_sql(mer))


def subset_indices(n_total: int) -> list[int]:
    rng = random.Random(SUBSET_SEED)
    return sorted(rng.sample(range(n_total), SUBSET_N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gpt41", "qwen", "deepseek", "gemini"])
    ap.add_argument("variant", choices=["v1", "v2", "v3"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subset", type=int, default=SUBSET_N,
                    help="use only the first K of the sampled subset")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.load(open(SPIDER_EN_DATA))
    idxs = subset_indices(len(data))[: args.subset]
    run_name = f"a9_{args.model}_s2_{args.variant}_en"
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
        db_path = str(SPIDER_DB_DIR / q["db_id"] / (q["db_id"] + ".sqlite"))
        work.append((idx, q, db_path))
    print(f"{run_name}: {len(work)} queries to run "
          f"(subset {len(idxs)}, seed {SUBSET_SEED})")

    lock_every = 10
    t_start = time.time()

    def process(item):
        idx, q, db_path = item
        pipeline = S2Variant(args.model, VARIANTS[args.variant], args.variant)
        pipeline._seed = 42
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
        correct = compare_results(gold_result, pred_result)
        return {"query_idx": idx, "db_id": db_id, "question": question[:120],
                "gold_sql": gold_sql[:200], "pred_sql": pred_sql[:200],
                "correct": correct, "gen_time": round(gen_time, 2),
                "call_count": r.get("call_count", 0),
                "error": (pred_err[:100] if pred_err else None) if has_sql
                         else "no SQL generated"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, w): w for w in work}
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % lock_every == 0:
                acc = sum(1 for r in results if r["correct"]) / len(results)
                json.dump({"results": results}, open(partial_file, "w"), indent=1)
                el = time.time() - t_start
                print(f"  {n}/{len(work)}  acc={acc*100:.2f}%  "
                      f"({el/n:.1f}s/q)", flush=True)

    acc = sum(1 for r in results if r["correct"]) / len(results) * 100
    meta = {"run": run_name, "model": args.model, "variant": args.variant,
            "language": "en", "dataset": "spider_dev",
            "subset_seed": SUBSET_SEED, "subset_n": len(idxs),
            "subset_indices": idxs,
            "temperature": 0, "seed": 42,
            "accuracy": acc / 100, "n": len(results),
            "elapsed_sec": round(time.time() - t_start, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump({"metadata": meta, "results": sorted(results, key=lambda r: r["query_idx"])},
              open(out_file, "w"), indent=1)
    if partial_file.exists():
        partial_file.unlink()
    print(f"DONE {run_name}: acc={acc:.2f}% n={len(results)} -> {out_file}")


if __name__ == "__main__":
    main()
