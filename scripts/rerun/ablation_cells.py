"""Prompt-template ablation for S1 and S2 under the fixed harness (IS revision: R1-W3, R3-2).

The matrix fixes one template per strategy. This script re-runs S1 (schema filtering) and
S2 (question decomposition) under two reworded template sets each, on the fixed 400-query
subset used by the original ablation (Spider dev indices, seed 42), in any of the three
languages (the Spider-family benchmarks share query positions).

  v1  canonical templates  -> not run here; taken from the corrected matrix
  v2  "analyst" rewording of every stage, stricter output-format instructions
  v3  few-shot variant: the first stage carries one worked toy example

S2 variants are the ones defined for the original ablation (scripts/a9_s2_template_ablation.py);
the S1 variants are defined below with the same two manipulations. Pipeline structure, parsing,
extraction and scoring are identical to the matrix (same classes, same `clean_sql`).

Output: results/rerun_2026_11/ablation/<bench>__<model>__<strategy>-<variant>.jsonl

Usage: python3 scripts/rerun/ablation_cells.py qwen25c7b en s1 v2 [--window 18:00-07:30]

When an ablation is run on a different machine from the matrix (Gemma-3n-E4B, 2026-11-26, on
an Apple-silicon host because the DGX was unavailable), the two arms taken from the matrix
must be re-run on that machine as well, so that every arm of the paired comparison shares one
serving environment: `s0 v1` runs direct generation and `s1 v1` / `s2 v1` the canonical
pipeline, with the matrix's own classes, on the same 400-query subset. Use --out-dir to keep
such runs apart from the DGX/API ablation records.

  python3 scripts/rerun/ablation_cells.py gemma4 en s0 v1 --out-dir results/rerun_2026_11/ablation_mac
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_cells as rc  # noqa: E402  (score, in_window, harness_id)
from common import REPO, load_queries  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from a9_s2_template_ablation import VARIANTS as S2_VARIANTS, S2Variant, subset_indices  # noqa: E402
from src.pipelines.s1_schema_filter import S1SchemaFilter  # noqa: E402
from src.utils.api_clients import MODEL_CONFIG  # noqa: E402
from src.utils.schema_utils import filter_schema_by_tables, get_schema  # noqa: E402
from src.utils.sql_utils import clean_sql, clean_sql_legacy, execute_sql  # noqa: E402

OUT_DIR = REPO / "results/rerun_2026_11/ablation"

# ---- S1 template variants (the JSON answer format is kept, so the canonical parser applies) ----
S1_V2_LINK = """You are a senior data analyst preparing to write a SQL query.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Select the MINIMAL set of tables and columns that the query will need. Include a table
only if the query must read from it or join through it.

Respond with a single JSON object and nothing else, in exactly this form:
{{"tables": ["table1", "table2"], "columns": ["table1.col1", "table2.col2"]}}"""

S1_V2_GENERATE = """You are a senior data analyst. Answer the question with exactly one
SQLite SELECT statement.

Schema (already reduced to the relevant tables):
{schema}

Question: {question}
{evidence_section}

Constraints: SQLite dialect; only tables/columns present in the schema above;
no comments, no explanation, no markdown fences -- output the SQL only."""

S1_V3_EXAMPLE = """Example (toy schema: employee(id, name, dept_id, salary); department(id, name); office(id, city)):
Question: Which department has the highest average salary?
{{"tables": ["employee", "department"], "columns": ["employee.dept_id", "employee.salary", "department.id", "department.name"]}}
"""

S1_V3_LINK = """Identify the tables and columns needed to answer the database question
below, following the style of this example.

""" + S1_V3_EXAMPLE + """
Database Schema:
{schema}

Question: {question}
{evidence_section}

Output only the JSON object for the question above."""

S1_V3_GENERATE = """Write one SQLite query that answers the question.

Relevant Schema (filtered):
{schema}

Question: {question}
{evidence_section}

Use only schema tables and columns. Output only the raw SQL."""

S1_VARIANTS = {"v2": {"link": S1_V2_LINK, "gen": S1_V2_GENERATE},
               "v3": {"link": S1_V3_LINK, "gen": S1_V3_GENERATE}}


class S1Variant(S1SchemaFilter):
    """S1 with injectable templates; mirrors S1SchemaFilter.run step for step."""

    def __init__(self, model_name: str, templates: dict, variant: str):
        super().__init__(model_name)
        self.templates = templates
        self.name = f"s1_{variant}"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""
        link_raw = self._call_llm(self.templates["link"].format(
            schema=schema, question=question, evidence_section=evidence_section), "schema_link")
        tables = self._parse_tables(link_raw)
        if tables:
            filtered = filter_schema_by_tables(schema, tables)
        else:
            filtered = schema
            self.steps.append({"step": "schema_link_fallback", "raw": "parse failed, using full schema"})
        gen_raw = self._call_llm(self.templates["gen"].format(
            schema=filtered, question=question, evidence_section=evidence_section), "generate_sql")
        return self._make_result(clean_sql(gen_raw))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODEL_CONFIG))
    ap.add_argument("bench", choices=["en", "zh-tw", "ja"])
    ap.add_argument("strategy", choices=["s0", "s1", "s2"])
    ap.add_argument("variant", choices=["v1", "v2", "v3"])
    ap.add_argument("--window", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    if args.strategy == "s0" and args.variant != "v1":
        ap.error("s0 has the canonical template only (v1)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bench}__{args.model}__{args.strategy}-{args.variant}.jsonl"
    tag = f"[abl {args.model}/{args.bench}/{args.strategy}-{args.variant}]"
    queries = load_queries(args.bench)
    todo = subset_indices(len(queries))
    if args.limit:
        todo = todo[:args.limit]
    done = {json.loads(line)["idx"] for line in open(out, encoding="utf-8") if line.strip()} if out.exists() else set()
    pending = [i for i in todo if i not in done]
    print(f"{tag} todo {len(todo)}  done {len(done & set(todo))}  running {len(pending)}", flush=True)
    if not pending:
        return

    if args.variant == "v1":   # canonical template: the matrix's own pipeline classes
        pipeline = rc.S0Direct(args.model) if args.strategy == "s0" else rc.PIPELINES[args.strategy](args.model)
    elif args.strategy == "s1":
        pipeline = S1Variant(args.model, S1_VARIANTS[args.variant], args.variant)
    else:
        pipeline = S2Variant(args.model, S2_VARIANTS[args.variant], args.variant)
    pipeline._seed = rc.SEED
    harness = rc.harness_id()
    schema_cache: dict[str, str] = {}
    t0_all, n_ok, n_fail = time.time(), 0, 0

    with open(out, "a", encoding="utf-8") as fh:
        for n, idx in enumerate(pending, 1):
            if not rc.in_window(args.window):
                print(f"{tag} outside window {args.window}; stopping after {n - 1} queries", flush=True)
                break
            q = queries[idx]
            rec = {"bench": args.bench, "model": args.model, "model_id": MODEL_CONFIG[args.model]["model"],
                   "strategy": args.strategy, "variant": args.variant, "idx": idx, "db_id": q["db_id"],
                   "question": q["question"], "gold_sql": q["gold_sql"], "seed": rc.SEED, "harness": harness,
                   "ts": dt.datetime.now().isoformat(timespec="seconds")}
            gold_result, gold_err = execute_sql(q["db_path"], q["gold_sql"])
            if gold_err:
                rec.update(error="gold_error: " + gold_err[:200], pred_sql="", steps=[], call_count=0,
                           correct=False, correct_official=False)
            else:
                if q["db_path"] not in schema_cache:
                    schema_cache[q["db_path"]] = get_schema(q["db_path"])
                t0 = time.time()
                try:
                    res = pipeline.run(q["question"], schema_cache[q["db_path"]], evidence=q["evidence"],
                                       db_path=q["db_path"])
                except Exception as e:  # noqa: BLE001
                    res = {"pred_sql": "", "call_count": pipeline.call_count,
                           "steps": list(pipeline.steps) + [{"step": "pipeline_error", "raw": repr(e)[:500]}]}
                steps = res.get("steps", [])
                failed = [s for s in steps if str(s.get("raw", "")).startswith("ERROR:")
                          or s.get("step") == "pipeline_error"]
                if failed:
                    n_fail += 1
                    print(f"{tag} idx {idx} call failed, left pending: {str(failed[0].get('raw'))[:160]}", flush=True)
                    if n_fail >= rc.MAX_CONSECUTIVE_FAILURES:
                        print(f"{tag} {n_fail} consecutive failures; aborting", flush=True)
                        sys.exit(2)
                    continue
                n_fail = 0
                pred = res.get("pred_sql", "") or ""
                err, strict, official = rc.score(args.bench, q, pred, gold_result)
                final_raw = next((s.get("raw", "") for s in reversed(steps) if "prompt_sha1" in s), "")
                legacy = clean_sql_legacy(final_raw) if final_raw else ""
                l_err, l_strict, _ = (err, strict, official) if legacy == pred else rc.score(args.bench, q, legacy,
                                                                                            gold_result)
                rec.update(gen_time=round(time.time() - t0, 2), pred_sql=pred, error=err, correct=strict,
                           correct_official=official, legacy_pred_sql=None if legacy == pred else legacy,
                           legacy_error=l_err, legacy_correct=l_strict, call_count=res.get("call_count", 0),
                           steps=steps)
                n_ok += strict
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 50 == 0 or n == len(pending):
                el = time.time() - t0_all
                print(f"{tag} {n}/{len(pending)}  strict-acc {100 * n_ok / n:.1f}%  {el / n:.1f}s/q", flush=True)


if __name__ == "__main__":
    main()
