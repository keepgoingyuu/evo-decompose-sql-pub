"""Arctic-Text2SQL-R1 7B rerun for the IS revision (R3 point 5).

The original uniform-protocol run capped generation at 256 tokens. The model writes its
reasoning before the query, so about a third of its outputs contained no SQL at all, and the
reported 7.88-25.34% reflected that cap rather than the prompt format. This rerun removes the
cap and separates the factors the reviewer listed:

  split       held fixed  - the same four benchmarks as the matrix
  decoding    held fixed  - greedy (temperature 0), same seed
  prompt      varied      - "uniform": the matrix's S0 prompt
                            "native" : the OmniSQL-style prompt the model was trained with,
                                       as described by its authors (asks for step-by-step
                                       reasoning and a fenced ```sql answer)

Full raw outputs are stored; SQL is extracted with the matrix's `clean_sql` (last fenced block,
else first statement) and scored under both rules.

Runs under the vLLM 0.6.6 environment used originally:
  ~/evo-decompose-sql/.venv/bin/python scripts/rerun/arctic_rerun.py --gpu-util 0.6 --window 18:00-07:30
Output: results/rerun_2026_11/arctic/<bench>__arctic__<prompt>.jsonl  (append-only, resumable)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCHMARKS, REPO, load_queries  # noqa: E402

sys.path.insert(0, str(REPO))
from src.utils.prompt_templates import S0_DIRECT  # noqa: E402
from src.utils.schema_utils import get_schema  # noqa: E402
from src.utils.sql_utils import (clean_sql, compare_results, compare_results_bird,  # noqa: E402
                                 compare_results_spider, execute_sql)

MODEL_ID = "Snowflake/Arctic-Text2SQL-R1-7B"
SEED = 42
OUT_DIR = REPO / "results/rerun_2026_11/arctic"

NATIVE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}{evidence_section}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


def build_prompt(kind: str, q: dict, schema: str) -> str:
    if kind == "native":
        ev = "\nEvidence: " + q["evidence"] if q["evidence"] else ""
        return NATIVE.format(schema=schema, question=q["question"], evidence_section=ev)
    ev = "Evidence: " + q["evidence"] if q["evidence"] else ""
    return S0_DIRECT.format(schema=schema, question=q["question"], evidence_section=ev)


def in_window(window):
    if not window:
        return True
    a, b = (dt.time.fromisoformat(x) for x in window.split("-"))
    now = dt.datetime.now().time()
    return (a <= now or now < b) if a > b else (a <= now < b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-util", type=float, required=True, help="fraction of each card vLLM may use")
    ap.add_argument("--max-model-len", type=int, default=4096)   # same context as the original run
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--window", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", default="bird:native,bird:uniform,en:native,en:uniform,"
                                      "zh-tw:native,zh-tw:uniform,ja:native,ja:uniform")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL_ID, dtype="float16", tensor_parallel_size=2, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, enforce_eager=True, seed=SEED)
    tok = llm.get_tokenizer()
    print(f"[arctic] model loaded in {time.time() - t0:.0f}s", flush=True)

    schema_cache: dict[str, str] = {}
    gold_cache: dict[tuple, tuple] = {}
    for job in args.jobs.split(","):
        bench, kind = job.split(":")
        out = OUT_DIR / f"{bench}__arctic__{kind}.jsonl"
        queries = load_queries(bench)
        if args.limit:
            queries = queries[:args.limit]
        done = {json.loads(line)["idx"] for line in open(out, encoding="utf-8") if line.strip()} if out.exists() else set()
        pending = [q for q in queries if q["idx"] not in done]
        print(f"[arctic/{bench}/{kind}] todo {len(queries)} done {len(done)} running {len(pending)}", flush=True)
        n_ok = n = 0
        t_job = time.time()
        for c in range(0, len(pending), args.chunk):
            if not in_window(args.window):
                print(f"[arctic] outside window {args.window}; stopping", flush=True)
                return
            batch = pending[c:c + args.chunk]
            prompts, params, metas = [], [], []
            for q in batch:
                if q["db_path"] not in schema_cache:
                    schema_cache[q["db_path"]] = get_schema(q["db_path"])
                text = tok.apply_chat_template([{"role": "user", "content": build_prompt(kind, q, schema_cache[q["db_path"]])}],
                                               tokenize=False, add_generation_prompt=True)
                n_in = len(tok(text).input_ids)
                room = args.max_model_len - n_in - 8
                metas.append((q, n_in, room))
                if room < 64:
                    continue  # prompt does not fit; recorded below as an error, never silently dropped
                prompts.append(text)
                params.append(SamplingParams(temperature=0.0, top_p=1.0, max_tokens=min(args.max_tokens, room), seed=SEED))
            outs = iter(llm.generate(prompts, params, use_tqdm=False)) if prompts else iter(())
            with open(out, "a", encoding="utf-8") as fh:
                for q, n_in, room in metas:
                    rec = {"bench": bench, "model": "arctic", "model_id": MODEL_ID, "prompt": kind, "idx": q["idx"],
                           "question_id": q["question_id"], "db_id": q["db_id"], "question": q["question"],
                           "gold_sql": q["gold_sql"], "seed": SEED, "in_tok": n_in,
                           "max_model_len": args.max_model_len, "ts": dt.datetime.now().isoformat(timespec="seconds")}
                    if room < 64:
                        rec.update(raw="", pred_sql="", error="prompt_too_long", finish=None, out_tok=0,
                                   correct=False, correct_official=False)
                    else:
                        o = next(outs).outputs[0]
                        raw = o.text
                        pred = clean_sql(raw)
                        key = (q["db_path"], q["gold_sql"])
                        if key not in gold_cache:
                            if len(gold_cache) > 2000:
                                gold_cache.clear()
                            gold_cache[key] = execute_sql(q["db_path"], q["gold_sql"])
                        g, g_err = gold_cache[key]
                        err, strict, official = None, False, False
                        if g_err:
                            err = "gold_error: " + g_err[:200]
                        elif not pred or "SELECT" not in pred.upper():
                            err = "no SQL generated"
                        else:
                            p, p_err = execute_sql(q["db_path"], pred)
                            if p_err:
                                err = p_err[:300]
                            else:
                                strict = compare_results(g, p)
                                official = (compare_results_bird(g, p) if BENCHMARKS[bench]["family"] == "bird"
                                            else compare_results_spider(g, p, q["gold_sql"]))
                        rec.update(raw=raw, pred_sql=pred, error=err, finish=o.finish_reason,
                                   out_tok=len(o.token_ids), correct=strict, correct_official=official)
                        n_ok += strict
                    n += 1
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            el = time.time() - t_job
            print(f"[arctic/{bench}/{kind}] {n}/{len(pending)}  strict-acc {100 * n_ok / max(1, n):.1f}%  "
                  f"{el / max(1, n):.2f}s/q  eta {el / max(1, n) * (len(pending) - n) / 60:.0f} min", flush=True)
    print("[arctic] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
