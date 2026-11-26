"""Shared definitions for the 2026-11 IS-revision rerun: benchmarks, models, legacy result files.

Every benchmark is addressed by the position of the query in its source file (`idx`),
which is what the original runs stored as `query_idx` (verified 2026-11-24). The three
API models on BIRD stored the BIRD `question_id` instead; `legacy_records` maps it back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BENCHMARKS = {
    "bird": {"queries": "data/bird/dev_20240627/dev.json", "db_dir": "data/bird/dev_20240627/dev_databases",
             "sql_key": "SQL", "family": "bird"},
    "en": {"queries": "data/spider/spider_data/dev.json", "db_dir": "data/spider/spider_data/spider_data/database",
           "sql_key": "query", "family": "spider"},
    "zh-tw": {"queries": "data/cspider/dev_tw.json", "db_dir": "data/spider/spider_data/spider_data/database",
              "sql_key": "query", "family": "spider"},
    "ja": {"queries": "data/multispider/dataset/multispider/with_english_value/dev_ja.json",
           "db_dir": "data/spider/spider_data/spider_data/database", "sql_key": "query", "family": "spider"},
}
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]
API_MODELS = ["gpt41", "deepseek", "gemini"]
LOCAL_MODELS = ["qwen25c7b", "gemma4"]
PANEL = API_MODELS + LOCAL_MODELS

# Stored error text left by the WITH-stripping extraction defect.
STRIP_SIGNATURE = 'near ")": syntax error'
# Errors that may come from an exhausted output budget rather than from the model's SQL.
BUDGET_ERRORS = ("incomplete input", "no SQL")


def load_queries(bench: str) -> list[dict]:
    """Queries of one benchmark in source-file order, normalised to
    {idx, db_id, question, gold_sql, evidence, question_id, db_path}."""
    cfg = BENCHMARKS[bench]
    rows = json.load(open(REPO / cfg["queries"], encoding="utf-8"))
    if bench == "bird":
        rows = [r for r in rows if r.get("difficulty") in ("moderate", "challenging")]
    out = []
    for i, r in enumerate(rows):
        out.append({
            "idx": i, "db_id": r["db_id"], "question": r["question"], "gold_sql": r[cfg["sql_key"]],
            "evidence": r.get("evidence", "") or "", "question_id": r.get("question_id"),
            "difficulty": r.get("difficulty"),
            "db_path": str(REPO / cfg["db_dir"] / r["db_id"] / (r["db_id"] + ".sqlite")),
        })
    return out


def legacy_file(bench: str, model: str, strategy: str) -> Path:
    """Result file of the original (pre-2026-11) run for one matrix cell."""
    p4, v21 = REPO / "results/thesis_2026_02", REPO / "results/q1_2026_04/main"
    if bench == "bird":
        if model in API_MODELS:
            if strategy == "s0":
                return p4 / {"gpt41": "phase1_gpt41_baseline.json", "deepseek": "phase1_deepseek_baseline.json",
                             "gemini": "phase1_gemini3flash_baseline.json"}[model]
            return p4 / f"phase2_{model}_{strategy}.json"
        return v21 / f"v21_bird_main_{model}_{strategy}_seed42.json"
    if model in API_MODELS:
        if bench == "ja":
            return v21 / f"v21_{model}_japanese_{strategy}_seed42.json"
        return p4 / f"phase4_{model}_{strategy}_{bench}.json"
    name = {"en": "xlingual_spider_en", "zh-tw": "xlingual_cspider_zh_tw", "ja": "xlingual_multispider_ja"}[bench]
    return v21 / f"v21_{name}_{model}_{strategy}_seed42.json"


def legacy_records(bench: str, model: str, strategy: str) -> dict[int, dict]:
    """{idx: stored record} for one cell of the original runs."""
    rows = json.load(open(legacy_file(bench, model, strategy), encoding="utf-8"))["results"]
    if bench == "bird" and model in API_MODELS:
        pos = {q["question_id"]: q["idx"] for q in load_queries("bird")}
        return {pos[int(r["query_id"])]: r for r in rows if int(r["query_id"]) in pos}
    return {int(r["query_idx"]): r for r in rows}


# Queries whose reference (gold) query the harness could not evaluate when the records were
# generated, so that the model was not called for them in part or all of the matrix:
#   BIRD 205, 245        the gold query runs for about 30 s or more, at the execution limit;
#                        whether it finished depended on the machine
#   Spider-family 455, 456   the gold result contains text that is not valid UTF-8 (database wta_1)
# They are scored as failures for every model and strategy, under both rules, so that all cells
# are treated alike. Indices are positions in the benchmark file (BIRD: in the mod+chal subset).
UNSCORABLE = {"bird": {205, 245}, "en": {455, 456}, "zh-tw": {455, 456}, "ja": {455, 456}}
UNSCORABLE_ERROR = "reference query not evaluable; scored as failure for all systems"


def score_sql(bench: str, q: dict, sql: str, gold_result):
    """(error, strict, official) for one predicted SQL string against an executed gold result.
    Spider-family databases are read with lenient text decoding, as the official evaluator does."""
    from src.utils.sql_utils import compare_results, compare_results_bird, compare_results_spider, execute_sql
    if not sql or "SELECT" not in sql.upper():
        return "no SQL generated", False, False
    spider = BENCHMARKS[bench]["family"] == "spider"
    pred_result, err = execute_sql(q["db_path"], sql, lenient_text=spider)
    if err:
        return err[:300], False, False
    if q["idx"] in UNSCORABLE[bench]:
        return None, False, False
    strict = compare_results(gold_result, pred_result)
    official = (compare_results_spider(gold_result, pred_result, q["gold_sql"]) if spider
                else compare_results_bird(gold_result, pred_result))
    return None, bool(strict), bool(official)


def gold_result(bench: str, q: dict):
    """Executed gold result, or None for the queries listed in UNSCORABLE."""
    from src.utils.sql_utils import execute_sql
    if q["idx"] in UNSCORABLE[bench]:
        return None
    res, err = execute_sql(q["db_path"], q["gold_sql"], lenient_text=BENCHMARKS[bench]["family"] == "spider")
    if err:
        raise RuntimeError(f"gold query failed: {bench} idx {q['idx']}: {err}")
    return res
