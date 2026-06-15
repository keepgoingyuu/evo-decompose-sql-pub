"""Phase 1: OmniSQL-7B baseline on full BIRD dev set (S0 direct generation).

Uses enriched DDL schema with column descriptions and sample values,
matching OmniSQL's training format (see OmniSQL paper Section 3.2).
"""
import json, sqlite3, time, requests, csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
DB_DIR = BIRD_DIR / "dev_databases"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "omnisql-7b"
TIMEOUT = 120
RESULT_FILE = "phase1_omnisql_baseline.json"
PARTIAL_FILE = "phase1_omnisql_baseline_partial.json"

def call_llm(prompt):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 2048}
        }, timeout=TIMEOUT)
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        return "ERROR: " + str(e)

def clean_sql(raw):
    text = raw.strip()
    # 1) Try to extract from ```sql ... ``` code blocks first
    import re
    code_blocks = re.findall(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        # Use the last code block (usually the final SQL)
        sql = code_blocks[-1].strip()
    else:
        # Fallback: strip tags and find SELECT
        for tag in ["<s>", "</s>", "[SQL]", "[/SQL]"]:
            text = text.replace(tag, "")
        text = text.strip()
        if "SELECT" in text.upper():
            text = text[text.upper().find("SELECT"):]
        sql = text.strip()
    # Remove comments at the start
    lines = sql.split("\n")
    sql_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--") and not sql_lines:
            continue  # skip leading comments
        if stripped.startswith("**") or stripped.startswith("Note:") or stripped.startswith("This query") or stripped.startswith("Explanation"):
            break
        sql_lines.append(line)
    sql = "\n".join(sql_lines).strip()
    sql = sql.rstrip(";").strip()
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    return sql

def load_column_descriptions(db_id):
    """Load BIRD column descriptions from database_description CSVs."""
    desc_dir = DB_DIR / db_id / "database_description"
    col_descs = {}  # (table_name, col_name) -> description string
    if not desc_dir.exists():
        return col_descs
    for csv_path in desc_dir.glob("*.csv"):
        table_name = csv_path.stem
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig = row.get("original_column_name", "").strip()
                    desc = row.get("column_description", "").strip()
                    val_desc = row.get("value_description", "").strip()
                    if not orig:
                        continue
                    parts = []
                    if desc and desc.lower() != orig.lower():
                        parts.append(desc)
                    if val_desc and val_desc.lower() not in ("", "not useful"):
                        # Truncate long value descriptions
                        parts.append(val_desc[:120])
                    if parts:
                        col_descs[(table_name.lower(), orig.lower())] = "; ".join(parts)
        except Exception:
            continue
    return col_descs

def get_sample_values(db_path, table_name, col_name, limit=3):
    """Get a few distinct sample values for a column."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 3000")
        cur = conn.cursor()
        # Quote column name to handle spaces/special chars
        cur.execute(
            "SELECT DISTINCT `%s` FROM `%s` WHERE `%s` IS NOT NULL LIMIT %d"
            % (col_name, table_name, col_name, limit)
        )
        vals = [str(row[0])[:50] for row in cur.fetchall()]
        conn.close()
        return vals
    except Exception:
        return []

def get_schema(db_path, db_id=None):
    """Build enriched DDL schema with column descriptions and sample values."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Load column descriptions if db_id provided
    col_descs = load_column_descriptions(db_id) if db_id else {}

    # Get table list
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    enriched_schemas = []
    for table in tables:
        # Get column info
        cur.execute("PRAGMA table_info(`%s`)" % table)
        columns = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        if not columns:
            continue

        # Get foreign keys
        cur.execute("PRAGMA foreign_key_list(`%s`)" % table)
        fks = cur.fetchall()  # (id, seq, table, from, to, ...)
        fk_map = {}
        for fk in fks:
            fk_map[fk[3]] = (fk[2], fk[4])  # from_col -> (ref_table, ref_col)

        lines = ["CREATE TABLE `%s` (" % table]
        col_lines = []
        pk_cols = [c[1] for c in columns if c[5]]

        for col in columns:
            cid, name, ctype, notnull, dflt, pk = col
            parts = ["    `%s`" % name]
            parts.append(ctype if ctype else "TEXT")
            if pk and len(pk_cols) == 1:
                parts.append("PRIMARY KEY")
            if notnull:
                parts.append("NOT NULL")

            # Build comment
            comment_parts = []
            # Column description from BIRD metadata
            desc_key = (table.lower(), name.lower())
            if desc_key in col_descs:
                comment_parts.append(col_descs[desc_key])
            # Foreign key info
            if name in fk_map:
                ref_t, ref_c = fk_map[name]
                comment_parts.append("references `%s`.`%s`" % (ref_t, ref_c))
            # Sample values
            samples = get_sample_values(db_path, table, name)
            if samples:
                comment_parts.append("examples: [%s]" % ", ".join(samples))

            col_line = " ".join(parts)
            if comment_parts:
                col_line += "  -- " + "; ".join(comment_parts)
            col_lines.append(col_line)

        lines.append(",\n".join(col_lines))

        # Composite primary key
        if len(pk_cols) > 1:
            lines.append(",\n    PRIMARY KEY (%s)" % ", ".join("`%s`" % c for c in pk_cols))

        lines.append(")")
        enriched_schemas.append("\n".join(lines))

    conn.close()
    return "\n\n".join(enriched_schemas)

def execute_sql(db_path, sql):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall(), None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

# OmniSQL official prompt template
PROMPT_TMPL = """Task Overview:
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

def run_baseline():
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)

    print("=" * 70)
    print("PHASE 1 BASELINE: OmniSQL-7B on BIRD dev set")
    print("Total queries: %d" % len(bird))
    print("Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    diff_counts = defaultdict(int)
    for q in bird:
        diff_counts[q["difficulty"]] += 1
    for d, c in sorted(diff_counts.items()):
        print("  %s: %d queries" % (d, c))

    # Resume from partial results
    results = []
    done_ids = set()
    partial_path = PROJECT_ROOT / "results" / PARTIAL_FILE
    if partial_path.exists():
        with open(partial_path) as pf:
            partial = json.load(pf)
            results = partial.get("results", [])
            done_ids = {r["query_id"] for r in results}
            print("  Resuming from %d completed queries" % len(done_ids))

    stats = defaultdict(lambda: {"total": 0, "correct": 0, "exec_err": 0, "no_sql": 0})
    for r in results:
        d = r["difficulty"]
        stats[d]["total"] += 1
        if r.get("correct"):
            stats[d]["correct"] += 1
        if r.get("error") and r["error"] != "no SQL generated" and "gold_error" not in r.get("error", ""):
            stats[d]["exec_err"] += 1
        if r.get("error") == "no SQL generated":
            stats[d]["no_sql"] += 1

    start_time = time.time()

    for i, q in enumerate(bird):
        if i in done_ids:
            continue

        db_id = q["db_id"]
        diff = q["difficulty"]
        question = q["question"]
        gold_sql = q["SQL"]
        evidence = q.get("evidence", "")
        db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))

        if not Path(db_path).exists():
            continue

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            stats[diff]["total"] += 1
            results.append({
                "query_id": i, "db_id": db_id, "difficulty": diff,
                "correct": False, "error": "gold_error: " + gold_err[:100]
            })
            continue

        schema = get_schema(db_path, db_id=db_id)
        evidence_section = "\nEvidence: " + evidence if evidence else ""
        prompt = PROMPT_TMPL.format(question=question, schema=schema, evidence_section=evidence_section)

        t0 = time.time()
        raw = call_llm(prompt)
        gen_time = time.time() - t0
        pred_sql = clean_sql(raw)

        has_sql = pred_sql and "SELECT" in pred_sql.upper()
        if has_sql:
            pred_result, pred_err = execute_sql(db_path, pred_sql)
        else:
            pred_result, pred_err = None, "no SQL generated"

        correct = str(gold_result) == str(pred_result) if gold_result and pred_result else False

        stats[diff]["total"] += 1
        if correct:
            stats[diff]["correct"] += 1
        if pred_err and has_sql:
            stats[diff]["exec_err"] += 1
        if not has_sql:
            stats[diff]["no_sql"] += 1

        results.append({
            "query_id": i, "db_id": db_id, "difficulty": diff,
            "question": question[:100],
            "gold_sql": gold_sql[:200], "pred_sql": pred_sql[:200],
            "correct": correct, "gen_time": gen_time,
            "error": pred_err[:100] if pred_err else None
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            remaining = (len(bird) - i - 1) / rate if rate > 0 else 0
            total_correct = sum(s["correct"] for s in stats.values())
            total_done = sum(s["total"] for s in stats.values())
            acc = total_correct / total_done * 100 if total_done else 0
            print("  [%d/%d] %.1f%% acc | %.0f q/min | ~%.0f min left" % (
                i+1, len(bird), acc, rate, remaining), flush=True)

        if (i + 1) % 100 == 0:
            with open(PROJECT_ROOT / "results" / PARTIAL_FILE, "w") as f:
                json.dump({"stats": dict(stats), "results": results}, f, ensure_ascii=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 1 BASELINE RESULTS")
    print("=" * 70)
    print("Model: %s" % MODEL)
    print("Total time: %.1f minutes" % (elapsed / 60))
    print()

    print("%-15s %8s %8s %8s %8s" % ("Difficulty", "Total", "Correct", "Acc%", "Exec Err"))
    print("-" * 50)
    overall_total = 0
    overall_correct = 0
    for d in ["simple", "moderate", "challenging"]:
        s = stats[d]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print("%-15s %8d %8d %7.1f%% %8d" % (d, s["total"], s["correct"], acc, s["exec_err"]))
        overall_total += s["total"]
        overall_correct += s["correct"]

    overall_acc = overall_correct / overall_total * 100 if overall_total else 0
    print("-" * 50)
    print("%-15s %8d %8d %7.1f%%" % ("OVERALL", overall_total, overall_correct, overall_acc))

    output = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_minutes": elapsed / 60,
        "stats": {},
        "results": results,
    }
    for d in ["simple", "moderate", "challenging"]:
        s = stats[d]
        output["stats"][d] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": s["correct"] / s["total"] if s["total"] else 0,
            "exec_errors": s["exec_err"],
            "no_sql": s["no_sql"],
        }
    output["stats"]["overall"] = {
        "total": overall_total,
        "correct": overall_correct,
        "accuracy": overall_correct / overall_total if overall_total else 0,
    }

    with open(PROJECT_ROOT / "results" / RESULT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved to results/%s" % RESULT_FILE)
    print("Finished: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    run_baseline()
