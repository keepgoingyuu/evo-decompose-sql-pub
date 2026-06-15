"""Gemini slow background run: 15 queries × 2 models, 10s delay between calls."""
import json, sqlite3, time, requests, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
DB_DIR = BIRD_DIR / "dev_databases"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
DELAY = 2  # seconds between API calls (Tier 1 paid)

def call_gemini(prompt, model, retries=8):
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + GEMINI_KEY
    for attempt in range(retries):
        try:
            r = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048}
            }, timeout=90)
            data = r.json()
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                err = data["error"]
                if err.get("code") == 429 or "RESOURCE_EXHAUSTED" in str(err) or "high demand" in str(err).lower():
                    wait = min((attempt + 1) * 30, 120)
                    print("    [rate-limit, wait %ds, attempt %d/%d]" % (wait, attempt+1, retries), flush=True)
                    time.sleep(wait)
                    continue
                return "ERROR: " + err.get("message", "")[:200]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(15)
                continue
            return "ERROR: " + str(e)
    return "ERROR: max retries after %d attempts" % retries

def clean_sql(raw):
    sql = raw.strip()
    for tag in ["```sql", "```", "[SQL]", "[/SQL]", "sql\n"]:
        sql = sql.replace(tag, "")
    sql = sql.strip()
    # Find first SELECT
    upper = sql.upper()
    if "SELECT" in upper:
        sql = sql[upper.find("SELECT"):]
    # Remove trailing text after the query
    sql = sql.rstrip(";").strip()
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    # Remove trailing markdown or explanation
    lines = sql.split("\n")
    sql_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**") or stripped.startswith("Note:") or stripped.startswith("This "):
            break
        sql_lines.append(line)
    sql = "\n".join(sql_lines).strip()
    return sql

def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schemas = [row[0] for row in cur.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)

def execute_sql(db_path, sql):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall(), None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

S0_TMPL = """You are a SQL expert. Generate a SQLite query to answer the question.

Database Schema:
{schema}

Question: {question}
{ev}

CRITICAL RULES:
- SQLite syntax ONLY
- Only use tables/columns from the schema above
- Output ONLY the raw SQL query - no explanations, no markdown, no comments
"""

S4_TMPL = """You are a SQL expert. Generate a SQLite query by following these steps internally:
1. Identify relevant tables and columns
2. Plan JOINs, WHERE, GROUP BY, ORDER BY
3. Write the final query

Database Schema:
{schema}

Question: {question}
{ev}

CRITICAL RULES:
- SQLite syntax ONLY
- Only use tables/columns from the schema above
- Output ONLY the final raw SQL query - no explanations, no steps, no markdown
"""

def strategy_s0(question, schema, evidence, model):
    ev = "\nEvidence: " + evidence if evidence else ""
    raw = call_gemini(S0_TMPL.format(question=question, schema=schema, ev=ev), model)
    time.sleep(DELAY)
    return clean_sql(raw)

def strategy_s4(question, schema, evidence, model):
    ev = "\nEvidence: " + evidence if evidence else ""
    raw = call_gemini(S4_TMPL.format(question=question, schema=schema, ev=ev), model)
    time.sleep(DELAY)
    return clean_sql(raw)

# Load queries
with open(BIRD_DIR / "dev.json") as f:
    bird = json.load(f)

def select_queries(difficulty, count=5):
    queries = [q for q in bird if q["difficulty"] == difficulty]
    selected, seen_db = [], set()
    for q in queries:
        if len(selected) >= count:
            break
        p = DB_DIR / q["db_id"] / (q["db_id"] + ".sqlite")
        if not p.exists():
            continue
        if q["db_id"] in seen_db and len(selected) < count - 1:
            continue
        conn = sqlite3.connect(str(p))
        try:
            c = conn.cursor()
            c.execute(q["SQL"])
            if c.fetchall():
                selected.append(q)
                seen_db.add(q["db_id"])
        except:
            pass
        finally:
            conn.close()
    return selected

test_queries = []
for d in ["simple", "moderate", "challenging"]:
    qs = select_queries(d, 5)
    for q in qs:
        q["_difficulty"] = d
    test_queries.extend(qs)

MODELS = ["gemini-3-flash-preview", "gemini-2.0-flash"]

print("=" * 70)
print("GEMINI SLOW RUN: %d queries x %d models x 2 strategies" % (len(test_queries), len(MODELS)))
print("Delay: %ds between calls. Estimated: ~%d minutes per model" % (DELAY, len(test_queries) * 2 * DELAY // 60 + 5))
print("Started: " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 70)

all_results = {}

for model in MODELS:
    print("\n--- %s ---\n" % model)
    results = []

    for i, q in enumerate(test_queries):
        db_id = q["db_id"]
        diff = q["_difficulty"]
        question = q["question"]
        gold_sql = q["SQL"]
        evidence = q.get("evidence", "")
        db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))
        schema = get_schema(db_path)

        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            print("  Q%d GOLD ERR: %s" % (i+1, gold_err[:50]))
            continue

        print("  Q%d/%d [%s][%s]" % (i+1, len(test_queries), diff, db_id), flush=True)

        t0 = time.time()
        s0_sql = strategy_s0(question, schema, evidence, model)
        s0_time = time.time() - t0
        has_sql = s0_sql and "SELECT" in s0_sql.upper()
        s0_res, s0_err = execute_sql(db_path, s0_sql) if has_sql else (None, "no SQL")
        s0_ok = str(gold_result) == str(s0_res) if gold_result and s0_res else False
        print("    S0: %s (%ds) %s" % ("CORRECT" if s0_ok else "WRONG", int(s0_time), ("err: " + str(s0_err)[:60]) if s0_err and not s0_ok else ""), flush=True)

        t0 = time.time()
        s4_sql = strategy_s4(question, schema, evidence, model)
        s4_time = time.time() - t0
        has_sql = s4_sql and "SELECT" in s4_sql.upper()
        s4_res, s4_err = execute_sql(db_path, s4_sql) if has_sql else (None, "no SQL")
        s4_ok = str(gold_result) == str(s4_res) if gold_result and s4_res else False
        print("    S4: %s (%ds) %s" % ("CORRECT" if s4_ok else "WRONG", int(s4_time), ("err: " + str(s4_err)[:60]) if s4_err and not s4_ok else ""), flush=True)

        if s0_ok != s4_ok:
            print("    >>> DIFFERENT OUTCOME: S0=%s S4=%s" % ("OK" if s0_ok else "X", "OK" if s4_ok else "X"), flush=True)

        results.append({
            "model": model, "query_id": i, "db_id": db_id, "difficulty": diff,
            "question": question[:100], "gold_sql": gold_sql[:200],
            "s0_sql": s0_sql[:300], "s4_sql": s4_sql[:300],
            "s0_correct": s0_ok, "s4_correct": s4_ok,
            "s0_time": s0_time, "s4_time": s4_time,
            "sql_different": s0_sql != s4_sql,
            "outcome_different": s0_ok != s4_ok,
        })

        # Save incrementally
        fn = "gemini_" + model.replace(".", "_").replace("-", "_") + "_results.json"
        with open(PROJECT_ROOT / "results" / fn, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    all_results[model] = results
    total = len(results)
    if total == 0:
        continue
    s0c = sum(1 for r in results if r["s0_correct"])
    s4c = sum(1 for r in results if r["s4_correct"])
    od = sum(1 for r in results if r["outcome_different"])
    print("\n  %s SUMMARY: S0=%d/%d S4=%d/%d | Outcome diff: %d/%d" % (model, s0c, total, s4c, total, od, total))

# Final
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print("%-25s %8s %8s %10s" % ("Model", "S0", "S4", "Out Diff"))
print("-" * 55)
print("%-25s %8s %8s %10s" % ("Qwen2.5-Coder-7B", "3/15", "3/15", "0/15"))
for m, res in all_results.items():
    t = len(res)
    if t == 0: continue
    s0 = sum(1 for r in res if r["s0_correct"])
    s4 = sum(1 for r in res if r["s4_correct"])
    od = sum(1 for r in res if r["outcome_different"])
    print("%-25s %8s %8s %10s" % (m, "%d/%d" % (s0,t), "%d/%d" % (s4,t), "%d/%d" % (od,t)))
print("\nFinished: " + time.strftime("%Y-%m-%d %H:%M:%S"))
