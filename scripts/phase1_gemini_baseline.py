"""Phase 1: Gemini 3 Flash baseline on full BIRD dev set (S0 direct generation)."""
import json, sqlite3, time, requests, os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
DB_DIR = BIRD_DIR / "dev_databases"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3-flash-preview"
DELAY = 0.5  # Tier 1 paid, aggressive
RESULT_FILE = "phase1_gemini3flash_baseline.json"
PARTIAL_FILE = "phase1_gemini3flash_baseline_partial.json"

def call_gemini(prompt, retries=5):
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + MODEL + ":generateContent?key=" + GEMINI_KEY
    for attempt in range(retries):
        try:
            r = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}
            }, timeout=60)
            data = r.json()
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                err = data["error"]
                if err.get("code") == 429 or "RESOURCE_EXHAUSTED" in str(err):
                    wait = min((attempt + 1) * 15, 60)
                    print(" [rl %ds]" % wait, end="", flush=True)
                    time.sleep(wait)
                    continue
                return "ERROR: " + err.get("message", "")[:200]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return "ERROR: " + str(e)
    return "ERROR: max retries"

def clean_sql(raw):
    sql = raw.strip()
    for tag in ["<s>", "</s>", "[SQL]", "[/SQL]", "```sql", "```", "sql\n"]:
        sql = sql.replace(tag, "")
    sql = sql.strip()
    if "SELECT" in sql.upper():
        sql = sql[sql.upper().find("SELECT"):]
    sql = sql.rstrip(";").strip()
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    # Remove trailing markdown/explanation
    lines = sql.split("\n")
    sql_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**") or stripped.startswith("Note:") or stripped.startswith("This "):
            break
        sql_lines.append(line)
    return "\n".join(sql_lines).strip()

def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schemas = [row[0] for row in cur.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)

def execute_sql(db_path, sql, timeout=30):
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

PROMPT_TMPL = """You are a SQL expert. Generate a SQLite-compatible SQL query to answer the question.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Rules:
- SQLite syntax ONLY (no ILIKE, no ::type, no date_part, no FILTER)
- Only use tables and columns from the schema above
- Output ONLY the raw SQL query, no explanations, no markdown
"""

def run_baseline():
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)

    print("=" * 70)
    print("PHASE 1 BASELINE: Gemini 3 Flash on BIRD dev set")
    print("Total queries: %d" % len(bird))
    print("Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    diff_counts = defaultdict(int)
    for q in bird:
        diff_counts[q["difficulty"]] += 1
    for d, c in sorted(diff_counts.items()):
        print("  %s: %d queries" % (d, c))

    # Resume from partial results if available
    results = []
    done_ids = set()
    partial_path = PROJECT_ROOT / "results" / PARTIAL_FILE
    if partial_path.exists():
        with open(partial_path) as pf:
            partial = json.load(pf)
            results = partial.get("results", [])
            done_ids = {r["query_id"] for r in results}
            # Rebuild stats from partial results
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

        schema = get_schema(db_path)
        evidence_section = "Evidence: " + evidence if evidence else ""
        prompt = PROMPT_TMPL.format(question=question, schema=schema, evidence_section=evidence_section)

        t0 = time.time()
        raw = call_gemini(prompt)
        gen_time = time.time() - t0
        pred_sql = clean_sql(raw)

        # Rate limit delay
        time.sleep(DELAY)

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

        # Progress every 50 queries
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            remaining = (len(bird) - i - 1) / rate if rate > 0 else 0
            total_correct = sum(s["correct"] for s in stats.values())
            total_done = sum(s["total"] for s in stats.values())
            acc = total_correct / total_done * 100 if total_done else 0
            print("  [%d/%d] %.1f%% acc | %.0f q/min | ~%.0f min remaining" % (
                i+1, len(bird), acc, rate, remaining), flush=True)

        # Save every 100 queries
        if (i + 1) % 100 == 0:
            with open(PROJECT_ROOT / "results" / PARTIAL_FILE, "w") as f:
                json.dump({"stats": dict(stats), "results": results}, f, ensure_ascii=False)

    # Final summary
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
