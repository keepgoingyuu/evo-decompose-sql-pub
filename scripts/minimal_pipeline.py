"""
Minimal Pipeline: 驗證不同分解策略是否產生不同結果
Week 3-4 核心實驗

策略：
  S0: 直接生成（不分解）
  S4: DIN-SQL 風格分解（schema linking → 查詢分類 → 分步生成）

目標：
  - 在 10 筆 BIRD challenging 查詢上比較兩種策略
  - 驗證 end-to-end pipeline 可跑
  - 觀察不同策略是否有不同結果
"""

import json
import sqlite3
import os
import time
import requests
from pathlib import Path

# === Configuration ===
PROJECT_ROOT = Path(__file__).parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
BIRD_DB_DIR = BIRD_DIR / "dev_databases"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "sqlcoder"
TIMEOUT = 120

# === 10 Selected Challenging Queries (diverse databases + SQL patterns) ===
SELECTED_IDS = [
    # california_schools: JOIN + admin lookup
    1,    # "Under whose administration is the school with the highest..."
    # california_schools: JOIN + COUNT + WHERE
    2,    # "What is the total number of non-chartered schools..."
    # financial: 3 JOINs + subquery + aggregation
    5,    # "List out the account numbers of female clients..."
    # financial: CAST + SUM + percentage
    6,    # "For the branch which located in south Bohemia..."
    # financial: growth rate CASE WHEN + STRFTIME
    10,   # "What was the growth rate of the total amount of loans..."
    # toxicology: nested subquery + AVG + GROUP BY
    12,   # "On average how many carcinogenic molecules are single bonded?"
    # toxicology: CASE WHEN + COUNT DISTINCT + percentage
    17,   # "What percentage of carcinogenic-type molecules does not contain fluorine?"
    # superhero: multiple JOINs + WHERE conditions
    None, # will find by db_id
    # card_games: percentage + CASE WHEN
    None,
    # european_football_2: JOINs + GROUP BY + ORDER BY
    None,
]


def load_bird_data():
    """Load BIRD dev data and select 10 challenging queries."""
    with open(BIRD_DIR / "dev.json", "r") as f:
        bird_data = json.load(f)

    challenging = [item for item in bird_data if item["difficulty"] == "challenging"]

    # Select specific queries by index + first from underrepresented DBs
    selected = []
    indices = [i for i in SELECTED_IDS if i is not None]

    for idx in indices:
        selected.append(challenging[idx])

    # Add first challenging from superhero, card_games, european_football_2
    for db in ["superhero", "card_games", "european_football_2"]:
        item = next(i for i in challenging if i["db_id"] == db)
        selected.append(item)

    return selected[:10]


def get_schema(db_path):
    """Get CREATE TABLE statements from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schemas = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)


def clean_sql(raw):
    """Clean model output to valid SQL."""
    sql = raw.strip()
    sql = sql.replace("<s>", "").replace("</s>", "")
    sql = sql.replace("[SQL]", "").replace("[/SQL]", "")
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.strip()
    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()
    # Take only first statement
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    return sql


def call_llm(prompt, model=MODEL):
    """Call Ollama API and return response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        return response.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def execute_sql(db_path, sql):
    """Execute SQL and return results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        return cursor.fetchall(), None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


# === Strategy S0: Direct Generation (No Decomposition) ===
S0_PROMPT = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

{evidence_section}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""


def strategy_s0(question, schema, evidence=""):
    """S0: Direct generation without decomposition."""
    evidence_section = f"### Evidence\n{evidence}" if evidence else ""
    prompt = S0_PROMPT.format(
        question=question, schema=schema, evidence_section=evidence_section
    )
    raw = call_llm(prompt)
    return clean_sql(raw)


# === Strategy S4: DIN-SQL Style Decomposition ===
S4_STEP1_PROMPT = """### Task: Schema Linking
Given the question and database schema, identify the relevant tables and columns.

### Database Schema
{schema}

### Question
{question}

{evidence_section}

### Instructions
List the relevant tables and columns needed to answer this question.
Format: table_name.column_name

### Relevant Schema Elements:
"""

S4_STEP2_PROMPT = """### Task: Query Classification and Decomposition
Given the question and relevant schema, classify the query type and decompose it into sub-steps.

### Question
{question}

### Relevant Schema
{relevant_schema}

{evidence_section}

### Instructions
1. Classify the query: SIMPLE (single table, no join) / MODERATE (1-2 joins) / COMPLEX (3+ joins, subqueries, aggregations)
2. Break down the query into logical sub-steps

### Classification and Steps:
"""

S4_STEP3_PROMPT = """### Task: SQL Generation
Generate a SQL query following the decomposed steps.

### Database Schema
{schema}

### Question
{question}

### Relevant Schema Elements
{relevant_schema}

### Query Plan (follow these steps)
{query_plan}

{evidence_section}

### Instructions
Generate a SQLite-compatible SQL query. Use the decomposed steps as a guide.
Only output the SQL query.

### SQL Query:
"""


def strategy_s4(question, schema, evidence=""):
    """S4: DIN-SQL style decomposition (3 steps)."""
    evidence_section = f"### Evidence\n{evidence}" if evidence else ""

    # Step 1: Schema Linking
    step1_prompt = S4_STEP1_PROMPT.format(
        schema=schema, question=question, evidence_section=evidence_section
    )
    relevant_schema = call_llm(step1_prompt)

    # Step 2: Classification + Decomposition
    step2_prompt = S4_STEP2_PROMPT.format(
        question=question,
        relevant_schema=relevant_schema.strip(),
        evidence_section=evidence_section,
    )
    query_plan = call_llm(step2_prompt)

    # Step 3: SQL Generation with plan
    step3_prompt = S4_STEP3_PROMPT.format(
        schema=schema,
        question=question,
        relevant_schema=relevant_schema.strip(),
        query_plan=query_plan.strip(),
        evidence_section=evidence_section,
    )
    raw = call_llm(step3_prompt)
    return clean_sql(raw)


def compare_results(gold_result, pred_result):
    """Compare execution results (Execution Accuracy)."""
    if gold_result is None or pred_result is None:
        return False
    return str(gold_result) == str(pred_result)


def run_pipeline():
    """Run the minimal pipeline experiment."""
    print("=" * 70)
    print("MINIMAL PIPELINE: S0 (Direct) vs S4 (DIN-SQL Decomposition)")
    print(f"Model: {MODEL}")
    print("=" * 70)

    queries = load_bird_data()
    print(f"\nSelected {len(queries)} challenging queries from {len(set(q['db_id'] for q in queries))} databases\n")

    results = []

    for i, query in enumerate(queries):
        db_id = query["db_id"]
        question = query["question"]
        gold_sql = query["SQL"]
        evidence = query.get("evidence", "")
        db_path = str(BIRD_DB_DIR / db_id / f"{db_id}.sqlite")

        print(f"\n{'─'*70}")
        print(f"Query {i+1}/10 [{db_id}]")
        print(f"Q: {question[:80]}...")
        if evidence:
            print(f"Evidence: {evidence[:80]}...")
        print(f"Gold SQL: {gold_sql[:80]}...")

        # Get schema
        schema = get_schema(db_path)

        # Execute gold SQL
        gold_result, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            print(f"  Gold SQL ERROR: {gold_err}")
            results.append({"query_id": i, "db_id": db_id, "s0_correct": False, "s4_correct": False, "error": "gold_error"})
            continue

        # Strategy S0
        print("\n  [S0 - Direct]", end=" ", flush=True)
        t0 = time.time()
        s0_sql = strategy_s0(question, schema, evidence)
        s0_time = time.time() - t0
        s0_result, s0_err = execute_sql(db_path, s0_sql) if s0_sql else (None, "empty")
        s0_correct = compare_results(gold_result, s0_result)
        print(f"{'CORRECT' if s0_correct else 'WRONG'} ({s0_time:.1f}s)")
        print(f"    SQL: {s0_sql[:80]}")
        if s0_err:
            print(f"    Error: {s0_err[:60]}")

        # Strategy S4
        print("  [S4 - DIN-SQL]", end=" ", flush=True)
        t0 = time.time()
        s4_sql = strategy_s4(question, schema, evidence)
        s4_time = time.time() - t0
        s4_result, s4_err = execute_sql(db_path, s4_sql) if s4_sql else (None, "empty")
        s4_correct = compare_results(gold_result, s4_result)
        print(f"{'CORRECT' if s4_correct else 'WRONG'} ({s4_time:.1f}s)")
        print(f"    SQL: {s4_sql[:80]}")
        if s4_err:
            print(f"    Error: {s4_err[:60]}")

        # Compare
        different = (s0_sql != s4_sql)
        outcome_different = (s0_correct != s4_correct)
        if outcome_different:
            print(f"  >>> DIFFERENT OUTCOME: S0={'CORRECT' if s0_correct else 'WRONG'}, S4={'CORRECT' if s4_correct else 'WRONG'}")

        results.append({
            "query_id": i,
            "db_id": db_id,
            "question": question[:100],
            "gold_sql": gold_sql[:200],
            "s0_sql": s0_sql[:200],
            "s4_sql": s4_sql[:200],
            "s0_correct": s0_correct,
            "s4_correct": s4_correct,
            "s0_time": s0_time,
            "s4_time": s4_time,
            "sql_different": different,
            "outcome_different": outcome_different,
        })

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    s0_acc = sum(1 for r in results if r["s0_correct"]) / len(results)
    s4_acc = sum(1 for r in results if r["s4_correct"]) / len(results)
    sql_diff = sum(1 for r in results if r.get("sql_different", False))
    outcome_diff = sum(1 for r in results if r.get("outcome_different", False))
    s0_total_time = sum(r.get("s0_time", 0) for r in results)
    s4_total_time = sum(r.get("s4_time", 0) for r in results)

    print(f"\nS0 (Direct):      {sum(1 for r in results if r['s0_correct'])}/10 correct ({s0_acc:.0%})")
    print(f"S4 (DIN-SQL):     {sum(1 for r in results if r['s4_correct'])}/10 correct ({s4_acc:.0%})")
    print(f"\nSQL Different:    {sql_diff}/10 queries generated different SQL")
    print(f"Outcome Different: {outcome_diff}/10 queries had different correctness")
    print(f"\nTotal time S0:    {s0_total_time:.1f}s ({s0_total_time/10:.1f}s avg)")
    print(f"Total time S4:    {s4_total_time:.1f}s ({s4_total_time/10:.1f}s avg)")

    print("\nPer-query results:")
    print(f"{'#':>2} {'DB':>25} {'S0':>8} {'S4':>8} {'Diff?':>6}")
    print("-" * 55)
    for r in results:
        s0 = "CORRECT" if r["s0_correct"] else "WRONG"
        s4 = "CORRECT" if r["s4_correct"] else "WRONG"
        diff = "YES" if r.get("outcome_different") else ""
        print(f"{r['query_id']+1:>2} {r['db_id']:>25} {s0:>8} {s4:>8} {diff:>6}")

    # Go/No-Go Decision
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)
    if outcome_diff >= 2:
        print(f"GO: {outcome_diff} queries had different outcomes between strategies.")
        print("Different decomposition strategies DO produce different results.")
        print("Proceed with full experiment design.")
    elif sql_diff >= 5:
        print(f"CAUTIOUS GO: {sql_diff} queries had different SQL but only {outcome_diff} different outcomes.")
        print("Strategies produce different SQL, but accuracy difference is small.")
        print("Consider refining decomposition strategies.")
    else:
        print(f"NO-GO: Only {sql_diff} queries had different SQL, {outcome_diff} different outcomes.")
        print("Need to redesign decomposition strategies for more diversity.")

    # Save results
    output_path = PROJECT_ROOT / "results"
    output_path.mkdir(exist_ok=True)
    with open(output_path / "minimal_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results/minimal_pipeline_results.json")


if __name__ == "__main__":
    run_pipeline()
