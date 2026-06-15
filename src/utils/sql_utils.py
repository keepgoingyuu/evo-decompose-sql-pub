"""SQL extraction, execution, and comparison utilities."""
import re
import sqlite3
import time


def clean_sql(raw: str) -> str:
    """Extract clean SQL from LLM response text.

    Handles: ```sql code blocks, tag wrappers, leading comments,
    trailing explanations, multiple statements.
    """
    text = raw.strip()

    # 1) Try to extract from ```sql ... ``` code blocks first
    code_blocks = re.findall(r"```\w*\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        sql = code_blocks[-1].strip()
    else:
        # Fallback: strip tags and find SELECT
        for tag in ["<s>", "</s>", "[SQL]", "[/SQL]"]:
            text = text.replace(tag, "")
        text = text.strip()
        if "SELECT" in text.upper():
            text = text[text.upper().find("SELECT"):]
        sql = text.strip()

    # Remove leading comments
    lines = sql.split("\n")
    sql_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--") and not sql_lines:
            continue
        if stripped.startswith("**") or stripped.startswith("Note:") or \
           stripped.startswith("This query") or stripped.startswith("This ") or \
           stripped.startswith("Explanation"):
            break
        sql_lines.append(line)

    sql = "\n".join(sql_lines).strip()
    sql = sql.rstrip(";").strip()
    # Keep only first statement
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    return sql


def execute_sql(db_path: str, sql: str, timeout_ms: int = 5000, exec_timeout: int = 30):
    """Execute SQL and return (results, error). Error is None on success.

    Uses sqlite3 progress_handler to interrupt long-running queries.
    exec_timeout: max seconds for query execution (prevents runaway queries).
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = %d" % timeout_ms)

    # Use progress_handler to check elapsed time periodically
    start = time.monotonic()

    def _check_timeout():
        if time.monotonic() - start > exec_timeout:
            return 1  # non-zero aborts the query
        return 0

    conn.set_progress_handler(_check_timeout, 10000)  # check every 10k SQLite ops

    try:
        cur = conn.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        return result, None
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return None, "execution_timeout"
        return None, str(e)
    except Exception as e:
        return None, str(e)
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def compare_results(gold_result, pred_result) -> bool:
    """Compare gold and predicted SQL results by string equality."""
    if gold_result is None or pred_result is None:
        return False
    return str(gold_result) == str(pred_result)
