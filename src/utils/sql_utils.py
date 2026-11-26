"""SQL extraction, execution, and comparison utilities."""
import re
import sqlite3
import time


# A leading common-table expression: WITH [RECURSIVE] name [(cols)] AS [[NOT] MATERIALIZED] (
_CTE_HEAD = re.compile(
    r"\bWITH\s+(?:RECURSIVE\s+)?(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|\w+)\s*(?:\([^()]*\)\s*)?"
    r"AS\s+(?:(?:NOT\s+)?MATERIALIZED\s+)?\(",
    re.IGNORECASE,
)
_SELECT_HEAD = re.compile(r"\bSELECT\b", re.IGNORECASE)


def _statement_start(text: str) -> int:
    """Offset of the first SQL statement in free text.

    The statement starts at the first SELECT, unless a CTE header (WITH name AS ()
    opens before it, in which case it starts at that WITH. Until 2026-11 this
    function's caller cut at the first SELECT unconditionally, which removed the
    `WITH name AS (` prefix of every un-fenced CTE query and left invalid SQL
    (see clean_sql_legacy and IS revision notes).
    """
    sel = _SELECT_HEAD.search(text)
    cte = _CTE_HEAD.search(text)
    if cte and (sel is None or cte.start() < sel.start()):
        return cte.start()
    return sel.start() if sel else 0


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
        text = text[_statement_start(text):]
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


def clean_sql_legacy(raw: str) -> str:
    """Extraction as used for all runs before 2026-11 (strips a leading WITH prefix). Kept only to quantify that defect.

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


def compare_results_bird(gold_result, pred_result) -> bool:
    """BIRD official EX rule (evaluation.py): set equality of result rows."""
    if gold_result is None or pred_result is None:
        return False
    try:
        return set(pred_result) == set(gold_result)
    except TypeError:
        return set(map(repr, pred_result)) == set(map(repr, gold_result))


def compare_results_spider(gold_result, pred_result, gold_sql: str) -> bool:
    """Spider execution-match rule, re-implemented after test-suite-sql-eval `result_eq`:
    multiset equality of rows up to a column permutation; row order matters only when
    the gold query has ORDER BY. (Re-implementation; cross-check against the official
    script before reporting.)"""
    import itertools
    from collections import Counter
    if gold_result is None or pred_result is None:
        return False
    g, p = list(gold_result), list(pred_result)
    if len(g) == 0 and len(p) == 0:
        return True
    if len(g) != len(p):
        return False
    nc = len(g[0])
    if len(p[0]) != nc:
        return False
    order_matters = "order by" in gold_sql.lower()

    def bag(rows):
        try:
            return Counter(rows)
        except TypeError:
            return Counter(map(repr, rows))

    try:
        gs = [{r[i] for r in g} for i in range(nc)]
        ps = [{r[i] for r in p} for i in range(nc)]
        cand = [[j for j in range(nc) if ps[j] == gs[i]] for i in range(nc)]
    except TypeError:
        cand = [list(range(nc))] * nc
    if any(len(c) == 0 for c in cand):
        return False
    total = 1
    for c in cand:
        total *= len(c)
    perms = itertools.product(*cand) if total <= 5000 else [tuple(range(nc))]
    for perm in perms:
        if len(set(perm)) != nc:
            continue
        pp = p if nc == 1 else [tuple(r[j] for j in perm) for r in p]
        if order_matters:
            if g == pp:
                return True
        elif bag(g) == bag(pp):
            return True
    return False
