"""Schema loading, enrichment, and filtering utilities."""
import csv
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
DB_DIR = BIRD_DIR / "dev_databases"


def get_schema(db_path: str) -> str:
    """Get raw DDL schema from sqlite_master."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schemas = [row[0] for row in cur.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)


def load_column_descriptions(db_id: str) -> dict:
    """Load BIRD column descriptions from database_description CSVs."""
    desc_dir = DB_DIR / db_id / "database_description"
    col_descs = {}
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
                        parts.append(val_desc[:120])
                    if parts:
                        col_descs[(table_name.lower(), orig.lower())] = "; ".join(parts)
        except Exception:
            continue
    return col_descs


def _get_sample_values(db_path: str, table_name: str, col_name: str, limit: int = 3) -> list:
    """Get distinct sample values for a column."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 3000")
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT `%s` FROM `%s` WHERE `%s` IS NOT NULL LIMIT %d"
            % (col_name, table_name, col_name, limit)
        )
        vals = [str(row[0])[:50] for row in cur.fetchall()]
        conn.close()
        return vals
    except Exception:
        return []


def get_enriched_schema(db_path: str, db_id: str) -> str:
    """Build enriched DDL with column descriptions and sample values."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    col_descs = load_column_descriptions(db_id)

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    enriched = []
    for table in tables:
        cur.execute("PRAGMA table_info(`%s`)" % table)
        columns = cur.fetchall()
        if not columns:
            continue

        cur.execute("PRAGMA foreign_key_list(`%s`)" % table)
        fk_map = {}
        for fk in cur.fetchall():
            fk_map[fk[3]] = (fk[2], fk[4])

        lines = ["CREATE TABLE `%s` (" % table]
        col_lines = []
        pk_cols = [c[1] for c in columns if c[5]]

        for col in columns:
            cid, name, ctype, notnull, dflt, pk = col
            parts = ["    `%s`" % name, ctype if ctype else "TEXT"]
            if pk and len(pk_cols) == 1:
                parts.append("PRIMARY KEY")
            if notnull:
                parts.append("NOT NULL")

            comment_parts = []
            desc_key = (table.lower(), name.lower())
            if desc_key in col_descs:
                comment_parts.append(col_descs[desc_key])
            if name in fk_map:
                ref_t, ref_c = fk_map[name]
                comment_parts.append("references `%s`.`%s`" % (ref_t, ref_c))
            samples = _get_sample_values(db_path, table, name)
            if samples:
                comment_parts.append("examples: [%s]" % ", ".join(samples))

            col_line = " ".join(parts)
            if comment_parts:
                col_line += "  -- " + "; ".join(comment_parts)
            col_lines.append(col_line)

        lines.append(",\n".join(col_lines))
        if len(pk_cols) > 1:
            lines.append(",\n    PRIMARY KEY (%s)" % ", ".join("`%s`" % c for c in pk_cols))
        lines.append(")")
        enriched.append("\n".join(lines))

    conn.close()
    return "\n\n".join(enriched)


def get_table_names(db_path: str) -> list:
    """List all table names in the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    names = [row[0] for row in cur.fetchall()]
    conn.close()
    return names


def filter_schema_by_tables(full_schema: str, relevant_tables: list) -> str:
    """Keep only CREATE TABLE blocks matching relevant_tables (case-insensitive)."""
    relevant_lower = {t.lower().strip() for t in relevant_tables}
    blocks = re.split(r'\n\n+', full_schema)
    kept = []
    for block in blocks:
        match = re.match(r'CREATE\s+TABLE\s+[`"\[]?(\w+)', block, re.IGNORECASE)
        if match and match.group(1).lower() in relevant_lower:
            kept.append(block)
    if not kept:
        return full_schema  # fallback: return everything if nothing matched
    return "\n\n".join(kept)
