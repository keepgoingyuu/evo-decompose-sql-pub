"""All prompt templates for S0-S4 pipelines and self-correction."""

# ============================================================
# S0: Direct generation (same as Phase 1 baseline)
# ============================================================

S0_DIRECT = """You are a SQL expert. Generate a SQLite-compatible SQL query to answer the question.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Rules:
- SQLite syntax ONLY (no ILIKE, no ::type, no date_part, no FILTER)
- Only use tables and columns from the schema above
- Output ONLY the raw SQL query, no explanations, no markdown"""

# ============================================================
# S1: Schema Filtering Pipeline (2 calls)
# ============================================================

S1_SCHEMA_LINK = """Given a natural language question and a database schema, identify the relevant tables and columns needed to answer the question.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Output a JSON object with this exact format:
{{"tables": ["table1", "table2"], "columns": ["table1.col1", "table2.col2"]}}

Only output the JSON, nothing else."""

S1_GENERATE = """You are a SQL expert. Generate a SQLite-compatible SQL query to answer the question.

Relevant Schema (filtered):
{schema}

Question: {question}
{evidence_section}

Rules:
- SQLite syntax ONLY (no ILIKE, no ::type, no date_part, no FILTER)
- Only use tables and columns from the schema above
- Output ONLY the raw SQL query, no explanations, no markdown"""

# ============================================================
# S2: Question Decomposition Pipeline (3+ calls)
# ============================================================

S2_DECOMPOSE = """Break down this complex database question into 2-4 simpler sub-questions that can each be answered with a single SQL query.

Database Schema:
{schema}

Original Question: {question}
{evidence_section}

Output a numbered list of sub-questions:
1. [first sub-question]
2. [second sub-question]
...

Each sub-question should be self-contained and answerable from the schema.
If the question is already simple enough, output just one sub-question identical to the original.
Only output the numbered list, nothing else."""

S2_SUB_SQL = """You are a SQL expert. Generate a SQLite-compatible SQL query for this sub-question.

Database Schema:
{schema}

Sub-question: {sub_question}
Context (original question): {question}
{evidence_section}

Rules:
- SQLite syntax ONLY
- Only use tables and columns from the schema above
- Output ONLY the raw SQL query, no explanations, no markdown"""

S2_MERGE = """You are a SQL expert. Given sub-queries that answer parts of a complex question, combine them into a single final SQL query.

Database Schema:
{schema}

Original Question: {question}
{evidence_section}

Sub-questions and their SQL:
{sub_queries_section}

Generate a single SQLite-compatible SQL query that answers the original question by combining the sub-queries above.
You may use CTEs (WITH clauses), subqueries, JOINs, or any approach to combine them.
Output ONLY the raw SQL query, no explanations, no markdown."""

# ============================================================
# S3: Skeleton-then-Fill Pipeline (2 calls)
# ============================================================

S3_SKELETON = """You are a SQL expert. Generate a SQL skeleton (template) for this question using placeholders.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Generate a SQL skeleton where:
- Use actual SQL keywords (SELECT, FROM, WHERE, JOIN, GROUP BY, etc.)
- Use [TABLE] for table names you're unsure about
- Use [COLUMN] for column names you're unsure about
- Use [VALUE] for literal values you're unsure about
- You CAN use actual names if you're confident about them

Example output:
SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] = [VALUE] GROUP BY [COLUMN]

Output ONLY the SQL skeleton, no explanations."""

S3_FILL = """You are a SQL expert. Fill in the placeholders in this SQL skeleton with actual table names, column names, and values from the schema.

Database Schema:
{schema}

Question: {question}
{evidence_section}

SQL Skeleton:
{skeleton}

Replace all [TABLE], [COLUMN], and [VALUE] placeholders with actual names from the schema.
Make sure the final SQL is valid SQLite syntax.
Output ONLY the completed SQL query, no explanations, no markdown."""

# ============================================================
# S4: DIN-SQL Style Pipeline (3 calls)
# ============================================================

S4_SCHEMA_LINK = """Analyze the question and identify the relevant database elements needed to write the SQL query.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Identify:
1. Relevant tables and their roles
2. Key columns needed (for SELECT, WHERE, JOIN, GROUP BY)
3. Join conditions between tables
4. Filter conditions and their values

Output format:
Tables: table1 (role), table2 (role)
Columns: table1.col1 (for SELECT), table2.col2 (for WHERE)
Joins: table1.col = table2.col
Conditions: column operator value"""

S4_CLASSIFY = """Classify the difficulty of this SQL query based on the question and schema analysis.

Question: {question}
Schema Analysis:
{schema_analysis}

Classification rules:
- EASY: Single table, no joins, simple aggregation or filtering
- MEDIUM: 2-3 table joins, moderate aggregation, straightforward conditions
- HARD: 4+ table joins, nested subqueries, set operations (UNION/INTERSECT/EXCEPT), complex aggregation with HAVING, or correlated subqueries

Output ONLY one word: EASY, MEDIUM, or HARD"""

S4_GENERATE_EASY = """You are a SQL expert. Generate a simple SQLite query.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Schema Analysis:
{schema_analysis}

This is a simple query (single table, basic operations).
Output ONLY the SQL query, no explanations."""

S4_GENERATE_MEDIUM = """You are a SQL expert. Generate a SQLite query with joins.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Schema Analysis:
{schema_analysis}

This query requires joining tables. Pay attention to:
- Correct join conditions from the schema analysis
- Proper column references with table aliases
Output ONLY the SQL query, no explanations."""

S4_GENERATE_HARD = """You are a SQL expert. Generate a complex SQLite query.

Database Schema:
{schema}

Question: {question}
{evidence_section}

Schema Analysis:
{schema_analysis}

This is a complex query. Think step by step:
1. Identify the main information requested
2. Determine the joins and subqueries needed
3. Add proper filtering and aggregation
4. Verify the query logic

Output ONLY the SQL query, no explanations, no markdown."""

# ============================================================
# Self-Correction
# ============================================================

SELF_CORRECT = """The following SQL query produced an execution error. Fix it.

Database Schema:
{schema}

Question: {question}

Original SQL:
{sql}

Error message:
{error}

Fix the SQL to resolve this error while still answering the question correctly.
Common fixes: correct table/column names, fix syntax, add missing quotes, handle NULL values.
Output ONLY the corrected SQL query, no explanations, no markdown."""
