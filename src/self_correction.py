"""Self-correction wrapper: retry on execution errors by asking LLM to fix."""
from src.utils.sql_utils import clean_sql, execute_sql
from src.utils.prompt_templates import SELF_CORRECT


class SelfCorrectionWrapper:
    """Wraps any pipeline with execution-error-driven self-correction."""

    def __init__(self, pipeline, max_retries=2):
        self.pipeline = pipeline
        self.max_retries = max_retries

    def run(self, question, schema, evidence="", db_path=""):
        result = self.pipeline.run(question, schema, evidence, db_path)
        sql = result["pred_sql"]
        correction_attempts = 0

        for attempt in range(self.max_retries):
            if not sql or "SELECT" not in sql.upper():
                break
            _, error = execute_sql(db_path, sql)
            if error is None:
                break  # SQL executes successfully

            correction_attempts = attempt + 1
            fix_prompt = SELF_CORRECT.format(
                sql=sql, error=error, schema=schema, question=question
            )
            fix_raw = self.pipeline._call_llm(fix_prompt, "self_correct_%d" % (attempt + 1))
            sql = clean_sql(fix_raw)

        result["pred_sql"] = sql
        result["correction_attempts"] = correction_attempts
        result["steps"] = self.pipeline.steps
        result["call_count"] = self.pipeline.call_count
        return result
