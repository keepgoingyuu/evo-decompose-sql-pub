"""S3: Skeleton-then-Fill Pipeline (2 LLM calls).

Step 1: Generate SQL skeleton with placeholders
Step 2: Fill placeholders with actual schema values
"""
from src.pipelines.base import BasePipeline
from src.utils.sql_utils import clean_sql
from src.utils.prompt_templates import S0_DIRECT, S3_SKELETON, S3_FILL


class S3Skeleton(BasePipeline):
    name = "s3_skeleton"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""

        # Step 1: Generate skeleton
        skeleton_prompt = S3_SKELETON.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        skeleton_raw = self._call_llm(skeleton_prompt, "skeleton")
        skeleton = clean_sql(skeleton_raw)

        # Fallback: if skeleton is empty or has no SELECT
        if not skeleton or "SELECT" not in skeleton.upper():
            self.steps.append({"step": "skeleton_fallback", "raw": "empty skeleton, using S0 direct"})
            direct_prompt = S0_DIRECT.format(
                schema=schema, question=question, evidence_section=evidence_section
            )
            gen_raw = self._call_llm(direct_prompt, "direct_generate")
            return self._make_result(clean_sql(gen_raw))

        # Step 2: Fill placeholders
        fill_prompt = S3_FILL.format(
            schema=schema, question=question, evidence_section=evidence_section,
            skeleton=skeleton
        )
        fill_raw = self._call_llm(fill_prompt, "fill")
        pred_sql = clean_sql(fill_raw)

        return self._make_result(pred_sql)
