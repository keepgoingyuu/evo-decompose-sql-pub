"""S2: Question Decomposition Pipeline (3+ LLM calls).

Step 1: Decompose question into sub-questions
Step 2-N: Generate SQL for each sub-question
Step N+1: Merge sub-SQLs into final query
"""
import re

from src.pipelines.base import BasePipeline
from src.utils.sql_utils import clean_sql
from src.utils.prompt_templates import S0_DIRECT, S2_DECOMPOSE, S2_SUB_SQL, S2_MERGE


class S2Decompose(BasePipeline):
    name = "s2_decompose"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""

        # Step 1: Decompose question
        decompose_prompt = S2_DECOMPOSE.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        decompose_raw = self._call_llm(decompose_prompt, "decompose")
        sub_questions = self._parse_sub_questions(decompose_raw)

        # Fallback: if only 1 or 0 sub-questions, do direct generation (S0)
        if len(sub_questions) <= 1:
            self.steps.append({"step": "decompose_fallback", "raw": "simple question, using S0 direct"})
            direct_prompt = S0_DIRECT.format(
                schema=schema, question=question, evidence_section=evidence_section
            )
            gen_raw = self._call_llm(direct_prompt, "direct_generate")
            return self._make_result(clean_sql(gen_raw))

        # Steps 2-N: Generate SQL for each sub-question
        sub_sqls = []
        for idx, sub_q in enumerate(sub_questions):
            sub_prompt = S2_SUB_SQL.format(
                schema=schema, sub_question=sub_q, question=question,
                evidence_section=evidence_section
            )
            sub_raw = self._call_llm(sub_prompt, "sub_sql_%d" % (idx + 1))
            sub_sqls.append({"sub_question": sub_q, "sql": clean_sql(sub_raw)})

        # Step N+1: Merge
        sub_queries_section = "\n".join(
            "Sub-question %d: %s\nSQL: %s\n" % (i + 1, s["sub_question"], s["sql"])
            for i, s in enumerate(sub_sqls)
        )
        merge_prompt = S2_MERGE.format(
            schema=schema, question=question, evidence_section=evidence_section,
            sub_queries_section=sub_queries_section
        )
        merge_raw = self._call_llm(merge_prompt, "merge")
        pred_sql = clean_sql(merge_raw)

        return self._make_result(pred_sql)

    def _parse_sub_questions(self, raw: str) -> list:
        """Extract numbered sub-questions from decomposition response."""
        lines = raw.strip().split("\n")
        questions = []
        for line in lines:
            # Match patterns like "1. ...", "1) ...", "- ..."
            match = re.match(r'^\s*(?:\d+[\.\)]\s*|-\s*)(.*)', line)
            if match:
                q = match.group(1).strip()
                if q:
                    questions.append(q)
        return questions
