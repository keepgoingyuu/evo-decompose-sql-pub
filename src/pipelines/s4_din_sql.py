"""S4: DIN-SQL Style Pipeline (3 LLM calls).

Step 1: Schema linking — identify tables, columns, joins, conditions
Step 2: Difficulty classification — EASY / MEDIUM / HARD
Step 3: SQL generation with difficulty-appropriate prompt
"""
from src.pipelines.base import BasePipeline
from src.utils.sql_utils import clean_sql
from src.utils.prompt_templates import (
    S4_SCHEMA_LINK, S4_CLASSIFY,
    S4_GENERATE_EASY, S4_GENERATE_MEDIUM, S4_GENERATE_HARD,
)


class S4DinSQL(BasePipeline):
    name = "s4_din_sql"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""

        # Step 1: Schema linking
        link_prompt = S4_SCHEMA_LINK.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        link_raw = self._call_llm(link_prompt, "schema_link")
        schema_analysis = link_raw.strip()

        # Fallback if schema linking returned nothing useful
        if not schema_analysis or len(schema_analysis) < 10:
            schema_analysis = "Full schema analysis unavailable. Use all tables."
            self.steps.append({"step": "link_fallback", "raw": "schema linking too short"})

        # Step 2: Difficulty classification
        classify_prompt = S4_CLASSIFY.format(
            question=question, schema_analysis=schema_analysis
        )
        classify_raw = self._call_llm(classify_prompt, "classify")
        difficulty = self._parse_difficulty(classify_raw)

        # Step 3: Generate SQL with difficulty-specific prompt
        gen_templates = {
            "EASY": S4_GENERATE_EASY,
            "MEDIUM": S4_GENERATE_MEDIUM,
            "HARD": S4_GENERATE_HARD,
        }
        gen_template = gen_templates[difficulty]
        gen_prompt = gen_template.format(
            schema=schema, question=question, evidence_section=evidence_section,
            schema_analysis=schema_analysis
        )
        gen_raw = self._call_llm(gen_prompt, "generate_%s" % difficulty.lower())
        pred_sql = clean_sql(gen_raw)

        return self._make_result(pred_sql)

    def _parse_difficulty(self, raw: str) -> str:
        """Extract EASY/MEDIUM/HARD from classification response."""
        text = raw.strip().upper()
        for level in ["HARD", "MEDIUM", "EASY"]:
            if level in text:
                return level
        return "HARD"  # default to hardest if unparseable
