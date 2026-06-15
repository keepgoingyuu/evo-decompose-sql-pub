"""S1: Schema Filtering Pipeline (2 LLM calls).

Step 1: Identify relevant tables/columns from full schema
Step 2: Generate SQL using filtered schema only
"""
import json
import re

from src.pipelines.base import BasePipeline
from src.utils.sql_utils import clean_sql
from src.utils.schema_utils import filter_schema_by_tables
from src.utils.prompt_templates import S1_SCHEMA_LINK, S1_GENERATE


class S1SchemaFilter(BasePipeline):
    name = "s1_schema_filter"

    def run(self, question, schema, evidence="", db_path=""):
        self._reset()
        evidence_section = "Evidence: " + evidence if evidence else ""

        # Step 1: Schema linking
        link_prompt = S1_SCHEMA_LINK.format(
            schema=schema, question=question, evidence_section=evidence_section
        )
        link_raw = self._call_llm(link_prompt, "schema_link")

        # Parse relevant tables from JSON response
        relevant_tables = self._parse_tables(link_raw)

        # Filter schema (fallback to full if parsing failed)
        if relevant_tables:
            filtered_schema = filter_schema_by_tables(schema, relevant_tables)
        else:
            filtered_schema = schema
            self.steps.append({"step": "schema_link_fallback", "raw": "parse failed, using full schema"})

        # Step 2: Generate SQL with filtered schema
        gen_prompt = S1_GENERATE.format(
            schema=filtered_schema, question=question, evidence_section=evidence_section
        )
        gen_raw = self._call_llm(gen_prompt, "generate_sql")
        pred_sql = clean_sql(gen_raw)

        return self._make_result(pred_sql)

    def _parse_tables(self, raw: str) -> list:
        """Extract table names from schema linking response."""
        # Try JSON parsing first
        try:
            # Find JSON in response
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                tables = data.get("tables", [])
                if tables:
                    return tables
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract table names mentioned after "tables:" or similar patterns
        tables = re.findall(r'["`](\w+)["`]', raw)
        return list(set(tables)) if tables else []
