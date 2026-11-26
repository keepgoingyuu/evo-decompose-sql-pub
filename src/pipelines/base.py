"""Base pipeline class for all multi-step strategies."""
import hashlib

from src.utils.api_clients import call_llm, last_call_meta
from src.utils.sql_utils import clean_sql


class BasePipeline:
    """Abstract base for S0-S4 pipelines."""

    name = "base"

    def __init__(self, model_name: str):
        self.model = model_name
        self.call_count = 0
        self.steps = []
        self._seed = None  # set by external runner if needed
        self._temperature = 0.0  # set by external runner for sampling experiments

    def run(self, question: str, schema: str, evidence: str = "", db_path: str = "") -> dict:
        """Run the pipeline. Returns {"pred_sql", "steps", "call_count"}."""
        raise NotImplementedError

    def _call_llm(self, prompt: str, step_name: str = "", seed=None) -> str:
        """Call LLM and track the step. seed param overrides self._seed if given."""
        self.call_count += 1
        s = seed if seed is not None else self._seed
        raw = call_llm(prompt, self.model, seed=s, temperature=self._temperature)
        # Full raw response is kept (it was cut to 500 chars and never saved before 2026-11);
        # the prompt is identified by hash and length, since schema + template reproduce it.
        self.steps.append({
            "step": step_name or f"call_{self.call_count}", "raw": raw,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(), "prompt_chars": len(prompt),
            **last_call_meta(),
        })
        return raw

    def _reset(self):
        """Reset per-query state."""
        self.call_count = 0
        self.steps = []

    def _make_result(self, pred_sql: str) -> dict:
        return {
            "pred_sql": pred_sql,
            "steps": self.steps,
            "call_count": self.call_count,
            "pipeline": self.name,
        }
