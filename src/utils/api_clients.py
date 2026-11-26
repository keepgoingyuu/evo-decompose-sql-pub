"""Unified LLM client.

Original 4 models (kept for reproducing thesis Phase 1-4):
  qwen, gemini, deepseek, gpt41

Q1-path 5 models (April 2026 refresh):
  gpt5mini    - OpenAI GPT-5 mini (replaces gpt41 as US strong API anchor)
  gemini3f    - Gemini 3 Flash Preview (= same as 'gemini' alias, kept for clarity)
  deepseekv32 - DeepSeek V3.2 (deepseek-chat at api.deepseek.com points to V3.2 since Dec 2025)
  qwen25c7b   - Qwen2.5-Coder-7B local via Ollama (alias for 'qwen', preserved)
  gemma4      - Gemma 3 / 4 local via Ollama or transformers (TBD: model size)

All API calls use temperature=0 + seed parameter where supported, for
multi-seed reproducibility (seeds 1, 2, 3 by default).
"""
import os
import time
import requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

MODEL_CONFIG = {
    # ============ Original thesis models (keep, for back-compat) ============
    "qwen": {
        "url": "http://localhost:11434/api/chat",  # qwen
        "model": "qwen2.5-coder:7b",
        "timeout": 120, "delay": 0, "type": "ollama",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
        "model": "gemini-3-flash-preview",
        "timeout": 60, "delay": 0.1, "type": "gemini",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "timeout": 60, "delay": 0.3, "type": "openai_compat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "gpt41": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4.1",
        "timeout": 60, "delay": 0.2, "type": "openai_compat",
        "key_env": "OPENAI_API_KEY",
    },

    # ============ Q1-path refresh models (April 2026) ============
    "gpt5mini": {
        # GPT-5 mini: cheap reasoning model, 2026-Q1 Q1-path strong-anchor
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-5-mini",
        "timeout": 120, "delay": 0.2, "type": "openai_compat",
        "key_env": "OPENAI_API_KEY",
        "reasoning_effort": "minimal",  # avoid huge thinking traces, save $
    },
    "gpt54mini": {
        # GPT-5.4 mini (2026-03): upper capability anchor added 2026-09 to test
        # whether the capability-benefit relation holds across a model generation.
        # Verified 2026-09-01 to emit SQL with reasoning_tokens=0 under this config,
        # so its output behaviour matches the non-reasoning panel models.
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-5.4-mini",
        "timeout": 120, "delay": 0.2, "type": "openai_compat",
        "key_env": "OPENAI_API_KEY",
        "reasoning_effort": "none",  # this model rejects 'minimal'; 'none' keeps reasoning_tokens at 0
    },
    "gemini3f": {
        # Gemini 3 Flash Preview (alias of 'gemini', explicit name for paper)
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
        "model": "gemini-3-flash-preview",
        "timeout": 60, "delay": 0.1, "type": "gemini",
    },
    "deepseekv32": {
        # DeepSeek V3.2 (deepseek-chat API alias, V3.2 since Dec 2025)
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "timeout": 60, "delay": 0.3, "type": "openai_compat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "qwen25c7b": {
        # Alias of 'qwen', explicit version name for paper
        "url": "http://localhost:11434/api/chat",  # qwen
        "model": "qwen2.5-coder:7b",
        "timeout": 120, "delay": 0, "type": "ollama",
    },
    "gemma4": {
        # Gemma 4 26B MoE local via Ollama on GPU 1 (Q4 quant, ~3.8B active)
        "url": "http://localhost:11435/api/chat",
        "model": "gemma3n:e4b",
        "timeout": 180, "delay": 0, "type": "ollama",
    },
    "arctic": {
        # Arctic-Text2SQL-R1 7B local via Ollama on GPU 2 (RL baseline for RQ5)
        "url": "http://localhost:11436/api/chat",
        "model": "arctic-text2sql-r1:7b",  # TBD: actual ollama tag
        "timeout": 120, "delay": 0, "type": "ollama",
    },
}

# Cost per token (input, output) in USD per million tokens
COST_PER_M = {
    # Original thesis models
    "qwen": (0, 0),
    "gemini": (0.50, 3.00),     # 3 Flash Preview pricing (was wrong: 0.15/0.60 was 2.0 Flash)
    "deepseek": (0.27, 1.10),
    "gpt41": (2.00, 8.00),
    # Q1-path
    "gpt5mini": (0.25, 2.00),    # GPT-5 mini: $0.25/M in, $2/M out (note: reasoning tokens billed as output)
    "gpt54mini": (0.25, 2.00),   # assumed same tier as gpt-5-mini; NOT verified against official pricing 2026-09
    "gemini3f": (0.50, 3.00),
    "deepseekv32": (0.27, 1.10),
    "qwen25c7b": (0, 0),
    "gemma4": (0, 0),
    "arctic": (0, 0),
}

# Metadata of the most recent call in this thread (finish reason, token usage).
# Read by BasePipeline right after each call so truncated outputs can be detected.
import threading
_LAST = threading.local()


def last_call_meta() -> dict:
    return dict(getattr(_LAST, "meta", {}) or {})


_KEY_CACHE = {
    "DEEPSEEK_API_KEY": DEEPSEEK_KEY,
    "OPENAI_API_KEY": OPENAI_KEY,
    "GEMINI_API_KEY": GEMINI_KEY,
}


def _get_key(env_name: str) -> str:
    return _KEY_CACHE.get(env_name) or os.environ.get(env_name, "")


def _call_ollama(prompt, cfg, seed=None, temperature=0.0):
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # num_ctx was never set before 2026-11, i.e. the server default (4096 in the
        # Ollama builds used) applied; it is now explicit so the value is recorded.
        "options": {"temperature": temperature, "num_predict": 4096,
                    "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096"))},
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    r = requests.post(cfg["url"], json=payload, timeout=cfg["timeout"])
    data = r.json()
    _LAST.meta = {"finish": data.get("done_reason"), "in_tok": data.get("prompt_eval_count"),
                  "out_tok": data.get("eval_count"), "num_ctx": payload["options"]["num_ctx"]}
    return data.get("message", {}).get("content", "")


def _call_gemini(prompt, cfg, seed=None, temperature=0.0):
    key = _get_key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    url = cfg["url"] + "?key=" + key
    # Gemini 3 Flash thinks before answering and the thinking shares this budget; at 4096 about
    # 4% of BIRD answers were cut off mid-query (IS revision, 2026-11). Overridable per run.
    gen_cfg = {"temperature": temperature, "maxOutputTokens": int(os.environ.get("GEMINI_MAX_OUTPUT", "4096"))}
    if seed is not None:
        gen_cfg["seed"] = seed
    r = requests.post(url, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }, timeout=cfg["timeout"])
    data = r.json()
    if "candidates" in data:
        cand = data["candidates"][0]
        um = data.get("usageMetadata", {})
        _LAST.meta = {"finish": cand.get("finishReason"), "in_tok": um.get("promptTokenCount"),
                      "out_tok": um.get("candidatesTokenCount"), "think_tok": um.get("thoughtsTokenCount"),
                      "model_version": data.get("modelVersion"), "max_output": gen_cfg["maxOutputTokens"]}
        return cand["content"]["parts"][0]["text"]
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    return ""


def _call_openai_compat(prompt, cfg, seed=None, temperature=0.0):
    key = _get_key(cfg["key_env"])
    if not key:
        raise RuntimeError("%s not set in .env" % cfg["key_env"])
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
    }

    # GPT-5 series: uses max_completion_tokens, supports reasoning_effort, no temperature
    is_gpt5 = cfg["model"].startswith("gpt-5") or cfg["model"].startswith("o1") or cfg["model"].startswith("o3") or cfg["model"].startswith("o4")
    if is_gpt5:
        payload["max_completion_tokens"] = 8192  # higher to accommodate reasoning tokens
        if "reasoning_effort" in cfg:
            payload["reasoning_effort"] = cfg["reasoning_effort"]
    else:
        payload["max_tokens"] = 4096
        payload["temperature"] = float(temperature)
    if seed is not None:
        payload["seed"] = seed

    r = requests.post(cfg["url"], headers=headers, json=payload, timeout=cfg["timeout"])
    data = r.json()
    if "choices" in data:
        u = data.get("usage", {}) or {}
        _LAST.meta = {"finish": data["choices"][0].get("finish_reason"), "in_tok": u.get("prompt_tokens"),
                      "out_tok": u.get("completion_tokens"), "model_version": data.get("model"),
                      "fingerprint": data.get("system_fingerprint")}
        return data["choices"][0]["message"]["content"]
    if "error" in data:
        err = data["error"]
        if "rate" in str(err).lower() or "429" in str(err):
            raise ConnectionError("rate_limit")
        raise RuntimeError(str(err.get("message", "")))
    return ""


def call_llm(prompt: str, model_name: str, retries: int = 5, seed=None, temperature: float = 0.0) -> str:
    """Call LLM with unified interface.

    Args:
        prompt: text prompt
        model_name: key from MODEL_CONFIG
        retries: max retry on transient failures
        seed: integer seed for multi-seed reproducibility (where API supports it)
        temperature: sampling temperature (default 0.0 = deterministic protocol)

    Returns:
        raw text response from the model
    """
    cfg = dict(MODEL_CONFIG[model_name])
    # OLLAMA_HOST env override: lets parallel runners pin a model to a specific
    # ollama instance (e.g. for 4-GPU parallel cross-lingual run).
    # Format: "http://127.0.0.1:11437" (no /api path)
    ollama_host_override = os.environ.get("OLLAMA_HOST")
    if ollama_host_override and cfg.get("type") == "ollama":
        # Strip protocol/path, then rebuild
        host = ollama_host_override.replace("http://", "").replace("https://", "").rstrip("/")
        # Preserve the api path from original url (e.g. /api/chat or /api/generate)
        old = cfg["url"]
        api_path = "/api/" + old.rsplit("/api/", 1)[1] if "/api/" in old else "/api/chat"
        cfg["url"] = f"http://{host}{api_path}"
    dispatch = {
        "ollama": _call_ollama,
        "gemini": _call_gemini,
        "openai_compat": _call_openai_compat,
    }
    fn = dispatch[cfg["type"]]

    # Paid APIs: reuse any response already paid for (see _cache_get). Local models are free.
    ckey = None
    if cfg["type"] != "ollama" and os.environ.get("LLM_CACHE", "1") != "0":
        ckey = _cache_key(cfg, prompt, seed, temperature)
        hit = _cache_get(ckey)
        if hit is not None:
            _LAST.meta = dict(hit["meta"], cached=True)
            return hit["text"]

    for attempt in range(retries):
        try:
            _LAST.meta = {}
            result = fn(prompt, cfg, seed=seed, temperature=temperature)
            if ckey and _LAST.meta:  # meta is set only when the API returned a real answer
                _cache_put(ckey, result, _LAST.meta)
            if cfg["delay"]:
                time.sleep(cfg["delay"])
            return result
        except ConnectionError:
            wait = min((attempt + 1) * 10, 60)
            time.sleep(wait)
            continue
        except Exception as e:
            # Billing stops (spend cap, empty prepay balance) do not clear by retrying.
            if any(s in str(e).lower() for s in _BILLING_STOP):
                return "ERROR: " + str(e)
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return "ERROR: " + str(e)
    return "ERROR: max retries"


# ---- paid-response cache (added 2026-11-25) --------------------------------------------------
# A multi-call strategy (S2-S4) whose last call failed used to be rerun from its first call,
# paying again for calls that had already succeeded. Every successful paid response is now kept
# in a local SQLite store keyed by everything that determines the request; a rerun of the same
# request returns the stored answer (marked cached=True in the call metadata) instead of paying
# twice. Store: $LLM_CACHE_DB (default results/llm_cache.sqlite, not committed). LLM_CACHE=0 disables.
_BILLING_STOP = ("spending cap", "spend cap", "prepay", "credit balance", "billing", "insufficient_quota",
                 "insufficient balance", "exceeded your current quota")
_CACHE_LOCK = threading.Lock()


def _cache_key(cfg, prompt, seed, temperature) -> str:
    import hashlib
    import json as _json
    budget = os.environ.get("GEMINI_MAX_OUTPUT", "4096") if cfg["type"] == "gemini" else ""
    raw = _json.dumps([cfg["model"], cfg.get("reasoning_effort"), seed, float(temperature), budget, prompt],
                      ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_conn():
    import sqlite3
    path = os.environ.get("LLM_CACHE_DB") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "llm_cache.sqlite")
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS resp (k TEXT PRIMARY KEY, text TEXT, meta TEXT, ts TEXT)")
    return con


def _cache_get(k):
    import json as _json
    with _CACHE_LOCK:
        con = _cache_conn()
        try:
            row = con.execute("SELECT text, meta FROM resp WHERE k=?", (k,)).fetchone()
        finally:
            con.close()
    return {"text": row[0], "meta": _json.loads(row[1])} if row else None


def _cache_put(k, text, meta):
    import json as _json
    with _CACHE_LOCK:
        con = _cache_conn()
        try:
            con.execute("INSERT OR REPLACE INTO resp VALUES (?,?,?,datetime('now'))",
                        (k, text, _json.dumps(meta, ensure_ascii=False)))
            con.commit()
        finally:
            con.close()


def estimate_cost(model_name: str, input_tokens: int = 500, output_tokens: int = 100) -> float:
    """Estimate API cost in USD for one call."""
    inp, out = COST_PER_M.get(model_name, (0, 0))
    return input_tokens * inp / 1e6 + output_tokens * out / 1e6
