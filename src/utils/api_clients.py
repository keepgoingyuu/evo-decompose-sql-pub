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
    "gemini3f": (0.50, 3.00),
    "deepseekv32": (0.27, 1.10),
    "qwen25c7b": (0, 0),
    "gemma4": (0, 0),
    "arctic": (0, 0),
}

_KEY_CACHE = {
    "DEEPSEEK_API_KEY": DEEPSEEK_KEY,
    "OPENAI_API_KEY": OPENAI_KEY,
    "GEMINI_API_KEY": GEMINI_KEY,
}


def _get_key(env_name: str) -> str:
    return _KEY_CACHE.get(env_name) or os.environ.get(env_name, "")


def _call_ollama(prompt, cfg, seed=None):
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    r = requests.post(cfg["url"], json=payload, timeout=cfg["timeout"])
    return r.json().get("message", {}).get("content", "")


def _call_gemini(prompt, cfg, seed=None):
    key = _get_key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    url = cfg["url"] + "?key=" + key
    gen_cfg = {"temperature": 0, "maxOutputTokens": 4096}
    if seed is not None:
        gen_cfg["seed"] = seed
    r = requests.post(url, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }, timeout=cfg["timeout"])
    data = r.json()
    if "candidates" in data:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    return ""


def _call_openai_compat(prompt, cfg, seed=None):
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
        payload["temperature"] = 0.0
    if seed is not None:
        payload["seed"] = seed

    r = requests.post(cfg["url"], headers=headers, json=payload, timeout=cfg["timeout"])
    data = r.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    if "error" in data:
        err = data["error"]
        if "rate" in str(err).lower() or "429" in str(err):
            raise ConnectionError("rate_limit")
        raise RuntimeError(str(err.get("message", "")))
    return ""


def call_llm(prompt: str, model_name: str, retries: int = 5, seed=None) -> str:
    """Call LLM with unified interface.

    Args:
        prompt: text prompt
        model_name: key from MODEL_CONFIG
        retries: max retry on transient failures
        seed: integer seed for multi-seed reproducibility (where API supports it)

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

    for attempt in range(retries):
        try:
            result = fn(prompt, cfg, seed=seed)
            if cfg["delay"]:
                time.sleep(cfg["delay"])
            return result
        except ConnectionError:
            wait = min((attempt + 1) * 10, 60)
            time.sleep(wait)
            continue
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return "ERROR: " + str(e)
    return "ERROR: max retries"


def estimate_cost(model_name: str, input_tokens: int = 500, output_tokens: int = 100) -> float:
    """Estimate API cost in USD for one call."""
    inp, out = COST_PER_M.get(model_name, (0, 0))
    return input_tokens * inp / 1e6 + output_tokens * out / 1e6
