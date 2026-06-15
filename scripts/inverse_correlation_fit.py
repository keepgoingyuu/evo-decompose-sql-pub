#!/usr/bin/env python3
"""
Inverse Correlation Law fit (Headline contribution #1)

Fit: Δ_S_i(M) = -α_S_i · log(C_M) + β_S_i + ε
where C_M = S0 baseline accuracy of model M
      Δ_S_i(M) = EX_S_i(M) - C_M

Reports α, β, R² with 95% bootstrap CI (10000 resamples).

Usage:
    python scripts/inverse_correlation_fit.py [--data results/phase2_cross_model_summary.json] [--out data/results/analysis/]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats


def fit_log_linear(c_values: np.ndarray, delta_values: np.ndarray) -> dict:
    """
    Fit Δ = -α · log(C) + β + ε via OLS.
    Returns alpha, beta, r_squared, p_value, residuals.
    """
    log_c = np.log(c_values)
    # OLS: Δ = a + b · log(C)  →  α = -b, β = a
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_c, delta_values)
    return {
        "alpha": -slope,
        "beta": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "slope_std_err": std_err,
        "n": len(c_values),
    }


def bootstrap_ci(
    c_values: np.ndarray,
    delta_values: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for alpha, beta, r_squared via resampling pairs with replacement."""
    rng = np.random.default_rng(rng_seed)
    n = len(c_values)
    alphas, betas, r2s = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        c_b = c_values[idx]
        d_b = delta_values[idx]
        if len(np.unique(c_b)) < 2:
            continue
        try:
            log_c = np.log(c_b)
            slope, intercept, r_val, _, _ = stats.linregress(log_c, d_b)
            alphas.append(-slope)
            betas.append(intercept)
            r2s.append(r_val**2)
        except Exception:
            continue
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return {
        "alpha_ci": (float(np.percentile(alphas, lo)), float(np.percentile(alphas, hi))),
        "beta_ci": (float(np.percentile(betas, lo)), float(np.percentile(betas, hi))),
        "r_squared_ci": (float(np.percentile(r2s, lo)), float(np.percentile(r2s, hi))),
        "n_valid_resamples": len(alphas),
    }


def load_data_from_summary(path: str) -> dict[str, dict[str, float]]:
    """
    Load model accuracies from phase2_cross_model_summary.json.
    Returns {model_name: {s0: acc, s1: acc, ..., s4: acc}}.
    Skips invalidated models.
    """
    with open(path) as f:
        d = json.load(f)
    out = {}
    for model, strategies in d.get("model_accuracies", {}).items():
        if "_note" in strategies and "INVALIDATED" in strategies["_note"]:
            print(f"  [skip] {model}: {strategies['_note'][:60]}")
            continue
        out[model] = {s: v["accuracy"] for s, v in strategies.items() if not s.startswith("_")}
    return out


# Hardcoded fallback — from thesis Ch5 Table 5.1 (BIRD mod+chal)
BIRD_HARDCODED = {
    "DeepSeek-V3": {"s0": 40.07, "s1": 39.41, "s2": 34.98, "s3": 37.6, "s4": 37.27},
    "GPT-4.1": {"s0": 38.59, "s1": 41.05, "s2": 33.99, "s3": 38.75, "s4": 39.57},
    "Qwen2.5-Coder-7B": {"s0": 27.59, "s1": 32.51, "s2": 30.05, "s3": 27.09, "s4": 28.24},
    "Gemini-3-Flash-Preview": {"s0": 33.0, "s1": 52.9, "s2": 49.3, "s3": 46.1, "s4": 51.2},
}


def fit_per_strategy(model_data: dict[str, dict[str, float]], output_dir: Path) -> dict:
    """Fit log-linear law independently for each strategy S1-S4."""
    models = list(model_data.keys())
    c = np.array([model_data[m]["s0"] for m in models])

    results = {"models": models, "s0_capability": c.tolist(), "per_strategy": {}}

    print(f"\n{'=' * 70}")
    print(f"Inverse Correlation Law Fit (N={len(models)} models)")
    print(f"{'=' * 70}")
    print(f"Models (sorted by S0 capability): ")
    sorted_idx = np.argsort(c)
    for i in sorted_idx:
        print(f"  {models[i]:<35} S0={c[i]:.2f}%")

    print(f"\n{'Strategy':<10}{'α':>10}{'β':>10}{'R²':>10}{'p':>10}{'95% CI on R²':>25}")
    print("-" * 70)

    for strategy in ["s1", "s2", "s3", "s4"]:
        delta = np.array([model_data[m][strategy] - model_data[m]["s0"] for m in models])
        fit = fit_log_linear(c, delta)
        ci = bootstrap_ci(c, delta, n_resamples=10000)
        results["per_strategy"][strategy] = {
            "delta_per_model": delta.tolist(),
            **fit,
            **ci,
        }
        ci_low, ci_high = ci["r_squared_ci"]
        print(
            f"{strategy.upper():<10}"
            f"{fit['alpha']:>10.3f}"
            f"{fit['beta']:>10.3f}"
            f"{fit['r_squared']:>10.3f}"
            f"{fit['p_value']:>10.4f}"
            f"  [{ci_low:.3f}, {ci_high:.3f}]"
        )

    # Universal law check: are all alpha values positive (inverse correlation holds)?
    alphas = [results["per_strategy"][s]["alpha"] for s in ["s1", "s2", "s3", "s4"]]
    r2s = [results["per_strategy"][s]["r_squared"] for s in ["s1", "s2", "s3", "s4"]]
    all_inverse = all(a > 0 for a in alphas)
    all_high_r2 = all(r > 0.6 for r in r2s)
    results["universality"] = {
        "all_inverse": bool(all_inverse),
        "all_r2_above_0.6": bool(all_high_r2),
        "alphas": alphas,
        "r2s": r2s,
    }

    print(f"\n{'-' * 70}")
    print(f"Universality check:")
    print(f"  All α > 0 (inverse correlation): {'✓' if all_inverse else '✗'}")
    print(f"  All R² > 0.6 (universality threshold): {'✓' if all_high_r2 else '✗'}")
    print(f"{'=' * 70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "inverse_correlation_fit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to phase2 summary JSON")
    parser.add_argument(
        "--out",
        default="data/results/analysis",
        help="Output directory (relative to repo root)",
    )
    parser.add_argument(
        "--use-hardcoded",
        action="store_true",
        help="Use hardcoded BIRD numbers from Ch5 Table 5.1 (when summary file is incomplete)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.out

    if args.use_hardcoded or args.data is None:
        print("Using hardcoded BIRD mod+chal numbers from Ch5 Table 5.1")
        data = BIRD_HARDCODED
    else:
        data_path = repo_root / args.data
        print(f"Loading from {data_path}")
        data = load_data_from_summary(str(data_path))
        # Augment with Gemini hardcoded if missing
        if "gemini" not in {k.lower() for k in data}:
            print("  Gemini missing in summary — adding from hardcoded Ch5 Table 5.1")
            data["Gemini-3-Flash-Preview"] = BIRD_HARDCODED["Gemini-3-Flash-Preview"]

    fit_per_strategy(data, output_dir)


if __name__ == "__main__":
    main()
