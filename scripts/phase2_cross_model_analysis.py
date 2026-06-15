"""Phase 2 Cross-Model Analysis: Compare S0-S4 across all models.

Produces:
  1. Accuracy table: S0-S4 × models
  2. Per-difficulty breakdown
  3. Per-database strategy recommendations
  4. Oracle potential per model
  5. Error propagation comparison
  6. Strategy complementarity heatmap

Auto-detects available results — run anytime.

Usage:
    python scripts/phase2_cross_model_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.schema_utils import BIRD_DIR

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS = ["deepseek", "gemini", "gpt41", "qwen"]
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]

# Model display names
MODEL_NAMES = {
    "deepseek": "DeepSeek-V3",
    "gemini": "Gemini 2.0 Flash",
    "gpt41": "GPT-4.1",
    "qwen": "Qwen2.5-Coder-7B",
}

# ============================================================
# 1. Load data
# ============================================================

def load_bird():
    """Load BIRD dev set, return mod+chal indexed by query_id."""
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)
    return {i: q for i, q in enumerate(bird)
            if q["difficulty"] in ("moderate", "challenging")}


def find_result_file(model, strategy):
    """Find result file for a model+strategy combo."""
    if strategy == "s0":
        # Phase 1 baseline
        candidates = [
            f"phase1_{model}_baseline.json",
            f"phase1_{model}3flash_baseline.json",  # gemini variant
        ]
    else:
        candidates = [
            f"phase2_{model}_{strategy}.json",
        ]
    for name in candidates:
        path = RESULTS_DIR / name
        if path.exists():
            return path
    return None


def load_all_results():
    """Load all available model×strategy results."""
    bird = load_bird()
    mod_chal_ids = set(bird.keys())

    all_results = {}  # (model, strategy) -> {query_id: result}
    available = defaultdict(list)  # model -> [strategies]

    for model in MODELS:
        for strat in STRATEGIES:
            path = find_result_file(model, strat)
            if path is None:
                continue
            with open(path) as f:
                data = json.load(f)

            results = {}
            for r in data["results"]:
                qid = r["query_id"]
                if qid in mod_chal_ids:
                    results[qid] = r
            all_results[(model, strat)] = results
            available[model].append(strat)

    return bird, all_results, dict(available)


# ============================================================
# 2. Accuracy table
# ============================================================

def accuracy_table(bird, all_results, available):
    """Print S0-S4 accuracy for each model."""
    print("=" * 70)
    print("ACCURACY TABLE: S0-S4 × Models (mod+chal, 609 queries)")
    print("=" * 70)

    header = f"{'Model':<22}" + "".join(f"{'S'+str(i):>8}" for i in range(5)) + "  | Best  Δ vs S0"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        if model not in available:
            continue
        row = f"{MODEL_NAMES[model]:<22}"
        accs = {}
        for strat in STRATEGIES:
            if (model, strat) in all_results:
                results = all_results[(model, strat)]
                correct = sum(1 for r in results.values() if r.get("correct"))
                total = len(results)
                acc = correct / total * 100 if total else 0
                accs[strat] = acc
                row += f"{acc:>7.1f}%"
            else:
                row += f"{'—':>8}"

        if accs:
            s0_acc = accs.get("s0", 0)
            best_strat = max(accs, key=lambda k: accs[k])
            best_acc = accs[best_strat]
            delta = best_acc - s0_acc
            row += f"  | {best_strat.upper():>4} {delta:>+5.1f}pp"
        print(row)
    print()


# ============================================================
# 3. Per-difficulty breakdown
# ============================================================

def difficulty_breakdown(bird, all_results, available):
    """Show accuracy by difficulty level for each model."""
    print("=" * 70)
    print("PER-DIFFICULTY BREAKDOWN")
    print("=" * 70)

    for diff in ["moderate", "challenging"]:
        diff_ids = {qid for qid, q in bird.items() if q["difficulty"] == diff}
        n = len(diff_ids)
        print(f"\n--- {diff.upper()} ({n} queries) ---")

        header = f"{'Model':<22}" + "".join(f"{'S'+str(i):>8}" for i in range(5))
        print(header)
        print("-" * len(header))

        for model in MODELS:
            if model not in available:
                continue
            row = f"{MODEL_NAMES[model]:<22}"
            for strat in STRATEGIES:
                if (model, strat) in all_results:
                    results = all_results[(model, strat)]
                    correct = sum(1 for qid in diff_ids
                                  if qid in results and results[qid].get("correct"))
                    acc = correct / n * 100 if n else 0
                    row += f"{acc:>7.1f}%"
                else:
                    row += f"{'—':>8}"
            print(row)
    print()


# ============================================================
# 4. Oracle analysis per model
# ============================================================

def oracle_analysis(bird, all_results, available):
    """For each model, compute best-per-query (oracle) upper bound."""
    print("=" * 70)
    print("ORACLE ANALYSIS: Best-per-query upper bound")
    print("=" * 70)

    header = f"{'Model':<22} {'S0':>8} {'Oracle':>8} {'Δ':>8} {'Gain%':>8}  Best-strategy distribution"
    print(header)
    print("-" * 90)

    for model in MODELS:
        if model not in available or "s0" not in available.get(model, []):
            continue

        strats_avail = [s for s in STRATEGIES if (model, s) in all_results]
        if len(strats_avail) < 2:
            continue

        # Get common query IDs
        common_ids = set(bird.keys())
        for s in strats_avail:
            common_ids &= set(all_results[(model, s)].keys())

        s0_correct = sum(1 for qid in common_ids
                        if all_results[(model, "s0")][qid].get("correct"))
        s0_acc = s0_correct / len(common_ids) * 100

        oracle_correct = 0
        best_counts = defaultdict(int)
        for qid in common_ids:
            best = None
            for s in strats_avail:
                if all_results[(model, s)][qid].get("correct"):
                    best = s
                    break
            if best is None:
                # None got it right — check if any did
                for s in strats_avail:
                    if all_results[(model, s)][qid].get("correct"):
                        best = s
                        break
            # Oracle: correct if ANY strategy got it right
            any_correct = any(all_results[(model, s)][qid].get("correct")
                             for s in strats_avail)
            if any_correct:
                oracle_correct += 1
                # Which strategy(ies) got it right?
                for s in strats_avail:
                    if all_results[(model, s)][qid].get("correct"):
                        best_counts[s] += 1

        oracle_acc = oracle_correct / len(common_ids) * 100
        delta = oracle_acc - s0_acc

        dist = " ".join(f"{s.upper()}={best_counts.get(s,0)}"
                       for s in strats_avail)
        print(f"{MODEL_NAMES[model]:<22} {s0_acc:>7.1f}% {oracle_acc:>7.1f}% {delta:>+7.1f}pp"
              f" {delta/s0_acc*100 if s0_acc else 0:>7.1f}%  {dist}")

    print()


# ============================================================
# 5. Error propagation comparison
# ============================================================

def error_propagation(bird, all_results, available):
    """Compare error propagation rates across models."""
    print("=" * 70)
    print("ERROR PROPAGATION: S0-correct → Si-wrong rate")
    print("=" * 70)

    header = f"{'Model':<22}" + "".join(f"{'S'+str(i):>8}" for i in range(1, 5))
    print(header)
    print("-" * len(header))

    for model in MODELS:
        if model not in available or "s0" not in available.get(model, []):
            continue

        s0_results = all_results.get((model, "s0"), {})
        s0_correct_ids = {qid for qid, r in s0_results.items() if r.get("correct")}

        if not s0_correct_ids:
            continue

        row = f"{MODEL_NAMES[model]:<22}"
        for strat in ["s1", "s2", "s3", "s4"]:
            if (model, strat) in all_results:
                si_results = all_results[(model, strat)]
                lost = sum(1 for qid in s0_correct_ids
                          if qid in si_results and not si_results[qid].get("correct"))
                rate = lost / len(s0_correct_ids) * 100
                row += f"{rate:>7.1f}%"
            else:
                row += f"{'—':>8}"
        print(row)

    print()
    print("(Lower = safer; the strategy doesn't break what S0 already solves)")
    print()


# ============================================================
# 6. Rescue analysis comparison
# ============================================================

def rescue_analysis(bird, all_results, available):
    """Compare rescue rates across models: S0-wrong → Si-correct."""
    print("=" * 70)
    print("RESCUE ANALYSIS: S0-wrong → Si-correct rate")
    print("=" * 70)

    header = f"{'Model':<22}" + "".join(f"{'S'+str(i):>8}" for i in range(1, 5)) + f"{'ANY':>8}"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        if model not in available or "s0" not in available.get(model, []):
            continue

        s0_results = all_results.get((model, "s0"), {})
        s0_wrong_ids = {qid for qid, r in s0_results.items() if not r.get("correct")}

        if not s0_wrong_ids:
            continue

        row = f"{MODEL_NAMES[model]:<22}"
        any_rescued = set()
        for strat in ["s1", "s2", "s3", "s4"]:
            if (model, strat) in all_results:
                si_results = all_results[(model, strat)]
                rescued = {qid for qid in s0_wrong_ids
                          if qid in si_results and si_results[qid].get("correct")}
                any_rescued |= rescued
                rate = len(rescued) / len(s0_wrong_ids) * 100
                row += f"{rate:>7.1f}%"
            else:
                row += f"{'—':>8}"

        any_rate = len(any_rescued) / len(s0_wrong_ids) * 100
        row += f"{any_rate:>7.1f}%"
        print(row)

    print()
    print("(Higher = more queries recovered; ANY = at least one Si rescues)")
    print()


# ============================================================
# 7. Per-database top strategies
# ============================================================

def per_database_analysis(bird, all_results, available):
    """Show best strategy per database for each model."""
    print("=" * 70)
    print("PER-DATABASE BEST STRATEGIES (top 10 databases with most queries)")
    print("=" * 70)

    # Group queries by database
    db_queries = defaultdict(set)
    for qid, q in bird.items():
        db_queries[q["db_id"]].add(qid)

    # Sort by count, take top 10
    top_dbs = sorted(db_queries.items(), key=lambda x: -len(x[1]))[:10]

    for model in MODELS:
        if model not in available or len(available.get(model, [])) < 2:
            continue

        strats_avail = [s for s in STRATEGIES if (model, s) in all_results]

        print(f"\n--- {MODEL_NAMES[model]} ---")
        header = f"{'Database':<30} {'N':>3}" + "".join(f"{'S'+s[1]:>7}" for s in strats_avail) + "  Best"
        print(header)
        print("-" * len(header))

        for db_id, qids in top_dbs:
            row = f"{db_id:<30} {len(qids):>3}"
            best_strat = None
            best_acc = -1
            for strat in strats_avail:
                results = all_results[(model, strat)]
                correct = sum(1 for qid in qids
                            if qid in results and results[qid].get("correct"))
                acc = correct / len(qids) * 100
                row += f"{acc:>6.0f}%"
                if acc > best_acc:
                    best_acc = acc
                    best_strat = strat
            s0_results = all_results[(model, "s0")]
            s0_acc = sum(1 for qid in qids if qid in s0_results and s0_results[qid].get("correct")) / len(qids) * 100
            delta = best_acc - s0_acc
            if delta > 0:
                row += f"  {best_strat.upper()} (+{delta:.0f}pp)"
            else:
                row += f"  S0 (best)"
            print(row)
    print()


# ============================================================
# 8. Cross-model agreement
# ============================================================

def cross_model_agreement(bird, all_results, available):
    """Show how often models agree on S0 performance."""
    models_with_s0 = [m for m in MODELS if "s0" in available.get(m, [])]
    if len(models_with_s0) < 2:
        return

    print("=" * 70)
    print(f"CROSS-MODEL S0 AGREEMENT ({len(models_with_s0)} models)")
    print("=" * 70)

    common_ids = set(bird.keys())
    for m in models_with_s0:
        common_ids &= set(all_results[(m, "s0")].keys())

    all_correct = 0
    all_wrong = 0
    mixed = 0
    for qid in common_ids:
        correct_count = sum(1 for m in models_with_s0
                           if all_results[(m, "s0")][qid].get("correct"))
        if correct_count == len(models_with_s0):
            all_correct += 1
        elif correct_count == 0:
            all_wrong += 1
        else:
            mixed += 1

    total = len(common_ids)
    print(f"All correct:  {all_correct:>4} ({all_correct/total*100:.1f}%)")
    print(f"All wrong:    {all_wrong:>4} ({all_wrong/total*100:.1f}%)")
    print(f"Mixed:        {mixed:>4} ({mixed/total*100:.1f}%)")

    # Pairwise agreement
    print(f"\nPairwise S0 agreement rates:")
    header = f"{'':>22}" + "".join(f"{MODEL_NAMES[m][:8]:>10}" for m in models_with_s0)
    print(header)
    for m1 in models_with_s0:
        row = f"{MODEL_NAMES[m1]:<22}"
        for m2 in models_with_s0:
            agree = sum(1 for qid in common_ids
                       if all_results[(m1, "s0")][qid].get("correct") ==
                          all_results[(m2, "s0")][qid].get("correct"))
            rate = agree / total * 100
            row += f"{rate:>9.1f}%"
        print(row)
    print()


# ============================================================
# 9. Summary + save
# ============================================================

def generate_summary(bird, all_results, available):
    """Generate JSON summary for later use."""
    summary = {
        "models_available": {m: available.get(m, []) for m in MODELS},
        "total_queries": len(bird),
        "model_accuracies": {},
    }

    for model in MODELS:
        if model not in available:
            continue
        model_data = {}
        for strat in STRATEGIES:
            if (model, strat) in all_results:
                results = all_results[(model, strat)]
                correct = sum(1 for r in results.values() if r.get("correct"))
                model_data[strat] = {
                    "correct": correct,
                    "total": len(results),
                    "accuracy": round(correct / len(results) * 100, 2) if results else 0,
                }
        summary["model_accuracies"][model] = model_data

    return summary


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 2 CROSS-MODEL ANALYSIS")
    print("=" * 70)

    bird, all_results, available = load_all_results()

    print(f"\nBIRD mod+chal queries: {len(bird)}")
    print("\nAvailable results:")
    for model in MODELS:
        if model in available:
            strats = ", ".join(s.upper() for s in available[model])
            print(f"  {MODEL_NAMES[model]:<22}: {strats}")
        else:
            print(f"  {MODEL_NAMES[model]:<22}: (no results)")
    print()

    # Run all analyses
    accuracy_table(bird, all_results, available)
    difficulty_breakdown(bird, all_results, available)
    oracle_analysis(bird, all_results, available)
    error_propagation(bird, all_results, available)
    rescue_analysis(bird, all_results, available)
    per_database_analysis(bird, all_results, available)
    cross_model_agreement(bird, all_results, available)

    # Save summary
    summary = generate_summary(bird, all_results, available)
    out_path = RESULTS_DIR / "phase2_cross_model_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
