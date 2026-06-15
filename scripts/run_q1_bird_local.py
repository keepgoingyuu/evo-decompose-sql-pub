"""Q1-path BIRD main exp (LOCAL ONLY): 2 models × 5 strategies × 609 queries × seed=42.

Resumable. Per-query checkpoint every 25 queries.
Qwen first (faster, ~11h), then Gemma 4 (~33h).
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-use pilot's run_one
from scripts.run_q1_pilot_local import run_one, BIRD_DIR, RESULTS_DIR, SEED


def load_bird_full_modchal():
    """Load all 609 BIRD moderate+challenging queries."""
    full = json.load(open(BIRD_DIR / "dev.json"))
    return [q for q in full if q.get("difficulty") in ("moderate", "challenging")]


def main():
    print(f"=== Q1 BIRD MAIN (LOCAL ONLY, seed={SEED}) ===")
    queries = load_bird_full_modchal()
    print(f"Loaded {len(queries)} BIRD moderate+challenging queries")

    # Run order: faster model first to get partial signal early
    models = ["qwen25c7b", "gemma4"]
    strategies = ["s0", "s1", "s2", "s3", "s4"]

    summary = {}
    grand_t0 = time.time()
    for model in models:
        print(f"\n========== MODEL: {model} ==========")
        for strategy in strategies:
            print(f"\n>>> {model} × {strategy}  ({len(queries)} queries) <<<")
            t0 = time.time()
            try:
                # Override output filename to distinguish from pilot
                # We monkey-patch by using pilot's run_one but write to different path
                r = run_one(model, strategy, queries)
                summary[f"{model}_{strategy}"] = r["metadata"]["accuracy"]
                # Move the output file from pilot naming to main naming
                old_path = RESULTS_DIR / f"q1_pilot_local_{model}_{strategy}_seed{SEED}.json"
                new_path = RESULTS_DIR / f"q1_bird_local_{model}_{strategy}_seed{SEED}.json"
                if old_path.exists() and not new_path.exists():
                    old_path.rename(new_path)
            except Exception as e:
                print(f"  ERROR in {model}/{strategy}: {e}")
                summary[f"{model}_{strategy}"] = None

            # Save running summary
            running_summary_path = RESULTS_DIR / f"q1_bird_local_running_summary_seed{SEED}.json"
            with open(running_summary_path, "w") as f:
                json.dump({
                    "completed": summary,
                    "current_model": model, "current_strategy": strategy,
                    "elapsed_total_min": round((time.time() - grand_t0) / 60, 2),
                }, f, indent=2, ensure_ascii=False)

    grand_elapsed = time.time() - grand_t0
    print(f"\n=== BIRD MAIN COMPLETED in {grand_elapsed/3600:.1f} hours ===")
    print("=== ACCURACY MATRIX ===")
    for model in models:
        row = " ".join(f"{strategy}={summary.get(f'{model}_{strategy}', '?')*100 if summary.get(f'{model}_{strategy}') else 'X':.1f}" for strategy in strategies)
        print(f"  {model}: {row}")


if __name__ == "__main__":
    main()
