"""V2.1 BIRD main run, PARALLEL: 2 models × 5 strategies × 609 BIRD mod+chal × seed 42.

Two Python processes run concurrently:
  Process A: qwen25c7b -> http://localhost:11434  (GPU 0)
  Process B: gemma4    -> http://localhost:11435  (GPU 1, gemma3n:e4b)

api_clients.MODEL_CONFIG already maps each model to its dedicated Ollama port,
so we only need to fork two processes (one per model).

Requires: scripts/start_ollama_multi.sh already started.
Output: results/q1_2026_04/main/v21_bird_main_<model>_<strategy>_seed42.json
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = ["qwen25c7b", "gemma4"]
STRATEGIES = ["s0", "s1", "s2", "s3", "s4"]


def run_model_all_strategies(model: str) -> dict:
    """Worker: run one model × all 5 strategies sequentially.

    api_clients.MODEL_CONFIG has port routing baked in
    (qwen25c7b -> 11434, gemma4 -> 11435), so no env tricks needed.
    """
    from scripts.run_v21_bird_main import run_one, load_bird_full_modchal

    queries = load_bird_full_modchal()
    out = {}
    for strategy in STRATEGIES:
        print(f"[{model}] >>> {strategy} ({len(queries)}q) <<<", flush=True)
        try:
            r = run_one(model, strategy, queries)
            out[strategy] = r["metadata"]["accuracy"]
            print(f"[{model}] {strategy} acc={out[strategy]*100:.2f}%", flush=True)
        except Exception as e:
            print(f"[{model}] {strategy} ERROR: {e}", flush=True)
            out[strategy] = None
    return {"model": model, "results": out}


def main():
    print(f"=== V2.1 BIRD MAIN (PARALLEL) ===")
    print(f"Models: {MODELS}")

    grand_t0 = time.time()
    summary = {}
    with ProcessPoolExecutor(max_workers=len(MODELS)) as ex:
        futures = {ex.submit(run_model_all_strategies, m): m for m in MODELS}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                r = fut.result()
                summary[model] = r["results"]
                print(f"\n=== {model} FINISHED ===", flush=True)
            except Exception as e:
                print(f"\n=== {model} CRASHED: {e} ===", flush=True)

    elapsed = time.time() - grand_t0
    print(f"\n=== ALL DONE in {elapsed/3600:.2f} hours ===")
    print("=== ACCURACY MATRIX ===")
    for model in MODELS:
        if model not in summary:
            continue
        line = " ".join(
            f"{s}={(a*100):.1f}" if a is not None else f"{s}=X"
            for s, a in summary[model].items()
        )
        print(f"  {model}: {line}")


if __name__ == "__main__":
    main()
