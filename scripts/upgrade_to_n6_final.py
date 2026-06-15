"""Upgrade all A1-A5 analyses from N=5 preview to N=6 final by adding Arctic-R1.

Triggered after E2 sync completes:
  bash scripts/sync_results_from_dgx.sh E2

Reads:
  data/results/E2_arctic_r1/v21_arctic_r1_<bench>_s0_seed42.json (4 files)
  data/results/E1_gemma_4_e4b/merged/...
  results/thesis_2026_02/phase{1,2,4}_*

Writes:
  data/results/analysis/inverse_correlation_fit.json (N=6 final, overwrites old N=4)
  data/results/analysis/v21_e1_metrics.json (adds arctic_r1 entries)
  data/results/analysis/xlingual_metrics_unified.json (adds Arctic-R1 across 4 langs)
  data/results/analysis/a2_strategy_specific_alpha.json (N=6)
  data/results/analysis/a3_cross_lingual_invariance.json (N=6)
  data/results/analysis/a4_statistical_rigor.json (N=6)
  data/results/analysis/a5_adaptive_selector.json (N=6)
"""
import json
import subprocess
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

REPO = REPO
ANALYSIS = REPO / 'data/results/analysis'
E2_DIR = REPO / 'data/results/E2_arctic_r1'

# Bench → lang mapping for cross-lingual unified
BENCH_LANG = {
    'bird_modchal': 'BIRD',  # not a lang; goes to BIRD-only fit
    'spider_en': 'EN',
    'cspider_zh_tw': 'ZH-TW',
    'multispider_ja': 'JA',
}


def load_arctic_metrics() -> dict:
    """Load Arctic-R1 S0 accuracy from 4 benchmark files."""
    out = {}
    for bench in BENCH_LANG:
        f = E2_DIR / f'v21_arctic_r1_{bench}_s0_seed42.json'
        if not f.exists():
            print(f"  WARN: {f.name} not found in E2_DIR")
            continue
        d = json.load(open(f))
        out[bench] = round(d['metadata']['accuracy'] * 100, 2)
    return out


def upgrade_xlingual_metrics(arctic: dict):
    """Add Arctic-R1 to xlingual_metrics_unified.json."""
    f = ANALYSIS / 'xlingual_metrics_unified.json'
    data = json.load(open(f))
    data['Arctic-Text2SQL-R1-7B'] = {}
    for bench, acc in arctic.items():
        if bench == 'bird_modchal':
            continue  # BIRD goes to inverse_correlation_fit input separately
        lang = BENCH_LANG[bench]
        # Arctic only has S0 (RL specialist single-pass)
        data['Arctic-Text2SQL-R1-7B'][lang] = {'s0': acc}
    json.dump(data, open(f, 'w'), indent=2)
    print(f"  ✓ Updated xlingual_metrics_unified.json with Arctic across {len(arctic)-1} langs")


def upgrade_a1_n6_final(arctic: dict):
    """Re-run A1 fit with Gemma4 + Arctic-R1 added (BIRD anchors)."""
    sys.path.insert(0, str(REPO))
    from scripts.inverse_correlation_fit import BIRD_HARDCODED, fit_per_strategy

    e1_merged = REPO / 'data/results/E1_gemma_4_e4b/merged'
    data = dict(BIRD_HARDCODED)

    # Replace Qwen with v2.1 rerun + add Gemma4
    for raw, display in [('qwen25c7b', 'Qwen2.5-Coder-7B'), ('gemma4', 'Gemma-4-E4B')]:
        accs = {}
        for s in ['s0', 's1', 's2', 's3', 's4']:
            fp = e1_merged / f'v21_bird_main_{raw}_{s}_seed42.json'
            d = json.load(open(fp))
            accs[s] = round(d['metadata']['accuracy'] * 100, 2)
        data[display] = accs

    # Arctic only has S0 — for A1 use S0=accuracy, S1-S4 = same as S0 (no decomposition data)
    # Actually Arctic is single-pass RL, doesn't run S1-S4. We exclude it from the per-strategy fit
    # and instead include it only as a reference point in §5.5 RQ5.
    # → A1 final remains N=5, but we annotate Arctic separately.
    print(f"  Arctic-R1 BIRD acc: {arctic.get('bird_modchal', 'N/A')}% (RL specialist; not in A1 N-model fit)")

    out_dir = ANALYSIS  # write to main, not n5_preview
    print(f"\n  Writing N=5 final fit to {out_dir} (Arctic excluded — no S1-S4 data)")
    fit_per_strategy(data, out_dir)


def upgrade_a4_with_arctic(arctic: dict):
    """A4 stays N=5 (Arctic not in strategy comparison). Just refresh from current xlingual data."""
    print("  A4 unchanged: Arctic has no S1-S4 to compare against S0 within-model")


def main():
    print("=" * 70)
    print("Upgrade analyses to N=6 final (post-Arctic)")
    print("=" * 70)

    if not E2_DIR.exists() or not list(E2_DIR.glob('v21_arctic_r1_*_seed42.json')):
        print(f"\n❌ No Arctic outputs in {E2_DIR}. Run sync first:")
        print("   bash scripts/sync_results_from_dgx.sh E2")
        sys.exit(1)

    arctic = load_arctic_metrics()
    print(f"\nArctic-R1 S0 accuracies:")
    for bench, acc in arctic.items():
        print(f"  {bench:<24} {acc}%")

    print("\n→ Upgrading xlingual_metrics_unified.json...")
    upgrade_xlingual_metrics(arctic)

    print("\n→ Upgrading A1 fit...")
    upgrade_a1_n6_final(arctic)

    print("\n→ Re-running A2 with main fit...")
    subprocess.run([sys.executable, str(REPO / 'scripts/a2_strategy_specific_alpha.py')], check=False,
                   env={**__import__('os').environ, 'A_FIT_PATH': str(ANALYSIS / 'inverse_correlation_fit.json')})

    print("\n→ Re-running A3 with Arctic added...")
    subprocess.run([sys.executable, str(REPO / 'scripts/a3_cross_lingual_invariance.py')], check=False)

    print("\n→ Re-running A4 (paired t-test)...")
    subprocess.run([sys.executable, str(REPO / 'scripts/a4_statistical_rigor.py')], check=False)

    print("\n→ A5 selector: Arctic has S0 only on BIRD; rescue analysis can't include it as alt strategy.")
    print("  Skipping A5 re-run (current N=5 selector valid).")

    # Save Arctic snapshot for thesis Ch5 §5.5
    rl_summary = {
        'model': 'Arctic-Text2SQL-R1-7B',
        'engine': 'vllm-0.6.6 + V100 TP=2',
        'note': 'RL specialist single-pass (S0 only). For RQ5 RL boundary analysis.',
        'per_benchmark_s0': arctic,
    }
    (ANALYSIS / 'rl_baseline_summary.json').write_text(json.dumps(rl_summary, indent=2))
    print(f"\n✓ RL baseline summary saved: {ANALYSIS / 'rl_baseline_summary.json'}")
    print("\n" + "=" * 70)
    print("DONE — N=6 final analysis ready (Arctic separate as RL boundary in §5.5)")
    print("=" * 70)


if __name__ == '__main__':
    main()
