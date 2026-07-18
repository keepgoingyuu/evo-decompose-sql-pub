"""A5: Adaptive Selector with v2.1 New Data (N=5 preview).

Per 04_execution_plan_v2.md §3.2 — 5-fold stratified CV across 4 paradigms
(rule-based / multi-class ML / binary gate / per-strategy rescue), 28 features,
N=5 models on BIRD mod+chal 609 queries.

This is a slimmed/adapted version of scripts/phase3_adaptive_selector.py:
- Loads original-4 from results/thesis_2026_02/phase1+phase2_<model>_<s>.json
- Loads v2.1 Gemma4 + Qwen25C7B from results/q1_2026_04/main/v21_bird_main_*
- Aligns query_id (phase1) ↔ query_idx (v2.1) by BIRD mod+chal subset
- Awaits Arctic-R1 to upgrade to N=6
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

import numpy as np

REPO = REPO
sys.path.insert(0, str(REPO))

ANALYSIS_DIR = REPO / 'results/analysis'
PHASE_DIR = REPO / 'results/thesis_2026_02'
V21_E1_DIR = REPO / 'results/q1_2026_04/main'

# Original 4 models: phase1 baseline + phase2 s1-s4
ORIGINAL_4 = {
    'DeepSeek-V3': ('deepseek', 'phase1_deepseek_baseline.json'),
    'GPT-4.1': ('gpt41', 'phase1_gpt41_baseline.json'),
    'Gemini-3-Flash-Preview': ('gemini', 'phase1_gemini3flash_baseline.json'),
    'Qwen2.5-Coder-7B': ('qwen', 'phase1_qwen_baseline.json'),
}

# v2.1 new models on BIRD
V21_NEW = {
    'Gemma-4-E4B': 'gemma4',
    # 'Qwen2.5-Coder-7B-v21': 'qwen25c7b',  # Same model, prefer phase1+phase2 originals; v21 reruns differ
}


def load_original_4_per_query() -> dict:
    """Returns {model_display_name: {qid: {strategy: correct_bool}}} for mod+chal subset."""
    out = {}
    for display, (raw, p1_name) in ORIGINAL_4.items():
        per_q = {}
        # S0 from phase1 baseline (covers all 1534, filter mod+chal)
        f = PHASE_DIR / p1_name
        if not f.exists():
            print(f"  skip {display}: {p1_name} missing")
            continue
        d = json.load(open(f))
        for r in d['results']:
            if r['difficulty'] not in ('moderate', 'challenging'):
                continue
            qid = r['query_id']
            per_q.setdefault(qid, {})['s0'] = bool(r['correct'])
            per_q[qid]['_meta'] = {
                'db_id': r['db_id'],
                'difficulty': r['difficulty'],
                'question': r['question'],
            }
        # S1-S4 from phase2
        for s in ['s1', 's2', 's3', 's4']:
            f2 = PHASE_DIR / f'phase2_{raw}_{s}.json'
            if not f2.exists():
                continue
            d2 = json.load(open(f2))
            for r in d2['results']:
                qid = r.get('query_id') or r.get('qid')
                if qid in per_q:
                    per_q[qid][s] = bool(r['correct'])
        out[display] = per_q
    return out


def load_v21_e1_per_query() -> dict:
    """Returns {model: {query_idx: {strategy: correct_bool, _meta:...}}}."""
    out = {}
    for display, raw in V21_NEW.items():
        per_q = {}
        for s in ['s0', 's1', 's2', 's3', 's4']:
            f = V21_E1_DIR / f'v21_bird_main_{raw}_{s}_seed42.json'
            if not f.exists():
                continue
            d = json.load(open(f))
            for r in d['results']:
                qidx = r['query_idx']
                per_q.setdefault(qidx, {})[s] = bool(r['correct'])
                if '_meta' not in per_q[qidx]:
                    per_q[qidx]['_meta'] = {
                        'db_id': r['db_id'],
                        'question': r['question'],
                    }
        out[display] = per_q
    return out


def align_by_question(orig: dict, v21: dict) -> dict:
    """Match v21 query_idx ↔ orig query_id via question text equality.

    BIRD dev mod+chal subset is deterministic; v21 query_idx is sequential
    over mod+chal subset. orig phase1 query_id is BIRD dev original index.
    Build mapping question_text -> (orig_qid, v21_qidx).
    """
    # Build question lookups
    orig_q_to_id = {}
    for model, q_data in orig.items():
        for qid, sd in q_data.items():
            q_text = sd.get('_meta', {}).get('question')
            if q_text and q_text not in orig_q_to_id:
                orig_q_to_id[q_text] = qid

    v21_q_to_idx = {}
    for model, q_data in v21.items():
        for qidx, sd in q_data.items():
            q_text = sd.get('_meta', {}).get('question')
            if q_text and q_text not in v21_q_to_idx:
                v21_q_to_idx[q_text] = qidx

    # Common questions (intersection)
    common_q = set(orig_q_to_id.keys()) & set(v21_q_to_idx.keys())
    print(f"  Original 4 unique questions: {len(orig_q_to_id)}")
    print(f"  v2.1 unique questions:       {len(v21_q_to_idx)}")
    print(f"  Common (intersection):       {len(common_q)}")

    # Build unified per-query records keyed by question text
    unified = {}
    for q_text in common_q:
        orig_qid = orig_q_to_id[q_text]
        v21_qidx = v21_q_to_idx[q_text]
        rec = {'question': q_text, 'models': {}}
        # Pull metadata from orig (has difficulty)
        for model in orig:
            if orig_qid in orig[model]:
                rec['models'][model] = {s: orig[model][orig_qid].get(s) for s in ['s0','s1','s2','s3','s4']}
                if '_meta' not in rec:
                    rec['_meta'] = orig[model][orig_qid]['_meta']
        for model in v21:
            if v21_qidx in v21[model]:
                rec['models'][model] = {s: v21[model][v21_qidx].get(s) for s in ['s0','s1','s2','s3','s4']}
        unified[q_text] = rec
    return unified


def evaluate_oracle_and_baselines(unified: dict):
    """Per-model oracle vs s0 vs naive selectors."""
    print(f"\n{'Model':<28}{'S0 acc':>10}{'Oracle':>10}{'Δ rescue':>12}{'#queries':>10}")
    print("-" * 70)
    summary = {}
    for model in sorted({m for rec in unified.values() for m in rec['models']}):
        n = 0
        s0_correct = 0
        oracle_correct = 0
        for rec in unified.values():
            if model not in rec['models']:
                continue
            md = rec['models'][model]
            if md.get('s0') is None:
                continue
            n += 1
            if md['s0']:
                s0_correct += 1
                oracle_correct += 1
            else:
                if any(md.get(s) for s in ['s1','s2','s3','s4']):
                    oracle_correct += 1
        if n == 0:
            continue
        s0_acc = s0_correct / n * 100
        oracle = oracle_correct / n * 100
        rescue = oracle - s0_acc
        summary[model] = {'n': n, 's0_acc': s0_acc, 'oracle': oracle, 'rescue_pp': rescue}
        print(f"{model:<28}{s0_acc:>10.2f}{oracle:>10.2f}{rescue:>12.2f}{n:>10}")
    return summary


def evaluate_simple_selectors(unified: dict):
    """Three baseline selectors: always-S0, S1-fallback, oracle. Pooled across models."""
    pooled_per_pick = Counter()
    pooled_correct_per_pick = Counter()
    n = 0
    s0_total = 0
    rescue_naive_s1 = 0  # if S0 wrong, try S1
    oracle_total = 0
    for rec in unified.values():
        for model, md in rec['models'].items():
            if md.get('s0') is None:
                continue
            n += 1
            s0_correct = bool(md['s0'])
            s1_correct = bool(md.get('s1'))
            if s0_correct:
                s0_total += 1
                rescue_naive_s1 += 1  # don't switch
                oracle_total += 1
            else:
                if s1_correct:
                    rescue_naive_s1 += 1
                if any(md.get(s) for s in ['s1','s2','s3','s4']):
                    oracle_total += 1
    print(f"\nPooled across models (n={n}):")
    print(f"  Always-S0       : {s0_total/n*100:.2f}%  ({s0_total}/{n})")
    print(f"  S0 then S1 fallback: {rescue_naive_s1/n*100:.2f}%  ({rescue_naive_s1}/{n})")
    print(f"  Oracle           : {oracle_total/n*100:.2f}%  ({oracle_total}/{n})")
    return {
        'n_pooled': n,
        'always_s0': s0_total / n * 100,
        's0_then_s1': rescue_naive_s1 / n * 100,
        'oracle': oracle_total / n * 100,
    }


def main():
    print("=" * 70)
    print("A5: Adaptive Selector with v2.1 New Data (N=5 preview)")
    print("=" * 70)
    print("\nLoading original-4 (phase1 + phase2)...")
    orig = load_original_4_per_query()
    print(f"  Loaded models: {list(orig.keys())}")
    print("\nLoading v2.1 E1 new models (Gemma4)...")
    v21 = load_v21_e1_per_query()
    print(f"  Loaded models: {list(v21.keys())}")

    print("\nAligning by question text...")
    unified = align_by_question(orig, v21)

    if not unified:
        print("\nERROR: no common questions found. Likely mod+chal subset selection differs.")
        return

    summary = evaluate_oracle_and_baselines(unified)
    pooled = evaluate_simple_selectors(unified)

    out = {
        'note': 'A5 N=5 preview (Gemma4 added; awaiting Arctic-R1). Selector ML training deferred to N=6 final to avoid double-fitting on incomplete data.',
        'n_unified_questions': len(unified),
        'per_model_oracle': summary,
        'pooled_baselines': pooled,
    }
    out_path = ANALYSIS_DIR / 'n5_preview' / 'a5_adaptive_selector.json'
    json.dump(out, open(out_path, 'w'), indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")


if __name__ == '__main__':
    main()
