"""A11: Embedding-based selector variants (JIIS R1-6 / R2-4).

Reviewers noted the ML selectors use only hand-crafted surface features and
asked whether richer semantic representations change the conclusion. This
script re-runs the strongest selector paradigm (binary S0-failure gate with
fixed S1 fallback) with sentence-embedding features:

  emb_gate_lr     LogisticRegression on 384-d MiniLM question embeddings
  emb_gate_gb     GradientBoosting on the same embeddings
  embhc_gate_lr   embeddings concatenated with the 28 hand-crafted features
  embhc_gate_gb   idem, GB

Everything else is held identical to phase3_v3_n5_selector.py: same 609-query
gold-sql-joined dataset (canonical v21 Qwen run), same query-level 5-fold
StratifiedKFold (seed 42), same anchors, same end-to-end evaluation.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (questions are
English; embeddings computed once per unique question, CPU).

Output: results/analysis/n5_final/embedding_selector_results.json

Usage: venv/bin/python scripts/a11_embedding_selector.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "p3", REPO / "scripts/phase3_v3_n5_selector.py")
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)

OUT = REPO / "results/analysis/n5_final/embedding_selector_results.json"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    print("=" * 70)
    print("A11: Embedding-based gate selectors (all-MiniLM-L6-v2)")
    print("=" * 70)

    orig = p3.load_original_4_per_query()
    v21 = p3.load_v21_e1_per_query()
    unified = p3.align_by_gold_sql(orig, v21)
    models = sorted(set(orig.keys()) | set(v21.keys()))
    records, per_query_meta = p3.build_pooled_dataset(unified, models)
    anchors = p3.compute_anchors(records)
    folds = p3.make_query_level_folds(per_query_meta, n_splits=p3.N_FOLDS, seed=p3.SEED)
    print(f"records={len(records)} queries={len(per_query_meta)} "
          f"anchors: S0={anchors['always_s0_pct']:.2f} "
          f"rule={anchors['s0_then_s1_pct']:.2f} oracle={anchors['oracle_pct']:.2f}")

    # ---- embeddings: one per unique question (full untruncated text) ----
    print(f"\nembedding {len(per_query_meta)} unique questions with {EMB_MODEL} ...")
    q_texts = {}
    for r in records:
        q_texts.setdefault(r["q_idx"], r["q_text"])
    st = SentenceTransformer(EMB_MODEL)
    order = sorted(q_texts)
    embs = st.encode([q_texts[i] for i in order],
                     batch_size=64, show_progress_bar=False,
                     normalize_embeddings=True)
    emb_by_q = {qi: embs[k] for k, qi in enumerate(order)}
    dim = embs.shape[1]
    print(f"done: dim={dim}, {time.time()-t0:.0f}s elapsed")

    def with_features(feats_fn):
        out = []
        for r in records:
            r2 = dict(r)
            r2["features"] = feats_fn(r)
            out.append(r2)
        return out

    def emb_only(r):
        return {f"e{k:03d}": float(v) for k, v in enumerate(emb_by_q[r["q_idx"]])}

    def emb_plus_hc(r):
        f = emb_only(r)
        f.update({f"hc_{k}": v for k, v in r["features"].items()})
        return f

    variants = {
        "emb_gate_lr":   (emb_only,    LogisticRegression(max_iter=2000)),
        "emb_gate_gb":   (emb_only,    GradientBoostingClassifier(
                              n_estimators=50, max_depth=3, min_samples_leaf=5,
                              random_state=p3.SEED)),
        "embhc_gate_lr": (emb_plus_hc, LogisticRegression(max_iter=2000)),
        "embhc_gate_gb": (emb_plus_hc, GradientBoostingClassifier(
                              n_estimators=50, max_depth=3, min_samples_leaf=5,
                              random_state=p3.SEED)),
    }

    results = {}
    for name, (fn, clf) in variants.items():
        recs = with_features(fn)
        t1 = time.time()
        res = p3.train_binary_gate(recs, folds, name, clf, fallback="s1")
        results[name] = res
        gm = res["gate_metrics"]
        print(f"{name:<16} end-to-end {res['end_to_end_acc_pct']:.2f}% "
              f"(Δ {res['rescue_pp']:+.2f}pp, rescue "
              f"{res['rescue_pp']/anchors['rescue_ceiling_pp']*100:.1f}%) | "
              f"AUROC={gm['auroc']:.3f} AUPRC={gm['auprc']:.3f} | {time.time()-t1:.0f}s")

    OUT.write_text(json.dumps({
        "experiment_id": "a11",
        "name": "embedding_gate_selectors",
        "embedding_model": EMB_MODEL,
        "embedding_dim": int(dim),
        "seed": p3.SEED,
        "anchors": anchors,
        "selectors": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
