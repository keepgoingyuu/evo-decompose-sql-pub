"""Phase 3 v3: N=5 Panel Adaptive Selector with Full ML Evaluation.

Implements the plan in scripts/n5_selector_plan.md.

Key differences from phase3_adaptive_selector.py (single-model exploratory):
  - N=5 panel (DeepSeek-V3 / GPT-4.1 / Gemini Flash / Qwen2.5-Coder-7B / Gemma-4-E4B)
  - 5-fold StratifiedKFold (was KFold) — matches paper §4.3 stratification claim
  - Query-level fold splits (prevents leakage from same query × different model)
  - Full metric family: Accuracy / Precision / Recall / F1 / AUROC / AUPRC /
    Confusion Matrix / Wilson 95% CI / Feature importance
  - 11 selectors evaluated: 2 rule-based + 3 multi-class + 3 binary gate + 3 cascade

Arctic-R1 (RL specialist) is intentionally excluded from the selector panel: it
is single-pass and does not produce S1–S4 outputs, so the selector decision space
is undefined for it (paper §5.5 line 266).

Usage:
    python scripts/phase3_v3_n5_selector.py
    python scripts/phase3_v3_n5_selector.py --dry-run   # 1 fold, 1 selector
    python scripts/phase3_v3_n5_selector.py --output-dir <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.schema_utils import DB_DIR, BIRD_DIR  # noqa: E402

# ============================================================
# Paths
# ============================================================
ANALYSIS_DIR = REPO / "results/analysis"
PHASE_DIR = REPO / "results/thesis_2026_02"

# Gemma-4 v21 results location: try multiple paths since DGX (pre-reorg) and
# Mac (post-reorg) have different layouts. First match wins.
V21_E1_CANDIDATES = [
    REPO / "data/results/E1_gemma_4_e4b/merged",   # Mac after reorg
    REPO / "results/q1_2026_04/main",              # DGX original
]
V21_E1_DIR = next((p for p in V21_E1_CANDIDATES if p.exists()), V21_E1_CANDIDATES[0])

DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "n5_final"

# Original 4 models: phase1 baseline + phase2 s1-s4
ORIGINAL_4 = {
    "DeepSeek-V3":             ("deepseek", "phase1_deepseek_baseline.json"),
    "GPT-4.1":                 ("gpt41",    "phase1_gpt41_baseline.json"),
    "Gemini-3-Flash-Preview":  ("gemini",   "phase1_gemini3flash_baseline.json"),
    "Qwen2.5-Coder-7B":        ("qwen",     "phase1_qwen_baseline.json"),
}

# v2.1 new models on BIRD
V21_NEW = {
    "Gemma-4-E4B": "gemma4",
}

SEED = 42
N_FOLDS = 5

# ============================================================
# 1. Data loading (adapted from a5_adaptive_selector_v21.py)
# ============================================================

def load_original_4_per_query() -> dict:
    """Returns {model_name: {qid: {strategy: correct_bool, _meta: {...}}}}."""
    out = {}
    for display, (raw, p1_name) in ORIGINAL_4.items():
        per_q = {}
        f = PHASE_DIR / p1_name
        if not f.exists():
            print(f"  skip {display}: {p1_name} missing")
            continue
        d = json.load(open(f))
        for r in d["results"]:
            if r["difficulty"] not in ("moderate", "challenging"):
                continue
            qid = r["query_id"]
            per_q.setdefault(qid, {})["s0"] = bool(r["correct"])
            per_q[qid]["_meta"] = {
                "db_id": r["db_id"],
                "difficulty": r["difficulty"],
                "question": r["question"],
                "evidence": r.get("evidence", ""),
            }
        for s in ["s1", "s2", "s3", "s4"]:
            f2 = PHASE_DIR / f"phase2_{raw}_{s}.json"
            if not f2.exists():
                continue
            d2 = json.load(open(f2))
            for r in d2["results"]:
                qid = r.get("query_id") or r.get("qid")
                if qid in per_q:
                    per_q[qid][s] = bool(r["correct"])
        out[display] = per_q
    return out


def load_v21_e1_per_query() -> dict:
    """Returns {model: {query_idx: {strategy: correct_bool, _meta: {...}}}}."""
    out = {}
    for display, raw in V21_NEW.items():
        per_q = {}
        for s in ["s0", "s1", "s2", "s3", "s4"]:
            f = V21_E1_DIR / f"v21_bird_main_{raw}_{s}_seed42.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            for r in d["results"]:
                qidx = r["query_idx"]
                per_q.setdefault(qidx, {})[s] = bool(r["correct"])
                if "_meta" not in per_q[qidx]:
                    per_q[qidx]["_meta"] = {
                        "db_id": r["db_id"],
                        "question": r["question"],
                        "evidence": r.get("evidence", ""),
                    }
        out[display] = per_q
    return out


def align_by_question(orig: dict, v21: dict) -> dict:
    """Match by question text. Returns {q_text: {models: {model: {s0..s4}}, _meta: {...}}}."""
    orig_q_to_id = {}
    for model, q_data in orig.items():
        for qid, sd in q_data.items():
            q_text = sd.get("_meta", {}).get("question")
            if q_text and q_text not in orig_q_to_id:
                orig_q_to_id[q_text] = qid

    v21_q_to_idx = {}
    for model, q_data in v21.items():
        for qidx, sd in q_data.items():
            q_text = sd.get("_meta", {}).get("question")
            if q_text and q_text not in v21_q_to_idx:
                v21_q_to_idx[q_text] = qidx

    common_q = set(orig_q_to_id.keys()) & set(v21_q_to_idx.keys())
    print(f"  Original 4 unique questions: {len(orig_q_to_id)}")
    print(f"  v2.1 unique questions:       {len(v21_q_to_idx)}")
    print(f"  Common (intersection):       {len(common_q)}")

    unified = {}
    for q_text in common_q:
        orig_qid = orig_q_to_id[q_text]
        v21_qidx = v21_q_to_idx[q_text]
        rec = {"question": q_text, "models": {}}
        for model in orig:
            if orig_qid in orig[model]:
                rec["models"][model] = {
                    s: orig[model][orig_qid].get(s) for s in ["s0", "s1", "s2", "s3", "s4"]
                }
                if "_meta" not in rec:
                    rec["_meta"] = orig[model][orig_qid]["_meta"]
        for model in v21:
            if v21_qidx in v21[model]:
                rec["models"][model] = {
                    s: v21[model][v21_qidx].get(s) for s in ["s0", "s1", "s2", "s3", "s4"]
                }
        unified[q_text] = rec
    return unified


# ============================================================
# 2. Feature extraction (adapted from phase3_adaptive_selector.py, 28 features)
# ============================================================

def extract_features(question: str, evidence: str, db_path: str, db_id: str) -> dict:
    q_lower = question.lower()
    feat = {}

    # Question length
    feat["q_length"] = len(question)
    feat["q_words"] = len(question.split())

    # Evidence
    feat["has_evidence"] = 1 if evidence and evidence.strip() else 0
    feat["evidence_length"] = len(evidence) if evidence else 0
    feat["evidence_words"] = len(evidence.split()) if evidence and evidence.strip() else 0

    # SQL complexity keywords
    agg_words = ["average", "total", "how many", "count", "sum", "maximum", "minimum",
                 "highest", "lowest", "most", "least", "avg", "max", "min"]
    feat["kw_aggregation"] = sum(1 for w in agg_words if w in q_lower)

    group_words = ["each", "per", "every", "group", "by category", "by type"]
    feat["kw_groupby"] = sum(1 for w in group_words if w in q_lower)

    compare_words = ["more than", "less than", "greater", "at least", "at most",
                     "between", "above", "below", "exceed"]
    feat["kw_compare"] = sum(1 for w in compare_words if w in q_lower)

    negation_words = ["not", "except", "without", "exclude", "other than", "never"]
    feat["kw_negation"] = sum(1 for w in negation_words if w in q_lower)

    join_words = ["and", "with", "both", "along with", "together", "related"]
    feat["kw_join_hint"] = sum(1 for w in join_words if w in q_lower)

    subq_words = ["among those", "of those", "within", "who also", "that have"]
    feat["kw_subquery_hint"] = sum(1 for w in subq_words if w in q_lower)

    temporal_words = ["before", "after", "during", "since", "until", "year", "month",
                      "date", "time", "when", "recent", "latest", "earliest"]
    feat["kw_temporal"] = sum(1 for w in temporal_words if w in q_lower)

    ranking_words = ["top", "bottom", "rank", "first", "last", "nth", "second",
                     "third", "order by", "sort"]
    feat["kw_ranking"] = sum(1 for w in ranking_words if w in q_lower)

    math_words = ["percentage", "ratio", "proportion", "difference", "divide",
                  "multiply", "subtract", "rate", "growth"]
    feat["kw_math"] = sum(1 for w in math_words if w in q_lower)

    # Question structure
    feat["has_multiple_conditions"] = 1 if q_lower.count(" and ") >= 2 else 0
    feat["num_question_marks"] = question.count("?")
    feat["has_or"] = 1 if " or " in q_lower else 0

    # Schema complexity
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        feat["num_tables"] = cur.fetchone()[0]
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        total_cols = 0
        fk_count = 0
        for t in tables:
            cur.execute("PRAGMA table_info(`%s`)" % t)
            total_cols += len(cur.fetchall())
            cur.execute("PRAGMA foreign_key_list(`%s`)" % t)
            fk_count += len(cur.fetchall())
        feat["num_columns"] = total_cols
        feat["avg_cols_per_table"] = total_cols / len(tables) if tables else 0
        feat["num_fks"] = fk_count
        feat["schema_complexity"] = feat["num_tables"] * feat["avg_cols_per_table"]
        conn.close()
    except Exception:
        feat["num_tables"] = 0
        feat["num_columns"] = 0
        feat["avg_cols_per_table"] = 0
        feat["num_fks"] = 0
        feat["schema_complexity"] = 0

    # Top BIRD DB one-hots
    top_dbs = ["california_schools", "card_games", "toxicology", "financial",
               "european_football_2", "thrombosis_prediction", "formula_1", "student_club"]
    for db in top_dbs:
        feat["db_%s" % db] = 1 if db_id == db else 0

    return feat


# ============================================================
# 3. Build pooled N=5 dataset
# ============================================================

def build_pooled_dataset(unified: dict, models: list[str]) -> tuple[list, list]:
    """Build pooled records (one record per (query, model) pair).

    Returns:
        records: list of {qid, q_text, model, features, correct: {s0..s4}, best, s0_correct}
        per_query_meta: list aligned with unique queries (for query-level CV folding)
    """
    # Check if BIRD DBs are accessible. If not, schema features (5 of 28) will be 0.
    bird_dbs_available = Path(DB_DIR).exists()
    if not bird_dbs_available:
        print(f"  ⚠ BIRD DBs not found at {DB_DIR}")
        print(f"    Schema-complexity features (5 of 28) will be 0 — proceeding with reduced feature set.")
        print(f"    This is documented in the limitations section of the output.")

    # Filter to queries where ALL models have S0 result (anchor consistency)
    valid_q = []
    for q_text, rec in unified.items():
        if all(m in rec["models"] and rec["models"][m].get("s0") is not None for m in models):
            valid_q.append(q_text)
    print(f"  Queries with all {len(models)} models having S0: {len(valid_q)}")

    records = []
    per_query_meta = []
    for q_idx, q_text in enumerate(sorted(valid_q)):
        rec = unified[q_text]
        meta = rec.get("_meta", {})
        db_id = meta.get("db_id", "")
        evidence = meta.get("evidence", "")
        db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))

        # If BIRD DBs are not on disk, schema features will fall back to 0
        # via extract_features's try/except. We do not skip the query.
        feat = extract_features(q_text, evidence, db_path, db_id)

        # Per-query S0 majority (for stratification)
        s0_votes = sum(1 for m in models if rec["models"][m].get("s0"))
        per_query_s0_majority = 1 if s0_votes > len(models) / 2 else 0

        per_query_meta.append({
            "qid_text": q_text[:50],
            "q_idx": q_idx,
            "s0_majority": per_query_s0_majority,
            "db_id": db_id,
        })

        for model in models:
            md = rec["models"][model]
            correct = {s: bool(md.get(s, False)) for s in ["s0", "s1", "s2", "s3", "s4"]}

            # Best strategy: prefer S0 if correct, else first correct alternative
            if correct["s0"]:
                best = "s0"
            else:
                best = "s0"
                for s in ["s1", "s3", "s4", "s2"]:
                    if correct.get(s):
                        best = s
                        break

            records.append({
                "q_idx": q_idx,
                "q_text": q_text,
                "model": model,
                "features": feat,
                "correct": correct,
                "best": best,
                "s0_correct": correct["s0"],
                "s0_wrong": not correct["s0"],
                "any_rescue": (not correct["s0"]) and any(correct.get(s) for s in ["s1", "s2", "s3", "s4"]),
            })

    return records, per_query_meta


# ============================================================
# 4. CV setup (query-level StratifiedKFold)
# ============================================================

def make_query_level_folds(per_query_meta: list, n_splits: int = N_FOLDS, seed: int = SEED):
    """Returns list of (train_q_idx, test_q_idx) tuples at query level."""
    from sklearn.model_selection import StratifiedKFold

    q_idx_array = np.array([m["q_idx"] for m in per_query_meta])
    y_strat = np.array([m["s0_majority"] for m in per_query_meta])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_q_pos, test_q_pos in skf.split(q_idx_array, y_strat):
        train_q_set = set(q_idx_array[train_q_pos].tolist())
        test_q_set = set(q_idx_array[test_q_pos].tolist())
        folds.append((train_q_set, test_q_set))
    return folds


def split_records_by_q_set(records: list, train_q_set: set, test_q_set: set):
    train_idx = [i for i, r in enumerate(records) if r["q_idx"] in train_q_set]
    test_idx = [i for i, r in enumerate(records) if r["q_idx"] in test_q_set]
    return np.array(train_idx), np.array(test_idx)


# ============================================================
# 5. Statistical helpers
# ============================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ============================================================
# 6. Rule-based selectors (Paradigm A — no training)
# ============================================================

def rule_v1(feat: dict) -> str:
    if feat["num_tables"] >= 8 and feat["kw_aggregation"] >= 1:
        return "s1"
    if feat["q_words"] >= 20 and feat["has_multiple_conditions"]:
        return "s4"
    if feat["kw_subquery_hint"] >= 1:
        return "s3"
    if feat["kw_negation"] >= 1 and feat["num_tables"] >= 5:
        return "s4"
    return "s0"


def rule_v2(feat: dict) -> str:
    if feat["evidence_length"] > 100 and feat["kw_math"] >= 1:
        return "s3"
    if feat["num_tables"] >= 8 and feat["kw_aggregation"] >= 1:
        return "s1"
    if feat["kw_ranking"] >= 1 and feat["num_columns"] >= 30:
        return "s4"
    if feat["q_words"] >= 25 and feat["kw_temporal"] >= 1:
        return "s3"
    if feat["has_multiple_conditions"] and feat["kw_negation"] >= 1:
        return "s4"
    if feat["kw_subquery_hint"] >= 1 and feat["num_tables"] >= 4:
        return "s3"
    return "s0"


def evaluate_rule_selector(records: list, rule_fn, name: str) -> dict:
    n = len(records)
    correct = 0
    picks = Counter()
    pick_correct = Counter()
    s0_correct = sum(1 for r in records if r["correct"]["s0"])

    for r in records:
        pick = rule_fn(r["features"])
        picks[pick] += 1
        if r["correct"].get(pick, False):
            correct += 1
            pick_correct[pick] += 1

    acc = correct / n * 100
    rescue_pp = acc - (s0_correct / n * 100)
    ci_lo, ci_hi = wilson_ci(correct, n)
    return {
        "name": name,
        "paradigm": "A",
        "n": n,
        "end_to_end_correct": correct,
        "end_to_end_acc_pct": acc,
        "rescue_pp": rescue_pp,
        "wilson_ci_pct": [ci_lo * 100, ci_hi * 100],
        "picks": dict(picks),
        "pick_correct": dict(pick_correct),
    }


# ============================================================
# 7. Multi-class ML selector (Paradigm B)
# ============================================================

def train_multiclass(records: list, folds: list, clf_name: str, clf_template) -> dict:
    """5-class predictor: predict best strategy directly."""
    from sklearn.metrics import (
        confusion_matrix, precision_recall_fscore_support, classification_report,
    )

    feature_names = sorted(records[0]["features"].keys())
    X = np.array([[r["features"][f] for f in feature_names] for r in records])
    y = np.array([r["best"] for r in records])

    all_y_true = []
    all_y_pred = []
    end_to_end_correct = 0
    s0_correct_total = 0
    n_evaluated = 0
    picks_total = Counter()
    importances = []

    for train_q_set, test_q_set in folds:
        train_idx, test_idx = split_records_by_q_set(records, train_q_set, test_q_set)
        clf = type(clf_template)(**clf_template.get_params())
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])

        all_y_true.extend(y[test_idx].tolist())
        all_y_pred.extend(preds.tolist())

        for i, pred in zip(test_idx, preds):
            picks_total[pred] += 1
            n_evaluated += 1
            if records[i]["correct"].get(pred, False):
                end_to_end_correct += 1
            if records[i]["correct"]["s0"]:
                s0_correct_total += 1

        if hasattr(clf, "feature_importances_"):
            importances.append(clf.feature_importances_)

    n = n_evaluated  # actual records evaluated across folds
    acc = end_to_end_correct / n * 100 if n else 0.0
    rescue_pp = acc - (s0_correct_total / n * 100 if n else 0.0)
    ci_lo, ci_hi = wilson_ci(end_to_end_correct, n)

    classes = sorted(set(all_y_true) | set(all_y_pred))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=classes, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average="weighted", zero_division=0
    )

    feat_imp_top = []
    if importances:
        avg_imp = np.mean(importances, axis=0)
        for idx in np.argsort(avg_imp)[::-1][:10]:
            feat_imp_top.append({"name": feature_names[idx], "imp": float(avg_imp[idx])})

    return {
        "name": clf_name,
        "paradigm": "B",
        "n": n,
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_acc_pct": acc,
        "rescue_pp": rescue_pp,
        "wilson_ci_pct": [ci_lo * 100, ci_hi * 100],
        "picks": dict(picks_total),
        "classifier_metrics": {
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "precision_per_class": dict(zip(classes, p_per.tolist())),
            "recall_per_class": dict(zip(classes, r_per.tolist())),
            "f1_per_class": dict(zip(classes, f1_per.tolist())),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        },
        "feature_importance_top10": feat_imp_top,
    }


# ============================================================
# 8. Binary gate selector (Paradigm C)
# ============================================================

def train_binary_gate(records: list, folds: list, clf_name: str, clf_template,
                      fallback: str = "s1") -> dict:
    """Predict S0 will fail; on predicted fail, fallback to fixed strategy."""
    from sklearn.metrics import (
        confusion_matrix, precision_recall_fscore_support, roc_auc_score,
        average_precision_score,
    )

    feature_names = sorted(records[0]["features"].keys())
    X = np.array([[r["features"][f] for f in feature_names] for r in records])
    y_s0_fail = np.array([0 if r["s0_correct"] else 1 for r in records])

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    end_to_end_correct = 0
    s0_correct_total = 0
    n_evaluated = 0
    picks_total = Counter()
    importances = []

    for train_q_set, test_q_set in folds:
        train_idx, test_idx = split_records_by_q_set(records, train_q_set, test_q_set)
        clf = type(clf_template)(**clf_template.get_params())
        clf.fit(X[train_idx], y_s0_fail[train_idx])
        preds = clf.predict(X[test_idx])

        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(X[test_idx])
            fail_class_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
            proba_fail = probas[:, fail_class_idx] if fail_class_idx >= 0 else np.zeros(len(test_idx))
        else:
            proba_fail = preds.astype(float)

        all_y_true.extend(y_s0_fail[test_idx].tolist())
        all_y_pred.extend(preds.tolist())
        all_y_proba.extend(proba_fail.tolist())

        for i, pred in zip(test_idx, preds):
            pick = "s0" if pred == 0 else fallback
            picks_total[pick] += 1
            n_evaluated += 1
            if records[i]["correct"].get(pick, False):
                end_to_end_correct += 1
            if records[i]["correct"]["s0"]:
                s0_correct_total += 1

        if hasattr(clf, "feature_importances_"):
            importances.append(clf.feature_importances_)

    n = n_evaluated
    acc = end_to_end_correct / n * 100 if n else 0.0
    rescue_pp = acc - (s0_correct_total / n * 100 if n else 0.0)
    ci_lo, ci_hi = wilson_ci(end_to_end_correct, n)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=[0, 1], zero_division=0
    )
    try:
        auroc = float(roc_auc_score(all_y_true, all_y_proba))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(all_y_true, all_y_proba))
    except ValueError:
        auprc = float("nan")

    feat_imp_top = []
    if importances:
        avg_imp = np.mean(importances, axis=0)
        for idx in np.argsort(avg_imp)[::-1][:10]:
            feat_imp_top.append({"name": feature_names[idx], "imp": float(avg_imp[idx])})

    return {
        "name": clf_name,
        "paradigm": "C",
        "fallback": fallback,
        "n": n,
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_acc_pct": acc,
        "rescue_pp": rescue_pp,
        "wilson_ci_pct": [ci_lo * 100, ci_hi * 100],
        "picks": dict(picks_total),
        "gate_metrics": {
            "confusion_matrix": cm.tolist(),
            "precision_s0_succeed": float(p_per[0]),
            "precision_s0_fail":    float(p_per[1]),
            "recall_s0_succeed":    float(r_per[0]),
            "recall_s0_fail":       float(r_per[1]),
            "f1_s0_succeed":        float(f1_per[0]),
            "f1_s0_fail":           float(f1_per[1]),
            "auroc": auroc,
            "auprc": auprc,
        },
        "feature_importance_top10": feat_imp_top,
    }


# ============================================================
# 9. Cascade selector (Paradigm D)
# ============================================================

def train_cascade(records: list, folds: list, clf_name: str, clf_template) -> dict:
    """Gate (S0-fail) + per-strategy rescue classifiers in priority order S1→S3→S4→S2."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, confusion_matrix,
        precision_recall_fscore_support,
    )

    feature_names = sorted(records[0]["features"].keys())
    X = np.array([[r["features"][f] for f in feature_names] for r in records])
    y_s0_fail = np.array([0 if r["s0_correct"] else 1 for r in records])
    rescue_priority = ["s1", "s3", "s4", "s2"]

    end_to_end_correct = 0
    s0_correct_total = 0
    n_evaluated = 0
    picks_total = Counter()

    # Aggregated rescue metrics across folds
    rescue_y_true_agg = {s: [] for s in rescue_priority}
    rescue_y_pred_agg = {s: [] for s in rescue_priority}
    rescue_y_proba_agg = {s: [] for s in rescue_priority}

    gate_y_true_agg = []
    gate_y_pred_agg = []
    gate_y_proba_agg = []

    for train_q_set, test_q_set in folds:
        train_idx, test_idx = split_records_by_q_set(records, train_q_set, test_q_set)

        # Train gate
        gate = type(clf_template)(**clf_template.get_params())
        gate.fit(X[train_idx], y_s0_fail[train_idx])
        gate_pred = gate.predict(X[test_idx])
        if hasattr(gate, "predict_proba"):
            fc = list(gate.classes_).index(1) if 1 in gate.classes_ else -1
            gate_proba = gate.predict_proba(X[test_idx])[:, fc] if fc >= 0 else np.zeros(len(test_idx))
        else:
            gate_proba = gate_pred.astype(float)

        gate_y_true_agg.extend(y_s0_fail[test_idx].tolist())
        gate_y_pred_agg.extend(gate_pred.tolist())
        gate_y_proba_agg.extend(gate_proba.tolist())

        # Train rescue classifiers on S0-fail subset of training data
        train_fail_mask = y_s0_fail[train_idx] == 1
        train_fail_global = train_idx[train_fail_mask]
        X_train_fail = X[train_fail_global]

        trained_rescue = {}
        for s in rescue_priority:
            y_rescue_full = np.array([1 if records[i]["correct"][s] else 0 for i in train_fail_global])
            if y_rescue_full.sum() >= 3 and len(y_rescue_full) >= 10:
                r_clf = type(clf_template)(**clf_template.get_params())
                r_clf.fit(X_train_fail, y_rescue_full)
                trained_rescue[s] = r_clf

        # Per-strategy rescue evaluation on test set's S0-fail subset
        test_fail_mask_local = y_s0_fail[test_idx] == 1
        test_fail_global = test_idx[test_fail_mask_local]
        if len(test_fail_global) > 0:
            X_test_fail = X[test_fail_global]
            for s in rescue_priority:
                y_true_rescue = np.array([1 if records[i]["correct"][s] else 0 for i in test_fail_global])
                if s in trained_rescue:
                    r_pred = trained_rescue[s].predict(X_test_fail)
                    if hasattr(trained_rescue[s], "predict_proba"):
                        rc = list(trained_rescue[s].classes_).index(1) if 1 in trained_rescue[s].classes_ else -1
                        r_proba = trained_rescue[s].predict_proba(X_test_fail)[:, rc] if rc >= 0 else np.zeros(len(X_test_fail))
                    else:
                        r_proba = r_pred.astype(float)
                else:
                    r_pred = np.zeros(len(X_test_fail), dtype=int)
                    r_proba = np.zeros(len(X_test_fail))
                rescue_y_true_agg[s].extend(y_true_rescue.tolist())
                rescue_y_pred_agg[s].extend(r_pred.tolist())
                rescue_y_proba_agg[s].extend(r_proba.tolist())

        # Cascade end-to-end pick
        for i, gp in zip(test_idx, gate_pred):
            if gp == 0:
                pick = "s0"
            else:
                pick = "s0"  # default if no rescue triggers
                for s in rescue_priority:
                    if s in trained_rescue:
                        rp = trained_rescue[s].predict(X[i:i+1])
                        if rp[0] == 1:
                            pick = s
                            break
                else:
                    # exhausted rescue priorities without trigger → fallback s1
                    pick = "s1"

            picks_total[pick] += 1
            n_evaluated += 1
            if records[i]["correct"].get(pick, False):
                end_to_end_correct += 1
            if records[i]["correct"]["s0"]:
                s0_correct_total += 1

    n = n_evaluated
    acc = end_to_end_correct / n * 100 if n else 0.0
    rescue_pp = acc - (s0_correct_total / n * 100 if n else 0.0)
    ci_lo, ci_hi = wilson_ci(end_to_end_correct, n)

    # Gate metrics
    gate_cm = confusion_matrix(gate_y_true_agg, gate_y_pred_agg, labels=[0, 1])
    p_g, r_g, f1_g, _ = precision_recall_fscore_support(
        gate_y_true_agg, gate_y_pred_agg, labels=[0, 1], zero_division=0
    )
    try:
        gate_auroc = float(roc_auc_score(gate_y_true_agg, gate_y_proba_agg))
    except ValueError:
        gate_auroc = float("nan")
    try:
        gate_auprc = float(average_precision_score(gate_y_true_agg, gate_y_proba_agg))
    except ValueError:
        gate_auprc = float("nan")

    # Per-strategy rescue metrics (Imbalanced — main metric is AUPRC)
    rescue_metrics = {}
    for s in rescue_priority:
        yt = rescue_y_true_agg[s]
        yp = rescue_y_pred_agg[s]
        ypr = rescue_y_proba_agg[s]
        if len(yt) == 0 or sum(yt) == 0:
            rescue_metrics[s] = {"support": int(sum(yt)), "n_test_fail": len(yt)}
            continue
        try:
            auroc = float(roc_auc_score(yt, ypr))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(yt, ypr))
        except ValueError:
            auprc = float("nan")
        prf = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
        rescue_metrics[s] = {
            "support": int(sum(yt)),
            "n_test_fail": len(yt),
            "positive_rate": float(np.mean(yt)),
            "precision": float(prf[0][1]),
            "recall":    float(prf[1][1]),
            "f1":        float(prf[2][1]),
            "auroc": auroc,
            "auprc": auprc,
        }

    return {
        "name": clf_name,
        "paradigm": "D",
        "n": n,
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_acc_pct": acc,
        "rescue_pp": rescue_pp,
        "wilson_ci_pct": [ci_lo * 100, ci_hi * 100],
        "picks": dict(picks_total),
        "gate_metrics": {
            "confusion_matrix": gate_cm.tolist(),
            "precision_s0_succeed": float(p_g[0]),
            "precision_s0_fail":    float(p_g[1]),
            "recall_s0_succeed":    float(r_g[0]),
            "recall_s0_fail":       float(r_g[1]),
            "f1_s0_succeed":        float(f1_g[0]),
            "f1_s0_fail":           float(f1_g[1]),
            "auroc": gate_auroc,
            "auprc": gate_auprc,
        },
        "rescue_metrics": rescue_metrics,
    }


# ============================================================
# 10. Anchor baselines (always-S0, S0→S1, oracle)
# ============================================================

def compute_anchors(records: list) -> dict:
    n = len(records)
    s0_total = sum(1 for r in records if r["correct"]["s0"])
    s0_then_s1 = sum(1 for r in records if r["correct"]["s0"] or r["correct"]["s1"])
    oracle = sum(1 for r in records if any(r["correct"][s] for s in ["s0", "s1", "s2", "s3", "s4"]))
    return {
        "n_pooled": n,
        "always_s0_correct": s0_total,
        "always_s0_pct": s0_total / n * 100,
        "s0_then_s1_correct": s0_then_s1,
        "s0_then_s1_pct": s0_then_s1 / n * 100,
        "oracle_correct": oracle,
        "oracle_pct": oracle / n * 100,
        "rescue_ceiling_pp": (oracle - s0_total) / n * 100,
        "wilson_ci_always_s0_pct": [c * 100 for c in wilson_ci(s0_total, n)],
        "wilson_ci_oracle_pct": [c * 100 for c in wilson_ci(oracle, n)],
    }


# ============================================================
# 11. Visualisations (matplotlib)
# ============================================================

def save_figures(results: dict, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-class confusion matrices
    for sel_name in ["multiclass_dt", "multiclass_rf", "multiclass_gb"]:
        sel = results["selectors"].get(sel_name)
        if not sel or "classifier_metrics" not in sel:
            continue
        cm = np.array(sel["classifier_metrics"]["confusion_matrix"])
        classes = sel["classifier_metrics"]["classes"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True (best strategy)")
        ax.set_title(f"Confusion Matrix — {sel_name}")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_matrix_{sel_name}.png", dpi=120)
        plt.close()

    # 2. Gate confusion matrices
    for sel_name in ["gate_dt", "gate_rf", "gate_gb"]:
        sel = results["selectors"].get(sel_name)
        if not sel or "gate_metrics" not in sel:
            continue
        cm = np.array(sel["gate_metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["S0 succeed", "S0 fail"])
        ax.set_yticklabels(["S0 succeed", "S0 fail"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Gate Confusion Matrix — {sel_name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_matrix_{sel_name}.png", dpi=120)
        plt.close()

    # 3. Bar chart: end-to-end accuracy per selector + anchors
    anchors = results["anchors"]
    sel_names_order = ["always_s0", "s0_then_s1",
                       "rule_v1", "rule_v2",
                       "multiclass_dt", "multiclass_rf", "multiclass_gb",
                       "gate_dt", "gate_rf", "gate_gb",
                       "cascade_dt", "cascade_rf", "cascade_gb",
                       "oracle"]
    accs = []
    labels = []
    for nm in sel_names_order:
        if nm == "always_s0":
            accs.append(anchors["always_s0_pct"]); labels.append("Always-S0")
        elif nm == "s0_then_s1":
            accs.append(anchors["s0_then_s1_pct"]); labels.append("S0→S1 fallback")
        elif nm == "oracle":
            accs.append(anchors["oracle_pct"]); labels.append("Oracle")
        elif nm in results["selectors"]:
            accs.append(results["selectors"][nm]["end_to_end_acc_pct"])
            labels.append(nm)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#888"] + ["#888"] + ["#4a90e2"] * 2 + ["#7ed321"] * 3 + \
             ["#f5a623"] * 3 + ["#d0021b"] * 3 + ["#888"]
    colors = colors[:len(accs)]
    ax.bar(labels, accs, color=colors)
    ax.set_ylabel("End-to-end accuracy (%)")
    ax.set_title("Selector end-to-end accuracy vs anchors (pooled n=%d)" % anchors["n_pooled"])
    ax.axhline(anchors["always_s0_pct"], linestyle="--", color="gray", alpha=0.5)
    ax.axhline(anchors["oracle_pct"], linestyle="--", color="gray", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "selector_accuracy_comparison.png", dpi=120)
    plt.close()

    print(f"  Figures saved to {fig_dir}")


# ============================================================
# 12. Main
# ============================================================

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only 1 fold + 1 multiclass selector for pipeline verification")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 3 v3: N=5 Adaptive Selector Full ML Evaluation")
    if args.dry_run:
        print("           [DRY RUN MODE — 1 fold, 1 selector]")
    print("=" * 70)
    t_start = time.time()

    # ---- Load data ----
    print("\n[1] Loading original-4 (phase1 + phase2)...")
    orig = load_original_4_per_query()
    print(f"  Loaded models: {list(orig.keys())}")

    print("\n[2] Loading v2.1 E1 new models (Gemma4)...")
    v21 = load_v21_e1_per_query()
    print(f"  Loaded models: {list(v21.keys())}")

    print("\n[3] Aligning by question text...")
    unified = align_by_question(orig, v21)

    if not unified:
        print("\nERROR: no common questions found.")
        return

    all_models = sorted(set(orig.keys()) | set(v21.keys()))
    print(f"\n[4] All N={len(all_models)} models: {all_models}")

    # ---- Build pooled dataset ----
    print(f"\n[5] Building pooled dataset (N={len(all_models)} models, query-level features)...")
    records, per_query_meta = build_pooled_dataset(unified, all_models)
    print(f"  Total records: {len(records)} (queries × models)")
    print(f"  Unique queries: {len(per_query_meta)}")

    # Sanity check: oracle/anchors should match a5
    print(f"\n[6] Computing anchor baselines...")
    anchors = compute_anchors(records)
    print(f"  Always-S0:     {anchors['always_s0_pct']:.2f}%  (CI {anchors['wilson_ci_always_s0_pct'][0]:.2f}-{anchors['wilson_ci_always_s0_pct'][1]:.2f}%)")
    print(f"  S0→S1:         {anchors['s0_then_s1_pct']:.2f}%")
    print(f"  Oracle:        {anchors['oracle_pct']:.2f}%  (CI {anchors['wilson_ci_oracle_pct'][0]:.2f}-{anchors['wilson_ci_oracle_pct'][1]:.2f}%)")
    print(f"  Rescue ceiling: {anchors['rescue_ceiling_pp']:.2f} pp")

    expected_oracle = 50.20
    if abs(anchors["oracle_pct"] - expected_oracle) > 1.0:
        print(f"\n  ⚠ WARNING: Oracle {anchors['oracle_pct']:.2f}% differs from a5 ({expected_oracle}%) by >1pp")

    # ---- CV folds ----
    print(f"\n[7] Building {N_FOLDS}-fold StratifiedKFold (query level, stratify on majority S0)...")
    folds = make_query_level_folds(per_query_meta, n_splits=N_FOLDS, seed=SEED)
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i+1}: train_q={len(tr)}, test_q={len(te)}")

    if args.dry_run:
        folds = folds[:1]

    # ---- Sklearn imports ----
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    selectors = {}

    # ---- Paradigm A: Rule-based ----
    if not args.dry_run:
        print("\n[8] Paradigm A: Rule-based selectors")
        selectors["rule_v1"] = evaluate_rule_selector(records, rule_v1, "rule_v1")
        print(f"  rule_v1: {selectors['rule_v1']['end_to_end_acc_pct']:.2f}% (Δ {selectors['rule_v1']['rescue_pp']:+.2f}pp)")
        selectors["rule_v2"] = evaluate_rule_selector(records, rule_v2, "rule_v2")
        print(f"  rule_v2: {selectors['rule_v2']['end_to_end_acc_pct']:.2f}% (Δ {selectors['rule_v2']['rescue_pp']:+.2f}pp)")

    # ---- Paradigm B: Multi-class ML ----
    print("\n[9] Paradigm B: Multi-class ML selectors")
    multiclass_clfs = [
        ("multiclass_dt", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED)),
        ("multiclass_rf", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ("multiclass_gb", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                      min_samples_leaf=10, random_state=SEED)),
    ]
    if args.dry_run:
        multiclass_clfs = multiclass_clfs[1:2]  # only RF in dry run

    for nm, clf in multiclass_clfs:
        t0 = time.time()
        result = train_multiclass(records, folds, nm, clf)
        elapsed = time.time() - t0
        selectors[nm] = result
        cm_metrics = result["classifier_metrics"]
        print(f"  {nm}: end-to-end {result['end_to_end_acc_pct']:.2f}% (Δ {result['rescue_pp']:+.2f}pp) | "
              f"f1_macro={cm_metrics['f1_macro']:.3f} f1_weighted={cm_metrics['f1_weighted']:.3f} | {elapsed:.1f}s")

    if args.dry_run:
        print("\n[DRY RUN] Skipping Paradigms C & D and figure generation. Inspect output and re-run without --dry-run.")
    else:
        # ---- Paradigm C: Binary gate ----
        print("\n[10] Paradigm C: Binary gate selectors (fallback=S1)")
        gate_clfs = [
            ("gate_dt", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED)),
            ("gate_rf", RandomForestClassifier(n_estimators=100, max_depth=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1)),
            ("gate_gb", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                    min_samples_leaf=5, random_state=SEED)),
        ]
        for nm, clf in gate_clfs:
            t0 = time.time()
            result = train_binary_gate(records, folds, nm, clf, fallback="s1")
            elapsed = time.time() - t0
            selectors[nm] = result
            gm = result["gate_metrics"]
            print(f"  {nm}: end-to-end {result['end_to_end_acc_pct']:.2f}% (Δ {result['rescue_pp']:+.2f}pp) | "
                  f"AUROC={gm['auroc']:.3f} AUPRC={gm['auprc']:.3f} f1_fail={gm['f1_s0_fail']:.3f} | {elapsed:.1f}s")

        # ---- Paradigm D: Cascade ----
        print("\n[11] Paradigm D: Cascade selectors (gate → S1→S3→S4→S2 rescue)")
        cascade_clfs = [
            ("cascade_dt", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED)),
            ("cascade_rf", RandomForestClassifier(n_estimators=100, max_depth=5,
                                                   class_weight="balanced", random_state=SEED, n_jobs=-1)),
            ("cascade_gb", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                       min_samples_leaf=5, random_state=SEED)),
        ]
        for nm, clf in cascade_clfs:
            t0 = time.time()
            result = train_cascade(records, folds, nm, clf)
            elapsed = time.time() - t0
            selectors[nm] = result
            print(f"  {nm}: end-to-end {result['end_to_end_acc_pct']:.2f}% (Δ {result['rescue_pp']:+.2f}pp) | "
                  f"gate AUROC={result['gate_metrics']['auroc']:.3f} | {elapsed:.1f}s")
            for s, rm in result["rescue_metrics"].items():
                if "auprc" in rm:
                    print(f"      rescue {s}: support={rm['support']}/{rm['n_test_fail']} "
                          f"({rm['positive_rate']*100:.1f}%) AUPRC={rm['auprc']:.3f} F1={rm['f1']:.3f}")

    # ---- Per-model breakdown ----
    print("\n[12] Per-model breakdown (always-S0 / oracle)")
    per_model = {}
    for m in all_models:
        m_records = [r for r in records if r["model"] == m]
        n_m = len(m_records)
        s0 = sum(1 for r in m_records if r["correct"]["s0"])
        oracle = sum(1 for r in m_records if any(r["correct"][s] for s in ["s0","s1","s2","s3","s4"]))
        per_model[m] = {
            "n": n_m,
            "s0_pct": s0 / n_m * 100 if n_m else 0,
            "oracle_pct": oracle / n_m * 100 if n_m else 0,
            "rescue_pp": (oracle - s0) / n_m * 100 if n_m else 0,
        }
        print(f"  {m}: n={n_m} S0={per_model[m]['s0_pct']:.2f}% Oracle={per_model[m]['oracle_pct']:.2f}% (Δ {per_model[m]['rescue_pp']:+.2f}pp)")

    # ---- Assemble results ----
    timestamp = datetime.now().isoformat(timespec="seconds")
    bird_dbs_available = Path(DB_DIR).exists()
    results = {
        "meta": {
            "timestamp": timestamp,
            "n_models": len(all_models),
            "models": all_models,
            "n_unique_queries": len(per_query_meta),
            "n_pooled_records": len(records),
            "cv_strategy": f"{N_FOLDS}-fold StratifiedKFold (query-level, stratify on majority S0)",
            "stratify_target": "per-query S0-correctness majority across N=5",
            "seed": SEED,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t_start, 1),
            "rl_specialist_excluded_reason": "Arctic-R1 is single-pass; does not produce S1-S4 outputs (paper §5.5 line 266)",
            "bird_dbs_available": bird_dbs_available,
            "feature_set": "full 28 features" if bird_dbs_available
                           else "23 features (schema-complexity 5 features set to 0; BIRD dev DBs not on disk)",
        },
        "anchors": anchors,
        "selectors": selectors,
        "per_model_breakdown": per_model,
    }

    out_path = output_dir / "n5_selector_results.json"
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print(f"\n[13] Results saved → {out_path}")

    if not args.dry_run and not args.skip_figures:
        print("\n[14] Generating figures...")
        save_figures(results, output_dir)

    print(f"\n{'='*70}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")
    print(f"{'='*70}")

    # ---- Summary table ----
    print("\nSUMMARY TABLE (end-to-end accuracy, pooled n={})".format(len(records)))
    print(f"{'Selector':<22}{'Acc %':>8}{'Δ pp':>8}{'Wilson CI':>20}")
    print("-" * 60)
    print(f"{'Always-S0 (anchor)':<22}{anchors['always_s0_pct']:>8.2f}{0.00:>8.2f}"
          f"  [{anchors['wilson_ci_always_s0_pct'][0]:.2f}, {anchors['wilson_ci_always_s0_pct'][1]:.2f}]")
    print(f"{'S0→S1 (anchor)':<22}{anchors['s0_then_s1_pct']:>8.2f}"
          f"{anchors['s0_then_s1_pct'] - anchors['always_s0_pct']:>8.2f}")
    for nm, sel in selectors.items():
        ci = sel["wilson_ci_pct"]
        print(f"{nm:<22}{sel['end_to_end_acc_pct']:>8.2f}{sel['rescue_pp']:>8.2f}"
              f"  [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"{'Oracle (ceiling)':<22}{anchors['oracle_pct']:>8.2f}{anchors['rescue_ceiling_pp']:>8.2f}"
          f"  [{anchors['wilson_ci_oracle_pct'][0]:.2f}, {anchors['wilson_ci_oracle_pct'][1]:.2f}]")


if __name__ == "__main__":
    main()
