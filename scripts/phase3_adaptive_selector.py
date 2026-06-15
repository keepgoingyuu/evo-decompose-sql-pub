"""Phase 3: Adaptive Strategy Selector — predict best pipeline per query.

Uses query-time features (no gold SQL) to select between S0-S4.
Trains on model results, evaluates with cross-validation.

Three selector paradigms:
  1. Multi-class: predict best strategy directly (S0-S4)
  2. Binary gate: predict "will S0 fail?" → if yes, pick best alternative
  3. Per-strategy rescue: for each Si, predict "will Si succeed where S0 fails?"

Usage:
    python scripts/phase3_adaptive_selector.py [model]
    model: deepseek (default), gemini, gpt41
"""
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.schema_utils import DB_DIR, BIRD_DIR

# ============================================================
# 1. Load all results
# ============================================================

S0_FILE_MAP = {
    "deepseek": "phase1_deepseek_baseline.json",
    "gemini": "phase1_gemini3flash_baseline.json",
    "gpt41": "phase1_gpt41_baseline.json",
    "qwen": "phase1_qwen_baseline.json",
}


def load_results(model="deepseek"):
    """Load S0-S4 results for a given model on mod+chal."""
    s0_file = PROJECT_ROOT / "results" / S0_FILE_MAP[model]
    with open(s0_file) as f:
        s0_data = json.load(f)
    s0 = {r["query_id"]: r for r in s0_data["results"]}

    strategies = {}
    for name in ["s1", "s2", "s3", "s4"]:
        p = PROJECT_ROOT / "results" / ("phase2_%s_%s.json" % (model, name))
        if not p.exists():
            print("  WARNING: %s not found, skipping" % p.name)
            continue
        with open(p) as f:
            data = json.load(f)
        strategies[name] = {r["query_id"]: r for r in data["results"]}

    return s0, strategies


def load_bird():
    """Load BIRD dev set, return mod+chal queries indexed by position."""
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)
    return {i: q for i, q in enumerate(bird)
            if q["difficulty"] in ("moderate", "challenging")}

# ============================================================
# 2. Feature extraction (query-time only, no gold SQL)
# ============================================================

def extract_features(question, evidence, db_path, db_id):
    """Extract features available at query time."""
    q_lower = question.lower()
    feat = {}

    # Question length features
    feat["q_length"] = len(question)
    feat["q_words"] = len(question.split())

    # Evidence features
    feat["has_evidence"] = 1 if evidence.strip() else 0
    feat["evidence_length"] = len(evidence)
    feat["evidence_words"] = len(evidence.split()) if evidence.strip() else 0

    # Keyword indicators (proxy for SQL complexity)
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

    # Question structure indicators
    feat["has_multiple_conditions"] = 1 if q_lower.count(" and ") >= 2 else 0
    feat["num_question_marks"] = question.count("?")
    feat["has_or"] = 1 if " or " in q_lower else 0

    # Schema complexity (from database)
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

    # Database ID as categorical (one-hot top DBs)
    top_dbs = ["california_schools", "card_games", "toxicology", "financial",
               "european_football_2", "thrombosis_prediction", "formula_1", "student_club"]
    for db in top_dbs:
        feat["db_%s" % db] = 1 if db_id == db else 0

    return feat

# ============================================================
# 3. Build dataset
# ============================================================

def build_dataset(s0, strategies, bird_by_id):
    """Build feature + correctness data for all mod+chal queries."""
    available_strats = sorted(strategies.keys())
    dataset = []

    for qid, q in bird_by_id.items():
        db_id = q["db_id"]
        db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))
        if not Path(db_path).exists():
            continue

        feat = extract_features(q["question"], q.get("evidence", ""), db_path, db_id)

        correct = {}
        correct["s0"] = s0.get(qid, {}).get("correct", False)
        for name in available_strats:
            correct[name] = strategies[name].get(qid, {}).get("correct", False)

        # Best strategy: prefer s0 if correct (cheapest), else first correct alternative
        if correct["s0"]:
            best = "s0"
        else:
            best = "s0"
            for name in ["s1", "s3", "s4", "s2"]:
                if name in available_strats and correct.get(name, False):
                    best = name
                    break

        # Binary label: can any non-S0 strategy rescue this query?
        s0_wrong = not correct["s0"]
        any_rescue = s0_wrong and any(correct.get(s, False) for s in available_strats)

        dataset.append({
            "qid": qid,
            "db_id": db_id,
            "difficulty": q["difficulty"],
            "features": feat,
            "best": best,
            "correct": correct,
            "s0_wrong": s0_wrong,
            "any_rescue": any_rescue,
        })

    return dataset

# ============================================================
# 4. Evaluation helpers
# ============================================================

def evaluate_selector(dataset, selector_fn, name="selector"):
    """Evaluate a selector function on the dataset."""
    correct = 0
    total = len(dataset)
    strategy_picks = Counter()
    strategy_correct = Counter()

    for d in dataset:
        pick = selector_fn(d["features"])
        strategy_picks[pick] += 1
        if d["correct"].get(pick, False):
            correct += 1
            strategy_correct[pick] += 1

    acc = correct / total * 100 if total else 0
    print("  %s: %d/%d = %.1f%%" % (name, correct, total, acc))
    for s in ["s0", "s1", "s2", "s3", "s4"]:
        if strategy_picks[s]:
            s_acc = strategy_correct[s] / strategy_picks[s] * 100
            print("    %s: picked %d times, %.1f%% correct" % (s, strategy_picks[s], s_acc))
    return acc


def oracle_accuracy(dataset):
    """Upper bound: pick best strategy per query."""
    correct = sum(1 for d in dataset if any(d["correct"].values()))
    total = len(dataset)
    acc = correct / total * 100
    print("  Oracle: %d/%d = %.1f%%" % (correct, total, acc))
    return acc

# ============================================================
# 5. Selectors
# ============================================================

def majority_vote_selector(feat):
    """Always pick S0 (baseline)."""
    return "s0"


def rule_based_selector_v1(feat):
    """Simple rule-based strategy selection (v1)."""
    if feat["num_tables"] >= 8 and feat["kw_aggregation"] >= 1:
        return "s1"
    if feat["q_words"] >= 20 and feat["has_multiple_conditions"]:
        return "s4"
    if feat["kw_subquery_hint"] >= 1:
        return "s3"
    if feat["kw_negation"] >= 1 and feat["num_tables"] >= 5:
        return "s4"
    return "s0"


def rule_based_selector_v2(feat):
    """Enhanced rule-based selector (v2) — tuned from data analysis."""
    # Evidence-driven: long evidence often means complex domain knowledge
    if feat["evidence_length"] > 100 and feat["kw_math"] >= 1:
        return "s3"  # skeleton helps structure calculations

    # Complex schema + aggregation → schema filter
    if feat["num_tables"] >= 8 and feat["kw_aggregation"] >= 1:
        return "s1"

    # Ranking + large schema → DIN-SQL style
    if feat["kw_ranking"] >= 1 and feat["num_columns"] >= 30:
        return "s4"

    # Long question with temporal + groupby → decompose or skeleton
    if feat["q_words"] >= 25 and feat["kw_temporal"] >= 1:
        return "s3"

    # Multiple conditions + negation → structured approach
    if feat["has_multiple_conditions"] and feat["kw_negation"] >= 1:
        return "s4"

    # Subquery hints
    if feat["kw_subquery_hint"] >= 1 and feat["num_tables"] >= 4:
        return "s3"

    return "s0"

# ============================================================
# 6. ML-based Selectors
# ============================================================

def run_ml_selectors(dataset):
    """Train and evaluate ML-based selectors."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.metrics import classification_report
        import numpy as np
    except ImportError:
        print("  sklearn not installed, skipping ML selectors")
        return

    feature_names = sorted(dataset[0]["features"].keys())
    X = np.array([[d["features"][f] for f in feature_names] for d in dataset])

    # ---- Approach A: Multi-class (predict best strategy) ----
    print("\n--- A. Multi-class: Predict Best Strategy ---")
    y_best = np.array([d["best"] for d in dataset])
    print("  Class distribution: %s" % dict(Counter(y_best).most_common()))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for clf_name, clf in [
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                          min_samples_leaf=10, random_state=42)),
    ]:
        cv_correct = 0
        cv_total = 0
        for train_idx, test_idx in kf.split(X):
            fold_clf = type(clf)(**clf.get_params())
            fold_clf.fit(X[train_idx], y_best[train_idx])
            preds = fold_clf.predict(X[test_idx])
            for idx, pred in zip(test_idx, preds):
                if dataset[idx]["correct"].get(pred, False):
                    cv_correct += 1
                cv_total += 1

        print("  %s CV: %d/%d = %.1f%%" % (clf_name, cv_correct, cv_total,
                                             cv_correct / cv_total * 100))

    # ---- Approach B: Binary Gate (predict S0 failure → then pick alternative) ----
    print("\n--- B. Binary Gate: Predict S0 Failure ---")
    y_s0_fail = np.array([1 if d["s0_wrong"] else 0 for d in dataset])
    print("  S0 fail rate: %d/%d = %.1f%%" % (y_s0_fail.sum(), len(y_s0_fail),
                                                y_s0_fail.mean() * 100))

    # Among S0-failure queries, which alternative is best?
    s0_fail_indices = [i for i, d in enumerate(dataset) if d["s0_wrong"]]
    rescue_stats = Counter()
    for i in s0_fail_indices:
        for s in ["s1", "s3", "s4", "s2"]:
            if dataset[i]["correct"][s]:
                rescue_stats[s] += 1
                break
        else:
            rescue_stats["none"] += 1
    print("  Among S0-fail: rescue by %s" % dict(rescue_stats.most_common()))

    # Find best single fallback strategy
    best_fallback_scores = {}
    for fallback in ["s1", "s2", "s3", "s4"]:
        score = sum(1 for i in s0_fail_indices if dataset[i]["correct"][fallback])
        best_fallback_scores[fallback] = score
    print("  Fallback scores (on S0-fail): %s" % best_fallback_scores)
    best_single_fallback = max(best_fallback_scores, key=lambda k: best_fallback_scores[k])
    print("  Best single fallback: %s" % best_single_fallback)

    # Binary gate classifiers
    for clf_name, clf in [
        ("DT Gate", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)),
        ("RF Gate", RandomForestClassifier(n_estimators=100, max_depth=5,
                                            class_weight="balanced", random_state=42)),
        ("GB Gate", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                min_samples_leaf=5, random_state=42)),
    ]:
        # For each fallback strategy
        for fallback in [best_single_fallback, "best_per_query"]:
            cv_correct = 0
            cv_total = 0
            cv_switches = 0

            for train_idx, test_idx in kf.split(X):
                fold_clf = type(clf)(**clf.get_params())
                fold_clf.fit(X[train_idx], y_s0_fail[train_idx])
                preds = fold_clf.predict(X[test_idx])

                for idx, pred in zip(test_idx, preds):
                    if pred == 0:
                        # Predict S0 will succeed → use S0
                        pick = "s0"
                    else:
                        cv_switches += 1
                        if fallback == "best_per_query":
                            # Try each alternative, pick first correct (optimistic)
                            pick = "s0"
                            for s in ["s1", "s3", "s4", "s2"]:
                                if dataset[idx]["correct"][s]:
                                    pick = s
                                    break
                        else:
                            pick = fallback

                    if dataset[idx]["correct"].get(pick, False):
                        cv_correct += 1
                    cv_total += 1

            fb_label = fallback if fallback != "best_per_query" else "oracle-fb"
            print("  %s → %s: %d/%d = %.1f%% (switched %d)" % (
                clf_name, fb_label, cv_correct, cv_total,
                cv_correct / cv_total * 100, cv_switches))

    # ---- Approach C: Per-Strategy Rescue Classifiers ----
    print("\n--- C. Per-Strategy Rescue (S0 wrong → will Si fix it?) ---")

    # Only train on S0-fail queries
    X_fail = X[y_s0_fail == 1]
    dataset_fail = [d for d in dataset if d["s0_wrong"]]

    for strat in ["s1", "s2", "s3", "s4"]:
        y_rescue = np.array([1 if d["correct"][strat] else 0 for d in dataset_fail])
        if y_rescue.sum() < 5:
            print("  %s: too few rescues (%d), skip" % (strat, y_rescue.sum()))
            continue

        clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                      class_weight="balanced", random_state=42)
        # Cross-val predict on the fail subset
        kf_fail = KFold(n_splits=min(5, len(X_fail)), shuffle=True, random_state=42)
        preds = cross_val_predict(clf, X_fail, y_rescue, cv=kf_fail)
        tp = sum(1 for p, y in zip(preds, y_rescue) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, y_rescue) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, y_rescue) if p == 0 and y == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        print("  %s rescue: TP=%d FP=%d FN=%d | prec=%.2f rec=%.2f" % (
            strat, tp, fp, fn, prec, rec))

    # ---- Approach D: Cascade selector with probability thresholds ----
    print("\n--- D. Cascade Selector (Gate + Rescue priority) ---")

    gate_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                       class_weight="balanced", random_state=42)

    # Train rescue classifiers for each strategy (on S0-fail data only)
    rescue_clfs = {}
    for strat in ["s1", "s3", "s4", "s2"]:  # priority order
        y_rescue = np.array([1 if d["correct"][strat] else 0 for d in dataset_fail])
        if y_rescue.sum() >= 5:
            r_clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                            class_weight="balanced", random_state=42)
            rescue_clfs[strat] = (r_clf, y_rescue)

    # Full CV with cascade
    cv_correct = 0
    cv_total = 0
    cv_routing = Counter()

    for train_idx, test_idx in kf.split(X):
        # Step 1: Train gate
        gate = type(gate_clf)(**gate_clf.get_params())
        gate.fit(X[train_idx], y_s0_fail[train_idx])

        # Step 2: Train rescue classifiers on S0-fail subset of training data
        train_fail_mask = y_s0_fail[train_idx] == 1
        X_train_fail = X[train_idx][train_fail_mask]
        trained_rescue = {}
        for strat, (r_clf_template, y_full_rescue) in rescue_clfs.items():
            y_train_rescue = y_full_rescue[np.isin(
                np.where(y_s0_fail == 1)[0],
                train_idx
            )]
            # Map back: find which positions in dataset_fail correspond to train_idx
            fail_global_indices = np.where(y_s0_fail == 1)[0]
            train_fail_local = np.isin(fail_global_indices, train_idx)
            X_t = X[fail_global_indices[train_fail_local]]
            y_t = y_full_rescue[train_fail_local]
            if y_t.sum() >= 3 and len(y_t) >= 10:
                r = type(r_clf_template)(**r_clf_template.get_params())
                r.fit(X_t, y_t)
                trained_rescue[strat] = r

        # Step 3: Predict on test set
        gate_preds = gate.predict(X[test_idx])
        # Also get probabilities for confidence
        if hasattr(gate, "predict_proba"):
            gate_probs = gate.predict_proba(X[test_idx])
            fail_class_idx = list(gate.classes_).index(1) if 1 in gate.classes_ else -1
        else:
            gate_probs = None
            fail_class_idx = -1

        for i, (idx, gate_pred) in enumerate(zip(test_idx, gate_preds)):
            if gate_pred == 0:
                pick = "s0"
            else:
                # Gate says S0 will fail → try rescue classifiers
                pick = "s0"  # default fallback
                for strat in ["s1", "s3", "s4", "s2"]:
                    if strat in trained_rescue:
                        r_pred = trained_rescue[strat].predict(X[idx:idx+1])
                        if r_pred[0] == 1:
                            pick = strat
                            break

            cv_routing[pick] += 1
            if dataset[idx]["correct"].get(pick, False):
                cv_correct += 1
            cv_total += 1

    print("  Cascade CV: %d/%d = %.1f%%" % (cv_correct, cv_total,
                                              cv_correct / cv_total * 100))
    print("  Routing: %s" % dict(cv_routing.most_common()))

    # ---- Feature Importance Analysis ----
    print("\n--- Feature Importance (for S0-failure prediction) ---")
    gate_full = RandomForestClassifier(n_estimators=100, max_depth=5,
                                        class_weight="balanced", random_state=42)
    gate_full.fit(X, y_s0_fail)
    importances = sorted(zip(feature_names, gate_full.feature_importances_),
                         key=lambda x: -x[1])
    print("  Top features for predicting S0 failure:")
    for fname, imp in importances[:15]:
        if imp > 0.01:
            print("    %s: %.3f" % (fname, imp))

    return feature_names, gate_full

# ============================================================
# 7. Detailed Analysis
# ============================================================

def analyze_rescue_patterns(dataset):
    """Analyze which query patterns are rescued by non-S0 strategies."""
    print("\n" + "=" * 70)
    print("RESCUE PATTERN ANALYSIS")
    print("=" * 70)

    s0_wrong = [d for d in dataset if d["s0_wrong"]]
    rescued = [d for d in s0_wrong if d["any_rescue"]]
    unrescuable = [d for d in s0_wrong if not d["any_rescue"]]

    print("\n  S0 wrong: %d queries" % len(s0_wrong))
    print("  Rescued by S1-S4: %d (%.1f%%)" % (len(rescued),
                                                 len(rescued) / len(s0_wrong) * 100 if s0_wrong else 0))
    print("  No strategy works: %d (%.1f%%)" % (len(unrescuable),
                                                  len(unrescuable) / len(s0_wrong) * 100 if s0_wrong else 0))

    # Per-strategy rescue counts
    print("\n  Rescue by strategy (among S0-wrong):")
    for s in ["s1", "s2", "s3", "s4"]:
        count = sum(1 for d in s0_wrong if d["correct"][s])
        excl = sum(1 for d in s0_wrong if d["correct"][s] and
                   not any(d["correct"][s2] for s2 in ["s1", "s2", "s3", "s4"] if s2 != s))
        print("    %s: %d rescues (%d exclusive)" % (s, count, excl))

    # Feature comparison: rescued vs unrescuable
    print("\n  Feature comparison (rescued vs unrescuable):")
    numeric_feats = [f for f in dataset[0]["features"].keys() if not f.startswith("db_")]
    for fname in sorted(numeric_feats):
        r_vals = [d["features"][fname] for d in rescued]
        u_vals = [d["features"][fname] for d in unrescuable]
        r_mean = sum(r_vals) / len(r_vals) if r_vals else 0
        u_mean = sum(u_vals) / len(u_vals) if u_vals else 0
        if abs(r_mean - u_mean) > 0.1 * max(abs(r_mean), abs(u_mean), 0.01):
            print("    %s: rescued=%.2f vs unrescuable=%.2f" % (fname, r_mean, u_mean))

    # Difficulty breakdown
    print("\n  By difficulty:")
    for diff in ["moderate", "challenging"]:
        diff_total = [d for d in dataset if d["difficulty"] == diff]
        diff_s0_wrong = [d for d in diff_total if d["s0_wrong"]]
        diff_rescued = [d for d in diff_s0_wrong if d["any_rescue"]]
        print("    %s: %d total, %d S0-wrong, %d rescued (%.1f%% rescue rate)" % (
            diff, len(diff_total), len(diff_s0_wrong), len(diff_rescued),
            len(diff_rescued) / len(diff_s0_wrong) * 100 if diff_s0_wrong else 0))

    # Database breakdown (top DBs with most rescues)
    print("\n  Top databases with rescuable queries:")
    db_rescue = Counter(d["db_id"] for d in rescued)
    for db_id, count in db_rescue.most_common(10):
        db_total = sum(1 for d in dataset if d["db_id"] == db_id)
        db_s0_wrong = sum(1 for d in s0_wrong if d["db_id"] == db_id)
        print("    %s: %d rescued / %d S0-wrong / %d total" % (db_id, count, db_s0_wrong, db_total))

# ============================================================
# Main
# ============================================================

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    if model not in S0_FILE_MAP:
        print("Unknown model: %s. Choose from: %s" % (model, list(S0_FILE_MAP.keys())))
        sys.exit(1)

    print("=" * 70)
    print("PHASE 3: Adaptive Strategy Selector (%s)" % model.upper())
    print("=" * 70)

    print("\nLoading data...")
    s0, strategies = load_results(model)
    if not strategies:
        print("  ERROR: No S1-S4 results found for %s" % model)
        sys.exit(1)
    print("  Loaded strategies: %s" % sorted(strategies.keys()))
    bird = load_bird()
    print("  %d mod+chal queries" % len(bird))

    print("\nBuilding dataset...")
    dataset = build_dataset(s0, strategies, bird)
    print("  %d queries with features" % len(dataset))

    # Strategy distribution
    best_dist = Counter(d["best"] for d in dataset)
    print("\n  Best strategy distribution:")
    for s, c in best_dist.most_common():
        print("    %s: %d (%.1f%%)" % (s, c, c / len(dataset) * 100))

    # ---- Baselines ----
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    print("\n--- Always S0 ---")
    s0_acc = evaluate_selector(dataset, majority_vote_selector, "Always S0")

    print("\n--- Oracle ---")
    oracle_acc = oracle_accuracy(dataset)
    gap = oracle_acc - s0_acc
    print("  Gap (Oracle - S0): %.1f percentage points" % gap)

    # ---- Rule-based ----
    print("\n" + "=" * 70)
    print("RULE-BASED SELECTORS")
    print("=" * 70)

    print("\n--- Rule-based v1 (original) ---")
    evaluate_selector(dataset, rule_based_selector_v1, "Rule-v1")

    print("\n--- Rule-based v2 (enhanced) ---")
    evaluate_selector(dataset, rule_based_selector_v2, "Rule-v2")

    # ---- ML Selectors ----
    print("\n" + "=" * 70)
    print("ML-BASED SELECTORS")
    print("=" * 70)
    run_ml_selectors(dataset)

    # ---- Rescue Pattern Analysis ----
    analyze_rescue_patterns(dataset)

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("  Always S0:     %.1f%%" % s0_acc)
    print("  Rule-based v1: %.1f%%" % evaluate_selector(dataset, rule_based_selector_v1, "R-v1"))
    print("  Rule-based v2: %.1f%%" % evaluate_selector(dataset, rule_based_selector_v2, "R-v2"))
    print("  Oracle:        %.1f%%" % oracle_acc)
    print("  Gap to close:  %.1f pp" % gap)


if __name__ == "__main__":
    main()
