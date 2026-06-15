"""Phase 1 Error Analysis: Cross-model error patterns on BIRD dev set.

Analyzes:
1. Query structure classification (JOINs, subqueries, aggregations, etc.)
2. Cross-model agreement/disagreement patterns
3. Error type breakdown per model
4. Per-database difficulty analysis
5. Structural complexity vs accuracy
"""
import json, sqlite3, re
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird" / "dev_20240627"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS = {
    "qwen": "phase1_qwen_baseline.json",
    "gemini": "phase1_gemini3flash_baseline.json",
    "deepseek": "phase1_deepseek_baseline.json",
    "gpt41": "phase1_gpt41_baseline.json",
}

# ── Query Structure Classification ──────────────────────────────────

def classify_sql(sql):
    """Classify a SQL query by structural features."""
    sql_upper = sql.upper()
    features = {}

    # JOIN count
    features["join_count"] = len(re.findall(r'\bJOIN\b', sql_upper))
    features["has_join"] = features["join_count"] > 0

    # Subquery
    # Count SELECT occurrences (first one is main query)
    select_count = len(re.findall(r'\bSELECT\b', sql_upper))
    features["subquery_count"] = max(0, select_count - 1)
    features["has_subquery"] = features["subquery_count"] > 0

    # Aggregation functions
    agg_funcs = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
    features["agg_funcs"] = [f for f in agg_funcs if re.search(r'\b%s\s*\(' % f, sql_upper)]
    features["has_aggregation"] = len(features["agg_funcs"]) > 0

    # GROUP BY / HAVING
    features["has_group_by"] = bool(re.search(r'\bGROUP\s+BY\b', sql_upper))
    features["has_having"] = bool(re.search(r'\bHAVING\b', sql_upper))

    # Set operations
    features["has_union"] = bool(re.search(r'\bUNION\b', sql_upper))
    features["has_intersect"] = bool(re.search(r'\bINTERSECT\b', sql_upper))
    features["has_except"] = bool(re.search(r'\bEXCEPT\b', sql_upper))
    features["has_set_op"] = features["has_union"] or features["has_intersect"] or features["has_except"]

    # ORDER BY / LIMIT
    features["has_order_by"] = bool(re.search(r'\bORDER\s+BY\b', sql_upper))
    features["has_limit"] = bool(re.search(r'\bLIMIT\b', sql_upper))

    # WHERE conditions
    features["has_where"] = bool(re.search(r'\bWHERE\b', sql_upper))
    features["has_like"] = bool(re.search(r'\bLIKE\b', sql_upper))
    features["has_between"] = bool(re.search(r'\bBETWEEN\b', sql_upper))
    features["has_in"] = bool(re.search(r'\bIN\s*\(', sql_upper))
    features["has_case"] = bool(re.search(r'\bCASE\b', sql_upper))
    features["has_distinct"] = bool(re.search(r'\bDISTINCT\b', sql_upper))

    # Complexity score (simple heuristic)
    complexity = 0
    complexity += features["join_count"] * 2
    complexity += features["subquery_count"] * 3
    complexity += len(features["agg_funcs"])
    complexity += features["has_group_by"] * 1
    complexity += features["has_having"] * 2
    complexity += features["has_set_op"] * 3
    complexity += features["has_case"] * 2
    features["complexity_score"] = complexity

    # Structural category
    if features["has_set_op"]:
        features["category"] = "set_operation"
    elif features["has_subquery"] and features["has_join"]:
        features["category"] = "nested_join"
    elif features["has_subquery"]:
        features["category"] = "nested"
    elif features["join_count"] >= 3:
        features["category"] = "multi_join"
    elif features["has_join"] and features["has_aggregation"]:
        features["category"] = "join_agg"
    elif features["has_join"]:
        features["category"] = "join"
    elif features["has_aggregation"]:
        features["category"] = "aggregation"
    else:
        features["category"] = "simple_select"

    return features


# ── Error Classification ────────────────────────────────────────────

def classify_error(result):
    """Classify the error type for a wrong prediction."""
    if result["correct"]:
        return "correct"
    err = result.get("error")
    if err and "gold_error" in str(err):
        return "gold_error"
    if err == "no SQL generated":
        return "no_sql"
    if err and err != "no SQL generated":
        # Further classify execution errors
        err_lower = str(err).lower()
        if "no such column" in err_lower:
            return "err_column"
        if "no such table" in err_lower:
            return "err_table"
        if "syntax error" in err_lower or "near" in err_lower:
            return "err_syntax"
        if "ambiguous" in err_lower:
            return "err_ambiguous"
        return "err_other"
    return "wrong_result"


# ── Main Analysis ───────────────────────────────────────────────────

def run_analysis():
    # Load BIRD dev data
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)
    print("Loaded %d BIRD dev queries" % len(bird))

    # Load all model results
    model_results = {}
    for name, filename in MODELS.items():
        with open(RESULTS_DIR / filename) as f:
            data = json.load(f)
        model_results[name] = {r["query_id"]: r for r in data["results"]}
        print("  %s: %d results" % (name, len(data["results"])))

    # ── 1. Query Structure Classification ───────────────────────────
    print("\n" + "=" * 70)
    print("1. QUERY STRUCTURE CLASSIFICATION")
    print("=" * 70)

    categories = defaultdict(list)
    all_features = []
    for i, q in enumerate(bird):
        feat = classify_sql(q["SQL"])
        feat["query_id"] = i
        feat["difficulty"] = q["difficulty"]
        feat["db_id"] = q["db_id"]
        all_features.append(feat)
        categories[feat["category"]].append(i)

    print("\n%-18s %6s %8s %8s %8s" % ("Category", "Count", "Simple", "Moderate", "Chall."))
    print("-" * 55)
    for cat in ["simple_select", "aggregation", "join", "join_agg", "multi_join", "nested", "nested_join", "set_operation"]:
        ids = categories[cat]
        diffs = Counter(all_features[i]["difficulty"] for i in ids)
        print("%-18s %6d %8d %8d %8d" % (
            cat, len(ids), diffs.get("simple", 0), diffs.get("moderate", 0), diffs.get("challenging", 0)
        ))
    print("-" * 55)
    print("%-18s %6d" % ("TOTAL", len(bird)))

    # ── 2. Per-Category Accuracy ────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. ACCURACY BY QUERY STRUCTURE × MODEL")
    print("=" * 70)

    cat_order = ["simple_select", "aggregation", "join", "join_agg", "multi_join", "nested", "nested_join", "set_operation"]
    model_names = list(MODELS.keys())

    print("\n%-18s %5s | %8s %8s %8s %8s" % ("Category", "N", *[m[:8] for m in model_names]))
    print("-" * 70)
    for cat in cat_order:
        ids = categories[cat]
        accs = []
        for m in model_names:
            correct = sum(1 for qid in ids if model_results[m].get(qid, {}).get("correct", False))
            accs.append(correct / len(ids) * 100 if ids else 0)
        print("%-18s %5d | %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (cat, len(ids), *accs))

    # Overall
    accs = []
    for m in model_names:
        correct = sum(1 for r in model_results[m].values() if r.get("correct", False))
        accs.append(correct / len(bird) * 100)
    print("-" * 70)
    print("%-18s %5d | %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % ("OVERALL", len(bird), *accs))

    # ── 3. Error Type Breakdown ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. ERROR TYPE BREAKDOWN PER MODEL")
    print("=" * 70)

    err_types = ["correct", "wrong_result", "err_syntax", "err_column", "err_table", "err_ambiguous", "err_other", "no_sql", "gold_error"]
    for m in model_names:
        counts = Counter()
        for r in model_results[m].values():
            counts[classify_error(r)] += 1
        print("\n%s:" % m)
        for et in err_types:
            if counts[et]:
                print("  %-15s %5d (%5.1f%%)" % (et, counts[et], counts[et] / len(bird) * 100))

    # ── 4. Cross-Model Agreement ────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. CROSS-MODEL AGREEMENT ANALYSIS")
    print("=" * 70)

    all_correct = 0
    all_wrong = 0
    some_correct = 0
    agreement_patterns = Counter()

    for i in range(len(bird)):
        pattern = tuple(1 if model_results[m].get(i, {}).get("correct", False) else 0 for m in model_names)
        agreement_patterns[pattern] += 1
        n_correct = sum(pattern)
        if n_correct == len(model_names):
            all_correct += 1
        elif n_correct == 0:
            all_wrong += 1
        else:
            some_correct += 1

    print("\nAll 4 models correct:   %4d (%5.1f%%)" % (all_correct, all_correct / len(bird) * 100))
    print("Some models correct:    %4d (%5.1f%%)" % (some_correct, some_correct / len(bird) * 100))
    print("All 4 models wrong:     %4d (%5.1f%%)" % (all_wrong, all_wrong / len(bird) * 100))

    # Show top disagreement patterns
    print("\nTop agreement patterns (qwen, gemini, deepseek, gpt41):")
    print("%-25s %5s %6s" % ("Pattern", "Count", "%"))
    print("-" * 40)
    for pattern, count in agreement_patterns.most_common(10):
        labels = ["".join(m[0].upper() if v else "." for m, v in zip(model_names, pattern))]
        print("%-25s %5d %5.1f%%" % (str(pattern), count, count / len(bird) * 100))

    # ── 5. All-wrong Query Analysis ─────────────────────────────────
    print("\n" + "=" * 70)
    print("5. ALL-MODELS-WRONG QUERY ANALYSIS")
    print("=" * 70)

    all_wrong_ids = []
    for i in range(len(bird)):
        if all(not model_results[m].get(i, {}).get("correct", False) for m in model_names):
            all_wrong_ids.append(i)

    # Distribution by difficulty
    aw_diffs = Counter(bird[i]["difficulty"] for i in all_wrong_ids)
    print("\nDifficulty distribution of all-wrong queries:")
    for d in ["simple", "moderate", "challenging"]:
        total_d = sum(1 for q in bird if q["difficulty"] == d)
        aw_d = aw_diffs.get(d, 0)
        print("  %-12s: %4d / %4d (%5.1f%% of %s queries are universally wrong)" % (
            d, aw_d, total_d, aw_d / total_d * 100 if total_d else 0, d))

    # Distribution by structure
    aw_cats = Counter(all_features[i]["category"] for i in all_wrong_ids)
    print("\nStructure distribution of all-wrong queries:")
    for cat in cat_order:
        total_c = len(categories[cat])
        aw_c = aw_cats.get(cat, 0)
        print("  %-18s: %4d / %4d (%5.1f%%)" % (cat, aw_c, total_c, aw_c / total_c * 100 if total_c else 0))

    # Distribution by database
    aw_dbs = Counter(bird[i]["db_id"] for i in all_wrong_ids)
    print("\nTop 10 databases with most all-wrong queries:")
    for db_id, count in aw_dbs.most_common(10):
        total_db = sum(1 for q in bird if q["db_id"] == db_id)
        print("  %-30s: %3d / %3d (%5.1f%%)" % (db_id, count, total_db, count / total_db * 100))

    # ── 6. Per-Database Accuracy ────────────────────────────────────
    print("\n" + "=" * 70)
    print("6. PER-DATABASE ACCURACY (sorted by avg accuracy)")
    print("=" * 70)

    db_stats = defaultdict(lambda: {"total": 0, "accs": defaultdict(int)})
    for i, q in enumerate(bird):
        db_stats[q["db_id"]]["total"] += 1
        for m in model_names:
            if model_results[m].get(i, {}).get("correct", False):
                db_stats[q["db_id"]]["accs"][m] += 1

    print("\n%-30s %5s | %8s %8s %8s %8s | %5s" % ("Database", "N", *[m[:8] for m in model_names], "Avg"))
    print("-" * 95)
    db_list = []
    for db_id, st in db_stats.items():
        avg_acc = sum(st["accs"][m] / st["total"] * 100 for m in model_names) / len(model_names)
        db_list.append((db_id, st["total"], avg_acc, st["accs"]))
    db_list.sort(key=lambda x: x[2])  # sort by avg accuracy

    for db_id, total, avg_acc, accs in db_list:
        model_accs = [accs[m] / total * 100 for m in model_names]
        print("%-30s %5d | %7.1f%% %7.1f%% %7.1f%% %7.1f%% | %4.1f%%" % (
            db_id[:30], total, *model_accs, avg_acc))

    # ── 7. Complexity Score vs Accuracy ─────────────────────────────
    print("\n" + "=" * 70)
    print("7. COMPLEXITY SCORE VS ACCURACY")
    print("=" * 70)

    # Bin complexity scores
    bins = [(0, 0, "0 (trivial)"), (1, 2, "1-2 (low)"), (3, 5, "3-5 (medium)"), (6, 9, "6-9 (high)"), (10, 99, "10+ (very high)")]
    print("\n%-18s %5s | %8s %8s %8s %8s" % ("Complexity", "N", *[m[:8] for m in model_names]))
    print("-" * 70)
    for lo, hi, label in bins:
        ids = [f["query_id"] for f in all_features if lo <= f["complexity_score"] <= hi]
        if not ids:
            continue
        accs = []
        for m in model_names:
            correct = sum(1 for qid in ids if model_results[m].get(qid, {}).get("correct", False))
            accs.append(correct / len(ids) * 100)
        print("%-18s %5d | %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (label, len(ids), *accs))

    # ── 8. Unique Strengths per Model ───────────────────────────────
    print("\n" + "=" * 70)
    print("8. UNIQUE MODEL STRENGTHS (only this model correct)")
    print("=" * 70)

    for m_idx, m in enumerate(model_names):
        unique_correct = []
        for i in range(len(bird)):
            if model_results[m].get(i, {}).get("correct", False):
                others_wrong = all(
                    not model_results[m2].get(i, {}).get("correct", False)
                    for m2_idx, m2 in enumerate(model_names) if m2_idx != m_idx
                )
                if others_wrong:
                    unique_correct.append(i)

        diff_dist = Counter(bird[i]["difficulty"] for i in unique_correct)
        cat_dist = Counter(all_features[i]["category"] for i in unique_correct)
        print("\n%s: %d uniquely correct queries" % (m, len(unique_correct)))
        print("  By difficulty: %s" % dict(diff_dist))
        print("  By structure:  %s" % dict(cat_dist.most_common(5)))

    # ── Save analysis results ───────────────────────────────────────
    analysis_output = {
        "query_features": all_features,
        "categories": {k: v for k, v in categories.items()},
        "all_wrong_ids": all_wrong_ids,
    }
    out_path = RESULTS_DIR / "phase1_error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis_output, f, ensure_ascii=False)
    print("\n\nSaved analysis data to %s" % out_path)


if __name__ == "__main__":
    run_analysis()
