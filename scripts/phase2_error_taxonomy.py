"""Phase 2 Error Taxonomy: Analyze error patterns across S0-S4 pipelines.

Produces:
  1. Error type distribution per pipeline
  2. Error propagation analysis (S0 correct → Si wrong)
  3. Rescue analysis (S0 wrong → Si correct)
  4. Per-database and per-difficulty breakdown
  5. Cross-pipeline agreement analysis

Usage:
    python scripts/phase2_error_taxonomy.py [model]

Models: deepseek (default), gemini, gpt41, qwen
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
from src.utils.sql_utils import execute_sql

MODEL_NAMES = {
    "deepseek": "DeepSeek-V3",
    "gemini": "Gemini 2.0 Flash",
    "gpt41": "GPT-4.1",
    "qwen": "Qwen2.5-Coder-7B",
}

# ============================================================
# 1. Load data
# ============================================================

def find_s0_file(model):
    """Find S0 baseline file for a model."""
    candidates = [
        PROJECT_ROOT / "results" / f"phase1_{model}_baseline.json",
        PROJECT_ROOT / "results" / f"phase1_{model}3flash_baseline.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_all(model="deepseek"):
    """Load S0-S4 results + BIRD metadata for given model."""
    with open(BIRD_DIR / "dev.json") as f:
        bird = json.load(f)
    bird_by_id = {i: q for i, q in enumerate(bird)
                  if q["difficulty"] in ("moderate", "challenging")}

    results = {}
    available = []

    # S0
    s0_path = find_s0_file(model)
    if s0_path is None:
        print(f"ERROR: No S0 baseline found for {model}")
        sys.exit(1)
    with open(s0_path) as f:
        data = json.load(f)
    results["s0"] = {r["query_id"]: r for r in data["results"]}
    available.append("s0")

    # S1-S4
    for name in ["s1", "s2", "s3", "s4"]:
        path = PROJECT_ROOT / "results" / f"phase2_{model}_{name}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[name] = {r["query_id"]: r for r in data["results"]}
            available.append(name)

    return bird_by_id, results, available


def classify_error(result, db_path):
    """Classify the error type for a single result."""
    if result.get("correct", False):
        return "correct"

    pred_sql = result.get("pred_sql", "")
    if not pred_sql or "SELECT" not in pred_sql.upper():
        return "no_sql"

    # Check if it's an execution error
    exec_err = result.get("exec_error")
    if exec_err:
        err_lower = str(exec_err).lower()
        if "no such column" in err_lower:
            return "err_column"
        if "no such table" in err_lower:
            return "err_table"
        if "syntax error" in err_lower or "near" in err_lower:
            return "err_syntax"
        if "ambiguous" in err_lower:
            return "err_ambiguous"
        return "err_other"

    # SQL executes but wrong result
    return "wrong_result"

# ============================================================
# 2. Error Type Distribution
# ============================================================

def analyze_error_types(bird_by_id, results):
    """Error type distribution per pipeline."""
    print("=" * 70)
    print("1. ERROR TYPE DISTRIBUTION PER PIPELINE")
    print("=" * 70)

    strategies = ["s0", "s1", "s2", "s3", "s4"]
    all_types = ["correct", "wrong_result", "err_column", "err_table",
                 "err_syntax", "err_ambiguous", "err_other", "no_sql"]

    # Count by pipeline
    counts = {s: Counter() for s in strategies}
    for qid in bird_by_id:
        for s in strategies:
            r = results[s].get(qid, {})
            db_id = bird_by_id[qid]["db_id"]
            db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))
            err_type = classify_error(r, db_path)
            counts[s][err_type] += 1

    total = len(bird_by_id)

    # Print table
    print("\n  %-15s" % "Error Type", end="")
    for s in strategies:
        print(" %8s" % s.upper(), end="")
    print()
    print("  " + "-" * 60)

    for etype in all_types:
        print("  %-15s" % etype, end="")
        for s in strategies:
            c = counts[s][etype]
            pct = c / total * 100
            print(" %4d(%2.0f%%)" % (c, pct), end="")
        print()

    # Exec error summary
    print("\n  Execution errors (total):")
    for s in strategies:
        exec_errs = sum(counts[s][t] for t in all_types if t.startswith("err_"))
        print("    %s: %d/%d (%.1f%%)" % (s, exec_errs, total, exec_errs / total * 100))

    return counts

# ============================================================
# 3. Error Propagation Analysis
# ============================================================

def analyze_error_propagation(bird_by_id, results):
    """Analyze when pipelines introduce errors where S0 was correct."""
    print("\n" + "=" * 70)
    print("2. ERROR PROPAGATION (S0 correct → Si wrong)")
    print("=" * 70)

    s0_correct_ids = [qid for qid in bird_by_id
                      if results["s0"].get(qid, {}).get("correct", False)]
    total_s0_correct = len(s0_correct_ids)
    print("\n  S0 correct: %d queries" % total_s0_correct)

    for s in ["s1", "s2", "s3", "s4"]:
        propagated = [qid for qid in s0_correct_ids
                      if not results[s].get(qid, {}).get("correct", False)]
        print("\n  %s introduces errors in %d/%d (%.1f%%) of S0-correct queries" % (
            s.upper(), len(propagated), total_s0_correct,
            len(propagated) / total_s0_correct * 100))

        # What types of errors does Si introduce?
        err_types = Counter()
        for qid in propagated:
            r = results[s].get(qid, {})
            db_id = bird_by_id[qid]["db_id"]
            db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))
            err_types[classify_error(r, db_path)] += 1
        for etype, count in err_types.most_common():
            print("    %s: %d" % (etype, count))

    # S0 correct → ALL Si wrong (robust S0 queries)
    robust = [qid for qid in s0_correct_ids
              if all(not results[s].get(qid, {}).get("correct", False)
                     for s in ["s1", "s2", "s3", "s4"])]
    print("\n  S0 correct + ALL Si wrong: %d queries (%.1f%% of S0-correct)" % (
        len(robust), len(robust) / total_s0_correct * 100))

    # S0 correct → ALL Si also correct (easy queries)
    easy = [qid for qid in s0_correct_ids
            if all(results[s].get(qid, {}).get("correct", False)
                   for s in ["s1", "s2", "s3", "s4"])]
    print("  S0 correct + ALL Si correct: %d queries (%.1f%% of S0-correct)" % (
        len(easy), len(easy) / total_s0_correct * 100))

# ============================================================
# 4. Rescue Analysis
# ============================================================

def analyze_rescue_detailed(bird_by_id, results):
    """Detailed analysis of queries rescued by pipelines."""
    print("\n" + "=" * 70)
    print("3. RESCUE ANALYSIS (S0 wrong → Si correct)")
    print("=" * 70)

    s0_wrong_ids = [qid for qid in bird_by_id
                    if not results["s0"].get(qid, {}).get("correct", False)]
    total_wrong = len(s0_wrong_ids)
    print("\n  S0 wrong: %d queries" % total_wrong)

    # Per-strategy rescue
    rescue_map = defaultdict(set)  # strategy → set of rescued qids
    for s in ["s1", "s2", "s3", "s4"]:
        for qid in s0_wrong_ids:
            if results[s].get(qid, {}).get("correct", False):
                rescue_map[s].add(qid)

    for s in ["s1", "s2", "s3", "s4"]:
        print("\n  %s rescues %d/%d (%.1f%%)" % (
            s.upper(), len(rescue_map[s]), total_wrong,
            len(rescue_map[s]) / total_wrong * 100))

    # Overlap analysis
    print("\n  Rescue overlap (Jaccard similarity):")
    strats = ["s1", "s2", "s3", "s4"]
    for i, s1 in enumerate(strats):
        for s2 in strats[i+1:]:
            intersection = rescue_map[s1] & rescue_map[s2]
            union = rescue_map[s1] | rescue_map[s2]
            jaccard = len(intersection) / len(union) if union else 0
            print("    %s ∩ %s: %d shared / %d union (J=%.2f)" % (
                s1.upper(), s2.upper(), len(intersection), len(union), jaccard))

    # What S0 error type gets rescued?
    print("\n  S0 error types that get rescued:")
    for s in ["s1", "s2", "s3", "s4"]:
        s0_err_types = Counter()
        for qid in rescue_map[s]:
            r = results["s0"].get(qid, {})
            db_id = bird_by_id[qid]["db_id"]
            db_path = str(DB_DIR / db_id / (db_id + ".sqlite"))
            s0_err_types[classify_error(r, db_path)] += 1
        print("    %s rescues: %s" % (s.upper(), dict(s0_err_types.most_common())))

    # Unique rescues (only one strategy can fix)
    all_rescued = set()
    for s in strats:
        all_rescued |= rescue_map[s]

    print("\n  Unique rescues (only 1 strategy works):")
    for s in strats:
        unique = rescue_map[s] - set().union(*(rescue_map[s2] for s2 in strats if s2 != s))
        print("    %s: %d unique" % (s.upper(), len(unique)))

    print("\n  Total rescuable: %d/%d (%.1f%%)" % (
        len(all_rescued), total_wrong, len(all_rescued) / total_wrong * 100))

    return rescue_map

# ============================================================
# 5. Per-Database Analysis
# ============================================================

def analyze_per_database(bird_by_id, results):
    """Per-database accuracy comparison."""
    print("\n" + "=" * 70)
    print("4. PER-DATABASE ANALYSIS")
    print("=" * 70)

    db_stats = defaultdict(lambda: {s: {"correct": 0, "total": 0}
                                     for s in ["s0", "s1", "s2", "s3", "s4"]})

    for qid, q in bird_by_id.items():
        db_id = q["db_id"]
        for s in ["s0", "s1", "s2", "s3", "s4"]:
            db_stats[db_id][s]["total"] += 1
            if results[s].get(qid, {}).get("correct", False):
                db_stats[db_id][s]["correct"] += 1

    # Sort by S0 accuracy (ascending = hardest first)
    sorted_dbs = sorted(db_stats.items(),
                        key=lambda x: x[1]["s0"]["correct"] / max(x[1]["s0"]["total"], 1))

    print("\n  %-28s %5s" % ("Database", "N"), end="")
    for s in ["s0", "s1", "s2", "s3", "s4"]:
        print(" %6s" % s.upper(), end="")
    print("  Best")
    print("  " + "-" * 80)

    for db_id, stats in sorted_dbs:
        n = stats["s0"]["total"]
        if n < 5:
            continue
        print("  %-28s %5d" % (db_id[:28], n), end="")
        accs = {}
        for s in ["s0", "s1", "s2", "s3", "s4"]:
            acc = stats[s]["correct"] / n * 100
            accs[s] = acc
            print(" %5.1f%%" % acc, end="")
        best = max(accs, key=lambda k: accs[k])
        marker = " ←" if best != "s0" else ""
        print("  %s%s" % (best.upper(), marker))

# ============================================================
# 6. Per-Difficulty Analysis
# ============================================================

def analyze_per_difficulty(bird_by_id, results):
    """Per-difficulty accuracy comparison."""
    print("\n" + "=" * 70)
    print("5. PER-DIFFICULTY ANALYSIS")
    print("=" * 70)

    for diff in ["moderate", "challenging"]:
        qids = [qid for qid, q in bird_by_id.items() if q["difficulty"] == diff]
        print("\n  %s (%d queries):" % (diff.upper(), len(qids)))
        for s in ["s0", "s1", "s2", "s3", "s4"]:
            correct = sum(1 for qid in qids
                          if results[s].get(qid, {}).get("correct", False))
            exec_errs = sum(1 for qid in qids
                            if results[s].get(qid, {}).get("exec_error"))
            print("    %s: %d/%d = %.1f%% (exec_err=%d)" % (
                s.upper(), correct, len(qids), correct / len(qids) * 100, exec_errs))

# ============================================================
# 7. Cross-Pipeline Agreement
# ============================================================

def analyze_agreement(bird_by_id, results):
    """How often do pipelines agree/disagree?"""
    print("\n" + "=" * 70)
    print("6. CROSS-PIPELINE AGREEMENT")
    print("=" * 70)

    patterns = Counter()
    for qid in bird_by_id:
        pattern = tuple(1 if results[s].get(qid, {}).get("correct", False) else 0
                        for s in ["s0", "s1", "s2", "s3", "s4"])
        patterns[pattern] += 1

    total = len(bird_by_id)

    # All agree
    all_correct = patterns.get((1, 1, 1, 1, 1), 0)
    all_wrong = patterns.get((0, 0, 0, 0, 0), 0)
    print("\n  All 5 correct: %d (%.1f%%)" % (all_correct, all_correct / total * 100))
    print("  All 5 wrong:   %d (%.1f%%)" % (all_wrong, all_wrong / total * 100))
    print("  Mixed:         %d (%.1f%%)" % (total - all_correct - all_wrong,
                                             (total - all_correct - all_wrong) / total * 100))

    # Most common patterns
    print("\n  Top 15 correctness patterns (S0,S1,S2,S3,S4):")
    for pattern, count in patterns.most_common(15):
        pct = count / total * 100
        labels = ["S%d=%s" % (i, "✓" if v else "✗") for i, v in enumerate(pattern)]
        print("    %s : %3d (%.1f%%)" % (" ".join(labels), count, pct))

    # Pairwise agreement
    print("\n  Pairwise agreement rates:")
    strats = ["s0", "s1", "s2", "s3", "s4"]
    print("  %-5s" % "", end="")
    for s in strats:
        print(" %6s" % s.upper(), end="")
    print()
    for i, s1 in enumerate(strats):
        print("  %-5s" % s1.upper(), end="")
        for j, s2 in enumerate(strats):
            if j <= i:
                print(" %6s" % "—", end="")
            else:
                agree = sum(1 for qid in bird_by_id
                            if (results[s1].get(qid, {}).get("correct", False) ==
                                results[s2].get(qid, {}).get("correct", False)))
                print(" %5.1f%%" % (agree / total * 100), end="")
        print()

# ============================================================
# 8. Pipeline-Specific Strengths
# ============================================================

def analyze_pipeline_strengths(bird_by_id, results):
    """Identify query characteristics where each pipeline excels."""
    print("\n" + "=" * 70)
    print("7. PIPELINE-SPECIFIC STRENGTHS")
    print("=" * 70)

    # For each pipeline, find queries it uniquely solves
    strats = ["s0", "s1", "s2", "s3", "s4"]
    for s in strats:
        unique_qids = []
        for qid in bird_by_id:
            if results[s].get(qid, {}).get("correct", False):
                if all(not results[s2].get(qid, {}).get("correct", False)
                       for s2 in strats if s2 != s):
                    unique_qids.append(qid)

        if not unique_qids:
            print("\n  %s: no unique wins" % s.upper())
            continue

        print("\n  %s: %d unique wins" % (s.upper(), len(unique_qids)))

        # Database distribution
        db_dist = Counter(bird_by_id[qid]["db_id"] for qid in unique_qids)
        print("    DBs: %s" % dict(db_dist.most_common(5)))

        # Difficulty distribution
        diff_dist = Counter(bird_by_id[qid]["difficulty"] for qid in unique_qids)
        print("    Difficulty: %s" % dict(diff_dist))

        # Show a few example questions
        print("    Examples:")
        for qid in unique_qids[:3]:
            q = bird_by_id[qid]
            question = q["question"][:80]
            print("      [%d] %s (%s)" % (qid, question, q["db_id"]))

# ============================================================
# Main
# ============================================================

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    if model not in MODEL_NAMES:
        print(f"Unknown model: {model}. Choose from: {', '.join(MODEL_NAMES)}")
        sys.exit(1)

    print("=" * 70)
    print(f"PHASE 2 ERROR TAXONOMY ({MODEL_NAMES[model]} S0-S4, mod+chal)")
    print("=" * 70)

    print("\nLoading data...")
    bird_by_id, results, available = load_all(model)
    print(f"  {len(bird_by_id)} queries, strategies: {', '.join(s.upper() for s in available)}")

    if len(available) < 2:
        print(f"\nWARNING: Only S0 available for {model}. Need S1-S4 for full analysis.")
        print("Run the pipelines first: python scripts/phase2_run_pipeline.py %s s1 --subset mod" % model)
        return

    # Run all analyses
    error_counts = analyze_error_types(bird_by_id, results)
    analyze_error_propagation(bird_by_id, results)
    rescue_map = analyze_rescue_detailed(bird_by_id, results)
    analyze_per_database(bird_by_id, results)
    analyze_per_difficulty(bird_by_id, results)
    analyze_agreement(bird_by_id, results)
    analyze_pipeline_strengths(bird_by_id, results)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(bird_by_id)
    for s in available:
        correct = sum(1 for qid in bird_by_id
                      if results[s].get(qid, {}).get("correct", False))
        exec_errs = sum(1 for qid in bird_by_id
                        if results[s].get(qid, {}).get("exec_error"))
        print("  %s: %d/%d = %.1f%% (exec_err=%d)" % (
            s.upper(), correct, total, correct / total * 100, exec_errs))

    oracle = sum(1 for qid in bird_by_id
                 if any(results[s].get(qid, {}).get("correct", False)
                        for s in available))
    print("  Oracle: %d/%d = %.1f%%" % (oracle, total, oracle / total * 100))


if __name__ == "__main__":
    main()
