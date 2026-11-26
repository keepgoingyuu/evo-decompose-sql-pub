"""Strategy-selector analysis on the corrected matrix (IS revision; R3 points 3c and 4).

Part A  BIRD mod+chal, pooled over the five panel models, query-level 5-fold CV:
        reference points, the eleven selectors of the original study (code reused from
        scripts/phase3_v3_n5_selector.py), sentence-embedding gates, and gates that are also
        given the execution outcome of S0 (matched information). Every row reports mean LLM
        calls per query, so methods can be compared at matched cost.
Part B  The same reference points and the gradient-boosting gate on each Spider-family language.
Part C  Transfer: a gate trained on one benchmark/language and applied to another. Folds are
        made on the query index, which the three Spider-family languages share, so a test
        query is never seen in training in any language.

Output: results/rerun_2026_11/analysis/selector_<rule>.json
Usage: venv/bin/python scripts/rerun/revision_selector.py [--rule strict|official]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PANEL, REPO, STRATEGIES, load_queries  # noqa: E402

spec = importlib.util.spec_from_file_location("p3", REPO / "scripts/phase3_v3_n5_selector.py")
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)

AN = REPO / "results/rerun_2026_11/analysis"
SEED = 42
EMB = {"en": "sentence-transformers/all-MiniLM-L6-v2",
       "multi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}


def build(bench: str, rule: str, M: dict) -> tuple[list, list, float]:
    qs = load_queries(bench)
    records, meta = [], []
    for q in qs:
        i = q["idx"]
        feat = p3.extract_features(q["question"], q["evidence"], q["db_path"], q["db_id"])
        votes = sum(M[bench, m, "s0"][i][rule] for m in PANEL)
        meta.append({"q_idx": i, "s0_majority": int(votes > len(PANEL) / 2), "db_id": q["db_id"]})
        for m in PANEL:
            correct = {s: bool(M[bench, m, s][i][rule]) for s in STRATEGIES}
            best = "s0" if correct["s0"] else next((s for s in ["s1", "s3", "s4", "s2"] if correct[s]), "s0")
            records.append({"q_idx": i, "q_text": q["question"], "model": m, "features": feat, "correct": correct,
                            "best": best, "s0_correct": correct["s0"], "s0_wrong": not correct["s0"],
                            "s0_exec_error": int(M[bench, m, "s0"][i]["exec_error"]),
                            "any_rescue": (not correct["s0"]) and any(correct[s] for s in STRATEGIES[1:])})
    s2_calls = float(np.mean([r["call_count"] for m in PANEL for r in M[bench, m, "s2"].values() if r["call_count"]]))
    return records, meta, s2_calls


def calls_pre(picks: dict, s2_calls: float) -> float:
    """Mean calls when the strategy is chosen before any generation."""
    c = {"s0": 1, "s1": 2, "s2": s2_calls, "s3": 2, "s4": 3}
    n = sum(picks.values())
    return round(sum(c[s] * k for s, k in picks.items()) / n, 2)


def refs(records: list, s2_calls: float) -> dict:
    n = len(records)
    pct = lambda f: round(100 * sum(f(r) for r in records) / n, 2)  # noqa: E731
    trig = sum(r["s0_exec_error"] for r in records) / n
    return {"n": n,
            "always_s0": {"acc": pct(lambda r: r["correct"]["s0"]), "calls": 1.0},
            "always_s1": {"acc": pct(lambda r: r["correct"]["s1"]), "calls": 2.0},
            "always_s2": {"acc": pct(lambda r: r["correct"]["s2"]), "calls": round(s2_calls, 2)},
            "always_s3": {"acc": pct(lambda r: r["correct"]["s3"]), "calls": 2.0},
            "always_s4": {"acc": pct(lambda r: r["correct"]["s4"]), "calls": 3.0},
            "exec_gate_s0_s1": {"acc": pct(lambda r: r["correct"]["s1"] if r["s0_exec_error"] else r["correct"]["s0"]),
                                "calls": round(1 + 2 * trig, 2), "trigger_pct": round(100 * trig, 2)},
            "oracle_s0_or_s1": {"acc": pct(lambda r: r["correct"]["s0"] or r["correct"]["s1"])},
            "oracle_all": {"acc": pct(lambda r: any(r["correct"].values()))}}


def with_features(records: list, fn) -> list:
    out = []
    for r in records:
        r2 = dict(r)
        r2["features"] = fn(r)
        out.append(r2)
    return out


def embed(texts: dict, model: str) -> dict:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model)
    keys = sorted(texts)
    embs = st.encode([texts[k] for k in keys], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return {k: embs[j] for j, k in enumerate(keys)}


def gate_fit_apply(train: list, test: list, clf) -> dict:
    """Train an S0-failure gate on `train`, apply to `test` with fixed S1 fallback."""
    from sklearn.metrics import roc_auc_score
    names = sorted(train[0]["features"])
    X = np.array([[r["features"][f] for f in names] for r in train])
    y = np.array([int(not r["s0_correct"]) for r in train])
    Xt = np.array([[r["features"][f] for f in names] for r in test])
    yt = np.array([int(not r["s0_correct"]) for r in test])
    clf = type(clf)(**clf.get_params())
    clf.fit(X, y)
    pred = clf.predict(Xt)
    proba = clf.predict_proba(Xt)[:, list(clf.classes_).index(1)]
    hit = sum(r["correct"]["s1" if p else "s0"] for r, p in zip(test, pred))
    return {"hit": int(hit), "n": len(test), "to_s1": int(pred.sum()),
            "auroc": float(roc_auc_score(yt, proba)) if len(set(yt)) > 1 else float("nan")}



def budget_gate(records: list, folds: list, clf, k: int) -> dict:
    """Matched-cost gate: out-of-fold P(S0 wrong) from `clf`; the k queries with the highest
    probability go to S1, all others keep S0. k is chosen by the caller so that the mean call
    count equals that of the execution gate."""
    names = sorted(records[0]["features"])
    X = np.array([[r["features"][f] for f in names] for r in records])
    y = np.array([int(not r["s0_correct"]) for r in records])
    proba = np.zeros(len(records))
    for train_q, test_q in folds:
        tr, te = p3.split_records_by_q_set(records, train_q, test_q)
        c = type(clf)(**clf.get_params())
        c.fit(X[tr], y[tr])
        proba[te] = c.predict_proba(X[te])[:, list(c.classes_).index(1)]
    order = np.argsort(-proba, kind="stable")
    to_s1 = set(order[:k].tolist())
    hit = sum(r["correct"]["s1" if i in to_s1 else "s0"] for i, r in enumerate(records))
    return {"acc": round(100 * hit / len(records), 2), "to_s1": k}


def main() -> None:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    rule = sys.argv[sys.argv.index("--rule") + 1] if "--rule" in sys.argv else "strict"
    M = defaultdict(dict)
    for line in open(AN / "matrix.jsonl", encoding="utf-8"):
        r = json.loads(line)
        M[r["bench"], r["model"], r["strategy"]][r["idx"]] = r
    data = {b: build(b, rule, M) for b in ["bird", "en", "zh-tw", "ja"]}
    gb = lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=5, random_state=SEED)  # noqa: E731
    out: dict = {"rule": rule}

    # ---------------- Part A: BIRD ----------------
    records, meta, s2c = data["bird"]
    folds = p3.make_query_level_folds(meta, n_splits=5, seed=SEED)
    R = refs(records, s2c)
    base, ceil = R["always_s0"]["acc"], R["oracle_all"]["acc"] - R["always_s0"]["acc"]
    rows = {}

    def add(name, res, info, calls):
        rows[name] = {"acc": round(res["end_to_end_acc_pct"], 2), "delta": round(res["end_to_end_acc_pct"] - base, 2),
                      "rescue_pct": round(100 * (res["end_to_end_acc_pct"] - base) / ceil, 1), "calls": calls,
                      "info": info, "ci": [round(x, 2) for x in res["wilson_ci_pct"]],
                      "auroc": round(res.get("gate_metrics", {}).get("auroc", float("nan")), 3)}

    for nm, fn in [("rule_v1", p3.rule_v1), ("rule_v2", p3.rule_v2)]:
        res = p3.evaluate_rule_selector(records, fn, nm)
        add(nm, res, "pre", calls_pre(res["picks"], s2c))
    trees = {"dt": lambda **k: DecisionTreeClassifier(max_depth=5, random_state=SEED, **k),
             "rf": lambda **k: RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced",
                                                      random_state=SEED, n_jobs=-1, **k)}
    mc = {"dt": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
          "rf": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1),
          "gb": GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=10, random_state=SEED)}
    gates = {"dt": trees["dt"](min_samples_leaf=5), "rf": trees["rf"](), "gb": gb()}
    for k, clf in mc.items():
        res = p3.train_multiclass(records, folds, f"multiclass_{k}", clf)
        add(f"multiclass_{k}", res, "pre", calls_pre(res["picks"], s2c))
    for k, clf in gates.items():
        res = p3.train_binary_gate(records, folds, f"gate_{k}", clf, fallback="s1")
        add(f"gate_{k}", res, "pre", calls_pre(res["picks"], s2c))
    for k, clf in gates.items():
        res = p3.train_cascade(records, folds, f"cascade_{k}", clf)
        add(f"cascade_{k}", res, "pre", calls_pre(res["picks"], s2c))

    emb = embed({r["q_idx"]: r["q_text"] for r in records}, EMB["en"])
    e_only = lambda r: {f"e{j:03d}": float(v) for j, v in enumerate(emb[r["q_idx"]])}  # noqa: E731
    e_hc = lambda r: {**e_only(r), **{f"hc_{k}": v for k, v in r["features"].items()}}  # noqa: E731
    for nm, fn, clf in [("emb_gate_lr", e_only, LogisticRegression(max_iter=2000)), ("emb_gate_gb", e_only, gb()),
                        ("embhc_gate_lr", e_hc, LogisticRegression(max_iter=2000)), ("embhc_gate_gb", e_hc, gb())]:
        res = p3.train_binary_gate(with_features(records, fn), folds, nm, clf, fallback="s1")
        add(nm, res, "pre", calls_pre(res["picks"], s2c))

    # matched information: the gate also sees whether S0's SQL executed (S0 has been run: 1 call;
    # falling back to S1 costs 2 more)
    hc_x = lambda r: {**r["features"], "s0_exec_error": r["s0_exec_error"]}  # noqa: E731
    e_hc_x = lambda r: {**e_hc(r), "s0_exec_error": r["s0_exec_error"]}  # noqa: E731
    for nm, fn, clf in [("xgate_gb", hc_x, gb()), ("xgate_rf", hc_x, gates["rf"]), ("xgate_embhc_gb", e_hc_x, gb())]:
        res = p3.train_binary_gate(with_features(records, fn), folds, nm, clf, fallback="s1")
        to_s1 = res["picks"].get("s1", 0) / res["n"]
        add(nm, res, "post-exec", round(1 + 2 * to_s1, 2))
    # matched cost: every selector held to the execution gate's mean call count
    n_q = len(records)
    k_post = sum(r["s0_exec_error"] for r in records)          # gate: 1 + 2 calls on these, 1 elsewhere
    gate_calls = round(1 + 2 * k_post / n_q, 2)
    k_pre = round((gate_calls - 1) * n_q)                       # pre-generation: S1 costs 2 calls, S0 one
    matched = {}
    for nm, fn, clf, k, info in [("matched_xgate_rf", hc_x, gates["rf"], k_post, "post-exec"),
                                 ("matched_xgate_gb", hc_x, gb(), k_post, "post-exec"),
                                 ("matched_gate_rf", lambda r: r["features"], gates["rf"], k_pre, "pre"),
                                 ("matched_gate_gb", lambda r: r["features"], gb(), k_pre, "pre")]:
        res = budget_gate(with_features(records, fn), folds, clf, k)
        calls = gate_calls if info == "post-exec" else round(1 + k / n_q, 2)
        matched[nm] = {**res, "delta": round(res["acc"] - base, 2), "calls": calls, "info": info}
    out["bird_matched_cost"] = {"gate_calls": gate_calls, "k_post": k_post, "k_pre": k_pre, "selectors": matched}
    print(f"  matched cost ({gate_calls} calls/query):", {k: (v["acc"], v["delta"]) for k, v in matched.items()})
    out["bird"] = {"refs": R, "rescue_ceiling": round(ceil, 2), "selectors": rows,
                   "per_model": {m: refs([r for r in records if r["model"] == m], s2c) for m in PANEL}}

    print(f"== Part A: BIRD pooled n={R['n']} ({rule}) ==")
    for k in ["always_s0", "always_s1", "always_s3", "always_s4", "always_s2", "exec_gate_s0_s1"]:
        v = R[k]
        print(f"  {k:18} acc {v['acc']:6.2f}  Δ {v['acc'] - base:+5.2f}  rescue {100 * (v['acc'] - base) / ceil:5.1f}%  "
              f"calls {v['calls']:.2f}")
    print(f"  oracle S0|S1 {R['oracle_s0_or_s1']['acc']:.2f}   oracle all {R['oracle_all']['acc']:.2f}  (ceiling {ceil:.2f} pp)")
    for k, v in rows.items():
        print(f"  {k:18} acc {v['acc']:6.2f}  Δ {v['delta']:+5.2f}  rescue {v['rescue_pct']:5.1f}%  calls {v['calls']:.2f}  "
              f"[{v['info']}]  AUROC {v['auroc']}")

    # ---------------- Part B: each Spider-family language ----------------
    out["spider"] = {}
    print("\n== Part B: Spider-family, in-language 5-fold CV (gate = GB, 28 features) ==")
    for b in ["en", "zh-tw", "ja"]:
        recs, mt, c2 = data[b]
        fl = p3.make_query_level_folds(mt, n_splits=5, seed=SEED)
        Rb = refs(recs, c2)
        g = p3.train_binary_gate(recs, fl, "gate_gb", gb(), fallback="s1")
        x = p3.train_binary_gate(with_features(recs, hc_x), fl, "xgate_gb", gb(), fallback="s1")
        out["spider"][b] = {"refs": Rb,
                            "gate_gb": {"acc": round(g["end_to_end_acc_pct"], 2), "calls": calls_pre(g["picks"], c2),
                                        "auroc": round(g["gate_metrics"]["auroc"], 3)},
                            "xgate_gb": {"acc": round(x["end_to_end_acc_pct"], 2),
                                         "calls": round(1 + 2 * x["picks"].get("s1", 0) / x["n"], 2)}}
        print(f"  {b:6} S0 {Rb['always_s0']['acc']:.2f} | S1 {Rb['always_s1']['acc']:.2f} | exec-gate "
              f"{Rb['exec_gate_s0_s1']['acc']:.2f} ({Rb['exec_gate_s0_s1']['calls']} calls) | gate_gb "
              f"{g['end_to_end_acc_pct']:.2f} (AUROC {g['gate_metrics']['auroc']:.3f}) | xgate_gb "
              f"{x['end_to_end_acc_pct']:.2f} | oracle {Rb['oracle_all']['acc']:.2f}")

    # ---------------- Part C: transfer ----------------
    print("\n== Part C: transfer of a learned gate (accuracy; Δ vs always-S0 of the test set) ==")
    sp = {b: data[b][0] for b in ["en", "zh-tw", "ja"]}
    memb = {b: embed({r["q_idx"]: r["q_text"] for r in sp[b]}, EMB["multi"]) for b in sp}
    feats = {"hand-crafted": lambda b: (lambda r: r["features"]),
             "multilingual-emb": lambda b: (lambda r: {f"e{j:03d}": float(v) for j, v in enumerate(memb[b][r["q_idx"]])})}
    folds_sp = p3.make_query_level_folds(data["en"][1], n_splits=5, seed=SEED)
    transfer = {}
    for fname, mk in feats.items():
        F = {b: with_features(sp[b], mk(b)) for b in sp}
        for src in ["en", "zh-tw", "ja"]:
            for tgt in ["en", "zh-tw", "ja"]:
                agg = Counter()
                aur = []
                for tr_q, te_q in folds_sp:
                    res = gate_fit_apply([r for r in F[src] if r["q_idx"] in tr_q],
                                         [r for r in F[tgt] if r["q_idx"] in te_q], gb())
                    agg.update({k: res[k] for k in ("hit", "n", "to_s1")})
                    aur.append(res["auroc"])
                s0 = 100 * np.mean([r["correct"]["s0"] for r in sp[tgt]])
                transfer[f"{fname}|{src}->{tgt}"] = {"acc": round(100 * agg["hit"] / agg["n"], 2),
                                                     "delta_vs_s0": round(100 * agg["hit"] / agg["n"] - s0, 2),
                                                     "to_s1_pct": round(100 * agg["to_s1"] / agg["n"], 1),
                                                     "auroc": round(float(np.mean(aur)), 3)}
        print(f"  [{fname}] rows = trained on, columns = tested on (en, zh-tw, ja): Δ vs always-S0 / AUROC")
        for src in ["en", "zh-tw", "ja"]:
            print(f"    {src:6}" + "  ".join(
                f"{transfer[f'{fname}|{src}->{t}']['delta_vs_s0']:+5.2f}/{transfer[f'{fname}|{src}->{t}']['auroc']:.2f}"
                for t in ["en", "zh-tw", "ja"]))
    # cross-benchmark: BIRD -> Spider EN and back (disjoint databases, no fold needed)
    for src, tgt in [("bird", "en"), ("en", "bird")]:
        res = gate_fit_apply(data[src][0], data[tgt][0], gb())
        s0 = 100 * np.mean([r["correct"]["s0"] for r in data[tgt][0]])
        transfer[f"hand-crafted|{src}->{tgt}"] = {"acc": round(100 * res["hit"] / res["n"], 2),
                                                 "delta_vs_s0": round(100 * res["hit"] / res["n"] - s0, 2),
                                                 "to_s1_pct": round(100 * res["to_s1"] / res["n"], 1),
                                                 "auroc": round(res["auroc"], 3)}
        print(f"  [hand-crafted] {src} -> {tgt}: {transfer[f'hand-crafted|{src}->{tgt}']}")
    out["transfer"] = transfer
    for b in ["en", "zh-tw", "ja"]:
        Rb = out["spider"][b]["refs"]
        print(f"  reference {b}: always-S1 Δ {Rb['always_s1']['acc'] - Rb['always_s0']['acc']:+.2f}, "
              f"exec-gate Δ {Rb['exec_gate_s0_s1']['acc'] - Rb['always_s0']['acc']:+.2f}")

    json.dump(out, open(AN / f"selector_{rule}.json", "w"), indent=1, default=float)
    print(f"\n-> {AN / f'selector_{rule}.json'}")


if __name__ == "__main__":
    main()
