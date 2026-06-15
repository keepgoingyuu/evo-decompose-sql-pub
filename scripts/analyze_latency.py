#!/usr/bin/env python3
"""
響應時間 / 效能聚合 (Response Time / Latency analysis).

設計原則 (踩過的坑):
  1. 絕不跨 benchmark 混平均 latency: BIRD(困難) 與 Spider(簡單) 每題時間差 3-5x。
  2. 地端模型 (gemma4/qwen25c7b/arctic): 用 metadata.elapsed_min (wall-clock, 獨佔 GPU 無排隊)。
     - 只取實跑分片 (chunk0/chunk1 與單檔), 排除 elapsed_min=None 的 accuracy 合併檔,
       否則分母灌水、latency 被低估。
  3. 雲端模型 (gpt41/deepseek/gemini): 主指標用 per-query gen_time 中位數 (p50)。
     wall-clock (total_time_minutes) 受 API rate-limit retry 汙染 (e.g. gemini s3 zh-tw=1005min),
     只列為參考、不當主數據。
  4. 同一 benchmark 的多語言 (en/zh-tw/zh-cn) 分開列, 不合併。

輸出: data/results/analysis/latency_metrics.json
用法: python scripts/analyze_latency.py
"""
import json
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCH_NORM = {
    "bird mod+chal": "BIRD", "bird_modchal": "BIRD", "birdmodchal": "BIRD",
    "cspider_zh_tw": "CSpider-TW", "multispider_ja": "MultiSpider-JA",
    "spider_en": "Spider-EN", "multispider_ja_dev": "MultiSpider-JA",
    "multispider ja dev": "MultiSpider-JA",  # 新日文補跑檔 metadata 用空格
}


def norm_bench(b):
    return BENCH_NORM.get(str(b).strip().lower(), str(b))


def analyze_local():
    """地端: elapsed_min wall-clock, 分 (model, strategy, benchmark)."""
    acc = defaultdict(lambda: {"t": 0.0, "n": 0, "calls": [], "acc_n": 0, "acc_w": 0.0})
    skipped = {"none": 0, "nometa": 0}

    files = glob.glob(str(ROOT / "results/q1_2026_04/main/*.json"))
    files += glob.glob(str(ROOT / "data/results/E2_arctic_r1/*.json"))
    for f in files:
        d = json.load(open(f))
        if "metadata" not in d:
            skipped["nometa"] += 1
            continue
        m = d["metadata"]
        n, e = m.get("n_queries"), m.get("elapsed_min")
        if not n or e is None:
            skipped["none"] += 1
            continue
        model = m["model"]
        if model == "arctic-text2sql-r1-7b":
            model = "arctic-r1"
        key = (model, m.get("strategy", "s0"), norm_bench(m.get("benchmark")))
        a = acc[key]
        a["t"] += e * 60.0
        a["n"] += n
        ccs = [r.get("call_count") for r in d.get("results", []) if r.get("call_count")]
        if ccs:
            a["calls"].extend(ccs)
        if m.get("accuracy") is not None:
            a["acc_w"] += m["accuracy"] * n
            a["acc_n"] += n

    out = {}
    for (model, strat, bench), a in acc.items():
        out.setdefault(model, {}).setdefault(strat, {})[bench] = {
            "sec_per_query": round(a["t"] / a["n"], 2),
            "n_queries": a["n"],
            "mean_calls": round(st.mean(a["calls"]), 2) if a["calls"] else None,
            "accuracy": round(a["acc_w"] / a["acc_n"], 4) if a["acc_n"] else None,
            "timing_basis": "wall_clock_elapsed",
        }
    return out, skipped


def analyze_cloud():
    """雲端: per-query gen_time p50 為主, wall-clock 參考, 分 (model, strategy, lang)."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/thesis_2026_02/phase4_*.json"))):
        if "comparison" in f:
            continue
        d = json.load(open(f))
        m = d.get("metadata", {})
        res = d.get("results", [])
        gens = [r.get("gen_time") for r in res if r.get("gen_time") is not None]
        ccs = [r.get("call_count") for r in res if r.get("call_count")]
        if not gens:
            continue
        model, strat, lang = m.get("model"), m.get("strategy"), m.get("lang", "?")
        bench = f"MultiSpider-{lang}"  # phase4 都是 MultiSpider 的多語言版本
        out.setdefault(model, {}).setdefault(strat, {})[bench] = {
            "gen_time_p50": round(st.median(gens), 2),
            "gen_time_mean": round(st.mean(gens), 2),
            "gen_time_p95": round(sorted(gens)[int(len(gens) * 0.95)], 2),
            "n_queries": len(gens),
            "mean_calls": round(st.mean(ccs), 2) if ccs else None,
            "accuracy": round(m["accuracy"], 4) if m.get("accuracy") is not None else None,
            "api_cost_usd_full_set": m.get("api_cost_estimate"),
            "wall_clock_min_ref": m.get("total_time_minutes"),
            "timing_basis": "per_query_gen_time",
        }
    return out


def analyze_cloud_bird(out):
    """雲端 BIRD mod+chal: S0 來自 phase1 baseline (需篩 difficulty mod+chal=609),
    S1-S4 來自 phase2 (已是 BIRD 609). 補進 out[model]['sX']['BIRD'].
    phase1/phase2 是早期 thesis_2026_02 批次, 結構與 phase4 不同 (無 metadata.lang)."""
    base = ROOT / "results/thesis_2026_02"
    models = {"gpt41": "gpt41", "deepseek": "deepseek", "gemini": "gemini"}

    def record(gens, ccs, extra):
        e = {
            "gen_time_p50": round(st.median(gens), 2),
            "gen_time_mean": round(st.mean(gens), 2),
            "gen_time_p95": round(sorted(gens)[int(len(gens) * 0.95)], 2),
            "n_queries": len(gens),
            "mean_calls": round(st.mean(ccs), 2) if ccs else None,
            "timing_basis": "per_query_gen_time",
        }
        e.update(extra)
        return e

    for model, phase1_key in models.items():
        # --- S0: phase1 baseline, 篩 mod+chal ---
        p1 = {"gpt41": "phase1_gpt41_baseline.json",
              "deepseek": "phase1_deepseek_baseline.json",
              "gemini": "phase1_gemini3flash_baseline.json"}[model]
        p1path = base / p1
        if p1path.exists():
            d = json.load(open(p1path))
            res = d.get("results", [])
            gens = [r["gen_time"] for r in res
                    if r.get("difficulty") in ("moderate", "challenging")
                    and r.get("gen_time") is not None]
            if gens:
                # 註: phase1 的 total_time_minutes / api_cost 是全 BIRD(1534)的,
                # 非 mod+chal(609);gen_time 已逐筆篩 mod+chal 故 p50 正確,
                # 但 wall_clock/cost 不可直接除 609,故不在此記成 per-query 成本。
                out.setdefault(model, {}).setdefault("s0", {})["BIRD"] = record(
                    gens, [], {"wall_clock_min_ref_full_bird": d.get("total_time_minutes")})

        # --- S1-S4: phase2 (已是 BIRD 609) ---
        for s in ("s1", "s2", "s3", "s4"):
            p2path = base / f"phase2_{model}_{s}.json"
            if not p2path.exists():
                continue
            d = json.load(open(p2path))
            res = d["results"] if isinstance(d, dict) else d
            gens = [r["gen_time"] for r in res
                    if isinstance(r, dict) and r.get("gen_time") is not None]
            ccs = [r.get("call_count") for r in res
                   if isinstance(r, dict) and r.get("call_count")]
            if gens:
                out.setdefault(model, {}).setdefault(s, {})["BIRD"] = record(
                    gens, ccs, {})
    return out


def main():
    local, skipped = analyze_local()
    cloud = analyze_cloud()
    cloud = analyze_cloud_bird(cloud)
    result = {
        "_meta": {
            "note": "地端用 wall-clock elapsed; 雲端用 per-query gen_time p50 (wall-clock 受 API retry 汙染). 絕不跨 benchmark 混平均.",
            "local_skipped_merged_none_files": skipped["none"],
            "local_skipped_no_metadata": skipped["nometa"],
        },
        "local_models": local,
        "cloud_models": cloud,
    }
    outpath = ROOT / "data/results/analysis/latency_metrics.json"
    outpath.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"寫出: {outpath}")

    # 同時印出地端 BIRD vs Spider 的 S0->S4 倍數作為 sanity check
    print("\n=== 地端 S0->S4 latency (s/query) ===")
    for model in ("qwen25c7b", "gemma4"):
        if model not in local:
            continue
        for bench in ("BIRD", "Spider-EN"):
            row = {s: local[model].get(s, {}).get(bench, {}).get("sec_per_query")
                   for s in ("s0", "s1", "s2", "s3", "s4")}
            if any(v is not None for v in row.values()):
                s0 = row.get("s0")
                mult = f"(S2={row['s2']/s0:.1f}x)" if s0 and row.get("s2") else ""
                print(f"  {model:10} {bench:10} {row} {mult}")

    print("\n=== 雲端 gen_time p50 (s/query) MultiSpider-en ===")
    for model in ("gpt41", "deepseek", "gemini"):
        if model not in cloud:
            continue
        row = {s: cloud[model].get(s, {}).get("MultiSpider-en", {}).get("gen_time_p50")
               for s in ("s0", "s1", "s2", "s3", "s4")}
        print(f"  {model:10} {row}")


if __name__ == "__main__":
    main()
