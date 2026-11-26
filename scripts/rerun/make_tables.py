"""LaTeX table bodies for the revised manuscript, generated from the analysis JSON files so that
no number in the paper is typed by hand.

Output: results/rerun_2026_11/analysis/tables/<name>.tex  (rows only; the manuscript owns the
table environment, caption and notes)
Usage: python3 scripts/rerun/make_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

AN = REPO / "results/rerun_2026_11/analysis"
OUT = AN / "tables"
NAME = {"gemini": "Gemini Flash", "deepseek": "DeepSeek-V4.1-Flash", "gpt41": "GPT-4.1",
        "qwen25c7b": "Qwen2.5-Coder-7B", "gemma4": "Gemma-3n-E4B", "gpt54mini": "GPT-5.4-mini"}
ORDER = ["gemini", "deepseek", "gpt41", "qwen25c7b", "gemma4"]  # by S0 accuracy on BIRD mod+chal
BENCH = {"bird": r"\shortstack[l]{BIRD mod+chal\\\scriptsize EN, $n{=}609$}",
         "en": r"\shortstack[l]{Spider\\\scriptsize EN, $n{=}1{,}034$}",
         "zh-tw": r"\shortstack[l]{CSpider\\\scriptsize ZH-TW, $n{=}1{,}034$}",
         "ja": r"\shortstack[l]{MultiSpider 2.0\\\scriptsize JA, $n{=}1{,}034$}"}
S = ["s0", "s1", "s2", "s3", "s4"]


def sgn(x: float) -> str:
    return f"${'+' if x >= 0 else '-'}{abs(x):.2f}$"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    st = {r: json.load(open(AN / f"stats_{r}.json")) for r in ("strict", "official")}
    main_ = st["strict"]["main"]

    # ---- main table: strict | official ----
    lines = []
    for b in BENCH:
        models = ORDER + (["gpt54mini"] if b == "bird" else [])
        for j, m in enumerate(models):
            cells = []
            for rule in ("strict", "official"):
                v = [main_[f"{b}|{m}|{s}"][rule] for s in S]
                cells += [(r"\textbf{%.2f}" if x == max(v) else "%.2f") % x for x in v]
            head = (r"\multirow{%d}{*}{%s}" % (len(models), BENCH[b])) if j == 0 else ""
            if m == "gpt54mini":
                lines.append(r"\cmidrule(l){2-12}")
            lines.append(f"{head} & {NAME[m]}{r'\tnote{c}' if m == 'gpt54mini' else ''} & " + " & ".join(cells) + r" \\")
        if b != "ja":
            lines.append(r"\midrule")
    (OUT / "main.tex").write_text("\n".join(lines) + "\n")

    # ---- S0 vs S2 per cell ----
    con = {r: {(c["bench"], c["model"], c["strategy"]): c for c in st[r]["contrast"]} for r in st}
    pooled = {r: {(p["bench"], p["strategy"]): p for p in st[r]["pooled"]} for r in st}
    for strat in ("s2", "s1", "s3", "s4"):
        lines = []
        for m in ORDER:
            row = [NAME[m]]
            for b in BENCH:
                for r in ("strict", "official"):
                    c = con[r][b, m, strat]
                    star = r"\rlap{$^{*}$}" if c["p_holm"] < 0.05 else ""
                    row.append(sgn(c["drop"]) + star)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
        row = ["Panel mean"]
        for b in BENCH:
            for r in ("strict", "official"):
                row.append(sgn(pooled[r][b, strat]["drop"]))
        lines.append(" & ".join(row) + r" \\")
        row = [r"\quad 95\% CI"]
        for b in BENCH:
            for r in ("strict", "official"):
                lo, hi = pooled[r][b, strat]["ci"]
                row.append(f"\\scriptsize$[{lo:+.1f},{hi:+.1f}]$")
        lines.append(" & ".join(row) + r" \\")
        (OUT / f"contrast_{strat}.tex").write_text("\n".join(lines) + "\n")

    # ---- cell counts per strategy ----
    lines = []
    for strat, label in [("s1", "$S_1$ schema filtering"), ("s2", "$S_2$ question decomposition"),
                         ("s3", "$S_3$ skeleton-then-fill"), ("s4", "$S_4$ classify-then-generate")]:
        row = [label]
        for r in ("strict", "official"):
            sel = [c for c in st[r]["contrast"] if c["strategy"] == strat and c["model"] in ORDER]
            row += [str(sum(c["drop"] > 0 for c in sel)), str(sum(c["p_holm"] < .05 and c["drop"] > 0 for c in sel)),
                    str(sum(c["p_holm"] < .05 and c["drop"] < 0 for c in sel))]
        lines.append(" & ".join(row) + r" \\")
    (OUT / "cellcounts.tex").write_text("\n".join(lines) + "\n")

    # ---- ablation ----
    ab = json.load(open(AN / "ablation.json"))
    lab = {"v1": "V1 (canonical)", "v2": "V2 (reworded)", "v3": "V3 (few-shot)"}
    for strat in ("s2", "s1"):
        lines = []
        groups = [(m, b) for m in ["gpt41", "deepseek", "qwen25c7b"] for b in ["en", "zh-tw", "ja"]
                  if f"{m}|{b}|{strat}|strict" in ab]
        for g, (m, b) in enumerate(groups):
            d, do = ab[f"{m}|{b}|{strat}|strict"], ab[f"{m}|{b}|{strat}|official"]
            for j, ver in enumerate(["v1", "v2", "v3"]):
                v, vo = d["variants"][ver], do["variants"][ver]
                p = f"{v['p']:.3f}" if v["p"] >= 0.001 else "$<$.001"
                row = [f"{NAME[m]}" if j == 0 else "", {"en": "EN", "zh-tw": "ZH-TW", "ja": "JA"}[b] if j == 0 else "",
                       lab[ver], f"{v['acc']:.2f}", sgn(v["drop"]), f"{v['broken']}/{v['rescued']}", p, sgn(vo["drop"])]
                if strat == "s2":
                    nd = v["no_decomposition"]
                    row += [f"{v['calls']:.2f}", f"{nd[0]:.0f}" + (r"\tnote{d}" if nd[1] < d["n"] else "")]
                lines.append(" & ".join(row) + r" \\")
            if g < len(groups) - 1:
                lines.append(r"\addlinespace")
        (OUT / f"ablation_{strat}.tex").write_text("\n".join(lines) + "\n")
    s0_lines = [f"{NAME[m]} {b}: S0 strict {ab[f'{m}|{b}|s2|strict']['s0']:.2f}, official {ab[f'{m}|{b}|s2|official']['s0']:.2f}"
                for m in ["gpt41", "deepseek", "qwen25c7b"] for b in ["en", "zh-tw", "ja"] if f"{m}|{b}|s2|strict" in ab]
    (OUT / "ablation_s0.txt").write_text("\n".join(s0_lines) + "\n")
    print("\n".join(f"--- {p.name}\n{p.read_text()}" for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
