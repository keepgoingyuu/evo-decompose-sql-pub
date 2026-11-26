"""Figures 1-5 of the revised manuscript, generated from the analysis JSON files.

For each figure a TikZ/pgfplots body is written to figures/src/, compiled inside an
empty elsarticle page (so fonts and \\linewidth match the manuscript), and cropped to its ink
with PyMuPDF. Output: figures/Figure_<n>.pdf (numbered by order of appearance).

Usage: python3 scripts/rerun/make_figures.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

AN = REPO / "results/rerun_2026_11/analysis"
SRC = REPO / "figures/src"
DST = REPO / "figures"
ORDER = ["gemini", "deepseek", "gpt41", "qwen25c7b", "gemma4"]
SHORT = {"gemini": "Gemini Flash", "deepseek": "DeepSeek-V4.1-F.", "gpt41": "GPT-4.1", "qwen25c7b": "Qwen2.5-C.-7B",
         "gemma4": "Gemma-3n-E4B"}
PREAMBLE = r"""\documentclass[preprint,3p,number]{elsarticle}
\usepackage{amsmath,amssymb,graphicx,xcolor,tikz,pgfplots}
\usetikzlibrary{arrows.meta,positioning}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.17}
\pagestyle{empty}
\begin{document}\thispagestyle{empty}\noindent
"""


def fig_pipeline() -> str:
    return r"""\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[
    box/.style={draw=blue!40!black, rounded corners=2pt, minimum width=16mm, minimum height=7mm,
                align=center, font=\scriptsize, fill=blue!10},
    llm/.style={draw=orange!60!black, rounded corners=2pt, minimum width=16mm, minimum height=7mm,
                align=center, font=\scriptsize, fill=orange!25},
    outbox/.style={draw=green!50!black, rounded corners=2pt, minimum width=12mm, minimum height=7mm,
                align=center, font=\scriptsize, fill=green!25},
    lbl/.style={font=\scriptsize\bfseries, anchor=west},
    note/.style={font=\scriptsize\itshape, anchor=west, gray!40!black},
    arr/.style={->, >=stealth, thick, gray!60!black},
]
\def\cA{0}\def\cB{2.3}\def\cC{4.6}\def\cD{6.9}\def\cE{9.2}\def\cF{11.5}\def\cG{13.8}\def\cH{15.7}
\foreach \y/\name in {0/$S_0$ Direct, -1.2/$S_1$ Schema, -2.4/$S_2$ Decomp., -3.6/$S_3$ Skeleton, -4.8/$S_4$ Classify}{
  \node[lbl] at (-3.1,\y) {\name};
}
% S0
\node[box] (a0) at (\cA,0) {$\mathcal{S}_D, q$}; \node[llm] (b0) at (\cB,0) {LLM\\\textit{gen}}; \node[outbox] (c0) at (\cC,0) {$\hat{Q}$};
\draw[arr] (a0)--(b0); \draw[arr] (b0)--(c0); \node[note] at (\cH,0) {1 call};
% S1
\node[box] (a1) at (\cA,-1.2) {$\mathcal{S}_D, q$}; \node[llm] (b1) at (\cB,-1.2) {LLM\\\textit{filter}};
\node[box] (c1) at (\cC,-1.2) {$\mathcal{S}_D'\subset\mathcal{S}_D$}; \node[llm] (d1) at (\cD,-1.2) {LLM\\\textit{gen}}; \node[outbox] (e1) at (\cE,-1.2) {$\hat{Q}$};
\draw[arr] (a1)--(b1); \draw[arr] (b1)--(c1); \draw[arr] (c1)--(d1); \draw[arr] (d1)--(e1); \node[note] at (\cH,-1.2) {2 calls};
% S2
\node[box] (a2) at (\cA,-2.4) {$\mathcal{S}_D, q$}; \node[llm] (b2) at (\cB,-2.4) {LLM\\\textit{split}};
\node[box] (c2) at (\cC,-2.4) {$q_1,\ldots,q_k$}; \node[llm] (d2) at (\cD,-2.4) {LLM $\times k$\\\textit{sub-SQL}};
\node[box] (e2) at (\cE,-2.4) {$Q_1,\ldots,Q_k$}; \node[llm] (f2) at (\cF,-2.4) {LLM\\\textit{merge}}; \node[outbox] (g2) at (\cG,-2.4) {$\hat{Q}$};
\draw[arr] (a2)--(b2); \draw[arr] (b2)--(c2); \draw[arr] (c2)--(d2); \draw[arr] (d2)--(e2); \draw[arr] (e2)--(f2); \draw[arr] (f2)--(g2);
\node[note] at (\cH,-2.4) {$k{+}2$ calls};
% S3
\node[box] (a3) at (\cA,-3.6) {$\mathcal{S}_D, q$}; \node[llm] (b3) at (\cB,-3.6) {LLM\\\textit{skel.}};
\node[box] (c3) at (\cC,-3.6) {SQL skel.}; \node[llm] (d3) at (\cD,-3.6) {LLM\\\textit{fill}}; \node[outbox] (e3) at (\cE,-3.6) {$\hat{Q}$};
\draw[arr] (a3)--(b3); \draw[arr] (b3)--(c3); \draw[arr] (c3)--(d3); \draw[arr] (d3)--(e3); \node[note] at (\cH,-3.6) {2 calls};
% S4
\node[box] (a4) at (\cA,-4.8) {$\mathcal{S}_D, q$}; \node[llm] (b4) at (\cB,-4.8) {LLM\\\textit{link}};
\node[box] (c4) at (\cC,-4.8) {linking\\analysis}; \node[llm] (d4) at (\cD,-4.8) {LLM\\\textit{classify}};
\node[box] (e4) at (\cE,-4.8) {Easy/Med./\\Hard}; \node[llm] (f4) at (\cF,-4.8) {LLM\\\textit{gen}$_d$}; \node[outbox] (g4) at (\cG,-4.8) {$\hat{Q}$};
\draw[arr] (a4)--(b4); \draw[arr] (b4)--(c4); \draw[arr] (c4)--(d4); \draw[arr] (d4)--(e4); \draw[arr] (e4)--(f4); \draw[arr] (f4)--(g4);
\node[note] at (\cH,-4.8) {3 calls};
\end{tikzpicture}}
"""


def fig_s2(st: dict) -> str:
    pooled = {(p["bench"], p["strategy"]): p for p in st["pooled"]}
    rows = [("bird", "BIRD mod+chal", 1), ("ja", "JA", 2), ("zh-tw", "ZH-TW", 3), ("en", "EN", 4)]
    dots, bars = [], []
    for b, _, y in rows:
        p = pooled[b, "s2"]
        for j, d in enumerate(p["per_model"]):
            dots.append(f"({d:.2f},{y + (0.13 if j % 2 == 0 else -0.13):.2f})")
        lo, hi = p["ci"]
        bars.append(f"\\draw[blue!70!black, line width=1pt] (axis cs:{lo},{y})--(axis cs:{hi},{y});\n"
                    f"\\draw[blue!70!black, line width=1pt] (axis cs:{lo},{y - .1})--(axis cs:{lo},{y + .1});\n"
                    f"\\draw[blue!70!black, line width=1pt] (axis cs:{hi},{y - .1})--(axis cs:{hi},{y + .1});\n"
                    f"\\filldraw[blue!55, draw=blue!70!black, line width=0.6pt] (axis cs:{p['drop']},{y}) circle (2.6pt);\n"
                    f"\\node[font=\\scriptsize, blue!50!black, anchor=south] at (axis cs:{p['drop']},{y + .2}) {{${p['drop']:+.2f}$}};")
    return r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\linewidth, height=0.42\linewidth,
    xmin=-2, xmax=9,
    xlabel={\footnotesize Accuracy drop from $S_0$ to $S_2$ (pp)},
    ytick={1,2,3,4}, yticklabels={""" + ", ".join(r[1] for r in rows) + r"""},
    ymin=0.5, ymax=4.75, xtick={-2,0,2,4,6,8},
    tick label style={font=\footnotesize}, label style={font=\footnotesize},
    axis line style={-, line width=0.4pt}, xmajorgrids, grid style={gray!12, line width=0.4pt}, clip=false,
    tick style={line width=0.4pt},
]
\draw[dashed, gray!55, line width=0.6pt] (axis cs:0,0.55) -- (axis cs:0,4.45);
\node[font=\scriptsize, gray!55!black, anchor=south] at (axis cs:0,4.45) {$S_2{=}S_0$};
\addplot[only marks, mark=*, mark size=1.3pt, color=blue!35!black,
         mark options={fill=blue!35!black, fill opacity=0.55, draw opacity=0.55}]
  coordinates {""" + " ".join(dots) + "};\n" + "\n".join(bars) + "\n\\end{axis}\n\\end{tikzpicture}\n"


def fig_heatmap(st: dict) -> str:
    acc = lambda b, m, s: st["main"][f"{b}|{m}|{s}"]["strict"]  # noqa: E731
    panels = []
    for b, title in [("en", "EN"), ("zh-tw", "ZH-TW"), ("ja", "JA")]:
        rows = "\n".join(f"{j + 1} {i + 1} {acc(b, m, s) - acc(b, m, 's0'):.2f}"
                         for i, m in enumerate(ORDER) for j, s in enumerate(["s1", "s2", "s3", "s4"]))
        yl = "yticklabels={" + ", ".join(SHORT[m] for m in ORDER) + "}" if b == "en" else ""
        panels.append(f"\\nextgroupplot[title={{{title}}}, {yl}]\n\\addplot[matrix plot*, mesh/cols=4, point meta=explicit, "
                      f"nodes near coords] table[meta=z] {{\nx y z\n{rows}\n}};")
    return r"""\pgfplotsset{
  colormap={divergingRB}{rgb255=(178,24,43) rgb255=(247,247,247) rgb255=(33,102,172)},
  langpanel/.style={
    width=0.30\linewidth, height=0.40\linewidth,
    enlarge x limits={abs=0.5}, enlarge y limits={abs=0.5},
    xmin=1, xmax=4, ymin=1, ymax=5, xtick={1,2,3,4}, xticklabels={$S_1$,$S_2$,$S_3$,$S_4$},
    ytick={1,2,3,4,5}, y dir=reverse, point meta min=-9, point meta max=9,
    tick label style={font=\small}, title style={font=\small, yshift=-2pt}, axis on top,
    tick style={line width=0.4pt}, axis line style={line width=0.4pt},
    nodes near coords style={font=\scriptsize, /pgf/number format/fixed, /pgf/number format/precision=1,
                             /pgf/number format/print sign},
    every node near coord/.append style={anchor=center},
  },
}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1, horizontal sep=14pt, ylabels at=edge left, yticklabels at=edge left}, langpanel]
""" + "\n".join(panels) + "\n\\end{groupplot}\n\\end{tikzpicture}\n"


def fig_fit(st: dict) -> str:
    acc = lambda m, s: st["main"][f"bird|{m}|{s}"]["strict"]  # noqa: E731
    ms = list(reversed(ORDER))  # ascending capability
    xs = {"gemma4": "Gemma", "qwen25c7b": "Qwen", "gpt41": "GPT-4.1", "deepseek": "DeepSeek", "gemini": "Gemini"}
    style = {"s1": ("*", "blue!70!black", "blue!55"), "s2": ("square*", "orange!80!black", "orange!70"),
             "s3": ("triangle*", "green!50!black", "green!45"), "s4": ("diamond*", "red!70!black", "red!45")}
    plots = []
    for s, (mk, c, f) in style.items():
        pts = " ".join(f"({xs[m]},{acc(m, s) - acc(m, 's0'):.2f})" for m in ms)
        plots.append(f"\\addplot[only marks, mark={mk}, mark size=2.2pt, color={c}, mark options={{fill={f}}}] coordinates {{{pts}}};\n"
                     f"\\addlegendentry{{$S_{s[1]}$}}")
    caps = "\n".join(f"\\node[font=\\scriptsize, gray!60!black] at (axis cs:{xs[m]},-9.6) {{${'C_M{=}' if i == 0 else ''}{acc(m, 's0'):.1f}$}};"
                     for i, m in enumerate(ms))
    return r"""\begin{tikzpicture}
\begin{axis}[
    width=0.85\linewidth, height=0.46\linewidth,
    xlabel={\small Prompting model (ordered by $S_0$ accuracy $C_M$ on BIRD mod+chal)},
    ylabel={\small Strategy benefit $\Delta_{S_i}(M)$ (pp)},
    ymin=-10.5, ymax=4, ytick={-8,-6,-4,-2,0,2,4},
    symbolic x coords={""" + ",".join(xs[m] for m in ms) + r"""}, xtick=data,
    xticklabel style={font=\scriptsize}, enlarge x limits=0.12,
    grid=major, grid style={gray!18, line width=0.4pt},
    legend columns=4,
    legend style={font=\scriptsize, draw=gray!50, fill=white, at={(0.5,1.04)}, anchor=south, /tikz/every even column/.append style={column sep=4pt}},
    label style={font=\small}, tick label style={font=\scriptsize},
    axis line style={-, line width=0.4pt}, tick style={line width=0.4pt}, clip=false,
]
""" + caps + "\n\\addplot[gray!70, line width=0.6pt, forget plot, sharp plot] coordinates {(Gemma,0) (Gemini,0)};\n" + "\n".join(plots) + "\n\\end{axis}\n\\end{tikzpicture}\n"


def fig_selector(sel: dict) -> str:
    R, S = sel["bird"]["refs"], sel["bird"]["selectors"]
    pre = max(v["acc"] for k, v in S.items() if v["info"] == "pre")
    post = max(v["acc"] for k, v in S.items() if v["info"] == "post-exec")
    bars = [("Always-$S_0$", R["always_s0"]["acc"], "gray!45"), ("Best learned\\\\(before gen.)", pre, "blue!30"),
            ("Best learned\\\\(+exec.\\ outcome)", post, "blue!30"), ("Execution\\\\gate", R["exec_gate_s0_s1"]["acc"], "blue!55"),
            ("Oracle\\\\$\\{S_0,S_1\\}$", R["oracle_s0_or_s1"]["acc"], "orange!45"),
            ("Oracle\\\\$\\{S_0..S_4\\}$", R["oracle_all"]["acc"], "orange!45")]
    plots = "\n".join(f"\\addplot[fill={c}, draw=black!60, line width=0.5pt, bar shift=0pt] coordinates {{({i + 1},{v})}};"
                      for i, (_, v, c) in enumerate(bars))
    return r"""\begin{tikzpicture}
\begin{axis}[
    width=0.9\linewidth, height=0.46\linewidth, ybar, bar width=26pt,
    ymin=0, ymax=60, ytick={0,10,20,30,40,50,60},
    ylabel={\small Pooled execution accuracy (\%)},
    xtick={1,2,3,4,5,6}, xticklabels={""" + ",".join("{" + b[0] + "}" for b in bars) + r"""},
    xticklabel style={font=\scriptsize, align=center}, xmin=0.4, xmax=6.6,
    nodes near coords, nodes near coords style={font=\footnotesize, /pgf/number format/fixed, /pgf/number format/precision=2, /pgf/number format/fixed zerofill},
    label style={font=\small}, tick label style={font=\small},
    ymajorgrids, grid style={gray!15, line width=0.4pt},
    axis line style={-, line width=0.4pt}, tick style={line width=0.4pt},
]
""" + plots + "\n\\end{axis}\n\\end{tikzpicture}\n"


def build(name: str, body: str, out: Path) -> None:
    (SRC / f"{name}.tex").write_text("% Generated by scripts/rerun/make_figures.py - do not edit by hand.\n" + body, encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        Path(td, "f.tex").write_text(PREAMBLE + body + "\\end{document}\n", encoding="utf-8")
        env = dict(os.environ, TEXMFHOME="/tmp/empty_texmf", TEXINPUTS=f"{DST}:")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=td, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(name + "\n" + r.stdout[-1500:])
        doc = fitz.open(Path(td, "f.pdf"))
        page = doc[0]
        rect = fitz.Rect()
        for d in page.get_drawings():
            rect |= d["rect"]
        for blk in page.get_text("blocks"):
            rect |= fitz.Rect(blk[:4])
        rect = (rect + (-2, -2, 2, 2)) & page.rect
        page.set_cropbox(rect)
        doc.save(out)
        print(f"{out.name}: {rect.width:.0f} x {rect.height:.0f} pt")


def main() -> None:
    st = json.load(open(AN / "stats_strict.json"))
    sel = json.load(open(AN / "selector_strict.json"))
    build("fig0_pipeline", fig_pipeline(), DST / "Figure_1.pdf")
    build("fig3_cross_lingual_s2", fig_s2(st), DST / "Figure_2.pdf")
    build("fig4_lang_heatmap", fig_heatmap(st), DST / "Figure_3.pdf")
    build("fig1_a1_fit", fig_fit(st), DST / "Figure_4.pdf")
    build("fig2_selector_rescue", fig_selector(sel), DST / "Figure_5.pdf")


if __name__ == "__main__":
    main()
