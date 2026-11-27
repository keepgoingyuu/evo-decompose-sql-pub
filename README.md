# Which Prompting Strategy Should a Text-to-SQL Deployment Use?

Reproducibility package for the paper:

> **Which Prompting Strategy Should a Text-to-SQL Deployment Use? An Experimental Reproducible Survey Across Models, Languages, and Benchmarks**
> Chih-Yu Lin, Huan-Yu Chen (National Taichung University of Science and Technology)
> Under journal review (revised version, November 2026).

The repository holds the per-query records, the evaluation code and the scripts that
produce every table and figure of the paper from those records.

> **Read this first if you used an earlier version of this repository.**
> An audit carried out in November 2026 found defects in the original evaluation
> harness. The most consequential one removed the leading `WITH` clause of queries
> written as common table expressions, so that they failed to execute; stored SQL was
> also cut to 200 characters. About 51,000 of the 96,000 records were regenerated with
> the repaired harness, and several conclusions of the earlier version changed.
> **The numbers of the paper come only from `results/rerun_2026_11/`.** The older
> result directories are kept for provenance and are described under
> [Superseded material](#superseded-material).

---

## The experimental matrix

| Axis | Values |
|---|---|
| Models (panel of 5) | GPT-4.1, DeepSeek-V4.1-Flash, Gemini 3 Flash Preview, Qwen2.5-Coder-7B, Gemma-3n-E4B |
| Additional models | GPT-5.4-mini (BIRD only), Arctic-Text2SQL-R1 7B (BIRD only) |
| Strategies | S0 direct generation, S1 schema filtering, S2 question decomposition, S3 skeleton-then-fill, S4 classify-then-generate |
| Benchmarks | BIRD dev, moderate + challenging (609), Spider dev EN (1,034), CSpider dev in Traditional Chinese (1,034), MultiSpider JA (1,034) |
| Reference system | MAC-SQL re-run on GPT-4.1, BIRD subset |

Decoding is deterministic (temperature 0, seed 42 where the endpoint accepts a seed).
Model keys used in file names: `gpt41`, `deepseek`, `gemini`, `qwen25c7b`,
`gemma4` (= Gemma-3n-E4B, Ollama tag `gemma3n:e4b`), `gpt54mini`, `arctic`.

---

## What is here

```
results/rerun_2026_11/
├── <bench>__<model>__<strategy>.jsonl   regenerated per-query records (105 files)
├── ablation/                            S1 and S2 template variants (V2, V3) on a 400-query subset
├── ablation_mac/                        Gemma-3n-E4B: S0 and S1 V1/V2/V3 on the same subset, run on a second machine
├── arctic/                              Arctic-Text2SQL-R1 on BIRD, uniform and native prompt
├── worklist.json                        which records were regenerated, and why
├── kept_verdicts.jsonl                  verdicts of the records kept from the original runs, re-scored
├── bird_official_check.json             agreement of our BIRD scoring with the official script
├── spider_official_check.json           agreement of our Spider scoring with the official script
├── summary_<model>.json                 per-model before/after summary of the regeneration
└── analysis/
    ├── matrix.jsonl                     one row per (benchmark, model, strategy, query): 95,820 rows
    ├── stats_strict.json                main table, paired tests, rankings, fits (strict rule)
    ├── stats_official.json              the same under each benchmark's official rule
    ├── selector_strict.json             selector study, including cross-lingual transfer
    ├── selector_official.json
    ├── ablation.json                    template ablation
    ├── ablation_mac.json                Gemma-3n-E4B S1 ablation, with its agreement with the matrix
    ├── macsql.json                      MAC-SQL against single strategies, with call and token counts
    └── tables/                          LaTeX table bodies used in the paper
baselines/MAC-SQL/outputs/modchal609/    MAC-SQL outputs and API trace for the 609 BIRD queries
figures/                                 the five figures (PDF) and their generated TikZ sources
scripts/rerun/                           regeneration and analysis scripts (see below)
src/                                     pipelines S0 to S4, model clients, SQL extraction and scoring
```

### Record format (`results/rerun_2026_11/*.jsonl`)

One JSON object per query. `idx` is the position of the query in the benchmark's
source file (for BIRD, in the moderate + challenging subset).

| Field | Meaning |
|---|---|
| `bench`, `model`, `model_id`, `strategy`, `idx`, `db_id`, `question`, `gold_sql` | identification; `model_id` is the identifier sent to the endpoint |
| `pred_sql`, `error` | final SQL, complete, and its execution error if any |
| `correct` | verdict under the strict rule (ordered equality of result rows) |
| `correct_official` | verdict under the benchmark's official rule |
| `at_generation` | the values of `error`, `correct`, `correct_official`, `legacy_error`, `legacy_correct` computed when the record was generated; superseded by the fields above (see below) |
| `scored` | SQLite version and date of the re-scoring |
| `legacy_pred_sql`, `legacy_error`, `legacy_correct` | what the original extraction routine would have produced from the same response |
| `call_count`, `gen_time` | model calls and wall-clock seconds |
| `steps` | every model call in order: step name, full raw response (`raw`), finish reason, token counts, reported model version, SHA-1 of the prompt |
| `harness` | hash of the harness code that produced the record |

`analysis/matrix.jsonl` merges regenerated records (`src: "new"`) with records kept from
the original runs (`src: "old"`), and carries both verdicts for every row.

### The two scoring rules

* **strict**: the predicted and gold result rows must be equal in the order SQLite returns them.
* **official**: for BIRD, set equality of result rows, as in the official BIRD evaluator;
  for the Spider family, the test-suite execution rule (order matters only when the gold
  query has `ORDER BY`), with `DISTINCT` kept.

Both official rules were checked against the benchmarks' own evaluation scripts:
`bird_official_check.json` (all 19,488 BIRD predictions, 100% agreement) and
`spider_official_check.json` (all 74,627 executable Spider-family predictions of the panel,
regenerated and kept, 100% agreement with `DISTINCT` kept; 97.5% in the script's default mode,
which strips `DISTINCT`).

**One scoring environment.** The verdicts stored at generation time had been computed on two
machines with different SQLite versions, which changed the outcome for a few queries
(floating-point aggregates, slow reference queries). `scripts/rerun/rescore.py` recomputes every
verdict from the stored SQL in one environment (SQLite 3.45.3); the fields `error`, `correct`
and `correct_official` hold the re-scored values and `at_generation` the earlier ones (they
differ for 43 of the 95,820 rows of the matrix).

**Four queries scored as failures for all systems.** The reference query of BIRD mod+chal
positions 205 and 245 runs for about 30 s or longer, and that of Spider-family positions 455
and 456 returns text that is not valid UTF-8; the original harness did not call the models for
them in part or all of the matrix. They are counted as incorrect for every model and strategy
under both rules (`UNSCORABLE` in `scripts/rerun/common.py`).

Both are implemented in `src/utils/sql_utils.py`. The same file contains the repaired
extraction routine (`clean_sql`) and the original one, kept so that the effect of the
defect can be measured.

---

## Reproducing the paper

No model inference is needed. The benchmark databases are needed only for the steps that
execute SQL (see below and [Benchmark data](#benchmark-data-not-included)).

```bash
pip install -r requirements.txt

python scripts/rerun/rescore.py                         # optional: recompute every verdict from the stored SQL
python scripts/rerun/build_matrix.py                    # -> analysis/matrix.jsonl (both rules)
python scripts/rerun/revision_stats.py --rule strict    # -> analysis/stats_strict.json
python scripts/rerun/revision_stats.py --rule official  # -> analysis/stats_official.json
python scripts/rerun/revision_ablation.py               # -> analysis/ablation.json
python scripts/rerun/revision_ablation_mac.py           # -> analysis/ablation_mac.json (needs the databases)
python scripts/rerun/revision_macsql.py                 # -> analysis/macsql.json
python scripts/rerun/revision_selector.py --rule strict     # -> analysis/selector_strict.json
python scripts/rerun/revision_selector.py --rule official   # -> analysis/selector_official.json
python scripts/rerun/make_tables.py                     # -> analysis/tables/*.tex
python scripts/rerun/make_figures.py                    # -> figures/Figure_1..5.pdf (needs pdflatex, PyMuPDF)
```

`revision_selector.py` needs `sentence-transformers` for the embedding-based selectors.
`build_matrix.py` and the scripts after it need the benchmark question files but not the
databases (`rescore.py`, `revision_macsql.py` and the two official-script checks
execute SQL and do need them). Verdicts can depend on the SQLite version in rare cases; the
shipped ones were computed with SQLite 3.45.3.

### Paper item, script, output

| Paper item | Script | Output |
|---|---|---|
| Main accuracy table, both rules | `build_matrix.py`, `revision_stats.py` | `stats_*.json` (`main`), `tables/main.tex` |
| Paired contrasts with S0 per cell (exact McNemar, Holm), panel means with clustered permutation test and bootstrap interval | `revision_stats.py` | `stats_*.json` (`contrast`, `pooled`), `tables/contrast_s*.tex`, `tables/cellcounts.tex` |
| Stage at which S2 fails | `s2_error_stage.py` | printed to standard output |
| Accuracy by number of sub-questions | `revision_stats.py` | `stats_*.json` (`depth`) |
| Template ablation for S1 and S2 | `revision_ablation.py` | `ablation.json`, `tables/ablation_s*.tex` |
| Template ablation for S1, Gemma-3n-E4B rows of Table 8 (second machine; agreement with the matrix) | `revision_ablation_mac.py` | `ablation_mac.json` |
| Ranking concordance across languages (Kendall's W, exact permutation) | `revision_stats.py` | `stats_*.json` |
| Capability fit | `revision_stats.py` | `stats_*.json` |
| Selector study: fixed rules, execution gate, learned selectors, oracle, cross-lingual transfer | `revision_selector.py` | `selector_*.json` |
| Cost in calls and tokens | `revision_stats.py` | `stats_*.json` |
| Arctic-Text2SQL-R1 on BIRD | `arctic_rerun.py` (generation), `build_matrix.py` | `arctic/*.jsonl` |
| MAC-SQL against single strategies | `revision_macsql.py` | `macsql.json` |
| Agreement with the official Spider script | `spider_official_check.py` (needs `exec_eval.py` of test-suite-sql-eval under `baselines/MAC-SQL/evaluation/`) | `spider_official_check.json` |
| Agreement with the official BIRD script | `bird_official_check.py --official <path to BIRD's evaluation.py>` | `bird_official_check.json` |
| Figures 1 to 5 | `make_figures.py` | `figures/` |

### Regenerating records (optional, needs API keys or local models)

`scripts/rerun/build_worklist.py` derives `worklist.json` from the original result files;
`rerun_cells.py` regenerates the listed records, `ablation_cells.py` the template variants
and `arctic_rerun.py` the Arctic runs. Copy `.env.example` to `.env` and fill in the keys.
Hosted endpoints change over time, so regenerated outputs may differ from the stored ones;
each stored record carries the model version reported by the endpoint.

---

## Superseded material

These directories were produced by the original harness. They are kept because part of
their records are still used (the records that the audit showed to be unaffected; see
`worklist.json`) and so that the regeneration can be checked against them. Their stored
SQL is cut to 200 characters and they contain no raw responses. **Do not take accuracy
numbers from them.**

| Path | Content | Status |
|---|---|---|
| `results/thesis_2026_02/`, `results/q1_2026_04/` | original per-query files | partly kept through `matrix.jsonl`, otherwise superseded |
| `results/E2_arctic_r1/` | Arctic-R1 with a 256-token output limit, four benchmarks | superseded by `rerun_2026_11/arctic/` (BIRD only); the Spider-family results are withdrawn |
| `results/ablation_2026_08/` | earlier template and temperature ablations | superseded by `rerun_2026_11/ablation/` |
| `results/analysis/` (`n5_preview/`, `n5_final/`, and the files beside them) | statistics of the original submission | superseded by `rerun_2026_11/analysis/` |
| `scripts/a*.py`, `scripts/phase*.py`, `scripts/run_*.py` and similar | original runners and analyses | superseded by `scripts/rerun/`; `phase3_v3_n5_selector.py` is still imported by `revision_selector.py` |

---

## Benchmark data (not included)

The benchmark questions and databases carry their own licences and are not redistributed.
Download them from the official sources and place them as follows:

| Benchmark | Source | Expected path |
|---|---|---|
| BIRD dev | https://bird-bench.github.io/ | `data/bird/dev_20240627/` (`dev.json`, `dev_databases/`) |
| Spider | https://yale-lily.github.io/spider | `data/spider/spider_data/` (`dev.json`, `spider_data/database/`) |
| CSpider | https://taolusi.github.io/CSpider-explorer/ | `data/cspider/dev_tw.json`, obtained from the CSpider dev file with OpenCC (`s2twp`); see `scripts/convert_to_traditional.py` |
| MultiSpider | https://github.com/Longtao-Hu/MultiSpider | `data/multispider/dataset/multispider/with_english_value/dev_ja.json` |

The paths are defined in `scripts/rerun/common.py`.

---

## Licence

Code and analysis scripts: MIT (see `LICENSE`). Model outputs under `results/` and
`baselines/` are released for research reproducibility. Benchmark data, including the
questions and gold queries quoted inside the records, remain under their original licences.
