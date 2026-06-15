# N=5 ML Selector 完整評估計畫

**建立日期**：2026-05-09
**負責人**：林志諭
**目的**：補論文 §5.6 目前列為 future work 的「N=5 panel × 5-fold cross-validated full ML selector evaluation」

---

## 1. 概念背景

### 1.1 主任務 vs 子任務

```
        Text-to-SQL 研究
              │
   ┌──────────┴──────────┐
   ▼                     ▼
 主任務             子任務
(LLM 生成 SQL)    (策略選擇 Selector)
   │                     │
   ▼                     ▼
 5 策略 S0-S4         Selector
 指標：EX            指標：Accuracy / Precision /
                          Recall / F1 / AUROC /
                          AUPRC / Confusion Matrix
```

主任務指標 EX 是 Text-to-SQL 領域標準（類比 YOLO 的 mAP），不能改成 precision/recall/F1。
子任務 selector 是真正的 ML 二元/多類分類問題——這裡才是 ML 標準指標的主場。

### 1.2 Selector 的 4 種 Paradigm

| Paradigm | 機制 | ML 任務型態 |
|----------|------|-------------|
| **A: Rule-based** | 寫死的 if-else 規則（v1, v2）| 不訓練 |
| **B: Multi-class ML** | 直接預測 5 類中的最佳策略 | 5-class classification |
| **C: Binary gate** | 先預測「S0 會失敗嗎」，失敗則固定 fallback | binary classification |
| **D: Cascade** | Gate + per-strategy rescue classifiers，依序試 | 1 binary gate + 4 binary rescue |

### 1.3 RL Specialist 為什麼不進 selector panel

Arctic-Text2SQL-R1 是 single-pass RL specialist——只跑 S0、不跑 S1–S4。Selector 的工作是
「在 5 個策略中選最好的」，但 Arctic 只有 1 個策略可選——邏輯不對等。論文 §5.5 line 266
也明確寫「we do not run Arctic-R1 on S1–S4 because the specialist is a single-pass model
by design」。所以本實驗 N=5 panel 不含 Arctic-R1。

### 1.4 為什麼要做這個補強

1. **論文現狀風險**：§5.6 自己寫「we leave the 5-fold cross-validated full-panel ML selector
   evaluation as future work」——Q2 reviewer 可能挑「ML 是不是被刻意避開」
2. **直接回應老師關切**：老師明確要 precision/recall/accuracy/loss，子任務 selector 補做 ML
   指標就是直接回應
3. **論點強化**：論文 headline「規則就夠了」需要 ML 對照組才能站得住——「我評估 ML、發現
   規則不輸」比「我沒測 ML、宣稱規則夠」強很多
4. **可能揭露 over-engineering**：若 ML selector 只比規則多抓 1–3 pp，論文 finding 升級成
   「Text-to-SQL 領域過度依賴 ML selector」——跟主 finding「decomposition 對強模型有害」
   同一精神

---

## 2. 範圍

### 2.1 In scope（這次要做）

- **N=5 prompting models**：GPT-4.1 / DeepSeek-V3 / Gemini Flash / Qwen2.5-Coder-7B / Gemma-4-E4B
- **BIRD mod+chal** 上的 selector 完整評估
- **4 paradigms × 11 selectors**：Rule v1/v2 + Multi-class DT/RF/GB + Gate DT/RF/GB +
  Cascade DT/RF/GB
- **5-fold StratifiedKFold cross-validation**（修 phase3 用 KFold 與論文 §4.3 不一致的 bug）
- **完整指標族**：Accuracy / Precision / Recall / F1 / AUROC / AUPRC / Confusion Matrix /
  Rescue rate / Wilson 95% CI

### 2.2 Out of scope（這次不做）

- 跨 benchmark transfer（selector 訓練於 BIRD、應用到 Spider）—— ch6 limitations 已列
- Multi-seed evaluation —— ch6 limitations 已列
- Cross-lingual cell coverage 補齊 —— 屬於選項 3b，使用者選擇只做 3a
- N=6 (含 Arctic-R1) selector —— 邏輯不對等
- Calibration / reliability diagram —— 邊際效益遞減

### 2.3 已確認的決策

- **不等 N=6**：Arctic-R1 single-pass，不適合進 selector panel
- **修 KFold → StratifiedKFold**：論文 §4.3 line 66 寫的是 stratified，phase3 程式碼用的是
  KFold——這次新版直接修正
- **Stratify on**：S0-correctness 的 query-level label
- **Random seed**：42（與論文 main matrix 一致）
- **CV folding 層次**：在 query 層分（347 queries × 5 folds），不是 1735 pool 層——防 leakage：
  同一 query 在不同模型的 record 不能跨 fold

---

## 3. 輸入資料

### 3.1 已驗證存在的檔案

**Original 4 模型**（`results/thesis_2026_02/`）：
- `phase1_deepseek_baseline.json` + `phase2_deepseek_s{1,2,3,4}.json`
- `phase1_gpt41_baseline.json` + `phase2_gpt41_s{1,2,3,4}.json`
- `phase1_gemini3flash_baseline.json` + `phase2_gemini_s{1,2,3,4}.json`
- `phase1_qwen_baseline.json` + `phase2_qwen_s{1,2,3,4}.json`

**Gemma-4-E4B**（`data/results/E1_gemma_4_e4b/merged/`）：
- `v21_bird_main_gemma4_s{0,1,2,3,4}_seed42.json`

### 3.2 對齊邏輯

複用 `a5_adaptive_selector_v21.py` 已驗證的 question-text alignment：
- Original 4 用 `query_id` 索引、v21 用 `query_idx` 索引
- 透過 question 字串建立映射 → 347 common queries（與論文 §5.6 一致）
- Pooled 樣本：347 queries × 5 models = **1,735 records**

### 3.3 資料完整性檢查（dry-run 必做）

- [ ] 5 模型在 347 queries 上 S0–S4 全有結果
- [ ] 沒有 None / 缺值
- [ ] 1735 records 中 S0-correctness 比例 ≈ 27.6% (Always-S0 baseline)
- [ ] Oracle accuracy ≈ 50.20%（與 a5 已存結果一致）

---

## 4. 實驗設計

### 4.1 Feature 抽取

複用 `phase3_adaptive_selector.py` 的 28 個 query-time features，分 5 組：
1. **Question length**（2 個）：q_length, q_words
2. **Evidence**（3 個）：has_evidence, evidence_length, evidence_words
3. **SQL 複雜度關鍵字**（9 個）：kw_aggregation, kw_groupby, kw_compare, kw_negation,
   kw_join_hint, kw_subquery_hint, kw_temporal, kw_ranking, kw_math
4. **問題結構**（3 個）：has_multiple_conditions, num_question_marks, has_or
5. **Schema 複雜度**（5 個）：num_tables, num_columns, avg_cols_per_table, num_fks,
   schema_complexity
6. **Database one-hot**（8 個 + 1 個小計）：8 個最大 BIRD DB 的 indicator

每個 query 1 組 features → 347 個 unique feature vectors。

### 4.2 Cross-validation 設計

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  分層 key: query 層的 S0-correctness（pooled across N=5 models）
            ※ 對 query q，y_strat = 1 if 多數 model 在 q 上 S0 正確 else 0

  訓練/測試切分：在 347 queries 層次切（每 fold ~70 queries）
  訓練資料：5 fold × 4 train fold × ~277 queries × 5 models = ~5540 training records
  測試資料：每 fold ~70 queries × 5 models = ~350 test records
```

**為什麼 query 層切而非 record 層**：防 leakage。同一個 query 在不同 model 的 record
共享 features，若 record 層切會讓 train/test 看到相同 features。

### 4.3 11 個 Selector

| # | Paradigm | Algorithm | 訓練？ | 任務型態 |
|---|----------|-----------|--------|----------|
| 1 | A: Rule v1 | — | 否 | rule eval |
| 2 | A: Rule v2 | — | 否 | rule eval |
| 3 | B: Multi-class | DT | 是 | 5-class |
| 4 | B: Multi-class | RF (100 trees) | 是 | 5-class |
| 5 | B: Multi-class | GB (50 trees) | 是 | 5-class |
| 6 | C: Binary gate | DT | 是 | binary |
| 7 | C: Binary gate | RF | 是 | binary |
| 8 | C: Binary gate | GB | 是 | binary |
| 9 | D: Cascade | DT (gate + 4 rescue) | 是 | 1 binary + 4 binary |
| 10 | D: Cascade | RF (gate + 4 rescue) | 是 | 1 binary + 4 binary |
| 11 | D: Cascade | GB (gate + 4 rescue) | 是 | 1 binary + 4 binary |

### 4.4 指標計算

#### 主指標（每個 selector 都報，pooled n=1735）

| 指標 | 公式 / 計算 |
|------|-----------|
| **End-to-end accuracy** | selector 選定的策略執行後 EX 正確的 query 比例 |
| **Rescue rate** | $(\text{EX}_{\text{sel}} - \text{EX}_{S_0}) / (\text{EX}_{\text{oracle}} - \text{EX}_{S_0})$ |
| **Rescue pp** | $\text{EX}_{\text{sel}} - \text{EX}_{S_0}$（百分點）|
| **Wilson 95% CI** | 二元比例 CI（end-to-end accuracy 的 CI）|

#### 分類器層次指標（Paradigm B/C/D 的 underlying classifier）

| 指標 | 適用 | 為什麼 |
|------|------|--------|
| **Accuracy** | B/C/D 的 underlying classifier | 標準 |
| **Confusion matrix** | B/C/D | 看混淆模式 |
| **Precision / Recall / F1** | B/C/D, per-class + macro | 老師熟、imbalanced 必看 |
| **AUROC** | C 的 gate（balanced）| ROC 標準 |
| **AUPRC** | D 的 rescue classifiers（imbalanced）| Imbalanced 必報，不報 AUROC 會虛假樂觀 |
| **Feature importance** | RF / GB 的 gate / multi-class | 解釋性 |

#### 特殊處理

- **B Multi-class** 的 imbalance：S0 佔 77%、其他每個 ≤ 13% → 必看 macro F1，不只 weighted F1
- **C Gate** 是 balanced（S0 fail rate ~72%）→ AUROC 適用為主
- **D Rescue** 是 imbalanced（rescue 成功率 13–25%）→ AUPRC 為主

---

## 5. 程式架構

### 5.1 新檔案：`scripts/phase3_v3_n5_selector.py`

```
phase3_v3_n5_selector.py
├── 1. Imports & paths
├── 2. Data loading
│   ├── load_original_4_per_query()      # 抄 a5
│   ├── load_v21_e1_per_query()          # 抄 a5
│   └── align_by_question()              # 抄 a5
├── 3. Feature extraction
│   └── extract_features()               # 抄 phase3
├── 4. Build pooled dataset
│   └── build_n5_dataset()               # 1735 records, 含 (qid, model, features, correct dict, best, s0_correct)
├── 5. CV setup
│   └── make_stratified_folds()          # 347 query-level folds, 防 leakage
├── 6. Run selectors
│   ├── eval_rule_based()                # Paradigm A
│   ├── train_multiclass_ml()            # Paradigm B
│   ├── train_binary_gate()              # Paradigm C
│   └── train_cascade()                  # Paradigm D
├── 7. Compute metrics
│   ├── end_to_end_accuracy()
│   ├── classifier_metrics()             # Acc/P/R/F1/AUROC/AUPRC/CM
│   ├── wilson_ci()
│   └── feature_importance()
├── 8. Save results
│   ├── n5_selector_results.json
│   ├── n5_selector_log.txt
│   └── figures/                         # confusion matrix, ROC, PR curves
└── 9. main()
```

### 5.2 輸出格式

**`data/results/analysis/n5_final/n5_selector_results.json`**：

```json
{
  "meta": {
    "n_models": 5,
    "n_queries": 347,
    "n_pooled_records": 1735,
    "cv_strategy": "5-fold StratifiedKFold (query-level)",
    "stratify_on": "majority_s0_correctness",
    "seed": 42,
    "timestamp": "2026-05-09T..."
  },
  "anchors": {
    "always_s0_pct": 33.49,
    "s0_then_s1_pct": 42.77,
    "oracle_pct": 50.20,
    "rescue_ceiling_pp": 16.71
  },
  "selectors": {
    "rule_v1":      { "end_to_end_acc": ..., "rescue_pp": ..., "rescue_rate_pct": ..., "wilson_ci": [..., ...] },
    "rule_v2":      { ... },
    "multiclass_dt": {
      "end_to_end_acc": ...,
      "rescue_pp": ...,
      "rescue_rate_pct": ...,
      "wilson_ci": [..., ...],
      "classifier_metrics": {
        "accuracy_macro": ...,
        "precision_per_class": {"s0": ..., "s1": ..., ...},
        "recall_per_class": {...},
        "f1_per_class": {...},
        "f1_macro": ...,
        "f1_weighted": ...,
        "confusion_matrix": [[...], ...],
        "feature_importance_top10": [{"name": "kw_aggregation", "imp": 0.12}, ...]
      }
    },
    "multiclass_rf": { ... },
    "multiclass_gb": { ... },
    "gate_dt":      { ..., "auroc": ..., "auprc": ... },
    "gate_rf":      { ... },
    "gate_gb":      { ... },
    "cascade_dt":   {
      ...,
      "gate_metrics": { "auroc": ..., "auprc": ..., "f1": ... },
      "rescue_metrics": {
        "s1": { "auprc": ..., "auroc": ..., "f1": ..., "support": ... },
        "s2": { ... },
        "s3": { ... },
        "s4": { ... }
      }
    },
    "cascade_rf":   { ... },
    "cascade_gb":   { ... }
  },
  "per_model_breakdown": {
    "DeepSeek-V3":    { /* 各 selector 在這個模型上的 end-to-end acc */ },
    ...
  }
}
```

**`figures/`**：
- `confusion_matrix_multiclass_{dt,rf,gb}.png`
- `confusion_matrix_gate_{dt,rf,gb}.png`
- `roc_curve_gate_combined.png`（3 個 algorithms 疊一張）
- `pr_curve_rescue_per_strategy.png`（4 個 rescue × 3 algorithms）
- `feature_importance_gate_rf.png`

**`n5_selector_log.txt`**：
- 跑的時間、seed、輸入檔 hash
- 每個 selector 的 fold-by-fold 數字
- 任何警告（例如 fold 內單一 class）

---

## 6. 預期結果（骨架，跑完填數字）

### 6.1 Anchor 比較表（pooled n=1,735）

```
┌────────────────────────┬──────────┬───────────┬──────────┬──────────────┐
│ Selector               │ Acc (%)  │ Rescue pp │ Wilson CI│ Rescue rate% │
├────────────────────────┼──────────┼───────────┼──────────┼──────────────┤
│ Always-S0 (anchor)     │  33.49   │   0.00    │ [..,..]  │     0%       │
│ S0→S1 fallback (anchor)│  42.77   │   9.28    │ [..,..]  │    55.5%     │
│ Oracle (ceiling)       │  50.20   │  16.71    │ [..,..]  │   100%       │
├────────────────────────┼──────────┼───────────┼──────────┼──────────────┤
│ Rule v1                │   ?      │    ?      │ [..,..]  │     ?        │
│ Rule v2                │   ?      │    ?      │ [..,..]  │     ?        │
│ Multi-class DT         │   ?      │    ?      │ [..,..]  │     ?        │
│ Multi-class RF         │   ?      │    ?      │ [..,..]  │     ?        │
│ Multi-class GB         │   ?      │    ?      │ [..,..]  │     ?        │
│ Gate DT                │   ?      │    ?      │ [..,..]  │     ?        │
│ Gate RF                │   ?      │    ?      │ [..,..]  │     ?        │
│ Gate GB                │   ?      │    ?      │ [..,..]  │     ?        │
│ Cascade DT             │   ?      │    ?      │ [..,..]  │     ?        │
│ Cascade RF             │   ?      │    ?      │ [..,..]  │     ?        │
│ Cascade GB             │   ?      │    ?      │ [..,..]  │     ?        │
└────────────────────────┴──────────┴───────────┴──────────┴──────────────┘
```

### 6.2 三種可能 outcome 與論文意涵

| 結果情境 | 論文 §5.6 改寫方向 |
|---------|-------------------|
| **ML > 規則 ≥ 5pp** | "ML selector 把 rescue rate 從 55% 提到 X%；ML 訓練值得部署成本" |
| **ML ≈ 規則 ±2pp** | "ML 邊際提升小，規則 + S0→S1 fallback 是 production sweet spot" ⭐ 與 headline 同精神 |
| **ML < 規則** | "ML 在當前 feature set 下無法超越簡單規則——指出 feature engineering 不足，留 future work" |

三種情境都可寫；最有可能是中間（基於 single-model exploratory 的 13–25% 範圍推估）。

---

## 7. 執行步驟

### Step 0: 寫計畫文件 ⭐ 本文件
**狀態**：✅ 寫完
**輸出**：`scripts/n5_selector_plan.md`（本文件）

### Step 1: 寫程式（不執行）
**輸出**：`scripts/phase3_v3_n5_selector.py`
**檢查項**：
- [ ] Data loading 抄自 a5（已驗證對齊邏輯）
- [ ] Feature extraction 抄自 phase3（28 features 已驗證）
- [ ] StratifiedKFold（不是 KFold）
- [ ] Query-level folding（不是 record-level，防 leakage）
- [ ] 11 個 selector 全部實作
- [ ] 完整指標族計算
- [ ] JSON 輸出格式對齊 §5.2 規格
- [ ] 視覺化用 matplotlib，存 PNG

### Step 2: Dry-run 小規模驗證
**做法**：跑 1 fold × Multi-class RF（一個 selector）
**檢查項**：
- [ ] 1735 records 載入無誤
- [ ] StratifiedKFold 切出來的 train/test 大小合理
- [ ] sklearn 沒有 warning（class imbalance / single class in fold 等）
- [ ] 輸出指標數值合理範圍（accuracy 0–100%、AUPRC ≥ random baseline 等）

### Step 3: 跑完整實驗
**指令**：`python scripts/phase3_v3_n5_selector.py --output-dir data/results/analysis/n5_final/`
**預估時間**：CPU 上 5–15 分鐘（11 selectors × 5 folds × 1735 records，RF/GB 是主要瓶頸）

### Step 4: 整理結果與視覺化
**輸出**：JSON + 8 張 PNG figures + log

### Step 5: 寫論文 §5.6 重寫草稿
**輸出**：`thesis/journal_paper/notes/sec56_rewrite_draft.md`
**內容**：把現有 §5.6 line 233 的 "future work" 段落改寫成正式結果段落

### Step 6: 整合進論文
**手動操作**：使用者 review 草稿後，搬進 `chapters/ch5_experiments.tex`

---

## 8. 失敗模式與應對

| 風險 | 機率 | 應對 |
|------|------|------|
| StratifiedKFold 在 rescue 子集（n=70 per strategy）抓不到正例 | 中 | 改用 `RepeatedStratifiedKFold` 或 5→3 folds |
| AUPRC 在 single-positive fold 報 nan | 中 | catch、跳過該 fold、或用整體 cross_val_predict 替代 |
| Multi-class 全猜 S0 → all-S0 baseline | 高 | 這是 imbalance 的真實表現，不需"修"，照實報告 |
| Cascade 訓練時間長（5 折 × gate + 4 rescue × 3 algo） | 低 | RF n_jobs=-1，預估仍 < 15 分鐘 |
| 結果跟 a5 已報數字不一致 | 致命 | 立即停止、debug 對齊邏輯 |
| sklearn 版本差異 | 低 | requirements.txt 鎖定 |

---

## 9. 跑完後論文需更新的位置

- **§4.5 Statistical Instruments**：補一段「Selector evaluation metrics」，列 AUROC/AUPRC/F1
- **§5.6 Adaptive Selector**：line 233 的 "ML evaluation deferred to future work" 整段重寫
  - 原本：「earlier exploratory phase ... not directly comparable ... future work」
  - 改成：「N=5 panel × 5-fold StratifiedKFold ML selector evaluation」+ 結果表 + 結論
- **§6.3 Limitations**：移除「Selector ML evaluation」這項 limitation（如果有）
- **§6.4 Future Work**：第 (3) 項 "production deployment evaluation" 保留，但可加註「offline ML
  selector ceiling 已建立」

---

## 10. Audit trail

每次跑都記下：
- Git commit hash（執行時的）
- 輸入 JSON 檔的 file size + mtime
- random_state = 42
- sklearn version
- 完整 stdout 存到 `n5_selector_log.txt`

跑完後 archive 到 `data/results/analysis/n5_final/run_YYYYMMDD_HHMMSS/` 下。

---

## 附錄 A：與既有 selector script 的差異

| 項目 | `phase3_adaptive_selector.py` | `a5_adaptive_selector_v21.py` | **`phase3_v3_n5_selector.py`** (新) |
|------|------------------------------|------------------------------|--------------------------------------|
| 模型數 | 1 (single-model) | 5 (preview, 不訓練) | **5 (full ML training)** |
| CV | KFold ❌ | 不訓練 | **StratifiedKFold ✅** |
| ML training | 是（exploratory）| 否（deferred to N=6）| **是（最終版）** |
| 指標 | Accuracy + 部分 P/R | 只算 oracle | **完整族（含 AUROC/AUPRC/F1）** |
| 視覺化 | 否 | 否 | **是（CM/ROC/PR curves）** |
| 防 leakage | record-level CV ⚠️ | N/A | **query-level CV ✅** |
| 對應論文章節 | 未進論文 | §5.6 anchor 數字 | **§5.6 ML 結果（重寫後）** |
