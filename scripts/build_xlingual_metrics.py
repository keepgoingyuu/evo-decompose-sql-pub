"""Build unified cross-lingual metrics dataset.

Sources:
- Original 4 models EN/ZH-TW : results/thesis_2026_02/phase4_<model>_<strategy>_<lang>.json
- v2.1 Gemma4 / Qwen25C7B    : results/q1_2026_04/main/v21_xlingual_<bench>_<m>_<s>_seed42.json
- v2.1 frontier Japanese     : results/q1_2026_04/main/v21_<gpt41|deepseek|gemini>_japanese_<s>_seed42.json
- v2.1 Arctic-R1             : results/E2_arctic_r1/ (RL anchor; loaded separately, not part of this matrix)

Output schema: {model: {lang: {strategy: accuracy_pct}}}
"""
import json
import re
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent  # repo root

PHASE4_DIR = REPO / 'results/thesis_2026_02'
V21_E1_DIR = REPO / 'results/q1_2026_04/main'
V21_GPT41_JA_DIR = REPO / 'results/q1_2026_04/main'

MODEL_NAME_MAP = {
    'gpt41': 'GPT-4.1',
    'deepseek': 'DeepSeek-V3',
    'gemini': 'Gemini-3-Flash-Preview',
    'qwen': 'Qwen2.5-Coder-7B',
}

LANG_NORMALIZE = {
    'en': 'EN',
    'zh-tw': 'ZH-TW',
    'zh-cn': 'ZH-CN',  # we won't use zh-cn for v2.1 (focus is EN/ZH-TW/JA)
}

V21_BENCH_TO_LANG = {
    'xlingual_spider_en': 'EN',
    'xlingual_cspider_zh_tw': 'ZH-TW',
    'xlingual_multispider_ja': 'JA',
}


def load_phase4_originals() -> dict:
    """Original-4 models from thesis_2026_02 phase4 files."""
    out = {}
    pattern = re.compile(r'phase4_(\w+)_(s\d)_([\w-]+)\.json$')
    for f in PHASE4_DIR.glob('phase4_*.json'):
        m = pattern.search(f.name)
        if not m:
            continue
        raw_model, strategy, raw_lang = m.groups()
        if raw_model not in MODEL_NAME_MAP:
            continue
        if raw_lang not in LANG_NORMALIZE:
            continue
        if raw_lang == 'zh-cn':
            continue  # v2.1 focus is EN/ZH-TW/JA only
        model = MODEL_NAME_MAP[raw_model]
        lang = LANG_NORMALIZE[raw_lang]
        try:
            d = json.load(open(f))
            acc = d['metadata']['accuracy'] * 100
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        out.setdefault(model, {}).setdefault(lang, {})[strategy] = round(acc, 2)
    return out


def load_v21_e1() -> dict:
    """v2.1 Gemma4 + Qwen25C7B from E1 merged."""
    out = {}
    name_map = {'gemma4': 'Gemma-4-E4B', 'qwen25c7b': 'Qwen2.5-Coder-7B'}
    for raw_m, model in name_map.items():
        for bench, lang in V21_BENCH_TO_LANG.items():
            for s in ['s0', 's1', 's2', 's3', 's4']:
                f = V21_E1_DIR / f'v21_{bench}_{raw_m}_{s}_seed42.json'
                if not f.exists():
                    continue
                d = json.load(open(f))
                acc = d['metadata']['accuracy'] * 100
                out.setdefault(model, {}).setdefault(lang, {})[s] = round(acc, 2)
    return out


def load_v21_frontier_ja() -> dict:
    """Frontier-model Japanese runs (GPT-4.1 / DeepSeek-V3 / Gemini): v21_<m>_japanese_<s>_seed42.json."""
    out = {}
    for raw_m in ['gpt41', 'deepseek', 'gemini']:
        model = MODEL_NAME_MAP[raw_m]
        for s in ['s0', 's1', 's2', 's3', 's4']:
            f = V21_GPT41_JA_DIR / f'v21_{raw_m}_japanese_{s}_seed42.json'
            if not f.exists():
                continue
            d = json.load(open(f))
            # try multiple schema possibilities
            meta = d.get('metadata', d)
            acc = meta.get('accuracy')
            if acc is None:
                # fall back to recomputing from per-query records
                r = d.get('results')
                if not r:
                    continue
                acc = sum(1 for x in r if x.get('correct')) / len(r)
            if acc <= 1.0:
                acc *= 100
            out.setdefault(model, {}).setdefault('JA', {})[s] = round(acc, 2)
    return out


def merge(*sources) -> dict:
    out = {}
    for src in sources:
        for model, langs in src.items():
            for lang, strats in langs.items():
                for s, acc in strats.items():
                    out.setdefault(model, {}).setdefault(lang, {})[s] = acc
    return out


def main():
    p4 = load_phase4_originals()
    v21_e1 = load_v21_e1()
    v21_frontier_ja = load_v21_frontier_ja()
    merged = merge(p4, v21_e1, v21_frontier_ja)

    # Coverage report
    print("=" * 70)
    print("Cross-Lingual Coverage Report")
    print("=" * 70)
    all_langs = ['EN', 'ZH-TW', 'JA']
    all_strats = ['s0', 's1', 's2', 's3', 's4']
    print(f"\n{'Model':<28}" + "".join(f"{lang:<10}" for lang in all_langs))
    print("-" * 60)
    for model in sorted(merged.keys()):
        row = f"{model:<28}"
        for lang in all_langs:
            strats = merged[model].get(lang, {})
            n = len(strats)
            mark = '✓' if n == 5 else (f'{n}/5' if n > 0 else '-')
            row += f"{mark:<10}"
        print(row)

    # Accuracy table per language
    for lang in all_langs:
        print(f"\n=== {lang} ===")
        models_in_lang = [m for m in merged if lang in merged[m]]
        if not models_in_lang:
            print("  (no data)")
            continue
        print(f"{'Model':<28}" + "".join(f"{s.upper():>8}" for s in all_strats))
        for m in sorted(models_in_lang, key=lambda x: -merged[x][lang].get('s0', 0)):
            row = f"{m:<28}"
            for s in all_strats:
                v = merged[m][lang].get(s, '--')
                row += f"{v:>8}" if isinstance(v, str) else f"{v:>8.2f}"
            print(row)

    out_path = REPO / 'results/analysis/xlingual_metrics_unified.json'
    json.dump(merged, open(out_path, 'w'), indent=2)
    print(f"\n✓ Saved: {out_path}")


if __name__ == '__main__':
    main()
