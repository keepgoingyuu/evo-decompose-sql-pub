#!/usr/bin/env python3
"""
使用 OpenCC 將簡體中文資料集轉換成台灣繁體中文

支援轉換：
- CSpider
- DuSQL
- 其他簡體中文 Text-to-SQL 資料集

使用 s2twp 模式：
- s2t: Simplified to Traditional (基本字元轉換)
- s2twp: Simplified to Traditional (Taiwan) with Phrases
  → 包含台灣慣用語轉換（字段→欄位, 文件→檔案 等）
"""

import os
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def install_opencc():
    """安裝 OpenCC（如果沒有）"""
    try:
        from opencc import OpenCC
        return True
    except ImportError:
        print("\n📦 安裝 OpenCC...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'opencc-python-reimplemented'])
        from opencc import OpenCC
        return True

def test_opencc():
    """測試 OpenCC 轉換"""
    from opencc import OpenCC

    # 測試不同模式
    modes = {
        's2t': OpenCC('s2t'),      # 基本轉換
        's2twp': OpenCC('s2twp'),  # 台灣慣用語
    }

    test_cases = [
        "查询所有学生的姓名和成绩",
        "数据库字段名称",
        "文件系统",
        "软件配置",
    ]

    print("\n" + "="*80)
    print("🧪 OpenCC 轉換測試")
    print("="*80)

    for test in test_cases:
        print(f"\n簡體：{test}")
        for mode_name, converter in modes.items():
            result = converter.convert(test)
            print(f"  {mode_name}: {result}")

    print("\n✅ 推薦使用 s2twp（包含台灣慣用語轉換）")

def convert_dataset(input_path, output_path, dataset_name, converter):
    """
    轉換單個資料集檔案

    Args:
        input_path: 簡體資料集路徑
        output_path: 繁體資料集輸出路徑
        dataset_name: 資料集名稱（用於顯示）
        converter: OpenCC 轉換器
    """
    if not os.path.exists(input_path):
        print(f"⚠️  找不到：{input_path}")
        return False

    print(f"\n⏳ 轉換：{dataset_name}")
    print(f"   輸入：{input_path}")
    print(f"   輸出：{output_path}")

    # 讀取簡體資料
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 轉換每一筆
    converted_data = []
    for item in tqdm(data, desc=f"處理 {dataset_name}"):
        converted_item = item.copy()

        # 轉換問題文字
        if 'question' in item:
            converted_item['question'] = converter.convert(item['question'])
            # 保留原始簡體（方便對比）
            converted_item['question_simplified'] = item['question']

        # 轉換其他中文欄位（如果有）
        if 'question_toks' in item:  # CSpider 特有
            converted_item['question_toks'] = [
                converter.convert(tok) for tok in item['question_toks']
            ]

        # SQL 和其他欄位保持不變
        converted_data.append(converted_item)

    # 建立輸出目錄
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 儲存繁體版本
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成：{len(converted_data)} 筆")

    return True

def convert_all_datasets():
    """轉換所有支援的資料集"""
    from opencc import OpenCC

    # 使用 s2twp（台灣慣用語）
    converter = OpenCC('s2twp')

    print("\n" + "="*80)
    print("🔄 簡體轉繁體（OpenCC s2twp）")
    print("="*80)

    datasets_to_convert = []

    # CSpider
    cspider_dir = os.path.join(DATA_DIR, 'cspider')
    cspider_tw_dir = os.path.join(DATA_DIR, 'cspider-tw')
    if os.path.exists(cspider_dir):
        datasets_to_convert.append({
            'name': 'CSpider',
            'files': [
                (os.path.join(cspider_dir, 'train.json'),
                 os.path.join(cspider_tw_dir, 'train.json')),
                (os.path.join(cspider_dir, 'dev.json'),
                 os.path.join(cspider_tw_dir, 'dev.json')),
            ]
        })

    # DuSQL
    dusql_dir = os.path.join(DATA_DIR, 'dusql')
    dusql_tw_dir = os.path.join(DATA_DIR, 'dusql-tw')
    if os.path.exists(dusql_dir):
        datasets_to_convert.append({
            'name': 'DuSQL',
            'files': [
                (os.path.join(dusql_dir, 'train.json'),
                 os.path.join(dusql_tw_dir, 'train.json')),
                (os.path.join(dusql_dir, 'dev.json'),
                 os.path.join(dusql_tw_dir, 'dev.json')),
            ]
        })

    # CSpider + DuSQL 合併版本（從 Hugging Face）
    merged_dir = os.path.join(DATA_DIR, 'cspider_dusql_merged')
    merged_tw_dir = os.path.join(DATA_DIR, 'cspider_dusql_merged-tw')
    if os.path.exists(merged_dir):
        datasets_to_convert.append({
            'name': 'CSpider+DuSQL Merged',
            'files': [
                (os.path.join(merged_dir, 'train.json'),
                 os.path.join(merged_tw_dir, 'train.json')),
                (os.path.join(merged_dir, 'dev.json'),
                 os.path.join(merged_tw_dir, 'dev.json')),
            ]
        })

    if not datasets_to_convert:
        print("\n⚠️  沒有找到可轉換的資料集")
        print("請先下載：")
        print("  - CSpider: python scripts/download_and_convert_cspider.py")
        print("  - DuSQL: python scripts/download_dusql.py")
        return False

    # 轉換所有資料集
    total_converted = 0
    for dataset in datasets_to_convert:
        print(f"\n{'='*80}")
        print(f"📁 {dataset['name']}")
        print(f"{'='*80}")

        for input_path, output_path in dataset['files']:
            if convert_dataset(
                input_path,
                output_path,
                f"{dataset['name']} - {os.path.basename(input_path)}",
                converter
            ):
                total_converted += 1

    return total_converted > 0

def show_conversion_examples():
    """顯示轉換範例"""
    print("\n" + "="*80)
    print("📝 轉換範例")
    print("="*80)

    # CSpider-TW
    cspider_tw_dev = os.path.join(DATA_DIR, 'cspider-tw', 'dev.json')
    if os.path.exists(cspider_tw_dev):
        print("\n【CSpider-TW】")
        with open(cspider_tw_dev, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, item in enumerate(data[:2], 1):
            print(f"\n範例 {i}:")
            print(f"  🇨🇳 簡體：{item.get('question_simplified', '')}")
            print(f"  🇹🇼 繁體：{item.get('question', '')}")

    # DuSQL-TW
    dusql_tw_dev = os.path.join(DATA_DIR, 'dusql-tw', 'dev.json')
    if os.path.exists(dusql_tw_dev):
        print("\n【DuSQL-TW】")
        with open(dusql_tw_dev, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, item in enumerate(data[:2], 1):
            print(f"\n範例 {i}:")
            print(f"  🇨🇳 簡體：{item.get('question_simplified', '')}")
            print(f"  🇹🇼 繁體：{item.get('question', '')}")

    # CSpider+DuSQL Merged-TW
    merged_tw_dev = os.path.join(DATA_DIR, 'cspider_dusql_merged-tw', 'dev.json')
    if os.path.exists(merged_tw_dev):
        print("\n【CSpider+DuSQL Merged-TW】")
        with open(merged_tw_dev, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, item in enumerate(data[:2], 1):
            print(f"\n範例 {i}:")
            print(f"  🇨🇳 簡體：{item.get('question_simplified', '')}")
            print(f"  🇹🇼 繁體：{item.get('question', '')}")

def main():
    """主程式"""
    print("\n" + "="*80)
    print("🚀 簡體轉繁體轉換工具")
    print("="*80)
    print(f"\n專案目錄：{PROJECT_ROOT}")
    print(f"資料目錄：{DATA_DIR}")

    # 安裝 OpenCC
    install_opencc()

    # 測試 OpenCC
    test_opencc()

    # 轉換所有資料集
    success = convert_all_datasets()

    if success:
        # 顯示範例
        show_conversion_examples()

        print("\n" + "="*80)
        print("✅ 全部轉換完成！")
        print("="*80)
        print("\n繁體資料集位置：")
        if os.path.exists(os.path.join(DATA_DIR, 'cspider-tw')):
            print(f"  📁 CSpider-TW: {os.path.join(DATA_DIR, 'cspider-tw')}")
        if os.path.exists(os.path.join(DATA_DIR, 'dusql-tw')):
            print(f"  📁 DuSQL-TW: {os.path.join(DATA_DIR, 'dusql-tw')}")
        print("\n下一步：")
        print("  1. 檢查轉換結果")
        print("  2. 上傳到 Google Drive")
        print("  3. 在 Colab 執行 baseline 測試")
        print()
    else:
        print("\n❌ 轉換失敗")
        print("請確認已下載簡體資料集")

if __name__ == '__main__':
    main()
