#!/usr/bin/env python3
"""
從 Hugging Face 下載 CSpider + DuSQL 合併資料集

資料集：HONG2525/CSpider_and_DUSQL_sql_create_context
- 包含 CSpider + DuSQL 合併版本
- 31,200 筆訓練資料
- 已經是 SQL create context 格式
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def download_huggingface_dataset():
    """從 Hugging Face 下載 CSpider + DuSQL 合併資料集"""
    print("\n" + "="*80)
    print("📥 從 Hugging Face 下載 CSpider + DuSQL")
    print("="*80)

    print("\n資料集：HONG2525/CSpider_and_DUSQL_sql_create_context")
    print("包含：CSpider + DuSQL 合併版本（簡體中文）")

    # 下載資料集
    print("\n⏬ 下載中...")
    dataset = load_dataset("HONG2525/CSpider_and_DUSQL_sql_create_context")

    train_data = dataset['train']

    print(f"\n✅ 下載完成：{len(train_data)} 筆資料")

    # 顯示範例
    print("\n📝 資料範例（前 3 筆）：")
    for i in range(3):
        item = train_data[i]
        print(f"\n範例 {i+1}:")
        question = item['question']
        answer = item['answer']
        print(f"  問題：{question[:60] if len(question) > 60 else question}")
        print(f"  SQL：{answer[:60] if len(answer) > 60 else answer}")

    # 轉換成原始格式（question + query）
    print("\n🔄 轉換成標準格式...")
    converted_data = []
    for i in tqdm(range(len(train_data)), desc="轉換資料"):
        item = train_data[i]
        converted_item = {
            'question': item['question'],
            'query': item['answer'],
            'context': item.get('context', ''),
        }
        converted_data.append(converted_item)

    # 分割成 train 和 dev（按照 80/20 比例）
    split_point = int(len(converted_data) * 0.8)
    train_split = converted_data[:split_point]
    dev_split = converted_data[split_point:]

    print(f"\n📊 資料分割：")
    print(f"  訓練集：{len(train_split)} 筆")
    print(f"  驗證集：{len(dev_split)} 筆")

    # 儲存到 data/cspider_dusql_merged/ 目錄
    output_dir = os.path.join(DATA_DIR, 'cspider_dusql_merged')
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.json')
    dev_path = os.path.join(output_dir, 'dev.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)

    with open(dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_split, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 儲存完成：")
    print(f"  {train_path}")
    print(f"  {dev_path}")

    return True

def main():
    """主程式"""
    print("\n" + "="*80)
    print("🚀 Hugging Face 資料集下載工具")
    print("="*80)
    print(f"\n專案目錄：{PROJECT_ROOT}")
    print(f"資料目錄：{DATA_DIR}")

    try:
        success = download_huggingface_dataset()

        if success:
            print("\n" + "="*80)
            print("✅ 下載完成！")
            print("="*80)
            print("\n下一步：轉換成繁體中文")
            print("  python3 scripts/convert_to_traditional.py")
            print()
    except Exception as e:
        print(f"\n❌ 下載失敗：{e}")
        print("\n可能原因：")
        print("  1. 需要安裝 datasets 套件：pip3 install datasets")
        print("  2. 網路連線問題")
        print("  3. Hugging Face 存取問題")

if __name__ == '__main__':
    main()
