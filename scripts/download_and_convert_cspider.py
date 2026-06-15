#!/usr/bin/env python3
"""
下載 CSpider 並轉換成繁體中文

執行方式：
    python scripts/download_and_convert_cspider.py

輸出：
    data/cspider/       - 簡體中文版本
    data/cspider-tw/    - 繁體中文版本
"""

import os
import json
import requests
from tqdm import tqdm
import zipfile

# 設定路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def download_file(url, output_path):
    """下載檔案並顯示進度"""
    print(f"⏬ 下載到：{output_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path

def download_cspider():
    """
    下載 CSpider 資料集

    注意：如果自動下載失敗，請手動下載：
    https://taolusi.github.io/CSpider-explorer/
    """
    print("\n" + "="*80)
    print("📥 下載 CSpider（簡體中文）")
    print("="*80)

    cspider_dir = os.path.join(DATA_DIR, 'cspider')
    os.makedirs(cspider_dir, exist_ok=True)

    # 嘗試從 Hugging Face 下載
    try:
        print("\n嘗試方法 1：從 Hugging Face 下載...")
        from datasets import load_dataset

        dataset = load_dataset("tomsherborne/cspider")

        # 儲存訓練集
        if 'train' in dataset:
            train_path = os.path.join(cspider_dir, 'train.json')
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump([dict(item) for item in dataset['train']], f, ensure_ascii=False, indent=2)
            print(f"✅ 訓練集已儲存：{train_path}")

        # 儲存驗證集
        if 'validation' in dataset:
            dev_path = os.path.join(cspider_dir, 'dev.json')
            with open(dev_path, 'w', encoding='utf-8') as f:
                json.dump([dict(item) for item in dataset['validation']], f, ensure_ascii=False, indent=2)
            print(f"✅ 驗證集已儲存：{dev_path}")

        return True

    except Exception as e:
        print(f"\n⚠️  方法 1 失敗：{e}")
        print("\n" + "="*80)
        print("請手動下載 CSpider：")
        print("="*80)
        print("\n1. 前往：https://taolusi.github.io/CSpider-explorer/")
        print("2. 下載資料集")
        print("   - Google Drive：https://drive.google.com/...")
        print("   - 百度網盤：提取碼 9gzb")
        print(f"3. 解壓縮後放到：{cspider_dir}")
        print("\n解壓後應該有這些檔案：")
        print(f"   {cspider_dir}/train.json")
        print(f"   {cspider_dir}/dev.json")
        print(f"   {cspider_dir}/tables.json")
        print()

        return False

def convert_to_traditional():
    """將 CSpider（簡體）轉換成繁體中文"""
    print("\n" + "="*80)
    print("🔄 簡體轉繁體（OpenCC）")
    print("="*80)

    # 安裝 OpenCC（如果沒有）
    try:
        from opencc import OpenCC
    except ImportError:
        print("\n安裝 OpenCC...")
        os.system("pip install opencc-python-reimplemented")
        from opencc import OpenCC

    # 建立轉換器
    converter = OpenCC('s2t')  # Simplified to Traditional

    # 測試轉換
    test = "查询所有学生的姓名和成绩"
    result = converter.convert(test)
    print(f"\n✅ OpenCC 測試：")
    print(f"   簡體：{test}")
    print(f"   繁體：{result}")

    # 轉換檔案
    cspider_dir = os.path.join(DATA_DIR, 'cspider')
    cspider_tw_dir = os.path.join(DATA_DIR, 'cspider-tw')
    os.makedirs(cspider_tw_dir, exist_ok=True)

    files_to_convert = ['train.json', 'dev.json']

    for filename in files_to_convert:
        input_path = os.path.join(cspider_dir, filename)
        output_path = os.path.join(cspider_tw_dir, filename)

        if not os.path.exists(input_path):
            print(f"\n⚠️  找不到：{input_path}")
            continue

        print(f"\n⏳ 轉換：{filename}")

        # 讀取簡體資料
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 轉換每一筆
        converted_data = []
        for item in tqdm(data, desc="處理中"):
            converted_item = item.copy()

            # 轉換問題（簡→繁）
            if 'question' in item:
                converted_item['question'] = converter.convert(item['question'])
                # 保留原始簡體（方便對比）
                converted_item['question_simplified'] = item['question']

            # SQL 和其他欄位不變
            converted_data.append(converted_item)

        # 儲存繁體版本
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 完成：{output_path}（{len(converted_data)} 筆）")

    # 顯示範例
    print("\n" + "="*80)
    print("📝 轉換範例（前 3 筆）")
    print("="*80)

    dev_tw_path = os.path.join(cspider_tw_dir, 'dev.json')
    if os.path.exists(dev_tw_path):
        with open(dev_tw_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, item in enumerate(data[:3], 1):
            print(f"\n範例 {i}:")
            print(f"  🇨🇳 簡體：{item.get('question_simplified', '')}")
            print(f"  🇹🇼 繁體：{item.get('question', '')}")
            print(f"  SQL: {item.get('query', '')[:60]}...")

    return True

def main():
    """主程式"""
    print("\n" + "="*80)
    print("🚀 CSpider 下載與繁體轉換工具")
    print("="*80)
    print(f"\n專案目錄：{PROJECT_ROOT}")
    print(f"資料目錄：{DATA_DIR}")

    # 步驟 1：下載 CSpider
    success = download_cspider()

    # 步驟 2：檢查檔案是否存在
    cspider_dev = os.path.join(DATA_DIR, 'cspider', 'dev.json')

    if not os.path.exists(cspider_dev):
        print("\n❌ CSpider 資料集不存在")
        print("請先手動下載並解壓縮到 data/cspider/")
        return

    # 步驟 3：轉換成繁體
    convert_to_traditional()

    # 完成
    print("\n" + "="*80)
    print("✅ 全部完成！")
    print("="*80)
    print("\n資料集位置：")
    print(f"  📁 簡體中文：{os.path.join(DATA_DIR, 'cspider')}")
    print(f"  📁 繁體中文：{os.path.join(DATA_DIR, 'cspider-tw')}")
    print("\n下一步：")
    print("  1. 檢查 data/cspider-tw/ 中的檔案")
    print("  2. 上傳到 Google Drive 或 Colab")
    print("  3. 執行 baseline 測試")
    print()

if __name__ == '__main__':
    main()
