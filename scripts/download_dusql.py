#!/usr/bin/env python3
"""
下載 DuSQL 資料集

DuSQL 是百度發布的原生中文 Text-to-SQL 資料集（2020 EMNLP）
- 200 個資料庫
- 23,797 個問答對
- 原生中文（非翻譯）
- 包含複雜計算邏輯

官網：https://aclanthology.org/2020.emnlp-main.562/
GitHub：https://github.com/luge-ai/luge-ai/blob/master/semantic-parsing.md
"""

import os
import json
import requests
from tqdm import tqdm
import zipfile
import tarfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def download_file(url, output_path):
    """下載檔案並顯示進度"""
    print(f"⏬ 下載：{os.path.basename(output_path)}")
    response = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path

def download_dusql():
    """下載 DuSQL 資料集"""
    print("\n" + "="*80)
    print("📥 下載 DuSQL（原生簡體中文）")
    print("="*80)

    dusql_dir = os.path.join(DATA_DIR, 'dusql')
    os.makedirs(dusql_dir, exist_ok=True)

    # DuSQL 下載連結（從 GitHub 或其他來源）
    # 注意：實際連結可能需要更新
    DUSQL_URLS = {
        'train': 'https://dataset-bj.cdn.bcebos.com/qianyan/DuSQL_train.json',
        'dev': 'https://dataset-bj.cdn.bcebos.com/qianyan/DuSQL_dev.json',
        'tables': 'https://dataset-bj.cdn.bcebos.com/qianyan/DuSQL_tables.json'
    }

    print("\n嘗試從百度雲下載 DuSQL...")

    success = True
    for name, url in DUSQL_URLS.items():
        output_path = os.path.join(dusql_dir, f'{name}.json')

        if os.path.exists(output_path):
            print(f"✅ {name}.json 已存在，跳過")
            continue

        try:
            download_file(url, output_path)
            print(f"✅ {name}.json 下載完成")
        except Exception as e:
            print(f"❌ {name}.json 下載失敗：{e}")
            success = False

    if not success:
        print("\n" + "="*80)
        print("⚠️  自動下載失敗，請手動下載：")
        print("="*80)
        print("\n方法 1：從官方 GitHub")
        print("  https://github.com/luge-ai/luge-ai/blob/master/semantic-parsing.md")
        print("\n方法 2：從百度雲盤")
        print("  連結：https://pan.baidu.com/...")
        print("\n方法 3：從 Hugging Face（如果有）")
        print("  huggingface.co/datasets/dusql")
        print(f"\n下載後放到：{dusql_dir}")
        print("  需要的檔案：")
        print("    - train.json")
        print("    - dev.json")
        print("    - tables.json")
        print()

        return False

    # 檢查檔案
    print("\n" + "="*80)
    print("📊 DuSQL 資料集資訊")
    print("="*80)

    train_path = os.path.join(dusql_dir, 'train.json')
    dev_path = os.path.join(dusql_dir, 'dev.json')

    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"\n✅ 訓練集：{len(train_data)} 筆")

    if os.path.exists(dev_path):
        with open(dev_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        print(f"✅ 驗證集：{len(dev_data)} 筆")

    # 顯示範例
    if os.path.exists(dev_path):
        print("\n📝 DuSQL 範例（前 2 筆）：")
        for i, item in enumerate(dev_data[:2], 1):
            print(f"\n範例 {i}:")
            print(f"  問題：{item.get('question', '')}")
            print(f"  SQL：{item.get('query', '')[:60]}...")
            print(f"  資料庫：{item.get('db_id', '')}")

    return True

def main():
    """主程式"""
    print("\n" + "="*80)
    print("🚀 DuSQL 下載工具")
    print("="*80)
    print(f"\n專案目錄：{PROJECT_ROOT}")
    print(f"資料目錄：{DATA_DIR}")

    success = download_dusql()

    if success:
        print("\n" + "="*80)
        print("✅ DuSQL 下載完成！")
        print("="*80)
        print(f"\n資料夾：{os.path.join(DATA_DIR, 'dusql')}")
        print("\n下一步：")
        print("  python scripts/convert_to_traditional.py")
        print()
    else:
        print("\n❌ 請手動下載 DuSQL")

if __name__ == '__main__':
    main()
