#!/usr/bin/env python3
"""
一鍵準備所有資料集

這個腳本會：
1. 下載 Spider 1.0（英文）
2. 下載 CSpider（簡體中文）
3. 下載 DuSQL（簡體中文）
4. 使用 OpenCC s2twp 轉換成台灣繁體
5. 產生統計報告

執行：
    python scripts/prepare_all_datasets.py

輸出：
    data/
    ├── spider/         # 英文 Spider 1.0
    ├── cspider/        # 簡體 CSpider
    ├── cspider-tw/     # 繁體 CSpider
    ├── dusql/          # 簡體 DuSQL
    └── dusql-tw/       # 繁體 DuSQL
"""

import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

def run_script(script_name):
    """執行子腳本"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"⚠️  找不到腳本：{script_path}")
        return False

    print(f"\n🔧 執行：{script_name}")
    print("="*80)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 執行失敗：{e}")
        return False

def check_dataset_status():
    """檢查資料集狀態"""
    print("\n" + "="*80)
    print("📊 資料集狀態檢查")
    print("="*80)

    datasets = {
        'Spider 1.0 (英文)': os.path.join(DATA_DIR, 'spider', 'dev.json'),
        'CSpider (簡體)': os.path.join(DATA_DIR, 'cspider', 'dev.json'),
        'CSpider-TW (繁體)': os.path.join(DATA_DIR, 'cspider-tw', 'dev.json'),
        'DuSQL (簡體)': os.path.join(DATA_DIR, 'dusql', 'dev.json'),
        'DuSQL-TW (繁體)': os.path.join(DATA_DIR, 'dusql-tw', 'dev.json'),
    }

    status = {}
    for name, path in datasets.items():
        exists = os.path.exists(path)
        status[name] = exists

        if exists:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ {name}: {len(data)} 筆")
        else:
            print(f"❌ {name}: 不存在")

    return status

def generate_summary():
    """產生資料集摘要"""
    print("\n" + "="*80)
    print("📋 產生資料集摘要")
    print("="*80)

    summary = {
        'datasets': {},
        'statistics': {},
        'conversion_method': 'OpenCC s2twp (Simplified to Traditional with Taiwan phrases)',
        'date': None
    }

    # Spider 1.0
    spider_dev = os.path.join(DATA_DIR, 'spider', 'dev.json')
    if os.path.exists(spider_dev):
        with open(spider_dev, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary['datasets']['spider_1.0'] = {
            'language': 'English',
            'size': len(data),
            'status': 'available',
            'description': '原始 Spider 1.0 資料集（英文）'
        }

    # CSpider-TW
    cspider_tw_dev = os.path.join(DATA_DIR, 'cspider-tw', 'dev.json')
    if os.path.exists(cspider_tw_dev):
        with open(cspider_tw_dev, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary['datasets']['cspider_tw'] = {
            'language': 'Traditional Chinese (Taiwan)',
            'size': len(data),
            'status': 'converted',
            'source': 'CSpider (Simplified Chinese)',
            'description': 'Spider 台灣繁體版（OpenCC s2twp 轉換）'
        }

    # DuSQL-TW
    dusql_tw_dev = os.path.join(DATA_DIR, 'dusql-tw', 'dev.json')
    if os.path.exists(dusql_tw_dev):
        with open(dusql_tw_dev, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary['datasets']['dusql_tw'] = {
            'language': 'Traditional Chinese (Taiwan)',
            'size': len(data),
            'status': 'converted',
            'source': 'DuSQL (Simplified Chinese)',
            'description': '百度 DuSQL 台灣繁體版（原生中文，OpenCC s2twp 轉換）'
        }

    # 儲存摘要
    summary_path = os.path.join(DATA_DIR, 'datasets_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ 摘要已儲存：{summary_path}")
    print("\n摘要內容：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def show_next_steps():
    """顯示下一步指示"""
    print("\n" + "="*80)
    print("🚀 下一步")
    print("="*80)

    print("\n1️⃣ 檢查資料集")
    print("   ls -lh data/*/dev.json")

    print("\n2️⃣ 查看轉換範例")
    print("   head -n 100 data/cspider-tw/dev.json")
    print("   head -n 100 data/dusql-tw/dev.json")

    print("\n3️⃣ 上傳到 Google Drive")
    print("   手動拖曳或使用 Google Drive 桌面版")
    print("   資料夾：")
    print("     - data/cspider-tw/")
    print("     - data/dusql-tw/")

    print("\n4️⃣ 在 Colab 執行 baseline 測試")
    print("   開啟 02_baseline_test.ipynb")
    print("   測試英文 Spider + 繁體 CSpider + 繁體 DuSQL")

    print("\n5️⃣ （選做）上傳到 Hugging Face")
    print("   分享你的繁體中文資料集給社群")
    print()

def main():
    """主程式"""
    print("\n" + "="*80)
    print("🎯 EvoDecomposeSQL - 資料集準備工具")
    print("="*80)
    print(f"\n專案目錄：{PROJECT_ROOT}")
    print(f"資料目錄：{DATA_DIR}")

    # 建立資料目錄
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n這個腳本會自動下載並轉換以下資料集：")
    print("  1. Spider 1.0（英文）")
    print("  2. CSpider（簡體 → 繁體）")
    print("  3. DuSQL（簡體 → 繁體）")

    response = input("\n是否繼續？[Y/n] ")
    if response.lower() == 'n':
        print("取消")
        return

    # 步驟 1: 下載 CSpider
    print("\n" + "="*80)
    print("步驟 1/3: 下載 CSpider")
    print("="*80)
    run_script('download_and_convert_cspider.py')

    # 步驟 2: 下載 DuSQL
    print("\n" + "="*80)
    print("步驟 2/3: 下載 DuSQL")
    print("="*80)
    run_script('download_dusql.py')

    # 步驟 3: 轉換成繁體
    print("\n" + "="*80)
    print("步驟 3/3: 轉換成繁體（OpenCC s2twp）")
    print("="*80)
    run_script('convert_to_traditional.py')

    # 檢查狀態
    status = check_dataset_status()

    # 產生摘要
    generate_summary()

    # 顯示下一步
    show_next_steps()

    # 最終狀態
    print("\n" + "="*80)
    print("✅ 資料集準備完成！")
    print("="*80)

    all_ready = all(status.values())
    if all_ready:
        print("\n🎉 所有資料集都已就緒！")
    else:
        print("\n⚠️  部分資料集需要手動下載")
        print("\n缺少的資料集：")
        for name, exists in status.items():
            if not exists:
                print(f"  ❌ {name}")

        print("\n請依照錯誤訊息中的指示手動下載")
        print("然後重新執行：python scripts/convert_to_traditional.py")

if __name__ == '__main__':
    main()
