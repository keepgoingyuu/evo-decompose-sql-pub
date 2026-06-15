#!/bin/bash
# EvoDecomposeSQL - 資料集下載腳本

set -e

DATA_DIR="../data"

download_spider2() {
    echo "📥 下載 Spider 2.0 資料集..."
    mkdir -p "$DATA_DIR/spider2"

    # Spider 2.0 官方連結
    # 注意：需要先到官網申請存取權限
    # https://spider2-sql.github.io/

    echo "⚠️  Spider 2.0 需要先到官網申請存取權限："
    echo "   https://spider2-sql.github.io/"
    echo ""
    echo "請手動下載後放到 $DATA_DIR/spider2/ 目錄"
    echo ""

    # 如果有直接下載連結，可以用：
    # wget -O "$DATA_DIR/spider2/spider2.zip" "https://..."
    # unzip "$DATA_DIR/spider2/spider2.zip" -d "$DATA_DIR/spider2/"
}

download_falcon() {
    echo "📥 下載 Falcon（中文）資料集..."
    mkdir -p "$DATA_DIR/falcon"

    # Falcon 資料集
    # https://www.arxiv.org/pdf/2510.24762

    echo "⚠️  Falcon 資料集請參考論文中的連結："
    echo "   https://www.arxiv.org/pdf/2510.24762"
    echo ""
    echo "請手動下載後放到 $DATA_DIR/falcon/ 目錄"
}

download_spider1() {
    echo "📥 下載 Spider 1.0 資料集（備用）..."
    mkdir -p "$DATA_DIR/spider1"

    # Spider 1.0 是公開的
    wget -O "$DATA_DIR/spider1/spider.zip" "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"

    if [ -f "$DATA_DIR/spider1/spider.zip" ]; then
        unzip "$DATA_DIR/spider1/spider.zip" -d "$DATA_DIR/spider1/"
        rm "$DATA_DIR/spider1/spider.zip"
        echo "✅ Spider 1.0 下載完成"
    else
        echo "❌ Spider 1.0 下載失敗，請手動下載："
        echo "   https://yale-lily.github.io/spider"
    fi
}

download_cspider() {
    echo "📥 下載 CSpider（中文 Spider）資料集..."
    mkdir -p "$DATA_DIR/cspider"

    echo "⚠️  CSpider 請到以下連結下載："
    echo "   https://taolusi.github.io/CSpider-explorer/"
    echo ""
    echo "請手動下載後放到 $DATA_DIR/cspider/ 目錄"
}

# 主程式
case "$1" in
    spider2)
        download_spider2
        ;;
    falcon)
        download_falcon
        ;;
    spider1)
        download_spider1
        ;;
    cspider)
        download_cspider
        ;;
    all)
        download_spider2
        download_falcon
        download_spider1
        download_cspider
        ;;
    *)
        echo "EvoDecomposeSQL 資料集下載工具"
        echo ""
        echo "用法: $0 [dataset]"
        echo ""
        echo "可用資料集:"
        echo "  spider2  - Spider 2.0（主要評估，需申請）"
        echo "  falcon   - Falcon 中文企業 benchmark"
        echo "  spider1  - Spider 1.0（備用）"
        echo "  cspider  - CSpider 中文版"
        echo "  all      - 下載全部"
        echo ""
        echo "範例:"
        echo "  $0 spider1"
        echo "  $0 all"
        ;;
esac
