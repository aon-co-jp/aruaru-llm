#!/bin/bash
# aruaru-llm/scripts/daily-news-collect.sh
#
# 毎日1回、主要各国のニュースをGET /v1/news/forで取得させ(=既存の
# per-country digest DATABASE `data/news_by_country.json`へ保存させる)、
# その後archive-news-to-github.shの入力(8日超過分のGitHubアーカイブ)へ
# 自然につながるようにする(2026-09-23新設、ユーザー指示「自動で
# メンテナンスの時間とバックグラウンドでインターネットニュースをGithubに
# 自動でDATABASE化するのは...主要な言語を網羅...毎日DATABASE化して」
# への対応)。
#
# 設計: このスクリプトは検索そのものは行わない——`GET /v1/news/for`を
# 叩くだけで、実際のGoogle Custom Search呼び出し・言語別クエリ構築・
# DATABASE保存は`aruaru-llm`本体(`news_geo.rs`)が担う。3時間TTLの
# キャッシュがあるため、同じ国を1日に何度叩いても実際の検索は1回のみ
# (無料枠の節約)。
#
# **正直な開示・対象外の国**: ユーザーが列挙した国のうち「北朝鮮」は
# 意図的に含めていない——自由な報道機関が存在せず、Google検索結果は
# 事実上すべて国外(主に英語圏)からの報道になるため、「北朝鮮の現地
# ニュース」として提示するのは誤解を招く。この判断はユーザーへ別途
# 報告済み(要相談)。
set -uo pipefail

ARUARU_LLM_BASE_URL="${ARUARU_LLM_BASE_URL:-http://127.0.0.1:4600}"

# ユーザー指示(2026-09-23)の列挙に基づく対象国一覧。
# 日本語/英語圏以外は現地のネイティブ言語で検索される(news_geo.rs::news_query_for_country)。
# India/Ukraine/Israelは英語(ユーザー指示により意図的)。北朝鮮は上記の理由で対象外。
COUNTRIES=(
  "Japan" "United States"
  "China" "Taiwan" "South Korea"
  "Philippines" "Cambodia" "Thailand" "Malaysia"
  "United Kingdom" "Germany" "Italy" "France" "Austria" "Switzerland"
  "India" "Russia" "Ukraine" "Israel"
)

echo "[daily-news-collect] collecting news for ${#COUNTRIES[@]} countries via ${ARUARU_LLM_BASE_URL}/v1/news/for"
fail_count=0
for country in "${COUNTRIES[@]}"; do
    encoded=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$country" 2>/dev/null || echo "$country" | sed 's/ /%20/g')
    response=$(curl -fsS "${ARUARU_LLM_BASE_URL}/v1/news/for?country=${encoded}" 2>&1)
    if [ $? -ne 0 ]; then
        echo "[daily-news-collect] WARN: failed to fetch news for '${country}': ${response}"
        fail_count=$((fail_count + 1))
    else
        item_count=$(echo "$response" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('items', [])))" 2>/dev/null || echo "?")
        echo "[daily-news-collect] ${country}: ${item_count} items"
    fi
    sleep 2
done

echo "[daily-news-collect] done (${fail_count}/${#COUNTRIES[@]} countries failed)"
exit 0
