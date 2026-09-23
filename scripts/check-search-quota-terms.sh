#!/bin/bash
# aruaru-llm/scripts/check-search-quota-terms.sh
#
# 毎朝1回、現在使用している検索API各社(SerpApi/Tavily/Exa/Brave/Google)の
# 料金・無料枠ページを巡回し、前回巡回時からの内容変化を検出する
# (2026-09-23新設、ユーザー指示「無料枠の範囲で利用しているGoogle検索や
# APIキーなどの無料枠の仕様変更やルール改正などを毎朝自動クロールで自動で
# 情報収集して」への対応)。
#
# **正直な開示(重要な設計判断)**: このスクリプトが行うのは「前回との
# 差分検出」までであり、ページの意味を理解して`web_search.rs`内の
# `SERPAPI_FREE_MONTHLY_LIMIT`等の定数を自動で書き換えることはしない。
# 実際に今回のセッションでGoogle Custom Search JSON APIの廃止(2027年
# 完全終了)を発見したのはユーザー自身がGoogle検索のAI要約を読んで
# 気づいたケースであり、機械的なテキスト差分検出だけでは「廃止」のような
# 重要な意味変化を正しく解釈できるとは限らない——誤った自動書き換えで
# 実際の無料枠を超過して課金が発生するリスクの方が、人間の確認を挟む
# 手間より大きいと判断した。差分が見つかった場合は
# `data/quota-terms-changes-pending.md`へ記録するに留め、実際の定数調整は
# 次回のClaude Codeセッションで人間が内容を読んでから行う運用とする。
set -uo pipefail

STATE_DIR="${STATE_DIR:-data}"
HASHES_FILE="${STATE_DIR}/quota-terms-hashes.json"
PENDING_FILE="${STATE_DIR}/quota-terms-changes-pending.md"
mkdir -p "$STATE_DIR"

# 巡回対象URLはPythonブロック内の`pages`辞書で定義している
# (現在web_search.rsが実際に使っている検索API各社の料金・無料枠ページ、
# URLを変える場合はそちらを更新すればよい)。

python3 - "$HASHES_FILE" "$PENDING_FILE" <<'PYEOF'
import sys, json, os, re, hashlib, urllib.request, datetime

hashes_file, pending_file = sys.argv[1], sys.argv[2]
pages = {
    "serpapi": "https://serpapi.com/pricing",
    "tavily": "https://www.tavily.com/pricing",
    "exa": "https://exa.ai/docs/admin/pricing",
    "brave": "https://brave.com/search/api/",
    "google_custom_search": "https://developers.google.com/custom-search/v1/overview",
}

try:
    with open(hashes_file, encoding="utf-8") as f:
        prev = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    prev = {}

today = datetime.date.today().isoformat()
changed = []
current = {}

for name, url in pages.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (aruaru-llm quota-terms-checker)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[check-search-quota-terms] WARN: failed to fetch {name} ({url}): {e}")
        continue
    # 素朴なタグ除去(広告・セッショントークン等のノイズを減らすため、
    # scriptタグの中身ごと除去した上でタグのみ除去する)。完全なHTML解析
    # ではないため、内容自体は変わらずレイアウトだけ変わった場合等に
    # 誤検知しうる正直な限界がある。
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    current[name] = {"hash": digest, "checked_at": today, "url": url}
    if name in prev and prev[name]["hash"] != digest:
        changed.append((name, url, prev[name].get("checked_at", "unknown")))
    elif name not in prev:
        print(f"[check-search-quota-terms] {name}: first-time baseline recorded")

with open(hashes_file, "w", encoding="utf-8") as f:
    json.dump(current, f, ensure_ascii=False, indent=2)

if changed:
    with open(pending_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {today} 検出された変更 / Detected changes\n\n")
        for name, url, since in changed:
            f.write(f"- **{name}**: {url} の内容が前回確認({since})から変化しています。無料枠・料金体系の変更が無いか、人間が内容を確認してください。\n")
            f.write(f"  {name}: content changed at {url} since last check ({since}). A human should review whether the free tier / pricing terms actually changed.\n")
    print(f"[check-search-quota-terms] {len(changed)} page(s) changed — see {pending_file}")
else:
    print("[check-search-quota-terms] no changes detected")
PYEOF
