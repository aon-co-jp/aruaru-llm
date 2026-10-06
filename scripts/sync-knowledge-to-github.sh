#!/bin/bash
# aruaru-llmのモデル非依存の知識(knowledge.rs)をGitHubの公開リポジトリ
# aon-co-jp/open-english の data/knowledge/knowledge.json へ退避する。
# VPSを作り直しても、aruaru-llmが起動時にこのファイル(raw.githubusercontent.com)から
# 知識を自動で取り込んで復元する(knowledge::sync_from_remote)。
# 公開してよい内容(接客技法の言い換え・出典URL付き)だけを扱う前提。個人情報・会話は入れない。
# 作業用の別cloneを使い、本番の作業ツリー(別の変更が残っている)には触れない。
set -euo pipefail
BASE="${ARUARU_LLM_BASE_URL:-http://127.0.0.1:4600}"
WORK="${KNOWLEDGE_PUSH_DIR:-/root/.cache/open-english-knowledge-push}"
REPO="https://github.com/aon-co-jp/open-english.git"

[ -d "$WORK/.git" ] || git clone --depth 1 "$REPO" "$WORK" >/dev/null 2>&1
cd "$WORK"
git pull -q --ff-only origin master
mkdir -p data/knowledge
TMP="$(mktemp)"
curl -fsS --max-time 20 "$BASE/v1/knowledge/search" | jq '.items' > "$TMP"
# 空・壊れた応答では既存を上書きしない(復元元を守る)
[ "$(jq 'length' "$TMP")" -gt 0 ] || { echo "empty knowledge; skip"; rm -f "$TMP"; exit 0; }
mv "$TMP" data/knowledge/knowledge.json
git add -f data/knowledge/knowledge.json
if git diff --cached --quiet; then
  echo "no change"
else
  git -c user.name="aruaru-llm knowledge sync" -c user.email="noreply@aon-co-jp" commit -q -m "data: knowledge snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
  git push -q origin HEAD:master
  echo pushed
fi
