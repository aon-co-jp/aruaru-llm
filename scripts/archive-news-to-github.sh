#!/bin/bash
# aruaru-llm/scripts/archive-news-to-github.sh
#
# 8日以上前になったニュースをaruaru-llm本体(常時稼働するRustサーバー)側の
# ローカルDATABASEから追い出し(`POST /v1/news/prune-archive`)、生成された
# Markdownをopen-englishリポジトリの`NEWS-TITLE-README.md`へ追記して
# GitHubへpushする(2026-09-23新設、ユーザー指示「インターネットニュースを
# 自動収集してGithubの無料のDATABASEを使って自動保存する」への対応)。
#
# 意図的な設計: このcommit/pushの操作は、常時稼働するaruaru-llm本体プロセス
# ではなく、VPS上のこの独立したスクリプト(systemdタイマー経由で日次実行)が
# 行う——サーバープロセス自体にGitHub書き込み資格情報を持たせるリスクを
# 避けるため。GitHub認証は、このVPSに既に設定済みのgit credential store
# (fine-grained PAT)をそのまま利用する(新規の資格情報配置は不要)。
#
# 冪等性: 追い出すべき古いニュースが無ければ`prune-archive`は0件を返し、
# pending fileも空のままなので、このスクリプトは何もコミットせず終了する
# (無駄なコミットを作らない)。
set -euo pipefail

ARUARU_LLM_DIR="${ARUARU_LLM_DIR:-/root/aruaru-llm}"
OPEN_ENGLISH_DIR="${OPEN_ENGLISH_DIR:-/root/easy-web.tokyo/open-english}"
ARUARU_LLM_BASE_URL="${ARUARU_LLM_BASE_URL:-http://127.0.0.1:4600}"
PENDING_FILE="${ARUARU_LLM_DIR}/data/news-archive-pending.md"
ARCHIVE_FILE="${OPEN_ENGLISH_DIR}/NEWS-TITLE-README.md"

echo "[archive-news-to-github] pruning stale news via ${ARUARU_LLM_BASE_URL}/v1/news/prune-archive"
pruned_response=$(curl -fsS -X POST "${ARUARU_LLM_BASE_URL}/v1/news/prune-archive" || echo '{"pruned":0}')
echo "[archive-news-to-github] prune response: ${pruned_response}"

if [ ! -s "${PENDING_FILE}" ]; then
    echo "[archive-news-to-github] no pending archive content — nothing to push"
    exit 0
fi

if [ ! -f "${ARCHIVE_FILE}" ]; then
    echo "[archive-news-to-github] ERROR: ${ARCHIVE_FILE} does not exist (expected to already be tracked in the open-english repo)" >&2
    exit 1
fi

cat "${PENDING_FILE}" >> "${ARCHIVE_FILE}"

cd "${OPEN_ENGLISH_DIR}"
git add NEWS-TITLE-README.md
if git diff --cached --quiet; then
    echo "[archive-news-to-github] no net change to NEWS-TITLE-README.md — skipping commit"
else
    git commit -q -m "news-archive: auto-archive stale news entries ($(date -u +%Y-%m-%d))

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
    git push -q
    echo "[archive-news-to-github] pushed archive update to GitHub"
fi

# push成功後にのみpendingファイルを空にする(push失敗時は次回再送できるよう残す)。
: > "${PENDING_FILE}"
echo "[archive-news-to-github] done"
