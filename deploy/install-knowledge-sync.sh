#!/bin/bash
# 知識のGitHub退避タイマーを新しいVPSへ入れる(お引越し用、何度実行しても同じ結果)。
# Install the knowledge-to-GitHub timer on a new VPS (idempotent).
# 前提 / Prerequisites: git, jq, curl; ~/.git-credentials にGitHub PAT(open-englishへpushできるもの);
# /root/aruaru-llm にこのリポジトリ。
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x scripts/sync-knowledge-to-github.sh
cp deploy/knowledge-sync-push.service deploy/knowledge-sync-push.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now knowledge-sync-push.timer
systemctl list-timers | grep knowledge || true
