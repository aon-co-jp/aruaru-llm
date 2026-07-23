#!/bin/sh
# aruaru-llm インストールスクリプト(AlmaLinux/Ubuntu/Debian/Fedora/RHEL等、
# systemdを使う主要Linuxディストリ共通)。
#
# 正直な開示: このバイナリは起動時に
# `models/multilingual-e5-small/`(470MB超、Hugging Face配布、MIT)を
# バイナリと同じ作業ディレクトリから相対パスで読み込む。モデル本体は
# ライセンス上の理由からこのインストーラーには同梱されていない――
# 初回起動前に以下のいずれかで必ず取得すること:
#   huggingface-cli download intfloat/multilingual-e5-small \
#     --local-dir /etc/aruaru-llm/models/multilingual-e5-small
# (または https://huggingface.co/intfloat/multilingual-e5-small/tree/main
#  から config.json/model.safetensors/sentencepiece.bpe.model/
#  special_tokens_map.json/tokenizer.json/tokenizer_config.json を
#  個別ダウンロードし同ディレクトリに配置する)
#
# 使い方:
#   curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
#   sudo ./install.sh

set -eu

BIN_SRC="$(dirname "$0")/aruaru-llm"
INSTALL_DIR="/usr/local/bin"
DATA_DIR="/etc/aruaru-llm"
SERVICE_FILE="/etc/systemd/system/aruaru-llm.service"

if [ "$(id -u)" -ne 0 ]; then
    echo "root権限で実行してください(例: sudo ./install.sh)" >&2
    exit 1
fi

if [ ! -f "$BIN_SRC" ]; then
    echo "aruaru-llm バイナリが見つかりません($BIN_SRC)。同梱のtar.gzを展開したディレクトリで実行してください。" >&2
    exit 1
fi

echo "==> バイナリを ${INSTALL_DIR}/aruaru-llm へ配置"
install -m 755 "$BIN_SRC" "${INSTALL_DIR}/aruaru-llm"

mkdir -p "${DATA_DIR}/models"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "==> systemdサービスを作成(${SERVICE_FILE})"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=aruaru-llm - 契約不要の独自AI(open-cuda x aruaru-llm SET)
After=network.target

[Service]
Type=simple
WorkingDirectory=${DATA_DIR}
ExecStart=${INSTALL_DIR}/aruaru-llm
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
else
    echo "==> 既存のsystemdサービスが見つかったため上書きしません(${SERVICE_FILE})"
fi

echo "==> 完了。起動前に必ずモデル重みを取得してください:"
echo "    huggingface-cli download intfloat/multilingual-e5-small --local-dir ${DATA_DIR}/models/multilingual-e5-small"
echo "    sudo systemctl enable --now aruaru-llm"
