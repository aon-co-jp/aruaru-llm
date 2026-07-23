# aruaru-llm インストールスクリプト(Windows / Windows Server 共通)。
#
# 正直な開示: このバイナリは起動時に
# `models/multilingual-e5-small/`(470MB超、Hugging Face配布、MIT)を
# バイナリと同じ作業ディレクトリから相対パスで読み込む。モデル本体は
# ライセンス上の理由からこのインストーラーには同梱されていない――
# 初回起動前に以下のいずれかで必ず取得すること:
#   huggingface-cli download intfloat/multilingual-e5-small `
#     --local-dir "C:\Program Files\aruaru-llm\models\multilingual-e5-small"
# (または https://huggingface.co/intfloat/multilingual-e5-small/tree/main
#  から個別ダウンロードし同ディレクトリに配置する)
#
# 使い方(管理者権限のPowerShellで):
#   Invoke-WebRequest -Uri "https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-windows-x86_64.zip" -OutFile aruaru-llm.zip
#   Expand-Archive aruaru-llm.zip -DestinationPath aruaru-llm
#   cd aruaru-llm
#   .\install.ps1

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$InstallDir = "C:\Program Files\aruaru-llm"
$ServiceName = "AruaruLlm"

Write-Host "==> インストール先: $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "models") | Out-Null

$BinSrc = Join-Path $PSScriptRoot "aruaru-llm.exe"
if (-not (Test-Path $BinSrc)) {
    Write-Error "aruaru-llm.exe が見つかりません($BinSrc)。zipを展開したディレクトリで実行してください。"
    exit 1
}
Copy-Item $BinSrc -Destination $InstallDir -Force

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "==> 既存のWindowsサービスが見つかったため、バイナリのみ更新しました(再起動は行いません)"
    Write-Host "    手動で再起動する場合: Restart-Service $ServiceName"
} else {
    Write-Host "==> Windowsサービスとして登録する場合の手順:"
    Write-Host "      New-Service -Name $ServiceName -BinaryPathName '$InstallDir\aruaru-llm.exe' -DisplayName 'aruaru-llm' -StartupType Automatic"
    Write-Host "      Start-Service $ServiceName"
}

Write-Host ""
Write-Host "==> 完了。起動前に必ずモデル重みを取得してください:"
Write-Host "    huggingface-cli download intfloat/multilingual-e5-small --local-dir `"$InstallDir\models\multilingual-e5-small`""
