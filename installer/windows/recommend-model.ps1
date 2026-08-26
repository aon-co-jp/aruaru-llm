# インストール中にハードウェアを検出し、推奨LLMサイズ・特徴を日英併記で
# 提示した上で、「もっと大きいモデルにしますか?」と質問して選択に応じて
# モデルを自動ダウンロードするスクリプト(2026-08-26新設、ユーザー指示
# 「インストール途中で、推薦のLLMを提示してLLMの特徴も英語と日本語で
# 提示してもう一つ大きなLLMにしますか?などの日本語と英語のメッセージ後に
# 選択してLLMをインストール可能として」への対応)。
#
# 正直な開示: モデル重み自体は既存の`aruaru-llm`のHTTP API
# (`GET /v1/recommend`・`POST /v1/recommend-and-download`・
# `POST /v1/download-larger`)を呼ぶだけで、新しいダウンロード経路を
# 追加実装したわけではない——このスクリプトは「インストーラーの中で
# 選択できるようにする」という導線を新設しただけ。失敗しても
# インストール自体は止めない(可用性優先、既存の設計方針を踏襲)。
param(
    [Parameter(Mandatory = $true)]
    [string]$AppDir
)

$ErrorActionPreference = "Stop"
$exePath = Join-Path $AppDir "aruaru-llm.exe"
if (-not (Test-Path $exePath)) {
    exit 0
}

# 既に起動中の常用インスタンス(既定ポート4600)と衝突しないよう、
# インストール中だけ使う一時ポートで起動する。
$tempPort = 47600
$env:ARUARU_LLM_BIND = "127.0.0.1:$tempPort"
$base = "http://127.0.0.1:$tempPort"

try {
    $proc = Start-Process -FilePath $exePath -WindowStyle Hidden -PassThru

    # 起動待ち(モデル埋め込み層の読み込み等で数秒かかることがある、最大20秒)。
    $healthy = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $null = Invoke-RestMethod -Uri "$base/healthz" -TimeoutSec 2
            $healthy = $true
            break
        } catch {
            # まだ起動していない、リトライ。
        }
    }
    if (-not $healthy) {
        Write-Output "recommend-model: aruaru-llm did not become healthy in time, skipping model recommendation (this is not fatal)."
        if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        exit 0
    }

    $rec = Invoke-RestMethod -Uri "$base/v1/recommend" -TimeoutSec 10
    $catalog = Invoke-RestMethod -Uri "$base/v1/models/catalog" -TimeoutSec 10
    $entries = $catalog.models
    $recommendedEntry = $entries | Where-Object { $_.id -eq $rec.recommended_model_id } | Select-Object -First 1
    $sortedByLargest = $entries | Sort-Object approx_size_mb
    $recommendedIndex = [array]::IndexOf(($sortedByLargest.id), $rec.recommended_model_id)
    $largerEntry = $null
    if ($recommendedIndex -ge 0 -and $recommendedIndex -lt ($sortedByLargest.Count - 1)) {
        $largerEntry = $sortedByLargest[$recommendedIndex + 1]
    }

    Add-Type -AssemblyName System.Windows.Forms

    $gpuLine = if ($rec.hardware.gpu_name) { $rec.hardware.gpu_name } else { "CPU only / CPUのみ(GPU未検出)" }
    $recName = if ($recommendedEntry) { $recommendedEntry.display_name_en } else { $rec.recommended_model_id }
    $recNameJa = if ($recommendedEntry) { $recommendedEntry.display_name_ja } else { $rec.recommended_model_id }
    $recSize = if ($recommendedEntry) { "$($recommendedEntry.approx_size_mb) MB" } else { "?" }

    $msg = @"
Detected hardware: $gpuLine

Recommended model for your device: $recName ($recSize)
$($rec.disclosure_ja)

Click "Yes" to install this recommended model now.
"@
    if ($largerEntry) {
        $msg += @"


A larger model is also available: $($largerEntry.display_name_en) ($($largerEntry.approx_size_mb) MB) —
generally higher quality but slower and uses more memory.
Click "No" instead to install this LARGER model.
"@
    }
    $msg += @"


------------------------------------------------------------------
検出したハードウェア: $gpuLine

お使いの端末への推奨モデル: $recNameJa ($recSize)
$($rec.disclosure_ja)

「はい」でこの推奨モデルを今すぐインストールします。
"@
    if ($largerEntry) {
        $msg += @"

もう一つ大きいモデルもあります: $($largerEntry.display_name_ja) ($($largerEntry.approx_size_mb) MB) ——
一般に品質は高くなりますが、動作は遅く、メモリ使用量も増えます。
「いいえ」を選ぶと、代わりにこちらの大きいモデルをインストールします。
"@
    }
    $msg += "`n`n(""Cancel"" / 「キャンセル」でモデルのダウンロードをスキップします。後でアプリ内の「Recommend LLM / おすすめLLM」ボタンからいつでも実行できます。)"

    $buttons = if ($largerEntry) { [System.Windows.Forms.MessageBoxButtons]::YesNoCancel } else { [System.Windows.Forms.MessageBoxButtons]::YesNoCancel }
    $result = [System.Windows.Forms.MessageBox]::Show($msg, "aruaru-llm — Recommend a model / おすすめのモデルを選択", $buttons, [System.Windows.Forms.MessageBoxIcon]::Information)

    if ($result -eq [System.Windows.Forms.DialogResult]::Yes) {
        Write-Output "recommend-model: installing recommended model $($rec.recommended_model_id)..."
        $null = Invoke-RestMethod -Uri "$base/v1/recommend-and-download" -Method Post -TimeoutSec 600
    } elseif ($result -eq [System.Windows.Forms.DialogResult]::No -and $largerEntry) {
        Write-Output "recommend-model: installing recommended model $($rec.recommended_model_id) then stepping up to larger model $($largerEntry.id)..."
        $null = Invoke-RestMethod -Uri "$base/v1/recommend-and-download" -Method Post -TimeoutSec 600
        $null = Invoke-RestMethod -Uri "$base/v1/download-larger" -Method Post -TimeoutSec 600
    } else {
        Write-Output "recommend-model: skipped by user (Cancel). You can run this anytime from the 'Recommend LLM' button in the app."
    }
} catch {
    Write-Output "recommend-model: skipped due to an error ($_). This does not affect the rest of the installation — you can use the 'Recommend LLM' button in the app instead."
} finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item Env:\ARUARU_LLM_BIND -ErrorAction SilentlyContinue
}
exit 0
