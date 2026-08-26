; aruaru-llm Windowsインストーラー(Inno Setup)。
;
; ユーザー指示「open-englishでaruaru-llmはopen-cudaとopen-directxと一緒に
; 使用したほうが本領を発揮するならSETでダウンロードしてインストールする
; 仕様にして、リポジトリ名-installer.exeという名前に統一して」への対応。
;
; 「SET」の実体(正直な開示): open-cuda/open-directxはaruaru-llmとは
; 別のアプリではなく、aruaru-llmがCargo path依存として直接リンクする
; Rustライブラリ(opencuda-vulkan/opencuda-directx crate)である。
; したがって「一緒にダウンロード・インストールする」とは、それらの
; GPUバックエンドを組み込んでビルドした「1本の実行ファイル」を
; 提供することを意味する——open-cuda/open-directxを個別のexeとして
; 同梱するわけではない(そもそもそのようなスタンドアロンアプリは
; 存在しない)。
;
; GPU版(installgpuタスク)は既定で未チェック。理由(正直な開示、
; CLAUDE.md 2026-08-23 HANDOFF参照): このプロジェクト自身の開発機
; (低スペックのNVIDIA GT 730)での実測で、GPU版はCPU版より**遅く**
; なることが確認されている(小さな計算ではGPUディスパッチの
; オーバーヘッドが支配的になるため)。効果は機種依存であり、
; 「本領を発揮する」ことを無条件には保証できない——このため既定は
; 安全側のCPU版のままとし、GPU版は利用者が明示的に選んだ場合のみ
; インストールする。
;
; ビルド方法: リポジトリルートで
;   cargo build --release --bin aruaru-llm
;   CARGO_TARGET_DIR=target-gpu cargo build --release --bin aruaru-llm --features hw-detect-vulkan,real-vulkan,hw-detect-directx,real-dx12
; を実行した後、このディレクトリで`ISCC.exe aruaru-llm.iss`を実行する。
; (GPU版のビルドが無い場合でも、installgpuタスクのチェックボックスは
; 表示されるが対応する[Files]行が見つからずインストール時にエラーに
; なる——CI(.github/workflows/release.yml)では両方を必ずビルドする)。

#define MyAppName "aruaru-llm"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-local-build"
#endif
#define MyAppPublisher "aon-co-jp"
#define MyAppURL "https://github.com/aon-co-jp/aruaru-llm"
#define MyAppExeName "aruaru-llm.exe"

[Setup]
; PrivilegesRequired=lowest: 管理者権限のPowerShellを手動で立ち上げる
; 必要をなくすため(ユーザー指摘「パワーシェルを管理者で立ち上げるのは
; 面倒くさいです」への対応)。UAC昇格プロンプト自体を不要にする——
; open-englishのインストーラーと同じ方針。
PrivilegesRequired=lowest
AppId={{3E7A1C4F-9B2D-4A6E-8F31-5D9C2E7B0A44}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
OutputDir=dist
; エコシステム全体の命名規則: <リポジトリ名>-installer.exe
; (バージョン番号なし、常に同じファイル名——open-englishと同じ方針)。
OutputBaseFilename=aruaru-llm-installer
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; GPU版(open-cuda Vulkan/DirectXバックエンド組み込み、「SET」の実体)。
; 既定は未チェック——上記の正直な開示(GPU版がCPU版より遅い実測結果が
; ある)に基づく安全側デフォルト。
Name: "installgpu"; Description: "GPU acceleration build (open-cuda Vulkan/DirectX combined — may be SLOWER on some GPUs, see README-INSTALLED.txt) / GPUアクセラレーション版(open-cuda Vulkan/DirectX組み込み——環境によっては逆に遅くなります、README-INSTALLED.txt参照)"; Flags: unchecked
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; 既定(CPU版)。installgpuタスクが選ばれなかった場合はこのまま使う。
Source: "..\..\target\release\{#MyAppExeName}"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion
; GPU版が選ばれた場合、上記を同名で上書きする([Files]は上から順に
; 処理されるため、この行が後から実行され最終的にこちらが有効になる)。
Source: "..\..\target-gpu\release\{#MyAppExeName}"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion; Tasks: installgpu
Source: "README-INSTALLED.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion; DestName: "install-service.ps1"
Source: "recommend-model.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; インストール中にハードウェア検出→推奨モデルの提示→選択に応じた
; ダウンロードまでを行う(ユーザー指示「インストール途中で、推薦の
; LLMを提示してLLMの特徴も英語と日本語で提示してもう一つ大きなLLMに
; しますか?などのメッセージ後に選択してLLMをインストール可能」への
; 対応)。ダイアログ表示のため`runhidden`は付けない
; (waituntilterminatedのみ——利用者の選択を待つ)。サイレント
; インストール(自動更新等)時はダイアログを出さず既定のままにする
; (`skipifsilent`)。
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\recommend-model.ps1"" -AppDir ""{app}"""; \
    StatusMsg: "Choosing a recommended AI model... / おすすめのAIモデルを選択中..."; \
    Flags: waituntilterminated skipifsilent
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
