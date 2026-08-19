//! 自動アップデート機能(2026-08-19新設)。
//!
//! ユーザー指示「`open-english/server/src/self_update.rs`と同様の
//! 自己更新の仕組みをエコシステム内の関連リポジトリへ展開してほしい」
//! への対応。`open-english`側はInno Setupインストーラー(`unins000.exe`+
//! 新インストーラーの無人差し替え)を前提とした設計だが、このリポジトリ
//! (`aruaru-llm`)はGitHub Releasesの配布物自体が単一の実行ファイル+
//! `install.sh`/`install.ps1`という**zip/tar.gz形式**であり、専用の
//! アンインストーラーを持たない(`.github/workflows/release.yml`参照)。
//! そのため、コピペではなくこのリポジトリの実際の配布形態に合わせ、
//! 「新バージョンの実行ファイルをダウンロード→現在の実行ファイルを
//! `.bak`として退避→新実行ファイルへ差し替え→再起動→`/healthz`で
//! ヘルスチェック→失敗時は退避しておいた旧バイナリへロールバック」
//! という設計にした。Windows/Linux/macOSのいずれも、差し替え自体は
//! **デタッチした一時スクリプトが、このプロセス終了後に行う**
//! (実行中の自分自身のファイルを、実行中のまま安全に上書きする方法は
//! プラットフォームに依らず確実ではないため——Windows は明示的な
//! ファイルロックで失敗しうる、Unix系はinodeの挙動上は可能だが、
//! この実装はWindows/Unix共通のスクリプトベースの安全側設計に統一した)。
//!
//! ## 正直な開示(最重要)
//!
//! - **実機E2E検証は未実施**: この開発機には`aon-co-jp/aruaru-llm`の
//!   実際のGitHub Releaseが存在するかどうかに関わらず、「新バージョンを
//!   実際にリリースし、起動中の旧バージョンがそれを検知して自己更新する」
//!   という一連の流れを最初から最後まで実行して確認するには至っていない。
//!   検証できたのはコンパイル成功・単体テスト(バージョン比較・アセット名
//!   判定ロジック)・`fetch_latest_release`がリリース不在時に正直に
//!   ログを出すだけでクラッシュしないこと、までに留まる。
//! - **バージョン情報の取得元**: `open-english`は隣接ファイル
//!   `version.json`を読む設計だったが、このリポジトリには元々そのような
//!   ファイルが存在しない。代わりに`env!("CARGO_PKG_VERSION")`
//!   (`Cargo.toml`の`version`フィールド、ビルド時に埋め込まれる)を
//!   ローカルバージョンとして使う——追加のファイル配置を必要としない
//!   利点がある一方、`cargo build`ではなく手元でバイナリを直接実行した
//!   開発ビルドでも常に何らかのバージョン文字列を持つため、
//!   `open-english`側にあった「`version.json`が無ければ更新対象外と
//!   判断してスキップする」という安全弁は使えない。誤ってこのバイナリを
//!   意図せず更新してしまうことを避けるため、既定では
//!   `ARUARU_LLM_ENABLE_SELF_UPDATE=1`(または`true`)を明示的に設定
//!   しない限り、この自動更新機構自体が起動時に何もしない
//!   (`check_and_apply_update`冒頭のガード参照)。
//! - **ヘルスチェック→ロールバック**: `/healthz`(既存エンドポイント)を
//!   使う。新バイナリ起動後`HEALTH_CHECK_SECS`秒以内に到達できなければ、
//!   退避しておいた旧バイナリを復元して起動し直す。この一連の流れ自体は
//!   実機で確認していない(コードレビューベース)。

use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

const GITHUB_REPO: &str = "aon-co-jp/aruaru-llm";

/// 新バージョン起動後、ヘルスチェックへ与える猶予秒数(`open-english`
/// 側の同名定数と同じ設計思想、複雑な指数バックオフ等は行わない)。
pub(crate) const HEALTH_CHECK_SECS: u64 = 12;

#[derive(Debug, Deserialize)]
pub(crate) struct ReleaseAsset {
    pub(crate) name: String,
    pub(crate) browser_download_url: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LatestRelease {
    pub(crate) tag_name: String,
    pub(crate) assets: Vec<ReleaseAsset>,
}

/// `"v0.2.3"`や`"0.2.3"`のようなタグ文字列を`(major, minor, patch)`へ
/// パースする。パース不可な部分は0扱い(不明時は最新と判断しない安全側)。
pub(crate) fn parse_version(raw: &str) -> (u64, u64, u64) {
    let trimmed = raw.trim().trim_start_matches('v');
    let mut parts = trimmed.split('.').map(|s| s.parse::<u64>().unwrap_or(0));
    (parts.next().unwrap_or(0), parts.next().unwrap_or(0), parts.next().unwrap_or(0))
}

pub(crate) fn is_newer(remote: &str, local: &str) -> bool {
    parse_version(remote) > parse_version(local)
}

fn self_update_enabled() -> bool {
    std::env::var("ARUARU_LLM_ENABLE_SELF_UPDATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

async fn fetch_latest_release() -> anyhow::Result<LatestRelease> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(10)).build()?;
    let url = format!("https://api.github.com/repos/{GITHUB_REPO}/releases/latest");
    let res = client.get(&url).header("User-Agent", "aruaru-llm-self-updater").send().await?;
    if !res.status().is_success() {
        anyhow::bail!("GitHub releases API returned HTTP {}", res.status());
    }
    Ok(res.json::<LatestRelease>().await?)
}

/// `.github/workflows/release.yml`が生成する`aruaru-llm-windows-x86_64.zip`/
/// `aruaru-llm-linux-x86_64.tar.gz`の命名規則に合わせてアセットを探す。
fn platform_asset(release: &LatestRelease) -> Option<&ReleaseAsset> {
    let is_windows = cfg!(target_os = "windows");
    release.assets.iter().find(|a| {
        let name = a.name.to_lowercase();
        if is_windows {
            name.ends_with(".zip") && name.contains("windows")
        } else {
            name.ends_with(".tar.gz") && name.contains("linux")
        }
    })
}

pub(crate) async fn download_to(url: &str, dest: &std::path::Path) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().build()?;
    let bytes = client.get(url).header("User-Agent", "aruaru-llm-self-updater").send().await?.bytes().await?;
    std::fs::write(dest, &bytes)?;
    Ok(())
}

/// アーカイブを展開し、中身のバイナリ(`aruaru-llm`/`aruaru-llm.exe`)への
/// パスを返す。Windows側は`Expand-Archive`(PowerShell標準)、Unix側は
/// `tar`をそれぞれ子プロセスとして呼ぶ(追加crate依存を増やさないための
/// 実用上の判断、`open-english`側も同様の設計)。
async fn extract_binary(archive_path: &std::path::Path, extract_dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    let _ = std::fs::remove_dir_all(extract_dir);
    std::fs::create_dir_all(extract_dir)?;

    if cfg!(target_os = "windows") {
        let status = tokio::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    archive_path.display(),
                    extract_dir.display()
                ),
            ])
            .status()
            .await?;
        if !status.success() {
            anyhow::bail!("Expand-Archive failed with status {status}");
        }
        Ok(extract_dir.join("aruaru-llm.exe"))
    } else {
        let status = tokio::process::Command::new("tar")
            .args(["xzf", &archive_path.to_string_lossy(), "-C"])
            .arg(extract_dir)
            .status()
            .await?;
        if !status.success() {
            anyhow::bail!("tar extraction failed with status {status}");
        }
        Ok(extract_dir.join("aruaru-llm"))
    }
}

/// デタッチしたスクリプトを起動し、このプロセス自身を終了する
/// (`std::process::exit`、呼び出し元には戻らない)。スクリプトは
/// 「少し待つ→現在の実行ファイルを`.bak`として退避→新バイナリへ
/// 差し替え→再起動→`/healthz`確認→失敗ならロールバック」を行う。
async fn apply_update(new_binary: &std::path::Path, bind_addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let backup = exe.with_extension("bak");

    if cfg!(target_os = "windows") {
        let script_path = std::env::temp_dir().join("aruaru-llm-self-update.bat");
        let mut script = String::from("@echo off\r\nping 127.0.0.1 -n 3 > nul\r\n");
        script += &format!("copy /Y \"{}\" \"{}\"\r\n", exe.display(), backup.display());
        script += &format!("copy /Y \"{}\" \"{}\"\r\n", new_binary.display(), exe.display());
        script += &format!("start \"\" \"{}\"\r\n", exe.display());
        script += &format!("ping 127.0.0.1 -n {} > nul\r\n", HEALTH_CHECK_SECS + 2);
        script += &format!(
            "powershell -NoProfile -Command \"try {{ $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://{bind_addr}/healthz'; if ($r.StatusCode -ne 200) {{ exit 1 }} }} catch {{ exit 1 }}\"\r\n"
        );
        script += "if %errorlevel% neq 0 (\r\n";
        script += &format!("  taskkill /IM \"{}\" /F >nul 2>nul\r\n", exe.file_name().and_then(|n| n.to_str()).unwrap_or("aruaru-llm.exe"));
        script += "  ping 127.0.0.1 -n 2 > nul\r\n";
        script += &format!("  copy /Y \"{}\" \"{}\"\r\n", backup.display(), exe.display());
        script += &format!("  start \"\" \"{}\"\r\n", exe.display());
        script += ")\r\n";
        script += &format!("del \"{}\"\r\n", script_path.display());
        std::fs::write(&script_path, script)?;
        tokio::process::Command::new("cmd").args(["/C", "start", "", script_path.to_string_lossy().as_ref()]).spawn()?;
    } else {
        let script_path = std::env::temp_dir().join("aruaru-llm-self-update.sh");
        let mut script = String::from("#!/bin/sh\nsleep 3\n");
        script += &format!("cp \"{}\" \"{}\"\n", exe.display(), backup.display());
        script += &format!("cp \"{}\" \"{}\"\n", new_binary.display(), exe.display());
        script += &format!("chmod +x \"{}\"\n", exe.display());
        script += &format!("nohup \"{}\" >/dev/null 2>&1 &\n", exe.display());
        script += &format!("sleep {}\n", HEALTH_CHECK_SECS + 2);
        script += &format!(
            "if ! curl -sf --max-time 3 'http://{bind_addr}/healthz' >/dev/null 2>&1; then\n"
        );
        script += &format!("  pkill -f \"{}\" 2>/dev/null\n", exe.display());
        script += "  sleep 1\n";
        script += &format!("  cp \"{}\" \"{}\"\n", backup.display(), exe.display());
        script += &format!("  chmod +x \"{}\"\n", exe.display());
        script += &format!("  nohup \"{}\" >/dev/null 2>&1 &\n", exe.display());
        script += "fi\n";
        std::fs::write(&script_path, &script)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        tokio::process::Command::new("sh").arg(&script_path).spawn()?;
    }

    tracing::info!("aruaru-llm self-update: launched swap+healthcheck script, exiting to release file locks");
    std::process::exit(0);
}

/// 起動時のメンテナンスチェックの一部として呼ぶ想定のエントリポイント。
/// `ARUARU_LLM_ENABLE_SELF_UPDATE`が未設定/falseの場合は何もしない
/// (既定off、上記モジュールdoc「正直な開示」参照)。
pub async fn check_and_apply_update(bind_addr: std::net::SocketAddr) {
    if !self_update_enabled() {
        tracing::debug!("aruaru-llm self-update: disabled (set ARUARU_LLM_ENABLE_SELF_UPDATE=1 to enable)");
        return;
    }

    let release = match fetch_latest_release().await {
        Ok(r) => r,
        Err(e) => {
            tracing::info!("aruaru-llm self-update: could not check GitHub releases ({e}) — continuing without updating");
            return;
        }
    };

    let local = env!("CARGO_PKG_VERSION");
    if !is_newer(&release.tag_name, local) {
        tracing::info!("aruaru-llm self-update: up to date (local {local}, latest release {})", release.tag_name);
        return;
    }

    let Some(asset) = platform_asset(&release) else {
        tracing::info!(
            "aruaru-llm self-update: newer release {} found but no matching platform asset attached — skipping",
            release.tag_name
        );
        return;
    };

    tracing::info!("aruaru-llm self-update: newer release {} found (local {local}), downloading {}", release.tag_name, asset.name);

    let dest = std::env::temp_dir().join(&asset.name);
    if let Err(e) = download_to(&asset.browser_download_url, &dest).await {
        tracing::info!("aruaru-llm self-update: download failed ({e}) — continuing without updating");
        return;
    }

    let extract_dir = std::env::temp_dir().join("aruaru-llm-self-update-extract");
    let new_binary = match extract_binary(&dest, &extract_dir).await {
        Ok(p) if p.exists() => p,
        Ok(p) => {
            tracing::info!("aruaru-llm self-update: extracted archive but expected binary not found at {}", p.display());
            return;
        }
        Err(e) => {
            tracing::info!("aruaru-llm self-update: extraction failed ({e}) — continuing without updating");
            return;
        }
    };

    if let Err(e) = apply_update(&new_binary, bind_addr).await {
        tracing::info!("aruaru-llm self-update: failed to launch update ({e})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_version_strings_with_and_without_v_prefix() {
        assert_eq!(parse_version("v0.2.3"), (0, 2, 3));
        assert_eq!(parse_version("0.2.3"), (0, 2, 3));
        assert_eq!(parse_version("1.0"), (1, 0, 0));
        assert_eq!(parse_version("garbage"), (0, 0, 0));
    }

    #[test]
    fn is_newer_compares_semver_correctly() {
        assert!(is_newer("v0.3.0", "0.2.3"));
        assert!(is_newer("v1.0.0", "0.9.9"));
        assert!(!is_newer("v0.2.3", "0.2.3"));
        assert!(!is_newer("v0.2.0", "0.2.3"));
    }

    #[test]
    fn platform_asset_finds_expected_naming() {
        let release = LatestRelease {
            tag_name: "v0.3.0".into(),
            assets: vec![
                ReleaseAsset { name: "aruaru-llm-windows-x86_64.zip".into(), browser_download_url: "https://example/win".into() },
                ReleaseAsset { name: "aruaru-llm-linux-x86_64.tar.gz".into(), browser_download_url: "https://example/linux".into() },
            ],
        };
        let asset = platform_asset(&release).expect("asset should be found");
        if cfg!(target_os = "windows") {
            assert!(asset.name.contains("windows"));
        } else {
            assert!(asset.name.contains("linux"));
        }
    }
}
