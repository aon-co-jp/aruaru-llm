//! ブラウザ/クライアントから送られた音声を書き起こす音声認識(ASR)
//! エンドポイント `POST /v1/transcribe` の実体(2026-08-29追加、
//! `open-english/docs/SPEECH_RECOGNITION_REDESIGN.md` の P2-β)。
//!
//! # 背景
//!
//! `open-english` の音声認識は当初ブラウザの Web Speech API 頼みで
//! 精度が低かった。P2-α でブラウザ内 Whisper(transformers.js /
//! ONNX Runtime Web)を追加したが、(1) 端末の GPU/NPU/CPU 性能に
//! 大きく左右される、(2) large-v3 級のモデルはブラウザには重すぎる、
//! という限界がある。P2-β は「利用者が自分の PC で起動している
//! `aruaru-llm`(既に open-cuda / open-directx / open-cpu の推論基盤に
//! 乗っている)に、より大きな Whisper モデルでの書き起こしを任せる」
//! 経路をサーバー側に用意する。
//!
//! # 実装方式(2026-08-29 方針変更): whisper.cpp プレビルド CLI の子プロセス
//!
//! 当初は `whisper-rs`(whisper.cpp の Rust バインディング)を直接リンク
//! する設計だったが、多言語再調査で **`whisper-rs-sys` は Windows(MSVC)
//! で bindgen が glibc 固有型を生成して破綻する既知ブロッカー**があり、
//! `whisper-rs 0.16.0` でも `WHISPER_DONT_GENERATE_BINDINGS=1` でも
//! 解消しないと判明した(§3.6)。open-english の主対象は Windows 上で
//! 利用者が起動する aruaru-llm なので、この方式は成立しない。
//!
//! 代わりに **whisper.cpp の公式リリース同梱プレビルド CLI
//! (`whisper-cli` / 旧名 `main`)を子プロセスとして起動**する。これは
//! このエコシステムが既に多用しているパターン——`Db::backup_postgres_
//! via_pg_dump` が `pg_dump` を、`component_update` が `Expand-Archive`
//! を、Android 連携が `adb` を子プロセスで呼ぶのと同じ。C++ リンク・
//! bindgen を完全に回避でき、GPU バックエンド(Vulkan/CUDA/Metal)は
//! プレビルド CLI 側で選ばれたものがそのまま使われる(open-cuda が使う
//! のと同じ物理 GPU 上で走る)。**Cargo feature は不要**(コンパイル時
//! 依存が無いため)。`is_available()` は「CLI 実行ファイルとモデルが
//! 実在するか」を実行時に判定する。
//!
//! # 検証状況(正直な開示)
//!
//! 実 CLI(`whisper-cli`)+ 実 GGML モデルでの書き起こし E2E は、この
//! 開発環境に両方が無いため **未検証**。既定ビルドが壊れないこと
//! (型・テスト・CLI/モデル不在時の `503` フォールバック)までを確認済み。
//! 次周: プレビルド `whisper-cli` + `ggml-base.bin` を用意して
//! `POST /v1/transcribe` を実 HTTP で検証する。

use std::io::Write;
use std::path::{Path, PathBuf};

/// 書き起こし結果。
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscribeOutput {
    /// 書き起こしたテキスト(前後の空白は除去済み)。
    pub text: String,
    /// Whisper が検出/使用した言語コード(例 `"en"` / `"ja"`)。
    pub language: String,
}

/// whisper.cpp CLI(`whisper-cli` / 旧名 `main`)のパス候補。
/// `ARUARU_LLM_WHISPER_CLI` で上書き可、既定は `<crate>/models/whisper/` 下。
pub fn cli_candidates() -> Vec<PathBuf> {
    if let Ok(p) = std::env::var("ARUARU_LLM_WHISPER_CLI") {
        return vec![PathBuf::from(p)];
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models").join("whisper");
    if cfg!(target_os = "windows") {
        vec![dir.join("whisper-cli.exe"), dir.join("main.exe")]
    } else {
        vec![dir.join("whisper-cli"), dir.join("main")]
    }
}

/// 実在する最初の CLI パス(無ければ最初の候補を返す — 表示用)。
pub fn cli_path() -> PathBuf {
    let cands = cli_candidates();
    cands.iter().find(|p| p.is_file()).cloned().unwrap_or_else(|| cands[0].clone())
}

pub fn cli_present() -> bool {
    cli_candidates().iter().any(|p| p.is_file())
}

/// GGML モデルファイルのパス。`ARUARU_LLM_WHISPER_MODEL` で上書き可、
/// 既定は `<crate>/models/whisper/ggml-base.bin`。
pub fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("ARUARU_LLM_WHISPER_MODEL") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("whisper")
        .join("ggml-base.bin")
}

pub fn model_present() -> bool {
    model_path().is_file()
}

/// `POST /v1/transcribe` が実際に書き起こせる状態か(CLI とモデルの両方が
/// 実在するか)。`/v1/runtime` の `whisper` 段と `/v1/transcribe` の可否
/// 判定に使う。
pub fn is_available() -> bool {
    cli_present() && model_present()
}

/// `/v1/runtime` の `whisper.backend` 表示。
pub fn backend_label() -> &'static str {
    if is_available() {
        "whisper.cpp-cli"
    } else {
        "not-available"
    }
}

/// 16kHz mono 16-bit PCM の最小 WAV を書き出す(whisper-cli は 16kHz WAV を
/// 要求するため。ヘッダ 44 バイト + i16 サンプル)。
fn write_wav_16k_mono(path: &Path, pcm: &[f32]) -> std::io::Result<()> {
    let data_len = (pcm.len() * 2) as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // fmt チャンクサイズ
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&16_000u32.to_le_bytes())?; // サンプルレート
    f.write_all(&32_000u32.to_le_bytes())?; // バイトレート = 16000*1*2
    f.write_all(&2u16.to_le_bytes())?; // ブロックアライン
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}

/// `temp_dir` 下にこのプロセス専用の一意なサブディレクトリを作る
/// (`tempfile` crate を実行時依存に加えないための最小実装)。
fn make_scratch_dir() -> std::io::Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("aruaru-llm-whisper-{}-{}-{}", std::process::id(), ts, n));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// whisper-cli を子プロセスで起動し、壁時計上限(既定 300 秒、
/// `ARUARU_LLM_WHISPER_TIMEOUT_SECS` で調整可)を超えたら kill する。
fn run_with_timeout(mut cmd: std::process::Command) -> Result<std::process::Output, String> {
    let timeout_secs: u64 = std::env::var("ARUARU_LLM_WHISPER_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(300);
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start whisper-cli: {e}"))?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                return child.wait_with_output().map_err(|e| format!("whisper-cli wait failed: {e}"));
            }
            Ok(None) => {
                if start.elapsed().as_secs() >= timeout_secs {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("whisper-cli timed out after {timeout_secs}s"));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => return Err(format!("whisper-cli try_wait failed: {e}")),
        }
    }
}

/// whisper-cli が書き出した JSON からテキストと言語を取り出す。
/// バージョン差に強いよう `serde_json::Value` で緩くパースする。
fn parse_whisper_json(bytes: &[u8]) -> Result<TranscribeOutput, String> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("whisper-cli JSON parse failed: {e}"))?;
    let mut text = String::new();
    if let Some(arr) = v.get("transcription").and_then(|t| t.as_array()) {
        for seg in arr {
            if let Some(s) = seg.get("text").and_then(|t| t.as_str()) {
                text.push_str(s);
            }
        }
    }
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("whisper-cli produced no text (silence or too short?)".to_string());
    }
    let language = v
        .get("result")
        .and_then(|r| r.get("language"))
        .and_then(|l| l.as_str())
        .unwrap_or("unknown")
        .to_string();
    Ok(TranscribeOutput { text, language })
}

/// 16kHz mono の f32 PCM(範囲 -1.0..=1.0)を書き起こす。
///
/// CLI もモデルも無ければ `Err`(呼び出し側 `main.rs::transcribe` が
/// `503` + 正直なメッセージを返す)。
pub fn transcribe_pcm16k(pcm: &[f32], language: Option<&str>) -> Result<TranscribeOutput, String> {
    if pcm.is_empty() {
        return Err("audio is empty".to_string());
    }
    if !cli_present() {
        return Err(format!(
            "whisper.cpp CLI not found (looked for {}); download a prebuilt whisper-cli from \
             https://github.com/ggml-org/whisper.cpp/releases and place it there, or set ARUARU_LLM_WHISPER_CLI",
            cli_candidates().iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" / ")
        ));
    }
    if !model_present() {
        return Err(format!(
            "whisper GGML model not found at {} (download e.g. ggml-base.bin, or set ARUARU_LLM_WHISPER_MODEL)",
            model_path().display()
        ));
    }

    let scratch = make_scratch_dir().map_err(|e| format!("could not create scratch dir: {e}"))?;
    let wav = scratch.join("audio.wav");
    let out_prefix = scratch.join("out");
    let out_json = scratch.join("out.json");

    let result = (|| {
        write_wav_16k_mono(&wav, pcm).map_err(|e| format!("could not write WAV: {e}"))?;

        let threads = (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) / 2).max(1);
        let lang = language
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .unwrap_or("auto");

        let mut cmd = std::process::Command::new(cli_path());
        cmd.arg("-m")
            .arg(model_path())
            .arg("-f")
            .arg(&wav)
            .arg("-l")
            .arg(lang)
            .arg("-oj") // JSON 出力(-of で指定した prefix + ".json")
            .arg("-of")
            .arg(&out_prefix)
            .arg("-nt") // テキストにタイムスタンプを付けない
            .arg("-np") // 進捗等を標準出力へ出さない
            .arg("-t")
            .arg(threads.to_string());

        let output = run_with_timeout(cmd)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "whisper-cli exited with {}: {}",
                output.status,
                stderr.lines().last().unwrap_or("").trim()
            ));
        }
        let json = std::fs::read(&out_json)
            .map_err(|e| format!("whisper-cli did not write {}: {e}", out_json.display()))?;
        parse_whisper_json(&json)
    })();

    let _ = std::fs::remove_dir_all(&scratch); // ベストエフォート後始末
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_is_false_without_cli_or_model_in_dev() {
        // 開発環境には whisper-cli も ggml モデルも無いので false のはず。
        // (env 上書きが無い前提。CI もこの状態。)
        if std::env::var("ARUARU_LLM_WHISPER_CLI").is_err()
            && std::env::var("ARUARU_LLM_WHISPER_MODEL").is_err()
        {
            assert!(!is_available());
            assert_eq!(backend_label(), "not-available");
        }
    }

    #[test]
    fn model_path_honors_the_env_override() {
        std::env::set_var("ARUARU_LLM_WHISPER_MODEL", "/tmp/some-model.bin");
        assert_eq!(model_path(), PathBuf::from("/tmp/some-model.bin"));
        std::env::remove_var("ARUARU_LLM_WHISPER_MODEL");
        assert!(model_path().ends_with("ggml-base.bin"));
    }

    #[test]
    fn cli_path_honors_the_env_override() {
        std::env::set_var("ARUARU_LLM_WHISPER_CLI", "/opt/whisper/whisper-cli");
        assert_eq!(cli_path(), PathBuf::from("/opt/whisper/whisper-cli"));
        std::env::remove_var("ARUARU_LLM_WHISPER_CLI");
    }

    #[test]
    fn transcribe_reports_unavailable_when_cli_missing() {
        if std::env::var("ARUARU_LLM_WHISPER_CLI").is_ok() {
            return; // 実 CLI がある環境ではスキップ
        }
        let r = transcribe_pcm16k(&[0.1_f32; 1600], Some("en"));
        assert!(r.is_err());
        let msg = r.unwrap_err();
        assert!(msg.contains("whisper.cpp CLI not found") || msg.contains("model not found"));
    }

    #[test]
    fn write_wav_16k_mono_has_a_valid_riff_header() {
        let dir = make_scratch_dir().unwrap();
        let p = dir.join("t.wav");
        write_wav_16k_mono(&p, &[0.0, 0.5, -0.5, 1.0]).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        // 44 バイトヘッダ + 4 サンプル * 2 バイト
        assert_eq!(bytes.len(), 44 + 8);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_whisper_json_extracts_text_and_language() {
        let j = r#"{"result":{"language":"ja"},"transcription":[{"text":" Hello "},{"text":"world"}]}"#;
        let out = parse_whisper_json(j.as_bytes()).unwrap();
        assert_eq!(out.text, "Hello world");
        assert_eq!(out.language, "ja");
    }

    #[test]
    fn parse_whisper_json_errs_on_empty_transcription() {
        let j = r#"{"result":{"language":"en"},"transcription":[{"text":"   "}]}"#;
        assert!(parse_whisper_json(j.as_bytes()).is_err());
    }
}
