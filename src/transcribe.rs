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
//! # 正直な開示(アーキテクチャ上の妥協、`nllb.rs` と同じ方針)
//!
//! whisper.cpp(`whisper-rs` crate 経由)は C++ ビルド(CMake +
//! C コンパイラ + bindgen/libclang)を要する。このエコシステムが
//! GPT-2・BERT で貫いてきた「手作り Rust 実装 + safetensors 直接
//! ロード、重量級フレームワーク非依存」からは外れる。そのため
//! `whisper-transcribe` Cargo feature(**既定オフ**)の背後に隔離し、
//! 未指定時のビルド(CI・VPS 本番)には一切影響しない。feature 無効時の
//! `POST /v1/transcribe` は `503` + 正直なエラーメッセージを返す。
//!
//! whisper.cpp 自身の GPU バックエンド(CUDA / Vulkan / Metal)は
//! `open-cuda` とは別実装だが、有効化すれば `open-cuda` が使うのと
//! **同じ物理 GPU** 上で走る。CPU 実行時は whisper.cpp が自前で
//! AVX2/FMA/NEON をディスパッチする(`open-cpu` とは別実装だが目的は同じ)。
//! feature 名の `open-cuda/open-directx/open-cpu に乗る` は「同じ
//! ハードウェアを共有する」という意味であって、これらのクレートを
//! 直接呼ぶわけではない(誇張しない)。
//!
//! # 検証状況(正直な開示)
//!
//! `whisper-transcribe` feature を有効化した実ビルド・実モデルロード・
//! 実書き起こしの E2E は、この開発環境に whisper.cpp のビルド
//! ツールチェーン(libclang)と GGML モデルファイルが無いため
//! **未検証**(`nllb-translate` と同じ状況)。既定ビルドが壊れない
//! こと(feature 無効時のフォールバック・型・テスト)までを確認済み。

/// 書き起こし結果。
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscribeOutput {
    /// 書き起こしたテキスト(前後の空白は除去済み)。
    pub text: String,
    /// Whisper が検出/使用した言語コード(例 `"en"` / `"ja"`)。
    /// 言語を明示指定した場合はそれをそのまま返す。
    pub language: String,
}

/// このビルドで whisper.cpp 書き起こしが利用可能か(feature フラグの
/// 実行時問い合わせ、`/v1/runtime` の `whisper` tier 表示と
/// `/v1/transcribe` の可否判定に使う)。
pub fn is_available() -> bool {
    cfg!(feature = "whisper-transcribe")
}

/// whisper.cpp のどの計算バックエンドがこのビルドに組み込まれているか
/// (`whisper-rs` の feature 転送で決まる)。`/v1/runtime` の
/// `whisper.backend` に出す。
pub fn backend_label() -> &'static str {
    if cfg!(feature = "whisper-cuda") {
        "cuda"
    } else if cfg!(feature = "whisper-vulkan") {
        "vulkan"
    } else if cfg!(feature = "whisper-transcribe") {
        "cpu"
    } else {
        "not-compiled-in"
    }
}

/// GGML モデルファイルのパス。`ARUARU_LLM_WHISPER_MODEL` で上書き可、
/// 既定は `<crate>/models/whisper/ggml-base.bin`。
pub fn model_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ARUARU_LLM_WHISPER_MODEL") {
        return std::path::PathBuf::from(p);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("whisper")
        .join("ggml-base.bin")
}

/// 現在の設定でモデルファイルが実在してロードを試せる状態か
/// (`/v1/runtime` の `whisper.model_present` に出す。feature が
/// 無効でもパスの存在チェックだけは常に行える)。
pub fn model_present() -> bool {
    model_path().is_file()
}

// ─────────────────────────────────────────────────────────────────────
// feature 有効時: whisper-rs(whisper.cpp)で実際に書き起こす
// ─────────────────────────────────────────────────────────────────────
#[cfg(feature = "whisper-transcribe")]
mod imp {
    use super::TranscribeOutput;
    use std::sync::Mutex;
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    /// モデルの遅延ロード + スレッド間共有(初回リクエストでのみ
    /// GGML をロード、以降は使い回す。`generation.rs` の GPT-2 重み
    /// ロードと同じ「初回コスト許容・ウォームアップ後は高速」方針)。
    static CTX: std::sync::OnceLock<Mutex<Result<WhisperContext, String>>> = std::sync::OnceLock::new();

    fn context() -> Result<std::sync::MutexGuard<'static, Result<WhisperContext, String>>, String> {
        let cell = CTX.get_or_init(|| {
            let path = super::model_path();
            let loaded = path
                .to_str()
                .ok_or_else(|| "whisper model path is not valid UTF-8".to_string())
                .and_then(|p| {
                    WhisperContext::new_with_params(p, WhisperContextParameters::default())
                        .map_err(|e| format!("failed to load whisper model {p}: {e}"))
                });
            Mutex::new(loaded)
        });
        cell.lock().map_err(|_| "whisper context lock poisoned".to_string())
    }

    /// 16kHz mono の f32 PCM(範囲 -1.0..=1.0)を書き起こす。
    pub fn transcribe_pcm16k(pcm: &[f32], language: Option<&str>) -> Result<TranscribeOutput, String> {
        if pcm.is_empty() {
            return Err("audio is empty".to_string());
        }
        let guard = context()?;
        let ctx = guard.as_ref().map_err(|e| e.clone())?;
        let mut state = ctx.create_state().map_err(|e| format!("failed to create whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // "auto" 相当: language 未指定なら Whisper に検出させる。
        params.set_language(language.or(Some("auto")));
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // whisper.cpp が自前で使うスレッド数(CPU 実行時)。open-cpu の
        // 検出ではなく whisper.cpp 側の既定に任せるが、論理コア数の
        // 半分程度に抑えてサーバーの他リクエストを圧迫しないようにする。
        let threads = (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) / 2).max(1);
        params.set_n_threads(threads as i32);

        state.full(params, pcm).map_err(|e| format!("whisper transcription failed: {e}"))?;

        let n = state.full_n_segments().map_err(|e| format!("full_n_segments failed: {e}"))?;
        let mut text = String::new();
        for i in 0..n {
            let seg = state
                .full_get_segment_text(i)
                .map_err(|e| format!("full_get_segment_text({i}) failed: {e}"))?;
            text.push_str(&seg);
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("whisper produced no text (silence or too short?)".to_string());
        }

        let detected = match language {
            Some(l) => l.to_string(),
            None => state
                .full_lang_id()
                .ok()
                .and_then(|id| whisper_rs::get_lang_str(id).map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".to_string()),
        };

        Ok(TranscribeOutput { text, language: detected })
    }
}

// ─────────────────────────────────────────────────────────────────────
// feature 無効時: 常にエラー(呼び出し側は 503 + 正直なメッセージを返す)
// ─────────────────────────────────────────────────────────────────────
#[cfg(not(feature = "whisper-transcribe"))]
mod imp {
    use super::TranscribeOutput;

    pub fn transcribe_pcm16k(_pcm: &[f32], _language: Option<&str>) -> Result<TranscribeOutput, String> {
        Err("whisper-transcribe feature not enabled at build time \
             (rebuild aruaru-llm with --features whisper-transcribe, and place a GGML model \
             at ARUARU_LLM_WHISPER_MODEL or <crate>/models/whisper/ggml-base.bin)"
            .to_string())
    }
}

pub use imp::transcribe_pcm16k;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_reflects_the_compiled_feature_flag() {
        assert_eq!(is_available(), cfg!(feature = "whisper-transcribe"));
    }

    #[test]
    fn backend_label_is_not_compiled_in_for_the_default_build() {
        if !cfg!(feature = "whisper-transcribe") {
            assert_eq!(backend_label(), "not-compiled-in");
        }
    }

    #[test]
    fn model_path_honors_the_env_override() {
        // 直列化のため専用の一意な値を使う(他テストと環境変数を共有
        // しない: このキーは transcribe 以外では読まれない)。
        std::env::set_var("ARUARU_LLM_WHISPER_MODEL", "/tmp/some-model.bin");
        assert_eq!(model_path(), std::path::PathBuf::from("/tmp/some-model.bin"));
        std::env::remove_var("ARUARU_LLM_WHISPER_MODEL");
        assert!(model_path().ends_with("ggml-base.bin"));
    }

    #[cfg(not(feature = "whisper-transcribe"))]
    #[test]
    fn transcribe_reports_unavailable_when_feature_disabled() {
        let r = transcribe_pcm16k(&[0.0_f32; 16], None);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("not enabled"));
    }
}
