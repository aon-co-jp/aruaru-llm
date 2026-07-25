//! `opencuda-llm::GptModel`(GPT-2 124M実学習済み重み)による本格的な
//! 自己回帰テキスト生成。`scoring.rs`(意図分類、軽量・高速)とは目的が
//! 異なるため無理に統合せず、別エンドポイント(`POST /v1/generate`)として
//! 提供する。役割分担: 意図分類=軽量・高速な定型応答振り分け、生成=
//! 本格的だが重い(548MBの重みロード+CPU推論)自由文生成。
//!
//! ## 正直な開示(最重要、CLAUDE.md参照)
//!
//! - GPT-2 124Mは2019年発表の小型モデルであり、GPT-4等の最新商用LLMと
//!   同等の性能・知識・指示追従能力は無い。あくまで「外部LLM契約(API課金)
//!   不要の自己完結型AIとして、実際に文法的な文章を自己回帰生成できる」
//!   ことの実証に留まる。
//! - 生成されるテキストは文法的には自然な英語になることが多いが、
//!   事実として正確である保証は無い(幻覚・意味不明な continuation の
//!   可能性がある)。ファインチューニング済みの対話モデルではなく、
//!   素の事前学習済み言語モデルの貪欲デコード(温度無し、サンプリング無し)
//!   であるため、対話的に「質問に答える」というより「文の続きを予測する」
//!   挙動になる。
//! - `opencuda-llm`自体もPagedAttention・連続バッチング等の本家vLLM最適化を
//!   持たない単一シーケンス逐次デコードのMVP実装(`opencuda-llm`の
//!   モジュールdocコメント参照)。

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use opencuda_core::GpuDevice;
use opencuda_llm::{GptModel, GptTokenizer};

/// GPT-2実重みのディレクトリパス。既定は`open-cuda`側で既に検証・
/// ダウンロード済みの`opencuda-llm/models/gpt2`(sibling path、`../open-cuda`
/// 前提、`PORTING.md`のsibling path依存パターンと同じ)。
/// `ARUARU_LLM_GPT2_DIR`環境変数で上書き可能。
fn model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ARUARU_LLM_GPT2_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from("../open-cuda/crates/opencuda-llm/models/gpt2")
}

struct LoadedGpt {
    model: GptModel,
    tokenizer: GptTokenizer,
}

static GPT2: OnceLock<Result<LoadedGpt, String>> = OnceLock::new();

fn load() -> &'static Result<LoadedGpt, String> {
    GPT2.get_or_init(|| {
        let dir = model_dir();
        let model = GptModel::load(&dir).map_err(|e| format!("failed to load GPT-2 weights from {dir:?}: {e:#}"))?;
        let tokenizer = GptTokenizer::load(&dir).map_err(|e| format!("failed to load GPT-2 tokenizer from {dir:?}: {e:#}"))?;
        Ok(LoadedGpt { model, tokenizer })
    })
}

/// 起動時ウォームアップ用(コールドスタート対策、`scoring::warmup`と同じ
/// 設計思想)。失敗しても致命的ではない(初回リクエスト時に再試行される)。
pub fn warmup() -> Result<()> {
    match load() {
        Ok(_) => Ok(()),
        Err(e) => anyhow::bail!("{e}"),
    }
}

/// エンジン識別子(`engine`フィールド用、常に実装方式を正直に返す方針)。
pub const ENGINE_GPT2_GREEDY: &str = "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu";

/// `prompt`の続きを`max_new_tokens`個、貪欲デコード(argmax、サンプリング
/// 無し)で生成する。GPT-2実重み・実トークナイザが読み込めない場合はエラー
/// を返す(黙って別経路にフォールバックしない、意図分類とは別軸のため
/// bag-of-words的な代替は存在しない)。
pub fn generate(device: &Arc<dyn GpuDevice>, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let loaded = load().as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    let prompt_ids = loaded.tokenizer.encode(prompt).context("opencuda-llm tokenizer encode failed")?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");
    let generated_ids = loaded.model.generate(device, &prompt_ids, max_new_tokens).context("GptModel::generate failed")?;
    loaded.tokenizer.decode(&generated_ids).context("opencuda-llm tokenizer decode failed")
}
