//! `open-cuda-llm::QwenModel`(Qwen2/Qwen2.5系、RoPE+GQA+RMSNorm+SwiGLU)に
//! よるテキスト生成(2026-09-11新設)。
//!
//! ## 位置づけ・正直な開示
//!
//! 既存の`generation.rs`(`GptModel`、GPT-2系)には**一切手を触れず**、
//! 完全に並行・独立した経路として追加した。理由:
//! - `generation.rs`は FP8量子化・MLA(PCA較正版含む)・DXILオフロード・
//!   層折りたたみ(Model Folding)・投機的デコード等、`GptModel`固有の
//!   高度な配線を大量に持つ1246行の成熟したモジュールで、これを
//!   `GptModel`/`QwenModel`両対応のenumへ汎用化するのは大規模な
//!   リファクタリングになり、既存機能への回帰リスクが高い。
//! - 今回はまず「QwenModelを実際に選択・生成できる」という最小限の
//!   経路を安全に追加することを優先した。
//! - **現時点でこのモジュールが持つのは「ロード→貪欲デコード生成」のみ**
//!   ——FP8/MLA/DXILオフロード/層折りたたみ/投機的デコードは未配線
//!   (`open-cuda-llm`側の`QwenModel`自体はMLA圧縮に対応済みだが、この
//!   サービング層からはまだ呼べない、次の増分)。
//! - `/v1/generate`(既存、GPT-2系)とは別に`/v1/generate-qwen`を新設した
//!   ——既存エンドポイントの挙動・契約を一切変えないため。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use open_cuda_llm::{GptTokenizer, QwenModel};
use opencuda_core::GpuDevice;

struct LoadedQwen {
    model: QwenModel,
    tokenizer: GptTokenizer,
    dir: PathBuf,
}

static ACTIVE_QWEN: std::sync::RwLock<Option<Arc<LoadedQwen>>> = std::sync::RwLock::new(None);

/// `dir`(`config.json`+`model.safetensors`+`tokenizer.json`)からQwenモデルを
/// ロードし、以降`/v1/generate-qwen`が使うアクティブモデルとして設定する
/// (`generation::select_model`と同じ「ホットスワップ、プロセス再起動不要」
/// 設計)。
pub fn select_qwen_model(dir: PathBuf) -> Result<()> {
    let tokenizer = GptTokenizer::load(&dir).with_context(|| format!("failed to load tokenizer.json from {dir:?}"))?;
    let model = QwenModel::load(&dir).with_context(|| format!("failed to load Qwen weights from {dir:?}"))?;
    *ACTIVE_QWEN.write().expect("qwen active model lock poisoned") = Some(Arc::new(LoadedQwen { model, tokenizer, dir }));
    Ok(())
}

/// 現在アクティブなQwenモデルのロード元ディレクトリ(未選択なら`None`)。
pub fn active_qwen_model_dir() -> Option<PathBuf> {
    ACTIVE_QWEN.read().expect("qwen active model lock poisoned").as_ref().map(|l| l.dir.clone())
}

/// アクティブなQwenモデルで貪欲デコード生成する(`POST /v1/generate-qwen`)。
/// `generation::generate`と同じ呼び出し規約(繰り返しペナルティは既定値)。
pub fn generate(device: &Arc<dyn GpuDevice>, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let loaded = ACTIVE_QWEN
        .read()
        .expect("qwen active model lock poisoned")
        .clone()
        .context("no Qwen model is currently selected — call POST /v1/qwen/select first")?;
    let prompt_ids = loaded.tokenizer.encode(prompt).context("tokenizer encode failed")?;
    let generated = loaded
        .model
        .generate_with_repetition_penalty(device, &prompt_ids, max_new_tokens, crate::generation::default_repetition_penalty())
        .context("QwenModel::generate_with_repetition_penalty failed")?;
    loaded.tokenizer.decode(&generated).context("tokenizer decode failed")
}
