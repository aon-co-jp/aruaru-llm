//! `open-cuda-llm::DeepseekModel`(DeepSeek-V2/V3系Multi-head Latent
//! Attention)によるテキスト生成(2026-09-13新設)。
//!
//! ## 位置づけ・正直な開示
//!
//! `qwen_generation.rs`と同じ設計(既存の`generation.rs`/GPT-2系には
//! 一切手を触れず、完全に並行・独立した経路として追加)を踏襲する。
//!
//! **重要な制約(誇張しない)**: `open_cuda_llm::DeepseekModel::load`は
//! MoE(DeepSeekMoE)層を読み込めない(`deepseek_arch.rs`のモジュールdoc
//! 参照)。実在するDeepSeek-V2/V2-Lite/V3の公開チェックポイントは
//! `first_k_dense_replace`(通常1)以降のほぼ全層がMoEのため、**現時点では
//! 実在の公開チェックポイントをこの経路でエンドツーエンドにロードする
//! ことはできない**。そのため`model_catalog.rs`には(Qwenと違い)自動
//! ダウンロード用のDeepSeekカタログを設けていない——存在しない互換性を
//! 装ってダウンロードボタンを出すのは不正直なため。このモジュールが
//! 提供するのは、(1) 将来のMoE対応後にそのまま使える「ロード→生成」の
//! サービング層、(2) ユーザーが自前で用意した「MLA構成だがMLP層は
//! dense SwiGLU」という互換チェックポイント(ローカルディレクトリ指定)を
//! 今すぐ試せる経路、の2点。
//!
//! `/v1/generate`(GPT-2系)・`/v1/generate-qwen`(Qwen系)とは別に
//! `/v1/generate-deepseek`を新設する(既存エンドポイントの挙動・契約を
//! 一切変えないため)。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use open_cuda_llm::{DeepseekModel, GptTokenizer};
use opencuda_core::GpuDevice;

struct LoadedDeepseek {
    model: DeepseekModel,
    tokenizer: GptTokenizer,
    dir: PathBuf,
}

static ACTIVE_DEEPSEEK: std::sync::RwLock<Option<Arc<LoadedDeepseek>>> = std::sync::RwLock::new(None);

/// `dir`(`config.json`+`model.safetensors`+`tokenizer.json`)からDeepSeek
/// MLAモデルをロードし、以降`/v1/generate-deepseek`が使うアクティブ
/// モデルとして設定する(`qwen_generation::select_qwen_model`と同じ
/// 「ホットスワップ、プロセス再起動不要」設計)。**正直な開示**:
/// MoE層を含む実在のDeepSeekチェックポイントは`DeepseekModel::load`内で
/// 明示的なエラー(どの層のどのテンソルが見つからないか)で失敗する
/// (`deepseek_arch.rs`の`load()`docコメント参照)。
pub fn select_deepseek_model(dir: PathBuf) -> Result<()> {
    let tokenizer = GptTokenizer::load(&dir).with_context(|| format!("failed to load tokenizer.json from {dir:?}"))?;
    let model = DeepseekModel::load(&dir).with_context(|| format!("failed to load DeepSeek MLA weights from {dir:?}"))?;
    *ACTIVE_DEEPSEEK.write().expect("deepseek active model lock poisoned") = Some(Arc::new(LoadedDeepseek { model, tokenizer, dir }));
    Ok(())
}

/// 現在アクティブなDeepSeekモデルのロード元ディレクトリ(未選択なら`None`)。
pub fn active_deepseek_model_dir() -> Option<PathBuf> {
    ACTIVE_DEEPSEEK.read().expect("deepseek active model lock poisoned").as_ref().map(|l| l.dir.clone())
}

/// アクティブなDeepSeekモデルで貪欲デコード生成する
/// (`POST /v1/generate-deepseek`)。`qwen_generation::generate`と同じ
/// 呼び出し規約(繰り返しペナルティは既定値)。
pub fn generate(device: &Arc<dyn GpuDevice>, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let loaded = ACTIVE_DEEPSEEK
        .read()
        .expect("deepseek active model lock poisoned")
        .clone()
        .context("no DeepSeek model is currently selected — call POST /v1/deepseek/select first")?;
    let prompt_ids = loaded.tokenizer.encode(prompt).context("tokenizer encode failed")?;
    let generated = loaded
        .model
        .generate_with_repetition_penalty(device, &prompt_ids, max_new_tokens, crate::generation::default_repetition_penalty())
        .context("DeepseekModel::generate_with_repetition_penalty failed")?;
    loaded.tokenizer.decode(&generated).context("tokenizer decode failed")
}
