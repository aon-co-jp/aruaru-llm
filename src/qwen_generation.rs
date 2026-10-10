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

/// 文脈 `prefix` のあとに各 `conts[i]` が続く対数尤度(アクティブな Qwen モデル、バッチ prefill)。
pub fn score_texts(_device: &Arc<dyn GpuDevice>, prefix: &str, conts: &[String]) -> Result<Vec<f32>> {
    let loaded = ACTIVE_QWEN.read().expect("qwen active model lock poisoned").clone().context("no Qwen model is currently selected")?;
    let device = _device;
    let prefix_ids = loaded.tokenizer.encode(prefix).context("tokenizer encode failed")?;
    let mut ids = Vec::with_capacity(conts.len());
    for c in conts {
        ids.push(if c.is_empty() { Vec::new() } else { loaded.tokenizer.encode(c).context("tokenizer encode failed")? });
    }
    loaded.model.score_continuations(device, &prefix_ids, &ids).context("QwenModel::score_continuations failed")
}

/// 文脈 `prefix` の次に来る文字列の上位 `k` 個(文字列, 対数確率)。
pub fn top_next_texts(device: &Arc<dyn GpuDevice>, prefix: &str, k: usize) -> Result<Vec<(String, f32)>> {
    let loaded = ACTIVE_QWEN.read().expect("qwen active model lock poisoned").clone().context("no Qwen model is currently selected")?;
    let prefix_ids = loaded.tokenizer.encode(prefix).context("tokenizer encode failed")?;
    let top = loaded.model.top_next_tokens(device, &prefix_ids, k).context("QwenModel::top_next_tokens failed")?;
    Ok(top.into_iter().filter_map(|(id, lp)| loaded.tokenizer.decode(&[id]).ok().map(|t| (t, lp))).collect())
}

/// 1 トークンぶんの範囲(文字単位)と、直前までの文脈つきの対数確率。
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpan {
    pub start: usize,
    pub len: usize,
    pub logp: f32,
}

/// 文章の各トークンの範囲と対数確率(アクティブな Qwen、1 回のバッチ prefill)。OCR の誤読の検出用。
/// バイト単位のトークン(文字の途中で切れるもの)は、文字が完成するまで次のトークンと合わせて 1 つの範囲にする。
pub fn scan_text(device: &Arc<dyn GpuDevice>, text: &str) -> Result<Vec<TokenSpan>> {
    let loaded = ACTIVE_QWEN.read().expect("qwen active model lock poisoned").clone().context("no Qwen model is currently selected")?;
    let ids = loaded.tokenizer.encode(text).context("tokenizer encode failed")?;
    if ids.len() < 2 {
        return Ok(Vec::new());
    }
    let lps = loaded.model.token_logprobs(device, &ids).context("QwenModel::token_logprobs failed")?;
    let mut spans = Vec::new();
    let mut done_chars = 0usize;
    let mut pending = 0.0f32;
    for k in 0..ids.len() {
        pending += lps[k];
        let dec = loaded.tokenizer.decode(&ids[..=k]).unwrap_or_default();
        let chars: Vec<char> = dec.chars().collect();
        // 末尾の不完全な文字(U+FFFD)は、まだ数えない。
        let complete = chars.iter().rposition(|&c| c != '\u{FFFD}').map(|p| p + 1).unwrap_or(0);
        let complete = if chars.last() == Some(&'\u{FFFD}') { complete } else { chars.len() };
        if complete > done_chars {
            spans.push(TokenSpan { start: done_chars, len: complete - done_chars, logp: pending });
            done_chars = complete;
            pending = 0.0;
        }
    }
    Ok(spans)
}

/// 頻出順(Qwen の語彙は、頻出の字ほど ID が小さい傾向がある)に並べた、1 文字で 1 トークンの漢字・かなの集合(最大 `n` 字)。
/// アプリが、残っているインクの形(字形)で候補を探すときの母集団にする。モデルが選ばれていなくても使えない(語彙はモデルのもの)。
pub fn common_chars(n: usize) -> Result<Vec<char>> {
    let loaded = ACTIVE_QWEN.read().expect("qwen active model lock poisoned").clone().context("no Qwen model is currently selected")?;
    let mut out: Vec<char> = Vec::new();
    for id in 0..120_000u32 {
        if out.len() >= n {
            break;
        }
        let Ok(t) = loaded.tokenizer.decode(&[id]) else {
            continue;
        };
        let mut it = t.chars();
        if let (Some(c), None) = (it.next(), it.next()) {
            let ok = matches!(c as u32, 0x3041..=0x3096 | 0x30A1..=0x30FA | 0x4E00..=0x9FFF);
            if ok && !out.contains(&c) {
                out.push(c);
            }
        }
    }
    Ok(out)
}
