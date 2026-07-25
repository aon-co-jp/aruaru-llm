//! 埋め込みモデル(`opencuda-bert`/multilingual-e5-small)の重みファイルが
//! `models/multilingual-e5-small/`に存在しない、またはロードに失敗した
//! 場合に自動的に使われるフォールバック経路(2026-07-25追加)。
//!
//! **正直な開示**: これは本物のニューラルネットワークによる意味理解では
//! なく、`scoring.rs`が2026-07-21移行前まで使っていた、固定語彙への
//! bag-of-wordsドット積という極めて単純なベクトル演算(要素積を
//! `opencuda_cpu::CpuDevice`上でカーネル実行し、ホスト側で合計するだけ)。
//! `opencuda-bert`の470MB超のモデル重みを用意できない環境でもサービスを
//! 完全に停止させないための可用性優先のフォールバックであり、`/v1/chat`の
//! `engine`フィールドには常にこのフォールバックが使われたことを正直に
//! 返す(`scoring::ENGINE_BOW_FALLBACK`)。分類精度は埋め込みベースより
//! 明確に劣る(意味理解ではなくキーワード一致のため)。

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx};

use crate::scoring::{Intent, INTENTS};

/// 固定語彙。この語彙に含まれる単語だけがベクトル化の対象になる
/// (単純なbag-of-wordsのため、語彙外の単語は無視される)。
const VOCAB: &[&str] = &[
    "申請", "手続き", "行政", "役所", "マイナンバー", "government", "application", "買いたい", "欲しい", "注文",
    "商品", "buy", "want", "order", "product", "仕入れ", "与信", "掛け", "売掛", "請求書", "credit", "invoice",
    "不動産", "土地", "間取り", "工務店", "賃貸", "real", "estate", "land", "house",
];

struct IntentKeywords {
    name: &'static str,
    keywords: &'static [&'static str],
}

/// `scoring::INTENTS`の各`name`に対応するキーワード集合
/// (旧bag-of-words実装、2026-07-21の埋め込み移行前のコミット`0045de8~1`から
/// 復元)。`scoring::INTENTS`にインテントを追加・改名する際はここも
/// 合わせて更新すること。
const INTENT_KEYWORDS: &[IntentKeywords] = &[
    IntentKeywords {
        name: "gov",
        keywords: &["申請", "手続き", "行政", "役所", "マイナンバー", "government", "application"],
    },
    IntentKeywords {
        name: "trade",
        keywords: &["買いたい", "欲しい", "注文", "商品", "buy", "want", "order", "product"],
    },
    IntentKeywords {
        name: "credit",
        keywords: &["仕入れ", "与信", "掛け", "売掛", "請求書", "credit", "invoice"],
    },
    IntentKeywords {
        name: "realestate",
        keywords: &["不動産", "土地", "間取り", "工務店", "賃貸", "real", "estate", "land", "house"],
    },
];

/// テキストを固定語彙に対するbag-of-wordsベクトル(0.0/1.0)へ変換する。
fn to_vector(text: &str, keywords: &[&str]) -> Vec<f32> {
    let lower = text.to_lowercase();
    VOCAB
        .iter()
        .map(|word| {
            let word_lower = word.to_lowercase();
            let present = keywords.iter().any(|k| k.eq_ignore_ascii_case(word)) || lower.contains(&word_lower);
            if present {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn to_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32スライスを読み取り専用のu8スライスとして見るだけ(open-cudaの
    // examples/vector_addと同じ最小キャストパターン、依存を増やさないため)。
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn from_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: 同上、可変版。
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// `a`と`b`の要素積を、open-cudaの`CpuDevice`上でカーネル実行して求める。
fn elementwise_multiply_via_opencuda(device: &Arc<dyn GpuDevice>, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let bytes = std::mem::size_of_val(a);

    let da = alloc_buffer(device, bytes)?;
    let db = alloc_buffer(device, bytes)?;
    let dc = alloc_buffer(device, bytes)?;

    da.copy_from_host(to_bytes(a))?;
    db.copy_from_host(to_bytes(b))?;

    let kernel = CompiledKernel::native("bow_multiply", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let i = ctx.global_id_x() as usize;
        let (a_ptr, a_len) = args[0].as_ptr().unwrap();
        let (b_ptr, _) = args[1].as_ptr().unwrap();
        let (c_ptr, _) = args[2].as_ptr().unwrap();
        let n = args[3].as_usize().unwrap();

        if i >= n {
            return;
        }
        debug_assert!((i + 1) * 4 <= a_len);

        // SAFETY: i < n、各バッファはn*4バイト確保済み。各スレッドは自分の
        // iのみ書くため競合しない(vector_addと同じ安全性の根拠)。
        unsafe {
            let a = (a_ptr as *const f32).add(i).read();
            let b = (b_ptr as *const f32).add(i).read();
            (c_ptr as *mut f32).add(i).write(a * b);
        }
    });

    let cfg = LaunchConfig::linear(n as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(n),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; n];
    dc.copy_to_host(from_bytes_mut(&mut c))?;
    Ok(c)
}

/// ユーザー発話ともっともスコアの高いインテントを、bag-of-wordsドット積で
/// 計算する(埋め込みモデルが使えないときのフォールバック専用)。全
/// インテントが0点ならNoneを返す。
pub fn best_intent_bow(device: &Arc<dyn GpuDevice>, user_text: &str) -> Result<Option<&'static Intent>> {
    let mut best: Option<(&'static Intent, f32)> = None;

    for ik in INTENT_KEYWORDS {
        let intent = INTENTS
            .iter()
            .find(|i| i.name == ik.name)
            .expect("bow_fallback::INTENT_KEYWORDS names must match scoring::INTENTS names");
        let msg_vec = to_vector(user_text, &[]);
        let intent_vec = to_vector("", ik.keywords);
        let product = elementwise_multiply_via_opencuda(device, &msg_vec, &intent_vec)?;
        let score: f32 = product.iter().sum();

        if score > 0.0 && best.as_ref().map(|(_, s)| score > *s).unwrap_or(true) {
            best = Some((intent, score));
        }
    }

    Ok(best.map(|(intent, _)| intent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;

    fn cpu_device() -> Arc<dyn GpuDevice> {
        CpuDevice::new(0)
    }

    #[test]
    fn bow_matches_government_intent() {
        let device = cpu_device();
        let intent = best_intent_bow(&device, "マイナンバーカードの申請をしたい").unwrap().unwrap();
        assert_eq!(intent.name, "gov");
    }

    #[test]
    fn bow_matches_trade_intent_case_insensitively() {
        let device = cpu_device();
        let intent = best_intent_bow(&device, "I want to BUY a speaker").unwrap().unwrap();
        assert_eq!(intent.name, "trade");
    }

    #[test]
    fn bow_matches_credit_intent() {
        let device = cpu_device();
        let intent = best_intent_bow(&device, "掛け仕入れについて教えて").unwrap().unwrap();
        assert_eq!(intent.name, "credit");
    }

    #[test]
    fn bow_matches_realestate_intent() {
        let device = cpu_device();
        let intent = best_intent_bow(&device, "土地を探しています").unwrap().unwrap();
        assert_eq!(intent.name, "realestate");
    }

    #[test]
    fn bow_returns_none_for_unmatched_text() {
        let device = cpu_device();
        let intent = best_intent_bow(&device, "こんにちは").unwrap();
        assert!(intent.is_none());
    }

    #[test]
    fn bow_elementwise_multiply_matches_expected_values() {
        let device = cpu_device();
        let a = vec![1.0, 2.0, 3.0, 0.0];
        let b = vec![1.0, 0.0, 3.0, 5.0];
        let result = elementwise_multiply_via_opencuda(&device, &a, &b).unwrap();
        assert_eq!(result, vec![1.0, 0.0, 9.0, 0.0]);
    }
}
