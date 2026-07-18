//! open-cuda連携の意図分類スコアリング(CLAUDE.mdの「SET構成」)。
//!
//! **これは実際にopen-cudaの`GpuDevice`実行パイプラインを通る**——文字列の
//! `contains()`比較ではなく、ユーザー発話と各インテントをbag-of-words
//! ベクトルに変換し、要素積カーネルを`opencuda_cpu::CpuDevice`上で実行
//! (`device.launch_kernel`)してから、ホスト側でその積を合計してドット積
//! スコアを求める。open-cudaの`examples/vector_add`と同じ「デバイス確保→
//! 転送→カーネル起動→回収」のパイプラインを、加算ではなく乗算カーネルで
//! 踏襲している。
//!
//! **正直な開示**: これは本物のニューラルネットワーク(埋め込み+
//! Attention等)による意味理解ではなく、固定語彙に対するbag-of-words
//! ドット積という、極めて単純なベクトル演算。「LLM」を名乗るこの
//! プロジェクトが、実際に何を計算しているかを誇張しないための開示。

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx};

/// 固定語彙。この語彙に含まれる単語だけがベクトル化の対象になる
/// (単純なbag-of-wordsのため、語彙外の単語は無視される)。
const VOCAB: &[&str] = &[
    "申請", "手続き", "行政", "役所", "マイナンバー", "government", "application",
    "買いたい", "欲しい", "注文", "商品", "buy", "want", "order", "product",
    "仕入れ", "与信", "掛け", "売掛", "請求書", "credit", "invoice",
    "不動産", "土地", "間取り", "工務店", "賃貸", "real", "estate", "land", "house",
];

pub struct Intent {
    pub name: &'static str,
    pub reply: &'static str,
    keywords: &'static [&'static str],
}

pub const INTENTS: &[Intent] = &[
    Intent {
        name: "gov",
        keywords: &["申請", "手続き", "行政", "役所", "マイナンバー", "government", "application"],
        reply: "eガバメント(デジタルガバメント)についてのご案内ですね。\
ペーパーレスでのオンライン申請、コンビニ端末(Loppi/Famiポート等)での手続き、\
金額に応じた段階的な本人確認に対応しています。詳しくは https://e-gov.info/gov をご覧ください。",
    },
    Intent {
        name: "trade",
        keywords: &["買いたい", "欲しい", "注文", "商品", "buy", "want", "order", "product"],
        reply: "オンライン貿易プラットフォームでのお買い物ですね。\
食料品・家電・自動車・オーディオ機器まで幅広く取り扱っています(現在は実在庫を伴わないサンプル運用です)。\
詳しくは https://e-gov.info/trade をご覧ください。",
    },
    Intent {
        name: "credit",
        keywords: &["仕入れ", "与信", "掛け", "売掛", "請求書", "credit", "invoice"],
        reply: "AI与信調査・掛け仕入れ・売掛保証についてのご質問ですね。\
与信スコアに応じた後払い仕入れ、電子請求書の重複調査、売掛債権の保証に対応予定です\
(現時点では設計方針の段階で、実際の与信審査機能はまだ搭載していません)。\
詳しくは https://e-gov.info/credit をご覧ください。",
    },
    Intent {
        name: "realestate",
        keywords: &["不動産", "土地", "間取り", "工務店", "賃貸", "real", "estate", "land", "house"],
        reply: "不動産投資・AI工務店についてのご質問ですね。\
検索した土地情報をもとにAIが間取りをご提案する機能を構想しています\
(電子契約は正式な許可が下りるまで未実装のサンプル・デモ段階です)。\
詳しくは https://e-gov.info/realestate をご覧ください。",
    },
];

pub const FALLBACK_REPLY: &str = "e-gov.infoへようこそ。\
「申請したい」「買いたい」「仕入れたい」「土地を探したい」のように\
教えていただければ、該当するページをご案内します。\
(本メッセージはopen-cudaのCPUバックエンドで計算したbag-of-wordsスコアに\
基づくルールベース応答です。実際のニューラルLLM推論は未実装、詳しくは\
CLAUDE.mdをご覧ください)";

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
/// (`examples/vector_add`の加算カーネルを乗算に置き換えたもの)。
fn elementwise_multiply_via_opencuda(device: &Arc<dyn GpuDevice>, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let bytes = n * std::mem::size_of::<f32>();

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

/// ユーザー発話ともっともスコアの高いインテントを、open-cudaのCPU
/// バックエンド経由で計算する。全インテントが0点ならNoneを返す
/// (呼び出し側で`FALLBACK_REPLY`にフォールバックする)。
pub fn best_intent(device: &Arc<dyn GpuDevice>, user_text: &str) -> Result<Option<&'static Intent>> {
    let mut best: Option<(&Intent, f32)> = None;

    for intent in INTENTS {
        let msg_vec = to_vector(user_text, &[]);
        let intent_vec = to_vector("", intent.keywords);
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
    fn matches_government_intent_via_opencuda() {
        let device = cpu_device();
        let intent = best_intent(&device, "マイナンバーカードの申請をしたい").unwrap().unwrap();
        assert_eq!(intent.name, "gov");
    }

    #[test]
    fn matches_trade_intent_case_insensitively() {
        let device = cpu_device();
        let intent = best_intent(&device, "I want to BUY a speaker").unwrap().unwrap();
        assert_eq!(intent.name, "trade");
    }

    #[test]
    fn matches_credit_intent() {
        let device = cpu_device();
        let intent = best_intent(&device, "掛け仕入れについて教えて").unwrap().unwrap();
        assert_eq!(intent.name, "credit");
    }

    #[test]
    fn matches_realestate_intent() {
        let device = cpu_device();
        let intent = best_intent(&device, "土地を探しています").unwrap().unwrap();
        assert_eq!(intent.name, "realestate");
    }

    #[test]
    fn returns_none_for_unmatched_text() {
        let device = cpu_device();
        let intent = best_intent(&device, "こんにちは").unwrap();
        assert!(intent.is_none());
    }

    #[test]
    fn elementwise_multiply_matches_expected_values() {
        let device = cpu_device();
        let a = vec![1.0, 2.0, 3.0, 0.0];
        let b = vec![1.0, 0.0, 3.0, 5.0];
        let result = elementwise_multiply_via_opencuda(&device, &a, &b).unwrap();
        assert_eq!(result, vec![1.0, 0.0, 9.0, 0.0]);
    }
}
