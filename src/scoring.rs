//! open-cuda連携の意図分類スコアリング(CLAUDE.mdの「SET構成」)。
//!
//! **2026-07-21移行: bag-of-wordsから実際の文埋め込みベースへ**。
//! 以前はユーザー発話・各インテントを固定語彙へのbag-of-words(0/1)
//! ベクトルへ変換し`opencuda_blas::sgemm`でドット積するだけの単純な
//! キーワードマッチングだった。現在は`opencuda-bert`(multilingual-e5-small、
//! Hugging Face、MITライセンス、日本語含む100言語対応)で実際に文を
//! 384次元の埋め込みベクトルへ変換し、各インテントの代表例文embeddingとの
//! コサイン類似度(`opencuda_bert::cosine_similarity`)で最も近いものを
//! 選ぶ。埋め込み計算自体、`opencuda-blas`の実GEMM(`sgemm`)・実Attention
//! (`scaled_dot_product_attention`)を`opencuda_cpu::CpuDevice`上で実行して
//! 求めている(スタブではない)。
//!
//! **正直な開示**: これは学習済みエンコーダによる**意味的類似度分類**で
//! あり、bag-of-wordsだった頃より意味理解の質は大きく向上した(実機検証:
//! 「マイナンバーカードの申請をしたい」と「行政手続き・マイナンバーに
//! 関するご案内」の類似度が「今日の天気は晴れです」より高くなることを
//! `opencuda-bert`側のテストで確認済み)。ただしこれは**エンコーダ専用**の
//! 分類であり、自己回帰デコーダによる文章生成(いわゆる対話生成としての
//! 「LLM」の能力)はまだ実装していない。「LLM」を名乗るこのプロジェクトが
//! 実際に何を計算しているかを誇張しないための開示(詳しくはCLAUDE.md参照)。

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use opencuda_bert::{cosine_similarity, embed_text, BertModel, BertTokenizer};
use opencuda_core::GpuDevice;

/// コサイン類似度がこの値未満のときはどのインテントにも一致しないと
/// みなし、`FALLBACK_REPLY`を返す。multilingual-e5-smallの実測値を基に
/// 調整した閾値(`cargo test`で無関係な発話が誤分類されないことを確認済み)。
const SIMILARITY_THRESHOLD: f32 = 0.86;

pub struct Intent {
    pub name: &'static str,
    pub reply: &'static str,
    /// 英語訳の定型応答文。`lang != "ja"`のリクエストはまずこれを使う
    /// (2026-07-22追記: e-gov.infoが本サービスへ問い合わせるようになり、
    /// e-gov.info自体は13言語対応なのに本サービス経由だと日本語固定に
    /// なってしまう非対称を解消するため)。
    pub reply_en: &'static str,
    /// この意図を表す代表的な例文(複数可)。起動後の初回呼び出し時に
    /// これらの埋め込みベクトルを平均・正規化してキャッシュし、
    /// ユーザー発話との類似度比較に用いる。
    examples: &'static [&'static str],
}

impl Intent {
    /// 要求言語に応じた応答文を返す。`(reply, actual_lang, was_fallback)`。
    /// `lang == "ja"`なら日本語、それ以外は英語(現状唯一の翻訳先)を返す。
    /// `lang`が`"ja"`でも`"en"`でもない(未対応言語)場合は、無言で
    /// 日本語へ落とすのではなく英語へフォールバックし、`was_fallback`で
    /// それを呼び出し側へ正直に伝える(「graceful degradation, never
    /// silent」というこのエコシステムの方針、CLAUDE.md参照)。
    pub fn reply_for(&self, lang: &str) -> (&'static str, &'static str, bool) {
        match lang {
            "ja" => (self.reply, "ja", false),
            "en" => (self.reply_en, "en", false),
            _ => (self.reply_en, "en", true),
        }
    }
}

pub const INTENTS: &[Intent] = &[
    Intent {
        name: "gov",
        examples: &[
            "マイナンバーカードの申請をしたい",
            "行政手続き・マイナンバーに関するご案内",
            "役所へのオンライン申請の方法を知りたい",
        ],
        reply: "eガバメント(デジタルガバメント)についてのご案内ですね。\
ペーパーレスでのオンライン申請、コンビニ端末(Loppi/Famiポート等)での手続き、\
金額に応じた段階的な本人確認に対応しています。詳しくは https://e-gov.info/gov をご覧ください。",
        reply_en: "It sounds like you're asking about e-Government (digital government) services. \
We support paperless online applications, procedures via convenience-store terminals (Loppi/Famiport, etc.), \
and tiered identity verification based on transaction amount. See https://e-gov.info/gov for details.",
    },
    Intent {
        name: "trade",
        examples: &[
            "商品を買いたい、注文したい",
            "I want to buy a product and place an order",
            "オンラインでの買い物について知りたい",
        ],
        reply: "オンライン貿易プラットフォームでのお買い物ですね。\
食料品・家電・自動車・オーディオ機器まで幅広く取り扱っています(現在は実在庫を伴わないサンプル運用です)。\
詳しくは https://e-gov.info/trade をご覧ください。",
        reply_en: "It sounds like you're interested in shopping on our online trade platform. \
We carry a wide range of goods, from groceries to home appliances, automobiles, and audio equipment \
(currently a sample operation with no real inventory). See https://e-gov.info/trade for details.",
    },
    Intent {
        name: "credit",
        examples: &[
            "掛け仕入れと与信審査について教えてほしい",
            "売掛金の保証や請求書の与信調査について知りたい",
            "credit and invoice financing for wholesale purchases",
        ],
        reply: "AI与信調査・掛け仕入れ・売掛保証についてのご質問ですね。\
与信スコアに応じた後払い仕入れ、電子請求書の重複調査、売掛債権の保証に対応予定です\
(現時点では設計方針の段階で、実際の与信審査機能はまだ搭載していません)。\
詳しくは https://e-gov.info/credit をご覧ください。",
        reply_en: "It sounds like you're asking about AI-based credit screening, buy-now-pay-later wholesale \
purchasing, or accounts-receivable guarantees. We plan to offer credit-score-based deferred payment for \
purchasing, duplicate-invoice detection, and receivables guarantees (this is currently at the design stage; \
actual credit screening is not yet implemented). See https://e-gov.info/credit for details.",
    },
    Intent {
        name: "realestate",
        examples: &[
            "不動産や土地、賃貸の間取りについて相談したい",
            "工務店に家の建築を依頼したい",
            "real estate, land, and house rental inquiries",
        ],
        reply: "不動産投資・AI工務店についてのご質問ですね。\
検索した土地情報をもとにAIが間取りをご提案する機能を構想しています\
(電子契約は正式な許可が下りるまで未実装のサンプル・デモ段階です)。\
詳しくは https://e-gov.info/realestate をご覧ください。",
        reply_en: "It sounds like you're asking about real estate investment or our AI-assisted builder service. \
We're planning a feature where AI suggests floor plans based on land data you search for \
(electronic contracts are not yet implemented and remain a sample/demo pending formal approval). \
See https://e-gov.info/realestate for details.",
    },
];

pub const FALLBACK_REPLY: &str = "e-gov.infoへようこそ。\
「申請したい」「買いたい」「仕入れたい」「土地を探したい」のように\
教えていただければ、該当するページをご案内します。\
(本メッセージはopen-cudaのCPUバックエンドで計算した文埋め込み\
コサイン類似度に基づく分類結果です。自己回帰的な対話生成はまだ\
実装していません、詳しくはCLAUDE.mdをご覧ください)";

pub const FALLBACK_REPLY_EN: &str = "Welcome to e-gov.info. \
Try telling us what you'd like to do, e.g. \"I want to apply\", \"I want to buy something\", \
\"I want to purchase inventory\", or \"I'm looking for land\", and we'll point you to the right page. \
(This message is a classification result based on text-embedding cosine similarity computed on the \
open-cuda CPU backend. Autoregressive dialogue generation is not yet implemented; see CLAUDE.md for details.)";

/// [`FALLBACK_REPLY`]の言語別版。[`Intent::reply_for`]と同じ規約:
/// `"ja"`は日本語、それ以外は英語(未対応言語は無言で日本語へ落とさず
/// 英語へフォールバックし、その旨を返す)。
pub fn fallback_reply_for(lang: &str) -> (&'static str, &'static str, bool) {
    match lang {
        "ja" => (FALLBACK_REPLY, "ja", false),
        "en" => (FALLBACK_REPLY_EN, "en", false),
        _ => (FALLBACK_REPLY_EN, "en", true),
    }
}

struct EmbeddingModel {
    model: BertModel,
    tokenizer: BertTokenizer,
}

/// `multilingual-e5-small`のモデル・トークナイザは初回呼び出し時に一度だけ
/// ロードし、プロセス内で使い回す(ロードに数秒かかるため、リクエストの
/// たびにロードし直すと極端に遅くなる)。
static MODEL: OnceLock<EmbeddingModel> = OnceLock::new();
/// 各インテントの代表例文embedding(平均・L2正規化済み)もプロセス内で
/// キャッシュする(テスト・リクエストのたびに毎回embeddingし直すと、
/// インテント数×例文数だけ余計な推論が走ってしまうため)。
static INTENT_EMBEDDINGS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();

fn model_dir() -> PathBuf {
    // aruaru-llm/models/multilingual-e5-small(CLAUDE.md記載のダウンロード済みモデル)。
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/multilingual-e5-small")
}

fn get_model() -> Result<&'static EmbeddingModel> {
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let dir = model_dir();
    let model = BertModel::load(&dir)
        .with_context(|| format!("opencuda-bert: multilingual-e5-smallのロードに失敗しました({dir:?})"))?;
    let tokenizer = BertTokenizer::load(&dir)
        .with_context(|| format!("opencuda-bert: tokenizer.jsonのロードに失敗しました({dir:?})"))?;
    // 別スレッドと競合してもどちらか片方が採用されればよい(結果は同一)。
    let _ = MODEL.set(EmbeddingModel { model, tokenizer });
    Ok(MODEL.get().expect("MODEL was just set"))
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn get_intent_embeddings(device: &Arc<dyn GpuDevice>) -> Result<&'static Vec<Vec<f32>>> {
    if let Some(e) = INTENT_EMBEDDINGS.get() {
        return Ok(e);
    }
    let m = get_model()?;
    let hidden_size = m.model.hidden_size();

    let mut embeddings = Vec::with_capacity(INTENTS.len());
    for intent in INTENTS {
        let mut acc = vec![0.0f32; hidden_size];
        for example in intent.examples {
            // multilingual-e5系は"passage: "接頭辞で登録側テキストを埋め込む規約。
            let text = format!("passage: {example}");
            let v = embed_text(&m.model, &m.tokenizer, device, &text)?;
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        normalize(&mut acc);
        embeddings.push(acc);
    }

    let _ = INTENT_EMBEDDINGS.set(embeddings);
    Ok(INTENT_EMBEDDINGS.get().expect("INTENT_EMBEDDINGS was just set"))
}

/// ユーザー発話ともっとも類似度の高いインテントを、open-cudaのCPU
/// バックエンド上で実行する実際のBERT系エンコーダ(`opencuda-bert`、
/// GEMM/Attentionは`opencuda-blas`の実カーネル)で計算する。すべての
/// インテントとの類似度が`SIMILARITY_THRESHOLD`未満ならNoneを返す
/// (呼び出し側で`FALLBACK_REPLY`にフォールバックする)。
pub fn best_intent(device: &Arc<dyn GpuDevice>, user_text: &str) -> Result<Option<&'static Intent>> {
    let m = get_model()?;
    let intent_embeddings = get_intent_embeddings(device)?;

    // multilingual-e5系は"query: "接頭辞で検索側テキストを埋め込む規約。
    let query_text = format!("query: {user_text}");
    let query_embedding = embed_text(&m.model, &m.tokenizer, device, &query_text)?;

    let mut best: Option<(usize, f32)> = None;
    for (i, intent_embedding) in intent_embeddings.iter().enumerate() {
        let sim = cosine_similarity(&query_embedding, intent_embedding);
        if best.map(|(_, best_sim)| sim > best_sim).unwrap_or(true) {
            best = Some((i, sim));
        }
    }

    Ok(best.filter(|(_, sim)| *sim >= SIMILARITY_THRESHOLD).map(|(i, _)| &INTENTS[i]))
}

/// コールドスタート対策(2026-07-22追記、CLAUDE.md 2026-07-22 HANDOFF参照):
/// `opencuda-bert`モデル・トークナイザのロードとインテント代表ベクトルの
/// 計算は、いずれも`OnceLock`により初回呼び出し時に一度だけ実行される
/// 設計だが、それを「サーバが接続を受け付け始めた後の最初の実リクエスト」
/// 任せにすると、呼び出し元(e-gov.info等)のタイムアウト(実測3秒)を
/// 超えてしまうことが実際に観測された。この関数を`main()`の起動処理で
/// (`Server::new(...).run(app)`より前に)一度呼び出すことで、モデルロード+
/// ダミー推論をサーバ起動時に前倒しし、実際のリクエストが来る頃には
/// すでにウォーム状態にしておく。
pub fn warmup(device: &Arc<dyn GpuDevice>) -> Result<()> {
    // best_intentと全く同じコードパス(get_model→get_intent_embeddings→
    // embed_text)を通すダミー推論。ここで計算した結果自体は捨ててよく、
    // 目的はOnceLockへのモデルロード・インテントembeddingキャッシュの
    // 前倒しのみ。
    let _ = best_intent(device, "warmup")?;
    Ok(())
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
    fn reply_for_ja_returns_japanese_unchanged() {
        let device = cpu_device();
        let intent = best_intent(&device, "マイナンバーカードの申請をしたい").unwrap().unwrap();
        let (reply, lang, fallback) = intent.reply_for("ja");
        assert_eq!(reply, intent.reply);
        assert_eq!(lang, "ja");
        assert!(!fallback);
        assert!(reply.contains("eガバメント"));
    }

    #[test]
    fn reply_for_en_returns_english_translation() {
        let device = cpu_device();
        let intent = best_intent(&device, "マイナンバーカードの申請をしたい").unwrap().unwrap();
        let (reply, lang, fallback) = intent.reply_for("en");
        assert_eq!(reply, intent.reply_en);
        assert_eq!(lang, "en");
        assert!(!fallback);
        assert!(reply.contains("e-Government"));
    }

    #[test]
    fn reply_for_unsupported_lang_falls_back_to_english_with_indicator() {
        let device = cpu_device();
        let intent = best_intent(&device, "マイナンバーカードの申請をしたい").unwrap().unwrap();
        let (reply, lang, fallback) = intent.reply_for("fr");
        assert_eq!(reply, intent.reply_en);
        assert_eq!(lang, "en");
        assert!(fallback, "unsupported language should fall back to English, not silently to Japanese");
    }

    #[test]
    fn fallback_reply_for_respects_lang_and_flags_unsupported() {
        let (ja_reply, ja_lang, ja_fallback) = fallback_reply_for("ja");
        assert_eq!(ja_reply, FALLBACK_REPLY);
        assert_eq!(ja_lang, "ja");
        assert!(!ja_fallback);

        let (en_reply, en_lang, en_fallback) = fallback_reply_for("en");
        assert_eq!(en_reply, FALLBACK_REPLY_EN);
        assert_eq!(en_lang, "en");
        assert!(!en_fallback);

        let (unsupported_reply, unsupported_lang, unsupported_fallback) = fallback_reply_for("zh");
        assert_eq!(unsupported_reply, FALLBACK_REPLY_EN);
        assert_eq!(unsupported_lang, "en");
        assert!(unsupported_fallback);
    }
}
