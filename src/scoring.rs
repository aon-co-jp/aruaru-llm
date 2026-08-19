//! open-cuda連携の意図分類スコアリング(CLAUDE.mdの「SET構成」)。
//!
//! **2026-07-21移行: bag-of-wordsから実際の文埋め込みベースへ**。
//! 以前はユーザー発話・各インテントを固定語彙へのbag-of-words(0/1)
//! ベクトルへ変換し`opencuda_blas::sgemm`でドット積するだけの単純な
//! キーワードマッチングだった。現在は`open-cuda-bert`(multilingual-e5-small、
//! Hugging Face、MITライセンス、日本語含む100言語対応)で実際に文を
//! 384次元の埋め込みベクトルへ変換し、各インテントの代表例文embeddingとの
//! コサイン類似度(`open_cuda_bert::cosine_similarity`)で最も近いものを
//! 選ぶ。埋め込み計算自体、`opencuda-blas`の実GEMM(`sgemm`)・実Attention
//! (`scaled_dot_product_attention`)を`opencuda_cpu::CpuDevice`上で実行して
//! 求めている(スタブではない)。
//!
//! **正直な開示**: これは学習済みエンコーダによる**意味的類似度分類**で
//! あり、bag-of-wordsだった頃より意味理解の質は大きく向上した(実機検証:
//! 「マイナンバーカードの申請をしたい」と「行政手続き・マイナンバーに
//! 関するご案内」の類似度が「今日の天気は晴れです」より高くなることを
//! `open-cuda-bert`側のテストで確認済み)。ただしこれは**エンコーダ専用**の
//! 分類であり、自己回帰デコーダによる文章生成(いわゆる対話生成としての
//! 「LLM」の能力)はまだ実装していない。「LLM」を名乗るこのプロジェクトが
//! 実際に何を計算しているかを誇張しないための開示(詳しくはCLAUDE.md参照)。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use open_cuda_bert::{cosine_similarity, embed_text, BertModel, BertTokenizer};
use opencuda_core::GpuDevice;

/// コサイン類似度がこの値未満のときはどのインテントにも一致しないと
/// みなし、`FALLBACK_REPLY`を返す。multilingual-e5-smallの実測値を基に
/// 調整した閾値(`cargo test`で無関係な発話が誤分類されないことを確認済み)。
const SIMILARITY_THRESHOLD: f32 = 0.86;

/// `/v1/chat`の`engine`フィールドに返す、実際に使われた分類経路の名前。
/// 呼び出し側が「本物の対話生成AIかどうか」「意味理解の質(埋め込み/
/// bag-of-words)」を判別できるよう、常に実際に使った経路を正直に返す
/// (CLAUDE.md「正直な開示」節参照)。**2026-08-06修正**: 以前は`-cpu`が
/// 実行経路(Vulkan/CPU)に関わらず常に固定文字列だった(既知の粗、
/// CLAUDE.md 2026-08-05 HANDOFF参照)。実際の分類経路を反映するには
/// [`engine_embedding_label`](実行時に`-cpu`/`-vulkan`を判定して返す)
/// を使うこと——この定数は後方互換の参考値として残す。
pub const ENGINE_EMBEDDING: &str = "embedding-cosine-v0-open-cuda-bert-cpu";
/// 埋め込みモデル(`models/multilingual-e5-small/`)が存在しない、または
/// ロードに失敗したときに自動的に使われるフォールバック経路
/// (2026-07-25追加、`bow_fallback.rs`参照)。
pub const ENGINE_BOW_FALLBACK: &str = "bow-dotproduct-v0-opencuda-cpu-fallback";
/// 埋め込み・bag-of-words双方が失敗した(通常発生しない異常系)場合。
pub const ENGINE_CLASSIFICATION_UNAVAILABLE: &str = "classification-unavailable-v0";

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
    // `ARUARU_LLM_EMBED_MODEL_DIR`環境変数で上書き可能(2026-08-11追加、
    // Android単体版向け——コンパイル時固定パス〈`CARGO_MANIFEST_DIR`〉は
    // Android実機上には存在しないため、実行時にアプリの内部ストレージへ
    // 展開したモデルディレクトリを指定できるようにする、`open-english/
    // server`の`OPEN_ENGLISH_SERVER_ROOT`と同じパターン)。
    if let Ok(dir) = std::env::var("ARUARU_LLM_EMBED_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    // カレントディレクトリ相対の`models/multilingual-e5-small`が存在すれば
    // それを優先する(2026-08-12追加。install.sh/install.ps1が案内する
    // 配置——systemdの`WorkingDirectory=/etc/aruaru-llm`やWindowsの
    // `C:\Program Files\aruaru-llm`——はバイナリの実行ディレクトリを
    // 前提としており、次の`CARGO_MANIFEST_DIR`フォールバックが指す
    // 開発機/CIランナー上のビルド時パスはGitHub Release配布バイナリを
    // 実際にインストールしたユーザー環境には存在しない。
    // `open-english/server::repo_root`で実機テストの上発見・修正した
    // 同種のバグと同じ修正パターン)。
    let cwd_candidate = PathBuf::from("models/multilingual-e5-small");
    if cwd_candidate.is_dir() {
        return cwd_candidate;
    }
    // 実行ファイルと同じディレクトリに`models/multilingual-e5-small`が
    // あればそちらも試す(手元でzipを展開してそのディレクトリのまま
    // 実行するケースの保険。上のカレントディレクトリと一致することが多いが、
    // 呼び出し元のCWDが別の場所になるケースへの追加フォールバック)。
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("models/multilingual-e5-small");
            if candidate.is_dir() {
                return candidate;
            }
        }
    }
    // aruaru-llm/models/multilingual-e5-small(CLAUDE.md記載のダウンロード済みモデル、
    // ビルド時のパスなので開発機/CIランナー上でのみ有効)。
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/multilingual-e5-small")
}

/// `generation.rs::default_matmul_spirv_path`と同じsibling path前提
/// (`../open-cuda`、`ARUARU_LLM_MATMUL_SPIRV`環境変数で共通に上書き可能)。
/// `open-cuda-bert::BertModel::set_matmul_spirv`(2026-08-06新設、
/// `open-cuda-llm::GptModel::set_matmul_spirv`と同じパターン)へ渡す。
#[cfg(feature = "real-vulkan")]
fn default_matmul_spirv_path() -> PathBuf {
    if let Ok(path) = std::env::var("ARUARU_LLM_MATMUL_SPIRV") {
        return PathBuf::from(path);
    }
    PathBuf::from("../open-cuda/examples/matmul_vulkan_real/shaders/matmul.spv")
}

/// コンパイル済み`softmax.spv`のパス(`real-vulkan` feature有効時のみ使用、
/// 2026-08-06追加)。`generation.rs::default_softmax_spirv_path`と同じ
/// sibling path前提、`open-cuda-bert::BertModel::set_softmax_spirv`
/// (2026-08-06新設)へ渡す。`ARUARU_LLM_SOFTMAX_SPIRV`環境変数で
/// (`generation.rs`側と共通に)上書き可能。
#[cfg(feature = "real-vulkan")]
fn default_softmax_spirv_path() -> PathBuf {
    if let Ok(path) = std::env::var("ARUARU_LLM_SOFTMAX_SPIRV") {
        return PathBuf::from(path);
    }
    PathBuf::from("../open-cuda/examples/softmax_vulkan_real/shaders/softmax.spv")
}

/// `open-cuda-bert::Linear::forward`が`GemmPath::VulkanGeneric`を使える
/// よう、コンパイル済み`matmul.spv`を配線する(2026-08-06新設、
/// `generation.rs::wire_matmul_spirv`と同じ設計、CLAUDE.md 2026-08-05
/// HANDOFFで報告されていた「`scoring`/`security`側には同様のVulkan GEMM
/// 配線が無く、`real-vulkan`有効時は起動時ウォームアップが失敗する」への
/// 対応)。読み込み失敗時はサービスを落とさず、CPU実行のまま継続する
/// (既存の「サービスを壊さない」設計方針を踏襲)。成功した場合のみ
/// [`SPIRV_WIRED`]をセットし、[`dispatch_suffix`]が正しく`-vulkan`を
/// 報告できるようにする。
#[cfg(feature = "real-vulkan")]
fn wire_matmul_spirv(model: &mut BertModel) {
    let path = default_matmul_spirv_path();
    match std::fs::read(&path) {
        Ok(bytes) => {
            let len = bytes.len();
            model.set_matmul_spirv(bytes);
            SPIRV_WIRED.store(true, Ordering::Relaxed);
            tracing::info!(
                "real-vulkan feature enabled: loaded matmul.spv ({len} bytes) from {} and wired into BertModel via set_matmul_spirv (scoring/security)",
                path.display()
            );
        }
        Err(err) => {
            tracing::warn!(
                "real-vulkan feature enabled but failed to load matmul.spv from {} ({err}); \
                 scoring/security Linear layers will keep using the CPU GEMM path even if VulkanDevice was selected \
                 (run tools/compile-vulkan-shaders.* in open-cuda first, or set ARUARU_LLM_MATMUL_SPIRV)",
                path.display()
            );
        }
    }
}

/// `open-cuda-bert::BertModel::set_softmax_spirv`が実際に成功したか
/// どうかを`wire_matmul_spirv`と同じ設計で配線する(2026-08-06追加、
/// `generation.rs::wire_softmax_spirv`と同じパターン)。`set_matmul_spirv`
/// と併用して初めてAttention計算全体(QKᵀ・softmax・P·V)がGPU常駐になる
/// (「GPU GEMM + CPU softmax」から「GPU GEMM + GPU softmax」への移行)。
/// 読み込み失敗時はサービスを落とさず、softmaxはホスト側CPU実行のまま
/// 継続する(既存の「サービスを壊さない」設計方針を踏襲)。
#[cfg(feature = "real-vulkan")]
fn wire_softmax_spirv(model: &mut BertModel) {
    let path = default_softmax_spirv_path();
    match std::fs::read(&path) {
        Ok(bytes) => {
            let len = bytes.len();
            model.set_softmax_spirv(bytes);
            tracing::info!(
                "real-vulkan feature enabled: loaded softmax.spv ({len} bytes) from {} and wired into BertModel via set_softmax_spirv (scoring/security)",
                path.display()
            );
        }
        Err(err) => {
            tracing::warn!(
                "real-vulkan feature enabled but failed to load softmax.spv from {} ({err}); \
                 scoring/security Attention softmax will keep using the CPU path even if matmul.spv is wired \
                 (run tools/compile-vulkan-shaders.* in open-cuda first, or set ARUARU_LLM_SOFTMAX_SPIRV)",
                path.display()
            );
        }
    }
}

/// [`wire_matmul_spirv`]が実際に成功したかどうか(=`BertModel`の全
/// `Linear`にコンパイル済みSPIR-Vが配線済みかどうか)。`real-vulkan`
/// feature無効時は常に`false`のまま(既定ビルドへの影響ゼロ)。
static SPIRV_WIRED: AtomicBool = AtomicBool::new(false);

/// `engine`フィールドに実際の実行経路(Vulkan/CPU)を反映させるための
/// 判定(2026-08-06追加、CLAUDE.md 2026-08-05 HANDOFFで指摘された
/// 「`engine`フィールドが実行経路に関わらず常に`-cpu`固定文字列を返す」
/// 粗の修正)。`device.supports_spirv()`(呼び出し側が実際に選択した
/// デバイス)と[`SPIRV_WIRED`](モデル側の配線が実際に成功したか)の
/// **両方**が真の場合のみ`-vulkan`を返す——デバイスがVulkanでも配線が
/// 失敗していればCPU GEMMパスへフォールバックしているため、その場合は
/// 正直に`-cpu`を返す。
pub fn dispatch_suffix(device: &Arc<dyn GpuDevice>) -> &'static str {
    if device.supports_spirv() && SPIRV_WIRED.load(Ordering::Relaxed) {
        "-vulkan"
    } else {
        "-cpu"
    }
}

/// [`ENGINE_EMBEDDING`]の動的版。実際の実行経路(Vulkan/CPU)を
/// `-vulkan`/`-cpu`接尾辞で反映する(2026-08-06追加)。
pub fn engine_embedding_label(device: &Arc<dyn GpuDevice>) -> String {
    format!("embedding-cosine-v0-open-cuda-bert{}", dispatch_suffix(device))
}

fn get_model() -> Result<&'static EmbeddingModel> {
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let dir = model_dir();
    #[allow(unused_mut)]
    let mut model = BertModel::load(&dir)
        .with_context(|| format!("open-cuda-bert: multilingual-e5-smallのロードに失敗しました({dir:?})"))?;
    let tokenizer = BertTokenizer::load(&dir)
        .with_context(|| format!("open-cuda-bert: tokenizer.jsonのロードに失敗しました({dir:?})"))?;
    #[cfg(feature = "real-vulkan")]
    {
        wire_matmul_spirv(&mut model);
        wire_softmax_spirv(&mut model);
    }
    // 別スレッドと競合してもどちらか片方が採用されればよい(結果は同一)。
    let _ = MODEL.set(EmbeddingModel { model, tokenizer });
    Ok(MODEL.get().expect("MODEL was just set"))
}

pub(crate) fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// multilingual-e5系の規約に沿って1テキストを埋め込む共有ヘルパー。
/// `security`モジュール等が、モデルを二重ロードせずに(同じ`OnceLock`
/// キャッシュを使って)埋め込みを計算できるよう`pub(crate)`で公開する。
/// `is_query`が`true`なら`"query: "`接頭辞(検索側)、`false`なら
/// `"passage: "`接頭辞(登録側)を付ける。
pub(crate) fn embed(device: &Arc<dyn GpuDevice>, text: &str, is_query: bool) -> Result<Vec<f32>> {
    let m = get_model()?;
    let prefix = if is_query { "query: " } else { "passage: " };
    let prefixed = format!("{prefix}{text}");
    embed_text(&m.model, &m.tokenizer, device, &prefixed)
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

/// `idle_background_fold`モジュール向けのデバッグ用アクセサ(2026-08-19
/// 新設)。既にウォームアップ済み(`INTENT_EMBEDDINGS`が計算済み)の
/// インテント代表ベクトル同士のコサイン類似度を、隣接するペアについて
/// 総当たりで計算して返す。**新規にモデル呼び出しは行わない**
/// (`OnceLock`未初期化ならデバイスを持たないため空配列を返す、
/// アイドル時バックグラウンド処理を予期しないタイミングで重いモデル
/// ロードへ導かないための安全策)。読み取り専用、モデル・重みは一切
/// 変更しない。
pub fn debug_intent_embedding_pairwise_similarities() -> Vec<(&'static str, &'static str, f32)> {
    let Some(embeddings) = INTENT_EMBEDDINGS.get() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for i in 0..INTENTS.len() {
        for j in (i + 1)..INTENTS.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            out.push((INTENTS[i].name, INTENTS[j].name, sim));
        }
    }
    out
}

/// `phone_task`モジュール向けのアクセサ(2026-08-19新設)。既に
/// ウォームアップ済みのインテント代表ベクトルから先頭2本を取り出し、
/// スマホ側へ配布する「コサイン類似度計算タスク」の題材として使う。
/// `debug_intent_embedding_pairwise_similarities`と同じく新規のモデル
/// 呼び出しは行わない(未ウォームアップなら`None`、呼び出し側が
/// フォールバックのデモベクトルへ切り替える)。
pub fn sample_embedding_pair_for_phone_task() -> Option<(Vec<f32>, Vec<f32>)> {
    let embeddings = INTENT_EMBEDDINGS.get()?;
    if embeddings.len() < 2 {
        return None;
    }
    Some((embeddings[0].clone(), embeddings[1].clone()))
}

/// ユーザー発話ともっとも類似度の高いインテントを、open-cudaのCPU
/// バックエンド上で実行する実際のBERT系エンコーダ(`open-cuda-bert`、
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

/// [`classify`]の結果。`engine`は実際に使われた分類経路
/// (`ENGINE_EMBEDDING`/`ENGINE_BOW_FALLBACK`/`ENGINE_CLASSIFICATION_UNAVAILABLE`)。
pub struct ClassifyResult {
    pub intent: Option<&'static Intent>,
    pub engine: String,
}

/// 意図分類のエントリポイント。まず`open-cuda-bert`による埋め込み
/// コサイン類似度分類(`best_intent`)を試み、モデル重み
/// (`models/multilingual-e5-small/`)が存在しない・ロードに失敗した等で
/// エラーになった場合は、自動的に`bow_fallback`の固定語彙bag-of-words
/// ドット積へフォールバックする(2026-07-25追加)。**正直な開示**:
/// フォールバック時は意味理解の質が明確に下がる(キーワード一致のみ)ため、
/// `engine`フィールドで必ずどちらの経路が使われたかを呼び出し側へ伝える。
pub fn classify(device: &Arc<dyn GpuDevice>, user_text: &str) -> ClassifyResult {
    match best_intent(device, user_text) {
        Ok(intent) => ClassifyResult { intent, engine: engine_embedding_label(device) },
        Err(err) => {
            tracing::warn!(
                "embedding-based classification unavailable ({err}); falling back to bag-of-words (models/multilingual-e5-small/ missing or failed to load)"
            );
            match crate::bow_fallback::best_intent_bow(device, user_text) {
                Ok(intent) => ClassifyResult { intent, engine: ENGINE_BOW_FALLBACK.to_string() },
                Err(bow_err) => {
                    tracing::warn!("bag-of-words fallback also failed: {bow_err}");
                    ClassifyResult { intent: None, engine: ENGINE_CLASSIFICATION_UNAVAILABLE.to_string() }
                }
            }
        }
    }
}

/// コールドスタート対策(2026-07-22追記、CLAUDE.md 2026-07-22 HANDOFF参照):
/// `open-cuda-bert`モデル・トークナイザのロードとインテント代表ベクトルの
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
