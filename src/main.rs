//! aruaru-llm — aruaruエコシステム共通の「AIチャットコマース」応答サービス。
//!
//! **正直な開示(最重要、詳細はCLAUDE.md参照)**: 2026-07-21時点では
//! 自己回帰デコーダによる文章生成(いわゆる対話生成としての「LLM」の
//! 能力)は実装していない。`open-cuda`の`open-cuda-bert`クレート
//! (multilingual-e5-small、MITライセンス)で実際に文を埋め込みベクトルへ
//! 変換し、`opencuda-blas`の実GEMM(`sgemm`)・実Attention
//! (`scaled_dot_product_attention`)を`opencuda_cpu::CpuDevice`上で実行して
//! 意図ごとの代表例文とのコサイン類似度を求める、エンコーダベースの
//! 意味的類似度分類(旧: 固定語彙bag-of-wordsのドット積)。
//!
//! **「分身の術」構成**: このサービスは1インスタンスを複数ドメインが
//! 共有する設計(`src/tenants.rs`参照)。ドメインを追加するたびに
//! 新しい`aruaru-llm`プロセスを個別インストールする必要はない——
//! `POST /admin/tenants`で動的登録するだけでよい。
//!
//! **Poem互換ファサード(RPoem)への移行(2026-07-31)**: 本家`poem`クレート
//! への直接依存を廃止し、`RPoem`(`open-runo-poem-compat`、
//! `open_runo_router::hyper_compat`のtokio/hyper直接実装をpoemと同じ
//! 呼び出し形状でラップした薄いファサード)へ移行した。`Data<T>`抽出子は
//! 提供されないため、共有状態(`device`/`registry`)はハンドラ登録時の
//! クロージャで`Arc`をキャプチャする形に置き換えている——ロジック自体は
//! 移行前と同一。

mod bow_fallback;
mod cache_optimizer;
mod device_pool;
mod geo_content;
mod referrals;
mod web_search;
mod generation;
mod hardware;
mod intrusion_detection;
mod model_catalog;
mod nllb;
mod scoring;
mod security;
mod signatures;
mod tenants;

use std::sync::Arc;

use bytes::Bytes;
use opencuda_core::GpuDevice;
use opencuda_cpu::CpuDevice;
use open_runo_poem_compat::hyper_compat::{fixed_body, json_response, Params};
use open_runo_poem_compat::{delete, get, handler_fn, post, Handler, Json, PathParams, Request, Response, Route, Server, StatusCode, TcpListener};
use serde::{Deserialize, Serialize};
use tenants::{TenantInfo, TenantRegistry};

/// poemの`Response::builder().status(...).body(&str)`相当(RPoemは
/// レスポンスボディのcontent-type自動判定を持たないため、プレーン
/// テキスト応答はこの薄いヘルパーで組み立てる)。
fn text_response(status: StatusCode, body: impl Into<String>) -> Response {
    hyper::Response::builder()
        .status(status)
        .body(fixed_body(Bytes::from(body.into())))
        .expect("building a response from a fixed set of valid headers cannot fail")
}

fn html_page_response(html: &'static str) -> Response {
    hyper::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(fixed_body(Bytes::from(html)))
        .expect("building a response from a fixed set of valid headers cannot fail")
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
    /// 呼び出し元ドメイン(任意)。登録済みでなくても応答は返す
    /// (テナント登録は可用性の制約ではなく、利用状況可視化のための
    /// 構造という位置づけ)。
    #[serde(default)]
    tenant: Option<String>,
    /// 応答言語(任意、既定`"ja"`)。2026-07-22追記: e-gov.info自体は
    /// 13言語対応だが、本サービス経由の応答は従来日本語固定だった非対称を
    /// 解消するために追加(CLAUDE.md 2026-07-22 HANDOFF参照)。未送信の
    /// 既存呼び出し元との後方互換のため`"ja"`をデフォルトにする。
    #[serde(default = "default_lang")]
    lang: String,
}

fn default_lang() -> String {
    "ja".to_string()
}

/// 「分身の術」テナント登録有無の確認を全エンドポイントで統一する
/// (2026-07-26修正: 以前は`/v1/chat`のみがこのチェックを行い、
/// `/v1/classify-security`・`/v1/generate`は`tenant`フィールドを受け取り
/// ながらレジストリに一切問い合わせず単にログへ流すだけという非対称が
/// あった。テナント未登録でも応答は返す設計〈可用性優先〉は3エンドポイント
/// とも共通のまま、少なくとも「未登録テナントからの呼び出し」の可視化
/// だけは全エンドポイントで揃える)。
fn log_tenant_usage(endpoint: &str, tenant: &Option<String>, registry: &TenantRegistry) {
    if let Some(tenant) = tenant {
        if !registry.contains(tenant) {
            tracing::info!("{endpoint} request from unregistered tenant: {tenant}");
        } else {
            tracing::info!("{endpoint} request from tenant: {tenant}");
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    reply: String,
    engine: String,
    matched_intent: Option<&'static str>,
    /// 実際に返した応答の言語(`"ja"`または`"en"`、現状の対応言語)。
    reply_lang: &'static str,
    /// `true`の場合、リクエストされた`lang`に対応する翻訳が無かったため
    /// 英語へフォールバックしたことを示す(黙って日本語へ落とさない、
    /// このエコシステムの「graceful degradation, never silent」方針、
    /// CLAUDE.md参照)。
    lang_fallback: bool,
}

async fn chat(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<ChatRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("chat", &req.tenant, &registry);

    // scoring::classifyは、埋め込みモデル(models/multilingual-e5-small/)が
    // 使える場合はコサイン類似度分類を、モデル重みが無い・ロードに失敗した
    // 場合は自動的にbag-of-wordsドット積へフォールバックする(2026-07-25
    // 追加、詳細はscoring.rs/bow_fallback.rs参照)。engineには実際に使われた
    // 経路を常に正直に返す。
    let result = scoring::classify(&device, &req.message);
    match result.intent {
        Some(intent) => {
            let (reply, reply_lang, lang_fallback) = intent.reply_for(&req.lang);
            json_response(
                StatusCode::OK,
                &ChatResponse { reply: reply.to_string(), engine: result.engine, matched_intent: Some(intent.name), reply_lang, lang_fallback },
            )
        }
        None => {
            let (reply, reply_lang, lang_fallback) = scoring::fallback_reply_for(&req.lang);
            json_response(StatusCode::OK, &ChatResponse { reply: reply.to_string(), engine: result.engine, matched_intent: None, reply_lang, lang_fallback })
        }
    }
}

#[derive(Debug, Deserialize)]
struct ClassifySecurityRequest {
    /// 判定対象のコード片、または振る舞いの説明文。
    text: String,
    #[serde(default)]
    tenant: Option<String>,
}

#[derive(Debug, Serialize)]
struct StaticSignalDto {
    tag: String,
    description: String,
    category_hint: &'static str,
}

#[derive(Debug, Serialize)]
struct ClassifySecurityResponse {
    label: &'static str,
    description: &'static str,
    score: f32,
    is_suspicious: bool,
    engine: String,
    /// 2026-07-26追加: embeddingコサイン類似度とは別に検出された、決定的に
    /// 検証可能な静的特徴(既知シグネチャ一致・エントロピー・API組み合わせ)。
    /// 空配列はそれらが何も検出されなかった(embeddingのみの判定)ことを示す。
    static_signals: Vec<StaticSignalDto>,
}

/// RS-Guardの「AI二次判定」用エンドポイント。静的ルールに引っかからない
/// コード片を受け取り、マルウェア/スパイウェア/常駐・自動巡回/正常の
/// いずれに最も近いかを判定する。2026-07-26更新: 汎用文埋め込みの
/// コサイン類似度に加え、`signatures::analyze`による決定的な静的特徴抽出
/// (既知マルウェア文字列シグネチャ・エンコード文字列のエントロピー・
/// 疑わしいAPI呼び出しの組み合わせ)を組み合わせて判定する。
/// **正直な開示**: それでも訓練済みマルウェア分類器ではなく、
/// 意味的類似度+決定的な文字列/数値ヒューリスティックの組み合わせに
/// すぎない。`engine`と`static_signals`にその内訳を常に明示する。
async fn classify_security(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<ClassifySecurityRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("classify-security", &req.tenant, &registry);
    // 2026-08-06修正: `engine`が実行経路(Vulkan/CPU)に関わらず常に
    // `-cpu`固定文字列だった粗を修正(CLAUDE.md 2026-08-05 HANDOFF参照)。
    // `scoring::dispatch_suffix`は`scoring::get_model()`内の`SPIRV_WIRED`を
    // 見るため、security側も同じ埋め込みモデルを共有している(`security.rs`
    // が`crate::scoring::embed`を呼ぶ設計)ことからそのまま使える。
    let suffix = scoring::dispatch_suffix(&device);
    match security::classify_security(&device, &req.text) {
        Ok(v) => {
            let had_static_signals = !v.static_signals.is_empty();
            json_response(
                StatusCode::OK,
                &ClassifySecurityResponse {
                    label: v.label,
                    description: v.description,
                    score: v.score,
                    is_suspicious: v.is_suspicious,
                    engine: if had_static_signals {
                        format!("embedding-cosine-heuristic-v0-open-cuda-bert{suffix}+static-signatures-v1-opencuda-cpu")
                    } else {
                        format!("embedding-cosine-heuristic-v0-open-cuda-bert{suffix}")
                    },
                    static_signals: v
                        .static_signals
                        .into_iter()
                        .map(|s| StaticSignalDto { tag: s.tag, description: s.description, category_hint: s.category_hint })
                        .collect(),
                },
            )
        }
        Err(err) => {
            tracing::warn!("classify_security failed: {err}");
            // 判定不能なときは黙って「安全」とは言わない——is_suspicious=false
            // だが、engineでエラーだったことを正直に示し、呼び出し側が
            // 静的結果のみで判断できるようにする。
            json_response(
                StatusCode::OK,
                &ClassifySecurityResponse {
                    label: "unknown",
                    description: "classification failed; rely on static findings only",
                    score: 0.0,
                    is_suspicious: false,
                    engine: format!("embedding-cosine-heuristic-v0-open-cuda-bert{suffix}-error"),
                    static_signals: Vec::new(),
                },
            )
        }
    }
}

#[derive(Debug, Deserialize)]
struct ClassifyTrafficRequest {
    /// RS-SmartTCP側が観測したトラフィック特徴量から組み立てた短い説明文
    /// (例: "one source IP probed 60 ports in 3 seconds")。生パケットや
    /// 数値特徴量そのものではなく、既に自然文化されたものを受け取る設計。
    description: String,
    #[serde(default)]
    tenant: Option<String>,
}

#[derive(Debug, Serialize)]
struct ClassifyTrafficResponse {
    label: &'static str,
    description: &'static str,
    score: f32,
    is_suspicious: bool,
    engine: String,
}

/// RS-SmartTCPの「AI侵入検知」プラグイン用エンドポイント(2026-08-11新設)。
/// `classify_security`と同じ`open-cuda-bert`埋め込み+コサイン類似度の
/// 仕組みで、トラフィック特徴量の説明文をポートスキャン/SYNフラッド/
/// ブルートフォース/データ持ち出し/正常のいずれかに分類する。
/// **正直な開示**: 攻撃トラフィックの実データで訓練した専用分類器では
/// なく、汎用文埋め込みモデルによる意味的類似度のヒューリスティック
/// (`intrusion_detection.rs`冒頭のモジュールdoc参照)。
async fn classify_traffic(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<ClassifyTrafficRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("security/classify-traffic", &req.tenant, &registry);
    let suffix = scoring::dispatch_suffix(&device);
    match intrusion_detection::classify_traffic(&device, &req.description) {
        Ok(v) => json_response(
            StatusCode::OK,
            &ClassifyTrafficResponse {
                label: v.label,
                description: v.description,
                score: v.score,
                is_suspicious: v.is_suspicious,
                engine: format!("embedding-cosine-heuristic-v0-open-cuda-bert{suffix}"),
            },
        ),
        Err(err) => {
            tracing::warn!("classify_traffic failed: {err}");
            json_response(
                StatusCode::OK,
                &ClassifyTrafficResponse {
                    label: "unknown",
                    description: "classification failed",
                    score: 0.0,
                    is_suspicious: false,
                    engine: format!("embedding-cosine-heuristic-v0-open-cuda-bert{suffix}-error"),
                },
            )
        }
    }
}

/// `E_GOV_LLM_ADMIN_TOKEN`が設定されていれば`x-admin-token`ヘッダとの
/// 一致を要求する。未設定の場合は誰でも管理APIを呼べてしまうため、
/// 本番運用では必ず設定すること(`open-web-server`の`TenantRegistry`
/// 管理APIと同じ設計)。
fn check_admin_token(req: &Request) -> bool {
    match std::env::var("E_GOV_LLM_ADMIN_TOKEN") {
        Ok(expected) => req.headers().get("x-admin-token").and_then(|v| v.to_str().ok()) == Some(expected.as_str()),
        Err(_) => true,
    }
}

async fn admin_register_tenant(req: Request, registry: Arc<TenantRegistry>) -> Response {
    if !check_admin_token(&req) {
        return text_response(StatusCode::UNAUTHORIZED, "invalid admin token");
    }
    let Json(info): Json<TenantInfo> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    tracing::info!("registering tenant: {}", info.host);
    registry.register(info);
    text_response(StatusCode::OK, "ok")
}

async fn admin_list_tenants(req: Request, registry: Arc<TenantRegistry>) -> Response {
    if !check_admin_token(&req) {
        return text_response(StatusCode::UNAUTHORIZED, "invalid admin token");
    }
    json_response(StatusCode::OK, &registry.list())
}

async fn admin_remove_tenant(req: Request, params: Params, registry: Arc<TenantRegistry>) -> Response {
    if !check_admin_token(&req) {
        return text_response(StatusCode::UNAUTHORIZED, "invalid admin token");
    }
    let host = PathParams::from(params).get("host").unwrap_or("").to_string();
    if registry.remove(&host) {
        text_response(StatusCode::OK, "ok")
    } else {
        text_response(StatusCode::NOT_FOUND, "tenant not found")
    }
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    /// 生成の起点となるプロンプト(英語推奨——GPT-2 124MのBPE語彙は
    /// 英語中心の学習データに由来するため、日本語入力は語彙効率・品質共に
    /// 低下する。CLAUDE.md「正直な開示」参照)。
    prompt: String,
    /// 追加生成トークン数(既定16、上限128——CPU逐次デコードのため大きい値は
    /// 応答時間が線形に伸びる)。
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    tenant: Option<String>,
}

fn default_max_new_tokens() -> usize {
    16
}

const MAX_NEW_TOKENS_LIMIT: usize = 128;

#[derive(Debug, Serialize)]
struct GenerateResponse {
    /// `prompt`に続けて生成されたテキスト(プロンプト自体は含まない)。
    completion: String,
    /// 2026-07-27修正: `model_catalog`経由のホットスワップ後も実際に
    /// 使用中のモデルディレクトリ名を反映するよう動的化
    /// (`generation::engine_label(&device)`)——固定文字列のままだと、例えば
    /// gpt2-mediumへ切り替えた後も"gpt2-124m-..."と表示され続け不正直
    /// だったため。
    engine: String,
    /// 正直な開示メッセージ(常に返す、GPT-2系モデルの性能限界を毎回明示)。
    disclosure: &'static str,
}

#[derive(Debug, Serialize)]
struct GenerateErrorResponse {
    error: String,
    engine: String,
}

/// `open-cuda-llm::GptModel`(GPT-2 124M実重み)による自己回帰テキスト生成。
/// `/v1/chat`(意図分類、軽量・高速)とは別目的の別エンドポイント——
/// 意図分類と生成は無理に統合しない設計方針(CLAUDE.md参照)。
async fn generate(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<GenerateRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("generate", &req.tenant, &registry);
    // 2026-08-07追加: 空prompt(または空白のみ)は、以前はトークナイザで
    // 0トークンにエンコードされた後`generation::generate`内部の
    // `ensure!(!prompt_ids.is_empty(), ...)`まで進んでから失敗し、その
    // エラーがバックエンド障害と同じ`503 Service Unavailable`として
    // 返っていた——呼び出し側からは「サーバーが落ちている」のか
    // 「自分のリクエストが不正」なのか区別できず不便だった。クライアント
    // 起因の入力不備は`400 Bad Request`で即座に、モデル側の状態には触れず
    // 明確なメッセージで返す(APIの使いやすさ改善、`open-cuda`側の変更は
    // 不要でこのリポジトリ単体で完結する修正)。
    if req.prompt.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &GenerateErrorResponse { error: "prompt must not be empty".to_string(), engine: generation::engine_label(&device) },
        );
    }
    let max_new_tokens = req.max_new_tokens.clamp(1, MAX_NEW_TOKENS_LIMIT);
    match generation::generate(&device, &req.prompt, max_new_tokens) {
        Ok(completion) => json_response(
            StatusCode::OK,
            &GenerateResponse {
                completion,
                engine: generation::engine_label(&device),
                disclosure: "GPT-2 family models (124M-1.5B) are small 2019-era models, not comparable to modern commercial LLMs (e.g. GPT-4). \
                    This demonstrates self-contained text generation without an external LLM API contract, not state-of-the-art quality. \
                    Output may be grammatically fluent but is not guaranteed to be factually accurate.",
            },
        ),
        Err(err) => {
            tracing::warn!("generate failed: {err:#}");
            json_response(StatusCode::SERVICE_UNAVAILABLE, &GenerateErrorResponse { error: format!("{err:#}"), engine: generation::engine_label(&device) })
        }
    }
}

/// `POST /v1/generate-with-search` — ユーザー指示「open-englishは、人が
/// しゃべったり文字を入力したら、その都度Google検索するような仕様に
/// して」への対応(ブリッジ式: 入力文をそのままGoogle Custom Search
/// JSON APIへ問い合わせ、上位数件のタイトル+スニペットを生成プロンプトへ
/// コンテキストとして埋め込んでから`/v1/generate`と同じ生成処理を呼ぶ)。
///
/// **正直な開示**: (1) 検索は`ARUARU_LLM_GOOGLE_SEARCH_API_KEY`/
/// `ARUARU_LLM_GOOGLE_SEARCH_CX`環境変数が設定されている場合のみ実際に
/// 行われる——未設定時は検索無しで通常の`/v1/generate`相当にフォール
/// バックする(`used_search: false`で判別可能、サービス全体を壊さない)。
/// (2) 検索結果を埋め込んでもGPT-2系の小型モデルが実際にそれを踏まえた
/// 応答をするとは保証されない(モデル自体は変わらない、既存の`disclosure`
/// と同じ限界)。
#[derive(Debug, Deserialize)]
struct GenerateWithSearchRequest {
    prompt: String,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    tenant: Option<String>,
}

#[derive(Debug, Serialize)]
struct GenerateWithSearchResponse {
    completion: String,
    engine: String,
    disclosure: &'static str,
    used_search: bool,
    search_results: Vec<web_search::SearchResult>,
    /// `used_search=false`の理由を正直に開示する診断フィールド(ユーザー
    /// 指摘「早くBUG修正して」への対応——ターミナルログを見なくても
    /// レスポンスだけで原因が分かるようにした。APIキーの値自体は
    /// 決して含めない、エラーメッセージはHTTPステータス・パース失敗等の
    /// 情報のみ)。
    #[serde(skip_serializing_if = "Option::is_none")]
    search_error: Option<String>,
}

async fn generate_with_search(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<GenerateWithSearchRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("generate-with-search", &req.tenant, &registry);
    if req.prompt.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &GenerateErrorResponse { error: "prompt must not be empty".to_string(), engine: generation::engine_label(&device) },
        );
    }

    let (augmented_prompt, used_search, search_results, search_error) = if web_search::is_configured() {
        match web_search::search(&req.prompt, 3).await {
            Ok(results) if !results.is_empty() => {
                let context = web_search::format_results_as_context(&results);
                (format!("Reference information from a web search:\n{context}\n\n{}", req.prompt), true, results, None)
            }
            Ok(_) => (req.prompt.clone(), false, Vec::new(), Some("Google returned zero results for this query.".to_string())),
            Err(err) => {
                let msg = format!("{err:#}");
                tracing::warn!("google custom search failed, falling back to no-search generation: {msg}");
                (req.prompt.clone(), false, Vec::new(), Some(msg))
            }
        }
    } else {
        (req.prompt.clone(), false, Vec::new(), Some("Google Custom Search is not configured (no API key/cx set).".to_string()))
    };

    let max_new_tokens = req.max_new_tokens.clamp(1, MAX_NEW_TOKENS_LIMIT);
    match generation::generate(&device, &augmented_prompt, max_new_tokens) {
        Ok(completion) => json_response(
            StatusCode::OK,
            &GenerateWithSearchResponse {
                completion,
                engine: generation::engine_label(&device),
                disclosure: "GPT-2 family models (124M-1.5B) are small 2019-era models, not comparable to modern commercial LLMs (e.g. GPT-4). \
                    Web search results (if any) are embedded as extra context, but the model is not guaranteed to actually use them correctly. \
                    This demonstrates self-contained text generation augmented with live search, not state-of-the-art quality.",
                used_search,
                search_results,
                search_error,
            },
        ),
        Err(err) => {
            tracing::warn!("generate_with_search failed: {err:#}");
            json_response(StatusCode::SERVICE_UNAVAILABLE, &GenerateErrorResponse { error: format!("{err:#}"), engine: generation::engine_label(&device) })
        }
    }
}

/// `POST /v1/settings/google-search` — ブラウザの設定パネルから
/// Google Custom Search APIキー/検索エンジンID(cx)を保存する
/// (ユーザー指示「利用者がAPIキーの取得とCOPYペーストが簡単な機能を
/// 搭載して」への対応)。**正直な開示**: メモリ上にのみ保持し、
/// ディスクへの書き込み・ログ出力は一切行わない(プロセス再起動で
/// 消える設計、`web_search`モジュールdocコメント参照)。
#[derive(Debug, Deserialize)]
struct GoogleSearchSettingsRequest {
    api_key: String,
    cx: String,
}

#[derive(Debug, Serialize)]
struct GoogleSearchSettingsStatusResponse {
    /// APIキーの値自体は絶対に返さない(この構造体にキーの値を保持する
    /// フィールドを持たせないことで、実装ミスによる漏洩を型レベルで
    /// 防ぐ設計)。
    configured: bool,
}

async fn set_google_search_settings(req: Request) -> Response {
    let Json(body): Json<GoogleSearchSettingsRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    web_search::set_runtime_credentials(body.api_key, body.cx);
    json_response(StatusCode::OK, &GoogleSearchSettingsStatusResponse { configured: web_search::is_configured() })
}

async fn clear_google_search_settings() -> Response {
    web_search::clear_runtime_credentials();
    json_response(StatusCode::OK, &GoogleSearchSettingsStatusResponse { configured: web_search::is_configured() })
}

async fn get_google_search_settings_status() -> Response {
    json_response(StatusCode::OK, &GoogleSearchSettingsStatusResponse { configured: web_search::is_configured() })
}

#[derive(Debug, Deserialize)]
struct TranslateRequest {
    /// 翻訳元テキスト。
    text: String,
    /// 翻訳先言語コード(表示ラベル、例: "English"/"Italian"/"日本語"等)。
    /// ISO言語コードそのものではなく人間可読な言語名文字列でよい
    /// (プロンプトへそのまま埋め込むため)。
    target_lang: String,
    /// 翻訳元言語コード(任意、省略時はプロンプトに明記しない=モデルに
    /// 自動判定させる)。
    #[serde(default)]
    source_lang: Option<String>,
    #[serde(default)]
    tenant: Option<String>,
}

#[derive(Debug, Serialize)]
struct TranslateResponse {
    translation: String,
    engine: String,
    disclosure: String,
}

#[derive(Debug, Serialize)]
struct TranslateErrorResponse {
    error: String,
    engine: String,
}

/// `POST /v1/translate` — GPT-2系モデル(`generation::generate`)を翻訳
/// プロンプトで呼び出す薄いラッパー(2026-08-04新設、ユーザー指示
/// 「aruaru-llmに自動翻訳機能を持たせて」への対応)。
///
/// **正直な開示(最重要)**: GPT-2 124M-1.5Bは英語中心の学習データで
/// 事前学習された素の言語モデルであり、指示追従(instruction-following)
/// のファインチューニングも翻訳専用の学習も一切受けていない。
/// 「Translate X to Y:」のようなプロンプトへの続き生成は、しばしば
/// 翻訳ではなく無関係な文の継続になる(特に日本語→非英語、非英語→
/// 非英語のような英語を経由しない組み合わせで品質が大きく劣化する)。
/// これは`/v1/generate`と同じ土台の上に構築した実装であり、専用の
/// 翻訳モデル(NLLB/M2M100等)ではないことを常にレスポンスへ明記する。
async fn translate(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<TranslateRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("translate", &req.tenant, &registry);

    // 2026-08-07追加: `/v1/generate`と同じ理由(下記コメント参照)で、
    // 空`text`・空`target_lang`は`400 Bad Request`で即座に返す
    // (以前は空`text`がGPT-2フォールバック経路で`503`扱いになっていた)。
    if req.text.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &TranslateErrorResponse { error: "text must not be empty".to_string(), engine: generation::engine_label(&device) },
        );
    }
    if req.target_lang.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &TranslateErrorResponse { error: "target_lang must not be empty".to_string(), engine: generation::engine_label(&device) },
        );
    }

    // `nllb-translate` feature有効時は、専用翻訳モデル(M2M100)をまず
    // 試みる(2026-08-04追加、実HTTP検証でGPT-2流用実装が実用に耐えないと
    // 判明したための対応、CLAUDE.md参照)。モデル未対応の言語指定・
    // 未ロード失敗時は、既存のGPT-2流用実装へ安全にフォールバックする
    // (`nllb::translate_with_nllb`は`nllb-translate`feature無効時は常に
    // `Err`を返す設計のため、この分岐はfeature無効ビルドでも無条件に
    // GPT-2流用実装のパスを通り、既存動作を一切変えない)。
    if let Ok(translation) = nllb::translate_with_nllb(&req.text, &req.target_lang, req.source_lang.as_deref()) {
        return json_response(
            StatusCode::OK,
            &TranslateResponse {
                translation,
                engine: "m2m100-rust-bert-v0".to_string(),
                disclosure: "This translation was produced by a dedicated open-source translation model (M2M100 via rust-bert), \
                    not the GPT-2 fallback. Quality is substantially better but still not guaranteed to be publication-ready for \
                    all language pairs; spot-check before publishing."
                    .to_string(),
            },
        );
    }

    let prompt = match &req.source_lang {
        Some(src) => format!("Translate the following text from {src} to {}:\n{}\nTranslation:", req.target_lang, req.text),
        None => format!("Translate the following text to {}:\n{}\nTranslation:", req.target_lang, req.text),
    };
    // 翻訳は継続生成が長すぎると無関係な文へ発散しやすいため、
    // /v1/generateの既定(16)より長め・上限より短めの64に固定する。
    match generation::generate(&device, &prompt, 64) {
        Ok(completion) => json_response(
            StatusCode::OK,
            &TranslateResponse {
                translation: completion.trim().to_string(),
                engine: generation::engine_label(&device),
                disclosure: format!(
                    "This endpoint reuses the GPT-2 family text-generation engine (124M-1.5B, 2019-era, English-centric \
                    pretraining) with a translation-style prompt — it is NOT a dedicated translation model (e.g. NLLB/M2M100) and has \
                    received no instruction-following or translation-specific fine-tuning. Quality is unreliable, especially for \
                    non-English source/target language pairs; always spot-check output before publishing it. \
                    {}",
                    if nllb::is_available() {
                        "(nllb-translate feature IS compiled into this build but failed to produce a translation for this request — \
                        check target_lang is a supported language name, or see server logs for the underlying M2M100 load/inference error.)"
                    } else {
                        "(This build was compiled WITHOUT the nllb-translate feature; rebuild with --features nllb-translate to use \
                        the dedicated M2M100 translation model instead of this GPT-2 fallback.)"
                    }
                ),
            },
        ),
        Err(err) => {
            tracing::warn!("translate failed: {err:#}");
            json_response(StatusCode::SERVICE_UNAVAILABLE, &TranslateErrorResponse { error: format!("{err:#}"), engine: generation::engine_label(&device) })
        }
    }
}

/// `GET /v1/models/catalog` — インストール可能なGPT-2アーキテクチャ互換
/// モデルの一覧(ダウンロード前、どのモデルが選択可能かを提示するため)。
#[derive(Debug, Serialize)]
struct CatalogResponse {
    models: &'static [model_catalog::CatalogEntry],
    installed_ids: Vec<&'static str>,
    /// 現在実際に生成に使われているモデルの読み込み元ディレクトリ
    /// (まだ一度もロードしていなければ`None`、2026-07-27追加)。
    active_model_dir: Option<String>,
    /// 正直な開示: このエンジンがロードできるのはGPT-2アーキテクチャ
    /// 互換モデルのみで、Llama/Mistral/Qwen等アーキテクチャの異なる
    /// オープンソースLLMは対象外であることを、APIレスポンス自体にも
    /// 明記する(UIがこの文言をそのまま表示できるように)。
    disclosure_ja: &'static str,
}

/// `POST /v1/models/optimize-cache` — 東芝SBM(シミュレーテッド分岐)で
/// 実際に解く、ディスク容量予算下でのモデルキャッシュ選択(0/1ナップサック
/// 問題)。**正直な開示**: この規模(カタログ5件)であれば全探索・動的
/// 計画法でも瞬時に厳密解が求まり、SBMを使う実用上の必要性は薄い——
/// これは「SBMの動作実証をこのサービスの実際の意思決定パスへ配線する」
/// ことを目的とした機能であり、`aruaru_llm::cache_optimizer`のモジュール
/// docコメントに詳細な調査結果・限界を明記している。**advisory専用**
/// (実際のディスク削除は行わない、`evict`は「削除を推奨するモデルID一覧」
/// を返すのみ)。
#[derive(Debug, Deserialize)]
struct OptimizeCacheRequest {
    budget_mb: u32,
    #[serde(default)]
    value_overrides: std::collections::HashMap<String, f64>,
    #[serde(default)]
    seed: Option<u64>,
}

async fn optimize_model_cache_handler(req: Request) -> Response {
    let body = match Json::<OptimizeCacheRequest>::from_body(req).await {
        Ok(Json(b)) => b,
        Err(resp) => return resp,
    };
    let entries: Vec<(&str, u32)> =
        model_catalog::CATALOG.iter().map(|e| (e.id, e.approx_size_mb)).collect();
    let seed = body.seed.unwrap_or(0xC0FFEE);
    let result = cache_optimizer::optimize_model_cache(&entries, body.budget_mb, &body.value_overrides, seed);
    json_response(StatusCode::OK, &result)
}

async fn list_model_catalog() -> Response {
    let models_root = model_catalog::models_root();
    json_response(
        StatusCode::OK,
        &CatalogResponse {
            models: model_catalog::CATALOG,
            installed_ids: model_catalog::installed_ids(&models_root),
            active_model_dir: generation::active_model_dir().map(|d| d.to_string_lossy().to_string()),
            disclosure_ja: "このカタログはGPT-2アーキテクチャ互換モデルのみを対象としています。\
                Llama/Mistral/Qwen等、異なるアーキテクチャのオープンソースLLMは現在のエンジンでは\
                ロードできません(config.json/model.safetensors/tokenizer.jsonの3ファイル構成で\
                GPT-2のテンソル名規約に従うモデルのみ対応)。",
        },
    )
}

#[derive(Debug, Deserialize)]
struct InstallModelRequest {
    /// `model_catalog::CatalogEntry::id`のいずれか。
    id: String,
}

#[derive(Debug, Serialize)]
struct InstallModelResponse {
    id: String,
    dir: String,
    message_ja: String,
}

#[derive(Debug, Serialize)]
struct InstallModelErrorResponse {
    error: String,
}

#[derive(Debug, Deserialize)]
struct SelectModelRequest {
    /// `model_catalog::CatalogEntry::id`のいずれか(ダウンロード済みで
    /// あることが前提、`GET /v1/models/catalog`の`installed_ids`参照)。
    id: String,
}

#[derive(Debug, Serialize)]
struct SelectModelResponse {
    id: String,
    dir: String,
    message_ja: String,
}

/// `POST /v1/models/select` — インストール済みモデルへプロセス再起動
/// 無しで切り替える(2026-07-27追加、`generation::select_model`参照)。
/// **読み込みに成功した場合のみ**現在使用中のモデルを置き換える——
/// 指定した`id`が未インストール・破損している等で読み込みに失敗した
/// 場合、現在動作中のモデルはそのまま維持され、サービスは壊れない。
async fn select_model(req: Request) -> Response {
    let Json(req): Json<SelectModelRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let dest_dir = model_catalog::models_root().join(&req.id);
    let dir_for_task = dest_dir.clone();
    let result = tokio::task::spawn_blocking(move || generation::select_model(dir_for_task)).await;
    match result {
        Ok(Ok(())) => json_response(
            StatusCode::OK,
            &SelectModelResponse {
                id: req.id.clone(),
                dir: dest_dir.to_string_lossy().to_string(),
                message_ja: format!("使用するモデルを{}に切り替えました(プロセス再起動は不要です)。", req.id),
            },
        ),
        Ok(Err(e)) => {
            tracing::warn!("select_model({}) failed: {e:#}", req.id);
            json_response(StatusCode::BAD_REQUEST, &InstallModelErrorResponse { error: format!("{e:#}") })
        }
        Err(e) => json_response(StatusCode::INTERNAL_SERVER_ERROR, &InstallModelErrorResponse { error: format!("select_model task panicked: {e}") }),
    }
}

/// `POST /v1/models/install` — カタログから選択したモデルをHugging Face
/// からダウンロードし、`{ARUARU_LLM_MODELS_ROOT}/{id}/`へ配置する。
/// ダウンロード自体は必ずこのエンドポイントへの明示的なリクエストからのみ
/// 起動する(サーバー起動時の自動ダウンロードは行わない設計)。
/// **正直な開示**: ダウンロード完了後、実際にそのモデルを使うには
/// `ARUARU_LLM_GPT2_DIR`をこのディレクトリへ向けて`/v1/generate`を叩く
/// 側のプロセスを再起動する必要がある(現状のロード方式が起動時
/// `OnceLock`のため、実行中のホットスワップには対応していない)。
async fn install_model(req: Request) -> Response {
    let Json(req): Json<InstallModelRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let Some(entry) = model_catalog::find(&req.id) else {
        return json_response(StatusCode::BAD_REQUEST, &InstallModelErrorResponse { error: format!("unknown model id: {}", req.id) });
    };

    let dest_dir = model_catalog::models_root().join(entry.id);
    match model_catalog::install(entry, &dest_dir).await {
        Ok(()) => json_response(
            StatusCode::OK,
            &InstallModelResponse {
                id: entry.id.to_string(),
                dir: dest_dir.to_string_lossy().to_string(),
                message_ja: format!(
                    "{}のダウンロードが完了しました。使用するには ARUARU_LLM_GPT2_DIR={} を設定してプロセスを再起動してください。",
                    entry.display_name_ja,
                    dest_dir.to_string_lossy()
                ),
            },
        ),
        Err(err) => {
            tracing::warn!("install_model({}) failed: {err:#}", entry.id);
            json_response(StatusCode::BAD_GATEWAY, &InstallModelErrorResponse { error: format!("{err:#}") })
        }
    }
}

/// `GET /v1/recommend` — ハードウェア検出+推奨モデルサイズ算出のみ
/// (ダウンロードは行わない、2026-07-27新設)。`hardware::recommend()`が
/// `open-directx`(DXGI)/`open-cuda`(Vulkan)いずれかの実GPU検出を試み、
/// VRAM容量から推奨モデルIDを算出する(正直な開示は`hardware.rs`
/// モジュールdoc参照)。
async fn recommend_model() -> Response {
    json_response(StatusCode::OK, &hardware::recommend())
}

#[derive(Debug, Serialize)]
struct RecommendAndDownloadResponse {
    recommendation: hardware::Recommendation,
    already_installed: bool,
    switched_to_recommended: bool,
    message_ja: String,
}

/// `POST /v1/recommend-and-download` — 「お勧めLLMをダウンロード」ボタンの
/// 受け口(2026-07-27新設)。(a)ハードウェア検出→推奨モデルサイズ算出、
/// (b)未ダウンロードならHugging Faceから取得(`model_catalog::install`、
/// 既にダウンロード済みなら再取得しない=冪等)、(c)ダウンロード
/// (または既存)完了後、`generation::select_model`でホットスワップし
/// `/v1/generate`が直ちにこのモデルを使えるようにする。**正直な開示**:
/// 失敗時(ダウンロード失敗・切り替え失敗)は現在動作中のモデルを維持
/// したまま、エラー内容を正直に返す(サービスを壊さない設計、
/// `select_model`と同じ思想)。
async fn recommend_and_download() -> Response {
    let rec = hardware::recommend();
    let Some(entry) = model_catalog::find(rec.recommended_model_id) else {
        return json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &InstallModelErrorResponse { error: format!("recommended id {} not in catalog (internal bug)", rec.recommended_model_id) },
        );
    };

    let models_root = model_catalog::models_root();
    let dest_dir = models_root.join(entry.id);
    let already_installed = model_catalog::installed_ids(&models_root).contains(&entry.id);

    if !already_installed {
        if let Err(err) = model_catalog::install(entry, &dest_dir).await {
            tracing::warn!("recommend_and_download: install({}) failed: {err:#}", entry.id);
            return json_response(StatusCode::BAD_GATEWAY, &InstallModelErrorResponse { error: format!("{err:#}") });
        }
    }

    let dir_for_task = dest_dir.clone();
    let switch_result = tokio::task::spawn_blocking(move || generation::select_model(dir_for_task)).await;
    let (switched_to_recommended, message_ja) = match switch_result {
        Ok(Ok(())) => (true, format!("推奨モデル{}({})のダウンロードと切り替えが完了しました。/v1/generateで使用中です。", entry.display_name_ja, entry.id)),
        Ok(Err(e)) => {
            tracing::warn!("recommend_and_download: select_model({}) failed: {e:#}", entry.id);
            (false, format!("ダウンロードは完了しましたが、切り替えに失敗しました({e:#})。現在動作中のモデルは維持されています。"))
        }
        Err(e) => (false, format!("切り替え処理がパニックしました({e})。現在動作中のモデルは維持されています。")),
    };

    json_response(StatusCode::OK, &RecommendAndDownloadResponse { recommendation: rec, already_installed, switched_to_recommended, message_ja })
}

/// 現在アクティブなモデルのカタログID(2026-07-27追加、「一つ大きい/
/// 小さいモデルをダウンロード」ボタンが「今どのサイズを基準に1段階
/// 動かすか」を判断するために使う)。`model_catalog::install`が
/// `models_root().join(entry.id)`という規約でディレクトリを作る
/// (`recommend_and_download`参照)ため、そのディレクトリ名の最後の
/// パス要素がそのままカタログIDになる。まだ一度もモデルを切り替えて
/// いない(起動時の既定モデルのまま)場合、既定モデルの読み込み元
/// ディレクトリ名も偶然`"gpt2"`という同じ規約に従う
/// (`generation::default_model_dir`参照)ため、素直にディレクトリ名を
/// 使うだけでよい。取得できない場合は安全側の"gpt2"を基準にする。
fn current_model_id() -> String {
    generation::active_model_dir().and_then(|dir| dir.file_name().map(|n| n.to_string_lossy().to_string())).unwrap_or_else(|| "gpt2".to_string())
}

#[derive(Debug, Serialize)]
struct StepModelResponse {
    from_id: String,
    to_id: Option<String>,
    already_installed: bool,
    switched: bool,
    message_ja: String,
}

/// `direction`に応じて「今使っているモデルより1段階大きい/小さい」
/// カタログエントリを探し、未取得ならダウンロードした上でホットスワップ
/// する共通処理(`POST /v1/download-larger`/`POST /v1/download-smaller`
/// 双方から呼ぶ、2026-07-27新設)。既に最大/最小サイズの場合は、正直に
/// その旨を伝えて何もしない(`to_id: None`、`switched: false`)。
async fn step_model_size(direction_larger: bool) -> Response {
    let from_id = current_model_id();
    let next = if direction_larger { model_catalog::next_larger(&from_id) } else { model_catalog::next_smaller(&from_id) };

    let Some(entry) = next else {
        let label = if direction_larger { "最大" } else { "最小" };
        return json_response(
            StatusCode::OK,
            &StepModelResponse {
                from_id: from_id.clone(),
                to_id: None,
                already_installed: true,
                switched: false,
                message_ja: format!("現在の{from_id}は既にカタログ内で{label}サイズです。これ以上{}できません。", if direction_larger { "大きくする" } else { "小さくする" }),
            },
        );
    };

    let models_root = model_catalog::models_root();
    let dest_dir = models_root.join(entry.id);
    let already_installed = model_catalog::installed_ids(&models_root).contains(&entry.id);

    if !already_installed {
        if let Err(err) = model_catalog::install(entry, &dest_dir).await {
            tracing::warn!("step_model_size: install({}) failed: {err:#}", entry.id);
            return json_response(StatusCode::BAD_GATEWAY, &InstallModelErrorResponse { error: format!("{err:#}") });
        }
    }

    let dir_for_task = dest_dir.clone();
    let switch_result = tokio::task::spawn_blocking(move || generation::select_model(dir_for_task)).await;
    let (switched, message_ja) = match switch_result {
        Ok(Ok(())) => (true, format!("{from_id} から {}({}) へ切り替えました。/v1/generateで使用中です。", entry.display_name_ja, entry.id)),
        Ok(Err(e)) => {
            tracing::warn!("step_model_size: select_model({}) failed: {e:#}", entry.id);
            (false, format!("ダウンロードは完了しましたが、切り替えに失敗しました({e:#})。現在動作中の{from_id}は維持されています。"))
        }
        Err(e) => (false, format!("切り替え処理がパニックしました({e})。現在動作中の{from_id}は維持されています。")),
    };

    json_response(StatusCode::OK, &StepModelResponse { from_id, to_id: Some(entry.id.to_string()), already_installed, switched, message_ja })
}

/// `POST /v1/download-larger` — 現在のモデルより1段階大きいカタログ
/// エントリをダウンロード・切り替える(2026-07-27新設、ユーザー指示
/// 「一つ大きなモデルをダウンロードする、と言うボタンも作って」)。
async fn download_larger_model() -> Response {
    step_model_size(true).await
}

/// `POST /v1/download-smaller` — 現在のモデルより1段階小さいカタログ
/// エントリをダウンロード・切り替える(2026-07-27新設、ユーザー指示
/// 「一つ小さなモデルをダウロードする、と言うボタンも作って」)。
async fn download_smaller_model() -> Response {
    step_model_size(false).await
}

/// 最小限の静的HTML UI(2026-07-27新設、ユーザー指示「お勧めLLMを
/// ダウンロード」ボタン1つ+進捗表示+生成テスト導線)。Tauri/Node.js/
/// TypeScript不使用、Rust側でのインライン静的HTML配信(既存エコシステム
/// 方針通り、過剰実装を避けフレームワーク追加無し)。
const INDEX_HTML: &str = include_str!("../static/index.html");

async fn index_page() -> Response {
    html_page_response(INDEX_HTML)
}

async fn healthz() -> Response {
    text_response(StatusCode::OK, "ok")
}

/// 日本の都道府県1件・米国の州1件・世界の首都1件をランダムに返す
/// (open-englishの自己紹介トレーニング用、2026-08-11新設)。
async fn geo_random() -> Response {
    json_response(StatusCode::OK, &geo_content::random_entry().await)
}

/// 富士山の安全上の注意+山小屋(吉田口ルート)予約先一覧を返す
/// (2026-08-11新設、ユーザー指示「富士山の話題が出たら日本語と英語で
/// 紹介して」への対応)。
async fn geo_fuji() -> Response {
    json_response(StatusCode::OK, &geo_content::fuji_info())
}

#[derive(Debug, Deserialize)]
struct GeoToursRequest {
    place: String,
}

#[derive(Debug, Serialize)]
struct GeoToursResponse {
    configured: bool,
    web_results: Vec<web_search::SearchResult>,
    youtube_search_url: String,
    disclosure_ja: String,
    disclosure_en: String,
}

/// 「日本も世界も観光で訪れるなら、観光ツアーの紹介とオンライン予約を
/// その都度検索して」への対応(2026-08-11新設)。既存のGoogle Custom
/// Search連携(`web_search.rs`、`/v1/generate-with-search`と共通の
/// APIキー設定)をそのまま再利用し、`"<place> 観光ツアー オンライン予約"`
/// で検索する。**正直な開示**: ユーザー自身のGoogle Search APIキー設定
/// (`POST /v1/settings/google-search`)が必要——未設定の場合は
/// `configured: false`+空の結果を正直に返す(黙って無関係な結果を
/// 返さない)。YouTube検索は同様のAPI契約が別途必要なため、実際の検索
/// 結果ではなくYouTube検索結果ページへの直リンク(URLエンコード済み)を
/// 返す設計とした(Google Custom Search同様の実装は今回のスコープ外、
/// 誇張しない)。
async fn geo_tours(req: Request) -> Response {
    let Json(body): Json<GeoToursRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let place = body.place.trim();
    if place.is_empty() {
        return json_response(StatusCode::BAD_REQUEST, &serde_json::json!({"error": "place must not be empty"}));
    }

    let youtube_query = format!("{place} 観光ツアー tour");
    let youtube_search_url = format!(
        "https://www.youtube.com/results?search_query={}",
        urlencoding_encode(&youtube_query)
    );

    if !web_search::is_configured() {
        return json_response(
            StatusCode::OK,
            &GeoToursResponse {
                configured: false,
                web_results: vec![],
                youtube_search_url,
                disclosure_ja: "Google検索(観光ツアー・オンライン予約)は未設定です。設定パネルからAPIキーとcxを保存してください。".to_string(),
                disclosure_en: "Google Search for tourism tours/online booking is not configured yet. Save your API key and cx in the settings panel.".to_string(),
            },
        );
    }

    let query = format!("{place} 観光ツアー オンライン予約 tour booking");
    let web_results = web_search::search(&query, 5).await.unwrap_or_default();
    json_response(
        StatusCode::OK,
        &GeoToursResponse {
            configured: true,
            web_results,
            youtube_search_url,
            disclosure_ja: "検索結果は外部サイト(Google)由来です。実際の予約はリンク先の各事業者サイトで行ってください。".to_string(),
            disclosure_en: "Results are sourced from an external site (Google). Complete any actual booking on the linked provider's own site.".to_string(),
        },
    )
}

#[derive(Debug, Deserialize)]
struct ReferralsCheckRequest {
    text: String,
}

#[derive(Debug, Serialize)]
struct ReferralsCheckResponse {
    matched: bool,
    referrals: Option<referrals::ReferralInfo>,
}

/// 発話文が就職・転職・観光の話題かどうかを判定し、該当すれば
/// aruaru.tokyo/nasa.tokyo/audiocafe.tokyo(aruaru・aruaru-lady)への
/// 紹介情報を返す(2026-08-11新設)。
async fn referrals_check(req: Request) -> Response {
    let Json(body): Json<ReferralsCheckRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let matched = referrals::mentions_career_or_tourism(&body.text);
    json_response(
        StatusCode::OK,
        &ReferralsCheckResponse {
            matched,
            referrals: if matched { Some(referrals::career_and_tourism_referrals()) } else { None },
        },
    )
}

/// 簡易URLエンコード(クエリ文字列用、外部クレート非依存)。
fn urlencoding_encode(s: &str) -> String {
    let mut out = String::new();
    for byte in s.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[derive(Debug, Deserialize)]
struct GeoLookupRequest {
    country: String,
}

/// 「今度オーストラリアに旅行/出張の予定がある」のような発話向けの
/// 国名検索(`{"country": "Australia"}`または`{"country": "日本"}`)。
async fn geo_lookup(req: Request) -> Response {
    let Json(body): Json<GeoLookupRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    json_response(StatusCode::OK, &geo_content::lookup_country(&body.country).await)
}

/// 引数を取らないハンドラを`handler_fn`のシグネチャへ橋渡しする。
fn plain(f: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> + Send + Sync + 'static) -> Handler {
    handler_fn(move |_req, _params| f())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // マルチコア/マルチスレッド前提: #[tokio::main]の既定フレーバーは
    // multi_thread(current_threadへの明示的固定はしていない)。CPU計算
    // (bag-of-wordsスコアリング)自体はopencuda-cpuのrayonが
    // 利用可能な全論理コアへ並列ディスパッチする(`CpuDevice::new`が
    // `std::thread::available_parallelism()`から検出)。
    // デバイス選択(2026-08-04追記、CLAUDE.md 2026-08-04 HANDOFF参照):
    // `real-vulkan` feature(既定無効のオプトイン)が有効な場合のみ
    // `opencuda_vulkan::real::VulkanDevice`へ切り替える。VulkanDeviceの
    // 構築(実GPU列挙・論理デバイス作成)に失敗した場合は、サービスを
    // 落とさずCPUへ安全側フォールバックする(既存の「サービスを壊さない」
    // 設計方針を踏襲、hardware.rsのGPU検出失敗時フォールバックと同じ考え方)。
    // CPU+GPU同時並列稼働(2026-08-15、ユーザー指示「CPU+システムメモリ+
    // GPUを同時に並列並行で動作させて」への対応、device_pool.rs参照)。
    // 従来は「VulkanDevice構築成功ならGPU、失敗ならCPU」という排他選択
    // だったが、CPU(rayon並列)は常にプールへ加え、GPU構築に成功した
    // 場合はCPUを置き換えるのではなく**追加**する。DevicePoolがリクエスト
    // ごとにラウンドロビンで両方へ振り分けるため、同時に複数リクエストが
    // 来た場合CPU担当分とGPU担当分が実際に並行して稼働する。
    let cpu_device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    let mut pool_devices: Vec<Arc<dyn GpuDevice>> = vec![Arc::clone(&cpu_device)];
    #[cfg(feature = "real-vulkan")]
    {
        match opencuda_vulkan::real::VulkanDevice::new(0) {
            Ok(vulkan_device) => {
                tracing::info!(
                    "real-vulkan feature enabled: adding VulkanDevice ({}) to the device pool alongside CpuDevice",
                    vulkan_device.info().name
                );
                pool_devices.push(vulkan_device);
            }
            Err(err) => {
                tracing::warn!(
                    "real-vulkan feature enabled but VulkanDevice::new failed ({err}); running CPU-only"
                );
            }
        }
    }
    let device_pool = device_pool::DevicePool::new(pool_devices);
    tracing::info!(
        "aruaru-llm device pool: {} device(s) — {}",
        device_pool.device_count(),
        device_pool.device_names().join(", ")
    );
    // warmup(モデルロード・埋め込みキャッシュ)は単一デバイス
    // (CpuDevice)で行う——起動時1回きりの処理であり、CPU/GPU双方で
    // 二重にロードする意味は無いため。実際のリクエスト処理は
    // `device_pool.next_device()`でラウンドロビン分散する。
    let device = cpu_device;

    // 翻訳プラグイン(nllb.rs、M2M100/rust-bert、2026-08-04追加)の状態を
    // 起動時に明示ログ出力する。このプラグインはCargo feature
    // (`nllb-translate`、既定オフ)としてのみ存在し、ビルド時にfeature
    // フラグを付けるかどうかが「インストール/アンインストール」に相当
    // する(実行時のプラグイン着脱ではなく、ビルド成果物自体に翻訳
    // モデル依存〈tch/libtorch〉が含まれるかどうかで切り替わる設計、
    // ユーザー指示「翻訳部分だけプラグインという形にして、必要な人だけ
    // インストール/アンインストールできるように」への対応)。
    if nllb::is_available() {
        tracing::info!("translation plugin: ENABLED (M2M100 via rust-bert) — built with --features nllb-translate");
    } else {
        tracing::info!("translation plugin: not installed (GPT-2 fallback only) — rebuild with --features nllb-translate to install it");
    }

    // コールドスタート対策(2026-07-22追記、CLAUDE.md HANDOFF参照):
    // open-cuda-bertのモデルロード+インテントembedding計算(数秒)を、
    // サーバがTCP接続を受け付け始める前にここで前倒しで済ませておく。
    // これをやらないと「実際のリクエストが来て初めてOnceLockへロードする」
    // ことになり、e-gov.info等の呼び出し元タイムアウト(実測3秒)を
    // 超える初回リクエスト遅延が発生する(実際に観測済み)。
    {
        let warmup_started = std::time::Instant::now();
        match scoring::warmup(&device) {
            Ok(()) => tracing::info!("warmup complete in {:?} (model loaded, intent embeddings cached)", warmup_started.elapsed()),
            Err(err) => tracing::warn!("warmup failed (will retry lazily on first request): {err}"),
        }
        // セキュリティ分類のカテゴリ代表ベクトルも起動時に前倒しキャッシュ。
        match security::warmup(&device) {
            Ok(()) => tracing::info!("security classifier warmup complete (category embeddings cached)"),
            Err(err) => tracing::warn!("security warmup failed (will retry lazily on first request): {err}"),
        }
        // RS-SmartTCPの「AI侵入検知」プラグイン用カテゴリ代表ベクトルも
        // 起動時に前倒しキャッシュ(2026-08-11追加)。
        match intrusion_detection::warmup(&device) {
            Ok(()) => tracing::info!("intrusion detection classifier warmup complete (category embeddings cached)"),
            Err(err) => tracing::warn!("intrusion detection warmup failed (will retry lazily on first request): {err}"),
        }
        // GPT-2 124M実重み(548MB)のロードも起動時に前倒し(2026-07-25追加)。
        // 失敗しても致命的ではない(/v1/generateへの初回リクエスト時に再試行)。
        match generation::warmup() {
            Ok(()) => tracing::info!("generation (GPT-2 124M) warmup complete"),
            Err(err) => tracing::warn!("generation warmup failed (will retry lazily on first /v1/generate request): {err}"),
        }
    }

    // 地理・観光DB(2026-08-11追加): aruaru-dbへベストエフォートでseedを
    // 投入する(接続できない/未設定ならログのみで正常起動を継続、
    // geo_content.rsのモジュールdoc参照)。
    geo_content::seed_database_if_configured().await;

    // 2026-07-27追記(使いやすさ改善): GPU検出feature(hw-detect-vulkan/
    // hw-detect-directx)はいずれも既定offのため、何も知らずにビルドした
    // ユーザーは常にCPU-onlyフォールバック(最小モデル固定推奨)に静かに
    // 誘導されてしまう——起動ログでその旨と有効化方法を明示し、この
    // 「気づかれないまま」の状態を防ぐ。
    #[cfg(not(any(feature = "hw-detect-vulkan", feature = "hw-detect-directx")))]
    tracing::info!(
        "GPU detection is disabled in this build (hw-detect-vulkan/hw-detect-directx features \
         are off) — /v1/recommend and /v1/recommend-and-download will always assume CPU-only \
         and recommend the smallest model. Rebuild with `cargo build --features hw-detect-vulkan` \
         (or hw-detect-directx on Windows) for hardware-based recommendations."
    );
    #[cfg(any(feature = "hw-detect-vulkan", feature = "hw-detect-directx"))]
    tracing::info!(
        "GPU detection is enabled in this build (feature(s): {}{}).",
        if cfg!(feature = "hw-detect-vulkan") { "hw-detect-vulkan " } else { "" },
        if cfg!(feature = "hw-detect-directx") { "hw-detect-directx" } else { "" },
    );

    let registry = Arc::new(TenantRegistry::new());

    // 2026-08-15(CPU+GPU同時並列稼働): 従来はハンドラ登録時に1回だけ
    // `Arc::clone(&device)`していたため全リクエストが同一デバイスを
    // 共有していた。`device_pool`をキャプチャし、**リクエストごとに**
    // `next_device()`を呼ぶよう変更——複数の同時リクエストがCPU/GPU
    // 双方へラウンドロビンで分散し、両方が実際に並行稼働する。
    let chat_pool = Arc::clone(&device_pool);
    let chat_registry = Arc::clone(&registry);
    let classify_pool = Arc::clone(&device_pool);
    let classify_registry = Arc::clone(&registry);
    let classify_traffic_pool = Arc::clone(&device_pool);
    let classify_traffic_registry = Arc::clone(&registry);
    let generate_pool = Arc::clone(&device_pool);
    let generate_registry = Arc::clone(&registry);
    let generate_search_pool = Arc::clone(&device_pool);
    let generate_search_registry = Arc::clone(&registry);
    let translate_pool = Arc::clone(&device_pool);
    let translate_registry = Arc::clone(&registry);
    let admin_register_registry = Arc::clone(&registry);
    let admin_list_registry = Arc::clone(&registry);
    let admin_remove_registry = Arc::clone(&registry);

    let app = Route::new()
        .at("/v1/chat", post(handler_fn(move |req, _p| { let device = chat_pool.next_device(); let registry = Arc::clone(&chat_registry); async move { chat(req, device, registry).await } })))
        .at(
            "/v1/classify-security",
            post(handler_fn(move |req, _p| {
                let device = classify_pool.next_device();
                let registry = Arc::clone(&classify_registry);
                async move { classify_security(req, device, registry).await }
            })),
        )
        .at(
            "/v1/security/classify-traffic",
            post(handler_fn(move |req, _p| {
                let device = classify_traffic_pool.next_device();
                let registry = Arc::clone(&classify_traffic_registry);
                async move { classify_traffic(req, device, registry).await }
            })),
        )
        .at(
            "/v1/generate",
            post(handler_fn(move |req, _p| {
                let device = generate_pool.next_device();
                let registry = Arc::clone(&generate_registry);
                async move { generate(req, device, registry).await }
            })),
        )
        .at(
            "/v1/generate-with-search",
            post(handler_fn(move |req, _p| {
                let device = generate_search_pool.next_device();
                let registry = Arc::clone(&generate_search_registry);
                async move { generate_with_search(req, device, registry).await }
            })),
        )
        .at(
            "/v1/translate",
            post(handler_fn(move |req, _p| {
                let device = translate_pool.next_device();
                let registry = Arc::clone(&translate_registry);
                async move { translate(req, device, registry).await }
            })),
        )
        .at("/v1/models/catalog", get(plain(|| Box::pin(list_model_catalog()))))
        .at("/v1/geo/random", get(plain(|| Box::pin(geo_random()))))
        .at("/v1/geo/fuji", get(plain(|| Box::pin(geo_fuji()))))
        .at("/v1/geo/tours", post(handler_fn(|req, _p| Box::pin(geo_tours(req)))))
        .at("/v1/referrals/check", post(handler_fn(|req, _p| Box::pin(referrals_check(req)))))
        .at("/v1/geo/lookup", post(handler_fn(|req, _p| Box::pin(geo_lookup(req)))))
        .at(
            "/v1/settings/google-search",
            post(handler_fn(|req, _p| Box::pin(set_google_search_settings(req))))
                .get(plain(|| Box::pin(get_google_search_settings_status())))
                .delete(plain(|| Box::pin(clear_google_search_settings()))),
        )
        .at(
            "/v1/models/optimize-cache",
            post(handler_fn(|req, _p| Box::pin(optimize_model_cache_handler(req)))),
        )
        .at("/v1/models/install", post(handler_fn(|req, _p| Box::pin(install_model(req)))))
        .at("/v1/models/select", post(handler_fn(|req, _p| Box::pin(select_model(req)))))
        .at("/v1/recommend", get(plain(|| Box::pin(recommend_model()))))
        .at("/v1/recommend-and-download", post(plain(|| Box::pin(recommend_and_download()))))
        .at("/v1/download-larger", post(plain(|| Box::pin(download_larger_model()))))
        .at("/v1/download-smaller", post(plain(|| Box::pin(download_smaller_model()))))
        .at("/", get(plain(|| Box::pin(index_page()))))
        .at(
            "/admin/tenants",
            post(handler_fn(move |req, _p| { let registry = Arc::clone(&admin_register_registry); async move { admin_register_tenant(req, registry).await } }))
                .get(handler_fn(move |req, _p| { let registry = Arc::clone(&admin_list_registry); async move { admin_list_tenants(req, registry).await } })),
        )
        .at(
            "/admin/tenants/:host",
            delete(handler_fn(move |req, params| { let registry = Arc::clone(&admin_remove_registry); async move { admin_remove_tenant(req, params, registry).await } })),
        )
        .at("/healthz", get(plain(|| Box::pin(healthz()))))
        .with_cors();

    // `ARUARU_LLM_BIND`環境変数で上書き可能(2026-08-11追加、Android
    // 単体版向け——端末上で自己完結させるため`127.0.0.1`限定で起動し、
    // 外部ネットワークへは一切listenしないようにする)。
    let bind_addr: std::net::SocketAddr = std::env::var("ARUARU_LLM_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "0.0.0.0:4600".parse().expect("static bind address is always valid"));
    tracing::info!("aruaru-llm listening on {bind_addr} (shared multi-tenant instance)");
    let (_addr, handle) = Server::new(TcpListener::bind(bind_addr)).run(app).await?;
    handle.await.map_err(|e| anyhow::anyhow!("server task panicked: {e}"))
}
