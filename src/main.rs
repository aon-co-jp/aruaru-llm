//! aruaru-llm — aruaruエコシステム共通の「AIチャットコマース」応答サービス。
//!
//! **正直な開示(最重要、詳細はCLAUDE.md参照)**: 2026-07-21時点では
//! 自己回帰デコーダによる文章生成(いわゆる対話生成としての「LLM」の
//! 能力)は実装していない。`open-cuda`の`opencuda-bert`クレート
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

mod bow_fallback;
mod generation;
mod scoring;
mod security;
mod signatures;
mod tenants;

use std::sync::Arc;

use opencuda_core::GpuDevice;
use opencuda_cpu::CpuDevice;
use poem::listener::TcpListener;
use poem::web::{Data, Json, Path};
use poem::{delete, get, handler, http::StatusCode, post, EndpointExt, Request, Response, Route, Server};
use serde::{Deserialize, Serialize};
use tenants::{TenantInfo, TenantRegistry};

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
    engine: &'static str,
    matched_intent: Option<&'static str>,
    /// 実際に返した応答の言語(`"ja"`または`"en"`、現状の対応言語)。
    reply_lang: &'static str,
    /// `true`の場合、リクエストされた`lang`に対応する翻訳が無かったため
    /// 英語へフォールバックしたことを示す(黙って日本語へ落とさない、
    /// このエコシステムの「graceful degradation, never silent」方針、
    /// CLAUDE.md参照)。
    lang_fallback: bool,
}

#[handler]
fn chat(
    Json(req): Json<ChatRequest>,
    Data(device): Data<&Arc<dyn GpuDevice>>,
    Data(registry): Data<&Arc<TenantRegistry>>,
) -> Json<ChatResponse> {
    log_tenant_usage("chat", &req.tenant, registry);

    // scoring::classifyは、埋め込みモデル(models/multilingual-e5-small/)が
    // 使える場合はコサイン類似度分類を、モデル重みが無い・ロードに失敗した
    // 場合は自動的にbag-of-wordsドット積へフォールバックする(2026-07-25
    // 追加、詳細はscoring.rs/bow_fallback.rs参照)。engineには実際に使われた
    // 経路を常に正直に返す。
    let result = scoring::classify(device, &req.message);
    match result.intent {
        Some(intent) => {
            let (reply, reply_lang, lang_fallback) = intent.reply_for(&req.lang);
            Json(ChatResponse {
                reply: reply.to_string(),
                engine: result.engine,
                matched_intent: Some(intent.name),
                reply_lang,
                lang_fallback,
            })
        }
        None => {
            let (reply, reply_lang, lang_fallback) = scoring::fallback_reply_for(&req.lang);
            Json(ChatResponse {
                reply: reply.to_string(),
                engine: result.engine,
                matched_intent: None,
                reply_lang,
                lang_fallback,
            })
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
    engine: &'static str,
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
#[handler]
fn classify_security(Json(req): Json<ClassifySecurityRequest>, Data(device): Data<&Arc<dyn GpuDevice>>, Data(registry): Data<&Arc<TenantRegistry>>) -> Json<ClassifySecurityResponse> {
    log_tenant_usage("classify-security", &req.tenant, registry);
    match security::classify_security(device, &req.text) {
        Ok(v) => {
            let had_static_signals = !v.static_signals.is_empty();
            Json(ClassifySecurityResponse {
                label: v.label,
                description: v.description,
                score: v.score,
                is_suspicious: v.is_suspicious,
                engine: if had_static_signals {
                    "embedding-cosine-heuristic-v0-opencuda-bert-cpu+static-signatures-v1-opencuda-cpu"
                } else {
                    "embedding-cosine-heuristic-v0-opencuda-bert-cpu"
                },
                static_signals: v
                    .static_signals
                    .into_iter()
                    .map(|s| StaticSignalDto { tag: s.tag, description: s.description, category_hint: s.category_hint })
                    .collect(),
            })
        }
        Err(err) => {
            tracing::warn!("classify_security failed: {err}");
            // 判定不能なときは黙って「安全」とは言わない——is_suspicious=false
            // だが、engineでエラーだったことを正直に示し、呼び出し側が
            // 静的結果のみで判断できるようにする。
            Json(ClassifySecurityResponse {
                label: "unknown",
                description: "classification failed; rely on static findings only",
                score: 0.0,
                is_suspicious: false,
                engine: "embedding-cosine-heuristic-v0-opencuda-bert-cpu-error",
                static_signals: Vec::new(),
            })
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

#[handler]
fn admin_register_tenant(req: &Request, Json(info): Json<TenantInfo>, Data(registry): Data<&Arc<TenantRegistry>>) -> Response {
    if !check_admin_token(req) {
        return Response::builder().status(StatusCode::UNAUTHORIZED).body("invalid admin token");
    }
    tracing::info!("registering tenant: {}", info.host);
    registry.register(info);
    Response::builder().status(StatusCode::OK).body("ok")
}

#[handler]
fn admin_list_tenants(req: &Request, Data(registry): Data<&Arc<TenantRegistry>>) -> Response {
    if !check_admin_token(req) {
        return Response::builder().status(StatusCode::UNAUTHORIZED).body("invalid admin token");
    }
    let body = serde_json::to_string(&registry.list()).unwrap_or_else(|_| "[]".to_string());
    Response::builder().status(StatusCode::OK).content_type("application/json").body(body)
}

#[handler]
fn admin_remove_tenant(req: &Request, Path(host): Path<String>, Data(registry): Data<&Arc<TenantRegistry>>) -> Response {
    if !check_admin_token(req) {
        return Response::builder().status(StatusCode::UNAUTHORIZED).body("invalid admin token");
    }
    if registry.remove(&host) {
        Response::builder().status(StatusCode::OK).body("ok")
    } else {
        Response::builder().status(StatusCode::NOT_FOUND).body("tenant not found")
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
    engine: &'static str,
    /// 正直な開示メッセージ(常に返す、GPT-2 124Mの性能限界を毎回明示)。
    disclosure: &'static str,
}

#[derive(Debug, Serialize)]
struct GenerateErrorResponse {
    error: String,
    engine: &'static str,
}

/// `opencuda-llm::GptModel`(GPT-2 124M実重み)による自己回帰テキスト生成。
/// `/v1/chat`(意図分類、軽量・高速)とは別目的の別エンドポイント——
/// 意図分類と生成は無理に統合しない設計方針(CLAUDE.md参照)。
#[handler]
fn generate(Json(req): Json<GenerateRequest>, Data(device): Data<&Arc<dyn GpuDevice>>, Data(registry): Data<&Arc<TenantRegistry>>) -> Response {
    log_tenant_usage("generate", &req.tenant, registry);
    let max_new_tokens = req.max_new_tokens.clamp(1, MAX_NEW_TOKENS_LIMIT);
    match generation::generate(device, &req.prompt, max_new_tokens) {
        Ok(completion) => {
            let body = serde_json::to_string(&GenerateResponse {
                completion,
                engine: generation::ENGINE_GPT2_GREEDY,
                disclosure: "GPT-2 124M is a small 2019-era model, not comparable to modern commercial LLMs (e.g. GPT-4). \
                    This demonstrates self-contained text generation without an external LLM API contract, not state-of-the-art quality. \
                    Output may be grammatically fluent but is not guaranteed to be factually accurate.",
            })
            .unwrap_or_else(|_| "{}".to_string());
            Response::builder().status(StatusCode::OK).content_type("application/json").body(body)
        }
        Err(err) => {
            tracing::warn!("generate failed: {err:#}");
            let body = serde_json::to_string(&GenerateErrorResponse { error: format!("{err:#}"), engine: generation::ENGINE_GPT2_GREEDY })
                .unwrap_or_else(|_| "{}".to_string());
            Response::builder().status(StatusCode::SERVICE_UNAVAILABLE).content_type("application/json").body(body)
        }
    }
}

#[handler]
fn healthz() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    tracing_subscriber::fmt::init();

    // マルチコア/マルチスレッド前提: #[tokio::main]の既定フレーバーは
    // multi_thread(current_threadへの明示的固定はしていない)。CPU計算
    // (bag-of-wordsスコアリング)自体はopencuda-cpuのrayonが
    // 利用可能な全論理コアへ並列ディスパッチする(`CpuDevice::new`が
    // `std::thread::available_parallelism()`から検出)。
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    tracing::info!("aruaru-llm using open-cuda device: {}", device.info().name);

    // コールドスタート対策(2026-07-22追記、CLAUDE.md HANDOFF参照):
    // opencuda-bertのモデルロード+インテントembedding計算(数秒)を、
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
        // GPT-2 124M実重み(548MB)のロードも起動時に前倒し(2026-07-25追加)。
        // 失敗しても致命的ではない(/v1/generateへの初回リクエスト時に再試行)。
        match generation::warmup() {
            Ok(()) => tracing::info!("generation (GPT-2 124M) warmup complete"),
            Err(err) => tracing::warn!("generation warmup failed (will retry lazily on first /v1/generate request): {err}"),
        }
    }

    let registry = Arc::new(TenantRegistry::new());

    let app = Route::new()
        .at("/v1/chat", post(chat))
        .at("/v1/classify-security", post(classify_security))
        .at("/v1/generate", post(generate))
        .at("/admin/tenants", post(admin_register_tenant).get(admin_list_tenants))
        .at("/admin/tenants/:host", delete(admin_remove_tenant))
        .at("/healthz", get(healthz))
        .data(device)
        .data(registry);

    let bind_addr = "0.0.0.0:4600";
    tracing::info!("aruaru-llm listening on {bind_addr} (shared multi-tenant instance)");
    Server::new(TcpListener::bind(bind_addr)).run(app).await
}
