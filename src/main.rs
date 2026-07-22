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

mod scoring;
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
    if let Some(tenant) = &req.tenant {
        if !registry.contains(tenant) {
            tracing::info!("chat request from unregistered tenant: {tenant}");
        }
    }

    match scoring::best_intent(device, &req.message) {
        Ok(Some(intent)) => {
            let (reply, reply_lang, lang_fallback) = intent.reply_for(&req.lang);
            Json(ChatResponse {
                reply: reply.to_string(),
                engine: "embedding-cosine-v0-opencuda-bert-cpu",
                matched_intent: Some(intent.name),
                reply_lang,
                lang_fallback,
            })
        }
        Ok(None) => {
            let (reply, reply_lang, lang_fallback) = scoring::fallback_reply_for(&req.lang);
            Json(ChatResponse {
                reply: reply.to_string(),
                engine: "embedding-cosine-v0-opencuda-bert-cpu",
                matched_intent: None,
                reply_lang,
                lang_fallback,
            })
        }
        Err(err) => {
            tracing::warn!("scoring failed: {err}");
            let (reply, reply_lang, lang_fallback) = scoring::fallback_reply_for(&req.lang);
            Json(ChatResponse {
                reply: reply.to_string(),
                engine: "embedding-cosine-v0-opencuda-bert-cpu-error",
                matched_intent: None,
                reply_lang,
                lang_fallback,
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
    }

    let registry = Arc::new(TenantRegistry::new());

    let app = Route::new()
        .at("/v1/chat", post(chat))
        .at("/admin/tenants", post(admin_register_tenant).get(admin_list_tenants))
        .at("/admin/tenants/:host", delete(admin_remove_tenant))
        .at("/healthz", get(healthz))
        .data(device)
        .data(registry);

    let bind_addr = "0.0.0.0:4600";
    tracing::info!("aruaru-llm listening on {bind_addr} (shared multi-tenant instance)");
    Server::new(TcpListener::bind(bind_addr)).run(app).await
}
