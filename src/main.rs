//! aruaru-llm — aruaruエコシステム共通の「AIチャットコマース」応答サービス。
//!
//! **正直な開示(最重要、詳細はCLAUDE.md参照)**: v0.1.0時点では実際の
//! ニューラルLLM推論を一切行わない。open-cuda(`opencuda-core`/
//! `opencuda-cpu`)のCPUバックエンドを使い、bag-of-wordsベクトルの
//! 要素積カーネルを実行してドット積スコアを求める、単純なベクトル演算
//! ベースの意図分類。「open-cudaとSET」という位置づけは、open-cudaの
//! GPU/CPU実行パイプラインを実際に呼び出している(Cargo依存だけでなく
//! 実行時に本当に通る)という意味であり、Attention機構等を伴う本物の
//! Transformer推論ではない。
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
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    reply: String,
    engine: &'static str,
    matched_intent: Option<&'static str>,
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
        Ok(Some(intent)) => Json(ChatResponse {
            reply: intent.reply.to_string(),
            engine: "rule-based-v0-opencuda-cpu",
            matched_intent: Some(intent.name),
        }),
        Ok(None) => Json(ChatResponse {
            reply: scoring::FALLBACK_REPLY.to_string(),
            engine: "rule-based-v0-opencuda-cpu",
            matched_intent: None,
        }),
        Err(err) => {
            tracing::warn!("scoring failed: {err}");
            Json(ChatResponse {
                reply: scoring::FALLBACK_REPLY.to_string(),
                engine: "rule-based-v0-opencuda-cpu-error",
                matched_intent: None,
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
