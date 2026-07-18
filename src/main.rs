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

mod scoring;

use std::sync::Arc;

use opencuda_core::GpuDevice;
use opencuda_cpu::CpuDevice;
use poem::listener::TcpListener;
use poem::web::{Data, Json};
use poem::{get, handler, post, EndpointExt, Route, Server};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    reply: String,
    engine: &'static str,
    matched_intent: Option<&'static str>,
}

#[handler]
fn chat(Json(req): Json<ChatRequest>, Data(device): Data<&Arc<dyn GpuDevice>>) -> Json<ChatResponse> {
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

#[handler]
fn healthz() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    tracing_subscriber::fmt::init();

    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    tracing::info!("aruaru-llm using open-cuda device: {}", device.info().name);

    let app = Route::new()
        .at("/v1/chat", post(chat))
        .at("/healthz", get(healthz))
        .data(device);

    let bind_addr = "127.0.0.1:4600";
    tracing::info!("aruaru-llm listening on {bind_addr}");
    Server::new(TcpListener::bind(bind_addr)).run(app).await
}
