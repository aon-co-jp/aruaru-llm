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
mod chat_providers;
mod device_pool;
mod geo_content;
mod github_search;
mod referrals;
mod web_search;
mod youtube_search;
mod generation;
mod hardware;
mod idle_background_fold;
mod intrusion_detection;
mod model_catalog;
mod news_geo;
mod nllb;
mod phone_task;
mod provider_priority;
mod scoring;
mod security;
mod self_update;
mod signatures;
mod tenants;
mod transcribe;

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine as _;
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
    // アイドル時バックグラウンドModel Folding準備スケジューラ(idle_background_fold.rs、
    // 2026-08-19新設)へ「今アクティブなリクエストがあった」ことを伝える。
    idle_background_fold::touch_activity();
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
    idle_background_fold::touch_activity();
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
    // 2026-08-15(CPU+GPU非同期並列化、ユーザー指示): `generation::generate`は
    // 同期的な重い計算(CPU rayon並列、またはGPU Vulkanディスパッチ+
    // フェンス待機)を行う。これをasyncハンドラ内で直接呼ぶとtokioの
    // ワーカースレッドを長時間ブロックし、他の非同期タスク(他リクエストの
    // 待受・ヘルスチェック等)の処理を妨げる——`spawn_blocking`(自動拡張する
    // 専用ブロッキングスレッドプール)へ逃がすことで、CPU担当リクエストと
    // GPU担当リクエストが実際に別スレッドで並行実行され、tokioのマルチコア
    // ワーカーを塞がない設計にする。開始/終了ログにデバイス名・
    // OSスレッドIDを含めることで、複数リクエストを同時に送った際に
    // 実際にどのスレッドでどのデバイスが並行稼働したかを事後計測できる
    // ようにする。
    let device_name = device.info().name.clone();
    let request_started = std::time::Instant::now();
    tracing::info!(
        "generate: dispatch start device={device_name} thread={:?}",
        std::thread::current().id()
    );
    let prompt = req.prompt.clone();
    let device_for_task = Arc::clone(&device);
    let generate_result = tokio::task::spawn_blocking(move || {
        let thread_id = std::thread::current().id();
        let result = generation::generate(&device_for_task, &prompt, max_new_tokens);
        (result, thread_id)
    })
    .await;
    let (generate_result, exec_thread_id) = match generate_result {
        Ok((result, thread_id)) => (result, Some(thread_id)),
        Err(join_err) => (
            Err(anyhow::anyhow!("generate task panicked: {join_err}")),
            None,
        ),
    };
    tracing::info!(
        "generate: dispatch end device={device_name} exec_thread={exec_thread_id:?} \
         elapsed_ms={} (caller_thread={:?})",
        request_started.elapsed().as_millis(),
        std::thread::current().id()
    );
    match generate_result {
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

#[derive(Debug, Deserialize)]
struct GenerateSpeculativeRequest {
    prompt: String,
    /// `model_catalog::CATALOG`のいずれか(例: `"distilgpt2"`)。ダウンロード
    /// 済みでない場合は`400`(`POST /v1/models/install`が必要)。
    draft_id: String,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    /// ドラフトモデルが1ラウンドで提案するトークン数(既定4)。
    #[serde(default = "default_draft_k")]
    draft_k: usize,
    #[serde(default)]
    tenant: Option<String>,
}

fn default_draft_k() -> usize {
    4
}

#[derive(Debug, Serialize)]
struct GenerateSpeculativeResponse {
    completion: String,
    engine: String,
    draft_id: String,
    /// 検証対象になったドラフト提案トークンの総数。
    proposed: usize,
    /// そのうち実際に採用された数。
    accepted: usize,
    acceptance_rate: f32,
    disclosure: &'static str,
}

/// `POST /v1/generate-speculative` — DeepSeekの「DSpark」(ロスレス投機的
/// デコード)方式を`open-cuda-llm::GptModel::generate_speculative`経由で
/// 呼ぶ、2026-08-17新設のオプトインエンドポイント(ユーザー承認、週次
/// リサーチルーティンでのDSpark/llama.cpp Multi-Token Prediction調査への
/// YES回答を受けて実装)。
///
/// **正直な開示(最重要)**: `open-cuda-llm`側で実機計測したところ
/// (CPU実行、ターゲット`gpt2`+ドラフト`distilgpt2`、`draft_k=4`)、
/// 採用率80%と高かったにもかかわらず**素の`/v1/generate`より実際には
/// 遅かった**(plain=4.63秒 vs speculative=7.65秒、`open-cuda-llm`側
/// テスト`real_gpt2_speculative_decoding_matches_plain_greedy_and_reports_
/// acceptance`の実測)。CPU素朴GEMM実装ではディスパッチ固定オーバー
/// ヘッドという「削減すべきコスト」自体がほぼ存在しないため、ドラフト
/// モデルの追加計算コストが純増分になってしまう——`real-vulkan`環境
/// (Vulkanディスパッチオーバーヘッドが支配的、本来の狙い)での速度検証は
/// 未実施のまま。この理由により`/v1/generate`の内部実装は置き換えず、
/// 明示的にオプトインする本エンドポイントとして提供する。出力の正しさ
/// (`/v1/generate`とビット完全一致するロスレス性)は実重み・実合成
/// フィクスチャ双方で検証済み。
async fn generate_speculative(req: Request, device: Arc<dyn GpuDevice>, registry: Arc<TenantRegistry>) -> Response {
    let Json(req): Json<GenerateSpeculativeRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("generate-speculative", &req.tenant, &registry);
    if req.prompt.trim().is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &GenerateErrorResponse { error: "prompt must not be empty".to_string(), engine: generation::engine_label(&device) },
        );
    }
    if req.draft_k == 0 {
        return json_response(
            StatusCode::BAD_REQUEST,
            &GenerateErrorResponse { error: "draft_k must be >= 1".to_string(), engine: generation::engine_label(&device) },
        );
    }
    let max_new_tokens = req.max_new_tokens.clamp(1, MAX_NEW_TOKENS_LIMIT);
    let draft_id = req.draft_id.clone();
    let prompt = req.prompt.clone();
    let draft_k = req.draft_k;
    let device_for_task = Arc::clone(&device);
    let result = tokio::task::spawn_blocking(move || generation::generate_speculative(&device_for_task, &draft_id, &prompt, max_new_tokens, draft_k)).await;
    let result = match result {
        Ok(r) => r,
        Err(join_err) => Err(anyhow::anyhow!("generate-speculative task panicked: {join_err}")),
    };
    match result {
        Ok((completion, stats)) => json_response(
            StatusCode::OK,
            &GenerateSpeculativeResponse {
                completion,
                engine: generation::engine_label(&device),
                draft_id: req.draft_id,
                proposed: stats.proposed,
                accepted: stats.accepted,
                acceptance_rate: stats.acceptance_rate(),
                disclosure: "Lossless speculative decoding (DeepSeek DSpark / Leviathan et al. style): output is byte-identical to plain \
                    greedy /v1/generate for the same prompt. Honest disclosure: measured SLOWER than plain /v1/generate on CPU-only \
                    hardware despite high acceptance rates, because CPU naive GEMM has little dispatch overhead to amortize. The intended \
                    benefit (reducing target-model dispatch count) is unverified on --features real-vulkan hardware so far.",
            },
        ),
        Err(err) => {
            tracing::warn!("generate-speculative failed: {err:#}");
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
    /// ブラウザ側の利用者が自分自身で用意したGoogle Custom Search
    /// APIキー/検索エンジンID(cx)(2026-08-25新設、ユーザー指示
    /// 「ブラウザ版は各自Google検索のAPIキーとIDを各自で設定してもらう
    /// 様に…開発者が設定したAPIキーとIDは、アクセス者は使わない、
    /// 消費しない様に」への対応)。**両方とも指定された場合のみ**、
    /// この値を使ってこのリクエスト限りの検索を行い、プロセス全体で
    /// 共有されるグローバル設定(`/v1/settings/google-search`・環境変数)
    /// には一切触れない・使わない——開発者が別デプロイで設定した
    /// キーを、この経路からのリクエストが消費することは無い。
    #[serde(default)]
    google_search_api_key: Option<String>,
    #[serde(default)]
    google_search_cx: Option<String>,
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

    // 利用者自身が持ち込んだキー(ブラウザ側の設定パネル入力→リクエスト
    // ボディ経由)があれば最優先で使い、グローバル共有設定
    // (`web_search::is_configured`/`web_search::search`)には一切触れない。
    // これにより、同じ`aruaru-llm`インスタンスを複数の訪問者が共有する
    // デプロイ(VPS上の共有デプロイ等)でも、ある訪問者の検索が
    // 開発者や他の訪問者のAPIキー・クォータを消費することは無い。
    let own_credentials = match (&req.google_search_api_key, &req.google_search_cx) {
        (Some(k), Some(c)) if !k.trim().is_empty() && !c.trim().is_empty() => Some((k.clone(), c.clone())),
        _ => None,
    };
    let (augmented_prompt, used_search, search_results, search_error) = if let Some((api_key, cx)) = own_credentials {
        match web_search::search_with_credentials(&req.prompt, 3, &api_key, &cx).await {
            Ok(results) if !results.is_empty() => {
                let context = web_search::format_results_as_context(&results);
                (web_search::build_search_augmented_prompt(&context, &req.prompt), true, results, None)
            }
            Ok(_) => (req.prompt.clone(), false, Vec::new(), Some("Google returned zero results for this query.".to_string())),
            Err(err) => {
                let msg = format!("{err:#}");
                tracing::warn!("google custom search (visitor-supplied key) failed, falling back to no-search generation: {msg}");
                (req.prompt.clone(), false, Vec::new(), Some(msg))
            }
        }
    } else if web_search::is_configured() {
        match web_search::search(&req.prompt, 3).await {
            Ok(results) if !results.is_empty() => {
                let context = web_search::format_results_as_context(&results);
                (web_search::build_search_augmented_prompt(&context, &req.prompt), true, results, None)
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

/// `POST /v1/settings/github-search` — ブラウザの設定パネルからGitHub
/// Personal Access Token(任意)を保存する(ユーザー指示「GitHub連携も、
/// チェックを付けられる機能と実際に連携する機能を付けて」への対応)。
/// **正直な開示**: トークンは検索の利用自体には必須ではない
/// (`github_search`モジュールdoc参照、未認証だとレート制限が10/分に
/// なるだけ)。
#[derive(Debug, Deserialize)]
struct GithubSearchSettingsRequest {
    #[serde(default)]
    token: String,
}

#[derive(Debug, Serialize)]
struct GithubSearchSettingsStatusResponse {
    token_configured: bool,
}

async fn set_github_search_settings(req: Request) -> Response {
    let Json(body): Json<GithubSearchSettingsRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    github_search::set_runtime_token(body.token);
    json_response(StatusCode::OK, &GithubSearchSettingsStatusResponse { token_configured: github_search::is_token_configured() })
}

async fn clear_github_search_settings() -> Response {
    github_search::clear_runtime_token();
    json_response(StatusCode::OK, &GithubSearchSettingsStatusResponse { token_configured: github_search::is_token_configured() })
}

async fn get_github_search_settings_status() -> Response {
    json_response(StatusCode::OK, &GithubSearchSettingsStatusResponse { token_configured: github_search::is_token_configured() })
}

/// `POST /v1/settings/youtube-search` — ブラウザの設定パネルからYouTube
/// Data API v3のAPIキーを保存する(ユーザー指示「Youtube連携機能も
/// 付けて」への対応、`set_google_search_settings`と同じ設計)。
#[derive(Debug, Deserialize)]
struct YoutubeSearchSettingsRequest {
    api_key: String,
}

#[derive(Debug, Serialize)]
struct YoutubeSearchSettingsStatusResponse {
    configured: bool,
}

async fn set_youtube_search_settings(req: Request) -> Response {
    let Json(body): Json<YoutubeSearchSettingsRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    youtube_search::set_runtime_key(body.api_key);
    json_response(StatusCode::OK, &YoutubeSearchSettingsStatusResponse { configured: youtube_search::is_configured() })
}

async fn clear_youtube_search_settings() -> Response {
    youtube_search::clear_runtime_key();
    json_response(StatusCode::OK, &YoutubeSearchSettingsStatusResponse { configured: youtube_search::is_configured() })
}

async fn get_youtube_search_settings_status() -> Response {
    json_response(StatusCode::OK, &YoutubeSearchSettingsStatusResponse { configured: youtube_search::is_configured() })
}

/// `POST /v1/settings/chat-providers` — ブラウザの設定パネルからChatGPT/
/// DeepSeek/Gemini/ClaudeのAPIキーを保存する(`set_google_search_settings`
/// と同じ設計: メモリ上にのみ保持、ディスク書き込み・ログ出力は一切
/// 行わない)。
#[derive(Debug, Deserialize)]
struct ChatProviderSettingsRequest {
    provider: chat_providers::Provider,
    api_key: String,
}

#[derive(Debug, Serialize)]
struct ChatProviderSettingsStatusResponse {
    /// APIキーの値自体は絶対に返さない(`GoogleSearchSettingsStatusResponse`
    /// と同じ設計)。設定済みのプロバイダ一覧のみを返す。
    configured_providers: Vec<chat_providers::Provider>,
}

async fn set_chat_provider_settings(req: Request) -> Response {
    let Json(body): Json<ChatProviderSettingsRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    chat_providers::set_runtime_key(body.provider, body.api_key);
    json_response(StatusCode::OK, &ChatProviderSettingsStatusResponse { configured_providers: chat_providers::configured_providers() })
}

async fn clear_chat_provider_settings() -> Response {
    chat_providers::clear_runtime_keys();
    json_response(StatusCode::OK, &ChatProviderSettingsStatusResponse { configured_providers: chat_providers::configured_providers() })
}

async fn get_chat_provider_settings_status() -> Response {
    json_response(StatusCode::OK, &ChatProviderSettingsStatusResponse { configured_providers: chat_providers::configured_providers() })
}

/// `POST /v1/chat-providers/complete` — 単体のプロバイダ(ChatGPT/
/// DeepSeek/Gemini/Claudeのいずれか1つ)を呼び出す。`api_key`が
/// リクエストボディに含まれていればこのリクエスト限りで使用し
/// (共有VPS上での意図しないキー消費を避ける、`generate_with_search`と
/// 同じ設計)、無ければ実行時/環境変数設定へフォールバックする。
#[derive(Debug, Deserialize)]
struct ChatProviderCompleteRequest {
    provider: chat_providers::Provider,
    prompt: String,
    #[serde(default)]
    api_key: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatProviderCompleteResponse {
    provider: chat_providers::Provider,
    text: String,
}

#[derive(Debug, Serialize)]
struct ChatProviderErrorResponse {
    provider: chat_providers::Provider,
    error: String,
}

/// 外部チャットプロバイダAPIへ渡すプロンプトの上限文字数(2026-08-26
/// セキュリティ見直しで新設)。**正直な開示**: これは悪意ある攻撃者を
/// 完全に防ぐものではなく、誤って巨大なテキスト(例: ファイルの中身を
/// まるごと貼り付けてしまった等)をそのまま外部の有料APIへ送ってしまい
/// 意図せず高額請求・無料枠の急速な枯渇を招く事故を防ぐための実用的な
/// 上限。各社の実際のトークン上限(モデルにより数千〜数十万トークン)
/// とは無関係な、このリポジトリ独自の保守的な安全弁。
const CHAT_PROVIDER_PROMPT_CHAR_LIMIT: usize = 20_000;

fn chat_provider_prompt_too_long(prompt: &str) -> bool {
    prompt.chars().count() > CHAT_PROVIDER_PROMPT_CHAR_LIMIT
}

async fn chat_provider_complete(req: Request) -> Response {
    let Json(req): Json<ChatProviderCompleteRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if chat_provider_prompt_too_long(&req.prompt) {
        return json_response(
            StatusCode::BAD_REQUEST,
            &serde_json::json!({"error": format!("prompt exceeds {CHAT_PROVIDER_PROMPT_CHAR_LIMIT} characters")}),
        );
    }
    let result = match req.api_key.as_deref() {
        Some(key) if !key.trim().is_empty() => chat_providers::complete_with_key(req.provider, key, &req.prompt).await,
        _ => chat_providers::complete(req.provider, &req.prompt).await,
    };
    match result {
        Ok(text) => json_response(StatusCode::OK, &ChatProviderCompleteResponse { provider: req.provider, text }),
        Err(err) => {
            tracing::warn!("chat_provider_complete failed: {err:#}");
            json_response(StatusCode::SERVICE_UNAVAILABLE, &ChatProviderErrorResponse { provider: req.provider, error: format!("{err:#}") })
        }
    }
}

/// `POST /v1/chat-providers/complete-multi` — 複数プロバイダを同時実行
/// (並列にHTTPリクエストを投げ、成功分をプロバイダ別に、失敗分も
/// エラー内容付きで正直に返す。1つの応答へ統合・要約する処理は
/// 行わない——呼び出し元・利用者が結果を比較できるようにするため)。
#[derive(Debug, Deserialize)]
struct ChatProviderCompleteMultiRequest {
    providers: Vec<chat_providers::Provider>,
    prompt: String,
    /// プロバイダごとの持ち込みAPIキー(任意、`provider`名をキーとする
    /// JSONオブジェクト)。指定が無いプロバイダは実行時/環境変数設定へ
    /// フォールバックする。
    #[serde(default)]
    api_keys: HashMap<chat_providers::Provider, String>,
}

#[derive(Debug, Serialize)]
struct ChatProviderCompleteMultiResponse {
    replies: Vec<chat_providers::ProviderReply>,
    failures: Vec<chat_providers::ProviderFailure>,
}

async fn chat_provider_complete_multi(req: Request) -> Response {
    let Json(req): Json<ChatProviderCompleteMultiRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if req.providers.is_empty() {
        return json_response(StatusCode::BAD_REQUEST, &serde_json::json!({"error": "providers must not be empty"}));
    }
    if req.prompt.trim().is_empty() {
        return json_response(StatusCode::BAD_REQUEST, &serde_json::json!({"error": "prompt must not be empty"}));
    }
    if chat_provider_prompt_too_long(&req.prompt) {
        return json_response(
            StatusCode::BAD_REQUEST,
            &serde_json::json!({"error": format!("prompt exceeds {CHAT_PROVIDER_PROMPT_CHAR_LIMIT} characters")}),
        );
    }
    let (replies, failures) = chat_providers::complete_multi(&req.providers, &req.api_keys, &req.prompt).await;
    json_response(StatusCode::OK, &ChatProviderCompleteMultiResponse { replies, failures })
}

/// `POST /v1/settings/provider-priority` — Google検索+ChatGPT/DeepSeek/
/// Gemini/Claudeの5サービスを横断した「無料枠を優先で使い切り、順番に
/// 使用」チェックボックスの有効/無効+優先順序を設定する(ユーザー指示
/// 「Google、ChatGPT/DeepSeek/Gemini/Claudeは、無料枠を優先で使い切り
/// 順番に使用、にチェックを付けられる様にして。Googleなどは、順番を
/// 入力したり、数字のラジオボタンを押すかのどちらかで優先の順番を
/// 変更可能にして」への対応)。`order`はフロントエンド側が「番号入力」
/// または「ラジオボタン」いずれのUIで組み立てても良い、サーバー側は
/// 並び替え済みの配列を受け取るだけ(UI実装の自由度を残す設計)。
#[derive(Debug, Deserialize)]
struct ProviderPrioritySettingsRequest {
    enabled: bool,
    #[serde(default)]
    order: Vec<provider_priority::PriorityService>,
}

async fn set_provider_priority_settings(req: Request) -> Response {
    let Json(body): Json<ProviderPrioritySettingsRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    provider_priority::set_priority(body.enabled, body.order);
    json_response(StatusCode::OK, &provider_priority::status())
}

async fn reset_provider_priority_settings() -> Response {
    provider_priority::reset_priority();
    json_response(StatusCode::OK, &provider_priority::status())
}

async fn get_provider_priority_settings_status() -> Response {
    json_response(StatusCode::OK, &provider_priority::status())
}

/// `POST /v1/chat-providers/complete-priority` — 上記の優先順位設定に
/// 従い、設定済みのプロバイダを順番に試し最初に成功したものを返す
/// (`chat_providers::complete_in_priority_order`)。優先順位機能が
/// 無効化されている場合でも呼び出しは可能(その場合は既定順序=
/// Google→OpenAI→DeepSeek→Gemini→Claudeで、設定済みの最初の1件のみを
/// 試す——`enabled`はUI側のチェックボックス状態を表すだけで、この
/// エンドポイント自体は常に「順番に試す」動作をする、無効時に何も
/// しないと利用者が混乱するのを避けるため)。
/// Google検索・GitHub検索・YouTube検索を任意で有効化するチェックボックス
/// (ユーザー指示「Github連携も、チェックを付けられる機能と実際に連携
/// する機能を付けて」「Youtube連携機能も付けて」への対応)。有効化された
/// 検索の結果は、プロンプト本文の前にコンテキストとして埋め込まれる
/// (`generate_with_search`と同じブリッジ式)。**正直な開示**: 各検索
/// APIが未設定/失敗した場合はその旨を`search_notes`で正直に開示し、
/// 検索無しでチャット補完自体は続行する(サービス全体を壊さない設計)。
#[derive(Debug, Deserialize)]
struct ChatProviderCompletePriorityRequest {
    prompt: String,
    #[serde(default)]
    use_google_search: bool,
    #[serde(default)]
    use_github_search: bool,
    #[serde(default)]
    use_youtube_search: bool,
    /// 利用者自身が持ち込んだ検索APIキー(任意、`generate_with_search`と
    /// 同じ「共有VPS上でグローバル設定を消費しない」設計)。
    #[serde(default)]
    google_search_api_key: Option<String>,
    #[serde(default)]
    google_search_cx: Option<String>,
    #[serde(default)]
    github_token: Option<String>,
    #[serde(default)]
    youtube_api_key: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatProviderCompletePriorityResponse {
    reply: Option<chat_providers::ProviderReply>,
    attempted: Vec<chat_providers::PriorityAttempt>,
    all_quota_exceeded: bool,
    /// 実際に使われた検索コンテキスト(空なら検索無しでの応答)。
    search_notes: Vec<String>,
}

async fn chat_provider_complete_priority(req: Request) -> Response {
    let Json(req): Json<ChatProviderCompletePriorityRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if req.prompt.trim().is_empty() {
        return json_response(StatusCode::BAD_REQUEST, &serde_json::json!({"error": "prompt must not be empty"}));
    }
    if chat_provider_prompt_too_long(&req.prompt) {
        return json_response(
            StatusCode::BAD_REQUEST,
            &serde_json::json!({"error": format!("prompt exceeds {CHAT_PROVIDER_PROMPT_CHAR_LIMIT} characters")}),
        );
    }

    let mut context_blocks: Vec<String> = Vec::new();
    let mut search_notes: Vec<String> = Vec::new();

    if req.use_google_search {
        let own_credentials = match (&req.google_search_api_key, &req.google_search_cx) {
            (Some(k), Some(c)) if !k.trim().is_empty() && !c.trim().is_empty() => Some((k.clone(), c.clone())),
            _ => None,
        };
        let result = match own_credentials {
            Some((api_key, cx)) => web_search::search_with_credentials(&req.prompt, 3, &api_key, &cx).await,
            None if web_search::is_configured() => web_search::search(&req.prompt, 3).await,
            None => Err(anyhow::anyhow!("Google Custom Search is not configured (no API key/cx set)")),
        };
        match result {
            Ok(results) if !results.is_empty() => {
                context_blocks.push(format!("Google search results:\n{}", web_search::format_results_as_context(&results)));
                search_notes.push(format!("google: {} result(s)", results.len()));
            }
            Ok(_) => search_notes.push("google: 0 results".to_string()),
            Err(err) => search_notes.push(format!("google: failed ({err:#})")),
        }
    }

    if req.use_github_search {
        let token = req.github_token.as_deref().filter(|t| !t.trim().is_empty());
        match github_search::search_with_optional_token(&req.prompt, 3, token).await {
            Ok(results) if !results.is_empty() => {
                context_blocks.push(format!("GitHub repository search results:\n{}", github_search::format_results_as_context(&results)));
                search_notes.push(format!("github: {} result(s)", results.len()));
            }
            Ok(_) => search_notes.push("github: 0 results".to_string()),
            Err(err) => search_notes.push(format!("github: failed ({err:#})")),
        }
    }

    if req.use_youtube_search {
        let key = req.youtube_api_key.as_deref().filter(|k| !k.trim().is_empty());
        let result = match key {
            Some(k) => youtube_search::search_with_key(&req.prompt, 3, k).await,
            None => youtube_search::search(&req.prompt, 3).await,
        };
        match result {
            Ok(results) if !results.is_empty() => {
                context_blocks.push(format!("YouTube video search results:\n{}", youtube_search::format_results_as_context(&results)));
                search_notes.push(format!("youtube: {} result(s)", results.len()));
            }
            Ok(_) => search_notes.push("youtube: 0 results".to_string()),
            Err(err) => search_notes.push(format!("youtube: failed ({err:#})")),
        }
    }

    let augmented_prompt = if context_blocks.is_empty() { req.prompt.clone() } else { format!("{}\n\n{}", context_blocks.join("\n\n"), req.prompt) };
    if chat_provider_prompt_too_long(&augmented_prompt) {
        return json_response(
            StatusCode::BAD_REQUEST,
            &serde_json::json!({"error": format!("prompt (with search context) exceeds {CHAT_PROVIDER_PROMPT_CHAR_LIMIT} characters")}),
        );
    }
    let result = chat_providers::complete_in_priority_order(&augmented_prompt).await;
    json_response(
        StatusCode::OK,
        &ChatProviderCompletePriorityResponse { reply: result.reply, attempted: result.attempted, all_quota_exceeded: result.all_quota_exceeded, search_notes },
    )
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

#[derive(Debug, Deserialize)]
struct TranscribeRequest {
    /// 16kHz mono の f32 PCM サンプル(範囲 -1.0..=1.0)をリトルエンディアン
    /// バイト列にして base64 エンコードしたもの。`open-english` の
    /// `blobToPcm16k()` が `OfflineAudioContext` で 16kHz mono へリサンプル
    /// 済みの `Float32Array` をそのまま送る想定。
    pcm_f32_base64: String,
    /// サンプルレート(Hz)。Whisper は 16000 固定のため、それ以外は `400`。
    #[serde(default = "default_transcribe_sample_rate")]
    sample_rate: u32,
    /// 言語コード(例 `"en"` / `"ja"`)。省略/`"auto"` なら Whisper に検出させる。
    #[serde(default)]
    language: Option<String>,
    /// contextual biasing 用のプロンプト(直前のトレーナー発話・練習問題の
    /// 期待語彙など)。whisper-cli の `--prompt` へ渡す。Whisper のデコーダ
    /// プロンプトは末尾 ~224 トークンしか効かないので、サーバー側で先頭を
    /// 切り詰める。
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    tenant: Option<String>,
}

fn default_transcribe_sample_rate() -> u32 {
    16000
}

#[derive(Debug, Serialize)]
struct TranscribeResponse {
    transcript: String,
    /// Whisper が検出/使用した言語コード。
    language: String,
    engine: &'static str,
    disclosure: &'static str,
}

#[derive(Debug, Serialize)]
struct TranscribeErrorResponse {
    error: String,
    engine: &'static str,
}

fn transcribe_engine_label() -> &'static str {
    if transcribe::is_available() {
        "whisper.cpp-cli-v0"
    } else {
        "whisper-not-available"
    }
}

/// `POST /v1/transcribe` — whisper.cpp のプレビルド CLI(`whisper-cli`)を
/// 子プロセス起動して音声を書き起こす(2026-08-29新設・方針変更、
/// `open-english/docs/SPEECH_RECOGNITION_REDESIGN.md` の P2-β。ブラウザ内
/// Whisper〈P2-α〉では端末性能が足りない利用者向けに、自分の PC で起動
/// している aruaru-llm 側で書き起こす経路)。
///
/// **正直な開示**: `whisper-cli` 実行ファイルと GGML モデル(いずれも
/// リポジトリ非同梱)が実在する場合のみ動作する。無ければ `503` +
/// 入手先を案内するエラーを返す。当初は `whisper-rs` を直接リンクする
/// 設計だったが、Windows/MSVC でビルド不能な上流ブロッカーがあるため
/// CLI サブプロセス方式へ変更した(`src/transcribe.rs` モジュール doc 参照)。
async fn transcribe(req: Request, registry: Arc<TenantRegistry>) -> Response {
    idle_background_fold::touch_activity();
    let Json(req): Json<TranscribeRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    log_tenant_usage("transcribe", &req.tenant, &registry);

    if !transcribe::is_available() {
        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &TranscribeErrorResponse {
                error: format!(
                    "whisper transcription is not available: {}. Download a prebuilt whisper-cli from \
                     https://github.com/ggml-org/whisper.cpp/releases and a GGML model (e.g. ggml-base.bin), \
                     place them at {} and {} (or set ARUARU_LLM_WHISPER_CLI / ARUARU_LLM_WHISPER_MODEL).",
                    if !transcribe::cli_present() { "whisper-cli not found" } else { "GGML model not found" },
                    transcribe::cli_path().display(),
                    transcribe::model_path().display(),
                ),
                engine: transcribe_engine_label(),
            },
        );
    }
    if req.sample_rate != 16000 {
        return json_response(
            StatusCode::BAD_REQUEST,
            &TranscribeErrorResponse {
                error: format!("sample_rate must be 16000 (got {}); resample to 16kHz mono client-side", req.sample_rate),
                engine: transcribe_engine_label(),
            },
        );
    }

    let raw = match base64::engine::general_purpose::STANDARD.decode(req.pcm_f32_base64.as_bytes()) {
        Ok(b) => b,
        Err(e) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &TranscribeErrorResponse { error: format!("pcm_f32_base64 is not valid base64: {e}"), engine: transcribe_engine_label() },
            );
        }
    };
    if raw.is_empty() || raw.len() % 4 != 0 {
        return json_response(
            StatusCode::BAD_REQUEST,
            &TranscribeErrorResponse {
                error: format!("decoded PCM must be a non-empty multiple of 4 bytes (little-endian f32); got {} bytes", raw.len()),
                engine: transcribe_engine_label(),
            },
        );
    }
    // 上限: 16kHz mono f32 で 10 分ぶん(= 16000 * 60 * 10 * 4 バイト ≒ 38MB)。
    const MAX_PCM_BYTES: usize = 16_000 * 60 * 10 * 4;
    if raw.len() > MAX_PCM_BYTES {
        return json_response(
            StatusCode::BAD_REQUEST,
            &TranscribeErrorResponse {
                error: format!("audio too long: {} bytes decoded, limit is {} (~10 minutes at 16kHz mono f32)", raw.len(), MAX_PCM_BYTES),
                engine: transcribe_engine_label(),
            },
        );
    }
    let pcm: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let language = req
        .language
        .as_deref()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("auto"))
        .map(str::to_string);

    // contextual biasing プロンプト(末尾 ~1000 文字に切り詰め。Whisper の
    // デコーダプロンプトは末尾しか効かないため末尾を残す)。
    let prompt = req
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            let chars: Vec<char> = p.chars().collect();
            if chars.len() > 1000 {
                chars[chars.len() - 1000..].iter().collect()
            } else {
                p.to_string()
            }
        });

    // whisper-cli の子プロセスは同期の重い処理(WAV 書き出し + プロセス
    // 起動 + 書き起こし待ち)。`generate` と同じく `spawn_blocking` へ
    // 逃がして tokio ワーカーを塞がない。
    let result = tokio::task::spawn_blocking(move || {
        transcribe::transcribe_pcm16k(&pcm, language.as_deref(), prompt.as_deref())
    })
    .await;

    match result {
        Ok(Ok(out)) => json_response(
            StatusCode::OK,
            &TranscribeResponse {
                transcript: out.text,
                language: out.language,
                engine: transcribe_engine_label(),
                disclosure: "Transcribed by whisper.cpp (GGML Whisper model) running on this aruaru-llm instance. \
                    Accuracy depends on the model size (ggml-base is fast but modest; use ggml-large-v3-turbo for best quality), \
                    audio quality, and background noise. Non-English accuracy varies by language.",
            },
        ),
        Ok(Err(e)) => {
            tracing::warn!("transcribe failed: {e}");
            json_response(
                StatusCode::SERVICE_UNAVAILABLE,
                &TranscribeErrorResponse { error: e, engine: transcribe_engine_label() },
            )
        }
        Err(join_err) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &TranscribeErrorResponse { error: format!("transcribe task panicked: {join_err}"), engine: transcribe_engine_label() },
        ),
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

#[derive(Debug, Deserialize, Default)]
struct LayerRedundancyRequest {
    /// トピックを分散させた複数の文を推奨。省略・空配列なら
    /// `mla_calibration_prompts()`(8文の一般英文)を既定として使う。
    #[serde(default)]
    sample_prompts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LayerRedundancyResponse {
    layers: Vec<open_cuda_llm::LayerRedundancyReport>,
    disclosure_ja: &'static str,
    disclosure_en: &'static str,
}

const LAYER_REDUNDANCY_DISCLOSURE_JA: &str = "これはDeepSeek固有の技術ではありません(調査の結果、「DeepSeekの折りたたみ理論」は\
    実在しないと判明しました)。ShortGPT(arXiv:2403.03853)/Gromov et al.(arXiv:2403.17887)方式の\
    層単位Block Influence冗長性検出です。block_influenceが低いほどその層は入力≒出力の恒等写像に近く\
    冗長と推定されます。これは少数のサンプル文からの推定であり、実際に層を除去する前の下調べです\
    (この分析自体はモデルを一切変更しません)。";
const LAYER_REDUNDANCY_DISCLOSURE_EN: &str = "This is NOT a DeepSeek-specific technique — our research found no such \
    thing as a 'DeepSeek folding theory'. This follows ShortGPT (arXiv:2403.03853) / Gromov et al. (arXiv:2403.17887): \
    layer-level Block Influence redundancy detection. A lower block_influence means that layer's output is closer to \
    an identity mapping of its input, suggesting redundancy. This is only an estimate from a small sample of prompts, \
    and a read-only preview before actually removing any layer (this analysis alone never modifies the model).";

/// `POST /v1/models/layer-redundancy` — 現在アクティブなモデルの各層に
/// ついて、Block Influence(層の入力≒出力ならほぼ0=冗長)を計算する
/// **読み取り専用**の分析。モデルは一切変更されない(`POST /v1/models/
/// fold-layers`とは異なる、実際に折りたたむ前の下調べ用エンドポイント)。
async fn layer_redundancy_handler(req: Request, device: Arc<dyn GpuDevice>) -> Response {
    let Json(body): Json<LayerRedundancyRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let result = tokio::task::spawn_blocking(move || generation::analyze_active_model_layer_redundancy(&device, &body.sample_prompts)).await;
    match result {
        Ok(Ok(layers)) => json_response(StatusCode::OK, &LayerRedundancyResponse { layers, disclosure_ja: LAYER_REDUNDANCY_DISCLOSURE_JA, disclosure_en: LAYER_REDUNDANCY_DISCLOSURE_EN }),
        Ok(Err(e)) => json_response(StatusCode::SERVICE_UNAVAILABLE, &InstallModelErrorResponse { error: format!("{e:#}") }),
        Err(e) => json_response(StatusCode::INTERNAL_SERVER_ERROR, &InstallModelErrorResponse { error: format!("layer_redundancy task panicked: {e}") }),
    }
}

#[derive(Debug, Deserialize)]
struct FoldLayersRequest {
    #[serde(default)]
    sample_prompts: Vec<String>,
    /// この値未満のBlock Influenceを持つ層を実際に除去する。既定
    /// `0.01`(=入力と出力のコサイン類似度が99%以上一致=ほぼ恒等写像の
    /// 層のみを対象、ShortGPT論文が報告する典型的な冗長層の水準より
    /// 保守的な既定値——「まず控えめに試す」ことを優先した)。
    /// **`num_layers_to_remove`が指定されている場合、この値は無視される**
    /// (下記参照)。
    #[serde(default = "default_block_influence_threshold")]
    block_influence_threshold: f32,
    /// **2026-09-01追加**: 指定した場合、独立閾値方式ではなく
    /// Gromov et al.(arXiv:2403.17887)方式の連続ブロック探索
    /// (`generation::fold_active_model_by_block`)を使う——複数層を
    /// まとめて除去したい場合はこちらを推奨(実測で明確に品質保持が
    /// 優れていることを確認済み、`aruaru-llm/CLAUDE.md`参照)。
    #[serde(default)]
    num_layers_to_remove: Option<usize>,
    /// **2026-09-01追加(さらに続き)**: `num_layers_to_remove`指定時のみ
    /// 有効。`true`なら除去ブロックを跡形もなく消すのではなく、
    /// 最小二乗法でフィットした軽量な線形アダプタ層へ置換する
    /// (`generation::fold_active_model_with_linear_adapter`)——極端な
    /// 予算(除去する層数がモデルの大部分を占める場合)で実測上、
    /// 完全な劣化ループを回避できることを確認済み(ただし完全な修正では
    /// ない、`aruaru-llm/CLAUDE.md`参照)。既定`false`(=跡形もなく削除)。
    #[serde(default)]
    use_linear_adapter: bool,
    /// **2026-09-01追加(さらに続き2、ridge_lambdaの外部調整可能化)**:
    /// `use_linear_adapter=true`のときのみ有効。線形アダプタのフィット
    /// (最小二乗法の閉形式リッジ回帰)に使う正則化係数。未指定
    /// (`null`)なら`open-cuda-llm`側の既定値`1e-2`を使う。値が非有限・
    /// 0以下の場合は`open-cuda-llm`側が`400`相当のエラーを返す
    /// (`fold_block_with_linear_adapter`のバリデーション)。較正データの
    /// 分散が大きい(例: 日本語プロンプトを混在させた場合)ほど正規方程式
    /// が悪条件になりやすく、`ridge_lambda`を大きくする必要が出ることが
    /// ある——この値を外部から調整可能にすることで、呼び出し側が
    /// 実測しながら最適値を探れるようにした。
    #[serde(default)]
    ridge_lambda: Option<f32>,
}

fn default_block_influence_threshold() -> f32 {
    0.01
}

#[derive(Debug, Serialize)]
struct FoldLayersResponse {
    redundancy: Vec<open_cuda_llm::LayerRedundancyReport>,
    original_layer_count: usize,
    pruned_layer_count: usize,
    removed_layer_indices: Vec<usize>,
    sample_prompt: String,
    completion_before_fold: String,
    completion_after_fold: String,
    disclosure_ja: String,
    disclosure_en: String,
    /// `open-cuda-llm`側`GptModel::prune_redundant_layers`が返す、
    /// アルゴリズム自体(Block Influence方式)の日英併記の開示文
    /// (上のdisclosure_ja/enは「DeepSeekとの混同」に関する開示、こちらは
    /// 「何のアルゴリズムを使ったか」の開示——別軸のため分けて返す)。
    layer_removal_technique_disclosure: &'static str,
    /// ブロック探索モード(`num_layers_to_remove`指定時)のみ設定される、
    /// `block_similarity`に基づく事前の品質見込み。閾値モードでは`null`。
    quality_hint: Option<&'static str>,
    /// **2026-09-01追加**: `use_linear_adapter=true`のときのみ設定される、
    /// 実際に使われた`ridge_lambda`値。リクエストで`ridge_lambda`を
    /// 指定した場合はその値がそのまま反映され、未指定(`null`)の場合は
    /// `open-cuda-llm`側の既定値`0.01`が反映される——呼び出し側が
    /// パラメータが黙って無視されていないことを確認できるようにする。
    ridge_lambda_used: Option<f32>,
    /// **2026-09-01追加**: `use_linear_adapter=true`のときのみ`true`。
    /// 挿入したアダプタ層が推論時にAttentionサブ層(QKV射影・softmax・
    /// P·V・KVキャッシュ)を丸ごとスキップし、除去したブロックぶんの
    /// Attention計算コストが実際に削減されることを示す(旧設計は出力を
    /// ゼロで捨てるだけで演算は残っていた)。閾値方式・ブロック探索方式
    /// (層を跡形もなく削除する)では`null`。
    attention_compute_skipped: Option<bool>,
}

/// `POST /v1/models/fold-layers` — 実際にモデルの層を除去し、**現在
/// アクティブなモデルを差し替える**。`num_layers_to_remove`を指定すると
/// Gromov et al.方式の連続ブロック探索(`generation::
/// fold_active_model_by_block`、複数層除去時に推奨)、未指定なら従来の
/// 独立閾値方式(`generation::fold_active_model`)を使う。折りたたみ前後の
/// 同一プロンプトへの生成結果を両方返すので、呼び出し側(UI等)は品質劣化の
/// 有無を実際の出力で確認できる。
async fn fold_layers_handler(req: Request, device: Arc<dyn GpuDevice>) -> Response {
    let Json(body): Json<FoldLayersRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let result = tokio::task::spawn_blocking(move || match (body.num_layers_to_remove, body.use_linear_adapter) {
        (Some(n), true) => generation::fold_active_model_with_linear_adapter(&device, &body.sample_prompts, n, body.ridge_lambda),
        (Some(n), false) => generation::fold_active_model_by_block(&device, &body.sample_prompts, n),
        (None, _) => generation::fold_active_model(&device, &body.sample_prompts, body.block_influence_threshold),
    })
    .await;
    match result {
        Ok(Ok(r)) => json_response(
            StatusCode::OK,
            &FoldLayersResponse {
                redundancy: r.redundancy,
                original_layer_count: r.prune_report.original_layer_count,
                pruned_layer_count: r.prune_report.pruned_layer_count,
                removed_layer_indices: r.prune_report.removed_layer_indices,
                sample_prompt: r.sample_prompt,
                completion_before_fold: r.completion_before_fold,
                completion_after_fold: r.completion_after_fold,
                disclosure_ja: r.disclosure_ja.to_string(),
                disclosure_en: r.disclosure_en.to_string(),
                layer_removal_technique_disclosure: r.prune_report.disclosure,
                quality_hint: r.quality_hint,
                ridge_lambda_used: r.ridge_lambda_used,
                attention_compute_skipped: r.attention_compute_skipped,
            },
        ),
        Ok(Err(e)) => {
            tracing::warn!("fold_active_model failed: {e:#}");
            json_response(StatusCode::BAD_REQUEST, &InstallModelErrorResponse { error: format!("{e:#}") })
        }
        Err(e) => json_response(StatusCode::INTERNAL_SERVER_ERROR, &InstallModelErrorResponse { error: format!("fold_layers task panicked: {e}") }),
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
/// モジュールdoc参照)。**2026-09-05追記**: 任意の`?precision=f16|f32|
/// f64|f128`クエリパラメータで、`hardware::InferencePrecision`
/// (2026-09-03新設だが`recommend()`からは呼び出せず「未接続」のまま
/// 残っていたギャップ)を考慮したVRAM見積もりを要求できる。未指定時は
/// 従来通りF32(後方互換)。不正な値は`400`で正直に拒否する。
async fn recommend_model(req: Request) -> Response {
    let precision = match req.uri().query().and_then(|q| {
        q.split('&').find_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            (k == "precision").then(|| v.to_string())
        })
    }) {
        Some(raw) => match hardware::InferencePrecision::parse(&raw) {
            Some(p) => p,
            None => {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({"error": format!("unknown precision '{raw}', expected one of: f16, f32, f64, f128")}),
                );
            }
        },
        None => hardware::InferencePrecision::F32,
    };
    json_response(StatusCode::OK, &hardware::recommend_at_precision(precision))
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

/// `GET /v1/runtime` — 現在の実行基盤(open-cudaデバイスプール・
/// ビルド時feature・アクティブモデル)を**正直に**返す(2026-08-22新設、
/// open-english側の「今どこで計算しているのか」表示用)。
///
/// **正直な開示**:
/// - このエンドポイントは「何が使われているか」を報告するだけで、
///   高速化そのものは一切行わない。
/// - `devices`は`device_pool::DevicePool`が実際に保持している
///   `opencuda_core::GpuDevice`の`info()`をそのまま出す。既定ビルドでは
///   `opencuda_cpu::CpuDevice`1台のみ(=GPUは使っていない)。
///   `--features real-vulkan`でビルドし、かつ`VulkanDevice::new`が実際に
///   成功した場合のみGPUが1台追加される。
/// - `open-directx`について: ここで参照し得るのは`open-cuda`内蔵の
///   `opencuda-directx`クレート(`hw-detect-directx` feature、既定オフ、
///   GPU**検出**のみで演算には使わない)であり、独立リポジトリ
///   `aon-co-jp/open-directx`とは無関係(CLAUDE.md 2026-08-20参照)。
#[derive(Debug, Serialize)]
struct RuntimeDeviceInfo {
    id: usize,
    name: String,
    vendor: String,
    total_memory_bytes: u64,
    compute_units: u32,
    supports_spirv: bool,
}

#[derive(Debug, Serialize)]
struct RuntimeInfoResponse {
    devices: Vec<RuntimeDeviceInfo>,
    device_count: usize,
    /// プール内に SPIR-V ディスパッチ可能な(=実GPU)デバイスが1台でもあるか。
    gpu_in_use: bool,
    /// ビルド時に有効化されているGPU関連feature名の一覧(空=CPUのみ)。
    enabled_gpu_features: Vec<&'static str>,
    /// `/v1/generate`が返すのと同じエンジン識別子(実行経路サフィックス付き)。
    engine: String,
    active_model_dir: Option<String>,
    /// 誇張しない一言サマリ(英日併記、open-englishがそのまま表示できる)。
    summary_en: String,
    summary_ja: String,
    /// CPU 推論経路で実際に選ばれている SIMD 実装(open-cpu の検出結果)。
    ///
    /// 2026-08-23 追加。単一命令の有無ではなく **組み合わせ**
    /// (AVX2+FMA3 等)で決まるプロファイルを返す。
    cpu_simd: CpuSimdInfo,
    /// 階層的アクセラレーション(CUDA → Vulkan → DirectX → CPU SIMD)の
    /// うち、**実際に**どの段が有効になっているか(2026-08-23追加)。
    acceleration: AccelerationInfo,
    /// llama.cpp 風のバックエンド行列(2026-09-03追加)。未実装の将来経路
    /// (Metal / HIP / SYCL / WebGPU)も含めた全体像。**報告のみ**。
    /// 正本: `open-cuda/OmniGPU-Design.md` §11.6 / §12.3。
    backend_matrix: Vec<BackendMatrixRow>,
    /// `POST /v1/transcribe`(whisper.cpp 音声認識)の状態(2026-08-29追加、
    /// SPEECH_RECOGNITION_REDESIGN.md P2-β)。
    whisper: WhisperTierInfo,
    disclosure: &'static str,
}

/// `POST /v1/transcribe`(whisper.cpp CLI 音声認識)の可否と実体パス
/// (2026-08-29新設・方針変更)。
///
/// **正直な開示**: 実装は whisper.cpp のプレビルド CLI(`whisper-cli`)を
/// 子プロセス起動する方式(`whisper-rs` の直接リンクは Windows/MSVC で
/// ビルド不能なため撤回)。`cli_present` と `model_present` の両方が
/// true のときだけ `/v1/transcribe` が実際に書き起こせる。いずれも
/// リポジトリには同梱していない。
#[derive(Debug, Serialize)]
struct WhisperTierInfo {
    /// CLI・モデルの両方が実在し `/v1/transcribe` が動作可能か。
    available: bool,
    /// `"whisper.cpp-cli"` / `"not-available"`。
    backend: &'static str,
    cli_path: String,
    cli_present: bool,
    model_path: String,
    model_present: bool,
    detail: &'static str,
}

fn whisper_tier_info() -> WhisperTierInfo {
    let cli_present = transcribe::cli_present();
    let model_present = transcribe::model_present();
    WhisperTierInfo {
        available: cli_present && model_present,
        backend: transcribe::backend_label(),
        cli_path: transcribe::cli_path().to_string_lossy().to_string(),
        cli_present,
        model_path: transcribe::model_path().to_string_lossy().to_string(),
        model_present,
        detail: if cli_present && model_present {
            "POST /v1/transcribe is ready (whisper-cli and a GGML model are both present)."
        } else if !cli_present && !model_present {
            "Neither whisper-cli nor a GGML model was found. Download a prebuilt whisper-cli from \
             github.com/ggml-org/whisper.cpp/releases and a model (e.g. ggml-base.bin); \
             or set ARUARU_LLM_WHISPER_CLI / ARUARU_LLM_WHISPER_MODEL."
        } else if !cli_present {
            "whisper-cli not found (a GGML model is present). Add a prebuilt whisper-cli or set ARUARU_LLM_WHISPER_CLI."
        } else {
            "GGML model not found (whisper-cli is present). Add ggml-base.bin or set ARUARU_LLM_WHISPER_MODEL."
        },
    }
}

/// 階層的アクセラレーション各段の状態(2026-08-23新設)。
///
/// **正直な開示**: `compiled_in`はビルド時featureの有無、`active`は
/// 実行時に実際にその経路が使われているかを表す。両方trueの段だけが
/// 「実際に効いている」——それ以外は下位段へフォールバックしている。
#[derive(Debug, Serialize)]
struct TierStatus {
    compiled_in: bool,
    active: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct AccelerationInfo {
    /// 実際に有効な最上位の段(`cuda` / `vulkan` / `directx-gemm` / `cpu-simd`)。
    tier: &'static str,
    /// 人間向けの短いラベル(open-englishのUIがそのまま表示できる)。
    tier_label_en: String,
    tier_label_ja: String,
    cuda: TierStatus,
    vulkan: TierStatus,
    directx: TierStatus,
    cpu_simd: TierStatus,
}

fn acceleration_info(pool: &device_pool::DevicePool) -> AccelerationInfo {
    let vulkan_device_present = pool.devices().iter().any(|d| d.supports_spirv());
    let vulkan_active = vulkan_device_present && generation::matmul_spirv_wired();
    let directx_active = generation::matmul_dxil_offloaded();

    // CUDA(NVIDIA専用ネイティブ経路)は open-cuda 側に実バックエンドが
    // 存在しない(`GemmPath::CuBlas`はスタブのまま)。NVIDIA GPUであっても
    // 実際にはVulkan経路で動く。ここで嘘をつかないよう常に非アクティブと
    // 報告する。
    let cuda = TierStatus {
        compiled_in: false,
        active: false,
        detail: "open-cuda has no native CUDA/cuBLAS backend yet (GemmPath::CuBlas is still a stub); \
                 NVIDIA GPUs are used through the Vulkan path instead."
            .to_string(),
    };
    let vulkan = TierStatus {
        compiled_in: cfg!(feature = "real-vulkan"),
        active: vulkan_active,
        detail: if vulkan_active {
            "Dense GEMM + attention dispatched to a real Vulkan compute device (SPIR-V).".to_string()
        } else if cfg!(feature = "real-vulkan") {
            "real-vulkan compiled in, but either no Vulkan device was created or matmul.spv was not wired.".to_string()
        } else {
            "Not compiled in (build with --features real-vulkan).".to_string()
        },
    };
    let directx = TierStatus {
        compiled_in: cfg!(feature = "real-dx12"),
        active: directx_active,
        detail: if directx_active {
            "Dense GEMM (QKV / attn_out / MLP / lm_head) offloaded to a real D3D12 compute device via matmul.dxil. \
             Attention, LayerNorm and GELU still run on the CPU device."
                .to_string()
        } else if cfg!(feature = "real-dx12") {
            "real-dx12 compiled in, but no D3D12 device could be created (or the Vulkan tier took priority).".to_string()
        } else {
            "Not compiled in (build with --features real-dx12).".to_string()
        },
    };
    let cpu_simd = TierStatus {
        compiled_in: true,
        active: !vulkan_active,
        detail: format!(
            "open-cpu runtime dispatch: {}. Always used for everything not offloaded to a GPU tier.",
            opencuda_blas::simd::cpu_features().describe()
        ),
    };

    let tier = if vulkan_active {
        "vulkan"
    } else if directx_active {
        "directx-gemm"
    } else {
        "cpu-simd"
    };
    let (tier_label_en, tier_label_ja) = match tier {
        "vulkan" => ("GPU (Vulkan compute)".to_string(), "GPU(Vulkan Compute)".to_string()),
        "directx-gemm" => (
            "GPU (DirectX 12 compute, dense GEMM only) + CPU SIMD".to_string(),
            "GPU(DirectX 12 Compute、密GEMMのみ)+ CPU SIMD".to_string(),
        ),
        _ => (
            format!("CPU SIMD ({})", opencuda_blas::simd::cpu_features().isa_profile()),
            format!("CPU SIMD({})", opencuda_blas::simd::cpu_features().isa_profile()),
        ),
    };

    AccelerationInfo { tier, tier_label_en, tier_label_ja, cuda, vulkan, directx, cpu_simd }
}

/// llama.cpp 風のバックエンド行列 1 行(2026-09-03 新設、`open-cuda`
/// `OmniGPU-Design.md` §12.3 の「バックエンド行列を llama.cpp に倣って
/// 明示する」方針)。**報告のみ**——この行列は挙動を一切変えない。
///
/// `status`:
/// - `"active"` = 今この経路で計算している
/// - `"compiled-in"` = ビルドに含まれるが実行時には選ばれていない
/// - `"not-compiled-in"` = feature を有効にすれば使える
/// - `"planned"` = 設計方針として記録済み・未実装
#[derive(Debug, Serialize)]
struct BackendMatrixRow {
    backend: &'static str,
    status: &'static str,
    note_en: &'static str,
    note_ja: &'static str,
}

/// `GET /v1/runtime` が返すバックエンド行列。`acceleration`(実際にどの段が
/// 効いているか)と重複するが、こちらは **未実装の将来経路(Metal / HIP /
/// SYCL / WebGPU)も含めた全体像** を llama.cpp のバックエンド表と同じ粒度で
/// 見せることが目的(正本: `open-cuda/OmniGPU-Design.md` §11.6 / §12.3)。
fn backend_matrix(accel: &AccelerationInfo) -> Vec<BackendMatrixRow> {
    let status = |compiled: bool, active: bool| -> &'static str {
        if active {
            "active"
        } else if compiled {
            "compiled-in"
        } else {
            "not-compiled-in"
        }
    };
    vec![
        BackendMatrixRow {
            backend: "cpu-simd",
            status: if accel.cpu_simd.active { "active" } else { "compiled-in" },
            note_en: "open-cpu runtime dispatch (AVX2+FMA3 measured ~3.34x vs scalar). Always built in; \
                      handles everything not offloaded to a GPU tier.",
            note_ja: "open-cpu の実行時ディスパッチ(AVX2+FMA3 でスカラー比 実測 約3.34倍)。\
                      常にビルドされ、GPU 段へオフロードされない全処理を担う。",
        },
        BackendMatrixRow {
            backend: "vulkan-spirv",
            status: status(accel.vulkan.compiled_in, accel.vulkan.active),
            note_en: "open-cuda SPIR-V compute. The portability backbone (NVIDIA/AMD/Intel on \
                      Linux/Windows, macOS via MoltenVK). Build with --features real-vulkan.",
            note_ja: "open-cuda の SPIR-V compute。移植性の背骨(Linux/Windows の NVIDIA/AMD/Intel、\
                      macOS は MoltenVK 経由)。--features real-vulkan でビルド。",
        },
        BackendMatrixRow {
            backend: "directx-dxil",
            status: status(accel.directx.compiled_in, accel.directx.active),
            note_en: "open-cuda D3D12/DXIL, dense GEMM only. Fallback for Windows without Vulkan. \
                      Build with --features real-dx12. Slower than CPU SIMD on this dev box (GT 730).",
            note_ja: "open-cuda の D3D12/DXIL、密 GEMM のみ。Vulkan が無い Windows 向けフォールバック。\
                      --features real-dx12 でビルド。この開発機(GT 730)では CPU SIMD より遅い。",
        },
        BackendMatrixRow {
            backend: "cuda",
            status: "not-compiled-in",
            note_en: "No native CUDA/cuBLAS backend in open-cuda (GemmPath::CuBlas is a stub). \
                      NVIDIA GPUs run through the Vulkan path.",
            note_ja: "open-cuda にネイティブ CUDA/cuBLAS バックエンドは無い(GemmPath::CuBlas はスタブ)。\
                      NVIDIA GPU も Vulkan 経路で動く。",
        },
        BackendMatrixRow {
            backend: "metal",
            status: "planned",
            note_en: "Reach Apple GPUs via MoltenVK (Vulkan-subset-on-Metal) with the existing SPIR-V \
                      kernels — no new backend. Real-machine verification pending. See OmniGPU-Design.md §11.3.",
            note_ja: "Apple GPU は MoltenVK(Vulkan サブセット on Metal)経由で既存の SPIR-V カーネルの\
                      まま到達する——新バックエンドは書かない。実機検証は保留。OmniGPU-Design.md §11.3。",
        },
        BackendMatrixRow {
            backend: "hip-rocm",
            status: "planned",
            note_en: "AMD native path (hipBLASLt). Optional accelerated path only; the portable route \
                      for AMD is Vulkan + VK_EXT_shader_float8 (shipping in Adrenalin 25.10.2+).",
            note_ja: "AMD ネイティブ経路(hipBLASLt)。任意の高速化経路のみ。AMD の移植性経路は \
                      Vulkan + VK_EXT_shader_float8(Adrenalin 25.10.2 以降で出荷)。",
        },
        BackendMatrixRow {
            backend: "sycl-levelzero",
            status: "planned",
            note_en: "Intel Arc/Xe native path (oneAPI Level Zero). Optional; Intel GPUs are covered by \
                      the Vulkan/SPIR-V path today.",
            note_ja: "Intel Arc/Xe ネイティブ経路(oneAPI Level Zero)。任意。Intel GPU は現状 \
                      Vulkan/SPIR-V 経路でカバーされる。",
        },
        BackendMatrixRow {
            backend: "webgpu-wasm",
            status: "planned",
            note_en: "Browser inference via wgpu/WebGPU (W3C Candidate Recommendation Draft, 2026-05). \
                      Future option; see the 2026-08-25 in-browser-AI plan and 'Llamas on the Web'.",
            note_ja: "wgpu/WebGPU によるブラウザ推論(W3C Candidate Recommendation Draft、2026-05)。\
                      将来オプション。2026-08-25 のブラウザ内 AI 構想・『Llamas on the Web』参照。",
        },
    ]
}

/// CPU 側 SIMD ディスパッチの状況(`open-cpu` + `opencuda-blas` の実測値)。
#[derive(Debug, Serialize)]
struct CpuSimdInfo {
    /// 検出された命令セットの組み合わせ(例 `"avx2+fma3+sse2"`)。
    features: String,
    /// `open-cpu` が判定した組み合わせプロファイル(例 `"avx2+fma3"`)。
    isa_profile: String,
    /// f32 GEMM/内積で AVX2+FMA3 経路が使われているか。
    avx2_fma_path: bool,
    /// int8 VNNI 経路が使えるか(このマシンでは false、実機未検証)。
    vnni_path: bool,
    /// AVX-512 経路が有効化されているか(`OPEN_CPU_ENABLE_AVX512=1` が必要)。
    avx512_opt_in: bool,
    note_ja: &'static str,
}

fn cpu_simd_info() -> CpuSimdInfo {
    let f = opencuda_blas::simd::cpu_features();
    CpuSimdInfo {
        features: f.describe(),
        isa_profile: f.isa_profile().to_string(),
        avx2_fma_path: f.has_avx2_fma(),
        vnni_path: f.has_vnni_path(),
        avx512_opt_in: open_cpu::avx512_opt_in(),
        note_ja: "CPU推論のGEMM/内積は open-cpu の検出結果に基づき実行時ディスパッチされる。                  AVX-512 経路は開発機に非搭載で実機未検証のため、既定では選択されない。",
    }
}

async fn runtime_info(pool: Arc<device_pool::DevicePool>) -> Response {
    let devices: Vec<RuntimeDeviceInfo> = pool
        .devices()
        .iter()
        .map(|d| {
            let info = d.info();
            RuntimeDeviceInfo {
                id: info.id,
                name: info.name.clone(),
                vendor: format!("{:?}", info.vendor),
                total_memory_bytes: info.total_memory,
                compute_units: info.compute_units,
                supports_spirv: d.supports_spirv(),
            }
        })
        .collect();
    let gpu_in_use = devices.iter().any(|d| d.supports_spirv);

    let mut enabled_gpu_features: Vec<&'static str> = Vec::new();
    if cfg!(feature = "real-vulkan") {
        enabled_gpu_features.push("real-vulkan");
    }
    if cfg!(feature = "hw-detect-vulkan") {
        enabled_gpu_features.push("hw-detect-vulkan");
    }
    if cfg!(feature = "hw-detect-directx") {
        enabled_gpu_features.push("hw-detect-directx");
    }
    if cfg!(feature = "real-dx12") {
        enabled_gpu_features.push("real-dx12");
    }

    // engine_labelはデバイスごとに実行経路が変わるため、プール先頭
    // (常にCpuDevice)ではなくGPUがあればGPU側を代表として使う。
    let representative = pool
        .devices()
        .iter()
        .find(|d| d.supports_spirv())
        .cloned()
        .unwrap_or_else(|| pool.next_device());
    let engine = generation::engine_label(&representative);
    let active_model_dir = generation::active_model_dir().map(|p| p.to_string_lossy().to_string());

    let acceleration = acceleration_info(&pool);
    let backend_matrix = backend_matrix(&acceleration);
    let names = devices.iter().map(|d| d.name.clone()).collect::<Vec<_>>().join(", ");
    let (summary_en, summary_ja) = if acceleration.tier == "directx-gemm" {
        (
            format!("Dense GEMM offloaded to DirectX 12 compute; everything else on CPU SIMD ({names})."),
            format!("密GEMMをDirectX 12 Computeへオフロード中。その他はCPU SIMDで実行({names})。"),
        )
    } else if gpu_in_use {
        (
            format!("GPU acceleration active via open-cuda ({names})."),
            format!("open-cuda経由でGPUを使用中({names})。"),
        )
    } else {
        (
            format!("CPU only via open-cuda CpuDevice ({names}); no GPU backend is compiled in or available."),
            format!("open-cudaのCpuDeviceによるCPU実行のみ({names})。GPUバックエンドはこのビルドに含まれていないか利用できません。"),
        )
    };

    json_response(
        StatusCode::OK,
        &RuntimeInfoResponse {
            device_count: devices.len(),
            devices,
            gpu_in_use,
            enabled_gpu_features,
            engine,
            active_model_dir,
            summary_en,
            summary_ja,
            cpu_simd: cpu_simd_info(),
            acceleration,
            backend_matrix,
            whisper: whisper_tier_info(),
            disclosure: "This endpoint only reports which open-cuda device backend is actually in use; \
                         it does not itself accelerate anything. The default build is CPU-only. \
                         The standalone aon-co-jp/open-directx repository is not involved.",
        },
    )
}

/// アイドル時バックグラウンドModel Folding準備スケジューラの進捗確認用
/// (2026-08-19新設、`idle_background_fold.rs`参照)。
async fn background_fold_status() -> Response {
    json_response(StatusCode::OK, &idle_background_fold::current_progress())
}

/// USB接続スマホ向けタスク配布(2026-08-19新設、`phone_task.rs`参照)。
/// 常に1件のタスクを返す(キューの空/満杯という概念は持たない簡易実装、
/// 呼び出すたびに新しい題材を生成する)。
async fn background_fold_task() -> Response {
    json_response(StatusCode::OK, &phone_task::next_task())
}

/// スマホ側が計算した結果を受け取る(2026-08-19新設)。受け取って記録する
/// のみで、実際のモデル推論・Model Foldingへは反映しない(`phone_task.rs`
/// モジュールdoc参照)。
async fn background_fold_task_result(req: Request) -> Response {
    let Json(body): Json<phone_task::TaskResultRequest> = match Json::from_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let record = phone_task::record_result(body);
    json_response(StatusCode::OK, &record)
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
/// サーバー接続先国のニュースを検出・取得しローカルDBへ保存する
/// (`POST /v1/news/refresh`、`news_geo.rs`参照)。open-englishの
/// メンテナンスバナー表示中に叩かれる想定。
async fn news_refresh() -> Response {
    let db = crate::news_geo::refresh().await;
    json_response(StatusCode::OK, &db)
}

/// 直近保存済みのニュースDBスナップショットを返す(`GET /v1/news/latest`)。
async fn news_latest() -> Response {
    let db = crate::news_geo::get_latest();
    json_response(StatusCode::OK, &db)
}

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

    // アイドル時バックグラウンドModel Folding準備スケジューラを起動
    // (2026-08-19新設)。専用の低優先度std::threadで動作し、HTTP処理を
    // 一切妨げない。詳細・正直な開示はidle_background_fold.rsのモジュール
    // docおよびCLAUDE.mdのHANDOFF(2026-08-19)参照。
    idle_background_fold::spawn();

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
    let runtime_info_pool = Arc::clone(&device_pool);
    let chat_pool = Arc::clone(&device_pool);
    let chat_registry = Arc::clone(&registry);
    let classify_pool = Arc::clone(&device_pool);
    let classify_registry = Arc::clone(&registry);
    let classify_traffic_pool = Arc::clone(&device_pool);
    let classify_traffic_registry = Arc::clone(&registry);
    let generate_pool = Arc::clone(&device_pool);
    let generate_registry = Arc::clone(&registry);
    let generate_speculative_pool = Arc::clone(&device_pool);
    let generate_speculative_registry = Arc::clone(&registry);
    let generate_search_pool = Arc::clone(&device_pool);
    let generate_search_registry = Arc::clone(&registry);
    let translate_pool = Arc::clone(&device_pool);
    let translate_registry = Arc::clone(&registry);
    let layer_redundancy_pool = Arc::clone(&device_pool);
    let fold_layers_pool = Arc::clone(&device_pool);
    let transcribe_registry = Arc::clone(&registry);
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
            "/v1/generate-speculative",
            post(handler_fn(move |req, _p| {
                let device = generate_speculative_pool.next_device();
                let registry = Arc::clone(&generate_speculative_registry);
                async move { generate_speculative(req, device, registry).await }
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
        .at(
            "/v1/transcribe",
            post(handler_fn(move |req, _p| {
                let registry = Arc::clone(&transcribe_registry);
                async move { transcribe(req, registry).await }
            })),
        )
        .at("/v1/models/catalog", get(plain(|| Box::pin(list_model_catalog()))))
        .at("/v1/geo/random", get(plain(|| Box::pin(geo_random()))))
        .at("/v1/geo/fuji", get(plain(|| Box::pin(geo_fuji()))))
        .at("/v1/geo/tours", post(handler_fn(|req, _p| Box::pin(geo_tours(req)))))
        .at("/v1/referrals/check", post(handler_fn(|req, _p| Box::pin(referrals_check(req)))))
        .at("/v1/news/refresh", post(plain(|| Box::pin(news_refresh()))))
        .at("/v1/news/latest", get(plain(|| Box::pin(news_latest()))))
        .at("/v1/geo/lookup", post(handler_fn(|req, _p| Box::pin(geo_lookup(req)))))
        .at(
            "/v1/settings/google-search",
            post(handler_fn(|req, _p| Box::pin(set_google_search_settings(req))))
                .get(plain(|| Box::pin(get_google_search_settings_status())))
                .delete(plain(|| Box::pin(clear_google_search_settings()))),
        )
        .at(
            "/v1/settings/github-search",
            post(handler_fn(|req, _p| Box::pin(set_github_search_settings(req))))
                .get(plain(|| Box::pin(get_github_search_settings_status())))
                .delete(plain(|| Box::pin(clear_github_search_settings()))),
        )
        .at(
            "/v1/settings/youtube-search",
            post(handler_fn(|req, _p| Box::pin(set_youtube_search_settings(req))))
                .get(plain(|| Box::pin(get_youtube_search_settings_status())))
                .delete(plain(|| Box::pin(clear_youtube_search_settings()))),
        )
        .at(
            "/v1/settings/chat-providers",
            post(handler_fn(|req, _p| Box::pin(set_chat_provider_settings(req))))
                .get(plain(|| Box::pin(get_chat_provider_settings_status())))
                .delete(plain(|| Box::pin(clear_chat_provider_settings()))),
        )
        .at("/v1/chat-providers/complete", post(handler_fn(|req, _p| Box::pin(chat_provider_complete(req)))))
        .at("/v1/chat-providers/complete-multi", post(handler_fn(|req, _p| Box::pin(chat_provider_complete_multi(req)))))
        .at("/v1/chat-providers/complete-priority", post(handler_fn(|req, _p| Box::pin(chat_provider_complete_priority(req)))))
        .at(
            "/v1/settings/provider-priority",
            post(handler_fn(|req, _p| Box::pin(set_provider_priority_settings(req))))
                .get(plain(|| Box::pin(get_provider_priority_settings_status())))
                .delete(plain(|| Box::pin(reset_provider_priority_settings()))),
        )
        .at(
            "/v1/models/optimize-cache",
            post(handler_fn(|req, _p| Box::pin(optimize_model_cache_handler(req)))),
        )
        .at("/v1/models/install", post(handler_fn(|req, _p| Box::pin(install_model(req)))))
        .at("/v1/models/select", post(handler_fn(|req, _p| Box::pin(select_model(req)))))
        .at(
            "/v1/models/layer-redundancy",
            post(handler_fn(move |req, _p| {
                let device = layer_redundancy_pool.next_device();
                async move { layer_redundancy_handler(req, device).await }
            })),
        )
        .at(
            "/v1/models/fold-layers",
            post(handler_fn(move |req, _p| {
                let device = fold_layers_pool.next_device();
                async move { fold_layers_handler(req, device).await }
            })),
        )
        .at("/v1/recommend", get(handler_fn(|req, _p| Box::pin(recommend_model(req)))))
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
        .at(
            "/v1/runtime",
            get(handler_fn(move |_req, _p| {
                let pool = Arc::clone(&runtime_info_pool);
                async move { runtime_info(pool).await }
            })),
        )
        .at("/v1/background-fold/status", get(plain(|| Box::pin(background_fold_status()))))
        .at("/v1/background-fold/task", get(plain(|| Box::pin(background_fold_task()))))
        .at("/v1/background-fold/task-result", post(handler_fn(|req, _p| Box::pin(background_fold_task_result(req)))))
        .with_cors();

    // `ARUARU_LLM_BIND`環境変数で上書き可能(2026-08-11追加、Android
    // 単体版向け——端末上で自己完結させるため`127.0.0.1`限定で起動し、
    // 外部ネットワークへは一切listenしないようにする)。
    let bind_addr: std::net::SocketAddr = std::env::var("ARUARU_LLM_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "0.0.0.0:4600".parse().expect("static bind address is always valid"));
    tracing::info!("aruaru-llm listening on {bind_addr} (shared multi-tenant instance)");

    // 自動アップデート機能(2026-08-19新設、self_update.rs参照)。既定は
    // 無効(ARUARU_LLM_ENABLE_SELF_UPDATE未設定)——サーバー起動をブロック
    // しないよう非同期タスクとして起動する。
    tokio::spawn(self_update::check_and_apply_update(bind_addr));

    let (_addr, handle) = Server::new(TcpListener::bind(bind_addr)).run(app).await?;
    handle.await.map_err(|e| anyhow::anyhow!("server task panicked: {e}"))
}

#[cfg(test)]
mod runtime_backend_matrix_tests {
    use super::*;

    fn dummy_tier(compiled: bool, active: bool) -> TierStatus {
        TierStatus { compiled_in: compiled, active, detail: String::new() }
    }

    fn dummy_accel(vulkan_active: bool, directx_compiled: bool) -> AccelerationInfo {
        AccelerationInfo {
            tier: if vulkan_active { "vulkan" } else { "cpu-simd" },
            tier_label_en: String::new(),
            tier_label_ja: String::new(),
            cuda: dummy_tier(false, false),
            vulkan: dummy_tier(vulkan_active, vulkan_active),
            directx: dummy_tier(directx_compiled, false),
            cpu_simd: dummy_tier(true, !vulkan_active),
        }
    }

    #[test]
    fn backend_matrix_covers_every_backend_with_a_valid_status() {
        let rows = backend_matrix(&dummy_accel(false, false));
        let names: Vec<&str> = rows.iter().map(|r| r.backend).collect();
        assert_eq!(
            names,
            vec![
                "cpu-simd",
                "vulkan-spirv",
                "directx-dxil",
                "cuda",
                "metal",
                "hip-rocm",
                "sycl-levelzero",
                "webgpu-wasm",
            ]
        );
        for r in &rows {
            assert!(
                matches!(r.status, "active" | "compiled-in" | "not-compiled-in" | "planned"),
                "backend {} has an unexpected status {:?}",
                r.backend,
                r.status
            );
            assert!(!r.note_en.is_empty() && !r.note_ja.is_empty(), "backend {} missing a note", r.backend);
        }
    }

    #[test]
    fn cpu_simd_is_always_at_least_compiled_in_never_missing() {
        for accel in [dummy_accel(false, false), dummy_accel(true, true)] {
            let rows = backend_matrix(&accel);
            let cpu = rows.iter().find(|r| r.backend == "cpu-simd").unwrap();
            assert_ne!(cpu.status, "not-compiled-in", "cpu-simd is always built in");
        }
    }

    #[test]
    fn vulkan_row_reflects_the_acceleration_tier_state() {
        let inactive = backend_matrix(&dummy_accel(false, false));
        assert_eq!(inactive.iter().find(|r| r.backend == "vulkan-spirv").unwrap().status, "not-compiled-in");

        let active = backend_matrix(&dummy_accel(true, false));
        assert_eq!(active.iter().find(|r| r.backend == "vulkan-spirv").unwrap().status, "active");
        // Vulkan が有効なら cpu-simd は "compiled-in"(active ではない)。
        assert_eq!(active.iter().find(|r| r.backend == "cpu-simd").unwrap().status, "compiled-in");
    }

    #[test]
    fn future_backends_are_marked_planned() {
        let rows = backend_matrix(&dummy_accel(true, true));
        for name in ["metal", "hip-rocm", "sycl-levelzero", "webgpu-wasm"] {
            assert_eq!(
                rows.iter().find(|r| r.backend == name).unwrap().status,
                "planned",
                "{name} should be reported as planned, not implemented"
            );
        }
    }
}
