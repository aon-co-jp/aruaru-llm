//! 外部チャットLLM API(ChatGPT/DeepSeek/Gemini/Claude)への薄いマルチ
//! プロバイダ連携(ユーザー指示「open-englishのGoogle検索APIキーの他に、
//! ChatGPT無料枠、DeepSeek無料枠、Gemini、Claudeを単体でも同時実行でも
//! 使えるようにしてほしい」への対応)。
//!
//! ## 正直な開示(最重要)
//!
//! - このモジュールは`web_search.rs`(Google Custom Search連携)と全く
//!   同じ設計パターンを踏襲する: 各プロバイダのAPIキーはユーザー自身が
//!   各社公式サイトで取得し、ブラウザの設定パネルからCOPY&PASTEするか
//!   環境変数で渡す(このリポジトリはいかなるAPIキーも同梱・保持しない)。
//! - 各社の無料枠情報は`provider-free-tiers.json`(`open-english`側)を
//!   参照——このモジュール自体は無料枠かどうかを判定・強制しない
//!   (呼び出し元がどのAPIキーを渡すかだけで決まる、課金は各社の契約に
//!   従う)。
//! - `aruaru-llm`本体(GPT-2/distilgpt2のローカル推論)は契約不要の
//!   自己完結型AIという設計思想だが、ここで連携する4プロバイダは
//!   いずれも外部サービスへの契約が前提の意図的な例外である
//!   (`web_search.rs`のGoogle Custom Searchと同じ位置づけ)。
//! - 未設定のプロバイダは黙って空応答を返さず、正直にエラーを返す
//!   (`is_configured`で呼び出し元が事前に判別できる設計)。
//! - 複数プロバイダの「同時実行」は、各プロバイダへ並列にHTTPリクエストを
//!   投げて結果を集約するだけであり、1つの応答へ統合・要約する処理は
//!   行わない(呼び出し元・利用者が結果を比較できるよう、プロバイダ別の
//!   生の応答をそのまま返す)。

use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::provider_priority::{self, PriorityService};

/// 対応プロバイダ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Openai,
    Deepseek,
    Gemini,
    Claude,
    /// Grok(xAI、2026-09-12追加、ユーザー指示「ChatGPTの次はGeminiの次は、
    /// DeepSeekの次はGrokの無料枠と順番に」への対応)。xAIのChat
    /// Completions APIはOpenAI互換のリクエスト/レスポンス形状のため、
    /// DeepSeekと同様に`OpenAiRequest`/`OpenAiResponse`をそのまま
    /// エンドポイントだけ変えて再利用する。
    Grok,
    /// Groq(2026-09-21追加)。OpenAI互換API(api.groq.com/openai/v1)。
    Groq,
    /// Cerebras(2026-09-21追加)。OpenAI互換API(api.cerebras.ai/v1)。
    Cerebras,
    /// Mistral(2026-09-21追加)。OpenAI互換API(api.mistral.ai/v1)。
    Mistral,
    /// Ollama(2026-09-21追加)。この端末で動くOllama(既定http://localhost:11434)の
    /// OpenAI互換API。APIキーは不要で、環境変数の値=**モデル名**として扱う。
    Ollama,
    /// OpenRouter(2026-09-21追加)。OpenAI互換API(openrouter.ai/api/v1)。
    OpenRouter,
    /// Cloudflare Workers AI(2026-09-21追加)。OpenAI互換API。キーは「アカウントID:APIトークン」の形式。
    Cloudflare,
}

impl Provider {
    fn env_var_name(self) -> &'static str {
        match self {
            Provider::Openai => "ARUARU_LLM_OPENAI_API_KEY",
            Provider::Deepseek => "ARUARU_LLM_DEEPSEEK_API_KEY",
            Provider::Gemini => "ARUARU_LLM_GEMINI_API_KEY",
            Provider::Claude => "ARUARU_LLM_ANTHROPIC_API_KEY",
            Provider::Grok => "ARUARU_LLM_GROK_API_KEY",
            Provider::Groq => "ARUARU_LLM_GROQ_API_KEY",
            Provider::Cerebras => "ARUARU_LLM_CEREBRAS_API_KEY",
            Provider::Mistral => "ARUARU_LLM_MISTRAL_API_KEY",
            // 値はAPIキーではなくモデル名(例: ministral-3b)。設定されていればOllama有効。
            Provider::Ollama => "ARUARU_LLM_OLLAMA_MODEL",
            Provider::OpenRouter => "ARUARU_LLM_OPENROUTER_API_KEY",
            // 値は「アカウントID:APIトークン」(1つの環境変数に収めるため)。
            Provider::Cloudflare => "ARUARU_LLM_CLOUDFLARE_API_KEY",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Provider::Openai => "openai",
            Provider::Deepseek => "deepseek",
            Provider::Gemini => "gemini",
            Provider::Claude => "claude",
            Provider::Grok => "grok",
            Provider::Groq => "groq",
            Provider::Cerebras => "cerebras",
            Provider::Mistral => "mistral",
            Provider::Ollama => "ollama",
            Provider::OpenRouter => "openrouter",
            Provider::Cloudflare => "cloudflare",
        }
    }

    fn all() -> [Provider; 11] {
        [
            Provider::Openai,
            Provider::Deepseek,
            Provider::Gemini,
            Provider::Claude,
            Provider::Grok,
            Provider::Groq,
            Provider::Cerebras,
            Provider::Mistral,
            Provider::Ollama,
            Provider::OpenRouter,
            Provider::Cloudflare,
        ]
    }

    /// `provider_priority::PriorityService`(Google検索を含む5サービス
    /// 共通の優先順位リスト)から、チャット補完系の4プロバイダに該当する
    /// ものだけを対応付ける(Google検索は`web_search.rs`側が別途扱う)。
    fn from_priority_service(svc: PriorityService) -> Option<Provider> {
        match svc {
            PriorityService::GoogleSearch => None,
            PriorityService::Openai => Some(Provider::Openai),
            PriorityService::Deepseek => Some(Provider::Deepseek),
            PriorityService::Gemini => Some(Provider::Gemini),
            PriorityService::Claude => Some(Provider::Claude),
            PriorityService::Grok => Some(Provider::Grok),
            PriorityService::Groq => Some(Provider::Groq),
            PriorityService::Cerebras => Some(Provider::Cerebras),
            PriorityService::Mistral => Some(Provider::Mistral),
            PriorityService::Ollama => Some(Provider::Ollama),
            PriorityService::OpenRouter => Some(Provider::OpenRouter),
            PriorityService::Cloudflare => Some(Provider::Cloudflare),
        }
    }
}

/// 利用者がブラウザの設定パネルから入力したAPIキーを、実行中のプロセスの
/// メモリ上にのみ保持する(`web_search::RUNTIME_CREDENTIALS`と同じ設計:
/// ディスク書き込み・ログ出力は一切行わず、プロセス再起動で消える)。
static RUNTIME_KEYS: RwLock<Option<HashMap<Provider, String>>> = RwLock::new(None);

/// `POST /v1/settings/chat-providers`から呼ばれる、単一プロバイダの
/// APIキーの実行時設定。空文字列を渡すとそのプロバイダの設定を消去する。
pub fn set_runtime_key(provider: Provider, api_key: String) {
    let mut guard = RUNTIME_KEYS.write().expect("runtime chat-provider keys lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    if api_key.trim().is_empty() {
        map.remove(&provider);
    } else {
        map.insert(provider, api_key);
    }
}

/// `DELETE /v1/settings/chat-providers`から呼ばれる、全プロバイダの
/// 実行時設定の消去。
pub fn clear_runtime_keys() {
    let mut guard = RUNTIME_KEYS.write().expect("runtime chat-provider keys lock poisoned");
    *guard = None;
}

/// 実行時設定(ブラウザの設定パネル経由)を優先し、無ければ環境変数
/// (起動時設定)にフォールバックする(`web_search::read_credentials`と
/// 同じ二段構え)。
fn read_key(provider: Provider) -> Option<String> {
    if let Some(key) = RUNTIME_KEYS.read().expect("runtime chat-provider keys lock poisoned").as_ref().and_then(|m| m.get(&provider)).cloned() {
        return Some(key);
    }
    let key = std::env::var(provider.env_var_name()).ok()?;
    if key.trim().is_empty() {
        return None;
    }
    Some(key)
}

/// 指定プロバイダのAPIキーが設定済みかどうか(空文字列は未設定として扱う)。
pub fn is_configured(provider: Provider) -> bool {
    read_key(provider).is_some()
}

/// 設定済みの全プロバイダ一覧(設定パネルの状態表示用)。
pub fn configured_providers() -> Vec<Provider> {
    Provider::all().into_iter().filter(|p| is_configured(*p)).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderReply {
    pub provider: Provider,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderFailure {
    pub provider: Provider,
    pub error: String,
    /// このプロバイダの無料枠(レート制限)を使い切ったと判定できたか
    /// (ユーザー指示「Google等のAIの一日の無料枠を使い切ると『本日の
    /// 無料枠は使い切りました』と英語と日本語で表示して」への対応)。
    /// **正直な開示**: HTTP 429(Too Many Requests、4社共通でレート
    /// 制限/無料枠超過時に返す規約上のステータスコード)を根拠に
    /// 判定しており、「一時的なトラフィック過多による429」と「本当に
    /// その日の無料枠を使い切った429」を区別する手段は無い——4社とも
    /// 両者を同じステータスコードで表現するため、これ以上細かい判別は
    /// 技術的にできない。
    pub quota_exceeded: bool,
}

/// HTTPステータス429(Too Many Requests)を無料枠/レート制限超過の
/// サインとして扱う。OpenAI・DeepSeek・Gemini・Claudeいずれも公式
/// ドキュメント上、レート制限・クォータ超過時に429を返す規約のため、
/// 4社共通のヒューリスティックとして採用する。
fn is_quota_exceeded_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

/// 429エラーを`bail!`する際に付けるマーカー接頭辞。呼び出し元
/// (`complete_multi`/`complete_in_priority_order`)がこの接頭辞の有無で
/// `quota_exceeded`を判定し、表示用のエラー文からは接頭辞を取り除く。
const QUOTA_EXCEEDED_MARKER: &str = "QUOTA_EXCEEDED::";

fn split_quota_exceeded(err: &anyhow::Error) -> (bool, String) {
    let full = format!("{err:#}");
    match full.strip_prefix(QUOTA_EXCEEDED_MARKER) {
        Some(rest) => (true, rest.to_string()),
        None => (false, full),
    }
}

/// 単一プロバイダを呼び出す(`web_search::search_with_credentials`と同じ
/// く、呼び出し元が明示的に渡したAPIキーのみを使う版。共有VPS上で
/// 他利用者のグローバル設定を誤って消費しないための設計)。
pub async fn complete_with_key(provider: Provider, api_key: &str, prompt: &str) -> Result<String> {
    if prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build().context("failed to build reqwest client for chat provider request")?;
    match provider {
        Provider::Openai => complete_openai(&client, api_key, prompt).await,
        Provider::Deepseek => complete_deepseek(&client, api_key, prompt).await,
        Provider::Gemini => complete_gemini(&client, api_key, prompt).await,
        Provider::Claude => complete_claude(&client, api_key, prompt).await,
        Provider::Grok => complete_grok(&client, api_key, prompt).await,
        Provider::Groq => {
            let model = std::env::var("ARUARU_LLM_GROQ_MODEL").unwrap_or_else(|_| "openai/gpt-oss-120b".to_string());
            complete_openai_compatible(&client, "Groq", "https://api.groq.com/openai/v1/chat/completions", &model, api_key, prompt).await
        }
        Provider::Cerebras => {
            let model = std::env::var("ARUARU_LLM_CEREBRAS_MODEL").unwrap_or_else(|_| "llama-3.3-70b".to_string());
            complete_openai_compatible(&client, "Cerebras", "https://api.cerebras.ai/v1/chat/completions", &model, api_key, prompt).await
        }
        Provider::OpenRouter => {
            // openrouter/free は、その時点で空いている無料モデルをOpenRouter側が自動選択する
            // (無料モデルの入れ替わりに追従するため既定にしている)。
            let model = std::env::var("ARUARU_LLM_OPENROUTER_MODEL").unwrap_or_else(|_| "openrouter/free".to_string());
            let slow_client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(60)).build().context("failed to build reqwest client for OpenRouter")?;
            complete_openai_compatible(&slow_client, "OpenRouter", "https://openrouter.ai/api/v1/chat/completions", &model, api_key, prompt).await
        }
        Provider::Cloudflare => {
            let (account_id, token) = api_key.split_once(':').context("Cloudflare key must be in the form ACCOUNT_ID:API_TOKEN")?;
            let model = std::env::var("ARUARU_LLM_CLOUDFLARE_MODEL").unwrap_or_else(|_| "@cf/meta/llama-3.3-70b-instruct-fp8-fast".to_string());
            let url = format!("https://api.cloudflare.com/client/v4/accounts/{}/ai/v1/chat/completions", account_id.trim());
            complete_openai_compatible(&client, "Cloudflare", &url, &model, token.trim(), prompt).await
        }
        Provider::Ollama => {
            // api_key引数はOllamaではモデル名。ローカルCPU実行は遅いことがあるため、
            // 専用の長いタイムアウト(120秒)のクライアントを使う。
            let base = std::env::var("ARUARU_LLM_OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
            let url = format!("{}/v1/chat/completions", base.trim_end_matches('/'));
            let slow_client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(120)).build().context("failed to build reqwest client for Ollama")?;
            complete_openai_compatible(&slow_client, "Ollama", &url, api_key, "ollama", prompt).await
        }
        Provider::Mistral => {
            let model = std::env::var("ARUARU_LLM_MISTRAL_MODEL").unwrap_or_else(|_| "ministral-14b-latest".to_string());
            complete_openai_compatible(&client, "Mistral", "https://api.mistral.ai/v1/chat/completions", &model, api_key, prompt).await
        }
    }
}

/// プロセス全体で共有される実行時/環境変数キーを使う版
/// (`POST /v1/chat-providers/complete`のうち、リクエストボディに
/// APIキーが渡されなかった場合のフォールバック経路)。
pub async fn complete(provider: Provider, prompt: &str) -> Result<String> {
    let key = read_key(provider).with_context(|| format!("{} API is not configured (set {} or use the settings panel)", provider.label(), provider.env_var_name()))?;
    complete_with_key(provider, &key, prompt).await
}

/// 複数プロバイダを並列に呼び出し、成功/失敗をプロバイダ別に分けて返す
/// (`tokio::spawn`でプロバイダごとに独立したタスクを起動し、全て
/// `await`することで並列実行する——可変長のため`tokio::join!`は使えず、
/// `futures`クレートへの新規依存を避けるため`JoinHandle`を手動で束ねる)。
pub async fn complete_multi(providers: &[Provider], keys: &HashMap<Provider, String>, prompt: &str) -> (Vec<ProviderReply>, Vec<ProviderFailure>) {
    let handles: Vec<_> = providers
        .iter()
        .copied()
        .map(|provider| {
            let prompt = prompt.to_string();
            let explicit_key = keys.get(&provider).cloned();
            tokio::spawn(async move {
                let result = match explicit_key {
                    Some(key) => complete_with_key(provider, &key, &prompt).await,
                    None => complete(provider, &prompt).await,
                };
                (provider, result)
            })
        })
        .collect();

    let mut replies = Vec::new();
    let mut failures = Vec::new();
    for handle in handles {
        match handle.await {
            Ok((provider, Ok(text))) => replies.push(ProviderReply { provider, text }),
            Ok((provider, Err(err))) => {
                let (quota_exceeded, error) = split_quota_exceeded(&err);
                failures.push(ProviderFailure { provider, error, quota_exceeded });
            }
            Err(join_err) => tracing::warn!("chat provider task panicked: {join_err:#}"),
        }
    }
    (replies, failures)
}

/// 「無料枠を優先で使い切り、順番に使用」機能(ユーザー指示、
/// `provider_priority`モジュール参照)。`provider_priority::current_order()`
/// の順に、設定済み(APIキーあり)のプロバイダを1つずつ試し、**最初に
/// 成功したもの**の結果を返す。失敗したプロバイダは`attempted`へ理由
/// 付きで記録し、どこまで試して何が起きたかを常に正直に開示する
/// (黙って1社だけ試して諦めない、`web_search`と同じ「サービスを
/// 壊さない」設計思想)。
#[derive(Debug, Clone, Serialize)]
pub struct PriorityAttempt {
    pub provider: Provider,
    pub error: String,
    pub quota_exceeded: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PriorityCompleteResult {
    pub reply: Option<ProviderReply>,
    pub attempted: Vec<PriorityAttempt>,
    /// 試行した全プロバイダが無料枠(レート制限)超過で失敗し、かつ
    /// 1件も成功しなかったか(ユーザー指示「Google等のAIの一日の無料枠を
    /// 使い切ると『本日の無料枠は使い切りました』と英語と日本語で表示
    /// して」への対応、フロントエンドがこの一言をそのまま出せるよう
    /// サーバー側で判定して返す)。**正直な開示**: 有料契約(課金設定
    /// 済み)のプロバイダは429を返さずそのまま成功するため、この
    /// フラグは自動的に`false`になる——「有料版も契約していたら
    /// 自動で継続する」という要件は、無料枠切れの判定を待たず単に
    /// 実際のAPI呼び出しが成功する、という既存の仕組みでそのまま
    /// 満たされる(有料/無料を明示的に切り替えるロジックは不要)。
    pub all_quota_exceeded: bool,
}

pub async fn complete_in_priority_order(prompt: &str) -> PriorityCompleteResult {
    complete_in_priority_order_skipping(prompt, &[]).await
}

/// skipに含まれるプロバイダを除いて、優先順に1つずつ試す。
pub async fn complete_in_priority_order_skipping(prompt: &str, skip: &[Provider]) -> PriorityCompleteResult {
    let order = provider_priority::current_order();
    let mut attempted = Vec::new();
    for svc in order {
        let Some(provider) = Provider::from_priority_service(svc) else {
            continue;
        };
        if skip.contains(&provider) || !is_configured(provider) {
            continue;
        }
        match complete(provider, prompt).await {
            Ok(text) => return PriorityCompleteResult { reply: Some(ProviderReply { provider, text }), attempted, all_quota_exceeded: false },
            Err(err) => {
                let (quota_exceeded, error) = split_quota_exceeded(&err);
                attempted.push(PriorityAttempt { provider, error, quota_exceeded });
            }
        }
    }
    let all_quota_exceeded = !attempted.is_empty() && attempted.iter().all(|a| a.quota_exceeded);
    PriorityCompleteResult { reply: None, attempted, all_quota_exceeded }
}

// --- OpenAI (ChatGPT) --------------------------------------------------

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
}

#[derive(Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(Deserialize)]
struct OpenAiChoiceMessage {
    #[serde(default)]
    content: String,
}

async fn complete_openai(client: &reqwest::Client, api_key: &str, prompt: &str) -> Result<String> {
    let body = OpenAiRequest { model: "gpt-3.5-turbo", messages: vec![OpenAiMessage { role: "user", content: prompt }] };
    let res = client.post("https://api.openai.com/v1/chat/completions").bearer_auth(api_key).json(&body).send().await.context("OpenAI request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        let marker = if is_quota_exceeded_status(status) { QUOTA_EXCEEDED_MARKER } else { "" };
        bail!("{marker}OpenAI returned HTTP {status}: {text}");
    }
    let parsed: OpenAiResponse = res.json().await.context("failed to parse OpenAI response")?;
    parsed.choices.into_iter().next().map(|c| c.message.content).context("OpenAI response contained no choices")
}

// --- DeepSeek (OpenAI-compatible API shape) -----------------------------

async fn complete_deepseek(client: &reqwest::Client, api_key: &str, prompt: &str) -> Result<String> {
    // DeepSeekのChat Completions APIはOpenAI互換のリクエスト/レスポンス
    // 形状を採用しているため、同じ構造体をエンドポイントだけ変えて再利用する。
    let body = OpenAiRequest { model: "deepseek-chat", messages: vec![OpenAiMessage { role: "user", content: prompt }] };
    let res = client.post("https://api.deepseek.com/chat/completions").bearer_auth(api_key).json(&body).send().await.context("DeepSeek request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        let marker = if is_quota_exceeded_status(status) { QUOTA_EXCEEDED_MARKER } else { "" };
        bail!("{marker}DeepSeek returned HTTP {status}: {text}");
    }
    let parsed: OpenAiResponse = res.json().await.context("failed to parse DeepSeek response")?;
    parsed.choices.into_iter().next().map(|c| c.message.content).context("DeepSeek response contained no choices")
}

// --- Google Gemini -------------------------------------------------------

#[derive(Serialize)]
struct GeminiRequest<'a> {
    contents: Vec<GeminiContent<'a>>,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    // Vertex AI窓口はroleが必須("user"/"model")。通常のGenerative Language
    // APIでも"user"を明示して問題無い。
    role: &'static str,
    parts: Vec<GeminiPart<'a>>,
}

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiCandidateContent,
}

#[derive(Deserialize)]
struct GeminiCandidateContent {
    #[serde(default)]
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
struct GeminiResponsePart {
    #[serde(default)]
    text: String,
}

async fn complete_gemini(client: &reqwest::Client, api_key: &str, prompt: &str) -> Result<String> {
    let body = GeminiRequest { contents: vec![GeminiContent { role: "user", parts: vec![GeminiPart { text: prompt }] }] };
    // 2026-09-20: Google AI Studioの通常キー(`AIza...`)はGenerative Language
    // API、Vertex AI(express mode)の新形式キー(`AQ.`で始まる)はVertex AIの
    // 窓口へ振り分ける(実機検証: `AQ.`キーは前者だと403
    // API_KEY_SERVICE_BLOCKED、後者だと gemini-2.5-flash が応答した)。
    let url = if api_key.starts_with("AQ.") {
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent"
    } else {
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    };
    // APIキーはクエリ文字列(`?key=...`)ではなくヘッダー(`x-goog-api-key`)
    // で渡す(Google公式が推奨する方式、2026-08-26セキュリティ見直しで
    // 変更——クエリ文字列だとリバースプロキシ・アクセスログ・ブラウザ
    // 履歴等にキーが平文で残りやすいため、ヘッダーの方が誤って
    // ログへ残るリスクが低い)。
    let res = client.post(url).header("x-goog-api-key", api_key).json(&body).send().await.context("Gemini request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        let marker = if is_quota_exceeded_status(status) { QUOTA_EXCEEDED_MARKER } else { "" };
        bail!("{marker}Gemini returned HTTP {status}: {text}");
    }
    let parsed: GeminiResponse = res.json().await.context("failed to parse Gemini response")?;
    parsed
        .candidates
        .into_iter()
        .next()
        .and_then(|c| c.content.parts.into_iter().next())
        .map(|p| p.text)
        .context("Gemini response contained no candidates")
}

// --- Anthropic Claude ------------------------------------------------------

#[derive(Serialize)]
struct ClaudeRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<OpenAiMessage<'a>>,
}

#[derive(Deserialize)]
struct ClaudeResponse {
    #[serde(default)]
    content: Vec<ClaudeContentBlock>,
}

#[derive(Deserialize)]
struct ClaudeContentBlock {
    #[serde(default)]
    text: String,
}

async fn complete_claude(client: &reqwest::Client, api_key: &str, prompt: &str) -> Result<String> {
    let body = ClaudeRequest { model: "claude-3-5-haiku-latest", max_tokens: 1024, messages: vec![OpenAiMessage { role: "user", content: prompt }] };
    let res = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .context("Claude request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        let marker = if is_quota_exceeded_status(status) { QUOTA_EXCEEDED_MARKER } else { "" };
        bail!("{marker}Claude returned HTTP {status}: {text}");
    }
    let parsed: ClaudeResponse = res.json().await.context("failed to parse Claude response")?;
    parsed.content.into_iter().next().map(|b| b.text).context("Claude response contained no content blocks")
}

// --- Grok (xAI, OpenAI-compatible API shape) ------------------------------

/// OpenAI互換のChat Completions API(Groq/Cerebras/Mistral共通、2026-09-21新設)。
/// モデル名は各社の提供状況が変わりやすいため、環境変数
/// (ARUARU_LLM_GROQ_MODEL等)で差し替えられるようにしてある。
async fn complete_openai_compatible(client: &reqwest::Client, name: &str, url: &str, model: &str, api_key: &str, prompt: &str) -> Result<String> {
    let body = OpenAiRequest { model, messages: vec![OpenAiMessage { role: "user", content: prompt }] };
    let res = client.post(url).bearer_auth(api_key).json(&body).send().await.with_context(|| format!("{name} request failed"))?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        let marker = if is_quota_exceeded_status(status) { QUOTA_EXCEEDED_MARKER } else { "" };
        bail!("{marker}{name} returned HTTP {status}: {text}");
    }
    let parsed: OpenAiResponse = res.json().await.with_context(|| format!("failed to parse {name} response"))?;
    parsed.choices.into_iter().next().map(|c| c.message.content).with_context(|| format!("{name} response contained no choices"))
}

async fn complete_grok(client: &reqwest::Client, api_key: &str, prompt: &str) -> Result<String> {
    // xAIのChat Completions APIはOpenAI互換のリクエスト/レスポンス形状を
    // 採用しているため、DeepSeekと同様に既存の`OpenAiRequest`/
    // `OpenAiResponse`をそのままエンドポイントだけ変えて再利用する。
    let body = OpenAiRequest { model: "grok-3-mini", messages: vec![OpenAiMessage { role: "user", content: prompt }] };
    let res = client.post("https://api.x.ai/v1/chat/completions").bearer_auth(api_key).json(&body).send().await.context("Grok request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        let marker = if is_quota_exceeded_status(status) { QUOTA_EXCEEDED_MARKER } else { "" };
        bail!("{marker}Grok returned HTTP {status}: {text}");
    }
    let parsed: OpenAiResponse = res.json().await.context("failed to parse Grok response")?;
    parsed.choices.into_iter().next().map(|c| c.message.content).context("Grok response contained no choices")
}

// --- ハイブリッド(良い所どり、2026-09-21新設) --------------------------
//
// ユーザー指示「Gemini2.5FlashとGroq(Llama 3.3 70B)とGrokを一緒に使用して(のちに
// Mistralも加えた4つに同時に質問して、2つ以上から回答があれば統合、1つだけならそのまま)、
// ハイブリッド使用で、良い所どりのAIの回答を目指して」。ハイブリッド群
// (Gemini・Groq・Grok・Mistral)のうち**設定済みのもの全てへ同時に問い合わせ**、
// 2つ以上から回答が得られたら、その回答群を1つのAIに渡して「正しい点を
// 組み合わせ、誤りや矛盾を捨てた最良の1回答」へ統合させる。1つしか得られ
// なければそのまま返し、全て失敗したら優先順チェーンの残り(Cerebras→
// ChatGPT→DeepSeek→Claude)へ進む。
//
// **正直な開示**: 統合は「複数AIの回答をもう1回AIに読ませてまとめる」方式で、
// 正しさを保証する仕組みではない(統合役も間違えうる)。1質問につき最大で
// 群の数(最大4)+1回のAPI呼び出しを消費するため、無料枠の減りは速くなる。
// ARUARU_LLM_HYBRID=off で無効化でき、その場合は従来の順次フォールバックのみ。
pub const HYBRID_GROUP: [Provider; 6] =
    [Provider::Gemini, Provider::Groq, Provider::Grok, Provider::Mistral, Provider::OpenRouter, Provider::Cloudflare];

#[derive(Debug, Clone, Serialize)]
pub struct HybridCompleteResult {
    pub reply: Option<ProviderReply>,
    /// 回答を出したハイブリッド群のプロバイダ(1つだけなら統合なし)。
    pub hybrid_providers: Vec<Provider>,
    /// 複数の回答を統合して作った回答か。
    pub synthesized: bool,
    pub attempted: Vec<PriorityAttempt>,
    pub all_quota_exceeded: bool,
}

pub fn hybrid_enabled() -> bool {
    !matches!(std::env::var("ARUARU_LLM_HYBRID").ok().as_deref(), Some("off") | Some("0") | Some("false"))
}

fn build_synthesis_prompt(question: &str, candidates: &[ProviderReply]) -> String {
    let mut s = String::from(
        "You are given a user's question and several candidate answers written by different AI assistants. \
Write the single best final answer: keep the correct and useful points from all candidates, drop anything \
wrong, unsupported or contradictory, and do not mention the candidates or that you merged them. \
Answer in the same language as the user's question, naturally and clearly, like a knowledgeable and friendly human.\n\n",
    );
    s.push_str("### User question\n");
    s.push_str(question);
    for (i, c) in candidates.iter().enumerate() {
        s.push_str(&format!("\n\n### Candidate answer {} ({})\n{}", i + 1, c.provider.label(), c.text));
    }
    s.push_str("\n\n### Final answer\n");
    s
}

// --- 予備への自動交代(2026-09-21、ユーザー指示「最初は二社でハイブリッド運用して、
// 無料枠が切れたり、無料期間が終了したものから、次のAIの使用に切り替えて、予備のAIが
// 欲しかった」) --------------------------------------------------------------------
//
// 同時に使うのは、優先順の上位ARUARU_LLM_HYBRID_SIZE社(既定2)。使えなくなった
// (枠切れ・無料期間終了・認証エラー等)AIは一定時間お休みさせ、その回のうちに
// 予備の次のAIへ交代する。お休み中のAIは以後の質問で最初から飛ばす(毎回失敗する
// 呼び出しで時間と枠を無駄にしない)。お休みが明けたら自動で復帰を試す。
// **正直な開示**: お休み状態はプロセスのメモリ上のみ(再起動で消える)。
static HYBRID_COOLDOWN: std::sync::Mutex<Option<HashMap<Provider, std::time::Instant>>> = std::sync::Mutex::new(None);

fn hybrid_size() -> usize {
    std::env::var("ARUARU_LLM_HYBRID_SIZE").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n >= 1).unwrap_or(2)
}

fn in_cooldown(provider: Provider) -> bool {
    let guard = HYBRID_COOLDOWN.lock().expect("hybrid cooldown lock poisoned");
    guard.as_ref().and_then(|m| m.get(&provider)).map_or(false, |until| std::time::Instant::now() < *until)
}

/// 失敗の種類に応じてお休みさせる: 枠切れ(429等)=24時間、認証/権限/モデル廃止
/// (401/403/404)=12時間(無料期間終了・キー失効・モデル廃止など)、それ以外(タイムアウト・
/// 5xx等の一時的な失敗)=2分。
///
/// 2026-09-22変更(ユーザー指示「一日の無料枠を超えて次のAIに移っても、一日経ったら、
/// その使用制限がリセットされたら自動でGeminiなどを自動で再度使用可能に」): 枠切れの
/// お休み時間を6時間→24時間に変更した。**正直な開示**: 各社の無料枠は元々「使えなく
/// なった後、お休み時間(この秒数)が経過すれば自動的に候補へ復帰する」仕組みが既に
/// あり(`in_cooldown`/`hybrid_candidates`参照、このコミット以前から動作済み)、今回の
/// 変更は「その時間を1日に近づける」調整のみである。ベンダー各社の実際のリセット時刻
/// (UTC深夜0時、リクエスト時刻から24時間後、等)はベンダーごとに異なり、かつ大半は
/// 公式ドキュメントで明示されていないため、正確なリセット時刻に合わせることはできない
/// ——「24時間経ったら再度試す」という近似にとどめている。もしリセット前に再度枠切れに
/// なった場合は、そこからまた24時間のお休みに入り、実際にリセットされるまで自動で
/// 再試行を繰り返す(既存の仕組みのまま)。
fn mark_cooldown(provider: Provider, quota_exceeded: bool, error: &str) {
    let secs = if quota_exceeded {
        24 * 3600
    } else if error.contains("HTTP 401") || error.contains("HTTP 403") || error.contains("HTTP 404") || error.contains("HTTP 402") {
        12 * 3600
    } else {
        120
    };
    let mut guard = HYBRID_COOLDOWN.lock().expect("hybrid cooldown lock poisoned");
    guard.get_or_insert_with(HashMap::new).insert(provider, std::time::Instant::now() + std::time::Duration::from_secs(secs));
}

/// ハイブリッド候補(優先順に並べた、設定済みで、お休み中でないAI)。
fn hybrid_candidates() -> Vec<Provider> {
    let mut out = Vec::new();
    for svc in provider_priority::current_order() {
        if let Some(p) = Provider::from_priority_service(svc) {
            if HYBRID_GROUP.contains(&p) && is_configured(p) && !in_cooldown(p) && !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// いまのハイブリッド運用状態(画面の「使用中の無料AI」表示用、2026-09-21新設)。
/// active=同時に使う上位N社、standby=予備(繰り上げ待ち)、resting=お休み中(枠切れ等)。
#[derive(Debug, Clone, Serialize)]
pub struct HybridStatus {
    pub hybrid_enabled: bool,
    pub size: usize,
    pub active: Vec<Provider>,
    pub standby: Vec<Provider>,
    pub resting: Vec<Provider>,
    /// 利用者が選べる(キー設定済みの)無料AI全部。
    pub available: Vec<Provider>,
}

pub fn hybrid_status() -> HybridStatus {
    let enabled = hybrid_enabled();
    let size = hybrid_size();
    let candidates = if enabled { hybrid_candidates() } else { Vec::new() };
    let active: Vec<Provider> = candidates.iter().copied().take(size).collect();
    let standby: Vec<Provider> = candidates.iter().copied().skip(size).collect();
    let resting: Vec<Provider> = HYBRID_GROUP.iter().copied().filter(|p| is_configured(*p) && in_cooldown(*p)).collect();
    let available: Vec<Provider> = HYBRID_GROUP.iter().copied().filter(|p| is_configured(*p)).collect();
    HybridStatus { hybrid_enabled: enabled, size, active, standby, resting, available }
}

/// 利用者が選べる同時利用AI数の上限(1=単独、2=ハイブリッド、3=トライブリッド)。
pub const MAX_SELECTED_PROVIDERS: usize = 3;

/// 無料枠のハイブリッド群から、利用者が選んだAI(重複除去・最大3個)を取り出す。
pub fn parse_selected_providers(names: &[String]) -> Vec<Provider> {
    let mut out: Vec<Provider> = Vec::new();
    for n in names {
        let lower = n.trim().to_lowercase();
        if let Some(p) = HYBRID_GROUP.iter().copied().find(|p| format!("{p:?}").to_lowercase() == lower) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
        if out.len() >= MAX_SELECTED_PROVIDERS {
            break;
        }
    }
    out
}

pub async fn complete_hybrid(prompt: &str) -> HybridCompleteResult {
    complete_hybrid_with(prompt, &[]).await
}

/// `selected`が空なら従来どおり(優先順の上位N社)。指定があれば、そのAI(1〜3個)を
/// 同時に使い、使えないものが出たら残りのハイブリッド群から予備を繰り上げる。
pub async fn complete_hybrid_with(prompt: &str, selected: &[Provider]) -> HybridCompleteResult {
    let mut candidates: Vec<Provider> = if hybrid_enabled() { hybrid_candidates() } else { Vec::new() };
    let mut size = hybrid_size();
    if !selected.is_empty() && hybrid_enabled() {
        let mut ordered: Vec<Provider> = selected.iter().copied().filter(|p| is_configured(*p) && !in_cooldown(*p)).collect();
        if !ordered.is_empty() {
            size = selected.len().min(MAX_SELECTED_PROVIDERS);
            for p in candidates.iter().copied() {
                if !ordered.contains(&p) {
                    ordered.push(p);
                }
            }
            candidates = ordered;
        }
    }
    if candidates.is_empty() {
        let r = complete_in_priority_order(prompt).await;
        return HybridCompleteResult { reply: r.reply, hybrid_providers: Vec::new(), synthesized: false, attempted: r.attempted, all_quota_exceeded: r.all_quota_exceeded };
    }
    let mut replies: Vec<ProviderReply> = Vec::new();
    let mut attempted: Vec<PriorityAttempt> = Vec::new();
    let mut tried: Vec<Provider> = Vec::new();
    // 必要な社数(size)が揃うまで、予備を1ラウンドずつ繰り上げて並列に呼ぶ。
    while replies.len() < size && !candidates.is_empty() {
        let need = size - replies.len();
        let batch: Vec<Provider> = candidates.drain(..need.min(candidates.len())).collect();
        tried.extend(batch.iter().copied());
        let (ok, failures) = complete_multi(&batch, &HashMap::new(), prompt).await;
        replies.extend(ok);
        for f in failures {
            mark_cooldown(f.provider, f.quota_exceeded, &f.error);
            attempted.push(PriorityAttempt { provider: f.provider, error: f.error, quota_exceeded: f.quota_exceeded });
        }
    }
    match replies.len() {
        0 => {
            // ハイブリッド群が全滅 → 残り(Ollama・有料等)を優先順に(群は再試行しない)
            let r = complete_in_priority_order_skipping(prompt, &HYBRID_GROUP).await;
            attempted.extend(r.attempted);
            let all_quota_exceeded = r.reply.is_none() && !attempted.is_empty() && attempted.iter().all(|a| a.quota_exceeded);
            HybridCompleteResult { reply: r.reply, hybrid_providers: Vec::new(), synthesized: false, attempted, all_quota_exceeded }
        }
        1 => {
            let only = replies.into_iter().next().expect("one reply");
            HybridCompleteResult { hybrid_providers: vec![only.provider], reply: Some(only), synthesized: false, attempted, all_quota_exceeded: false }
        }
        _ => {
            let providers: Vec<Provider> = replies.iter().map(|r| r.provider).collect();
            let synth_prompt = build_synthesis_prompt(prompt, &replies);
            // 統合役: 回答を出せたAIの先頭から順に。失敗したら次、全部失敗したら先頭の生の回答。
            for synthesizer in &providers {
                if let Ok(text) = complete(*synthesizer, &synth_prompt).await {
                    if !text.trim().is_empty() {
                        return HybridCompleteResult {
                            reply: Some(ProviderReply { provider: *synthesizer, text }),
                            hybrid_providers: providers,
                            synthesized: true,
                            attempted,
                            all_quota_exceeded: false,
                        };
                    }
                }
            }
            let first = replies.into_iter().next().expect("replies non-empty");
            HybridCompleteResult { reply: Some(first), hybrid_providers: providers, synthesized: false, attempted, all_quota_exceeded: false }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_configured_false_when_env_and_runtime_absent() {
        clear_runtime_keys();
        let saved = std::env::var("ARUARU_LLM_OPENAI_API_KEY").ok();
        std::env::remove_var("ARUARU_LLM_OPENAI_API_KEY");

        assert!(!is_configured(Provider::Openai));

        if let Some(v) = saved {
            std::env::set_var("ARUARU_LLM_OPENAI_API_KEY", v);
        }
    }

    #[test]
    fn set_and_clear_runtime_key_round_trips() {
        clear_runtime_keys();
        assert!(!is_configured(Provider::Claude));
        set_runtime_key(Provider::Claude, "sk-test".to_string());
        assert!(is_configured(Provider::Claude));
        set_runtime_key(Provider::Claude, String::new());
        assert!(!is_configured(Provider::Claude));
    }

    #[test]
    fn configured_providers_reflects_runtime_keys() {
        clear_runtime_keys();
        set_runtime_key(Provider::Gemini, "test-key".to_string());
        let configured = configured_providers();
        assert!(configured.contains(&Provider::Gemini));
        clear_runtime_keys();
    }

    #[tokio::test]
    async fn complete_with_key_rejects_empty_prompt() {
        let err = complete_with_key(Provider::Openai, "sk-test", "   ").await.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn split_quota_exceeded_detects_marker_and_strips_it() {
        let err = anyhow::anyhow!("{QUOTA_EXCEEDED_MARKER}OpenAI returned HTTP 429: rate limited");
        let (quota_exceeded, message) = split_quota_exceeded(&err);
        assert!(quota_exceeded);
        assert_eq!(message, "OpenAI returned HTTP 429: rate limited");
    }

    #[test]
    fn split_quota_exceeded_false_for_ordinary_errors() {
        let err = anyhow::anyhow!("Claude returned HTTP 401 Unauthorized: invalid key");
        let (quota_exceeded, message) = split_quota_exceeded(&err);
        assert!(!quota_exceeded);
        assert_eq!(message, "Claude returned HTTP 401 Unauthorized: invalid key");
    }

    #[test]
    fn is_quota_exceeded_status_matches_only_429() {
        assert!(is_quota_exceeded_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_quota_exceeded_status(reqwest::StatusCode::UNAUTHORIZED));
        assert!(!is_quota_exceeded_status(reqwest::StatusCode::OK));
    }

    /// 2026-09-22追加(ユーザー指示「一日経ったら、その使用制限がリセットされたら自動で
    /// Geminiなどを自動で再度使用可能に」): 枠切れ直後は約24時間お休みすることを確認する。
    #[test]
    fn quota_exceeded_cooldown_is_about_one_day() {
        {
            let mut guard = HYBRID_COOLDOWN.lock().expect("lock");
            guard.get_or_insert_with(HashMap::new).remove(&Provider::Gemini);
        }
        mark_cooldown(Provider::Gemini, true, "HTTP 429: rate limited");
        let guard = HYBRID_COOLDOWN.lock().expect("lock");
        let until = *guard.as_ref().unwrap().get(&Provider::Gemini).expect("cooldown set");
        let remaining = until.saturating_duration_since(std::time::Instant::now());
        // 24時間ちょうどは境界のタイミング差でずれうるので、23〜24時間の範囲で確認する。
        assert!(remaining.as_secs() > 23 * 3600 && remaining.as_secs() <= 24 * 3600, "remaining={remaining:?}");
    }

    /// 枠切れでお休みしたAIも、お休み時間(実質的な「1日経過」)を過ぎれば、次回の候補
    /// 選定(`hybrid_candidates`)へ自動的に復帰することを確認する(＝ユーザーが何も
    /// しなくても、リセット後は自動的にGeminiなどが再度使われる、という仕組みそのものの検証)。
    #[test]
    fn provider_auto_recovers_once_cooldown_instant_has_passed() {
        clear_runtime_keys();
        set_runtime_key(Provider::Gemini, "test-key".to_string());
        // 枠切れ直後: お休み中なので候補から外れている。
        mark_cooldown(Provider::Gemini, true, "HTTP 429: rate limited");
        assert!(in_cooldown(Provider::Gemini), "should be resting right after quota exceeded");
        assert!(!hybrid_candidates().contains(&Provider::Gemini));
        // お休み時間が「過去の時刻」まで進んだ状態(=無料枠がリセットされた後)を模擬する。
        {
            let mut guard = HYBRID_COOLDOWN.lock().expect("lock");
            guard.get_or_insert_with(HashMap::new).insert(Provider::Gemini, std::time::Instant::now() - std::time::Duration::from_secs(1));
        }
        assert!(!in_cooldown(Provider::Gemini), "should have recovered automatically once the cooldown time has passed");
        assert!(hybrid_candidates().contains(&Provider::Gemini), "must be usable again automatically, without any manual action");
        clear_runtime_keys();
        let mut guard = HYBRID_COOLDOWN.lock().expect("lock");
        guard.get_or_insert_with(HashMap::new).remove(&Provider::Gemini);
    }
}
