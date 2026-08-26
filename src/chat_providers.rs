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
}

impl Provider {
    fn env_var_name(self) -> &'static str {
        match self {
            Provider::Openai => "ARUARU_LLM_OPENAI_API_KEY",
            Provider::Deepseek => "ARUARU_LLM_DEEPSEEK_API_KEY",
            Provider::Gemini => "ARUARU_LLM_GEMINI_API_KEY",
            Provider::Claude => "ARUARU_LLM_ANTHROPIC_API_KEY",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Provider::Openai => "openai",
            Provider::Deepseek => "deepseek",
            Provider::Gemini => "gemini",
            Provider::Claude => "claude",
        }
    }

    fn all() -> [Provider; 4] {
        [Provider::Openai, Provider::Deepseek, Provider::Gemini, Provider::Claude]
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
    let order = provider_priority::current_order();
    let mut attempted = Vec::new();
    for svc in order {
        let Some(provider) = Provider::from_priority_service(svc) else {
            continue;
        };
        if !is_configured(provider) {
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
    let body = GeminiRequest { contents: vec![GeminiContent { parts: vec![GeminiPart { text: prompt }] }] };
    let url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";
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
}
