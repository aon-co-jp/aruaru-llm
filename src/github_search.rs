//! GitHub検索連携(ユーザー指示「GitHub連携も、チェックを付けられる機能と
//! 実際に連携する機能を付けて」への対応)。`web_search.rs`(Google Custom
//! Search)と全く同じ設計パターン——チェックボックスで有効化し、実際に
//! GitHub REST APIのリポジトリ検索エンドポイントへ本物のHTTPリクエストを
//! 送り、上位結果をチャット応答のプロンプトへコンテキストとして埋め込む。
//!
//! ## 正直な開示(最重要)
//!
//! - GitHubのリポジトリ検索API(`GET /search/repositories`)は**認証不要
//!   でも呼べる**が、未認証だと10リクエスト/分というかなり厳しい
//!   レート制限になる(GitHub公式ドキュメント記載)。個人アクセス
//!   トークン(Personal Access Token、`public_repo`スコープすら不要、
//!   単に認証済み扱いになるだけで十分)を設定すれば30リクエスト/分へ
//!   緩和される——トークンは`web_search.rs`と同じくプロセスメモリ上に
//!   のみ保持し、ディスクへの永続化は一切行わない。トークン無しでも
//!   機能自体は動作する(このリポジトリはGoogle検索と異なり、GitHub側の
//!   利用には契約・APIキー取得が必須ではない、ただし高頻度利用には
//!   トークン設定を推奨する旨をUIに明記すること)。
//! - 検索対象は**リポジトリのメタデータ(名前・説明・URL・スター数)
//!   のみ**——コード検索(`/search/code`)やファイル内容の取得は
//!   行わない(スコープを絞り、著作権上の配慮とAPI呼び出し回数の抑制を
//!   両立させるため)。
//! - 検索結果の説明文はGitHub上の公開リポジトリの著作物であり、本文を
//!   丸ごと転載せず、上位数件の名前・説明・URL・スター数のみをプロンプト
//!   へ埋め込む(`web_search.rs`と同じ引用範囲の配慮)。

use std::sync::RwLock;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const GITHUB_SEARCH_ENDPOINT: &str = "https://api.github.com/search/repositories";

/// 利用者がブラウザの設定パネルから入力したPersonal Access Tokenを、
/// 実行中のプロセスのメモリ上にのみ保持する(`web_search::RUNTIME_
/// CREDENTIALS`と同じ設計)。空文字列/未設定でも検索機能自体は動作する
/// (レート制限が10/分のまま、というだけ)。
static RUNTIME_TOKEN: RwLock<Option<String>> = RwLock::new(None);

pub fn set_runtime_token(token: String) {
    let mut guard = RUNTIME_TOKEN.write().expect("runtime github token lock poisoned");
    *guard = if token.trim().is_empty() { None } else { Some(token) };
}

pub fn clear_runtime_token() {
    set_runtime_token(String::new());
}

/// トークンが設定されているか(未設定でも検索自体は可能なため、
/// `web_search::is_configured`とは意味が異なる——こちらは「高いレート
/// 制限で使えるか」を示す)。
pub fn is_token_configured() -> bool {
    read_token().is_some()
}

fn read_token() -> Option<String> {
    if let Some(token) = RUNTIME_TOKEN.read().expect("runtime github token lock poisoned").clone() {
        return Some(token);
    }
    let token = std::env::var("ARUARU_LLM_GITHUB_TOKEN").ok()?;
    if token.trim().is_empty() {
        return None;
    }
    Some(token)
}

#[derive(Debug, Clone, Serialize)]
pub struct GithubResult {
    pub full_name: String,
    pub description: String,
    pub url: String,
    pub stars: u64,
}

#[derive(Debug, Deserialize)]
struct GithubSearchResponse {
    #[serde(default)]
    items: Vec<GithubItem>,
}

#[derive(Debug, Deserialize)]
struct GithubItem {
    #[serde(default)]
    full_name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    stargazers_count: u64,
}

/// GitHubリポジトリ検索を実行する(トークン設定済みなら優先利用、
/// 未設定でも動作する——`web_search::search`と異なり、未設定を
/// エラーにはしない)。
pub async fn search(query: &str, max_results: u8) -> Result<Vec<GithubResult>> {
    search_with_optional_token(query, max_results, read_token().as_deref()).await
}

/// `search()`と同じ検索処理だが、呼び出し元が明示的に渡したトークンのみを
/// 使う版(`web_search::search_with_credentials`と同じ設計、共有VPS上で
/// 他利用者のグローバル設定を誤って消費しないため)。トークンが`None`
/// でも未認証のまま検索を実行する(未認証は許可されている、レート制限が
/// 厳しくなるだけ)。
pub async fn search_with_optional_token(query: &str, max_results: u8, token: Option<&str>) -> Result<Vec<GithubResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }

    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(8)).build().context("failed to build reqwest client for GitHub search")?;

    let mut req = client
        .get(GITHUB_SEARCH_ENDPOINT)
        .header("User-Agent", "aruaru-llm-github-search")
        .header("Accept", "application/vnd.github+json")
        .query(&[("q", query), ("sort", "stars"), ("order", "desc"), ("per_page", &max_results.clamp(1, 20).to_string())]);
    if let Some(token) = token {
        if !token.trim().is_empty() {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
    }

    let res = req.send().await.context("GitHub search request failed")?;

    if !res.status().is_success() {
        let status = res.status();
        // GitHubのエラー本文には具体的な理由(レート制限超過・クエリ構文
        // エラー等)が入っており診断に有用——トークンの値自体が
        // エコーバックされることは無い(GitHub側の仕様、確認済み)ため
        // そのままエラーメッセージへ含めても安全。
        let body = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        bail!("GitHub search returned HTTP {status}: {body}");
    }

    let body: GithubSearchResponse = res.json().await.context("failed to parse GitHub search response")?;
    Ok(body
        .items
        .into_iter()
        .map(|item| GithubResult { full_name: item.full_name, description: item.description.unwrap_or_default(), url: item.html_url, stars: item.stargazers_count })
        .collect())
}

/// 検索結果を、生成プロンプトへ埋め込むための短いコンテキスト文字列へ
/// 整形する(名前+説明+スター数のみ、本文丸ごと転載はしない)。
pub fn format_results_as_context(results: &[GithubResult]) -> String {
    results
        .iter()
        .map(|r| format!("- {} ({}★): {}", r.full_name, r.stars, r.description))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_token_configured_false_when_env_and_runtime_absent() {
        clear_runtime_token();
        let saved = std::env::var("ARUARU_LLM_GITHUB_TOKEN").ok();
        std::env::remove_var("ARUARU_LLM_GITHUB_TOKEN");

        assert!(!is_token_configured());

        if let Some(v) = saved {
            std::env::set_var("ARUARU_LLM_GITHUB_TOKEN", v);
        }
    }

    #[test]
    fn set_and_clear_runtime_token_round_trips() {
        clear_runtime_token();
        assert!(!is_token_configured());
        set_runtime_token("ghp_test".to_string());
        assert!(is_token_configured());
        clear_runtime_token();
        assert!(!is_token_configured());
    }

    #[test]
    fn format_results_as_context_joins_name_stars_and_description() {
        let results = vec![
            GithubResult { full_name: "rust-lang/rust".to_string(), description: "The Rust language".to_string(), url: "https://github.com/rust-lang/rust".to_string(), stars: 90000 },
        ];
        let ctx = format_results_as_context(&results);
        assert_eq!(ctx, "- rust-lang/rust (90000★): The Rust language");
    }

    #[tokio::test]
    async fn search_with_optional_token_rejects_empty_query() {
        let err = search_with_optional_token("  ", 3, None).await.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }
}
