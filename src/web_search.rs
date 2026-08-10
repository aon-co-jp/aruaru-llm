//! Google Custom Search JSON APIによるWeb検索連携(ユーザー指示
//! 「open-englishは、人がしゃべったり文字を入力したら、その都度Google
//! 検索するような仕様にして」への対応、ブリッジ式——入力した英文の
//! 分からない単語/知識をGoogle検索し、その結果でAI応答を補強する)。
//!
//! ## 正直な開示(最重要)
//!
//! - Google Custom Search JSON APIの利用には、ユーザー自身が
//!   [Google Cloud Console](https://console.cloud.google.com/)で
//!   APIキーを、[Programmable Search Engine](https://programmablesearchengine.google.com/)
//!   で検索エンジンID(`cx`)を取得する必要がある(無料枠は1日100件まで、
//!   それ以降は有料)。このリポジトリはAPIキーを一切保持・同梱しない
//!   ——`ARUARU_LLM_GOOGLE_SEARCH_API_KEY`/`ARUARU_LLM_GOOGLE_SEARCH_CX`
//!   環境変数を起動時に設定するのはユーザー自身の責任(このエコシステム
//!   共通の「契約不要の独自AI」という設計思想〈`aruaru-llm`本体のGPT-2
//!   推論自体は外部契約不要〉に対し、この検索機能だけは意図的な例外
//!   ——ユーザー自身が明示的に選択したGoogle Custom Search API利用の
//!   結果であり、隠さず明記する)。
//! - 環境変数が未設定の場合、この機能は静かに無効化され(`is_configured()`
//!   が`false`を返す)、`/v1/generate-with-search`は検索無しの通常の
//!   `/v1/generate`相当の挙動へフォールバックする(サービス全体を壊さない
//!   設計、既存のGPT-2重み未取得時の503フォールバックと同じ思想)。
//! - 検索結果の要約(スニペット)はGoogle側の著作物であり、本文を丸ごと
//!   転載せず、上位数件のタイトル・スニペット・URLのみをプロンプトへ
//!   埋め込む(引用の範囲、既存の秋葉原メイドカフェ記事引用と同じ配慮)。

use std::sync::RwLock;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const GOOGLE_CSE_ENDPOINT: &str = "https://www.googleapis.com/customsearch/v1";

/// 利用者がブラウザの設定パネルから入力したAPIキー/cxを、実行中の
/// プロセスのメモリ上にのみ保持する(ユーザー指示「利用者がAPIキーの
/// 取得とCOPYペーストが簡単な機能を搭載して」への対応)。
///
/// **正直な開示・セキュリティ配慮**: ディスクへの書き込み・ログ出力は
/// 一切行わない。プロセス終了(サーバー再起動)で消える——永続化しない
/// 設計にすることで、誤ってGitリポジトリやログファイルへ紛れ込む
/// リスクを避けている。`GET /v1/settings/google-search`はキーの値自体を
/// 一切返さず、設定済みかどうかの真偽値のみを返す。
static RUNTIME_CREDENTIALS: RwLock<Option<(String, String)>> = RwLock::new(None);

/// ブラウザの設定パネルから送られたAPIキー/cxを実行時に設定する
/// (`POST /v1/settings/google-search`のハンドラから呼ばれる)。
pub fn set_runtime_credentials(api_key: String, cx: String) {
    let mut guard = RUNTIME_CREDENTIALS.write().expect("runtime credentials lock poisoned");
    if api_key.trim().is_empty() || cx.trim().is_empty() {
        *guard = None;
    } else {
        *guard = Some((api_key, cx));
    }
}

/// 実行時設定を消去する(`DELETE /v1/settings/google-search`)。
pub fn clear_runtime_credentials() {
    set_runtime_credentials(String::new(), String::new());
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub snippet: String,
    pub link: String,
}

#[derive(Debug, Deserialize)]
struct CseResponse {
    #[serde(default)]
    items: Vec<CseItem>,
}

#[derive(Debug, Deserialize)]
struct CseItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    link: String,
}

/// 環境変数`ARUARU_LLM_GOOGLE_SEARCH_API_KEY`/`ARUARU_LLM_GOOGLE_SEARCH_CX`
/// の両方が設定されているかどうか(空文字列は未設定として扱う)。
pub fn is_configured() -> bool {
    read_credentials().is_some()
}

/// 実行時設定(ブラウザの設定パネル経由)を優先し、無ければ環境変数
/// (起動時設定)にフォールバックする。
fn read_credentials() -> Option<(String, String)> {
    if let Some(creds) = RUNTIME_CREDENTIALS.read().expect("runtime credentials lock poisoned").clone() {
        return Some(creds);
    }
    let api_key = std::env::var("ARUARU_LLM_GOOGLE_SEARCH_API_KEY").ok()?;
    let cx = std::env::var("ARUARU_LLM_GOOGLE_SEARCH_CX").ok()?;
    if api_key.trim().is_empty() || cx.trim().is_empty() {
        return None;
    }
    Some((api_key, cx))
}

/// Google Custom Search JSON APIで検索を実行し、上位`max_results`件
/// (既定呼び出し側は3件程度を想定)を返す。認証情報未設定時は
/// 正直にエラーを返す(黙って空配列を返さない——呼び出し側が
/// `is_configured()`で事前に分岐する設計を前提とする)。
pub async fn search(query: &str, max_results: u8) -> Result<Vec<SearchResult>> {
    let (api_key, cx) = read_credentials().context("Google Custom Search API is not configured (set ARUARU_LLM_GOOGLE_SEARCH_API_KEY and ARUARU_LLM_GOOGLE_SEARCH_CX)")?;
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for Google Custom Search")?;

    let res = client
        .get(GOOGLE_CSE_ENDPOINT)
        .query(&[
            ("key", api_key.as_str()),
            ("cx", cx.as_str()),
            ("q", query),
            ("num", &max_results.clamp(1, 10).to_string()),
        ])
        .send()
        .await
        .context("Google Custom Search request failed")?;

    if !res.status().is_success() {
        let status = res.status();
        // Googleのエラーレスポンス本文には具体的な理由(例: "API key not
        // valid"・"Invalid Value"等)が入っており、ステータスコードだけ
        // より診断に有用——正直な開示: この本文にAPIキーの値自体が
        // エコーバックされることは無い(Google側の仕様、確認済み)ため、
        // そのままエラーメッセージへ含めても安全。
        let body = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        bail!("Google Custom Search returned HTTP {status}: {body}");
    }

    let body: CseResponse = res.json().await.context("failed to parse Google Custom Search response")?;
    Ok(body
        .items
        .into_iter()
        .map(|item| SearchResult { title: item.title, snippet: item.snippet, link: item.link })
        .collect())
}

/// 検索結果を、生成プロンプトへ埋め込むための短いコンテキスト文字列へ
/// 整形する(タイトル+スニペットのみ、本文丸ごと転載はしない)。
pub fn format_results_as_context(results: &[SearchResult]) -> String {
    results
        .iter()
        .map(|r| format!("- {}: {}", r.title, r.snippet))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_configured_false_when_env_vars_absent() {
        // 実行環境の環境変数を汚さないよう、既存の値を保存・復元する。
        let saved_key = std::env::var("ARUARU_LLM_GOOGLE_SEARCH_API_KEY").ok();
        let saved_cx = std::env::var("ARUARU_LLM_GOOGLE_SEARCH_CX").ok();
        std::env::remove_var("ARUARU_LLM_GOOGLE_SEARCH_API_KEY");
        std::env::remove_var("ARUARU_LLM_GOOGLE_SEARCH_CX");

        assert!(!is_configured());

        if let Some(v) = saved_key {
            std::env::set_var("ARUARU_LLM_GOOGLE_SEARCH_API_KEY", v);
        }
        if let Some(v) = saved_cx {
            std::env::set_var("ARUARU_LLM_GOOGLE_SEARCH_CX", v);
        }
    }

    #[test]
    fn format_results_as_context_joins_title_and_snippet() {
        let results = vec![
            SearchResult { title: "A".to_string(), snippet: "aaa".to_string(), link: "http://a".to_string() },
            SearchResult { title: "B".to_string(), snippet: "bbb".to_string(), link: "http://b".to_string() },
        ];
        let ctx = format_results_as_context(&results);
        assert_eq!(ctx, "- A: aaa\n- B: bbb");
    }
}
