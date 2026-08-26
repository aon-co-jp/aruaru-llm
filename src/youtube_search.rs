//! YouTube検索連携(ユーザー指示「Youtube連携機能も付けて」への対応)。
//! `web_search.rs`(Google Custom Search)と全く同じ設計パターン——
//! チェックボックスで有効化し、実際にYouTube Data API v3の検索
//! エンドポイントへ本物のHTTPリクエストを送り、上位結果(動画タイトル・
//! チャンネル名・URL)をチャット応答のプロンプトへコンテキストとして
//! 埋め込む。
//!
//! ## 正直な開示(最重要)
//!
//! - YouTube Data API v3の利用には、ユーザー自身が
//!   [Google Cloud Console](https://console.cloud.google.com/)で
//!   「YouTube Data API v3」を有効化しAPIキーを取得する必要がある
//!   (無料枠は1日1万ユニット、検索1回=100ユニット消費のため実質1日
//!   約100回)。このリポジトリはAPIキーを一切保持・同梱しない
//!   (`web_search.rs`のGoogle Custom Searchと同じ、意図的な例外)。
//! - APIキー未設定の場合、この機能は静かに無効化され
//!   (`is_configured()`が`false`を返す)、呼び出し元はYouTube検索無しの
//!   通常応答へフォールバックする(サービス全体を壊さない設計)。
//! - 検索結果の動画タイトル・説明はYouTube上の著作物であり、本文を
//!   丸ごと転載せず、上位数件のタイトル・チャンネル名・URLのみを
//!   プロンプトへ埋め込む(`web_search.rs`と同じ引用範囲の配慮)。
//!   実際の動画内容(音声・字幕の書き起こし等)は一切取得しない。

use std::sync::RwLock;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const YOUTUBE_SEARCH_ENDPOINT: &str = "https://www.googleapis.com/youtube/v3/search";

/// 利用者がブラウザの設定パネルから入力したAPIキーを、実行中の
/// プロセスのメモリ上にのみ保持する(`web_search::RUNTIME_CREDENTIALS`と
/// 同じ設計、ディスクへの永続化は一切行わない)。
static RUNTIME_KEY: RwLock<Option<String>> = RwLock::new(None);

pub fn set_runtime_key(api_key: String) {
    let mut guard = RUNTIME_KEY.write().expect("runtime youtube key lock poisoned");
    *guard = if api_key.trim().is_empty() { None } else { Some(api_key) };
}

pub fn clear_runtime_key() {
    set_runtime_key(String::new());
}

pub fn is_configured() -> bool {
    read_key().is_some()
}

fn read_key() -> Option<String> {
    if let Some(key) = RUNTIME_KEY.read().expect("runtime youtube key lock poisoned").clone() {
        return Some(key);
    }
    let key = std::env::var("ARUARU_LLM_YOUTUBE_API_KEY").ok()?;
    if key.trim().is_empty() {
        return None;
    }
    Some(key)
}

#[derive(Debug, Clone, Serialize)]
pub struct YoutubeResult {
    pub title: String,
    pub channel_title: String,
    pub url: String,
}

#[derive(Debug, Deserialize)]
struct YoutubeSearchResponse {
    #[serde(default)]
    items: Vec<YoutubeItem>,
}

#[derive(Debug, Deserialize)]
struct YoutubeItem {
    #[serde(default)]
    id: YoutubeItemId,
    #[serde(default)]
    snippet: YoutubeSnippet,
}

#[derive(Debug, Deserialize, Default)]
struct YoutubeItemId {
    #[serde(default)]
    video_id: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct YoutubeSnippet {
    #[serde(default)]
    title: String,
    #[serde(default)]
    channel_title: String,
}

pub async fn search(query: &str, max_results: u8) -> Result<Vec<YoutubeResult>> {
    let key = read_key().context("YouTube Data API is not configured (set ARUARU_LLM_YOUTUBE_API_KEY or use the settings panel)")?;
    search_with_key(query, max_results, &key).await
}

/// `search()`と同じ検索処理だが、呼び出し元が明示的に渡したAPIキーのみを
/// 使う版(`web_search::search_with_credentials`と同じ設計、共有VPS上で
/// 他利用者のグローバル設定を誤って消費しないため)。
pub async fn search_with_key(query: &str, max_results: u8, api_key: &str) -> Result<Vec<YoutubeResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }

    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(8)).build().context("failed to build reqwest client for YouTube search")?;

    let res = client
        .get(YOUTUBE_SEARCH_ENDPOINT)
        .query(&[("key", api_key), ("q", query), ("part", "snippet"), ("type", "video"), ("maxResults", &max_results.clamp(1, 10).to_string())])
        .send()
        .await
        .context("YouTube search request failed")?;

    if !res.status().is_success() {
        let status = res.status();
        // Googleのエラー本文には具体的な理由が入っており診断に有用
        // ——APIキーの値自体がエコーバックされることは無い(Google側の
        // 仕様、web_search.rsと同じ確認済みの前提)ためそのまま含めても
        // 安全。
        let body = res.text().await.unwrap_or_else(|_| "(failed to read response body)".to_string());
        bail!("YouTube search returned HTTP {status}: {body}");
    }

    let body: YoutubeSearchResponse = res.json().await.context("failed to parse YouTube search response")?;
    Ok(body
        .items
        .into_iter()
        .filter_map(|item| {
            let video_id = item.id.video_id?;
            Some(YoutubeResult { title: item.snippet.title, channel_title: item.snippet.channel_title, url: format!("https://www.youtube.com/watch?v={video_id}") })
        })
        .collect())
}

/// 検索結果を、生成プロンプトへ埋め込むための短いコンテキスト文字列へ
/// 整形する(タイトル+チャンネル名のみ、本文丸ごと転載はしない)。
pub fn format_results_as_context(results: &[YoutubeResult]) -> String {
    results.iter().map(|r| format!("- {} ({})", r.title, r.channel_title)).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_configured_false_when_env_and_runtime_absent() {
        clear_runtime_key();
        let saved = std::env::var("ARUARU_LLM_YOUTUBE_API_KEY").ok();
        std::env::remove_var("ARUARU_LLM_YOUTUBE_API_KEY");

        assert!(!is_configured());

        if let Some(v) = saved {
            std::env::set_var("ARUARU_LLM_YOUTUBE_API_KEY", v);
        }
    }

    #[test]
    fn set_and_clear_runtime_key_round_trips() {
        clear_runtime_key();
        assert!(!is_configured());
        set_runtime_key("AIzaSy-test".to_string());
        assert!(is_configured());
        clear_runtime_key();
        assert!(!is_configured());
    }

    #[test]
    fn format_results_as_context_joins_title_and_channel() {
        let results = vec![YoutubeResult { title: "Learn Rust".to_string(), channel_title: "RustConf".to_string(), url: "https://www.youtube.com/watch?v=abc123".to_string() }];
        let ctx = format_results_as_context(&results);
        assert_eq!(ctx, "- Learn Rust (RustConf)");
    }

    #[tokio::test]
    async fn search_rejects_empty_query_even_without_key() {
        let err = search_with_key("  ", 3, "fake-key").await.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }
}
