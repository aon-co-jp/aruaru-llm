//! サーバーの接続先国のニュースを収集し、簡易ローカルDB(JSONファイル)へ
//! 保存する機能(ユーザー指示、2026-08-17「メンテナンス時にその人のIP
//! アドレスからその国のインターネットニュースを読んで情報収集、分析して
//! DATABASE化して、話題についていけるように努力して」への対応)。
//!
//! ## 正直な開示・スコープ(重要)
//!
//! - **IPアドレスの取得元**: `open-english`はローカル常駐サーバー
//!   (Phase 0設計、`open-english/CLAUDE.md`参照)であり、利用者の
//!   ブラウザは常に同一端末またはLAN上から`aruaru-llm`
//!   (`http://localhost:4600`)へ接続する。このためHTTPリクエストの
//!   接続元ソケットIPは常に`127.0.0.1`/プライベートIPとなり、そこから
//!   「利用者の国」を判定することはできない。本実装は代わりに、
//!   このサーバー自身が実際にインターネットへ到達する際に使う公開IP
//!   (ip-api.comの自己検出エンドポイント、`/json/`にIPを指定せず呼ぶと
//!   呼び出し元の公開IPを自動判定する)から国を判定する——「利用者の
//!   ブラウザの接続元」ではなく「このサーバーが実際に置かれている
//!   ネットワークの接続先国」を代理指標として使う設計であることを
//!   明示する。
//! - **ニュース取得**: 専用のニュースAPI契約は結ばず、既存の
//!   `web_search`(Google Custom Search)連携を再利用する
//!   (`ARUARU_LLM_GOOGLE_SEARCH_API_KEY`/`ARUARU_LLM_GOOGLE_SEARCH_CX`
//!   未設定なら正直に「未設定」を返し、黙って空のニュースを捏造しない)。
//! - **「DATABASE化」の実体**: SQLデータベースではなく、ローカル
//!   ファイル(`data/news_db.json`)への構造化JSON永続化に留まる
//!   (aruaru-dbのような本格的なDB接続は今回のスコープ外、軽量な
//!   ローカル保存として正直に開示する)。

use std::path::PathBuf;
use std::sync::RwLock;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::web_search::{self, SearchResult};

const IP_GEOLOCATION_ENDPOINT: &str = "http://ip-api.com/json/";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountryInfo {
    pub country: String,
    pub country_code: String,
    pub query_ip: String,
}

#[derive(Debug, Deserialize)]
struct IpApiResponse {
    status: String,
    #[serde(default)]
    country: String,
    #[serde(rename = "countryCode", default)]
    country_code: String,
    #[serde(default)]
    query: String,
}

/// このサーバー自身の公開IPから国を判定する(上記モジュールdoc参照、
/// IPを明示せず呼ぶことでip-api.com側が呼び出し元の公開IPを自動判定)。
pub async fn detect_server_country() -> Result<CountryInfo> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for IP geolocation")?;

    let res = client
        .get(IP_GEOLOCATION_ENDPOINT)
        .send()
        .await
        .context("IP geolocation request failed")?;
    let body: IpApiResponse = res.json().await.context("failed to parse IP geolocation response")?;

    if body.status != "success" {
        anyhow::bail!("IP geolocation lookup did not succeed (status={})", body.status);
    }

    Ok(CountryInfo { country: body.country, country_code: body.country_code, query_ip: body.query })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    pub title: String,
    pub snippet: String,
    pub link: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NewsDb {
    pub country: Option<CountryInfo>,
    pub items: Vec<NewsItem>,
    /// Unix秒。UI/呼び出し側が鮮度を判断できるよう保持する。
    pub fetched_at_unix: Option<u64>,
    /// Google Search未設定等で取得できなかった場合の正直な理由。
    pub last_error: Option<String>,
}

fn news_db_path() -> PathBuf {
    std::env::var("ARUARU_LLM_NEWS_DB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/news_db.json"))
}

static NEWS_DB: RwLock<Option<NewsDb>> = RwLock::new(None);

fn load_from_disk() -> NewsDb {
    let path = news_db_path();
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn persist_to_disk(db: &NewsDb) {
    let path = news_db_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(db) {
        let _ = std::fs::write(&path, json);
    }
}

/// 現在保持しているニュースDBのスナップショットを返す(`GET /v1/news/latest`)。
pub fn get_latest() -> NewsDb {
    {
        let guard = NEWS_DB.read().expect("news db lock poisoned");
        if let Some(db) = guard.clone() {
            return db;
        }
    }
    let loaded = load_from_disk();
    *NEWS_DB.write().expect("news db lock poisoned") = Some(loaded.clone());
    loaded
}

fn news_query_for_country(country: &str) -> String {
    if country == "Japan" {
        "日本 ニュース 今日 主要".to_string()
    } else {
        format!("{country} news today headlines")
    }
}

/// 国を検出し、その国のニュースをGoogle Custom Searchで取得してローカル
/// DBへ保存する(`POST /v1/news/refresh`)。**正直な開示**: いずれかの
/// 段階(IP判定・Google未設定)で失敗しても、それまでに分かった情報
/// (国名のみ等)を保存し、`last_error`に理由を正直に記録する——
/// サービス全体を落とさない既存の可用性優先方針を踏襲する。
pub async fn refresh() -> NewsDb {
    let country_result = detect_server_country().await;

    let mut db = NewsDb::default();
    let country = match country_result {
        Ok(c) => {
            db.country = Some(c.clone());
            Some(c)
        }
        Err(e) => {
            db.last_error = Some(format!("IP geolocation failed: {e}"));
            None
        }
    };

    if let Some(c) = country {
        if !web_search::is_configured() {
            db.last_error = Some(
                "Google Custom Search is not configured (set ARUARU_LLM_GOOGLE_SEARCH_API_KEY / \
                 ARUARU_LLM_GOOGLE_SEARCH_CX) — no news fetched".to_string(),
            );
        } else {
            let query = news_query_for_country(&c.country);
            match web_search::search(&query, 8).await {
                Ok(results) => {
                    db.items = results.into_iter().map(|r: SearchResult| NewsItem { title: r.title, snippet: r.snippet, link: r.link }).collect();
                }
                Err(e) => {
                    db.last_error = Some(format!("news search failed: {e}"));
                }
            }
        }
    }

    db.fetched_at_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs());

    persist_to_disk(&db);
    *NEWS_DB.write().expect("news db lock poisoned") = Some(db.clone());
    db
}

/// チャット応答へ短く織り込むための日英併記の要約行(open-english側の
/// 「話題についていけるように」に対応、上位2件のタイトルのみ)。
pub fn topic_context_line(db: &NewsDb) -> Option<String> {
    if db.items.is_empty() {
        return None;
    }
    let country = db.country.as_ref().map(|c| c.country.as_str()).unwrap_or("your area");
    let headlines = db.items.iter().take(2).map(|i| i.title.as_str()).collect::<Vec<_>>().join(" / ");
    Some(format!("Recent news from {country}: {headlines}"))
}

// ── AI/LLM ニュース(多言語、2026-09-11新設) ─────────────────────────
//
// ユーザー指示「起動時のメンテナンス時に自動でインターネットニュースを
// 英語と日本語と中国語と台湾語で読む機能」への対応。上の`refresh()`
// (サーバー接続先国の一般ニュース、単一言語)とは別物として追加する
// ——検索テーマを「AI/LLM関連」に固定し、言語は接続先国に関わらず常に
// 4言語(英語・日本語・簡体字中国語・繁体字中国語)で取得する。
//
// **正直な開示**: これは表示専用の情報収集であり、取得した内容を根拠に
// 使用モデルを自動で切り替えることは一切しない(モデル変更は既存の
// `/v1/recommend-and-download`・`/v1/download-larger`・
// `/v1/download-smaller`——いずれもユーザー操作起点——のみが行う)。
// `web_search`(Google Custom Search)未設定なら、他の機能同様に正直に
// 「未設定」を返し、ニュースを捏造しない。

/// 言語コード(表示用)と検索クエリの組。
const AI_NEWS_QUERIES: &[(&str, &str)] = &[
    ("en", "latest open source LLM AI model news 2026"),
    ("ja", "最新 オープンソース LLM AI モデル ニュース 2026"),
    ("zh-CN", "最新 开源 LLM AI 模型 新闻 2026"),
    ("zh-TW", "最新 開源 LLM AI 模型 新聞 2026"),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiNewsItem {
    pub lang: String,
    pub title: String,
    pub snippet: String,
    pub link: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiNewsDb {
    pub items: Vec<AiNewsItem>,
    pub fetched_at_unix: Option<u64>,
    pub last_error: Option<String>,
}

fn ai_news_db_path() -> PathBuf {
    std::env::var("ARUARU_LLM_AI_NEWS_DB_PATH").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("data/ai_news_db.json"))
}

static AI_NEWS_DB: RwLock<Option<AiNewsDb>> = RwLock::new(None);

fn load_ai_news_from_disk() -> AiNewsDb {
    let path = ai_news_db_path();
    std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

fn persist_ai_news_to_disk(db: &AiNewsDb) {
    let path = ai_news_db_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(db) {
        let _ = std::fs::write(&path, json);
    }
}

/// 直近保存済みのAI/LLMニュースDBスナップショットを返す
/// (`GET /v1/news/ai-latest`)。
pub fn get_latest_ai_news() -> AiNewsDb {
    {
        let guard = AI_NEWS_DB.read().expect("ai news db lock poisoned");
        if let Some(db) = guard.clone() {
            return db;
        }
    }
    let loaded = load_ai_news_from_disk();
    *AI_NEWS_DB.write().expect("ai news db lock poisoned") = Some(loaded.clone());
    loaded
}

/// 4言語でAI/LLM関連ニュースを取得しローカルDBへ保存する
/// (`POST /v1/news/ai-refresh`)。open-englishのメンテナンスバナー表示中に
/// 叩かれる想定(既存`refresh()`と同じ呼び出しタイミング)。1言語の検索が
/// 失敗しても他の言語は継続する(部分的な結果を正直に返す、全滅時のみ
/// `last_error`を設定)。
pub async fn refresh_ai_news() -> AiNewsDb {
    let mut db = AiNewsDb::default();

    if !web_search::is_configured() {
        db.last_error = Some(
            "Google Custom Search is not configured (set ARUARU_LLM_GOOGLE_SEARCH_API_KEY / \
             ARUARU_LLM_GOOGLE_SEARCH_CX) — no AI/LLM news fetched"
                .to_string(),
        );
        persist_ai_news_to_disk(&db);
        *AI_NEWS_DB.write().expect("ai news db lock poisoned") = Some(db.clone());
        return db;
    }

    let mut errors = Vec::new();
    for &(lang, query) in AI_NEWS_QUERIES {
        match web_search::search(query, 4).await {
            Ok(results) => {
                db.items.extend(results.into_iter().map(|r: SearchResult| AiNewsItem { lang: lang.to_string(), title: r.title, snippet: r.snippet, link: r.link }));
            }
            Err(e) => errors.push(format!("{lang}: {e}")),
        }
    }
    if db.items.is_empty() && !errors.is_empty() {
        db.last_error = Some(format!("all AI/LLM news searches failed: {}", errors.join(" | ")));
    } else if !errors.is_empty() {
        db.last_error = Some(format!("some AI/LLM news searches failed (partial results kept): {}", errors.join(" | ")));
    }

    db.fetched_at_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs());

    persist_ai_news_to_disk(&db);
    *AI_NEWS_DB.write().expect("ai news db lock poisoned") = Some(db.clone());
    db
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_news_queries_cover_english_japanese_simplified_and_traditional_chinese() {
        let langs: Vec<&str> = AI_NEWS_QUERIES.iter().map(|(lang, _)| *lang).collect();
        assert_eq!(langs, vec!["en", "ja", "zh-CN", "zh-TW"]);
    }

    #[test]
    fn news_query_for_country_uses_japanese_for_japan() {
        assert_eq!(news_query_for_country("Japan"), "日本 ニュース 今日 主要");
    }

    #[test]
    fn news_query_for_country_uses_english_for_others() {
        assert_eq!(news_query_for_country("France"), "France news today headlines");
    }

    #[test]
    fn topic_context_line_none_when_empty() {
        let db = NewsDb::default();
        assert!(topic_context_line(&db).is_none());
    }

    #[test]
    fn topic_context_line_joins_top_two_titles() {
        let db = NewsDb {
            country: Some(CountryInfo { country: "Japan".to_string(), country_code: "JP".to_string(), query_ip: "1.2.3.4".to_string() }),
            items: vec![
                NewsItem { title: "A".to_string(), snippet: "".to_string(), link: "".to_string() },
                NewsItem { title: "B".to_string(), snippet: "".to_string(), link: "".to_string() },
                NewsItem { title: "C".to_string(), snippet: "".to_string(), link: "".to_string() },
            ],
            fetched_at_unix: None,
            last_error: None,
        };
        assert_eq!(topic_context_line(&db).unwrap(), "Recent news from Japan: A / B");
    }
}
