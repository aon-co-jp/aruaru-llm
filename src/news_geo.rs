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

#[cfg(test)]
mod tests {
    use super::*;

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
