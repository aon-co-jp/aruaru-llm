//! open-english予備パス(訪問者自身が設定するGoogle Custom Search JSON
//! APIキー/cx)の毎朝の自己点検(ユーザー指示、2026-09-27:
//! 「aruaru-search自身は毎朝7時に自己点検・修復している。open-english側の
//! 予備パス〈訪問者自身のGoogle無料枠キー〉にこそ、この仕組みの必要性が
//! 高い」への対応)。
//!
//! ## 何を点検するか・何を点検しないか(正直な開示)
//!
//! - `aruaru-search`(自前メタ検索)の毎朝の自己点検は、スクレイピング先の
//!   HTML構造が変わってCSSセレクタが壊れていないかをAIで自動修復する仕組み
//!   だが、Google Custom Search JSON APIは構造化JSONを返す公式APIであり
//!   セレクタという概念自体が無い。ここで壊れ得るのは「エンドポイントの
//!   URL自体」「レスポンスのJSONスキーマ」の2点のみ。
//! - **訪問者ひとりひとりのAPIキー/cxを、サーバー側からテストすることは
//!   できない**(サーバーはそれらを一切保存・保持しない設計、
//!   `web_search.rs`冒頭のプライバシー方針を参照)。そのため、この点検は
//!   「Googleの契約(エンドポイントURL・エラー応答のJSON形状)が今日も
//!   変わっていないか」だけを検証する——個々の訪問者の鍵が有効かどうかは
//!   検証できない(検証しようがない)、という制約を正直に記す。
//! - **誰の無料枠(1日100件)も消費しない**: 意図的に無効なAPIキーで
//!   リクエストする。Googleは無効なキーを認証段階で拒否するため
//!   (実際に返るのはHTTP 400、`{"error":{"code":400,"message":...}}`
//!   形式)、クォータ消費前に弾かれる。これにより、開発者・利用者どちらの
//!   実クォータにも一切触れずに「エンドポイント疎通」と「エラー応答の
//!   JSON形状」の2点を確認できる。
//!
//! 毎朝7時(日本時間)と起動時に点検し、結果を`GET /v1/search/fallback-status`
//! で開示する。異常を検知しても`open-english`のGoogle検索補強機能自体を
//! 止めはしない(既存の「補助機能の失敗は権威パスをブロックしない」方針を
//! 踏襲)——あくまで開発者・利用者への早期警告のみが目的。

use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const GOOGLE_CSE_ENDPOINT: &str = "https://www.googleapis.com/customsearch/v1";
/// 実在しない・意図的に無効なキー/cx(誰のクォータも消費しないため)。
const CANARY_API_KEY: &str = "AIzaSyINVALID-canary-key-for-daily-health-check";
const CANARY_CX: &str = "000000000000000000000:aaaaaaaaaaa";

const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 3600);
/// 起動直後の点検は、他の起動処理(モデル読み込み等)と競合しないよう
/// 少し遅らせる。
const STARTUP_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackHealth {
    pub last_checked_unix: u64,
    /// エンドポイント自体に到達できたか(DNS・TLS・接続レベル)。
    pub endpoint_reachable: bool,
    /// エラー応答が期待通りのJSON形状(`error.code`/`error.message`)を
    /// 保っているか——保っていれば、Google側の契約(レスポンス形状)は
    /// 今日も変わっていないと判断できる。
    pub schema_ok: bool,
    pub note_ja: String,
    pub note_en: String,
}

impl Default for FallbackHealth {
    fn default() -> Self {
        Self {
            last_checked_unix: 0,
            endpoint_reachable: false,
            schema_ok: false,
            note_ja: "起動直後のため未点検です。".to_string(),
            note_en: "Not checked yet (just started up).".to_string(),
        }
    }
}

static HEALTH: RwLock<Option<FallbackHealth>> = RwLock::new(None);

pub fn current() -> FallbackHealth {
    HEALTH
        .read()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_default()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// カナリア(意図的に無効な鍵)でGoogle Custom Search JSON APIへ1回だけ
/// 問い合わせ、エンドポイント疎通とエラー応答の形状を確認する。
async fn check_once() -> FallbackHealth {
    let client = match reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            return FallbackHealth {
                last_checked_unix: now_unix(),
                endpoint_reachable: false,
                schema_ok: false,
                note_ja: format!("HTTPクライアントの初期化に失敗しました: {err:#}"),
                note_en: format!("Failed to build the HTTP client: {err:#}"),
            };
        }
    };

    let res = client
        .get(GOOGLE_CSE_ENDPOINT)
        .query(&[("key", CANARY_API_KEY), ("cx", CANARY_CX), ("q", "canary health check")])
        .send()
        .await;

    match res {
        Err(err) => FallbackHealth {
            last_checked_unix: now_unix(),
            endpoint_reachable: false,
            schema_ok: false,
            note_ja: format!(
                "Google Custom Search JSON APIのエンドポイントに到達できませんでした\
                 (open-english予備パスが機能しない可能性があります): {err:#}"
            ),
            note_en: format!(
                "Could not reach the Google Custom Search JSON API endpoint \
                 (open-english's backup search path may not work): {err:#}"
            ),
        },
        Ok(resp) => {
            let status = resp.status();
            let body: Result<serde_json::Value, _> = resp.json().await;
            match body {
                Ok(json) => {
                    // 無効なキーで問い合わせた場合、Googleは通常
                    // `{"error": {"code": 400, "message": "..."}}`形式で
                    // 400を返す。この形状が保たれていれば、レスポンス
                    // スキーマは変わっていないと判断する(成功時の
                    // `items[].{title,snippet,link}`形状は無効な鍵では
                    // 確認できないため、あくまで「契約が変わっていない
                    // ことの間接的な裏付け」に留まる、正直な開示)。
                    let schema_ok = json.get("error").and_then(|e| e.get("code")).is_some()
                        && json.get("error").and_then(|e| e.get("message")).is_some();
                    if schema_ok {
                        FallbackHealth {
                            last_checked_unix: now_unix(),
                            endpoint_reachable: true,
                            schema_ok: true,
                            note_ja: "正常: エンドポイントに到達でき、想定通りのエラー応答形状でした(誰の無料枠も消費していません)。".to_string(),
                            note_en: "OK: endpoint reachable, error response shape matched expectations (no one's free-tier quota was consumed).".to_string(),
                        }
                    } else {
                        FallbackHealth {
                            last_checked_unix: now_unix(),
                            endpoint_reachable: true,
                            schema_ok: false,
                            note_ja: format!(
                                "警告: エンドポイントには到達できましたが、応答のJSON形状が想定と異なります(HTTP {status})。\
                                 Google側の仕様変更の可能性があり、open-english予備パスの動作確認が必要です: {json}"
                            ),
                            note_en: format!(
                                "Warning: endpoint reachable but the response JSON shape did not match expectations (HTTP {status}). \
                                 Google may have changed the API contract; open-english's backup search path should be re-verified: {json}"
                            ),
                        }
                    }
                }
                Err(err) => FallbackHealth {
                    last_checked_unix: now_unix(),
                    endpoint_reachable: true,
                    schema_ok: false,
                    note_ja: format!(
                        "警告: エンドポイントには到達できましたが、応答をJSONとして解析できませんでした(HTTP {status}): {err:#}"
                    ),
                    note_en: format!(
                        "Warning: endpoint reachable but the response could not be parsed as JSON (HTTP {status}): {err:#}"
                    ),
                },
            }
        }
    }
}

/// 次の日本時間7:00までの秒数(既に7時を過ぎていれば翌日の7時まで)。
fn seconds_until_next_7am_jst() -> u64 {
    let now = now_unix();
    const JST_OFFSET_SECS: u64 = 9 * 3600;
    let jst_now = now + JST_OFFSET_SECS;
    let secs_into_day = jst_now % 86_400;
    let seven_am_secs = 7 * 3600;
    if secs_into_day < seven_am_secs {
        seven_am_secs - secs_into_day
    } else {
        86_400 - secs_into_day + seven_am_secs
    }
}

/// バックグラウンドで、起動時+毎朝7時(日本時間)+以降24時間おきに点検する。
pub fn spawn() {
    tokio::spawn(async move {
        tokio::time::sleep(STARTUP_DELAY).await;
        loop {
            let health = check_once().await;
            tracing::info!(
                endpoint_reachable = health.endpoint_reachable,
                schema_ok = health.schema_ok,
                "google_search_fallback_health: checked (open-english backup search path)"
            );
            if let Ok(mut g) = HEALTH.write() {
                *g = Some(health);
            }
            let wait = seconds_until_next_7am_jst().min(CHECK_INTERVAL.as_secs());
            tokio::time::sleep(Duration::from_secs(wait.max(60))).await;
        }
    });
}
