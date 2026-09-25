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

use std::sync::{Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const GOOGLE_CSE_ENDPOINT: &str = "https://www.googleapis.com/customsearch/v1";

/// 共有(開発者設定)キー経由の検索1日あたりの上限(2026-09-15追加、
/// ユーザー指示「開発者のGoogle検索API KEYの無料枠から公開してあるので
/// だれでも利用出来ます。しかし一日に百回までしか利用出来ず、それが
/// 終わったら…自動で移って下さい」への対応)。Google Custom Search JSON
/// APIの無料枠自体が1日100件までのため、それに合わせた値。
///
/// **2026-08-25の既存方針との関係**: 従来は「開発者が設定したキーは
/// アクセス者に一切消費させない」方針だった(`search_with_credentials`の
/// doc参照)。今回はデモ環境(`easy-web.tokyo/open-english/demo`)限定で
/// この方針を明示的に反転し、開発者の無料枠を来訪者へ公開した上で、
/// 1日100件という上限を全訪問者合算でグローバルに強制する
/// (訪問者ごとの個別カウントではなく、Google側の実際の無料枠と一致
/// させるための単純化)。上限到達後は本関数がエラーを返し、
/// 呼び出し元(`/v1/generate-with-search`・`/v1/chat-providers/
/// complete-priority`)は既存の設計どおり検索無し(またはChatGPT/Gemini/
/// DeepSeek/Grok/Claude優先順チェーン)へ自動的にフォールバックする。
const SHARED_SEARCH_DAILY_LIMIT: u32 = 100;

/// (該当日のUNIX日数, その日の使用回数)。プロセス再起動で0へ戻る
/// (永続化しない——1日の境界をまたいで多少ずれても実害が小さい単純な
/// カウンタで十分という判断、既存のチャットレート制限と同じ思想)。
static SHARED_SEARCH_DAILY_COUNT: Mutex<(u64, u32)> = Mutex::new((0, 0));

fn today_epoch_day() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() / 86_400).unwrap_or(0)
}

/// 共有キーの本日の残り利用可能回数を返す(UI表示用)。
pub fn shared_search_quota_remaining() -> u32 {
    let today = today_epoch_day();
    let guard = SHARED_SEARCH_DAILY_COUNT.lock().expect("shared search counter lock poisoned");
    if guard.0 == today {
        SHARED_SEARCH_DAILY_LIMIT.saturating_sub(guard.1)
    } else {
        SHARED_SEARCH_DAILY_LIMIT
    }
}

/// 共有キーを1回消費してよいか判定し、よければカウントを1増やす
/// (日付が変わっていれば0から数え直す)。
fn try_consume_shared_search_quota() -> bool {
    let today = today_epoch_day();
    let mut guard = SHARED_SEARCH_DAILY_COUNT.lock().expect("shared search counter lock poisoned");
    if guard.0 != today {
        *guard = (today, 0);
    }
    if guard.1 >= SHARED_SEARCH_DAILY_LIMIT {
        return false;
    }
    guard.1 += 1;
    true
}

// ── プロバイダーごとの無料枠カウンタ(2026-09-23新設、同日中に日次
// リセットへ再設計) ─────────────────────────────────────────
//
// ユーザー指示「SerpApiを実装してハイブリッド検索機能搭載として。
// ただし無料で利用できる前提です」「無料枠を超えたら、その日は使用を
// 一旦止めて、次の日にリセットが掛かったら再び利用を再開して」への対応。
//
// **正直な開示(重要)**: SerpApiの実際の無料枠はGoogle Custom Search
// (1日100件)とは異なり**月単位**(月100件)。ユーザーが明示的に「日次
// リセット」を希望したため、月間無料枠をそのまま1日の上限にはせず
// (それだと月初の数日で使い切ってしまう)、**月間無料枠を日数(30日)で
// 割った安全な日割り上限**を1日あたりの上限として採用し、それを毎日
// リセットする——100÷30≒3件/日。これにより「毎日リセットされ、待たされる
// 期間も短い」というご要望と、「月間無料枠を使い切って課金が発生しない」
// という安全性の両方を満たす(満遍なく使えば1ヶ月で最大90件、実際の
// 月間無料枠100件以内に収まる)。
//
// **2026-09-23 Bing Search API削除の経緯**: 当初はBing Search API(Azure)も
// SerpApiと並ぶ第二のハイブリッド検索先として実装したが、ユーザーが
// 実際にAzure Portalで新規作成しようとしたところ、Bing Search APIは
// 2025-08-11付でMicrosoftにより完全に廃止(新規リソース作成不可)されて
// いることが判明した。後継の「Grounding with Bing Search」はAIエージェント
// への検索グラウンディング機能として設計されており、本モジュールのように
// 「生の検索結果JSONをAPIキー単体で直接取得する」用途には利用規約上
// 使えない(エージェント経由の回答生成に検索結果を内包させる設計であり、
// スタンドアロンな検索結果取得は想定されていない)。そのため、実装済み
// だったBing関連コード(`search_bing`/`BING_ENDPOINT`等)は全て削除し、
// SerpApiのみを実質的な主力の代替候補として残した。
// 2026-09-23修正: 公式料金ページ(https://serpapi.com/pricing)実機確認により
// 100ではなく250件/月が正しい無料枠と判明(WebFetchで実ページの記載
// "250 searches per month"を確認)。
const SERPAPI_FREE_MONTHLY_LIMIT: u32 = 250;
const DAYS_PER_MONTH_APPROX: u32 = 30;
const SERPAPI_SAFE_DAILY_LIMIT: u32 = SERPAPI_FREE_MONTHLY_LIMIT / DAYS_PER_MONTH_APPROX;

// **2026-09-23追加(Tavily/Exa)**: ユーザー指示「Tavily/Exaも登録してopen-english
// などで使用したい」への対応。どちらもAIエージェント/RAG向けに最適化された検索API
// (クレジットカード登録不要の無料枠あり)。SerpApiと同じ「月間無料枠÷30日」の
// 安全な日割り上限+日次リセットの設計を踏襲する。
// - Tavily: 月1,000クレジット、Basic検索1回=1クレジット消費 → 1000÷30≒33件/日。
// - Exa: 毎月自動付与される$10ぶんの無料クレジット(新規登録時の$20は使い切り型の
//   ボーナスのため、継続的に使える方の$10だけを根拠にする)、通常検索は$7/1000件
//   (1回≒$0.007)→ $10÷$0.007≒1428件/月 → 1428÷30≒47件/日。
const TAVILY_FREE_MONTHLY_LIMIT: u32 = 1000;
const TAVILY_SAFE_DAILY_LIMIT: u32 = TAVILY_FREE_MONTHLY_LIMIT / DAYS_PER_MONTH_APPROX;
const EXA_FREE_MONTHLY_REQUESTS_APPROX: u32 = 1428;
const EXA_SAFE_DAILY_LIMIT: u32 = EXA_FREE_MONTHLY_REQUESTS_APPROX / DAYS_PER_MONTH_APPROX;

// **2026-09-23追加(Jina AI)**: ユーザー指示「Jina AIも追加で組み込んで」
// への対応。無料枠がAPIキー発行時に1,000万トークン付与される方式
// (月次リセットではなく「トークンを使い切るまで」の一括付与)。1検索
// あたり最低1万トークン消費が公式ドキュメントに明記されているため、
// 1,000万÷1万=1,000回が理論上の総回数。日割り上限は他社と同じ考え方で
// 30日分の目安として計算するが、**実際は月次リセットではなく総量制**
// という点が他社と異なることを正直に開示しておく(このAPIの無料枠を
// 使い切ったら、新しいAPIキーを取得し直す運用になる)。
const JINA_FREE_TOTAL_REQUESTS_APPROX: u32 = 1000;
const JINA_SAFE_DAILY_LIMIT: u32 = JINA_FREE_TOTAL_REQUESTS_APPROX / DAYS_PER_MONTH_APPROX;

static SERPAPI_DAILY_COUNT: Mutex<(u64, u32)> = Mutex::new((0, 0));
static TAVILY_DAILY_COUNT: Mutex<(u64, u32)> = Mutex::new((0, 0));
static EXA_DAILY_COUNT: Mutex<(u64, u32)> = Mutex::new((0, 0));
static JINA_DAILY_COUNT: Mutex<(u64, u32)> = Mutex::new((0, 0));

/// [`try_consume_shared_search_quota`]と同じ考え方のプロバイダー別版
/// (純粋関数として分離、`consume_bucket`が実処理・テストしやすい形)。
fn consume_bucket(state: &mut (u64, u32), current_bucket: u64, limit: u32) -> bool {
    if state.0 != current_bucket {
        *state = (current_bucket, 0);
    }
    if state.1 >= limit {
        return false;
    }
    state.1 += 1;
    true
}

fn try_consume_serpapi_quota() -> bool {
    let bucket = today_epoch_day();
    let mut guard = SERPAPI_DAILY_COUNT.lock().expect("serpapi quota lock poisoned");
    consume_bucket(&mut guard, bucket, SERPAPI_SAFE_DAILY_LIMIT)
}

fn try_consume_tavily_quota() -> bool {
    let bucket = today_epoch_day();
    let mut guard = TAVILY_DAILY_COUNT.lock().expect("tavily quota lock poisoned");
    consume_bucket(&mut guard, bucket, TAVILY_SAFE_DAILY_LIMIT)
}

fn try_consume_exa_quota() -> bool {
    let bucket = today_epoch_day();
    let mut guard = EXA_DAILY_COUNT.lock().expect("exa quota lock poisoned");
    consume_bucket(&mut guard, bucket, EXA_SAFE_DAILY_LIMIT)
}

fn try_consume_jina_quota() -> bool {
    let bucket = today_epoch_day();
    let mut guard = JINA_DAILY_COUNT.lock().expect("jina quota lock poisoned");
    consume_bucket(&mut guard, bucket, JINA_SAFE_DAILY_LIMIT)
}

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
    read_brave_key().is_some()
        || read_serpapi_key().is_some()
        || read_tavily_key().is_some()
        || read_exa_key().is_some()
        || read_jina_key().is_some()
        || read_credentials().is_some()
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

/// 共有(開発者設定)キーで検索する。2026-09-23変更(ユーザー指示
/// 「ARUARU_LLM_SERPAPI_KEYを最優先で実際に運用を開始して」): **SerpApi**
/// を第一候補、Brave Search APIを第二候補、従来のGoogle Custom Searchを
/// 最後の候補として順に試す(Google Custom Search JSON APIは新規
/// プロジェクトへの提供を終了しつつあり〈2027年完全終了予定〉、実機で
/// 403エラーを確認済みのため、動く保証はないが完全には外さない)。片方が
/// 失敗(キー未設定・上限・HTTPエラー)したら次の検索サービスへ自動で
/// 移り、全て失敗した場合のみエラーを返す(呼び出し側は検索無しで
/// ChatGPT→Gemini→DeepSeek→Grok→Claudeの優先順チェーンへ進む)。
/// 共有キー全体の1日上限(`SHARED_SEARCH_DAILY_LIMIT`)は、どの検索サービスを
/// 使っても1回の検索につき1回として数える。
pub async fn search(query: &str, max_results: u8) -> Result<Vec<SearchResult>> {
    search_with_locale(query, max_results, None, None).await
}

/// [`search`]のロケール指定版(2026-09-23新設、ユーザー報告「フランスの
/// ニュースを検索したら無関係な結果(学校の話)が返ってきた」への対応)。
///
/// **根本原因**: SerpApiは`gl`(国)/`hl`(言語)パラメータを渡さないと
/// 既定でアメリカ・英語向けのインデックスを検索してしまい、フランス語の
/// クエリ文字列を渡しても関連性の低い結果になりやすい(実機で
/// "France actualités aujourd'hui principales" が学校紹介動画を返した事例で
/// 確認済み)。`news_geo.rs`の国別ニュース取得はこちらを使い、`gl`/`hl`で
/// Google/SerpApi側に明示的に地域・言語を伝える。`search()`(一般的な
/// Q&A用途、国の概念が無い)は従来通り`gl`/`hl`無しのまま。
/// 2026-09-25追加(ユーザー指示「aruaru-searchはaruaruLLMとSETで連動して使用」):
/// 自前のメタ検索`aruaru-search`(https://github.com/aon-co-jp/aruaru-search、APIキー不要・
/// VPSで完全無料、世界約130言語対応)を**第一候補**にする。ここで結果が得られれば、共有キーの
/// 1日上限(`SHARED_SEARCH_DAILY_LIMIT`)も各社の無料枠も消費しない。aruaru-searchが動いていない・
/// 全検索元から拒否された・結果が0件のときは、従来どおり下の共有キー経由のチェーンへ自動で移る。
/// 接続先は環境変数`ARUARU_LLM_SEARCH_URL`(既定`http://127.0.0.1:4610`、空文字で無効化)。
async fn search_via_aruaru_search(query: &str, max_results: u8, gl: Option<&str>, hl: Option<&str>) -> Option<Vec<SearchResult>> {
    let base = std::env::var("ARUARU_LLM_SEARCH_URL").unwrap_or_else(|_| "http://127.0.0.1:4610".to_string());
    if base.trim().is_empty() {
        return None;
    }
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(2))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .ok()?;
    let body = serde_json::json!({
        "q": query,
        "n": max_results,
        "hl": hl.unwrap_or(""),
        "gl": gl.unwrap_or(""),
    });
    let resp = client.post(format!("{}/v1/search", base.trim_end_matches('/'))).json(&body).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().await.ok()?;
    let results: Vec<SearchResult> = json
        .get("results")?
        .as_array()?
        .iter()
        .filter_map(|r| {
            Some(SearchResult {
                title: r.get("title")?.as_str()?.to_string(),
                snippet: r.get("snippet").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                link: r.get("link")?.as_str()?.to_string(),
            })
        })
        .take(max_results as usize)
        .collect();
    if results.is_empty() {
        return None;
    }
    tracing::info!(backend = "aruaru-search", query, count = results.len(), "web_search: served by");
    Some(results)
}

pub async fn search_with_locale(query: &str, max_results: u8, gl: Option<&str>, hl: Option<&str>) -> Result<Vec<SearchResult>> {
    if let Some(results) = search_via_aruaru_search(query, max_results, gl, hl).await {
        return Ok(results);
    }
    let serpapi_key = read_serpapi_key();
    let tavily_key = read_tavily_key();
    let exa_key = read_exa_key();
    let jina_key = read_jina_key();
    let brave_key = read_brave_key();
    let google = read_credentials();
    if serpapi_key.is_none() && tavily_key.is_none() && exa_key.is_none() && jina_key.is_none() && brave_key.is_none() && google.is_none() {
        bail!(
            "no shared search backend is configured (set ARUARU_LLM_SERPAPI_KEY, ARUARU_LLM_TAVILY_KEY, \
             ARUARU_LLM_EXA_KEY, ARUARU_LLM_JINA_KEY, ARUARU_LLM_BRAVE_SEARCH_API_KEY, or ARUARU_LLM_GOOGLE_SEARCH_API_KEY and ARUARU_LLM_GOOGLE_SEARCH_CX)"
        );
    }
    if !try_consume_shared_search_quota() {
        bail!(
            "shared search quota exhausted for today ({SHARED_SEARCH_DAILY_LIMIT}/day) —              falling back to other providers"
        );
    }
    let mut errors: Vec<String> = Vec::new();
    // 2026-09-23変更(ユーザー指示「ARUARU_LLM_SERPAPI_KEYを最優先で実際に
    // 運用を開始して」): SerpApiを最優先候補にする(Braveより先)。
    if let Some(key) = serpapi_key {
        if try_consume_serpapi_quota() {
            match search_serpapi_localized(query, max_results, &key, gl, hl).await {
                Ok(results) if !results.is_empty() => {
                    tracing::info!(backend = "serpapi", query, count = results.len(), "web_search: served by");
                    return Ok(results);
                }
                Ok(_) => errors.push("serpapi: 0 results".to_string()),
                Err(err) => errors.push(format!("serpapi: {err:#}")),
            }
        } else {
            errors.push(format!("serpapi: today's safe daily quota exhausted ({SERPAPI_SAFE_DAILY_LIMIT}/day, derived from the {SERPAPI_FREE_MONTHLY_LIMIT}/month free tier) — resets automatically tomorrow"));
        }
    }
    // 2026-09-23追加(ユーザー指示「Tavily/Exaも登録してopen-englishなどで
    // 使用したい」): AIエージェント/RAG向けに最適化された2社をSerpApiの次、
    // Braveより先に試す(いずれも無料枠がクレジットカード登録不要のため)。
    if let Some(key) = tavily_key {
        if try_consume_tavily_quota() {
            match search_tavily(query, max_results, &key).await {
                Ok(results) if !results.is_empty() => {
                    tracing::info!(backend = "tavily", query, count = results.len(), "web_search: served by");
                    return Ok(results);
                }
                Ok(_) => errors.push("tavily: 0 results".to_string()),
                Err(err) => errors.push(format!("tavily: {err:#}")),
            }
        } else {
            errors.push(format!("tavily: today's safe daily quota exhausted ({TAVILY_SAFE_DAILY_LIMIT}/day, derived from the {TAVILY_FREE_MONTHLY_LIMIT}/month free tier) — resets automatically tomorrow"));
        }
    }
    if let Some(key) = exa_key {
        if try_consume_exa_quota() {
            match search_exa(query, max_results, &key).await {
                Ok(results) if !results.is_empty() => {
                    tracing::info!(backend = "exa", query, count = results.len(), "web_search: served by");
                    return Ok(results);
                }
                Ok(_) => errors.push("exa: 0 results".to_string()),
                Err(err) => errors.push(format!("exa: {err:#}")),
            }
        } else {
            errors.push(format!(
                "exa: today's safe daily quota exhausted ({EXA_SAFE_DAILY_LIMIT}/day, derived from the ~{EXA_FREE_MONTHLY_REQUESTS_APPROX}/month recurring free credit) — resets automatically tomorrow"
            ));
        }
    }
    // 2026-09-23追加(ユーザー指示「Jina AIも追加で組み込んで」)。無料枠が
    // 「APIキー発行時に1,000万トークン一括付与、消費し切るまで」という
    // 総量制のため、他社(月次リセット)と性質が異なることを開示した上で
    // 日割りの目安として最後に配置する。
    if let Some(key) = jina_key {
        if try_consume_jina_quota() {
            match search_jina(query, max_results, &key).await {
                Ok(results) if !results.is_empty() => {
                    tracing::info!(backend = "jina", query, count = results.len(), "web_search: served by");
                    return Ok(results);
                }
                Ok(_) => errors.push("jina: 0 results".to_string()),
                Err(err) => errors.push(format!("jina: {err:#}")),
            }
        } else {
            errors.push(format!(
                "jina: today's safe usage pace exhausted ({JINA_SAFE_DAILY_LIMIT}/day, derived from the ~{JINA_FREE_TOTAL_REQUESTS_APPROX}-request total free allowance, NOT a monthly reset) — resets automatically tomorrow (pacing only, does not add more total credit)"
            ));
        }
    }
    if let Some(key) = brave_key {
        match search_brave(query, max_results, &key).await {
            Ok(results) if !results.is_empty() => {
                tracing::info!(backend = "brave", query, count = results.len(), "web_search: served by");
                return Ok(results);
            }
            Ok(_) => errors.push("brave: 0 results".to_string()),
            Err(err) => errors.push(format!("brave: {err:#}")),
        }
    }
    if let Some((api_key, cx)) = google {
        match search_with_credentials(query, max_results, &api_key, &cx).await {
            Ok(results) => {
                tracing::info!(backend = "google", query, count = results.len(), "web_search: served by");
                return Ok(results);
            }
            Err(err) => errors.push(format!("google: {err:#}")),
        }
    }
    bail!("all shared search backends failed: {}", errors.join(" | "))
}

fn read_brave_key() -> Option<String> {
    let key = std::env::var("ARUARU_LLM_BRAVE_SEARCH_API_KEY").ok()?;
    let key = key.trim().to_string();
    if key.is_empty() { None } else { Some(key) }
}

const BRAVE_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";

#[derive(Debug, Deserialize)]
struct BraveResponse {
    #[serde(default)]
    web: Option<BraveWeb>,
}

#[derive(Debug, Deserialize)]
struct BraveWeb {
    #[serde(default)]
    results: Vec<BraveItem>,
}

#[derive(Debug, Deserialize)]
struct BraveItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    url: String,
}

/// Brave Search APIで検索する(`X-Subscription-Token`ヘッダにAPIキー)。
/// 結果のタイトル・説明文・URLのみを使う(Google版と同じ引用の範囲)。
pub async fn search_brave(query: &str, max_results: u8, api_key: &str) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for Brave Search")?;
    let res = client
        .get(BRAVE_ENDPOINT)
        .header("Accept", "application/json")
        .header("X-Subscription-Token", api_key)
        .query(&[("q", query), ("count", &max_results.clamp(1, 10).to_string())])
        .send()
        .await
        .context("Brave Search request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let body: String = body.chars().take(300).collect();
        bail!("Brave Search returned HTTP {status}: {body}");
    }
    let parsed: BraveResponse = res.json().await.context("failed to parse Brave Search response")?;
    Ok(parsed
        .web
        .map(|w| w.results)
        .unwrap_or_default()
        .into_iter()
        .take(max_results as usize)
        .map(|i| SearchResult { title: i.title, snippet: i.description, link: i.url })
        .collect())
}

fn read_serpapi_key() -> Option<String> {
    let key = std::env::var("ARUARU_LLM_SERPAPI_KEY").ok()?;
    let key = key.trim().to_string();
    if key.is_empty() { None } else { Some(key) }
}

const SERPAPI_ENDPOINT: &str = "https://serpapi.com/search.json";

#[derive(Debug, Deserialize)]
struct SerpApiResponse {
    #[serde(default)]
    organic_results: Vec<SerpApiItem>,
}

#[derive(Debug, Deserialize)]
struct SerpApiItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    link: String,
}

/// **SerpApi**(2026-09-23新設、ユーザー指示「Google Custom Search JSON APIが
/// 廃止方針〈2027-01-01終了〉のため代替を実装して」への対応)。Google検索結果を
/// 代行取得するスクレイピング代行サービス——`engine=google`を指定し、実質的に
/// Google Custom Searchの代替として使える。無料枠は月100件までで、それ以降は
/// 有料(`ARUARU_LLM_SERPAPI_KEY`はユーザー自身が[serpapi.com](https://serpapi.com/)
/// で取得する必要があり、このリポジトリはキーを一切保持・同梱しない——既存の
/// Google/Brave同様の方針)。
pub async fn search_serpapi(query: &str, max_results: u8, api_key: &str) -> Result<Vec<SearchResult>> {
    search_serpapi_localized(query, max_results, api_key, None, None).await
}

/// [`search_serpapi`]のロケール指定版。`gl`(2文字国コード、例: "fr")・
/// `hl`(言語コード、例: "fr")を渡すと、SerpApi(実体はGoogle検索)が
/// その国・言語向けの結果を返すようになる(2026-09-23新設、
/// [`search_with_locale`]のdoc参照)。
pub async fn search_serpapi_localized(query: &str, max_results: u8, api_key: &str, gl: Option<&str>, hl: Option<&str>) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for SerpApi")?;
    let mut params = vec![
        ("engine".to_string(), "google".to_string()),
        ("q".to_string(), query.to_string()),
        ("num".to_string(), max_results.clamp(1, 10).to_string()),
        ("api_key".to_string(), api_key.to_string()),
    ];
    if let Some(gl) = gl {
        params.push(("gl".to_string(), gl.to_string()));
    }
    if let Some(hl) = hl {
        params.push(("hl".to_string(), hl.to_string()));
    }
    let res = client.get(SERPAPI_ENDPOINT).query(&params).send().await.context("SerpApi request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let body: String = body.chars().take(300).collect();
        bail!("SerpApi returned HTTP {status}: {body}");
    }
    let parsed: SerpApiResponse = res.json().await.context("failed to parse SerpApi response")?;
    Ok(parsed
        .organic_results
        .into_iter()
        .take(max_results as usize)
        .map(|i| SearchResult { title: i.title, snippet: i.snippet, link: i.link })
        .collect())
}

fn read_tavily_key() -> Option<String> {
    let key = std::env::var("ARUARU_LLM_TAVILY_KEY").ok()?;
    let key = key.trim().to_string();
    if key.is_empty() { None } else { Some(key) }
}

const TAVILY_ENDPOINT: &str = "https://api.tavily.com/search";

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    results: Vec<TavilyItem>,
}

#[derive(Debug, Deserialize)]
struct TavilyItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    url: String,
}

/// **Tavily**(2026-09-23新設、ユーザー指示「Tavily/Exaも登録して
/// open-englishなどで使用したい」への対応)。AIエージェント/RAG向けに
/// 最適化された検索API——クレジットカード登録不要で月1,000クレジットの
/// 無料枠があり、Basic検索(`search_depth: "basic"`、本関数が使う方)は
/// 1回1クレジット消費(Advanced検索は2クレジットのためコスト面で使わない)。
/// `ARUARU_LLM_TAVILY_KEY`はユーザー自身が[tavily.com](https://www.tavily.com/)
/// で取得する必要があり、このリポジトリはキーを一切保持・同梱しない。
pub async fn search_tavily(query: &str, max_results: u8, api_key: &str) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for Tavily")?;
    let res = client
        .post(TAVILY_ENDPOINT)
        .json(&serde_json::json!({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results.clamp(1, 10),
        }))
        .send()
        .await
        .context("Tavily request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let body: String = body.chars().take(300).collect();
        bail!("Tavily returned HTTP {status}: {body}");
    }
    let parsed: TavilyResponse = res.json().await.context("failed to parse Tavily response")?;
    Ok(parsed
        .results
        .into_iter()
        .take(max_results as usize)
        .map(|i| SearchResult { title: i.title, snippet: i.content, link: i.url })
        .collect())
}

fn read_exa_key() -> Option<String> {
    let key = std::env::var("ARUARU_LLM_EXA_KEY").ok()?;
    let key = key.trim().to_string();
    if key.is_empty() { None } else { Some(key) }
}

const EXA_ENDPOINT: &str = "https://api.exa.ai/search";

#[derive(Debug, Deserialize)]
struct ExaResponse {
    #[serde(default)]
    results: Vec<ExaItem>,
}

#[derive(Debug, Deserialize)]
struct ExaItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    highlights: Vec<String>,
}

/// **Exa**(旧Metaphor、2026-09-23新設)。キーワード一致ではなく意味
/// (セマンティック)ベースの検索を行うニューラル検索エンジン。新規登録時に
/// $20ぶんの無料クレジット(使い切り型ボーナス)に加え、毎月$10ぶんが
/// 自動的に再付与される——本モジュールの日割り上限計算は、いずれ尽きる
/// $20ボーナスではなく継続的な$10/月の方だけを根拠にしている(正直な開示)。
///
/// **2026-09-23修正**: Exa公式の`build-with-exa`スキル(`npx skills use
/// "https://github.com/exa-labs/agent-skills" --skill "build-with-exa"`、
/// ユーザー指示により参照)によると、スニペット抽出の推奨方式は
/// `contents.text`+`maxCharacters`ではなく`contents.highlights: true`
/// (「ほぼ全てのタスクでbare `highlights: true`を使うべき、
/// `maxCharacters`は明示的な予算要件がある場合のみ」と明記)。当初の実装は
/// この推奨に反していたため、`highlights: true`へ修正し、レスポンスの
/// `results[].highlights`(文字列配列)を結合してスニペットとして使う形へ
/// 直した。`ARUARU_LLM_EXA_KEY`はユーザー自身が[exa.ai](https://exa.ai/)で
/// 取得する必要があり、このリポジトリはキーを一切保持・同梱しない。
pub async fn search_exa(query: &str, max_results: u8, api_key: &str) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for Exa")?;
    let res = client
        .post(EXA_ENDPOINT)
        .header("x-api-key", api_key)
        .json(&serde_json::json!({
            "query": query,
            "type": "auto",
            "numResults": max_results.clamp(1, 10),
            "contents": {"highlights": true},
        }))
        .send()
        .await
        .context("Exa request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let body: String = body.chars().take(300).collect();
        bail!("Exa returned HTTP {status}: {body}");
    }
    let parsed: ExaResponse = res.json().await.context("failed to parse Exa response")?;
    Ok(parsed
        .results
        .into_iter()
        .take(max_results as usize)
        .map(|i| SearchResult { title: i.title, snippet: i.highlights.join(" … "), link: i.url })
        .collect())
}

fn read_jina_key() -> Option<String> {
    let key = std::env::var("ARUARU_LLM_JINA_KEY").ok()?;
    let key = key.trim().to_string();
    if key.is_empty() { None } else { Some(key) }
}

const JINA_ENDPOINT: &str = "https://s.jina.ai/";

#[derive(Debug, Deserialize)]
struct JinaResponse {
    #[serde(default)]
    data: Vec<JinaItem>,
}

#[derive(Debug, Deserialize)]
struct JinaItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    description: String,
}

/// **Jina AI Search**(`s.jina.ai`、2026-09-23新設、ユーザー指示「Jina AIも
/// 追加で組み込んで」への対応)。無料枠はAPIキー発行時に1,000万トークンが
/// 一括付与される方式(1検索あたり最低1万トークン消費、公式ドキュメント
/// 記載)——月次リセットの他社とは性質が異なり、実質「使い切ったら新しい
/// APIキーを取得し直す」運用になる(`JINA_SAFE_DAILY_LIMIT`は月次リセット
/// を前提にしていない、あくまで消費ペースの目安)。
///
/// **実機検証済み(2026-09-23)**: このAPIは無料枠であってもAPIキー無しでは
/// `401 AuthenticationRequiredError`を返すことを確認済み。実際に有効な
/// APIキーで`curl "https://s.jina.ai/?q=..."`を叩き、
/// `{"code":200,"status":20000,"data":[{"title":...,"url":...,
/// "description":...,"date":...,"content":...}]}`という形を実際に確認した
/// ——実装時の推測(`title`/`url`/`content`、無ければ`description`)が
/// そのまま正しかった。
pub async fn search_jina(query: &str, max_results: u8, api_key: &str) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        bail!("search query must not be empty");
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("failed to build reqwest client for Jina AI Search")?;
    let res = client
        .get(JINA_ENDPOINT)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Accept", "application/json")
        .query(&[("q", query)])
        .send()
        .await
        .context("Jina AI Search request failed")?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        let body: String = body.chars().take(300).collect();
        bail!("Jina AI Search returned HTTP {status}: {body}");
    }
    let parsed: JinaResponse = res.json().await.context("failed to parse Jina AI Search response")?;
    Ok(parsed
        .data
        .into_iter()
        .take(max_results as usize)
        .map(|i| {
            let snippet = if !i.content.is_empty() { i.content.chars().take(300).collect() } else { i.description };
            SearchResult { title: i.title, snippet, link: i.url }
        })
        .collect())
}

/// `search()`と同じ検索処理だが、プロセス全体で共有される
/// `read_credentials()`(環境変数/`POST /v1/settings/google-search`で
/// 設定されたグローバルな認証情報)を一切参照・消費しない版
/// (2026-08-25新設、ユーザー指示「ブラウザ版は各自Google検索のAPIキーと
/// IDを各自で設定してもらう様に…開発者が設定したAPIキーとIDは、
/// アクセス者は使わない、消費しない様に」への対応)。
///
/// **設計上の理由**: 複数の利用者が同じ`aruaru-llm`インスタンス
/// (例: VPS上の共有デプロイ)へブラウザ経由でアクセスする場合、
/// `read_credentials()`はプロセス全体で1組しか保持できない
/// グローバルな設定であり、ある利用者が自分のキーを
/// `POST /v1/settings/google-search`で設定すると**他の全利用者の
/// 検索もそのキーへ切り替わってしまう**(意図しない共有・消費)。
/// この関数は呼び出し元(`/v1/generate-with-search`)がリクエスト
/// ボディで直接渡した認証情報のみを使い、グローバル状態には
/// 一切触れないため、各ブラウザ利用者が自分のキーを自分のリクエスト
/// にだけ使わせることができる——開発者が別途設定したグローバルな
/// キーがこの経路で消費されることは無い。
pub async fn search_with_credentials(query: &str, max_results: u8, api_key: &str, cx: &str) -> Result<Vec<SearchResult>> {
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
            ("key", api_key),
            ("cx", cx),
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
/// 検索結果をプロンプト埋め込み用の文脈テキストへ整形する。
///
/// 番号付き箇条書き(`1. Title: snippet`)にしているのは、GPT-2/distilgpt2の
/// ような指示追従ファインチューニング無しのモデルでも、事前学習コーパスに
/// 頻出する「番号付きリスト→それを参照した回答」というパターン補完に
/// 乗りやすくするため(`- `単純箇条書きより番号付きの方が、後続の
/// `Question: ... Answer:`形式との整合が取りやすい)。詳細は
/// CLAUDE.mdの2026-08-26追記(検索コンテキスト活用精度の改善)を参照。
pub fn format_results_as_context(results: &[SearchResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}. {}: {}", i + 1, r.title, r.snippet))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 検索結果コンテキストとユーザーの質問文から、GPT-2/distilgpt2の
/// 貪欲デコードで検索結果を踏まえた応答が出やすい形にプロンプトを組み立てる。
///
/// 単純な`"Reference information...\n{context}\n\n{prompt}"`連結
/// (旧実装)ではなく`"Search results: ... Question: ... Answer:"`という
/// QA形式にしているのは、GPT-2の事前学習コーパス(Webテキスト全般)に
/// この種のQ&A形式のテキストが大量に含まれているため、指示追従の
/// ファインチューニングが無いモデルでも「Answer:」の直後に検索結果を
/// 踏まえた続きを生成するパターン補完が起きやすいという狙い(仮説)。
/// **保証ではなく改善の試みである点に注意**——GPT-2系は依然として
/// 対話・指示追従のファインチューニングを受けていないため、この書式
/// 変更だけで検索結果の活用が保証されるわけではない。詳細は
/// CLAUDE.mdの2026-08-26追記(検索コンテキスト活用精度の改善)を参照。
pub fn build_search_augmented_prompt(context: &str, question: &str) -> String {
    format!(
        "Use the search results below to answer the question as accurately as possible. \
If the search results don't contain the answer, say so honestly.\n\n\
Search results:\n{context}\n\n\
Question: {question}\n\
Answer:"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `std::env::set_var`/`remove_var`はプロセス全体で共有される状態であり、
    /// `cargo test`はデフォルトで並列実行するため、環境変数を触るテスト同士が
    /// 競合するとフラーキーになる(2026-09-23実際に踏んだ: SerpApiキー用テストの
    /// 追加後、既存の`is_configured_false_when_env_vars_absent`が並列実行で
    /// 稀に失敗するようになった)。環境変数を操作するテストはこのロックを
    /// 取ってから行うことで、この2つが同時に走らないようにする。
    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn is_configured_false_when_env_vars_absent() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        // 実行環境の環境変数を汚さないよう、既存の値を保存・復元する。
        // 2026-09-23拡張: SerpApi/Tavily/Exaもis_configured()の判定対象に
        // なったため、同様に一時退避・復元する。
        let keys = [
            "ARUARU_LLM_GOOGLE_SEARCH_API_KEY",
            "ARUARU_LLM_GOOGLE_SEARCH_CX",
            "ARUARU_LLM_BRAVE_SEARCH_API_KEY",
            "ARUARU_LLM_SERPAPI_KEY",
            "ARUARU_LLM_TAVILY_KEY",
            "ARUARU_LLM_EXA_KEY",
            "ARUARU_LLM_JINA_KEY",
        ];
        let saved: Vec<Option<String>> = keys.iter().map(|k| std::env::var(k).ok()).collect();
        for k in keys {
            std::env::remove_var(k);
        }

        assert!(!is_configured());

        for (k, v) in keys.iter().zip(saved) {
            if let Some(v) = v {
                std::env::set_var(k, v);
            }
        }
    }

    #[test]
    fn format_results_as_context_joins_title_and_snippet() {
        let results = vec![
            SearchResult { title: "A".to_string(), snippet: "aaa".to_string(), link: "http://a".to_string() },
            SearchResult { title: "B".to_string(), snippet: "bbb".to_string(), link: "http://b".to_string() },
        ];
        let ctx = format_results_as_context(&results);
        assert_eq!(ctx, "1. A: aaa\n2. B: bbb");
    }

    #[test]
    fn build_search_augmented_prompt_uses_qa_format() {
        let prompt = build_search_augmented_prompt("1. A: aaa", "What is A?");
        assert!(prompt.contains("Search results:\n1. A: aaa"));
        assert!(prompt.contains("Question: What is A?"));
        assert!(prompt.trim_end().ends_with("Answer:"));
    }

    #[test]
    fn serpapi_response_parses_organic_results() {
        let json = r#"{"organic_results":[{"title":"T1","snippet":"S1","link":"http://a"},{"title":"T2","snippet":"S2","link":"http://b"}]}"#;
        let parsed: SerpApiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.organic_results.len(), 2);
        assert_eq!(parsed.organic_results[0].title, "T1");
        assert_eq!(parsed.organic_results[1].link, "http://b");
    }

    #[test]
    fn serpapi_response_missing_organic_results_defaults_to_empty() {
        let parsed: SerpApiResponse = serde_json::from_str("{}").unwrap();
        assert!(parsed.organic_results.is_empty());
    }

    #[test]
    fn tavily_response_parses_results() {
        let json = r#"{"results":[{"title":"T1","content":"C1","url":"http://a"},{"title":"T2","content":"C2","url":"http://b"}]}"#;
        let parsed: TavilyResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.results.len(), 2);
        assert_eq!(parsed.results[0].title, "T1");
        assert_eq!(parsed.results[1].url, "http://b");
    }

    #[test]
    fn tavily_response_missing_results_defaults_to_empty() {
        let parsed: TavilyResponse = serde_json::from_str("{}").unwrap();
        assert!(parsed.results.is_empty());
    }

    #[test]
    fn exa_response_parses_results() {
        let json = r#"{"results":[{"title":"T1","url":"http://a","highlights":["highlight one","highlight two"]}]}"#;
        let parsed: ExaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.results[0].highlights, vec!["highlight one", "highlight two"]);
    }

    #[test]
    fn exa_response_missing_results_defaults_to_empty() {
        let parsed: ExaResponse = serde_json::from_str("{}").unwrap();
        assert!(parsed.results.is_empty());
    }

    #[test]
    fn jina_response_parses_data_array() {
        let json = r#"{"code":200,"data":[{"title":"T1","url":"http://a","content":"C1"},{"title":"T2","url":"http://b","description":"D2"}]}"#;
        let parsed: JinaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data.len(), 2);
        assert_eq!(parsed.data[0].title, "T1");
        assert_eq!(parsed.data[1].description, "D2");
    }

    #[test]
    fn jina_response_missing_data_defaults_to_empty() {
        let parsed: JinaResponse = serde_json::from_str("{}").unwrap();
        assert!(parsed.data.is_empty());
    }

    #[test]
    fn consume_bucket_allows_up_to_limit_then_blocks() {
        let mut state = (0u64, 0u32);
        for _ in 0..3 {
            assert!(consume_bucket(&mut state, 100, 3));
        }
        assert!(!consume_bucket(&mut state, 100, 3), "4th call must be blocked once limit is reached");
    }

    #[test]
    fn consume_bucket_resets_when_bucket_changes() {
        let mut state = (100u64, 3u32); // 前の月(バケット100)で使い切った状態
        assert!(consume_bucket(&mut state, 101, 3), "a new bucket (new month) must start fresh even if the previous one was exhausted");
        assert_eq!(state, (101, 1));
    }

    #[test]
    fn is_configured_true_when_only_serpapi_key_present() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let keys = [
            "ARUARU_LLM_GOOGLE_SEARCH_API_KEY",
            "ARUARU_LLM_GOOGLE_SEARCH_CX",
            "ARUARU_LLM_BRAVE_SEARCH_API_KEY",
            "ARUARU_LLM_SERPAPI_KEY",
            "ARUARU_LLM_TAVILY_KEY",
            "ARUARU_LLM_EXA_KEY",
            "ARUARU_LLM_JINA_KEY",
        ];
        let saved: Vec<Option<String>> = keys.iter().map(|k| std::env::var(k).ok()).collect();
        for k in keys {
            std::env::remove_var(k);
        }
        std::env::set_var("ARUARU_LLM_SERPAPI_KEY", "dummy-key-for-test");

        assert!(is_configured());

        std::env::remove_var("ARUARU_LLM_SERPAPI_KEY");
        for (k, v) in keys.iter().zip(saved) {
            if let Some(v) = v {
                std::env::set_var(k, v);
            }
        }
    }
}
