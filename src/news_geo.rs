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
    /// このニュースをGoogle検索で取得した日時(Unix秒、2026-09-22追加)。
    /// **正直な開示**: これは記事自体の公開日時ではなく、**検索を実行した日時**。
    /// Google Custom Search JSON APIは記事の公開日を構造化データとして安定して
    /// 返さないため、実際に確認できる事実(いつ検索して得た情報か)だけを記録する
    /// (ユーザー指示「Google検索した日付と検索した結果にネットから得た情報にも
    /// 日付を付けてDATABASEで管理して」への対応)。`#[serde(default)]`により、
    /// 旧バージョンで保存されたこのフィールド無しの既存DATABASEも読み込める。
    #[serde(default)]
    pub retrieved_at_unix: u64,
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

/// 国ごとに、可能な限り現地のネイティブ言語でニュースを検索する
/// (2026-09-23拡張、ユーザー指示「現地のネイティブの言語のインターネット
/// ニュース...を網羅」への対応)。
///
/// **正直な開示・意図的に現地語を使わない国**: ユーザー自身の指示により、
/// インド・ウクライナ・イスラエルは(現地語での報道は存在するものの)
/// 英語のニュースを使う——インドは英語が事実上の全国共通語として広く
/// 使われているため、ウクライナは「英語版があれば」という条件付き指示、
/// イスラエルはヘブライ語が「難しくてマイナーすぎる」というユーザー自身の
/// 判断による。**北朝鮮は意図的に対象外**: 自由な報道機関が存在せず、
/// Google検索で得られる結果は事実上すべて国外(主に英語圏)からの報道に
/// なり、「北朝鮮の現地ニュース」として提示するのは誤解を招くため
/// (ユーザーに要相談、`daily-news-collect.sh`のコメント参照)。
/// スイスは公用語が独語・仏語・伊語・ロマンシュ語の4つあるが、話者数が
/// 最多のドイツ語を代表として使う簡略化であることも明記する。
pub(crate) fn news_query_for_country(country: &str) -> String {
    match country {
        "Japan" => "日本 ニュース 今日 主要".to_string(),
        "China" => "中国 新闻 今天 头条".to_string(),
        "Taiwan" => "台灣 新聞 今天 頭條".to_string(),
        "South Korea" => "한국 뉴스 오늘 주요".to_string(),
        "Philippines" => "Pilipinas balita ngayon pangunahing".to_string(),
        "Cambodia" => "កម្ពុជា ព័ត៌មាន ថ្ងៃនេះ".to_string(),
        "Thailand" => "ประเทศไทย ข่าว วันนี้".to_string(),
        "Malaysia" => "Malaysia berita hari ini utama".to_string(),
        "Germany" | "Austria" => "Deutschland Nachrichten heute wichtigste".to_string(),
        "Italy" => "Italia notizie oggi principali".to_string(),
        "France" => "France actualités aujourd'hui principales".to_string(),
        "Switzerland" => "Schweiz Nachrichten heute wichtigste".to_string(),
        "Russia" => "Россия новости сегодня главные".to_string(),
        "Brazil" => "Brasil notícias hoje principais".to_string(),
        "Myanmar" => "မြန်မာ သတင်း ယနေ့ အဓိက".to_string(),
        // 2026-09-23追加(ユーザー指示「ブラジルとミャンマーの毎日のネット
        // ニュースも現地語と英語と日本語も追加して」): この2ヶ国だけは
        // 現地語に加えて英語版・日本語版も別途収集する。既存の「1国=1
        // クエリ」というdata/news_by_country.jsonのキー設計を大きく変えず
        // 済むよう、"Brazil (English)"のような疑似国名をキーとして扱う
        // (daily-news-collect.shのCOUNTRIES配列に対応エントリを追加済み)。
        "Brazil (English)" | "Myanmar (English)" => {
            let base = country.split(" (").next().unwrap_or(country);
            format!("{base} news today headlines")
        }
        "Brazil (Japanese)" => "ブラジル ニュース 今日 主要".to_string(),
        "Myanmar (Japanese)" => "ミャンマー ニュース 今日 主要".to_string(),
        // India/Ukraine/Israel/United States/United Kingdom等はユーザー指示・
        // 実情により英語のまま(上記doc参照)。
        _ => format!("{country} news today headlines"),
    }
}

/// `fetch_for_country_cached`のキャッシュ有効期間。この間は同じ国への再検索をせず、
/// 保存済みのダイジェストをそのまま返す(2026-09-22追加、ユーザー指示「今日のニュースは？
/// の様なよくある質問などはGoogle検索後に重要と思える内容をダイジェストにしてDATABASE化
/// してそれを表示するようにして」)。
const NEWS_DIGEST_TTL_SECS: u64 = 3 * 3600;

fn news_by_country_db_path() -> std::path::PathBuf {
    std::env::var("ARUARU_LLM_NEWS_BY_COUNTRY_DB_PATH").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("data/news_by_country.json"))
}

static NEWS_BY_COUNTRY: RwLock<Option<std::collections::HashMap<String, NewsDb>>> = RwLock::new(None);

fn load_news_by_country() -> std::collections::HashMap<String, NewsDb> {
    {
        let guard = NEWS_BY_COUNTRY.read().expect("news-by-country lock poisoned");
        if let Some(map) = guard.as_ref() {
            return map.clone();
        }
    }
    let loaded = std::fs::read_to_string(news_by_country_db_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    *NEWS_BY_COUNTRY.write().expect("news-by-country lock poisoned") = Some(loaded);
    load_news_by_country() // 直前に書き込んだので次は必ずSomeから返る(再帰1回のみ)
}

fn save_news_by_country(country: &str, db: &NewsDb) {
    let mut map = load_news_by_country();
    map.insert(country.to_string(), db.clone());
    if let Some(parent) = news_by_country_db_path().parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&map) {
        let _ = std::fs::write(news_by_country_db_path(), json);
    }
    *NEWS_BY_COUNTRY.write().expect("news-by-country lock poisoned") = Some(map);
}

/// VPSのディスク容量を圧迫しないよう、この期間より古いエントリは生きているDATABASEから
/// 追い出す(2026-09-22追加、ユーザー指示「8日間以上前のニュース記事などは...GitHubへ
/// pushして...ストックして置いてそこにアクセスして」)。
const NEWS_ARCHIVE_AGE_SECS: u64 = 8 * 24 * 3600;

/// 「古くなった国別ニュースを、生きているDATABASEから取り除いてMarkdown形式で書き出す」
/// 純粋関数(副作用無し、テストしやすい形に分離)。戻り値は(残す新しいDB, 追い出した
/// (国名, NewsDb)の一覧)。
fn split_stale_entries(map: &std::collections::HashMap<String, NewsDb>, now: u64) -> (std::collections::HashMap<String, NewsDb>, Vec<(String, NewsDb)>) {
    let mut fresh = std::collections::HashMap::new();
    let mut stale = Vec::new();
    for (country, db) in map.clone() {
        let age = db.fetched_at_unix.map_or(u64::MAX, |t| now.saturating_sub(t));
        if age >= NEWS_ARCHIVE_AGE_SECS {
            stale.push((country, db));
        } else {
            fresh.insert(country, db);
        }
    }
    (fresh, stale)
}

/// 追い出したエントリをMarkdownの追記用ブロックへ整形する(2026-09-22追加)。
/// **正直な開示**: このMarkdownをGitHubへ実際にpush(コミット)する処理自体は、
/// このRustサーバー本体には実装していない(常時稼働するサーバープロセスへGitHub
/// 書き込み資格情報を持たせるリスクを避けるため)。この関数はアーカイブ内容の整形と
/// VPS側ローカルファイルへの追記までを行い、実際のGitHubへのcommit・pushは、
/// このリポジトリの他の変更と同じく開発者(Claude Code経由)が行う運用としている。
fn format_archive_markdown(entries: &[(String, NewsDb)], now: u64) -> String {
    let mut out = String::new();
    let date = format_unix_date(now);
    out.push_str(&format!("\n## アーカイブ日 / Archived on {date}\n\n"));
    for (country, db) in entries {
        let retrieved = db.fetched_at_unix.map(format_unix_date).unwrap_or_else(|| "unknown".to_string());
        out.push_str(&format!("### {country}(検索日時 / searched at: {retrieved})\n\n"));
        // 2026-09-22追加(ユーザー指示「GitHubを全文検索しないで良い用に、簡単なタグ分けや
        // カテゴリー分けを基本に行なっておいて」): 国名・年月(YYYY-MM)を単純なタグとして
        // 1行添える。GitHub上でファイル内検索(Ctrl+F)するだけで該当箇所へすぐ辿り着ける
        // ように、という簡易的な目的にとどめる(全文検索インデックス構築などは行わない)。
        out.push_str(&format!("Tags: `{country}` `{}`\n\n", &retrieved[..7.min(retrieved.len())]));
        if db.items.is_empty() {
            out.push_str("- (no items / 記事なし)\n");
        }
        for item in &db.items {
            out.push_str(&format!("- [{}]({}) — {}\n", item.title, item.link, item.snippet));
        }
        out.push('\n');
    }
    out
}

fn format_unix_date(secs: u64) -> String {
    // 依存クレートを増やさない簡易UTC日付変換(年月日のみ、時刻は省略)。
    let days = secs / 86_400;
    let (mut y, mut d) = (1970i64, days as i64);
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let year_len = if leap { 366 } else { 365 };
        if d < year_len {
            break;
        }
        d -= year_len;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_lens = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0usize;
    for (i, len) in month_lens.iter().enumerate() {
        if d < *len {
            m = i;
            break;
        }
        d -= len;
    }
    format!("{:04}-{:02}-{:02}", y, m + 1, d + 1)
}

fn news_archive_path() -> std::path::PathBuf {
    std::env::var("ARUARU_LLM_NEWS_ARCHIVE_PATH").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("data/news-archive-pending.md"))
}

/// 8日以上前のニュースを生きているDATABASEから追い出し、VPSローカルのMarkdown
/// アーカイブファイルへ追記する。戻り値は追い出した件数(0なら何もしなかった)。
pub fn prune_and_archive_stale_news() -> usize {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs()).unwrap_or(0);
    let map = load_news_by_country();
    let (fresh, stale) = split_stale_entries(&map, now);
    if stale.is_empty() {
        return 0;
    }
    let markdown = format_archive_markdown(&stale, now);
    let path = news_archive_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = f.write_all(markdown.as_bytes());
    }
    if let Some(parent) = news_by_country_db_path().parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&fresh) {
        let _ = std::fs::write(news_by_country_db_path(), json);
    }
    *NEWS_BY_COUNTRY.write().expect("news-by-country lock poisoned") = Some(fresh);
    stale.len()
}

/// 指定した国のニュースダイジェストを返す。`GET /v1/news/for`本体はこちらを使う
/// (2026-09-22変更): 保存済みのダイジェストが`NEWS_DIGEST_TTL_SECS`以内かつ
/// エラー無しで取得できていれば、それをそのままDATABASE(`data/news_by_country.json`)
/// から返し、検索は行わない。無い/古い/前回エラーだった場合のみ新たに検索し、
/// 結果(=「重要と思える内容」としてGoogle検索が返した上位数件のダイジェスト)を
/// DATABASEへ保存してから返す。共有無料枠(1日100件)の消費を、同じ国への
/// よくある質問(「今日のニュースは？」等)のたびに繰り返さないための仕組み。
pub async fn fetch_for_country_cached(country: &str) -> NewsDb {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs()).unwrap_or(0);
    {
        let map = load_news_by_country();
        if let Some(cached) = map.get(country) {
            let fresh = cached.fetched_at_unix.map_or(false, |t| now.saturating_sub(t) < NEWS_DIGEST_TTL_SECS);
            if fresh && cached.last_error.is_none() && !cached.items.is_empty() {
                return cached.clone();
            }
        }
    }
    let db = fetch_for_country(country).await;
    save_news_by_country(country, &db);
    db
}

/// 指定した国のニュースを、その場でGoogle Custom Searchして返す(2026-09-22新設)。
/// `refresh()`(サーバー接続先国を自動検出し、結果をディスクへ永続保存する定期処理)とは
/// 別に、open-english側から「日本語の質問なら日本のニュース、英語の質問ならアメリカの
/// ニュース」のように**利用者の言語に応じて国を指定**できるようにする(`GET /v1/news/for`)。
/// キャッシュ・DATABASE化は`fetch_for_country_cached`が担う——この関数自体は常に
/// 新たに検索する「その場限り」の下請け関数のまま。
pub async fn fetch_for_country(country: &str) -> NewsDb {
    let mut db = NewsDb { country: Some(CountryInfo { country: country.to_string(), country_code: String::new(), query_ip: String::new() }), ..Default::default() };
    if !web_search::is_configured() {
        db.last_error = Some(
            "Google Custom Search is not configured (set ARUARU_LLM_GOOGLE_SEARCH_API_KEY / \
             ARUARU_LLM_GOOGLE_SEARCH_CX) — no news fetched".to_string(),
        );
    } else {
        let query = news_query_for_country(country);
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs()).unwrap_or(0);
        match web_search::search(&query, 8).await {
            Ok(results) => {
                db.items = results.into_iter().map(|r: SearchResult| NewsItem { title: r.title, snippet: r.snippet, link: r.link, retrieved_at_unix: now }).collect();
            }
            Err(e) => {
                db.last_error = Some(format!("news search failed: {e}"));
            }
        }
    }
    db.fetched_at_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs());
    db
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
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs()).unwrap_or(0);
            match web_search::search(&query, 8).await {
                Ok(results) => {
                    db.items = results.into_iter().map(|r: SearchResult| NewsItem { title: r.title, snippet: r.snippet, link: r.link, retrieved_at_unix: now }).collect();
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
    fn news_query_for_country_uses_english_for_unlisted_countries() {
        assert_eq!(news_query_for_country("United States"), "United States news today headlines");
    }

    #[test]
    fn news_query_for_country_uses_english_for_india_ukraine_israel_by_user_instruction() {
        // ユーザー自身の指示により、これら3ヶ国は意図的に現地語を使わない
        // (news_query_for_countryのdocコメント参照)。
        for country in ["India", "Ukraine", "Israel"] {
            assert_eq!(news_query_for_country(country), format!("{country} news today headlines"));
        }
    }

    #[test]
    fn news_query_for_country_uses_native_language_for_covered_countries() {
        assert_eq!(news_query_for_country("France"), "France actualités aujourd'hui principales");
        assert_eq!(news_query_for_country("China"), "中国 新闻 今天 头条");
        assert_eq!(news_query_for_country("Taiwan"), "台灣 新聞 今天 頭條");
        assert_eq!(news_query_for_country("South Korea"), "한국 뉴스 오늘 주요");
        assert_eq!(news_query_for_country("Thailand"), "ประเทศไทย ข่าว วันนี้");
        assert_eq!(news_query_for_country("Russia"), "Россия новости сегодня главные");
        // Germany/Austriaは同じドイツ語クエリを共有する。
        assert_eq!(news_query_for_country("Germany"), news_query_for_country("Austria"));
    }

    #[test]
    fn news_query_for_country_covers_brazil_and_myanmar_in_three_languages() {
        assert_eq!(news_query_for_country("Brazil"), "Brasil notícias hoje principais");
        assert_eq!(news_query_for_country("Brazil (English)"), "Brazil news today headlines");
        assert_eq!(news_query_for_country("Brazil (Japanese)"), "ブラジル ニュース 今日 主要");
        assert_eq!(news_query_for_country("Myanmar"), "မြန်မာ သတင်း ယနေ့ အဓိက");
        assert_eq!(news_query_for_country("Myanmar (English)"), "Myanmar news today headlines");
        assert_eq!(news_query_for_country("Myanmar (Japanese)"), "ミャンマー ニュース 今日 主要");
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
                NewsItem { title: "A".to_string(), snippet: "".to_string(), link: "".to_string(), retrieved_at_unix: 0 },
                NewsItem { title: "B".to_string(), snippet: "".to_string(), link: "".to_string(), retrieved_at_unix: 0 },
                NewsItem { title: "C".to_string(), snippet: "".to_string(), link: "".to_string(), retrieved_at_unix: 0 },
            ],
            fetched_at_unix: None,
            last_error: None,
        };
        assert_eq!(topic_context_line(&db).unwrap(), "Recent news from Japan: A / B");
    }

    /// 2026-09-22追加(ユーザー指示「よくある質問はGoogle検索後にダイジェストにして
    /// DATABASE化してそれを表示する」): 保存→読み込みが往復し、新しいエントリが
    /// TTL内は「新鮮」、TTLを過ぎれば「古い」と判定されることを確認する。
    #[test]
    fn news_by_country_cache_round_trips_and_respects_ttl() {
        let path = std::env::temp_dir().join(format!("aruaru_llm_news_by_country_test_{}.json", std::process::id()));
        std::env::set_var("ARUARU_LLM_NEWS_BY_COUNTRY_DB_PATH", &path);
        *NEWS_BY_COUNTRY.write().expect("lock") = None; // 前のテスト/実行のキャッシュを忘れさせる

        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let fresh_db = NewsDb {
            country: Some(CountryInfo { country: "Japan".to_string(), country_code: "JP".to_string(), query_ip: String::new() }),
            items: vec![NewsItem { title: "Fresh headline".to_string(), snippet: "".to_string(), link: "".to_string(), retrieved_at_unix: now }],
            fetched_at_unix: Some(now),
            last_error: None,
        };
        save_news_by_country("Japan", &fresh_db);
        let loaded = load_news_by_country();
        assert_eq!(loaded.get("Japan").unwrap().items[0].title, "Fresh headline");

        let stale_db = NewsDb {
            fetched_at_unix: Some(now.saturating_sub(NEWS_DIGEST_TTL_SECS + 60)),
            ..fresh_db.clone()
        };
        save_news_by_country("Japan", &stale_db);
        let map = load_news_by_country();
        let cached = map.get("Japan").unwrap();
        let is_fresh = cached.fetched_at_unix.map_or(false, |t| now.saturating_sub(t) < NEWS_DIGEST_TTL_SECS);
        assert!(!is_fresh, "an entry older than the TTL must be treated as stale");

        std::env::remove_var("ARUARU_LLM_NEWS_BY_COUNTRY_DB_PATH");
        let _ = std::fs::remove_file(&path);
        *NEWS_BY_COUNTRY.write().expect("lock") = None;
    }

    /// 2026-09-22追加(ユーザー指示「8日間以上前のニュース記事などは...GitHubへpushして
    /// ...ストックして置いて」): 8日以上前のエントリだけが追い出され、8日未満は
    /// 生きているDATABASEに残ることを確認する。
    #[test]
    fn split_stale_entries_uses_eight_day_boundary() {
        let now = 1_000_000_000u64;
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Japan".to_string(),
            NewsDb { country: None, items: vec![], fetched_at_unix: Some(now - NEWS_ARCHIVE_AGE_SECS - 1), last_error: None },
        );
        map.insert(
            "United States".to_string(),
            NewsDb { country: None, items: vec![], fetched_at_unix: Some(now - NEWS_ARCHIVE_AGE_SECS + 1), last_error: None },
        );
        let (fresh, stale) = split_stale_entries(&map, now);
        assert_eq!(fresh.len(), 1, "an entry just under 8 days old must stay in the live DB");
        assert!(fresh.contains_key("United States"));
        assert_eq!(stale.len(), 1, "an entry over 8 days old must be archived");
        assert_eq!(stale[0].0, "Japan");
    }

    #[test]
    fn format_archive_markdown_includes_country_date_and_items() {
        let entries = vec![(
            "Japan".to_string(),
            NewsDb {
                country: None,
                items: vec![NewsItem { title: "Headline".to_string(), snippet: "Snippet text".to_string(), link: "https://example.com".to_string(), retrieved_at_unix: 1_000_000_000 }],
                fetched_at_unix: Some(1_000_000_000),
                last_error: None,
            },
        )];
        let md = format_archive_markdown(&entries, 1_000_100_000);
        assert!(md.contains("Japan"));
        assert!(md.contains("Headline"));
        assert!(md.contains("https://example.com"));
        assert!(md.contains("Snippet text"));
    }

    #[test]
    fn format_unix_date_matches_known_dates() {
        assert_eq!(format_unix_date(0), "1970-01-01");
        assert_eq!(format_unix_date(1_700_000_000), "2023-11-14");
    }
}
