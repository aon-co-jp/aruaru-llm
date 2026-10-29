//! 「今日のニュース」「最近のニュース」「一週間の出来事」などの質問への直接回答
//! (ユーザー指示 2026-10-29「maid-cafe.tokyo で今日のニュース/ニュース/最近のニュース/
//! 一週間の出来事 などの質問と類似語に対応する機能。英語でも対応」)。
//!
//! ## 設計
//! - 実体は公開RSS(Yahoo!ニュース トピックス、BBC News、The Japan Times、The New York Times)。
//!   見出しと元記事へのリンク・公開日時だけを扱い、記事本文は複製しない。
//! - 30分ごとに取得して`data/news-feed-history.json`へ**蓄積**する(14日分)。RSSは直近の
//!   数件〜数十件しか返さないため、「一週間の出来事」は蓄積があって初めて意味を持つ。
//!   記録開始より前の期間は回答に正直に注記する。
//! - AIを通さず、見出しの一覧をそのまま返す(`knowledge::verbatim_for`と同じ方針)。
//! - 誤爆防止: 質問が短く(50文字以内)、ニュース系の語を含み、「記事」「article」等の
//!   記事の解説依頼を示す語を含まないときだけ反応する(英語学習の会話を奪わないため)。

use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const KEEP_SECS: u64 = 14 * 24 * 3600;
const MAX_ITEMS: usize = 4000;
const REFRESH_EVERY: Duration = Duration::from_secs(30 * 60);
const ON_DEMAND_STALE: Duration = Duration::from_secs(10 * 60);

struct Feed {
    lang: &'static str,
    source: &'static str,
    category: &'static str,
    url: &'static str,
}

const FEEDS: &[Feed] = &[
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "主要", url: "https://news.yahoo.co.jp/rss/topics/top-picks.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "国内", url: "https://news.yahoo.co.jp/rss/topics/domestic.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "国際", url: "https://news.yahoo.co.jp/rss/topics/world.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "経済", url: "https://news.yahoo.co.jp/rss/topics/business.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "エンタメ", url: "https://news.yahoo.co.jp/rss/topics/entertainment.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "スポーツ", url: "https://news.yahoo.co.jp/rss/topics/sports.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "IT", url: "https://news.yahoo.co.jp/rss/topics/it.xml" },
    Feed { lang: "ja", source: "Yahoo!ニュース", category: "科学", url: "https://news.yahoo.co.jp/rss/topics/science.xml" },
    Feed { lang: "en", source: "BBC News", category: "Top stories", url: "https://feeds.bbci.co.uk/news/rss.xml" },
    Feed { lang: "en", source: "BBC News", category: "World", url: "https://feeds.bbci.co.uk/news/world/rss.xml" },
    Feed { lang: "en", source: "BBC News", category: "Business", url: "https://feeds.bbci.co.uk/news/business/rss.xml" },
    Feed { lang: "en", source: "BBC News", category: "Technology", url: "https://feeds.bbci.co.uk/news/technology/rss.xml" },
    Feed { lang: "en", source: "The Japan Times", category: "Japan", url: "https://www.japantimes.co.jp/feed/" },
    Feed { lang: "en", source: "The New York Times", category: "World (NYT)", url: "https://rss.nytimes.com/services/xml/rss/nyt/World.xml" },
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Item {
    lang: String,
    source: String,
    category: String,
    title: String,
    link: String,
    /// 記事の公開日時(Unix秒)。RSSに無い/読めないときは取得時刻。
    ts: u64,
}

#[derive(Default)]
struct State {
    items: Vec<Item>,
    loaded: bool,
    last_refresh: Option<Instant>,
    /// 蓄積を始めた日時(最古の`ts`ではなく、このストアが空から始まった時刻)。
    since: u64,
}

static STATE: Mutex<State> = Mutex::new(State { items: Vec::new(), loaded: false, last_refresh: None, since: 0 });

/// 長期アーカイブ(主要カテゴリの見出しだけを永続的に残す。月間・年間のまとめ用)。
static ARCHIVE: Mutex<(bool, Vec<Item>)> = Mutex::new((false, Vec::new()));
const ARCHIVE_MAX: usize = 40_000;
const ARCHIVE_CATS: &[&str] = &["主要", "国際", "Top stories", "World", "Japan"];

fn archive_path() -> std::path::PathBuf {
    std::env::var("ARUARU_LLM_NEWS_ARCHIVE_PATH").map(Into::into).unwrap_or_else(|_| "data/news-feed-archive.json".into())
}

fn archive_load(a: &mut (bool, Vec<Item>)) {
    if !a.0 {
        a.0 = true;
        a.1 = std::fs::read_to_string(archive_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    }
}

fn archive_add(fresh: &[Item]) {
    let mut a = ARCHIVE.lock().unwrap_or_else(|e| e.into_inner());
    archive_load(&mut a);
    let seen: std::collections::HashSet<String> = a.1.iter().map(|i| i.link.clone()).collect();
    let mut changed = false;
    for it in fresh.iter().filter(|i| ARCHIVE_CATS.contains(&i.category.as_str())) {
        if !seen.contains(&it.link) {
            a.1.push(it.clone());
            changed = true;
        }
    }
    if a.1.len() > ARCHIVE_MAX {
        a.1.sort_by(|x, y| x.ts.cmp(&y.ts));
        let cut = a.1.len() - ARCHIVE_MAX;
        a.1.drain(..cut);
    }
    if changed {
        if let Ok(json) = serde_json::to_string(&a.1) {
            let p = archive_path();
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let tmp = p.with_extension("json.tmp");
            if std::fs::write(&tmp, json).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }
}

fn archive_snapshot() -> Vec<Item> {
    let mut a = ARCHIVE.lock().unwrap_or_else(|e| e.into_inner());
    archive_load(&mut a);
    a.1.clone()
}

#[derive(Serialize, Deserialize, Default)]
struct Disk {
    since: u64,
    items: Vec<Item>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn path() -> std::path::PathBuf {
    std::env::var("ARUARU_LLM_NEWS_FEED_PATH").map(Into::into).unwrap_or_else(|_| "data/news-feed-history.json".into())
}

fn ensure_loaded(st: &mut State) {
    if st.loaded {
        return;
    }
    st.loaded = true;
    if let Some(d) = std::fs::read_to_string(path()).ok().and_then(|s| serde_json::from_str::<Disk>(&s).ok()) {
        st.items = d.items;
        st.since = d.since;
    }
    if st.since == 0 {
        st.since = now();
    }
}

fn persist(st: &State) {
    let d = Disk { since: st.since, items: st.items.clone() };
    if let Ok(json) = serde_json::to_string(&d) {
        let p = path();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = p.with_extension("json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, &p);
        }
    }
}

// ---------- RSSの解析(依存を増やさない簡易版) ----------

fn unescape(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("<![CDATA[").and_then(|x| x.strip_suffix("]]>")).unwrap_or(s);
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&#x27;", "'")
        .replace("&amp;", "&")
        .trim()
        .to_string()
}

fn tag<'a>(block: &'a str, name: &str) -> Option<&'a str> {
    let open = format!("<{name}");
    let start = block.find(&open)?;
    let after = &block[start + open.len()..];
    let gt = after.find('>')?;
    let body = &after[gt + 1..];
    let end = body.find(&format!("</{name}>"))?;
    Some(&body[..end])
}

/// `Thu, 29 Oct 2026 14:47:34 +0900` / `... GMT` → Unix秒。
fn parse_rfc2822(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.split_once(',').map(|(_, r)| r).unwrap_or(s).trim();
    let mut it = s.split_whitespace();
    let day: i64 = it.next()?.parse().ok()?;
    let mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        .iter()
        .position(|m| it.clone().next().is_some_and(|x| x.eq_ignore_ascii_case(m)))? as i64
        + 1;
    it.next();
    let year: i64 = it.next()?.parse().ok()?;
    let mut hms = it.next()?.split(':');
    let (h, mi): (i64, i64) = (hms.next()?.parse().ok()?, hms.next()?.parse().ok()?);
    let sec: i64 = hms.next().and_then(|x| x.parse().ok()).unwrap_or(0);
    let tz = it.next().unwrap_or("GMT");
    let off: i64 = if tz.len() >= 5 && (tz.starts_with('+') || tz.starts_with('-')) {
        let sign = if tz.starts_with('-') { -1 } else { 1 };
        let (hh, mm): (i64, i64) = (tz[1..3].parse().ok()?, tz[3..5].parse().ok()?);
        sign * (hh * 3600 + mm * 60)
    } else {
        0
    };
    // days_from_civil
    let (y, m) = if mon <= 2 { (year - 1, mon + 9) } else { (year, mon - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let t = days * 86400 + h * 3600 + mi * 60 + sec - off;
    u64::try_from(t).ok()
}

fn parse_feed(xml: &str, f: &Feed, fetched: u64) -> Vec<Item> {
    xml.split("<item").skip(1)
        .filter_map(|chunk| {
            let block = chunk.split("</item>").next()?;
            let title = unescape(tag(block, "title")?);
            let link = unescape(tag(block, "link")?);
            if title.is_empty() || !link.starts_with("http") {
                return None;
            }
            let ts = tag(block, "pubDate").and_then(|d| parse_rfc2822(&unescape(d))).unwrap_or(fetched);
            Some(Item { lang: f.lang.into(), source: f.source.into(), category: f.category.into(), title, link, ts })
        })
        .collect()
}

async fn fetch_all() -> Vec<Item> {
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(10)).user_agent("Mozilla/5.0 (aruaru-llm news-feed)").build() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let fetched = now();
    let tasks = FEEDS.iter().map(|f| {
        let client = client.clone();
        async move {
            match client.get(f.url).send().await {
                Ok(r) if r.status().is_success() => r.text().await.map(|t| parse_feed(&t, f, fetched)).unwrap_or_default(),
                _ => Vec::new(),
            }
        }
    });
    futures_join(tasks).await
}

async fn futures_join<F: std::future::Future<Output = Vec<Item>> + Send + 'static>(tasks: impl Iterator<Item = F>) -> Vec<Item> {
    let handles: Vec<_> = tasks.map(tokio::spawn).collect();
    let mut out = Vec::new();
    for h in handles {
        if let Ok(v) = h.await {
            out.extend(v);
        }
    }
    out
}

/// 取得して蓄積する(失敗した配信元は黙って飛ばす)。
pub async fn refresh() -> usize {
    let fresh = fetch_all().await;
    let n = fresh.len();
    archive_add(&fresh);
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    ensure_loaded(&mut st);
    let seen: std::collections::HashSet<(String, String)> = st.items.iter().map(|i| (i.category.clone(), i.link.clone())).collect();
    let mut added = 0;
    for it in fresh {
        if !seen.contains(&(it.category.clone(), it.link.clone())) {
            st.items.push(it);
            added += 1;
        }
    }
    let cutoff = now().saturating_sub(KEEP_SECS);
    st.items.retain(|i| i.ts >= cutoff);
    if st.items.len() > MAX_ITEMS {
        st.items.sort_by(|a, b| b.ts.cmp(&a.ts));
        st.items.truncate(MAX_ITEMS);
    }
    st.last_refresh = Some(Instant::now());
    if n > 0 {
        persist(&st);
    }
    added
}

pub fn spawn_background_refresh() {
    tokio::spawn(async {
        loop {
            let added = refresh().await;
            tracing::info!("news-feed: refreshed, {added} new item(s)");
            tokio::time::sleep(REFRESH_EVERY).await;
        }
    });
}

// ---------- 質問の判定 ----------

#[derive(Debug, PartialEq)]
pub struct Intent {
    pub ja: bool,
    pub hours: u64,
    pub label_ja: &'static str,
    pub label_en: &'static str,
    /// true なら「今月」(その月の1日0時JSTから)。false なら now - hours。
    pub this_month: bool,
}

fn has_cjk(s: &str) -> bool {
    s.chars().any(|c| matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xFF66..=0xFF9F))
}

pub fn intent(prompt: &str) -> Option<Intent> {
    let q = prompt.rsplit_once("Student:").map(|(_, s)| s).unwrap_or(prompt).trim();
    if q.is_empty() || q.chars().count() > 50 {
        return None;
    }
    let l = q.to_lowercase();
    let ja = has_cjk(q);
    let kw_ja = ["ニュース", "出来事", "できごと", "時事", "速報", "トピックス", "報道"];
    let kw_en = ["news", "headline", "what happened", "what's happening", "whats happening", "current events", "breaking"];
    let hit = if ja { kw_ja.iter().any(|k| q.contains(k)) } else { kw_en.iter().any(|k| l.contains(k)) };
    if !hit {
        return None;
    }
    // 記事そのものの解説・翻訳依頼などは奪わない。
    let skip = ["記事", "解説", "説明して", "翻訳", "訳して", "article", "explain", "translate", "meaning", "grammar", "vocabulary"];
    if skip.iter().any(|k| l.contains(&k.to_lowercase())) {
        return None;
    }
    let any = |ws: &[&str]| ws.iter().any(|w| l.contains(&w.to_lowercase()));
    let (hours, label_ja, label_en, this_month) = if any(&["今月", "this month", "of the month", "monthly"]) {
        (31 * 24, "今月のニュース", "This month's news", true)
    } else if any(&["一ヶ月", "1ヶ月", "一か月", "1か月", "一カ月", "1カ月", "一箇月", "1箇月", "ひと月", "30日", "past month", "last month", "a month", "one month", "30 days"]) {
        (30 * 24, "この1か月のニュース", "News from the past month", false)
    } else if any(&["一週間", "1週間", "今週", "週間", "先週", "this week", "past week", "last week", "7 days", "weekly", "the week"]) {
        (7 * 24, "一週間のニュース", "News from the past week", false)
    } else if any(&["今日", "本日", "きょう", "today", "tonight", "this morning"]) {
        (30, "今日のニュース", "Today's news", false)
    } else if any(&["最近", "最新", "直近", "latest", "recent", "lately", "newest"]) {
        (72, "最近のニュース", "Recent news", false)
    } else {
        (48, "最新のニュース", "Latest news", false)
    };
    Some(Intent { ja, hours, label_ja, label_en, this_month })
}

// ---------- 回答の組み立て ----------

/// Unix秒 → JST(+9)の(年, 月, 日, 時, 分)。
fn civil(ts: u64) -> (i64, i64, i64, i64, i64) {
    let t = ts as i64 + 9 * 3600;
    let days = t.div_euclid(86400);
    let secs = t.rem_euclid(86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d, secs / 3600, secs % 3600 / 60)
}

/// JSTのその月1日0時のUnix秒。
fn month_start(ts: u64) -> u64 {
    let (y, m, _, _, _) = civil(ts);
    let (yy, mm) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let doy = (153 * mm + 2) / 5;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    (days * 86400 - 9 * 3600).max(0) as u64
}

fn fmt_time(ts: u64, ja: bool) -> String {
    let (_, m, d, hh, mm) = civil(ts);
    if ja {
        format!("{m}/{d} {hh:02}:{mm:02}")
    } else {
        let name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][(m - 1) as usize];
        format!("{name} {d} {hh:02}:{mm:02} JST")
    }
}

pub fn build_answer(intent: &Intent, items: &[Item], since: u64, now_ts: u64) -> String {
    let ja = intent.ja;
    let start = if intent.this_month { month_start(now_ts) } else { now_ts.saturating_sub(intent.hours * 3600) };
    let per_cat = if intent.hours > 72 { 6 } else if intent.hours > 36 { 5 } else { 4 };
    let lang = if ja { "ja" } else { "en" };
    let mut cats: Vec<&str> = FEEDS.iter().filter(|f| f.lang == lang).map(|f| f.category).collect();
    cats.dedup();
    let mut out = String::new();
    out.push_str(&format!(
        "📰 {}{}\n",
        if ja { intent.label_ja } else { intent.label_en },
        if ja { format!("({} 時点)", fmt_time(now_ts, true)) } else { format!(" (as of {})", fmt_time(now_ts, false)) }
    ));
    let mut total = 0;
    let mut seen = std::collections::HashSet::new();
    for cat in cats {
        let mut v: Vec<&Item> = items.iter().filter(|i| i.lang == lang && i.category == cat && i.ts >= start).collect();
        v.sort_by(|a, b| b.ts.cmp(&a.ts));
        let v: Vec<&Item> = v.into_iter().filter(|i| seen.insert(i.title.clone())).take(per_cat).collect();
        if v.is_empty() {
            continue;
        }
        out.push_str(&format!("\n■ {cat}\n"));
        for i in v {
            total += 1;
            out.push_str(&format!("・{}({})\n  {}\n", i.title, fmt_time(i.ts, ja), i.link));
        }
    }
    if total == 0 {
        return if ja {
            "申し訳ありません。いまニュースの見出しを取得できませんでした。しばらくしてからもう一度お試しください。".to_string()
        } else {
            "Sorry, I could not fetch news headlines right now. Please try again in a little while.".to_string()
        };
    }
    let srcs: Vec<&str> = {
        let mut s: Vec<&str> = FEEDS.iter().filter(|f| f.lang == lang).map(|f| f.source).collect();
        s.dedup();
        s
    };
    if ja {
        out.push_str(&format!("\n出典: {}(RSS)。見出しと元記事へのリンクのみを表示しています。", srcs.join("・")));
    } else {
        out.push_str(&format!("\nSources: {} (RSS). Only headlines and links to the original articles are shown.", srcs.join(", ")));
    }
    if since > start {
        let d = fmt_time(since, ja);
        if ja {
            out.push_str(&format!("\n※この機能は{d}から見出しの記録を始めたため、それ以前の分は含まれません(一週間分に満たない場合があります)。"));
        } else {
            out.push_str(&format!("\nNote: headline recording started on {d}, so earlier items are not included (a full week may not be available yet)."));
        }
    }
    out
}

// ---------- 年間ダイジェスト ----------

#[derive(Debug, PartialEq)]
pub struct YearIntent {
    pub ja: bool,
    pub year: i64,
}

pub fn year_intent(prompt: &str, this_year: i64) -> Option<YearIntent> {
    let q = prompt.rsplit_once("Student:").map(|(_, s)| s).unwrap_or(prompt).trim();
    if q.is_empty() || q.chars().count() > 60 {
        return None;
    }
    let l = q.to_lowercase();
    let ja = has_cjk(q);
    let skip = ["記事", "解説", "翻訳", "article", "explain", "translate", "grammar"];
    if skip.iter().any(|k| l.contains(&k.to_lowercase())) {
        return None;
    }
    let year = q
        .as_bytes()
        .windows(4)
        .filter_map(|w| std::str::from_utf8(w).ok().and_then(|x| x.parse::<i64>().ok()))
        .find(|y| (2000..=2100).contains(y));
    let kw_ja = ["一年間", "1年間", "年間", "振り返", "振返", "振り替", "振替", "年末", "今年", "去年", "昨年", "年の出来事", "年のニュース"];
    let kw_en = ["year in review", "this year", "the year", "year-end", "year end", "recap", "last year", "annual"];
    let topic_ja = ["ニュース", "出来事", "できごと", "振り返", "振返", "振り替", "振替", "まとめ", "ダイジェスト"];
    let topic_en = ["news", "events", "happened", "review", "recap", "digest", "highlights"];
    let ok = if ja {
        (kw_ja.iter().any(|k| q.contains(k)) || year.is_some()) && topic_ja.iter().any(|k| q.contains(k))
    } else {
        kw_en.iter().any(|k| l.contains(k)) && topic_en.iter().any(|k| l.contains(k))
    };
    if !ok {
        return None;
    }
    // 「今月」「今週」「今日」など短い期間の語が入っていれば、年間ではなくそちらを優先する。
    let short = ["今月", "今週", "今日", "本日", "this month", "this week", "today"];
    if year.is_none() && short.iter().any(|k| l.contains(k)) && !q.contains("年間") {
        return None;
    }
    let y = if l.contains("last year") || q.contains("去年") || q.contains("昨年") { this_year - 1 } else { year.unwrap_or(this_year) };
    Some(YearIntent { ja, year: y })
}

fn spread<T: Clone>(v: &[T], n: usize) -> Vec<T> {
    if v.len() <= n {
        return v.to_vec();
    }
    (0..n).map(|i| v[i * (v.len() - 1) / (n - 1).max(1)].clone()).collect()
}

pub fn build_year_digest(yi: &YearIntent, archive: &[Item], stock: &[crate::news_stock::DigestMonth], since: u64) -> String {
    let ja = yi.ja;
    let lang = if ja { "ja" } else { "en" };
    let mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let mut out = String::new();
    out.push_str(&if ja { format!("📚 {}年のニュースの索引ダイジェスト\n", yi.year) } else { format!("📚 News index digest for {}\n", yi.year) });
    let mut any = false;
    for m in 1..=12i64 {
        let mut v: Vec<&Item> = archive.iter().filter(|i| i.lang == lang && civil(i.ts).0 == yi.year && civil(i.ts).1 == m).collect();
        let sk: Vec<&crate::news_stock::DigestMonth> = stock.iter().filter(|d| d.year == yi.year && d.month == m).collect();
        if v.is_empty() && sk.is_empty() {
            continue;
        }
        any = true;
        v.sort_by_key(|i| i.ts);
        out.push_str(&match (ja, v.is_empty()) {
            (true, false) => format!("\n■ {}年{}月(記録した主要ニュース {} 件)\n", yi.year, m, v.len()),
            (true, true) => format!("\n■ {}年{}月\n", yi.year, m),
            (false, false) => format!("\n■ {} {} ({} main headlines recorded)\n", mon[(m - 1) as usize], yi.year, v.len()),
            (false, true) => format!("\n■ {} {}\n", mon[(m - 1) as usize], yi.year),
        });
        for i in spread(&v, 8) {
            let (_, mm, d, _, _) = civil(i.ts);
            out.push_str(&format!("・{mm}/{d} {}\n  {}\n", i.title, i.link));
        }
        for d in sk {
            out.push_str(&if ja {
                format!("◇ ストックしている国別ニュース検索の索引: {} 件\n", d.total)
            } else {
                format!("◇ Stocked country news search index: {} items\n", d.total)
            });
            for (country, date, title, link) in &d.picks {
                out.push_str(&format!("・[{country} {date}] {title}\n  {link}\n"));
            }
        }
    }
    if !any {
        out.push_str(&if ja { format!("\n{}年のニュースの記録は、まだありません。\n", yi.year) } else { format!("\nThere are no recorded news items for {} yet.\n", yi.year) });
    }
    let d = fmt_time(since, ja);
    if ja {
        out.push_str(&format!("\n※見出しの記録は{d}から始めました(毎日蓄積されるため、年末に近づくほど充実します)。それ以前は、国別ニュース検索のストック(2026年9月下旬〜)の索引のみです。リンクをクリックすると元の記事を読めます。"));
    } else {
        out.push_str(&format!("\nNote: headline recording started on {d} and grows every day, so this digest becomes richer toward the end of the year. Before that, only the stocked country-news search index (from late September 2026) is available. Click a link to read the original article."));
    }
    out
}

/// 質問がニュース系なら、見出し一覧の回答を返す。そうでなければ`None`。
pub async fn answer_for(prompt: &str) -> Option<String> {
    let this_year = civil(now()).0;
    if let Some(yi) = year_intent(prompt, this_year) {
        {
            let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
            ensure_loaded(&mut st);
        }
        let since = STATE.lock().unwrap_or_else(|e| e.into_inner()).since;
        let archive = archive_snapshot();
        let stock = crate::news_stock::digest_by_month(yi.year);
        return Some(build_year_digest(&yi, &archive, &stock, since));
    }
    let intent = intent(prompt)?;
    let stale = {
        let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
        ensure_loaded(&mut st);
        st.items.is_empty() || st.last_refresh.is_some_and(|t| t.elapsed() >= ON_DEMAND_STALE * 3)
    };
    if stale {
        let _ = tokio::time::timeout(Duration::from_secs(20), refresh()).await;
    }
    let archive = if intent.hours > 14 * 24 || intent.this_month { archive_snapshot() } else { Vec::new() };
    let st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let mut items = st.items.clone();
    let seen: std::collections::HashSet<(String, String)> = items.iter().map(|i| (i.category.clone(), i.link.clone())).collect();
    items.extend(archive.into_iter().filter(|i| !seen.contains(&(i.category.clone(), i.link.clone()))));
    Some(build_answer(&intent, &items, st.since, now()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_japanese_and_english_news_questions() {
        let t = |s: &str| intent(s).map(|i| (i.ja, i.hours, i.this_month));
        assert_eq!(t("今日のニュース"), Some((true, 30, false)));
        assert_eq!(t("最近のニュースを教えて"), Some((true, 72, false)));
        assert_eq!(t("一週間の出来事"), Some((true, 168, false)));
        assert_eq!(t("ニュース"), Some((true, 48, false)));
        assert_eq!(t("今日の速報"), Some((true, 30, false)));
        assert_eq!(t("今月のニュース"), Some((true, 744, true)));
        assert_eq!(t("一ヶ月のニュース"), Some((true, 720, false)));
        assert_eq!(t("1か月の出来事"), Some((true, 720, false)));
        assert_eq!(t("What's the news today?"), Some((false, 30, false)));
        assert_eq!(t("tell me the latest headlines"), Some((false, 72, false)));
        assert_eq!(t("what happened this week"), Some((false, 168, false)));
        assert_eq!(t("news this month"), Some((false, 744, true)));
        assert_eq!(t("news from the past month"), Some((false, 720, false)));
        assert_eq!(t("news"), Some((false, 48, false)));
        assert_eq!(t("この記事を解説して ニュース"), None);
        assert_eq!(t("Please explain this news article"), None);
        assert_eq!(t("how do I order coffee"), None);
        assert_eq!(t("こんにちは"), None);
    }

    #[test]
    fn detects_year_review() {
        let t = |s: &str| year_intent(s, 2026).map(|y| (y.ja, y.year));
        assert_eq!(t("1年間のニュース"), Some((true, 2026)));
        assert_eq!(t("一年間を振り返って"), Some((true, 2026)));
        assert_eq!(t("2026年の出来事"), Some((true, 2026)));
        assert_eq!(t("今年を振り返って"), Some((true, 2026)));
        assert_eq!(t("2025年のニュース"), Some((true, 2025)));
        assert_eq!(t("year in review"), Some((false, 2026)));
        assert_eq!(t("news recap of this year"), Some((false, 2026)));
        assert_eq!(t("今日のニュース"), None);
        assert_eq!(t("今月のニュース"), None);
        assert_eq!(t("how was your year"), None);
    }

    #[test]
    fn rfc2822_parses() {
        assert_eq!(parse_rfc2822("Thu, 29 Oct 2026 14:47:34 +0900"), parse_rfc2822("Thu, 29 Oct 2026 05:47:34 GMT"));
        let ts = parse_rfc2822("Thu, 29 Oct 2026 05:47:34 GMT").unwrap();
        assert_eq!(civil(ts), (2026, 10, 29, 14, 47));
        assert_eq!(civil(month_start(ts)), (2026, 10, 1, 0, 0));
    }

    #[test]
    fn parses_rss_with_cdata_and_entities() {
        let f = &FEEDS[0];
        let xml = "<rss><channel><item><title><![CDATA[A &amp; B]]></title><link>https://x/1</link><pubDate>Thu, 29 Oct 2026 05:48:44 GMT</pubDate></item><item><title>T2 &amp; more</title><link>https://x/2</link></item></channel></rss>";
        let v = parse_feed(xml, f, 1);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].title, "A & B");
        assert_eq!(v[1].title, "T2 & more");
        assert_eq!(v[1].ts, 1);
    }

    #[test]
    fn builds_answers() {
        let now_ts = parse_rfc2822("Thu, 29 Oct 2026 06:00:00 GMT").unwrap();
        let mk = |lang: &str, cat: &str, title: &str, ago: u64| Item { lang: lang.into(), source: "S".into(), category: cat.into(), title: title.into(), link: format!("https://x/{title}"), ts: now_ts - ago };
        let items = vec![mk("ja", "主要", "今日の見出し", 3600), mk("ja", "主要", "三日前の見出し", 3 * 86400), mk("ja", "主要", "先月の見出し", 40 * 86400), mk("en", "World", "Headline now", 600)];
        let it = |hours, this_month| Intent { ja: true, hours, label_ja: "テスト", label_en: "Test", this_month };
        let a = build_answer(&it(30, false), &items, now_ts - 86400, now_ts);
        assert!(a.contains("今日の見出し") && !a.contains("三日前") && a.contains("https://x/今日の見出し") && a.contains("記録を始めた"));
        let m = build_answer(&it(744, true), &items, now_ts - 90 * 86400, now_ts);
        assert!(m.contains("今日の見出し") && m.contains("三日前の見出し") && !m.contains("先月の見出し"));
        let e = build_answer(&Intent { ja: false, hours: 30, label_ja: "", label_en: "Today's news", this_month: false }, &items, now_ts - 86400, now_ts);
        assert!(e.contains("Headline now") && !e.contains("今日の見出し"));
        let y = build_year_digest(&YearIntent { ja: true, year: 2026 }, &items, &[], now_ts - 86400);
        assert!(y.contains("2026年10月") && y.contains("今日の見出し") && y.contains("https://x/"));
    }
}
