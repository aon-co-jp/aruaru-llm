//! 共有ニュースストック(ユーザー指示 2026-10-24「ストックニュースは、realdata.proとmaid-cafe-englishで
//! 知識ベースとし情報共有してAIとの会話に活かして」)。
//!
//! ## 設計
//! - 実体は`ARUARU_LLM_NEWS_STOCK_DIR`(既定`data/news-stock`)直下のMarkdown(`*.md`)。
//!   open-englishの毎時アーカイブ処理(`archive-news-to-github.sh`)が、GitHubへ退避するのと同じ
//!   刈り取り済みニュースをここへも追記する。形式は`### 国名(検索日時 / searched at: 日付)`の
//!   見出しの下に`- [見出し](リンク) — 抜粋`が並ぶ、アーカイブと同じもの。
//! - aruaru-llmはrealdata.proもopen-englishも同じインスタンスを使うので、ここへ置けば両方の
//!   AI会話に同じストックが効く(`knowledge::context_for_prompt`から差し込む)。
//! - `knowledge.json`には入れない(数万件になると毎回の総なめ検索・世代バックアップが重くなるため)。
//! - 検索は単語の重なりのみ(埋め込みは使わない)。国名が質問に含まれれば、その国の新しい順。
//!   正直な開示: 見出しの単語一致なので、言い換えや日本語の国名(スペイン等)は拾えない。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

const RELOAD_EVERY: Duration = Duration::from_secs(60);
const MAX_ITEMS: usize = 200_000;

struct Item {
    country: String,
    date: String,
    title: String,
    words: HashSet<String>,
}

struct Cache {
    checked: Option<Instant>,
    stamp: (usize, u64, Option<SystemTime>),
    items: Vec<Item>,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache { checked: None, stamp: (0, 0, None), items: Vec::new() });

fn dir() -> PathBuf {
    std::env::var("ARUARU_LLM_NEWS_STOCK_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("data/news-stock"))
}

fn significant(w: &str) -> bool {
    w.chars().any(|c| !c.is_ascii()) || w.len() >= 4
}

fn parse(text: &str, out: &mut Vec<Item>) {
    let (mut country, mut date) = (String::new(), String::new());
    for line in text.lines() {
        if let Some(h) = line.strip_prefix("### ") {
            if let Some((c, rest)) = h.split_once('(') {
                country = c.trim().to_string();
                date = rest.rsplit_once("searched at:").map(|(_, d)| d.trim().trim_end_matches(')').to_string()).unwrap_or_default();
            }
        } else if let Some(item) = line.strip_prefix("- [") {
            if let Some((title, _)) = item.split_once("](") {
                let title = title.trim();
                if title.is_empty() || country.is_empty() {
                    continue;
                }
                let words = crate::knowledge::words(title).into_iter().filter(|w| significant(w)).collect();
                out.push(Item { country: country.clone(), date: date.clone(), title: title.to_string(), words });
            }
        }
    }
}

fn load(d: &PathBuf) -> ((usize, u64, Option<SystemTime>), Vec<Item>) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(d)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "md")).collect())
        .unwrap_or_default();
    files.sort();
    let (mut bytes, mut newest) = (0u64, None);
    let mut items = Vec::new();
    for f in &files {
        if let Ok(m) = std::fs::metadata(f) {
            bytes += m.len();
            newest = newest.max(m.modified().ok());
        }
        if let Ok(t) = std::fs::read_to_string(f) {
            parse(&t, &mut items);
        }
    }
    if items.len() > MAX_ITEMS {
        items.drain(..items.len() - MAX_ITEMS); // 古い方から捨てる(ファイル名の昇順=古い順)
    }
    ((files.len(), bytes, newest), items)
}

fn with_items<R>(f: impl FnOnce(&[Item]) -> R) -> R {
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if c.checked.map_or(true, |t| t.elapsed() >= RELOAD_EVERY) {
        let d = dir();
        let (stamp, items) = load(&d);
        if c.checked.is_none() || stamp != c.stamp {
            c.stamp = stamp;
            c.items = items;
        }
        c.checked = Some(Instant::now());
    }
    f(&c.items)
}

fn search_in(items: &[Item], query: &str, limit: usize) -> Vec<(String, String, String)> {
    let q: HashSet<String> = crate::knowledge::words(query).into_iter().filter(|w| significant(w)).collect();
    let ql = query.to_lowercase();
    let mut scored: Vec<(usize, &Item)> = items
        .iter()
        .filter_map(|it| {
            let mut s = it.words.intersection(&q).count();
            let by_title = s;
            if it.country.len() >= 3 && ql.contains(&it.country.to_lowercase()) {
                s += 3;
            } else if by_title < 2 {
                return None; // 単語1つの偶然の一致は雑音
            }
            Some((s, it))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.date.cmp(&a.1.date)));
    let mut seen = HashSet::new();
    scored
        .into_iter()
        .filter(|(_, it)| seen.insert(it.title.clone()))
        .take(limit)
        .map(|(_, it)| (it.country.clone(), it.date.clone(), it.title.clone()))
        .collect()
}

/// プロンプトへ足す短い文脈(該当無しなら`None`)。
pub fn context_for(query: &str) -> Option<String> {
    let hits = with_items(|items| search_in(items, query, 3));
    if hits.is_empty() {
        return None;
    }
    Some(format!(
        "Recent news (shared stock): {}",
        hits.iter().map(|(c, d, t)| format!("[{c} {d}] {t}")).collect::<Vec<_>>().join(" | ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "## アーカイブ日 / Archived on 2026-10-01\n\n### Spain(検索日時 / searched at: 2026-09-23)\n\nTags: `Spain`\n\n- [De la Fuente extends Spain contract to 2032](https://x/1) — excerpt\n- [Chaotic scenes as Spanish police clear Ceuta beaches](https://x/2) — e\n\n### Poland(検索日時 / searched at: 2026-09-24)\n\n- [Poland detains neo-Nazi youth planning attack](https://x/3)\n";

    #[test]
    fn parses_and_finds_by_country_and_by_title() {
        let mut v = Vec::new();
        parse(SAMPLE, &mut v);
        assert_eq!(v.len(), 3);
        let by_country = search_in(&v, "what is the latest news in Spain?", 3);
        assert_eq!(by_country.len(), 2);
        assert!(by_country.iter().all(|(c, _, _)| c == "Spain"));
        let by_title = search_in(&v, "tell me about the neo-Nazi youth attack", 3);
        assert_eq!(by_title[0].2, "Poland detains neo-Nazi youth planning attack");
        assert!(search_in(&v, "recipe for curry", 3).is_empty());
    }
}
