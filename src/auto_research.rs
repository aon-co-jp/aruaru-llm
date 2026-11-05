//! DATABASE(知識ストア)の自動再調査・自動再更新(ユーザー指示 2026-11-06)。
//!
//! DATABASEに固定回答が無い質問が来たら、質問の中心語(トピック)を取り出し、無料・無制限の自前メタ検索
//! `aruaru-search`と、蓄積したネットニュースの見出しで**自動で再検索**する。複数の独立した出典
//! (異なるサイト、またはニュース見出し+検索結果)で裏付けが取れた(=確信が得られた)ときだけ、
//! 出典つきの要約を知識ストアへ`自動調査`タグで保存・更新する。
//!
//! ## 安全側の設計(正直な開示)
//! - 保存するのは**検索結果の見出し・抜粋と出典**であり、AIが作った文章ではない。誤情報の固定化を避けるため。
//! - `固定回答`タグは付けない。つまり`knowledge::verbatim_for`(登録した文章をそのまま返す)の対象にはならず、
//!   あくまで回答時の**文脈**(`Known facts:`)として使われる。ユーザーが登録した固定回答は一切変更しない。
//! - 裏付けが2つ未満なら何も保存しない。同じ語は12時間以内には再調査しない。自動調査の項目は最大300件。
//! - 古くなった項目(3日以上)は、6時間ごとの背景処理で同じ語を再検索して置き換える。
//! - 検索は`aruaru-search`のみ(APIキー・課金なし)。

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::knowledge;

pub const TAG: &str = "自動調査";
const MIN_SOURCES: usize = 2;
const FRESH_SECS: u64 = 12 * 3600;
const STALE_SECS: u64 = 3 * 24 * 3600;
const MAX_AUTO_ENTRIES: usize = 300;

static LAST_RUN: AtomicU64 = AtomicU64::new(0);
static BUSY: AtomicBool = AtomicBool::new(false);

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 質問から中心語(トピック)を取り出す。助詞・依頼表現を取り除き、最も長い語を採る。無ければ`None`。
pub fn topic_term(question: &str) -> Option<String> {
    let mut s = question.to_string();
    for w in [
        "について教えて", "について", "とは何ですか", "とは何", "とは", "を教えて", "教えてください", "教えて", "ください", "ですか", "でしょうか",
        "どう思いますか", "どう思う", "なぜ", "どうして", "ってなに", "って何", "とは?", "?", "?", "。", "、", "!", "!", "「", "」",
    ] {
        s = s.replace(w, " ");
    }
    for p in ['を', 'は', 'が', 'の', 'に', 'で', 'と', 'も', 'へ', 'や'] {
        s = s.replace(p, " ");
    }
    s.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .map(str::trim)
        .filter(|t| {
            let n = t.chars().count();
            (2..=20).contains(&n) && !(t.is_ascii() && n < 4)
        })
        .max_by_key(|t| (!t.is_ascii(), t.chars().count()))
        .map(str::to_string)
}

fn host_of(link: &str) -> String {
    link.split("//").nth(1).unwrap_or(link).split('/').next().unwrap_or("").trim_start_matches("www.").to_lowercase()
}

/// 検索結果から、トピックを含み、独立した出典(ホスト)が`MIN_SOURCES`以上あるときだけ、保存する本文を返す。
pub fn build_entry_text(term: &str, results: &[(String, String, String)], headlines: Option<&str>) -> Option<String> {
    let t = term.to_lowercase();
    let mut hosts: HashSet<String> = HashSet::new();
    let mut lines: Vec<String> = Vec::new();
    for (title, snippet, link) in results {
        if !(title.to_lowercase().contains(&t) || snippet.to_lowercase().contains(&t)) {
            continue;
        }
        let host = host_of(link);
        if host.is_empty() || !hosts.insert(host.clone()) {
            continue;
        }
        let sn: String = snippet.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(140).collect();
        lines.push(format!("・{} — {} ({})", title.trim(), sn, host));
        if lines.len() >= 3 {
            break;
        }
    }
    let mut sources = hosts.len();
    if let Some(h) = headlines.filter(|h| h.to_lowercase().contains(&t)) {
        sources += 1;
        lines.push(format!("・{}", h.chars().take(300).collect::<String>()));
    }
    if sources < MIN_SOURCES || lines.is_empty() {
        return None;
    }
    let (y, m, d) = civil_date(now());
    Some(format!("【自動調査 {y}-{m:02}-{d:02}・出典{sources}件】{term}\n{}", lines.join("\n")))
}

fn civil_date(ts: u64) -> (i64, u32, u32) {
    // 1970-01-01からの日数 → 年月日(Howard Hinnantのアルゴリズム、UTC)
    let z = (ts / 86400) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 1語を再調査して、確信が得られたら知識ストアを更新する。更新したら`true`。
pub async fn research_term(term: &str, force: bool) -> bool {
    if !force {
        let fresh = knowledge::list().iter().any(|e| e.source == "auto-research" && e.tags.iter().any(|t| t == term) && now().saturating_sub(e.created_unix) < FRESH_SECS);
        if fresh {
            return false;
        }
    }
    let results = match tokio::time::timeout(Duration::from_secs(20), crate::web_search::search_free_only(term, 6, None, None)).await {
        Ok(Ok(r)) => r,
        _ => return false,
    };
    let triples: Vec<(String, String, String)> = results.into_iter().map(|r| (r.title, r.snippet, r.link)).collect();
    let headlines = crate::news_feed::context_for(term);
    let Some(text) = build_entry_text(term, &triples, headlines.as_deref()) else { return false };
    knowledge::upsert_auto(term, &text, MAX_AUTO_ENTRIES).is_ok()
}

/// 回答の背景で、質問のトピックを再調査する(5秒に1回まで、同時に1件まで)。
pub fn spawn_for_question(question: &str) {
    let Some(term) = topic_term(question) else { return };
    let t = now();
    if t.saturating_sub(LAST_RUN.load(Ordering::Relaxed)) < 5 || BUSY.swap(true, Ordering::AcqRel) {
        return;
    }
    LAST_RUN.store(t, Ordering::Relaxed);
    tokio::spawn(async move {
        let updated = research_term(&term, false).await;
        eprintln!("auto-research: {term} -> {}", if updated { "database updated" } else { "no change" });
        BUSY.store(false, Ordering::Release);
    });
}

/// 6時間ごと: 3日以上前の自動調査の項目を、同じ語で再検索して更新する(1回に最大10語)。
pub fn spawn_background_refresh() {
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_secs(300)).await;
        loop {
            let stale: Vec<String> = knowledge::list()
                .into_iter()
                .filter(|e| e.source == "auto-research" && now().saturating_sub(e.created_unix) >= STALE_SECS)
                .filter_map(|e| e.tags.iter().find(|t| t.as_str() != TAG).cloned())
                .take(10)
                .collect();
            for term in stale {
                let ok = research_term(&term, true).await;
                eprintln!("auto-research refresh: {term} -> {ok}");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
            tokio::time::sleep(Duration::from_secs(6 * 3600)).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_topic() {
        assert_eq!(topic_term("消費税について教えて").as_deref(), Some("消費税"));
        assert_eq!(topic_term("ウクライナ情勢とは").as_deref(), Some("ウクライナ情勢"));
        assert_eq!(topic_term("What is inflation?").as_deref(), Some("inflation"));
        assert_eq!(topic_term("あ"), None);
        // 日本語の語があれば、英語の長い語より優先する
        assert_eq!(topic_term("ウクライナ情勢について island-travel-guide").as_deref(), Some("ウクライナ情勢"));
    }

    #[test]
    fn needs_two_independent_sources() {
        let r = |t: &str, l: &str| (t.to_string(), "snippet about topic".to_string(), l.to_string());
        let one = vec![r("topic news", "https://a.example/1"), r("topic more", "https://a.example/2")];
        assert!(build_entry_text("topic", &one, None).is_none(), "同じサイト2件は独立した出典として数えない");
        let two = vec![r("topic news", "https://a.example/1"), r("topic report", "https://www.b.example/x")];
        let t = build_entry_text("topic", &two, None).unwrap();
        assert!(t.contains("出典2件") && t.contains("a.example") && t.contains("b.example"), "{t}");
        assert!(build_entry_text("other", &two, None).is_none(), "トピックを含まない結果は使わない");
        assert!(build_entry_text("topic", &one, Some("[BBC World] topic headline")).is_some(), "ニュース見出しが2つ目の出典になる");
    }

    #[test]
    fn civil_date_known() {
        assert_eq!(civil_date(0), (1970, 1, 1));
        assert_eq!(civil_date(1_794_000_000).0, 2026);
    }
}
