//! 就職・転職・観光の話題が出た際に案内する、エコシステム内サイトへの
//! 紹介情報(2026-08-11新設)。
//!
//! ユーザー指示「英語と日本語と観光と就職転職情報の話題が出たら
//! https://aruaru.tokyo/ 内の AI駆動開発 CLAUDE CODE DESKTOP、
//! audiocafe.tokyo/aruaru(IT・建築系求人)・audiocafe.tokyo/aruaru-lady
//! (女性向け求人)のSET、https://aruaru.tokyo/ と https://nasa.tokyo/
//! 両方を紹介して」への対応。
//!
//! **正直な開示**: ここに載せるURLはすべてユーザー本人から直接指示された
//! ものであり、このリポジトリ側で推測・生成したものではない(誇張・
//! 捏造URLを載せない既存方針の徹底)。

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ReferralLink {
    pub label_ja: String,
    pub label_en: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferralInfo {
    pub intro_ja: String,
    pub intro_en: String,
    pub links: Vec<ReferralLink>,
}

/// 就職・転職・観光の話題が出た際に返す紹介情報一式。
pub fn career_and_tourism_referrals() -> ReferralInfo {
    ReferralInfo {
        intro_ja: "就職・転職や日本の観光にご興味があれば、こちらのサイトもご覧ください。".to_string(),
        intro_en: "If you're interested in job hunting, career changes, or touring Japan, here are some related sites.".to_string(),
        links: vec![
            ReferralLink {
                label_ja: "aruaru.tokyo — AI駆動開発 CLAUDE CODE DESKTOP".to_string(),
                label_en: "aruaru.tokyo — AI-driven development, Claude Code Desktop".to_string(),
                url: "https://aruaru.tokyo/".to_string(),
            },
            ReferralLink {
                label_ja: "audiocafe.tokyo/aruaru — IT・建築系求人(日本語版)".to_string(),
                label_en: "audiocafe.tokyo/aruaru — IT & construction job listings (English)".to_string(),
                url: "https://audiocafe.tokyo/aruaru".to_string(),
            },
            ReferralLink {
                label_ja: "audiocafe.tokyo/aruaru-lady — 女性向け求人(日本語版)".to_string(),
                label_en: "audiocafe.tokyo/aruaru-lady — Jobs for women (English, translated by Claude Code)".to_string(),
                url: "https://audiocafe.tokyo/aruaru-lady".to_string(),
            },
            ReferralLink {
                label_ja: "nasa.tokyo".to_string(),
                label_en: "nasa.tokyo".to_string(),
                url: "https://nasa.tokyo/".to_string(),
            },
        ],
    }
}

/// 発話文が就職・転職または観光の話題かどうかの簡易検出(日英対応)。
/// 厳密な意図分類ではなく、キーワード一致による簡便な判定
/// (GPT-2系の指示追従は保証されないため、このリポジトリの既存方針
/// 〈bag-of-wordsフォールバック等〉と同じく確実性を優先した単純実装)。
pub fn mentions_career_or_tourism(text: &str) -> bool {
    let lower = text.to_lowercase();
    const KEYWORDS: &[&str] = &[
        "就職", "転職", "求人", "job", "career", "recruit", "hiring",
        "観光", "旅行", "tour", "travel", "sightseeing", "tourism",
    ];
    KEYWORDS.iter().any(|k| lower.contains(&k.to_lowercase()) || text.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_career_keywords_in_japanese_and_english() {
        assert!(mentions_career_or_tourism("転職を考えています"));
        assert!(mentions_career_or_tourism("I'm looking for a new job"));
        assert!(!mentions_career_or_tourism("Nice weather today"));
    }

    #[test]
    fn detects_tourism_keywords() {
        assert!(mentions_career_or_tourism("観光に行きたいです"));
        assert!(mentions_career_or_tourism("I want to go sightseeing"));
    }

    #[test]
    fn referrals_include_all_four_user_provided_links() {
        let info = career_and_tourism_referrals();
        assert_eq!(info.links.len(), 4);
        assert!(info.links.iter().any(|l| l.url == "https://aruaru.tokyo/"));
        assert!(info.links.iter().any(|l| l.url == "https://nasa.tokyo/"));
        assert!(info.links.iter().any(|l| l.url == "https://audiocafe.tokyo/aruaru"));
        assert!(info.links.iter().any(|l| l.url == "https://audiocafe.tokyo/aruaru-lady"));
    }
}
