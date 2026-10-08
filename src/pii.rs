//! 個人情報の二次判定(`POST /v1/classify-pii`、2026-10-08新設)。
//!
//! `backup-system`(aon-co-jp-backup)の個人情報スキャナが、ルール検出で
//! 「曖昧(warn)」とした断片だけを送ってくる。**値そのものは送らない**——
//! 呼び出し側が`<EMAIL>`/`<PHONE>`/`<NUM>`のようにマスクした行(周辺の文脈)だけが
//! 届く設計で、本モジュールも念のため再マスクする。
//!
//! **正直な開示**: 訓練済みの個人情報分類器ではなく、`generic_classify`と同じ
//! 埋め込み+コサイン類似度で「実在の人物のデータ一覧らしいか / サンプル・連絡先の
//! 記載らしいか」を比べるヒューリスティック。確信が持てないとき(差が`MARGIN`未満)は
//! `verdict="unsure"`を返し、呼び出し側(ルール検出)の判断を覆さない。

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::GpuDevice;

use crate::generic_classify::classify_many;

pub const CAT_PII: &str = "実在の人物の個人情報を並べたデータ(顧客名簿・利用者一覧・会員データ)";
pub const CAT_BENIGN: &str = "ソースコードのサンプル・テスト用ダミー値・問い合わせ先の記載";
const MARGIN: f32 = 0.03;
const MAX_SNIPPETS: usize = 8;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Verdict {
    Pii,
    Benign,
    Unsure,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pii => "pii",
            Verdict::Benign => "benign",
            Verdict::Unsure => "unsure",
        }
    }
}

/// 断片ごとの多数決。`pii_votes`/`benign_votes`は`MARGIN`を超えた断片の数。
pub fn decide(pii_votes: usize, benign_votes: usize) -> Verdict {
    if pii_votes == 0 && benign_votes == 0 {
        Verdict::Unsure
    } else if pii_votes > benign_votes {
        Verdict::Pii
    } else if benign_votes > pii_votes {
        Verdict::Benign
    } else {
        Verdict::Unsure
    }
}

/// メール・電話・長い数字列を伏せ字にする(防御的な再マスク)。
pub fn mask(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for token in line.split_inclusive(char::is_whitespace) {
        let t = token.trim();
        let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
        if t.contains('@') && t.contains('.') {
            out.push_str("<EMAIL>");
            if token.ends_with(char::is_whitespace) { out.push(' '); }
        } else if digits >= 6 {
            out.push_str("<NUM>");
            if token.ends_with(char::is_whitespace) { out.push(' '); }
        } else {
            out.push_str(token);
        }
    }
    out.chars().take(300).collect()
}

pub fn classify_pii(device: &Arc<dyn GpuDevice>, snippets: &[String]) -> Result<(Verdict, usize, usize)> {
    let masked: Vec<String> = snippets.iter().take(MAX_SNIPPETS).map(|s| mask(s)).collect();
    if masked.is_empty() {
        return Ok((Verdict::Unsure, 0, 0));
    }
    let cats = vec![CAT_PII.to_string(), CAT_BENIGN.to_string()];
    let (mut p, mut b) = (0usize, 0usize);
    for item in &masked {
        // 1断片ずつ2カテゴリと比べ、スコアの差が十分なときだけ票に数える。
        let both = classify_many(device, std::slice::from_ref(item), &cats)?;
        let best = &both[0];
        // 最良でないほうのスコアも要るので、反対カテゴリ単独でもう一度比べる。
        let other_cat = if best.category == CAT_PII { CAT_BENIGN } else { CAT_PII };
        let other = classify_many(device, std::slice::from_ref(item), &[other_cat.to_string()])?;
        if best.score - other[0].score >= MARGIN {
            if best.category == CAT_PII { p += 1 } else { b += 1 }
        }
    }
    Ok((decide(p, b), p, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_majority_and_ties() {
        assert_eq!(decide(3, 1), Verdict::Pii);
        assert_eq!(decide(0, 2), Verdict::Benign);
        assert_eq!(decide(1, 1), Verdict::Unsure);
        assert_eq!(decide(0, 0), Verdict::Unsure);
    }

    #[test]
    fn mask_hides_emails_and_numbers() {
        let m = mask("user: taro@example.co.jp tel 09012345678 name=Taro");
        assert!(!m.contains('@'));
        assert!(!m.contains("0901234"));
        assert!(m.contains("<EMAIL>") && m.contains("<NUM>"));
        assert!(m.contains("name=Taro"));
    }

    #[test]
    fn mask_truncates_long_lines() {
        assert!(mask(&"a ".repeat(1000)).chars().count() <= 300);
    }
}
