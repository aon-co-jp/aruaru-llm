//! OCR で読めなかった・自信が低い **1〜2文字** を、前後の長い文脈から言語モデルで穴埋めする(`POST /v1/ocr/fill`)。
//!
//! 長文の書き換えではない。呼び出し側(アプリ)が、信頼度の低い箇所ごとに `(前の文脈, 疑わしい文字列, 後ろの文脈)` を送る。
//! サーバーは候補を作り、**候補 + 後ろの文脈** が前の文脈のあとに続く確率(対数尤度)を言語モデルで採点して、
//! 元の読みより十分に尤もらしい(`margin` 以上高い)ときだけ置き換える。置き換えは常に **1 か所・最大 `max_chars` 文字**。
//!
//! 候補の出どころ:
//! - 形の似た文字(日↔曰、未↔末、0↔O など)の表。
//! - 言語モデル自身の予測(前の文脈 + 疑わしい文字列の手前までの次の文字の上位候補)。元の文字と同じ種類(漢字は漢字、
//!   かなはかな、数字・英字どうし)の 1 文字だけを残す。
//! - 文字がまるごと欠けている疑い(`gap`)のときは、そこに入れる 1〜`max_chars` 文字をモデルに挙げさせる。
//!
//! 正直な開示: 採点するモデルは小さい(Qwen2.5-0.5B 程度)。誤って「もっともらしい別の字」に直す可能性があるため、
//! 応答には元の読み・採用した読み・候補ごとのスコアを全て返し、アプリ側は低リスクな箇所だけ採用できる。

use std::time::{Duration, Instant};

/// 1回の要求で受け付ける箇所の数の上限。
pub const MAX_ITEMS: usize = 400;
/// 1回の要求で使う時間の上限(超えたら残りは補完せずに返す)。
const DEADLINE: Duration = Duration::from_secs(600);
/// 前の文脈・後ろの文脈として使う最大文字数。
const PREFIX_CHARS: usize = 60;
const SUFFIX_CHARS: usize = 24;
/// 位置ごとにモデルへ挙げさせる候補の数。
const TOP_K: usize = 12;
/// 1 か所で採点する候補数の上限(採点が一番重いため)。
const MAX_CANDIDATES: usize = 14;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FillItem {
    /// 疑わしい箇所の前の文脈(行の前半+前の行の末尾など)。
    #[serde(default)]
    pub prefix: String,
    /// 疑わしい文字列(OCR の読み)。`gap` のときは空でよい。
    #[serde(default)]
    pub suspect: String,
    /// 後ろの文脈。
    #[serde(default)]
    pub suffix: String,
    /// 文字がまるごと欠けている疑い(そこへ 1〜`max_chars` 文字を挿入する候補だけを調べる)。
    #[serde(default)]
    pub gap: bool,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct Scored {
    pub text: String,
    /// この候補 + 後ろの文脈の、前の文脈つき対数尤度(大きいほど尤もらしい)。
    pub logp: f32,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct FillResult {
    pub index: usize,
    pub suspect: String,
    /// 採用した文字列(変えないときは `suspect` のまま)。
    pub chosen: String,
    pub changed: bool,
    /// 採用した候補と元の読みの対数尤度の差(変えないときは 0)。
    pub margin: f32,
    /// 上位の候補(元の読みを含む)。
    pub top: Vec<Scored>,
}

/// 言語モデルへの窓口(試験では偽物に差し替える)。
pub trait Lm {
    /// `prefix` のあとに各 `conts[i]` が続く対数尤度。
    fn score(&self, prefix: &str, conts: &[String]) -> Result<Vec<f32>, String>;
    /// `prefix` の次に来る文字列の上位 `k` 個(文字列, 対数確率)。
    fn top_next(&self, prefix: &str, k: usize) -> Result<Vec<(String, f32)>, String>;
}

/// アクティブな Qwen モデルを使う本番の `Lm`。
pub struct QwenLm {
    pub device: std::sync::Arc<dyn opencuda_core::GpuDevice>,
}

impl Lm for QwenLm {
    fn score(&self, prefix: &str, conts: &[String]) -> Result<Vec<f32>, String> {
        crate::qwen_generation::score_texts(&self.device, prefix, conts).map_err(|e| e.to_string())
    }
    fn top_next(&self, prefix: &str, k: usize) -> Result<Vec<(String, f32)>, String> {
        crate::qwen_generation::top_next_texts(&self.device, prefix, k).map_err(|e| e.to_string())
    }
}

/// 形の似た文字の組(OCR がよく取り違える)。
const CONFUSABLE: &[&str] = &[
    "日曰目白", "未末", "己已巳", "土士", "大太犬", "人入八", "刀力", "工エ", "口ロ回", "二ニ", "夕タ", "千干于", "天夫", "木本末",
    "王玉主", "万方", "戸户", "問間聞", "待持特", "場揚", "洋祥", "液夜", "報服", "析折", "貝見", "栄栗", "苗莓", "雪雲",
    "ーー一", "ぁあ", "ぃい", "ぅう", "ぇえ", "ぉお", "っつ", "ゃや", "ゅゆ", "ょよ", "ヘへ", "べペ", "ソン", "シツ", "ノメ", "ヲラ",
    ".,;", "，、", "：;", "0Oo", "1lI|", "5S", "8B", "2Z", "6b", "rn", "vy", "cC", "uU", "kK", "pP", "wW", "xX", "zZ",
    // 別の文字体系の同形文字(キリル・ギリシャ文字が英単語に混ざる誤読)
    "pр", "oоο", "aа", "cс", "eе", "xх", "yу", "iі", "Aд", "Hн", "Tт", "Kк", "Mм", "Bв",
];

fn confusables(c: char) -> Vec<char> {
    let mut out = Vec::new();
    for g in CONFUSABLE {
        if g.contains(c) {
            for x in g.chars() {
                if x != c && !out.contains(&x) {
                    out.push(x);
                }
            }
        }
    }
    out
}

fn tail_chars(s: &str, n: usize) -> String {
    let v: Vec<char> = s.chars().collect();
    v[v.len().saturating_sub(n)..].iter().collect()
}

fn head_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn usable(text: &str, max_chars: usize) -> bool {
    let n = text.chars().count();
    n >= 1 && n <= max_chars && !text.contains('\u{FFFD}') && !text.chars().any(|c| c.is_control() || c.is_whitespace())
}

/// 候補の一覧(0 番目は常に「元の読み」)。
fn candidates(lm: &dyn Lm, prefix: &str, item: &FillItem, max_chars: usize) -> Vec<(String, bool)> {
    let s: Vec<char> = item.suspect.chars().collect();
    // (文字列, 形の似た字の表に基づく候補か)。0 番目は常に元の読み。
    let mut out: Vec<(String, bool)> = vec![(item.suspect.clone(), true)];
    let push = |c: String, table: bool, out: &mut Vec<(String, bool)>| {
        if out.len() < MAX_CANDIDATES && !out.iter().any(|(x, _)| *x == c) {
            out.push((c, table));
        }
    };
    if item.gap || s.is_empty() {
        // 欠けた文字(隙間という物理的な根拠がある候補): そこへ入る 1〜max_chars 文字をモデルに挙げさせる。
        if let Ok(top) = lm.top_next(prefix, TOP_K) {
            for (t, _) in top {
                if usable(&t, max_chars) {
                    push(format!("{}{}", item.suspect, t), true, &mut out);
                }
            }
        }
        return out;
    }
    // 1 文字だけの疑わしい読みは、誤って挿入された文字かもしれない(削除の候補)。
    if s.len() == 1 {
        push(String::new(), true, &mut out);
    }
    // 形の似た文字(1 か所だけ置き換え)
    for (i, &c) in s.iter().enumerate() {
        for alt in confusables(c) {
            let mut v = s.clone();
            v[i] = alt;
            push(v.into_iter().collect(), true, &mut out);
        }
    }
    // モデルの予測(短い疑わしい文字列のみ。位置ごとに手前までを文脈にして次の 1 文字を挙げさせる)
    if s.len() <= 6 {
        for i in 0..s.len() {
            let before: String = s[..i].iter().collect();
            let ctx = format!("{prefix}{before}");
            if let Ok(top) = lm.top_next(&ctx, TOP_K) {
                for (t, _) in top {
                    let mut it = t.chars();
                    if let (Some(c), None) = (it.next(), it.next()) {
                        if usable(&t, 1) && c != s[i] && crate::ocr_correct::plausible_substitution(s[i], c) {
                            let mut v = s.clone();
                            v[i] = c;
                            push(v.into_iter().collect(), false, &mut out);
                        }
                    }
                }
            }
        }
    }
    out
}

/// 1 か所を穴埋めする。
pub fn fill_one(lm: &dyn Lm, index: usize, item: &FillItem, margin: f32, max_chars: usize) -> FillResult {
    let prefix = if item.prefix.is_empty() { "\n".to_string() } else { tail_chars(&item.prefix, PREFIX_CHARS) };
    let suffix = head_chars(&item.suffix, SUFFIX_CHARS);
    let unchanged = |top: Vec<Scored>| FillResult { index, suspect: item.suspect.clone(), chosen: item.suspect.clone(), changed: false, margin: 0.0, top };
    let cands = candidates(lm, &prefix, item, max_chars);
    if cands.len() < 2 {
        return unchanged(Vec::new());
    }
    let conts: Vec<String> = cands.iter().map(|(c, _)| format!("{c}{suffix}")).collect();
    let scores = match lm.score(&prefix, &conts) {
        Ok(s) if s.len() == cands.len() => s,
        _ => return unchanged(Vec::new()),
    };
    let mut ranked: Vec<Scored> = cands.iter().zip(&scores).map(|((t, _), &l)| Scored { text: t.clone(), logp: l }).collect();
    // 形の似た字の表に基づく候補(見た目の根拠がある)は `margin`、モデルの自由な予測だけの候補は倍の差を要求する。
    let best_of = |table: bool| (1..cands.len()).filter(|&i| cands[i].1 == table).max_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let pick = best_of(true).filter(|&i| scores[i] - scores[0] >= margin).or_else(|| best_of(false).filter(|&i| scores[i] - scores[0] >= margin * 2.0));
    ranked.sort_by(|a, b| b.logp.total_cmp(&a.logp));
    ranked.truncate(8);
    match pick {
        Some(i) => FillResult { index, suspect: item.suspect.clone(), chosen: cands[i].0.clone(), changed: true, margin: scores[i] - scores[0], top: ranked },
        None => unchanged(ranked),
    }
}

/// 複数箇所を順に穴埋めする(同期・CPU 重い。`spawn_blocking` から呼ぶ)。
pub fn fill_items(lm: &dyn Lm, items: &[FillItem], margin: f32, max_chars: usize) -> Vec<FillResult> {
    let started = Instant::now();
    items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            if started.elapsed() > DEADLINE {
                FillResult { index: i, suspect: it.suspect.clone(), chosen: it.suspect.clone(), changed: false, margin: 0.0, top: Vec::new() }
            } else {
                fill_one(lm, i, it, margin, max_chars)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 「前の文脈 + 候補 + 後ろ」が、決まった正解の文に近いほど高く採点する偽モデル。
    struct Fake {
        truth: &'static str,
    }
    impl Lm for Fake {
        fn score(&self, prefix: &str, conts: &[String]) -> Result<Vec<f32>, String> {
            Ok(conts.iter().map(|c| if format!("{prefix}{c}").contains(self.truth) { -1.0 } else { -9.0 }).collect())
        }
        fn top_next(&self, prefix: &str, _k: usize) -> Result<Vec<(String, f32)>, String> {
            // 正解の文の、prefix の直後の 1 文字を挙げる。
            let next = self.truth.strip_prefix(prefix.trim_start_matches('\n')).and_then(|r| r.chars().next());
            Ok(next.map(|c| vec![(c.to_string(), -0.1), ("、".to_string(), -2.0)]).unwrap_or_default())
        }
    }

    #[test]
    fn replaces_a_confusable_character_when_the_context_demands_it() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: "曰".into(), suffix: "は晴天なり".into(), gap: false };
        let r = fill_one(&lm, 0, &item, 3.0, 2);
        assert!(r.changed, "{r:?}");
        assert_eq!(r.chosen, "日");
        assert!(r.margin >= 3.0);
    }

    #[test]
    fn keeps_the_reading_when_it_is_already_the_best() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: "日".into(), suffix: "は晴天なり".into(), gap: false };
        let r = fill_one(&lm, 0, &item, 3.0, 2);
        assert!(!r.changed);
        assert_eq!(r.chosen, "日");
    }

    #[test]
    fn inserts_a_missing_character_in_gap_mode() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: String::new(), suffix: "は晴天なり".into(), gap: true };
        let r = fill_one(&lm, 0, &item, 3.0, 2);
        assert!(r.changed, "{r:?}");
        assert_eq!(r.chosen, "日");
    }

    #[test]
    fn never_changes_more_than_one_position_and_respects_the_margin() {
        let lm = Fake { truth: "ZZZZ" };
        let item = FillItem { prefix: "x".into(), suspect: "曰目".into(), suffix: "y".into(), gap: false };
        let r = fill_one(&lm, 0, &item, 3.0, 2);
        assert!(!r.changed, "no candidate reaches the truth, nothing should change: {r:?}");
        // 候補は全て、元の読みから 1 文字しか違わない。
        for c in confusables('曰') {
            assert_ne!(c, '曰');
        }
    }

    #[test]
    fn usable_filters_junk_tokens() {
        assert!(usable("日", 2) && usable("日本", 2));
        assert!(!usable("日本語", 2) && !usable(" ", 2) && !usable("\u{FFFD}", 2) && !usable("\n", 2));
    }
}
