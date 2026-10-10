//! OCR の誤読を、前後の行の文脈から補正する(`POST /v1/ocr/correct`、2026-10-10新設)。
//!
//! # 方式
//!
//! アクティブな Qwen2.5(日本語・中国語・英語などに対応、`/v1/qwen/install` で導入)に、
//! 「前の行・対象の行・次の行」を見せて、**対象の行の誤読だけを直した1行**を出力させる。
//! 小さな言語モデルは、頼まれていない書き換え(幻覚)をしがちなので、**出力をそのまま採用せず**、
//! [`accept`] で次の条件をすべて満たした修正だけを受け入れる:
//! - 文字数の差は2文字以内、編集距離は行の長さの 1/5 以内(大きな書き換えは拒否)。
//! - 変わった文字は、字形や種類が近い置き換え(漢字↔漢字、かな↔かな、英字↔英字/数字、数字↔数字/英字、記号↔記号)だけ。
//!
//! 信頼度が高い行は、そもそも補正の対象にしない(文脈としてだけ使う)。
//!
//! **正直な開示(2026-10-10 実測、実験的機能)**: Qwen2.5-0.5B-Instruct(`/v1/qwen/install` で導入)を使った試験では、
//! (1) **速度**: このエンジンはプロンプトを1トークンずつ処理する(バッチ prefill が無い)ため、数百トークンの指示つきプロンプトで
//!     1行あたり約90秒、7行で10分37秒かかった(32スレッドの PC)。実用の速度ではない。
//! (2) **品質**: 0.5B は「指示に従って1行だけ直す」ことができず(前置きの説明を出力する)、検証 [`accept`] で全て却下され、修正は0件だった。
//!     文脈の続きを予測させる(穴埋め)使い方は速く(1〜5秒)、「吾」→「輩は」のように当たる例もあるが、「第一」→「章」「無」→「い」は外れた。
//! 実用にするには、(a) `open-cuda` の Qwen にバッチ prefill と KV キャッシュの共有を入れる、(b) より大きなモデル(1.5B 以上)と GPU を使う、
//! (c) Tesseract が返す文字ごとの別候補を、言語モデルで選び直す方式にする、のいずれか(または組み合わせ)が必要。
//! 誤った補正を避けるため、出力は [`accept`] で厳しく検証し、補正前後を両方返す。アプリ(pdf-r2l-rs)はまだこの機能を使っていない。

use std::sync::Arc;
use std::time::{Duration, Instant};

use opencuda_core::GpuDevice;

/// 1回の要求で受け付ける行数の上限。
pub const MAX_LINES: usize = 300;
/// 1回の要求で補正に使う時間の上限(超えたら残りは補正せずに返す)。
const DEADLINE: Duration = Duration::from_secs(600);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CorrectLine {
    pub text: String,
    /// 行の信頼度(0〜100)。`only_below` 以上の行は補正しない。
    #[serde(default = "default_conf")]
    pub conf: f32,
}

fn default_conf() -> f32 {
    0.0
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct CorrectResult {
    pub index: usize,
    pub original: String,
    pub corrected: String,
    pub changed: bool,
}

/// 文字の種類(置き換えの妥当性の判定用)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Han,
    Kana,
    Hangul,
    Letter,
    Digit,
    Punct,
    Other,
}

fn kind(c: char) -> Kind {
    match c as u32 {
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF => Kind::Han,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Kind::Kana,
        0xAC00..=0xD7AF | 0x1100..=0x11FF => Kind::Hangul,
        _ if c.is_ascii_digit() => Kind::Digit,
        _ if c.is_alphabetic() => Kind::Letter,
        _ if c.is_ascii_punctuation() || matches!(c as u32, 0x3000..=0x303F | 0xFF01..=0xFF0F | 0xFF1A..=0xFF20) => Kind::Punct,
        _ => Kind::Other,
    }
}

/// 置き換え `a → b` が、誤読として起こり得る範囲か。
fn plausible_substitution(a: char, b: char) -> bool {
    let (ka, kb) = (kind(a), kind(b));
    match (ka, kb) {
        (Kind::Han, Kind::Han) | (Kind::Kana, Kind::Kana) | (Kind::Hangul, Kind::Hangul) | (Kind::Punct, Kind::Punct) | (Kind::Digit, Kind::Digit) => true,
        // 数字と英字(0↔O、1↔l など)、英字どうし(rn↔m などの 1 文字ぶん)
        (Kind::Letter, Kind::Letter) | (Kind::Letter, Kind::Digit) | (Kind::Digit, Kind::Letter) => true,
        // 長音記号と漢数字の一(ー↔一)、など、かな/漢字/記号が入れ替わる代表的な誤読
        (Kind::Kana, Kind::Han) | (Kind::Han, Kind::Kana) => matches!((a, b), ('ー', '一') | ('一', 'ー') | ('ロ', '口') | ('口', 'ロ') | ('エ', '工') | ('工', 'エ') | ('カ', '力') | ('力', 'カ') | ('ニ', '二') | ('二', 'ニ') | ('ハ', '八') | ('八', 'ハ') | ('タ', '夕') | ('夕', 'タ')),
        (Kind::Punct, Kind::Han) | (Kind::Han, Kind::Punct) => matches!((a, b), ('一', '-') | ('-', '一') | ('一', '―') | ('―', '一')),
        _ => false,
    }
}

/// 編集距離と、置き換えだけで済むかの判定。`(編集距離, 追加・削除の数, 置き換えが全て妥当か)`
fn diff_stats(a: &[char], b: &[char]) -> (usize, usize, bool) {
    let (n, m) = (a.len(), b.len());
    // dp[i][j] = (距離, 追加削除の数, 置き換えが全て妥当か)
    let mut dp = vec![vec![(usize::MAX / 2, 0usize, true); m + 1]; n + 1];
    dp[0][0] = (0, 0, true);
    for i in 0..=n {
        for j in 0..=m {
            let cur = dp[i][j];
            if cur.0 >= usize::MAX / 2 {
                continue;
            }
            let mut relax = |ni: usize, nj: usize, cost: usize, indel: usize, ok: bool| {
                let cand = (cur.0 + cost, cur.1 + indel, cur.2 && ok);
                if cand.0 < dp[ni][nj].0 || (cand.0 == dp[ni][nj].0 && (cand.1, !cand.2) < (dp[ni][nj].1, !dp[ni][nj].2)) {
                    dp[ni][nj] = cand;
                }
            };
            if i < n && j < m {
                if a[i] == b[j] {
                    relax(i + 1, j + 1, 0, 0, true);
                } else {
                    relax(i + 1, j + 1, 1, 0, plausible_substitution(a[i], b[j]));
                }
            }
            if i < n {
                relax(i + 1, j, 1, 1, true);
            }
            if j < m {
                relax(i, j + 1, 1, 1, true);
            }
        }
    }
    dp[n][m]
}

/// モデルの出力が、安全な補正として受け入れられるか(受け入れるなら補正後の行)。
pub fn accept(original: &str, proposed: &str) -> Option<String> {
    // 最初の非空の行だけを見る。「対象の行:」のような前置きや引用符を落とす。
    let first = proposed.lines().map(|l| l.trim()).find(|l| !l.is_empty())?;
    let cleaned = first
        .trim_start_matches("対象の行:")
        .trim_start_matches("対象の行：")
        .trim()
        .trim_matches(|c| matches!(c, '「' | '」' | '"' | '“' | '”' | '`'))
        .trim();
    if cleaned.is_empty() || cleaned == original {
        return None;
    }
    let (a, b): (Vec<char>, Vec<char>) = (original.chars().collect(), cleaned.chars().collect());
    if a.len().abs_diff(b.len()) > 2 {
        return None;
    }
    let (dist, indel, subs_ok) = diff_stats(&a, &b);
    let limit = (a.len() / 5).max(2);
    if dist == 0 || dist > limit || indel > 2 || !subs_ok {
        return None;
    }
    Some(cleaned.to_string())
}

/// モデルへの指示(Qwen の chat 形式)。前後の行を文脈として見せる。
pub fn build_prompt(prev: Option<&str>, line: &str, next: Option<&str>) -> String {
    let system = "あなたは文字認識(OCR)の誤読を直す校正者です。前後の行の文脈から明らかな誤読の文字だけを直し、それ以外は一字も変えずに、直した対象の行を1行だけ出力します。説明は書きません。直す所が無ければ対象の行をそのまま出力します。";
    let user = format!(
        "例1)\n前の行: 第一章 はじめに\n対象の行: 吾輩は猫である。名前はまだ無ぃ。\n次の行: どこで生れたかとんと見当がつかぬ。\n出力: 吾輩は猫である。名前はまだ無い。\n\n例2)\n前の行: Chapter 1\n対象の行: The qnick brown fox jumps over the 1azy dog.\n次の行: Pack my box with five dozen liquor jugs.\n出力: The quick brown fox jumps over the lazy dog.\n\n本番)\n前の行: {}\n対象の行: {}\n次の行: {}\n出力:",
        prev.unwrap_or("(なし)"),
        line,
        next.unwrap_or("(なし)")
    );
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// 行ごとに補正する(同期・CPU 重い。`spawn_blocking` から呼ぶ)。
pub fn correct_lines(device: &Arc<dyn GpuDevice>, lines: &[CorrectLine], only_below: f32) -> Vec<CorrectResult> {
    let started = Instant::now();
    let mut out = Vec::with_capacity(lines.len());
    for (i, l) in lines.iter().enumerate() {
        let skip = l.conf >= only_below || l.text.chars().count() < 3 || started.elapsed() > DEADLINE;
        let mut corrected = l.text.clone();
        if !skip {
            let prompt = build_prompt(i.checked_sub(1).map(|p| lines[p].text.as_str()), &l.text, lines.get(i + 1).map(|n| n.text.as_str()));
            let max_new = (l.text.chars().count() * 2 + 8).min(120);
            if let Ok(gen) = crate::qwen_generation::generate(device, &prompt, max_new) {
                if let Some(ok) = accept(&l.text, &gen) {
                    corrected = ok;
                }
            }
        }
        out.push(CorrectResult { index: i, changed: corrected != l.text, original: l.text.clone(), corrected });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_small_plausible_fixes() {
        assert_eq!(accept("第一草 PDFの綴じ方向", "第一章 PDFの綴じ方向"), Some("第一章 PDFの綴じ方向".into()));
        assert_eq!(accept("名前はまだ無ぃ。", "名前はまだ無い。"), Some("名前はまだ無い。".into()));
        assert_eq!(accept("The qnick brown fox", "The quick brown fox"), Some("The quick brown fox".into()));
        // 前置き・引用符つきでも、中身が妥当なら受け入れる
        assert_eq!(accept("fom scanned pages", "対象の行: 「from scanned pages」"), Some("from scanned pages".into()));
        assert_eq!(accept("12,345円", "12,345円\n(説明)"), None, "unchanged");
    }

    #[test]
    fn rejects_rewrites_and_implausible_changes() {
        // 大きな書き換え
        assert_eq!(accept("名前はまだ無い。", "吾輩は猫である。名前はまだ無い。"), None);
        // 日本語の文を英語にする、全く別の文にする
        assert_eq!(accept("今日は晴れです", "It is sunny today"), None);
        // かな→英字のような不自然な置き換え
        assert_eq!(accept("今日は晴れです", "今日はAれです"), None);
        // 空・同じ
        assert_eq!(accept("abc", ""), None);
        assert_eq!(accept("abc", "abc"), None);
        // 複数箇所の書き換えが上限(長さの 1/5)を超える
        assert_eq!(accept("吾輩は猫である", "我輩ハ描デ在ル"), None);
    }

    #[test]
    fn prompt_contains_context_and_template() {
        let p = build_prompt(Some("前"), "対象", None);
        assert!(p.starts_with("<|im_start|>system"));
        assert!(p.contains("前の行: 前") && p.contains("対象の行: 対象") && p.contains("次の行: (なし)"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }
}
