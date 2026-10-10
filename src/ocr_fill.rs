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
const SUFFIX_CHARS: usize = 10;
/// 位置ごとにモデルへ挙げさせる候補の数。
const TOP_K: usize = 8;
/// 1 か所で採点する候補数の上限(採点が一番重いため)。
const MAX_CANDIDATES: usize = 16;
/// 連続して直せる最大の文字数(走査で隣り合う不自然なトークンを 1 か所にまとめる上限)。
pub const MAX_SPAN: usize = 4;
/// 検索で裏付けが取れた 1 件あたりの加点(nat)と、数える件数の上限。
const WEB_BONUS: f32 = 3.0;
const WEB_MAX_HITS: usize = 3;

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
    /// アプリが字形(残っているインクの形)から選んだ追加の候補(疑わしい文字列を置き換える完全な文字列)。
    #[serde(default)]
    pub extra: Vec<String>,
    /// OCR の信頼度(0〜100、不明なら負)。1 文字の疑わしい読みの「削除」は、信頼度が低いときだけ候補にする。
    #[serde(default = "unknown_conf")]
    pub conf: f32,
}

fn unknown_conf() -> f32 {
    -1.0
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
    /// 元の読みの対数尤度(候補を採点できなかったときは 0)。
    pub orig_logp: f32,
    /// `return_all` のとき、採点した全候補(元の読みを含む。アプリが字形の証拠と合わせて選び直す用)。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub all: Vec<Scored>,
}

/// 言語モデルへの窓口(試験では偽物に差し替える)。
pub trait Lm {
    /// `prefix` のあとに各 `conts[i]` が続く対数尤度。
    fn score(&self, prefix: &str, conts: &[String]) -> Result<Vec<f32>, String>;
    /// `prefix` の次に来る文字列の上位 `k` 個(文字列, 対数確率)。
    fn top_next(&self, prefix: &str, k: usize) -> Result<Vec<(String, f32)>, String>;
    /// 文章を各トークンの範囲(文字単位の開始・長さ)と対数確率に分ける(誤読の検出用)。
    fn scan(&self, _text: &str) -> Result<Vec<(usize, usize, f32)>, String> {
        Err("scan is not supported".to_string())
    }
}

/// 実在の文章での裏付けを取る窓口(検索 API。試験では偽物に差し替える)。
pub trait Web {
    /// 検索して、結果の文字列(タイトル・抜粋)を返す。使えないときは空。
    fn snippets(&self, query: &str) -> Vec<String>;
}

/// `aruaru-search`(APIキー不要・無制限)を使う本番の `Web`。tokio の実行環境の中の `spawn_blocking` から呼ぶ。
pub struct AruaruWeb;

impl Web for AruaruWeb {
    fn snippets(&self, query: &str) -> Vec<String> {
        match tokio::runtime::Handle::try_current() {
            Ok(h) => h.block_on(crate::web_search::snippets_via_aruaru_search(query, 10, Some("ja"))),
            Err(_) => Vec::new(),
        }
    }
}

/// 空白を除いた文字列。検索結果の抜粋は、語の間に空白や記号が入ることがあるため、比べるときに空白を無視する。
fn squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// `前の文脈の末尾 + 候補 + 後ろの文脈の先頭` が、そのまま現れる検索結果の数(上限 `WEB_MAX_HITS`)。実在の文章での裏付け。
pub fn web_hits(web: &dyn Web, prefix: &str, cand: &str, suffix: &str) -> usize {
    let (p, s) = (tail_chars(prefix, 6), head_chars(suffix, 6));
    let phrase = format!("{p}{cand}{s}");
    if phrase.chars().count() < 6 {
        return 0;
    }
    let want = squash(&phrase);
    web.snippets(&format!("\"{phrase}\"")).iter().filter(|t| squash(t).contains(&want)).count().min(WEB_MAX_HITS)
}

/// 検索結果の実文から、前の文脈と後ろの文脈にはさまれた文字列(1〜`max_len` 文字)を採掘する。
/// 例: 前が「…私たちは」・後ろが「朝、公園で…」なら、抜粋の中の「私たちは毎朝、公園で」から「毎」を拾う。
pub fn web_candidates(web: &dyn Web, prefix: &str, suffix: &str, max_len: usize) -> Vec<String> {
    let (p, s) = (tail_chars(prefix, 6), head_chars(suffix, 6));
    if p.chars().count() < 3 || s.chars().count() < 2 {
        return Vec::new();
    }
    let (pq, sq): (Vec<char>, Vec<char>) = (squash(&p).chars().collect(), squash(&s).chars().collect());
    let mut out: Vec<String> = Vec::new();
    for text in web.snippets(&format!("{p} {s}")) {
        let t: Vec<char> = squash(&text).chars().collect();
        // 前の文脈・後ろの文脈の、長い一致(最大 6 文字)から短い一致(前 3 文字・後ろ 2 文字)へ。抜粋には前後の語が切れて入ることがある。
        'anchors: for pk in (3..=pq.len()).rev() {
            let pc = &pq[pq.len() - pk..];
            for sk in (2..=sq.len()).rev() {
                let sc = &sq[..sk];
                let mut i = 0;
                while i + pc.len() <= t.len() {
                    if t[i..i + pc.len()] == *pc {
                        let from = i + pc.len();
                        for len in 1..=max_len {
                            if from + len + sc.len() <= t.len() && t[from + len..from + len + sc.len()] == *sc {
                                let mid: String = t[from..from + len].iter().collect();
                                if usable_span(&mid, max_len) && !out.contains(&mid) {
                                    out.push(mid);
                                }
                            }
                        }
                    }
                    i += 1;
                }
                if !out.is_empty() {
                    break 'anchors;
                }
            }
        }
    }
    out
}

/// 空白・制御文字・置換文字を含まない、1〜`max_chars` 文字の文字列か(連続した複数文字の候補用)。
fn usable_span(text: &str, max_chars: usize) -> bool {
    let n = text.chars().count();
    n >= 1 && n <= max_chars && !text.contains('\u{FFFD}') && !text.chars().any(|c| c.is_control() || c.is_whitespace())
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
    fn scan(&self, text: &str) -> Result<Vec<(usize, usize, f32)>, String> {
        crate::qwen_generation::scan_text(&self.device, text).map(|v| v.into_iter().map(|t| (t.start, t.len, t.logp)).collect()).map_err(|e| e.to_string())
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

/// 言語モデルに、`prefix` のあとに続く `len` 文字の文字列を、幅 `width` のビームで挙げさせる(各ステップで 1 文字ずつ)。
fn beam_chain(lm: &dyn Lm, prefix: &str, len: usize, width: usize) -> Vec<String> {
    let mut beams: Vec<(f32, String)> = vec![(0.0, String::new())];
    for _ in 0..len {
        let mut next: Vec<(f32, String)> = Vec::new();
        for (score, so_far) in &beams {
            let Ok(top) = lm.top_next(&format!("{prefix}{so_far}"), width * 3) else {
                continue;
            };
            for (t, lp) in top {
                let mut it = t.chars();
                if let (Some(c), None) = (it.next(), it.next()) {
                    if usable(&t, 1) {
                        next.push((score + lp, format!("{so_far}{c}")));
                    }
                }
            }
        }
        next.sort_by(|a, b| b.0.total_cmp(&a.0));
        next.truncate(width);
        if next.is_empty() {
            return Vec::new();
        }
        beams = next;
    }
    beams.into_iter().map(|(_, s)| s).collect()
}

/// 候補の一覧(0 番目は常に「元の読み」)。
fn candidates(lm: &dyn Lm, web: Option<&dyn Web>, prefix: &str, item: &FillItem, max_chars: usize, lm_free: bool) -> Vec<(String, bool)> {
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
        for e in &item.extra {
            push(format!("{}{}", item.suspect, e), true, &mut out);
        }
        if let Some(w) = web {
            for m in web_candidates(w, prefix, &item.suffix, max_chars) {
                push(format!("{}{}", item.suspect, m), true, &mut out);
            }
        }
        return out;
    }
    // 1 文字だけの疑わしい読みは、誤って挿入された文字かもしれない(削除の候補)。
    if s.len() == 1 && item.conf >= 0.0 && item.conf < 60.0 {
        push(String::new(), true, &mut out);
    }
    for e in &item.extra {
        push(e.clone(), true, &mut out);
    }
    // 連続した 2〜4 文字の読み違い: 言語モデルに、前の文脈のあとに続く同じ長さの文字列を挙げさせる(幅 3 のビーム)。
    if lm_free && (2..=MAX_SPAN).contains(&s.len()) {
        for cand in beam_chain(lm, prefix, s.len(), 3) {
            push(cand, false, &mut out);
        }
    }
    // 実在の文章(検索結果)から、前後の文脈にはさまれた文字列を採掘する(元の読みと近い長さのもの)。
    if let Some(w) = web {
        let n = s.len().max(1);
        for m in web_candidates(w, prefix, &item.suffix, (n + 1).min(MAX_SPAN)) {
            if m.chars().count() + 1 >= n {
                push(m, true, &mut out);
            }
        }
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
    if lm_free && s.len() <= 6 {
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
pub fn fill_one(lm: &dyn Lm, web: Option<&dyn Web>, index: usize, item: &FillItem, margin: f32, max_chars: usize, return_all: bool, lm_free: bool) -> FillResult {
    let prefix = if item.prefix.is_empty() { "\n".to_string() } else { tail_chars(&item.prefix, PREFIX_CHARS) };
    let suffix = head_chars(&item.suffix, SUFFIX_CHARS);
    let unchanged = |top: Vec<Scored>, orig_logp: f32, all: Vec<Scored>| FillResult { index, suspect: item.suspect.clone(), chosen: item.suspect.clone(), changed: false, margin: 0.0, top, orig_logp, all };
    let cands = candidates(lm, web, &prefix, item, max_chars, lm_free);
    if cands.len() < 2 {
        return unchanged(Vec::new(), 0.0, Vec::new());
    }
    let conts: Vec<String> = cands.iter().map(|(c, _)| format!("{c}{suffix}")).collect();
    let scores = match lm.score(&prefix, &conts) {
        Ok(s) if s.len() == cands.len() => s,
        _ => return unchanged(Vec::new(), 0.0, Vec::new()),
    };
    // 検索での裏付け: 言語モデルの上位の候補と元の読みだけ、前後の文脈ごとの完全一致を検索で調べて加点する(検索の回数を抑える)。
    let mut scores = scores;
    if let Some(w) = web {
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let mut check: Vec<usize> = order.into_iter().take(2).collect();
        if !check.contains(&0) {
            check.push(0);
        }
        for i in check {
            scores[i] += WEB_BONUS * web_hits(w, &prefix, &cands[i].0, &item.suffix) as f32;
        }
    }
    let mut ranked: Vec<Scored> = cands.iter().zip(&scores).map(|((t, _), &l)| Scored { text: t.clone(), logp: l }).collect();
    // 形の似た字の表に基づく候補(見た目の根拠がある)は `margin`、モデルの自由な予測だけの候補は倍の差を要求する。
    let best_of = |table: bool| (1..cands.len()).filter(|&i| cands[i].1 == table).max_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let pick = best_of(true).filter(|&i| scores[i] - scores[0] >= margin).or_else(|| best_of(false).filter(|&i| scores[i] - scores[0] >= margin * 2.0));
    ranked.sort_by(|a, b| b.logp.total_cmp(&a.logp));
    let all = if return_all { ranked.clone() } else { Vec::new() };
    ranked.truncate(8);
    match pick {
        Some(i) => FillResult { index, suspect: item.suspect.clone(), chosen: cands[i].0.clone(), changed: true, margin: scores[i] - scores[0], top: ranked, orig_logp: scores[0], all },
        None => unchanged(ranked, scores[0], all),
    }
}

/// 複数箇所を順に穴埋めする(同期・CPU 重い。`spawn_blocking` から呼ぶ)。
pub fn fill_items(lm: &dyn Lm, web: Option<&dyn Web>, items: &[FillItem], margin: f32, max_chars: usize, return_all: bool, lm_free: bool) -> Vec<FillResult> {
    let started = Instant::now();
    items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            if started.elapsed() > DEADLINE {
                FillResult { index: i, suspect: it.suspect.clone(), chosen: it.suspect.clone(), changed: false, margin: 0.0, top: Vec::new(), orig_logp: 0.0, all: Vec::new() }
            } else {
                fill_one(lm, web, i, it, margin, max_chars, return_all, lm_free)
            }
        })
        .collect()
}

/// 提案 1 件: 疑わしい文字列の `pos` 文字目(`gap` のときは 0 = 挿入位置)に入り得る 1 文字と、言語モデルの次文字確率。
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct Proposal {
    pub pos: usize,
    pub text: String,
    /// 言語モデルの次の文字としての対数確率(形の似た字の表から来た候補で、モデルの上位に無いときは `None`)。
    pub lm: Option<f32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct ProposeResult {
    pub index: usize,
    pub proposals: Vec<Proposal>,
}

/// 疑わしい箇所ごとに、入り得る 1 文字の候補を広く挙げる(アプリが字形で絞り込む前提。後ろの文脈は使わないので軽い)。
/// 位置ごとに、前の文脈 + 手前までの疑わしい文字列を文脈にして、言語モデルの次の文字の上位 `k` 個と、形の似た字を返す。
pub fn propose_one(lm: &dyn Lm, index: usize, item: &FillItem, k: usize) -> ProposeResult {
    let prefix = if item.prefix.is_empty() { "\n".to_string() } else { tail_chars(&item.prefix, PREFIX_CHARS) };
    let s: Vec<char> = item.suspect.chars().collect();
    let positions: Vec<usize> = if item.gap || s.is_empty() { vec![0] } else { (0..s.len().min(6)).collect() };
    let mut proposals: Vec<Proposal> = Vec::new();
    for pos in positions {
        let before: String = s.iter().take(if item.gap { s.len() } else { pos }).collect();
        let ctx = format!("{prefix}{before}");
        if let Ok(top) = lm.top_next(&ctx, k) {
            for (t, lp) in top {
                if usable(&t, 1) && !proposals.iter().any(|p| p.pos == pos && p.text == t) {
                    proposals.push(Proposal { pos, text: t, lm: Some(lp) });
                }
            }
        }
        if !item.gap && pos < s.len() {
            for alt in confusables(s[pos]) {
                let t = alt.to_string();
                if !proposals.iter().any(|p| p.pos == pos && p.text == t) {
                    proposals.push(Proposal { pos, text: t, lm: None });
                }
            }
        }
    }
    ProposeResult { index, proposals }
}

pub fn propose_items(lm: &dyn Lm, items: &[FillItem], k: usize) -> Vec<ProposeResult> {
    let started = Instant::now();
    items.iter().enumerate().map(|(i, it)| if started.elapsed() > DEADLINE { ProposeResult { index: i, proposals: Vec::new() } } else { propose_one(lm, i, it, k) }).collect()
}

/// 走査 1 件: `prefix`(前の行などの文脈。検査はしない)のあとの `text`(検査する 1 行)。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScanItem {
    #[serde(default)]
    pub prefix: String,
    pub text: String,
}

/// 不自然な箇所: `text` の中の文字の範囲と、その言語モデルの対数確率(低いほど不自然)。
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct ScanSpot {
    pub start: usize,
    pub len: usize,
    pub text: String,
    pub logp: f32,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct ScanResult {
    pub index: usize,
    pub spots: Vec<ScanSpot>,
}

/// 1 行を走査して、言語モデルにとって不自然(対数確率が `threshold` 未満)なトークンを `max_spots` 個まで挙げる。
/// OCR が別の字に読み間違えた箇所(「毎朝」→「耕朝」など)を、前後の文脈から見つける。
pub fn scan_one(lm: &dyn Lm, index: usize, item: &ScanItem, threshold: f32, max_spots: usize) -> ScanResult {
    let prefix = tail_chars(&item.prefix, PREFIX_CHARS);
    let pc = prefix.chars().count();
    let full = format!("{prefix}{}", item.text);
    let chars: Vec<char> = full.chars().collect();
    let Ok(spans) = lm.scan(&full) else {
        return ScanResult { index, spots: Vec::new() };
    };
    let mut spots: Vec<ScanSpot> = spans
        .into_iter()
        .filter(|&(start, len, lp)| start >= pc && start + len <= chars.len() && len <= MAX_SPAN && lp < threshold)
        .map(|(start, len, lp)| ScanSpot { start: start - pc, len, text: chars[start..start + len].iter().collect(), logp: lp })
        // 句読点・記号・空白だけのトークンは、読み違いというより文の癖なので対象にしない。
        .filter(|s| s.text.chars().any(|c| c.is_alphanumeric()))
        .collect();
    // 隣り合う(または 1 文字だけ挟んだ)不自然なトークンは、1 か所の連続した読み違いとして、最大 `MAX_SPAN` 文字までまとめる。
    spots.sort_by_key(|s| s.start);
    let mut merged: Vec<ScanSpot> = Vec::new();
    for sp in spots {
        match merged.last_mut() {
            Some(last) if sp.start <= last.start + last.len + 1 && sp.start + sp.len - last.start <= MAX_SPAN => {
                let end = sp.start + sp.len;
                last.len = end - last.start;
                last.text = chars[pc + last.start..pc + last.start + last.len].iter().collect();
                last.logp = last.logp.min(sp.logp);
            }
            _ => merged.push(sp),
        }
    }
    merged.sort_by(|a, b| a.logp.total_cmp(&b.logp));
    merged.truncate(max_spots);
    merged.sort_by_key(|s| s.start);
    ScanResult { index, spots: merged }
}

pub fn scan_items(lm: &dyn Lm, items: &[ScanItem], threshold: f32, max_spots: usize) -> Vec<ScanResult> {
    let started = Instant::now();
    items.iter().enumerate().map(|(i, it)| if started.elapsed() > DEADLINE { ScanResult { index: i, spots: Vec::new() } } else { scan_one(lm, i, it, threshold, max_spots) }).collect()
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
        let item = FillItem { prefix: "本".into(), suspect: "曰".into(), suffix: "は晴天なり".into(), gap: false, extra: vec![], conf: -1.0 };
        let r = fill_one(&lm, None, 0, &item, 3.0, 2, false, true);
        assert!(r.changed, "{r:?}");
        assert_eq!(r.chosen, "日");
        assert!(r.margin >= 3.0);
    }

    #[test]
    fn keeps_the_reading_when_it_is_already_the_best() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: "日".into(), suffix: "は晴天なり".into(), gap: false, extra: vec![], conf: -1.0 };
        let r = fill_one(&lm, None, 0, &item, 3.0, 2, false, true);
        assert!(!r.changed);
        assert_eq!(r.chosen, "日");
    }

    #[test]
    fn inserts_a_missing_character_in_gap_mode() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: String::new(), suffix: "は晴天なり".into(), gap: true, extra: vec![], conf: -1.0 };
        let r = fill_one(&lm, None, 0, &item, 3.0, 2, false, true);
        assert!(r.changed, "{r:?}");
        assert_eq!(r.chosen, "日");
    }

    #[test]
    fn never_changes_more_than_one_position_and_respects_the_margin() {
        let lm = Fake { truth: "ZZZZ" };
        let item = FillItem { prefix: "x".into(), suspect: "曰目".into(), suffix: "y".into(), gap: false, extra: vec![], conf: -1.0 };
        let r = fill_one(&lm, None, 0, &item, 3.0, 2, false, true);
        assert!(!r.changed, "no candidate reaches the truth, nothing should change: {r:?}");
        // 候補は全て、元の読みから 1 文字しか違わない。
        for c in confusables('曰') {
            assert_ne!(c, '曰');
        }
    }

    #[test]
    fn extra_candidates_from_the_app_are_scored_and_all_scores_can_be_returned() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: "X".into(), suffix: "は晴天なり".into(), gap: false, extra: vec!["日".into()], conf: -1.0 };
        let r = fill_one(&lm, None, 0, &item, 3.0, 2, true, true);
        assert!(r.changed && r.chosen == "日", "{r:?}");
        assert!(r.all.iter().any(|s| s.text == "X") && r.all.iter().any(|s| s.text == "日"));
        assert!(r.orig_logp < 0.0);
    }

    #[test]
    fn deleting_a_single_character_needs_a_low_confidence() {
        struct PreferEmpty;
        impl Lm for PreferEmpty {
            fn score(&self, _p: &str, conts: &[String]) -> Result<Vec<f32>, String> {
                Ok(conts.iter().map(|c| if c.starts_with('を') { -1.0 } else { -20.0 }).collect())
            }
            fn top_next(&self, _p: &str, _k: usize) -> Result<Vec<(String, f32)>, String> {
                Ok(vec![])
            }
        }
        let mk = |conf: f32| FillItem { prefix: "署名と".into(), suspect: "押".into(), suffix: "をお願い".into(), gap: false, extra: vec![], conf };
        assert!(!fill_one(&PreferEmpty, None, 0, &mk(90.0), 3.0, 2, false, true).changed, "confident reading must not be deleted");
        assert!(!fill_one(&PreferEmpty, None, 0, &mk(-1.0), 3.0, 2, false, true).changed, "unknown confidence must not be deleted");
        assert!(fill_one(&PreferEmpty, None, 0, &mk(30.0), 3.0, 2, false, true).changed);
    }

    #[test]
    fn propose_lists_model_and_look_alike_characters_per_position() {
        let lm = Fake { truth: "本日は晴天なり" };
        let item = FillItem { prefix: "本".into(), suspect: "曰".into(), suffix: String::new(), gap: false, extra: vec![], conf: 30.0 };
        let r = propose_one(&lm, 0, &item, 8);
        assert!(r.proposals.iter().any(|p| p.text == "日" && p.pos == 0));
        assert!(r.proposals.iter().any(|p| p.text == "目" && p.lm.is_none()), "look-alikes come without an lm score");
    }

    #[test]
    fn scan_flags_the_least_likely_tokens_and_ignores_the_context_and_punctuation() {
        struct S;
        impl Lm for S {
            fn score(&self, _p: &str, _c: &[String]) -> Result<Vec<f32>, String> {
                Ok(vec![])
            }
            fn top_next(&self, _p: &str, _k: usize) -> Result<Vec<(String, f32)>, String> {
                Ok(vec![])
            }
            fn scan(&self, text: &str) -> Result<Vec<(usize, usize, f32)>, String> {
                // 1 文字 = 1 トークン。「耕」と、句読点の「、」と、文脈中の「Z」だけ低い確率にする。
                Ok(text.chars().enumerate().map(|(i, c)| (i, 1, if matches!(c, '耕' | '、' | 'Z') { -12.0 } else { -1.0 })).collect())
            }
        }
        let item = ScanItem { prefix: "Z前".into(), text: "毎耕朝、晴".into() };
        let r = scan_one(&S, 0, &item, -8.0, 4);
        assert_eq!(r.spots.len(), 1, "{r:?}");
        assert_eq!((r.spots[0].start, r.spots[0].text.as_str()), (1, "耕"));
    }

    struct FakeWeb(Vec<&'static str>);
    impl Web for FakeWeb {
        fn snippets(&self, _q: &str) -> Vec<String> {
            self.0.iter().map(|s| s.to_string()).collect()
        }
    }

    #[test]
    fn web_mines_the_text_between_prefix_and_suffix_from_snippets() {
        let web = FakeWeb(vec!["... 私たちは 毎朝、公園で 散歩 ...", "関係のない文章です"]);
        let got = web_candidates(&web, "今日から私たちは", "朝、公園で散歩をして", 2);
        assert!(got.contains(&"毎".to_string()), "{got:?}");
    }

    #[test]
    fn web_hits_count_exact_phrase_matches_only() {
        let web = FakeWeb(vec!["私たちは毎朝、公園で散歩", "私たちは毎朝、公園で", "私たちは耕朝、公園で"]);
        assert_eq!(web_hits(&web, "私たちは", "毎", "朝、公園で"), 2);
        assert_eq!(web_hits(&web, "私たちは", "耕", "朝、公園で"), 1);
        assert_eq!(web_hits(&web, "私たちは", "艇", "朝、公園で"), 0);
    }

    #[test]
    fn a_multi_character_misread_is_fixed_with_the_web_and_the_model() {
        // 「本日わ天気」の「日わ」(2 文字)を、実文(検索)で裏付けられる「日は」に直す。
        let lm = Fake { truth: "本日は晴天なり" };
        let web = FakeWeb(vec!["本日は晴天なり と書かれています", "本日は晴天なりの意味"]);
        let item = FillItem { prefix: "本".into(), suspect: "曰わ".into(), suffix: "晴天なり".into(), gap: false, extra: vec![], conf: -1.0 };
        let r = fill_one(&lm, Some(&web), 0, &item, 3.0, 2, false, true);
        assert!(r.changed && r.chosen == "日は", "{r:?}");
    }

    #[test]
    fn scan_merges_adjacent_unlikely_tokens_up_to_four_characters() {
        struct S;
        impl Lm for S {
            fn score(&self, _p: &str, _c: &[String]) -> Result<Vec<f32>, String> {
                Ok(vec![])
            }
            fn top_next(&self, _p: &str, _k: usize) -> Result<Vec<(String, f32)>, String> {
                Ok(vec![])
            }
            fn scan(&self, text: &str) -> Result<Vec<(usize, usize, f32)>, String> {
                Ok(text.chars().enumerate().map(|(i, c)| (i, 1, if matches!(c, '耕' | '朝' | '艇') { -12.0 } else { -1.0 })).collect())
            }
        }
        let item = ScanItem { prefix: String::new(), text: "私は耕朝のあいうえお艇".into() };
        let r = scan_one(&S, 0, &item, -8.0, 4);
        let found: Vec<(usize, usize, &str)> = r.spots.iter().map(|s| (s.start, s.len, s.text.as_str())).collect();
        assert_eq!(found, vec![(2, 2, "耕朝"), (10, 1, "艇")], "{found:?}");
    }

    #[test]
    fn usable_filters_junk_tokens() {
        assert!(usable("日", 2) && usable("日本", 2));
        assert!(!usable("日本語", 2) && !usable(" ", 2) && !usable("\u{FFFD}", 2) && !usable("\n", 2));
    }
}
