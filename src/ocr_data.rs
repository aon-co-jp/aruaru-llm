//! OCR の言語データ(Tesseract の `*.traineddata`)と、書体フォントの「必要な時だけ取得してキャッシュ」
//! する仕組み(2026-10-10追加)。`POST /v1/ocr` と PDF 綴じ方向アプリ v1.0.1 が使う。
//!
//! # 取得元(上から順に試す)
//! 1. 非公開リポジトリ `aon-co-jp/tessdata-jpn`(環境変数 `ARUARU_LLM_TESSDATA_TOKEN` があるときだけ)
//! 2. 公開の上流: `tesseract-ocr/tessdata_{best,fast}`(言語データ)、`notofonts`(フォント、いずれも OFL/Apache-2.0)
//!
//! 一度取得したファイルは `ARUARU_LLM_TESSDATA_DIR`(既定 `<crate>/models/tessdata`)にキャッシュし、
//! 以後は取得しない。言語は1回の要求で最大 [`MAX_LANGS`] 個まで。`manifest.json`(非公開リポジトリ)が
//! 取得できたときは SHA-256 を検証する。

use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// 1回の OCR で選べる言語数の上限。
pub const MAX_LANGS: usize = 12;

const PRIVATE_BASE: &str = "https://raw.githubusercontent.com/aon-co-jp/tessdata-jpn/main";

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct LangInfo {
    pub code: &'static str,
    pub name_ja: &'static str,
    pub name_en: &'static str,
}

macro_rules! langs {
    ($(($c:expr, $j:expr, $e:expr)),* $(,)?) => { &[$(LangInfo { code: $c, name_ja: $j, name_en: $e }),*] };
}

/// 対応言語(`tessdata-jpn/manifest.json` と同じ50言語)。`osd`(向き判定)は言語としては選べない。
pub const LANGUAGES: &[LangInfo] = langs![
    ("jpn", "日本語(横書き)", "Japanese (horizontal)"),
    ("jpn_vert", "日本語(縦書き)", "Japanese (vertical)"),
    ("eng", "英語", "English"),
    ("chi_sim", "中国語(簡体字)", "Chinese (Simplified)"),
    ("chi_sim_vert", "中国語(簡体字・縦書き)", "Chinese (Simplified, vertical)"),
    ("chi_tra", "中国語(繁体字・台湾/香港)", "Chinese (Traditional, Taiwan/Hong Kong)"),
    ("chi_tra_vert", "中国語(繁体字・縦書き)", "Chinese (Traditional, vertical)"),
    ("kor", "韓国語", "Korean"),
    ("kor_vert", "韓国語(縦書き)", "Korean (vertical)"),
    ("rus", "ロシア語", "Russian"),
    ("ukr", "ウクライナ語", "Ukrainian"),
    ("fas", "ペルシャ語(イラン)", "Persian (Iran)"),
    ("ara", "アラビア語", "Arabic"),
    ("deu", "ドイツ語", "German"),
    ("fra", "フランス語", "French"),
    ("spa", "スペイン語", "Spanish"),
    ("ita", "イタリア語", "Italian"),
    ("por", "ポルトガル語", "Portuguese"),
    ("nld", "オランダ語", "Dutch"),
    ("pol", "ポーランド語", "Polish"),
    ("ces", "チェコ語", "Czech"),
    ("slk", "スロバキア語", "Slovak"),
    ("hun", "ハンガリー語", "Hungarian"),
    ("ron", "ルーマニア語", "Romanian"),
    ("bul", "ブルガリア語", "Bulgarian"),
    ("ell", "ギリシャ語", "Greek"),
    ("swe", "スウェーデン語", "Swedish"),
    ("dan", "デンマーク語", "Danish"),
    ("nor", "ノルウェー語", "Norwegian"),
    ("fin", "フィンランド語", "Finnish"),
    ("hrv", "クロアチア語", "Croatian"),
    ("slv", "スロベニア語", "Slovenian"),
    ("srp", "セルビア語", "Serbian"),
    ("lit", "リトアニア語", "Lithuanian"),
    ("lav", "ラトビア語", "Latvian"),
    ("est", "エストニア語", "Estonian"),
    ("cat", "カタルーニャ語", "Catalan"),
    ("eus", "バスク語", "Basque"),
    ("glg", "ガリシア語", "Galician"),
    ("isl", "アイスランド語", "Icelandic"),
    ("gle", "アイルランド語", "Irish"),
    ("mlt", "マルタ語", "Maltese"),
    ("sqi", "アルバニア語", "Albanian"),
    ("bos", "ボスニア語", "Bosnian"),
    ("mkd", "マケドニア語", "Macedonian"),
    ("bel", "ベラルーシ語", "Belarusian"),
    ("tur", "トルコ語", "Turkish"),
    ("heb", "ヘブライ語", "Hebrew"),
    ("hin", "ヒンディー語", "Hindi"),
];

/// 書体フォント(`tessdata-jpn/fonts/` と同名)。CJK は PDF に埋め込みやすい TrueType(glyf)の可変フォント Subset 版。
pub const FONTS: &[(&str, &str)] = &[
    ("NotoSansJP-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansJP-VF.ttf"),
    ("NotoSerifJP-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Serif/Variable/TTF/Subset/NotoSerifJP-VF.ttf"),
    ("NotoSansSC-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansSC-VF.ttf"),
    ("NotoSerifSC-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Serif/Variable/TTF/Subset/NotoSerifSC-VF.ttf"),
    ("NotoSansTC-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansTC-VF.ttf"),
    ("NotoSerifTC-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Serif/Variable/TTF/Subset/NotoSerifTC-VF.ttf"),
    ("NotoSansKR-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansKR-VF.ttf"),
    ("NotoSerifKR-VF.ttf", "https://github.com/notofonts/noto-cjk/raw/main/Serif/Variable/TTF/Subset/NotoSerifKR-VF.ttf"),
    ("NotoSans-Regular.ttf", "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf"),
    ("NotoSans-Bold.ttf", "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Bold.ttf"),
    ("NotoSerif-Regular.ttf", "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSerif/hinted/ttf/NotoSerif-Regular.ttf"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// 高精度(大きい・遅め)。
    Best,
    /// 高速(小さい・速い)。
    Fast,
}

impl Quality {
    pub fn parse(s: Option<&str>) -> Result<Quality, String> {
        match s.map(|s| s.to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("best") => Ok(Quality::Best),
            Some("fast") => Ok(Quality::Fast),
            Some(other) => Err(format!("quality must be \"best\" or \"fast\" (got {other})")),
        }
    }
    pub fn dir_name(self) -> &'static str {
        match self {
            Quality::Best => "best",
            Quality::Fast => "fast",
        }
    }
}

/// キャッシュの保存先ルート。
pub fn data_root() -> PathBuf {
    std::env::var("ARUARU_LLM_TESSDATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join("tessdata"))
}

/// `--tessdata-dir` に渡すディレクトリ(`best` か `fast`)。
pub fn variant_dir(q: Quality) -> PathBuf {
    data_root().join(q.dir_name())
}

pub fn fonts_dir() -> PathBuf {
    data_root().join("fonts")
}

pub fn is_known_language(code: &str) -> bool {
    LANGUAGES.iter().any(|l| l.code == code)
}

pub fn lang_cached(q: Quality, code: &str) -> bool {
    variant_dir(q).join(format!("{code}.traineddata")).is_file()
}

/// `jpn+eng` 形式または配列の言語指定を検証し、重複を除いた言語コードの列にする(1〜12個)。
pub fn normalize_languages(codes: &[String]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for c in codes {
        let c = c.trim();
        if c.is_empty() {
            continue;
        }
        if !is_known_language(c) {
            return Err(format!("unsupported language: {c}"));
        }
        if !out.iter().any(|o| o == c) {
            out.push(c.to_string());
        }
    }
    if out.is_empty() {
        return Err("at least one language is required".to_string());
    }
    if out.len() > MAX_LANGS {
        return Err(format!("at most {MAX_LANGS} languages can be selected (got {})", out.len()));
    }
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .map_err(|e| e.to_string())
}

fn token() -> Option<String> {
    std::env::var("ARUARU_LLM_TESSDATA_TOKEN").ok().filter(|t| !t.is_empty())
}

async fn fetch(url: &str, with_token: bool) -> Result<Vec<u8>, String> {
    let mut req = client()?.get(url);
    if with_token {
        if let Some(t) = token() {
            req = req.header("Authorization", format!("token {t}"));
        }
    }
    let resp = req.send().await.map_err(|e| format!("{url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("{url}: HTTP {}", resp.status()));
    }
    Ok(resp.bytes().await.map_err(|e| format!("{url}: {e}"))?.to_vec())
}

/// 非公開リポジトリの `manifest.json` から `lang`/`variant` の SHA-256 を引く(取得できなければ None)。
async fn manifest_sha(variant: &str, lang: &str) -> Option<String> {
    token()?;
    let body = fetch(&format!("{PRIVATE_BASE}/manifest.json"), true).await.ok()?;
    let v: serde_json::Value = serde_json::from_slice(&body).ok()?;
    v["languages"][lang][variant]["sha256"].as_str().map(|s| s.to_string())
}

fn write_atomic(path: &std::path::Path, data: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, data).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

/// 1言語ぶんを取得してキャッシュする(既にあれば何もしない)。取得した場合 true。
pub async fn ensure_language(q: Quality, code: &str) -> Result<bool, String> {
    if !is_known_language(code) {
        return Err(format!("unsupported language: {code}"));
    }
    if lang_cached(q, code) {
        return Ok(false);
    }
    let variant = q.dir_name();
    let mut sources: Vec<(String, bool)> = Vec::new();
    if token().is_some() {
        sources.push((format!("{PRIVATE_BASE}/{variant}/{code}.traineddata"), true));
    }
    sources.push((format!("https://github.com/tesseract-ocr/tessdata_{variant}/raw/main/{code}.traineddata"), false));
    let expected = manifest_sha(variant, code).await;
    let mut last_err = String::new();
    for (url, with_token) in sources {
        match fetch(&url, with_token).await {
            Ok(data) => {
                if data.len() < 50_000 {
                    last_err = format!("{url}: unexpectedly small file ({} bytes)", data.len());
                    continue;
                }
                if let Some(exp) = &expected {
                    let got = hex(&Sha256::digest(&data));
                    if &got != exp {
                        last_err = format!("{url}: SHA-256 mismatch");
                        continue;
                    }
                }
                write_atomic(&variant_dir(q).join(format!("{code}.traineddata")), &data)?;
                return Ok(true);
            }
            Err(e) => last_err = e,
        }
    }
    Err(format!("could not download language data '{code}': {last_err}"))
}

/// 複数言語を並行して取得する。取得した(新規にキャッシュした)言語コードの一覧を返す。
pub async fn ensure_languages(q: Quality, codes: &[String]) -> Result<Vec<String>, String> {
    let mut set = tokio::task::JoinSet::new();
    for c in codes {
        let c = c.clone();
        set.spawn(async move { ensure_language(q, &c).await.map(|new| (c, new)) });
    }
    let mut fetched = Vec::new();
    let mut first_err: Option<String> = None;
    while let Some(r) = set.join_next().await {
        match r.map_err(|e| e.to_string()).and_then(|x| x) {
            Ok((c, true)) => fetched.push(c),
            Ok(_) => {}
            Err(e) => first_err = first_err.or(Some(e)),
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(fetched),
    }
}

pub fn is_known_font(name: &str) -> bool {
    FONTS.iter().any(|(n, _)| *n == name)
}

/// フォントを取得してキャッシュし、ファイルパスを返す。
pub async fn ensure_font(name: &str) -> Result<PathBuf, String> {
    let (_, upstream) = FONTS.iter().find(|(n, _)| *n == name).ok_or_else(|| format!("unknown font: {name}"))?;
    let path = fonts_dir().join(name);
    if path.is_file() {
        return Ok(path);
    }
    let mut sources: Vec<(String, bool)> = Vec::new();
    if token().is_some() {
        sources.push((format!("{PRIVATE_BASE}/fonts/{name}"), true));
    }
    sources.push((upstream.to_string(), false));
    let mut last_err = String::new();
    for (url, with_token) in sources {
        match fetch(&url, with_token).await {
            Ok(data) if data.len() > 100_000 => {
                write_atomic(&path, &data)?;
                return Ok(path);
            }
            Ok(data) => last_err = format!("{url}: unexpectedly small file ({} bytes)", data.len()),
            Err(e) => last_err = e,
        }
    }
    Err(format!("could not download font '{name}': {last_err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_table_has_unique_codes() {
        let mut codes: Vec<_> = LANGUAGES.iter().map(|l| l.code).collect();
        let n = codes.len();
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), n);
        assert_eq!(n, 49); // tessdata-jpn の50ファイルから osd(向き判定)を除いた言語
    }

    #[test]
    fn normalizes_and_limits_languages() {
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(normalize_languages(&v(&["jpn", "eng", "jpn"])).unwrap(), v(&["jpn", "eng"]));
        assert!(normalize_languages(&v(&[])).is_err());
        assert!(normalize_languages(&v(&["xxx"])).is_err());
        let thirteen: Vec<String> = LANGUAGES.iter().take(13).map(|l| l.code.to_string()).collect();
        assert!(normalize_languages(&thirteen).is_err());
        let twelve: Vec<String> = LANGUAGES.iter().take(12).map(|l| l.code.to_string()).collect();
        assert_eq!(normalize_languages(&twelve).unwrap().len(), 12);
    }

    #[test]
    fn quality_parses() {
        assert!(Quality::parse(None).unwrap() == Quality::Best);
        assert!(Quality::parse(Some("FAST")).unwrap() == Quality::Fast);
        assert!(Quality::parse(Some("x")).is_err());
    }
}
