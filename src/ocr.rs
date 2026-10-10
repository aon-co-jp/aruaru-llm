//! 画像の文字認識(OCR)エンドポイント `POST /v1/ocr` の実体(2026-10-10追加)。
//!
//! `pdf-r2l-win` 等の PDF 綴じ方向変更アプリ v1.0.1 の「文字を読み取り、元の書体に
//! 合わせて文字を置き換え、検索できるようにする」機能のためのサーバー側。
//!
//! # 実装方式: Tesseract CLI の子プロセス
//!
//! `transcribe.rs`(whisper.cpp CLI)と同じパターン。`tesseract <画像> stdout -l <言語> tsv`
//! を起動し、TSV(単語ごとの外接矩形・信頼度)を行ごとにまとめて返す。C++リンクは不要で、
//! 実行ファイル(`tesseract`)と言語データ(`jpn` など)が実在する場合だけ動作する。
//! 無ければ `503` と導入方法を返す。
//!
//! # 資源保護
//!
//! 公開サーバーで CPU を使い切られないよう、同時実行数・画像サイズ・実行時間に上限を設ける。

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// 画像(デコード後)の上限バイト数。
pub const MAX_IMAGE_BYTES: usize = 12 * 1024 * 1024;
/// tesseract の実行時間上限。
const TIMEOUT: Duration = Duration::from_secs(120);
/// 同時に走らせる tesseract の数(VPS の CPU を守る)。
const MAX_CONCURRENT: usize = 2;

static SEMAPHORE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(MAX_CONCURRENT);

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct OcrWord {
    pub text: String,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub conf: f32,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct OcrLine {
    pub text: String,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    /// 行内の単語の平均信頼度(0〜100)。
    pub conf: f32,
    pub words: Vec<OcrWord>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OcrOutput {
    pub lines: Vec<OcrLine>,
}

/// `tesseract` 実行ファイルのパス。`ARUARU_LLM_TESSERACT` で上書き可、既定は PATH 上の `tesseract`。
pub fn cli_path() -> PathBuf {
    std::env::var("ARUARU_LLM_TESSERACT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("tesseract"))
}

/// 言語指定の検証(`jpn+eng` のような英数字と `_` と `+` のみ。コマンド注入防止)。
pub fn valid_languages(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 40
        && s.split('+').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
}

/// インストール済みの言語データ(`tesseract --list-langs`)。実行できなければ None。
pub async fn installed_languages() -> Option<Vec<String>> {
    let out = tokio::process::Command::new(cli_path())
        .arg("--list-langs")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(text.lines().skip(1).map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
}

/// 画像のマジックバイトで形式を判定(PNG / JPEG のみ受理)。
pub fn image_ext(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else {
        None
    }
}

/// tesseract の TSV 出力を行(block/par/line 単位)にまとめる。
/// 列: level page block par line word left top width height conf text
pub fn parse_tsv(tsv: &str) -> Vec<OcrLine> {
    let mut lines: Vec<((i32, i32, i32), OcrLine)> = Vec::new();
    for row in tsv.lines().skip(1) {
        let c: Vec<&str> = row.splitn(12, '\t').collect();
        if c.len() < 12 || c[0] != "5" {
            continue;
        }
        let text = c[11].trim();
        if text.is_empty() {
            continue;
        }
        let num = |i: usize| c[i].trim().parse::<i32>().unwrap_or(0);
        let conf = c[10].trim().parse::<f32>().unwrap_or(-1.0);
        if conf < 0.0 {
            continue;
        }
        let key = (num(2), num(3), num(4));
        let word = OcrWord { text: text.to_string(), x: num(6), y: num(7), w: num(8), h: num(9), conf };
        match lines.iter_mut().find(|(k, _)| *k == key) {
            Some((_, l)) => l.words.push(word),
            None => lines.push((key, OcrLine { text: String::new(), x: 0, y: 0, w: 0, h: 0, conf: 0.0, words: vec![word] })),
        }
    }
    lines
        .into_iter()
        .map(|(_, mut l)| {
            let x0 = l.words.iter().map(|w| w.x).min().unwrap_or(0);
            let y0 = l.words.iter().map(|w| w.y).min().unwrap_or(0);
            let x1 = l.words.iter().map(|w| w.x + w.w).max().unwrap_or(0);
            let y1 = l.words.iter().map(|w| w.y + w.h).max().unwrap_or(0);
            l.x = x0;
            l.y = y0;
            l.w = x1 - x0;
            l.h = y1 - y0;
            l.conf = l.words.iter().map(|w| w.conf).sum::<f32>() / l.words.len() as f32;
            // 日本語は単語間に空白を入れず、英数字どうしが隣り合う所だけ空白で区切る。
            let mut text = String::new();
            for w in &l.words {
                let a = text.chars().last();
                let b = w.text.chars().next();
                if let (Some(a), Some(b)) = (a, b) {
                    if a.is_ascii_alphanumeric() && b.is_ascii_alphanumeric() {
                        text.push(' ');
                    }
                }
                text.push_str(&w.text);
            }
            l.text = text;
            l
        })
        .collect()
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn scratch_dir() -> std::io::Result<PathBuf> {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("aruaru-llm-ocr-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// 画像を OCR する。`languages` は `jpn+eng` のような tesseract の言語指定。
pub async fn recognize(image: &[u8], languages: &str) -> Result<OcrOutput, String> {
    if !valid_languages(languages) {
        return Err(format!("invalid languages: {languages}"));
    }
    let ext = image_ext(image).ok_or("image must be PNG or JPEG")?;
    let _permit = SEMAPHORE.acquire().await.map_err(|e| e.to_string())?;
    let dir = scratch_dir().map_err(|e| format!("scratch dir: {e}"))?;
    let result = run_tesseract(&dir, image, ext, languages).await;
    let _ = std::fs::remove_dir_all(&dir);
    result
}

async fn run_tesseract(dir: &Path, image: &[u8], ext: &str, languages: &str) -> Result<OcrOutput, String> {
    let img_path = dir.join(format!("page.{ext}"));
    std::fs::write(&img_path, image).map_err(|e| format!("write image: {e}"))?;
    let child = tokio::process::Command::new(cli_path())
        .arg(&img_path)
        .arg("stdout")
        .args(["-l", languages, "--psm", "3", "tsv"])
        .env("OMP_THREAD_LIMIT", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("failed to start tesseract: {e}"))?;
    let out = tokio::time::timeout(TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| "tesseract timed out".to_string())?
        .map_err(|e| format!("tesseract: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("tesseract failed: {}", err.lines().next().unwrap_or("unknown error")));
    }
    Ok(OcrOutput { lines: parse_tsv(&String::from_utf8_lossy(&out.stdout)) })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TSV: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
1\t1\t0\t0\t0\t0\t0\t0\t800\t600\t-1\t\n\
5\t1\t1\t1\t1\t1\t10\t20\t50\t30\t96.5\tHello\n\
5\t1\t1\t1\t1\t2\t70\t22\t60\t28\t90.0\tworld\n\
5\t1\t1\t1\t2\t1\t10\t60\t80\t30\t88.0\t日本語\n\
5\t1\t1\t1\t2\t2\t95\t60\t40\t30\t-1\t\n";

    #[test]
    fn groups_words_into_lines() {
        let lines = parse_tsv(TSV);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "Hello world");
        assert_eq!((lines[0].x, lines[0].y, lines[0].w, lines[0].h), (10, 20, 120, 30));
        assert!((lines[0].conf - 93.25).abs() < 0.01);
        assert_eq!(lines[1].text, "日本語");
        assert_eq!(lines[1].words.len(), 1);
    }

    #[test]
    fn validates_languages_and_images() {
        assert!(valid_languages("jpn+eng"));
        assert!(valid_languages("jpn_vert"));
        assert!(!valid_languages("jpn;rm -rf"));
        assert!(!valid_languages(""));
        assert_eq!(image_ext(&[0x89, b'P', b'N', b'G', 0]), Some("png"));
        assert_eq!(image_ext(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpg"));
        assert_eq!(image_ext(b"GIF89a"), None);
    }
}
