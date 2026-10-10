//! 画像の文字認識(OCR)エンドポイント `POST /v1/ocr` の実体(2026-10-10追加)。
//!
//! `pdf-r2l-win` 等の PDF 綴じ方向変更アプリ v1.0.1 の「文字を読み取り、元の書体に
//! 合わせて文字を置き換え、検索できるようにする」機能のためのサーバー側。
//!
//! # 実装方式: Tesseract CLI の子プロセス
//!
//! `transcribe.rs`(whisper.cpp CLI)と同じパターン。`tesseract <画像> stdout -l <言語> tsv`
//! を起動し、TSV(単語ごとの外接矩形・信頼度)を行ごとにまとめて返す。C++リンクは不要で、
//! 実行ファイル(`tesseract`)と言語データが実在する場合だけ動作する。無ければ `503`。
//!
//! # 高速化
//!
//! - **ページをまとめて OCR**: 複数ページの画像を画像リスト(`.txt`)にして **1回の tesseract 起動**で
//!   処理し、起動と言語データ読み込みの回数を減らす(`recognize_batch`)。
//! - 言語データは必要な言語だけを取得してキャッシュする(`ocr_data.rs`)。`fast` を選べば小さく速い。
//!
//! # 資源保護
//!
//! 公開サーバーで CPU を使い切られないよう、同時実行数・画像サイズ・実行時間に上限を設ける。

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// 画像1枚(デコード後)の上限バイト数。
pub const MAX_IMAGE_BYTES: usize = 12 * 1024 * 1024;
/// 1回の要求で受け付ける画像(ページ)数の上限。
pub const MAX_BATCH_IMAGES: usize = 16;
/// 1回の要求の画像合計(デコード後)の上限バイト数。
pub const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;
/// tesseract の実行時間上限(バッチ全体)。
const TIMEOUT: Duration = Duration::from_secs(600);
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

/// `tesseract` 実行ファイルのパス。`ARUARU_LLM_TESSERACT` で上書き可、既定は PATH 上の `tesseract`。
pub fn cli_path() -> PathBuf {
    std::env::var("ARUARU_LLM_TESSERACT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("tesseract"))
}

/// `tesseract` が起動できるか。
pub async fn cli_available() -> bool {
    tokio::process::Command::new(cli_path())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
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

/// 複数画像の TSV をページごとに分けて行にまとめる。`page_num` 列(2列目)がページ番号。
/// 列: level page block par line word left top width height conf text
pub fn parse_tsv_pages(tsv: &str) -> Vec<Vec<OcrLine>> {
    let mut pages: Vec<(i32, Vec<&str>)> = Vec::new();
    for row in tsv.lines().skip(1) {
        let page = row.split('\t').nth(1).and_then(|p| p.trim().parse::<i32>().ok()).unwrap_or(1);
        match pages.last_mut() {
            Some((p, rows)) if *p == page => rows.push(row),
            _ => pages.push((page, vec![row])),
        }
    }
    pages.into_iter().map(|(_, rows)| parse_rows(&rows)).collect()
}

/// 1ページぶんの TSV を行にまとめる(ヘッダー付き)。
pub fn parse_tsv(tsv: &str) -> Vec<OcrLine> {
    parse_tsv_pages(tsv).into_iter().next().unwrap_or_default()
}

fn parse_rows(rows: &[&str]) -> Vec<OcrLine> {
    let mut lines: Vec<((i32, i32, i32), OcrLine)> = Vec::new();
    for row in rows {
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
            // 日本語などは単語間に空白を入れず、英数字どうしが隣り合う所だけ空白で区切る。
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

/// 複数の画像(ページ)を **1回の tesseract 起動**で OCR する。戻り値は画像と同じ順のページごとの行。
/// `languages` は検証済みの言語コード、`tessdata_dir` は `--tessdata-dir` に渡す場所。
pub async fn recognize_batch(images: &[Vec<u8>], languages: &[String], tessdata_dir: &Path) -> Result<Vec<Vec<OcrLine>>, String> {
    if images.is_empty() {
        return Err("no images".into());
    }
    let mut exts = Vec::new();
    for img in images {
        exts.push(image_ext(img).ok_or("image must be PNG or JPEG")?);
    }
    let _permit = SEMAPHORE.acquire().await.map_err(|e| e.to_string())?;
    let dir = scratch_dir().map_err(|e| format!("scratch dir: {e}"))?;
    let result = run_tesseract(&dir, images, &exts, languages, tessdata_dir).await;
    let _ = std::fs::remove_dir_all(&dir);
    result
}

async fn run_tesseract(dir: &Path, images: &[Vec<u8>], exts: &[&str], languages: &[String], tessdata_dir: &Path) -> Result<Vec<Vec<OcrLine>>, String> {
    let mut list = String::new();
    for (i, (img, ext)) in images.iter().zip(exts).enumerate() {
        let p = dir.join(format!("page{i:03}.{ext}"));
        std::fs::write(&p, img).map_err(|e| format!("write image: {e}"))?;
        list.push_str(&p.to_string_lossy());
        list.push('\n');
    }
    let list_path = dir.join("pages.txt");
    std::fs::write(&list_path, list).map_err(|e| format!("write list: {e}"))?;
    let child = tokio::process::Command::new(cli_path())
        .arg(&list_path)
        .arg("stdout")
        .arg("--tessdata-dir")
        .arg(tessdata_dir)
        .args(["-l", &languages.join("+"), "--psm", "3", "tsv"])
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
    let mut pages = parse_tsv_pages(&String::from_utf8_lossy(&out.stdout));
    // 文字が1つも見つからなかったページは TSV に行が出ないことがあるため、ページ数を画像数に揃える。
    pages.resize(images.len(), Vec::new());
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TSV: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
1\t1\t0\t0\t0\t0\t0\t0\t800\t600\t-1\t\n\
5\t1\t1\t1\t1\t1\t10\t20\t50\t30\t96.5\tHello\n\
5\t1\t1\t1\t1\t2\t70\t22\t60\t28\t90.0\tworld\n\
5\t1\t1\t1\t2\t1\t10\t60\t80\t30\t88.0\t日本語\n\
5\t1\t1\t1\t2\t2\t95\t60\t40\t30\t-1\t\n\
5\t2\t1\t1\t1\t1\t5\t6\t70\t20\t77.0\tSecond\n";

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
    fn splits_pages() {
        let pages = parse_tsv_pages(TSV);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].len(), 2);
        assert_eq!(pages[1].len(), 1);
        assert_eq!(pages[1][0].text, "Second");
    }

    #[test]
    fn detects_image_formats() {
        assert_eq!(image_ext(&[0x89, b'P', b'N', b'G', 0]), Some("png"));
        assert_eq!(image_ext(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpg"));
        assert_eq!(image_ext(b"GIF89a"), None);
    }
}
