//! 字形(グリフ)照合を **行列積(sgemm)** で行う(`POST /v1/ocr/glyph-rank`)。
//!
//! かすれた文字の升目(ページ画像の切り抜き、`N`×`N`)と、候補の文字をフォントで描いた字形を、升目の中で字形をずらしながら
//! 正規化相互相関(NCC)で比べる。升目の全ての窓(289 個)と、全ての候補の字形(頻出の漢字・かな 約 8,000 字 × 明朝/ゴシック)の
//! 内積は 1 回の行列積(289 × 256 × 約 16,000)になるので、`open-cuda` の `sgemm`(CPU 版は `open-cpu` の AVX2 カーネル、
//! GPU のあるマシンでは Vulkan など)で実行する。字形の行列は一度だけ作って覚えておく。
//! 計算の中身(窓・字形の作り方・得点の取りまとめ)は共有クレート `open-runo-glyph`(RPoem)で、アプリ側の CPU 参照実装と同じ。

use std::sync::{Arc, Mutex};

use opencuda_core::GpuDevice;
use open_runo_glyph::{windows, GlyphIndex, Glyphs, K, N, WINDOWS};

/// 使うフォント(明朝体とゴシック体。`tessdata-jpn/fonts` と同名)。
const FONT_FILES: [&str; 2] = ["NotoSerifJP-VF.ttf", "NotoSansJP-VF.ttf"];

struct Built {
    pool: Vec<char>,
    index: Arc<GlyphIndex>,
}

static INDEX: Mutex<Option<Built>> = Mutex::new(None);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RankCell {
    /// 升目の濃淡(`N`×`N`、0〜1)。
    pub map: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct RankResult {
    /// 類似度の高い上位の文字(文字, 類似度)。
    pub top: Vec<(String, f32)>,
    /// `query` に指定した文字の類似度(母集団に無い文字は含まれない)。
    pub query: Vec<(String, f32)>,
}

/// 字形の行列を用意する(母集団が変わったときだけ作り直す)。母集団 = `pool`(頻出字)+ `extra`(文書内の字など)。
pub async fn ensure_index(pool: &[char], extra: &[char]) -> Result<Arc<GlyphIndex>, String> {
    let mut want: Vec<char> = pool.to_vec();
    for &c in extra {
        if !want.contains(&c) {
            want.push(c);
        }
    }
    if let Ok(g) = INDEX.lock() {
        if let Some(b) = g.as_ref() {
            if want.iter().all(|c| b.pool.contains(c)) {
                return Ok(Arc::clone(&b.index));
            }
        }
    }
    // フォントを取得(キャッシュに無ければ、非公開リポジトリ/上流から取得)。
    let mut fonts: Vec<Vec<u8>> = Vec::new();
    for f in FONT_FILES {
        let path = crate::ocr_data::ensure_font(f).await?;
        fonts.push(std::fs::read(path).map_err(|e| e.to_string())?);
    }
    let built = tokio::task::spawn_blocking(move || {
        let refs: Vec<&[u8]> = fonts.iter().map(|v| &v[..]).collect();
        let glyphs = Glyphs::with_fonts(&refs).ok_or_else(|| "cannot read the fonts".to_string())?;
        let index = Arc::new(GlyphIndex::build(&glyphs, &want));
        Ok::<Built, String>(Built { pool: want, index })
    })
    .await
    .map_err(|e| e.to_string())??;
    let index = Arc::clone(&built.index);
    if let Ok(mut g) = INDEX.lock() {
        *g = Some(built);
    }
    Ok(index)
}

/// 升目ごとに、窓の行列 × 字形の行列を `sgemm` で計算して、上位の文字と `query` の文字の得点を返す(同期・CPU/GPU 重い)。
pub fn rank(device: &Arc<dyn GpuDevice>, index: &GlyphIndex, cells: &[RankCell], top: usize, query: &[Vec<char>]) -> Result<Vec<RankResult>, String> {
    let m = index.len();
    if m == 0 {
        return Err("empty glyph index".to_string());
    }
    let mut out = Vec::with_capacity(cells.len());
    let mut scores = vec![0f32; WINDOWS * m];
    for (i, cell) in cells.iter().enumerate() {
        if cell.map.len() != N * N {
            return Err(format!("cell {i}: expected {} values, got {}", N * N, cell.map.len()));
        }
        let a = windows(&cell.map);
        scores.iter_mut().for_each(|v| *v = 0.0);
        opencuda_blas::sgemm(device.as_ref(), WINDOWS, K, m, 1.0, &a, index.matrix(), 0.0, &mut scores, None).map_err(|e| e.to_string())?;
        let empty: Vec<char> = Vec::new();
        let (best, q) = index.reduce(&scores, top, query.get(i).unwrap_or(&empty));
        out.push(RankResult { top: best.into_iter().map(|(v, c)| (c.to_string(), v)).collect(), query: q.into_iter().map(|(c, v)| (c.to_string(), v)).collect() });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;

    #[test]
    fn sgemm_ranking_matches_the_reference_implementation() {
        let Ok(bytes) = std::fs::read("F:/tessdata-jpn/fonts/NotoSansJP-VF.ttf") else {
            return;
        };
        let glyphs = Glyphs::with_fonts(&[&bytes[..]]).unwrap();
        let pool: Vec<char> = "日曰目白毎朝耕聞末未持特待私契約".chars().collect();
        let index = GlyphIndex::build(&glyphs, &pool);
        // 「毎」の字形を升目のずれた位置に置く。
        let g = glyphs.render(0, '毎').unwrap();
        let mut cell = vec![0f32; N * N];
        for y in 0..open_runo_glyph::G {
            for x in 0..open_runo_glyph::G {
                cell[(y + 4) * N + x + 9] = g[y * open_runo_glyph::G + x];
            }
        }
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let got = rank(&device, &index, &[RankCell { map: cell.clone() }], 3, &[vec!['毎', '曰']]).unwrap();
        let (reference, _) = index.rank_cpu(&cell, 3, &['毎']);
        assert_eq!(got[0].top[0].0, "毎");
        assert_eq!(got[0].top.iter().map(|t| t.0.clone()).collect::<Vec<_>>(), reference.iter().map(|t| t.1.to_string()).collect::<Vec<_>>());
        assert!((got[0].top[0].1 - reference[0].0).abs() < 1e-4, "{} vs {}", got[0].top[0].1, reference[0].0);
        assert_eq!(got[0].query.len(), 2);
    }

    /// 速度の測定(`cargo test --release -- --ignored --nocapture bench`): 頻出字 約 8,000 字 × 2 フォントの照合 1 升あたりの時間。
    #[test]
    #[ignore]
    fn bench_sgemm_vs_reference() {
        let (Ok(sans), Ok(serif)) = (std::fs::read("F:/tessdata-jpn/fonts/NotoSansJP-VF.ttf"), std::fs::read("F:/tessdata-jpn/fonts/NotoSerifJP-VF.ttf")) else {
            return;
        };
        let glyphs = Glyphs::with_fonts(&[&serif[..], &sans[..]]).unwrap();
        let pool: Vec<char> = (0x4E00u32..0x4E00 + 7000).filter_map(char::from_u32).chain("あいうえおかきくけこ".chars()).collect();
        let t = std::time::Instant::now();
        let index = GlyphIndex::build(&glyphs, &pool);
        println!("index: {} columns built in {:?}", index.len(), t.elapsed());
        let cell: Vec<f32> = (0..N * N).map(|i| ((i * 37) % 101) as f32 / 101.0).collect();
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let t = std::time::Instant::now();
        for _ in 0..5 {
            rank(&device, &index, &[RankCell { map: cell.clone() }], 24, &[]).unwrap();
        }
        println!("sgemm (open-cuda CPU device): {:?} per cell", t.elapsed() / 5);
        let t = std::time::Instant::now();
        for _ in 0..5 {
            index.rank_cpu(&cell, 24, &[]);
        }
        println!("reference (threads):          {:?} per cell", t.elapsed() / 5);
    }
}
