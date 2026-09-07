//! 汎用テキスト分類(embed-cosine方式)。2026-09-07新設——
//! open-easy-webの「システムメモリ円グラフのその他データをaruaru-llmの
//! AIで分類してほしい」というユーザー指示への対応で、`scoring.rs`/
//! `security.rs`が使う「固定カテゴリの代表例と比較する」パターンから、
//! **呼び出し側がカテゴリ一覧そのものを自由に指定できる**汎用版へ
//! 一般化した(将来の他の呼び出し元でも再利用できるよう`/v1/classify`
//! として公開する)。
//!
//! **正直な開示**: `security.rs`のように各カテゴリに複数の代表例文を
//! 事前に持たせる設計ではなく、**カテゴリ名/説明文そのものを1つの
//! embeddingとして扱う**簡略版——精度は`security.rs`の複数例平均方式より
//! 劣る可能性があるが、呼び出し側が任意のカテゴリ集合を実行時に渡せる
//! 汎用性を優先した設計判断。

use std::sync::Arc;

use anyhow::{bail, Result};
use open_cuda_bert::cosine_similarity;
use opencuda_core::GpuDevice;

use crate::scoring::embed;

pub struct ClassifiedItem {
    pub item: String,
    pub category: String,
    pub score: f32,
}

/// `items`の各要素を、`categories`の中で最も埋め込みコサイン類似度が
/// 高いものへ分類する。`categories`が空、または埋め込み計算(モデル
/// 未ロード等)に失敗した場合は正直に`Err`を返す(黙って適当な値を
/// 返さない)。
pub fn classify_many(device: &Arc<dyn GpuDevice>, items: &[String], categories: &[String]) -> Result<Vec<ClassifiedItem>> {
    if categories.is_empty() {
        bail!("categories must not be empty");
    }
    let category_embeddings: Vec<Vec<f32>> =
        categories.iter().map(|c| embed(device, c, false)).collect::<Result<Vec<_>>>()?;

    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let item_embedding = embed(device, item, true)?;
        let mut best_idx = 0usize;
        let mut best_score = f32::MIN;
        for (i, ce) in category_embeddings.iter().enumerate() {
            let sim = cosine_similarity(&item_embedding, ce);
            if sim > best_score {
                best_score = sim;
                best_idx = i;
            }
        }
        out.push(ClassifiedItem { item: item.clone(), category: categories[best_idx].clone(), score: best_score });
    }
    Ok(out)
}
