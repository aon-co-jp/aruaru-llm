//! USB接続したAndroidスマホへ実際に計算タスクを配布するための最小の
//! タスクキュー(2026-08-19新設)。
//!
//! # 背景
//! `hardware::detect_accelerators`はUSB接続台数の「検出」までしか行って
//! おらず(`disclosure`フィールドに明記の通り)、検出したスマホへ実際に
//! 計算タスクを送る経路が存在しなかった。本モジュールはその最小の一歩
//! ――ポーリング方式のタスク配布・結果受信――を実装する。
//!
//! # 実際にできること(正直な開示)
//! - `GET /v1/background-fold/task` — 未処理タスクを1件返す(無ければ
//!   `available: false`)。タスクの中身は`idle_background_fold`が既に
//!   使っている「実インテント埋め込みベクトル同士のコサイン類似度計算」
//!   と同じ粒度の軽量な数値計算(ベクトル2本、次元は埋め込みモデルの
//!   `hidden_size`)——Model Foldingの層統合計算そのものではなく、その
//!   ための「層間類似度を求める」という下請けタスクの模擬版に留まる
//!   (`idle_background_fold`と同じ限界)。
//! - `POST /v1/background-fold/task-result` — スマホ側が計算した
//!   コサイン類似度を受け取り、ログとして記録する。
//!
//! # 実際にはできないこと(誇張しない)
//! - タスクの結果を実際のモデル推論・Model Folding計算へ反映する経路は
//!   無い(受け取って記録するのみ、`idle_background_fold`のPC側計算とは
//!   独立)。
//! - 認証・レート制限は無い(同一LAN/USBテザリング内の信頼された端末を
//!   想定した最小実装、公開インターネットへの露出は想定していない)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

/// 1件のタスク。
#[derive(Debug, Clone, Serialize)]
pub struct PhoneTask {
    pub task_id: u64,
    pub vec_a: Vec<f32>,
    pub vec_b: Vec<f32>,
    /// タスクの種類。将来複数種類のタスクを配布する余地を残すため
    /// 文字列にしている(現状は`"cosine_similarity"`固定)。
    pub kind: &'static str,
}

#[derive(Debug, Deserialize)]
pub struct TaskResultRequest {
    pub task_id: u64,
    pub similarity: f32,
    /// この結果を計算した端末の簡単な自己申告(シリアル番号等、任意)。
    #[serde(default)]
    pub device_label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskResultRecord {
    pub task_id: u64,
    pub similarity: f32,
    pub device_label: Option<String>,
    pub received_epoch_secs: u64,
}

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

struct ResultLog {
    results: Vec<TaskResultRecord>,
}

static RESULT_LOG: OnceLock<Mutex<ResultLog>> = OnceLock::new();

fn result_log() -> &'static Mutex<ResultLog> {
    RESULT_LOG.get_or_init(|| Mutex::new(ResultLog { results: Vec::new() }))
}

fn now_epoch_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 次のタスクを1件生成する。実際に稼働中の`scoring.rs`のインテント
/// 埋め込みキャッシュがあればそれを使い、無ければ小さな決め打きベクトル
/// (デモ用、次元8)にフォールバックする——`scoring`側のキャッシュ取得に
/// 失敗しても本エンドポイント自体は落とさない可用性優先の設計。
pub fn next_task() -> PhoneTask {
    let task_id = NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed);
    let (vec_a, vec_b) = crate::scoring::sample_embedding_pair_for_phone_task()
        .unwrap_or_else(|| {
            // フォールバック: 埋め込みモデル未ロード時でも一応の計算タスクを
            // 発行できるよう、決め打ちの8次元ベクトルを使う(意味のある
            // 埋め込みではない、実演用のプレースホルダ)。
            (
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            )
        });
    PhoneTask {
        task_id,
        vec_a,
        vec_b,
        kind: "cosine_similarity",
    }
}

pub fn record_result(req: TaskResultRequest) -> TaskResultRecord {
    let record = TaskResultRecord {
        task_id: req.task_id,
        similarity: req.similarity,
        device_label: req.device_label,
        received_epoch_secs: now_epoch_secs(),
    };
    let mut log = result_log().lock().unwrap_or_else(|e| e.into_inner());
    log.results.push(record.clone());
    // 無制限に溜め続けない(直近200件のみ保持、デモ用の簡易実装)。
    if log.results.len() > 200 {
        let excess = log.results.len() - 200;
        log.results.drain(0..excess);
    }
    record
}

pub fn recent_results(limit: usize) -> Vec<TaskResultRecord> {
    let log = result_log().lock().unwrap_or_else(|e| e.into_inner());
    log.results
        .iter()
        .rev()
        .take(limit)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_task_returns_a_task_with_nonzero_vectors() {
        let task = next_task();
        assert!(!task.vec_a.is_empty());
        assert!(!task.vec_b.is_empty());
        assert_eq!(task.kind, "cosine_similarity");
    }

    #[test]
    fn record_result_and_recent_results_round_trip() {
        let before = recent_results(1000).len();
        let record = record_result(TaskResultRequest {
            task_id: 999_999,
            similarity: 0.42,
            device_label: Some("test-phone".to_string()),
        });
        assert_eq!(record.task_id, 999_999);
        let results = recent_results(1000);
        assert_eq!(results.len(), before + 1);
        assert_eq!(results[0].task_id, 999_999);
    }
}
