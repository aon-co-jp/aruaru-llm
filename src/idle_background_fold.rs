//! バックグラウンド・アイドル時Model Folding準備スケジューラ(2026-08-19新設)。
//!
//! # 背景・ユーザー要望
//! 「再学習を1週間程度かけて、余ったAndroidスマホもフル活用しながらPCの
//! バックグラウンドで、実現可能な範囲で裏側で自動実行する仕様にして
//! ほしい」という要望への対応。詳細な実現可能性の評価は
//! `CLAUDE.md`のHANDOFF(2026-08-19)を参照。
//!
//! # このモジュールが実際に行うこと(正直な開示)
//! - システムがアイドル(=直近`IDLE_THRESHOLD_SECS`秒間、`/v1/chat`・
//!   `/v1/generate`等の推論系エンドポイントへのHTTPリクエストが無い)と
//!   判定された場合のみ、低優先度のバックグラウンドスレッド上で
//!   「1ステップ分」の軽量な処理を実行し、すぐスリープへ戻る
//!   (常時フル稼働ではなく、間欠的に少しずつ進める設計)。
//! - `touch_activity()`をHTTPハンドラ側から呼ぶことで、実際のリクエスト
//!   有無に基づいてアイドル判定する(ダミーのタイマーではなく実際の
//!   サーバー負荷を見る)。
//! - 進捗(実行済みステップ数・直近の一致度スコア)を`FoldProgress`として
//!   `GET /v1/background-fold/status`から確認できる。
//!
//! # このモジュールが実際には行っていないこと(誇張しない)
//! - **本物のModel Folding(ICLR 2025, arXiv:2502.10216)のアルゴリズムは
//!   実装していない**。Model Foldingはモデルの重み(各層のニューロン/
//!   チャネル)そのものをクラスタリングし統合する手法だが、
//!   `open-cuda-llm::GptModel`は各層の重みベクトルへの公開アクセサ
//!   (`pub`なgetter)を提供していない(`layers: Vec<DecoderLayer>`は
//!   private field)。重みへ実際にアクセスするには`open-cuda-llm`側へ
//!   新規のpublic APIを追加する必要があり、今回のセッションではその
//!   クレート内部変更までは行っていない(スコープを正直に開示)。
//! - 代わりに、今回は「1週間かけて少しずつ進める」という**スケジューリング
//!   基盤**(アイドル検知・低頻度実行・進捗の可視化)を実際に動作する形で
//!   実装した。各ステップの中身は、`scoring.rs`が既に保持している実際の
//!   文埋め込みベクトル(open-cuda-bert、multilingual-e5-small)同士の
//!   コサイン類似度を計算するプレースホルダ処理に留めている——これは
//!   Model Foldingの「層の類似度に基づき統合候補を探す」という発想を
//!   小さく模した検証用ステップであり、実際にモデルの重みを書き換えたり
//!   圧縮したりすることは一切ない(読み取り専用、モデルは変更されない)。
//! - この基盤が動作することを確認した上で、次のステップとして
//!   `open-cuda-llm`側に層の重みへの読み取り専用アクセサを追加すれば、
//!   本ステップ関数の中身だけを本物のModel Folding計算(層間コサイン
//!   類似度→k-meansクラスタリング→統合)へ置き換えられる設計にしてある
//!   (`FoldStepFn`が差し替え可能なクロージャである理由)。
//!
//! # スマホ(Android)活用について
//! このモジュール自体はPC側(このプロセス内)でのみ動作する。スマホ側の
//! 役割分担(推論の一部肩代わり・軽量な補助計算)は、既存の
//! `android/`クライアント(`tokyo.runo.aruarullm`)がHTTP経由でこの
//! サーバーへアクセスする既存の仕組みを再利用する形の**プロトコル設計の
//! みを`CLAUDE.md`のHANDOFFに記載**しており、実機(手持ちの余った
//! Androidスマホ)がこの開発環境に存在しないため、スマホ側の実装は
//! 行っていない(正直な開示、詳細はHANDOFF参照)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// この秒数だけHTTPリクエストが来なければ「アイドル」とみなす。
/// 常時稼働のGPUクラスタとは異なり、開発機のCPUを普段使いの邪魔をしない
/// ことを優先し、やや長め(2分)に設定している。
const IDLE_THRESHOLD_SECS: u64 = 120;

/// アイドル判定ループ自体のポーリング間隔。短くしすぎるとこのスレッド
/// 自体がCPUを消費してしまうため、数秒に1回のチェックに留める。
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// 1回のアイドルステップの後に必ず挟む休止時間。アイドルが続く場合でも
/// 「気づいたら少しずつ進んでいる」程度の頻度に抑え、CPUを占有しない。
const STEP_COOLDOWN: Duration = Duration::from_secs(30);

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

static LAST_ACTIVITY_EPOCH_SECS: AtomicU64 = AtomicU64::new(0);

/// HTTPハンドラ側から呼び、「今アクティブなリクエストがあった」ことを
/// 記録する。呼び忘れても致命的ではない(その場合はアイドルと誤判定
/// されバックグラウンド処理が動くだけで、サービス自体は壊れない)。
pub fn touch_activity() {
    LAST_ACTIVITY_EPOCH_SECS.store(now_epoch_secs(), Ordering::Relaxed);
}

fn idle_seconds() -> u64 {
    let last = LAST_ACTIVITY_EPOCH_SECS.load(Ordering::Relaxed);
    if last == 0 {
        // まだ一度もtouch_activityが呼ばれていない = 起動直後。
        // プロセス開始時刻を基準にする方が正確だが、簡略化のため
        // 「アイドルとみなす」安全側(=すぐ処理を試みる)を選ぶ。
        return IDLE_THRESHOLD_SECS + 1;
    }
    now_epoch_secs().saturating_sub(last)
}

/// バックグラウンド処理の進捗。`GET /v1/background-fold/status`で公開する。
#[derive(Debug, Clone, serde::Serialize)]
pub struct FoldProgress {
    pub steps_completed: u64,
    pub last_step_epoch_secs: u64,
    pub last_similarity_summary: Option<String>,
    pub currently_idle: bool,
    pub idle_seconds: u64,
    /// 誇張しないための正直な自己申告フィールド。呼び出し側(UI等)が
    /// 「本物のModel Foldingが実行中」と誤解しないよう、常にこの文字列を
    /// 一緒に返す。
    pub disclosure: &'static str,
}

struct ProgressState {
    steps_completed: u64,
    last_step_epoch_secs: u64,
    last_similarity_summary: Option<String>,
}

static PROGRESS: OnceLock<Mutex<ProgressState>> = OnceLock::new();

fn progress_state() -> &'static Mutex<ProgressState> {
    PROGRESS.get_or_init(|| {
        Mutex::new(ProgressState {
            steps_completed: 0,
            last_step_epoch_secs: 0,
            last_similarity_summary: None,
        })
    })
}

pub fn current_progress() -> FoldProgress {
    let state = progress_state().lock().unwrap_or_else(|e| e.into_inner());
    FoldProgress {
        steps_completed: state.steps_completed,
        last_step_epoch_secs: state.last_step_epoch_secs,
        last_similarity_summary: state.last_similarity_summary.clone(),
        currently_idle: idle_seconds() >= IDLE_THRESHOLD_SECS,
        idle_seconds: idle_seconds(),
        disclosure: "このバックグラウンド処理は本物のModel Folding(重みの \
            クラスタリング・統合)ではなく、アイドル検知・低頻度実行という \
            スケジューリング基盤の動作検証用プレースホルダです。既存の文 \
            埋め込みベクトル同士のコサイン類似度計算を1ステップとして反 \
            復しているだけで、モデルの重みは一切変更されません。詳細は \
            aruaru-llm/CLAUDE.mdのHANDOFF(2026-08-19)を参照してください。",
    }
}

/// 1回分の「アイドルステップ」。既存の`scoring.rs`が保持する実インテント
/// 埋め込みベクトル同士のコサイン類似度を計算する(読み取り専用、
/// モデルは一切変更しない)。将来、ここを本物のModel Folding計算
/// (層の重みの類似度→クラスタリング→統合)へ置き換える差し込み口として
/// 独立関数にしてある。
fn run_one_step() -> Option<String> {
    let pairs = crate::scoring::debug_intent_embedding_pairwise_similarities();
    if pairs.is_empty() {
        return None;
    }
    let summary = pairs
        .iter()
        .map(|(a, b, sim)| format!("{a}~{b}={sim:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(summary)
}

/// バックグラウンドスケジューラを起動する。`std::thread`(tokioの
/// ワーカースレッドプールとは独立)を使い、優先度の低い間欠実行として
/// メインのHTTP処理を一切妨げないようにする。
pub fn spawn() {
    std::thread::Builder::new()
        .name("idle-background-fold".to_string())
        .spawn(|| {
            tracing::info!(
                "idle_background_fold: scheduler started (idle_threshold={}s, poll_interval={}s)",
                IDLE_THRESHOLD_SECS,
                POLL_INTERVAL.as_secs()
            );
            loop {
                std::thread::sleep(POLL_INTERVAL);
                if idle_seconds() < IDLE_THRESHOLD_SECS {
                    continue;
                }
                let step_start = Instant::now();
                match run_one_step() {
                    Some(summary) => {
                        let mut state =
                            progress_state().lock().unwrap_or_else(|e| e.into_inner());
                        state.steps_completed += 1;
                        state.last_step_epoch_secs = now_epoch_secs();
                        state.last_similarity_summary = Some(summary.clone());
                        tracing::info!(
                            "idle_background_fold: step {} completed in {:?} ({})",
                            state.steps_completed,
                            step_start.elapsed(),
                            summary
                        );
                    }
                    None => {
                        tracing::debug!(
                            "idle_background_fold: skipped step (embeddings not warmed up yet)"
                        );
                    }
                }
                std::thread::sleep(STEP_COOLDOWN);
            }
        })
        .expect("failed to spawn idle-background-fold thread");
}
