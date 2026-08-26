//! `open-cuda-llm::GptModel`(GPT-2 124M実学習済み重み)による本格的な
//! 自己回帰テキスト生成。`scoring.rs`(意図分類、軽量・高速)とは目的が
//! 異なるため無理に統合せず、別エンドポイント(`POST /v1/generate`)として
//! 提供する。役割分担: 意図分類=軽量・高速な定型応答振り分け、生成=
//! 本格的だが重い(548MBの重みロード+CPU推論)自由文生成。
//!
//! ## 正直な開示(最重要、CLAUDE.md参照)
//!
//! - GPT-2 124Mは2019年発表の小型モデルであり、GPT-4等の最新商用LLMと
//!   同等の性能・知識・指示追従能力は無い。あくまで「外部LLM契約(API課金)
//!   不要の自己完結型AIとして、実際に文法的な文章を自己回帰生成できる」
//!   ことの実証に留まる。
//! - 生成されるテキストは文法的には自然な英語になることが多いが、
//!   事実として正確である保証は無い(幻覚・意味不明な continuation の
//!   可能性がある)。ファインチューニング済みの対話モデルではなく、
//!   素の事前学習済み言語モデルの貪欲デコード(温度無し、サンプリング無し)
//!   であるため、対話的に「質問に答える」というより「文の続きを予測する」
//!   挙動になる。
//! - `open-cuda-llm`自体もPagedAttention・連続バッチング等の本家vLLM最適化を
//!   持たない単一シーケンス逐次デコードのMVP実装(`open-cuda-llm`の
//!   モジュールdocコメント参照)。
//!
//! ## モデルのホットスワップ(2026-07-27追加)
//!
//! `model_catalog`経由でダウンロードした別のGPT-2互換モデルへ、プロセスを
//! 再起動せずに切り替えられるようにした(以前は起動時`OnceLock`だった
//! ため、`ARUARU_LLM_GPT2_DIR`を変えるにはプロセス再起動が必須だった
//! ——ユーザー指示「簡単にダウンロード・インストールを選択可能にして
//! ユーザビリティを高めて」の「選択可能」を、ダウンロードだけでなく
//! 実際の切り替えまで含めて満たすため)。[`select_model`]が新しいモデル
//! ディレクトリを読み込み、**読み込みに成功した場合のみ**現在の
//! アクティブモデルを置き換える(失敗時は現在動作中のモデルをそのまま
//! 維持し、サービスを壊さない設計)。

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use opencuda_core::GpuDevice;
use open_cuda_llm::{GptModel, GptTokenizer};

/// GPT-2実重みのディレクトリパス。既定は`open-cuda`側で既に検証・
/// ダウンロード済みの`open-cuda-llm/models/gpt2`(sibling path、`../open-cuda`
/// 前提、`PORTING.md`のsibling path依存パターンと同じ)。
/// `ARUARU_LLM_GPT2_DIR`環境変数で上書き可能。
fn default_model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ARUARU_LLM_GPT2_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from("../open-cuda/crates/open-cuda-llm/models/gpt2")
}

struct LoadedGpt {
    model: GptModel,
    tokenizer: GptTokenizer,
    /// 現在アクティブなモデルの読み込み元ディレクトリ(診断・
    /// `GET /v1/models/catalog`等での表示用)。
    dir: PathBuf,
    /// `wire_matmul_spirv`が実際に成功したか(=このモデルの全`Linear`に
    /// コンパイル済みSPIR-Vが配線済みかどうか)。2026-08-06追加、
    /// `engine_label`が実行経路(Vulkan/CPU)を正直に反映するために使う
    /// (CLAUDE.md 2026-08-05 HANDOFFで指摘された「`engine`が常に`-cpu`
    /// 固定文字列」の粗の修正)。`real-vulkan` feature無効時は常に`false`。
    matmul_spirv_wired: bool,
    /// `wire_matmul_dxil_offload`が実際に成功したか(=このモデルの全
    /// `Linear`の密GEMMが実D3D12デバイスへオフロードされているか)。
    /// 2026-08-23追加。`real-dx12` feature無効時、または`DirectXDevice`の
    /// 構築に失敗した場合は常に`false`(=CPU実行のまま、正直に報告する)。
    matmul_dxil_offloaded: bool,
}

/// 現在アクティブなモデル(`None`は「まだ一度もロードを試みていない」
/// ではなく「直近のロードが失敗した」ことも表す——エラー内容は
/// `load_or_get`が都度ロードを試みて返すため、ここでは成功状態のみを保持)。
static ACTIVE: RwLock<Option<Arc<LoadedGpt>>> = RwLock::new(None);

/// コンパイル済み`matmul.spv`のパス(`real-vulkan` feature有効時のみ使用)。
/// `default_model_dir()`と同じsibling path前提(`../open-cuda`、
/// `PORTING.md`のsibling path依存パターン)。`open-cuda`側の
/// `examples/matmul_vulkan_real/shaders/matmul.spv`が、`Linear::forward`
/// のVulkan GEMMテスト(`open-cuda-llm`側`set_matmul_spirv_makes_linear_
/// forward_use_vulkan_and_matches_cpu_output`)でも使っている既存の
/// 共有シェーダパスであり、新規にこのリポジトリ用のコピーは作らず
/// そのまま参照する(`tools/compile-vulkan-shaders.*`で生成済みのものを
/// 前提とする)。`ARUARU_LLM_MATMUL_SPIRV`環境変数で上書き可能。
#[cfg(feature = "real-vulkan")]
fn default_matmul_spirv_path() -> PathBuf {
    if let Ok(path) = std::env::var("ARUARU_LLM_MATMUL_SPIRV") {
        return PathBuf::from(path);
    }
    PathBuf::from("../open-cuda/examples/matmul_vulkan_real/shaders/matmul.spv")
}

/// コンパイル済み`softmax.spv`のパス(`real-vulkan` feature有効時のみ使用、
/// 2026-08-06追加)。`default_matmul_spirv_path`と同じsibling path前提。
/// `open-cuda`側`GptModel::set_softmax_spirv`(2026-08-06新設、
/// `opencuda_blas::softmax_vulkan_generic`との連携)へ渡す。
/// `ARUARU_LLM_SOFTMAX_SPIRV`環境変数で上書き可能。
#[cfg(feature = "real-vulkan")]
fn default_softmax_spirv_path() -> PathBuf {
    if let Ok(path) = std::env::var("ARUARU_LLM_SOFTMAX_SPIRV") {
        return PathBuf::from(path);
    }
    PathBuf::from("../open-cuda/examples/softmax_vulkan_real/shaders/softmax.spv")
}

/// `real-vulkan` feature有効時のみ、コンパイル済み`matmul.spv`を
/// `GptModel::set_matmul_spirv`経由で全`Linear`層(QKV/attn_out/
/// intermediate/output/lm_head)へ配線する(2026-08-05追加、`open-cuda`
/// 側commit `6452ae4`——`Linear::forward`がspirvをsgemmへ渡していなかった
/// バグの修正——を受けての配線)。これが無いと、`real-vulkan` feature
/// でVulkanDeviceを選択していても`Linear::forward`は`spirv_matmul: None`
/// のままなので`GemmPath::VulkanGeneric`に必要なSPIR-Vが無く、
/// `sgemm`が即座に失敗する(CLAUDE.md 2026-08-04 HANDOFF参照)。
/// 読み込み失敗時はサービスを落とさず、CPU実行と同じ`spirv_matmul: None`
/// のまま(=CPU GEMMパス)で継続する(既存の「サービスを壊さない」設計
/// 方針を踏襲)。
#[cfg(feature = "real-vulkan")]
fn wire_matmul_spirv(model: &mut GptModel) -> bool {
    let path = default_matmul_spirv_path();
    match std::fs::read(&path) {
        Ok(bytes) => {
            let len = bytes.len();
            model.set_matmul_spirv(bytes);
            tracing::info!(
                "real-vulkan feature enabled: loaded matmul.spv ({len} bytes) from {} and wired into GptModel via set_matmul_spirv",
                path.display()
            );
            true
        }
        Err(err) => {
            tracing::warn!(
                "real-vulkan feature enabled but failed to load matmul.spv from {} ({err}); \
                 Linear layers will keep using the CPU GEMM path even if VulkanDevice was selected \
                 (run tools/compile-vulkan-shaders.* in open-cuda first, or set ARUARU_LLM_MATMUL_SPIRV)",
                path.display()
            );
            false
        }
    }
}

/// `real-vulkan` feature有効時のみ、コンパイル済み`softmax.spv`を
/// `GptModel::set_softmax_spirv`経由で配線する(2026-08-06追加、
/// `open-cuda`側`GptModel::set_softmax_spirv`——`opencuda_blas::
/// softmax_vulkan_generic`との連携——が新設されたことを受けての配線)。
/// これと`wire_matmul_spirv`の両方が成功して初めて、Attention計算の
/// QKᵀ・softmax・P·Vのすべてが実Vulkanデバイス上でディスパッチされる
/// (「GPU GEMM + CPU softmax」のハイブリッドから「GPU GEMM + GPU
/// softmax」への移行)。読み込み失敗時はサービスを落とさず、softmaxは
/// ホスト側CPU実行のまま継続する(既存の「サービスを壊さない」設計
/// 方針を踏襲)。
#[cfg(feature = "real-vulkan")]
fn wire_softmax_spirv(model: &mut GptModel) -> bool {
    let path = default_softmax_spirv_path();
    match std::fs::read(&path) {
        Ok(bytes) => {
            let len = bytes.len();
            model.set_softmax_spirv(bytes);
            tracing::info!(
                "real-vulkan feature enabled: loaded softmax.spv ({len} bytes) from {} and wired into GptModel via set_softmax_spirv",
                path.display()
            );
            true
        }
        Err(err) => {
            tracing::warn!(
                "real-vulkan feature enabled but failed to load softmax.spv from {} ({err}); \
                 Attention softmax will keep using the CPU path even if matmul.spv is wired \
                 (run tools/compile-vulkan-shaders.* in open-cuda first, or set ARUARU_LLM_SOFTMAX_SPIRV)",
                path.display()
            );
            false
        }
    }
}

/// コンパイル済み`flash_attention.spv`のパス(`real-vulkan` feature有効時
/// のみ使用、2026-08-08追加)。`default_matmul_spirv_path`/`default_softmax_
/// spirv_path`と同じsibling path前提。`open-cuda`側`GptModel::
/// set_flash_attention_spirv`(2026-08-07新設、`open-cuda-llm`の
/// DecoderLayerへ実配線・実機検証済み——QKᵀ・オンラインsoftmax・P·Vが
/// 1回のディスパッチで完結するfusedカーネル)へ渡す。`Some`が設定されると
/// `GptModel`側の設計により`softmax_spirv`より**優先**される
/// (`open-cuda-llm/src/lib.rs`参照、`flash_attn_spirv`が`Some`ならそちらの
/// 経路を使う)。`ARUARU_LLM_FLASH_ATTENTION_SPIRV`環境変数で上書き可能。
/// 既定では**wireしない**(下記`ARUARU_LLM_ENABLE_FLASH_ATTENTION=1`で
/// 明示的にopt-inした場合のみ)——3経路(GEMM+CPU softmax/GEMM+GPU softmax/
/// GEMM+fused flash attention)を実際に比較計測できるようにするための設計。
#[cfg(feature = "real-vulkan")]
fn default_flash_attention_spirv_path() -> PathBuf {
    if let Ok(path) = std::env::var("ARUARU_LLM_FLASH_ATTENTION_SPIRV") {
        return PathBuf::from(path);
    }
    PathBuf::from("../open-cuda/examples/flash_attention_vulkan_real/shaders/flash_attention.spv")
}

/// `real-vulkan` feature有効時、かつ`ARUARU_LLM_ENABLE_FLASH_ATTENTION=1`
/// (または`true`)が明示的に設定されている場合のみ、コンパイル済み
/// `flash_attention.spv`を`GptModel::set_flash_attention_spirv`経由で
/// 配線する(2026-08-08追加)。既定でwireしないのは、`wire_softmax_spirv`と
/// 同時に有効化した場合`GptModel`側の設計によりflash_attentionが常に
/// softmaxより優先されてしまい、「GEMM+GPU softmax」経路を意図的に選ぶ
/// ことができなくなるため(3経路の使い分け・比較を可能にするための
/// 明示的opt-in)。読み込み失敗時はサービスを落とさず、既存のsoftmax/CPU
/// 経路のまま継続する。
#[cfg(feature = "real-vulkan")]
fn wire_flash_attention_spirv(model: &mut GptModel) -> bool {
    let enabled = std::env::var("ARUARU_LLM_ENABLE_FLASH_ATTENTION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !enabled {
        return false;
    }
    let path = default_flash_attention_spirv_path();
    match std::fs::read(&path) {
        Ok(bytes) => {
            let len = bytes.len();
            // open-cuda側の実機検証済みテストと同じblock_size=4を既定とする
            // (`opencuda-blas::flash_attention_with_spirv`はhead_dim/block_size
            // とも256を超えると失敗する既知の制約があるが、GPT-2 124Mの
            // head_dim=64はこれを十分下回る)。
            model.set_flash_attention_spirv(bytes, 4);
            tracing::info!(
                "real-vulkan feature enabled + ARUARU_LLM_ENABLE_FLASH_ATTENTION set: \
                 loaded flash_attention.spv ({len} bytes) from {} and wired into GptModel \
                 via set_flash_attention_spirv (this takes priority over softmax_spirv)",
                path.display()
            );
            true
        }
        Err(err) => {
            tracing::warn!(
                "real-vulkan feature enabled + ARUARU_LLM_ENABLE_FLASH_ATTENTION set but failed to \
                 load flash_attention.spv from {} ({err}); Attention will keep using the softmax/CPU \
                 path (run tools/compile-vulkan-shaders.* in open-cuda first, or set \
                 ARUARU_LLM_FLASH_ATTENTION_SPIRV)",
                path.display()
            );
            false
        }
    }
}

/// `open-cuda-llm::GptModel::enable_mla_kv_compression`(DeepSeek-V3の
/// Multi-Head Latent Attention風、KVキャッシュの低ランク圧縮、
/// `open-cuda`側2026-08-07実装・実機検証済み)をこのモデルへ配線するか
/// 判定するenv変数。**既定は無効(opt-in)**——`real-vulkan`のような
/// GPU専用機能とは異なりCPU実行でも成立する(`mla_compress_kv`/
/// `mla_decompress_kv`は`opencuda_blas::sgemm`をVulkan/CPU両対応で
/// 呼ぶだけの純粋な計算のため、デバイス種別に依存しない)。それでも
/// 既定offにしたのは速度ではなく**生成品質**の理由——下記
/// `wire_mla_kv_compression`のdocコメント参照。
/// `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`(または`true`)で有効化。
fn mla_kv_compression_enabled() -> bool {
    std::env::var("ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

/// KVキャッシュの圧縮後次元`d_c`(ヘッドあたり)。既定は
/// `head_dim / 4`(75%削減、`open-cuda`側テスト
/// `mla_kv_compression_enabled_model_generates_without_panicking`が
/// 使っているのと同じ削減率)、最低`1`を保証する。`0 < d_c < head_dim`
/// (`enable_mla_kv_compression`自身のassert)を満たさない値が
/// 環境変数で指定された場合は既定値へフォールバックする(サービスを
/// 壊さない設計)。`ARUARU_LLM_MLA_D_C`環境変数で上書き可能。
fn mla_d_c(head_dim: usize) -> usize {
    let default_d_c = (head_dim / 4).max(1);
    match std::env::var("ARUARU_LLM_MLA_D_C") {
        Ok(v) => match v.parse::<usize>() {
            Ok(d_c) if d_c > 0 && d_c < head_dim => d_c,
            _ => {
                tracing::warn!(
                    "ARUARU_LLM_MLA_D_C={v:?} is not a valid 0 < d_c < head_dim={head_dim}; falling back to default d_c={default_d_c}"
                );
                default_d_c
            }
        },
        Err(_) => default_d_c,
    }
}

/// `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`(または`true`)が明示的に
/// 設定されている場合のみ、`GptModel::enable_mla_kv_compression`
/// (2026-08-07新設、`open-cuda`側で実装・実機検証済み——KVキャッシュを
/// フル精度`head_dim`次元のまま保持せず、低ランク射影で`d_c`次元へ
/// 圧縮して保存し、Attention計算直前に復元する)を配線する
/// (2026-08-08追加)。
///
/// ## なぜ既定offか(最重要、正直な開示)
///
/// `wire_flash_attention_spirv`(既定off)は「GPUディスパッチオーバー
/// ヘッドがCPUより遅いことがある」という**速度**上の理由でopt-inに
/// したが、こちらは事情が異なる——`enable_mla_kv_compression`が使う
/// down-projection/up-projection行列は`open-cuda`側`open-cuda-llm/
/// src/lib.rs`の実装(`GptModel::enable_mla_kv_compression`)を確認した
/// 限り**学習済みではなく乱数初期化**(`random_vec`、DeepSeek-V3本家が
/// 大規模事前学習で獲得する射影とは無関係)。つまりこの圧縮は
/// **非可逆**であり、圧縮→復元後のK/Vはこのプロセスが読み込んだ実GPT-2
/// 124M(またはカタログの他モデル)の学習済み重みが持つ意味的な内容を
/// 実際に破壊する。`open-cuda`側の回帰テスト
/// `mla_kv_compression_actually_changes_generation_versus_uncompressed`
/// 自体が「圧縮ありと無しで生成結果が実際に異なることを確認する」
/// テストであり、`open-cuda`側は最初から「配線が正しく動くこと」の
/// 実証に留め生成品質の維持は主張していない(同ファイルのdoc
/// コメント参照)。`open-cuda`側の実機検証はすべて`GptConfig::tiny`
/// (ランダム初期化トイモデル)止まりで、**実学習済み重みでの品質検証は
/// 一度も行われていない**——このリポジトリでの実配線が初めての
/// 実学習済み重みでの検証機会となる(下記CLAUDE.md HANDOFF参照)。
/// このため、`wire_flash_attention_spirv`と同じ「複数経路の比較を
/// 可能にする」意図に加え、**既定で実ユーザー向け応答の品質を落とさ
/// ない**という可用性優先の理由からも既定offとした。読み込み失敗時
/// (`d_c`不正等)はサービスを落とさず、フル精度KVキャッシュのまま
/// 継続する。
fn wire_mla_kv_compression(model: &mut GptModel) -> bool {
    if !mla_kv_compression_enabled() {
        return false;
    }
    let config = model.config();
    if config.num_heads == 0 || config.hidden_size % config.num_heads != 0 {
        tracing::warn!(
            "ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION set but hidden_size={} is not evenly divisible by num_heads={}; skipping MLA wiring",
            config.hidden_size,
            config.num_heads
        );
        return false;
    }
    let head_dim = config.hidden_size / config.num_heads;
    let d_c = mla_d_c(head_dim);
    let seed: u64 = std::env::var("ARUARU_LLM_MLA_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(42);
    match model.enable_mla_kv_compression(d_c, seed) {
        Ok(()) => {
            let reduction_percent = 100.0 * (1.0 - (d_c as f64 / head_dim as f64));
            tracing::info!(
                "ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION set: wired MLA-style KV cache compression \
                 (head_dim={head_dim} -> d_c={d_c}, {reduction_percent:.1}% smaller per-token KV storage). \
                 WARNING: down/up-projection matrices are randomly initialized (not learned), so this is \
                 lossy and will change/degrade generation output versus the uncompressed path \
                 (see generation.rs doc comment for details)."
            );
            true
        }
        Err(err) => {
            tracing::warn!("ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION set but enable_mla_kv_compression failed ({err:#}); keeping full-precision KV cache");
            false
        }
    }
}

/// **2026-08-08追加**: `open-cuda`側`GptModel::enable_mla_kv_compression_calibrated`
/// (PCA較正版MLA、`open-cuda/CLAUDE.md`同日HANDOFF参照)を配線するか判定
/// するenv変数。`ARUARU_LLM_MLA_CALIBRATED=1`(または`true`)が設定されて
/// いる場合、乱数射影版(`wire_mla_kv_compression`)の代わりにこちらを使う
/// (両方同時には有効化しない、下記`wire_mla_kv_compression_any`参照)。
/// **既定off**——PCA較正版でも実測(open-cuda側同日HANDOFF)では非圧縮版
/// より明確に品質が劣化したままであり(乱数射影版ほど酷い反復破綻には
/// 陥らないが、意味的一貫性は依然低い)、実ユーザー向け応答の既定挙動を
/// これに置き換えるべきではないと判断したため。
fn mla_calibrated_enabled() -> bool {
    std::env::var("ARUARU_LLM_MLA_CALIBRATED").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

/// PCA較正の較正データに使う既定サンプル文(トピックを分散させた一般的な
/// 英文、open-cuda側テスト`calibrated_pca_mla_kv_compression_on_real_gpt2_
/// weights`と同じ発想)。`ARUARU_LLM_MLA_CALIBRATION_PROMPTS`環境変数
/// (`;`区切り)で上書き可能——実運用でのトラフィックの実文体に近い較正文を
/// 使いたい場合に差し替えられるようにする。
fn mla_calibration_prompts() -> Vec<String> {
    if let Ok(v) = std::env::var("ARUARU_LLM_MLA_CALIBRATION_PROMPTS") {
        let prompts: Vec<String> = v.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        if !prompts.is_empty() {
            return prompts;
        }
    }
    [
        "The weather today is quite pleasant and sunny.",
        "In economics, supply and demand determine prices in a market.",
        "She walked into the kitchen and started making breakfast.",
        "The history of ancient Rome spans over a thousand years.",
        "Computers process information using binary logic circuits.",
        "The mountain trail was steep but offered a beautiful view.",
        "Scientists discovered a new species of frog in the rainforest.",
        "He picked up his guitar and began to play a soft melody.",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// PCA較正版MLA(`enable_mla_kv_compression_calibrated`)を配線する。
/// 較正パス自体はGEMM計算のみでデバイス種別に依存しないため、
/// `real-vulkan` featureの有無に関わらず`opencuda_cpu::CpuDevice`で行う
/// (较正は起動時に1回だけ、比較的軽量な複数プリフィルで完結するため、
/// わざわざVulkanDeviceを別途構築する必要はないと判断)。
fn wire_mla_kv_compression_calibrated(model: &mut GptModel, tokenizer: &GptTokenizer) -> bool {
    let config = model.config();
    if config.num_heads == 0 || !config.hidden_size.is_multiple_of(config.num_heads) {
        tracing::warn!(
            "ARUARU_LLM_MLA_CALIBRATED set but hidden_size={} is not evenly divisible by num_heads={}; skipping calibrated MLA wiring",
            config.hidden_size,
            config.num_heads
        );
        return false;
    }
    let head_dim = config.hidden_size / config.num_heads;
    let d_c = mla_d_c(head_dim);

    let prompts = mla_calibration_prompts();
    let mut sample_prompts = Vec::with_capacity(prompts.len());
    for text in &prompts {
        match tokenizer.encode(text) {
            Ok(ids) if !ids.is_empty() => sample_prompts.push(ids),
            Ok(_) => tracing::warn!("ARUARU_LLM_MLA_CALIBRATED: calibration prompt {text:?} tokenized to zero tokens; skipping it"),
            Err(err) => tracing::warn!("ARUARU_LLM_MLA_CALIBRATED: failed to tokenize calibration prompt {text:?} ({err:#}); skipping it"),
        }
    }
    if sample_prompts.is_empty() {
        tracing::warn!("ARUARU_LLM_MLA_CALIBRATED set but no calibration prompts could be tokenized; skipping calibrated MLA wiring");
        return false;
    }

    let device: Arc<dyn GpuDevice> = opencuda_cpu::CpuDevice::new(0);
    match model.enable_mla_kv_compression_calibrated(d_c, device.as_ref(), &sample_prompts) {
        Ok(()) => {
            let reduction_percent = 100.0 * (1.0 - (d_c as f64 / head_dim as f64));
            tracing::info!(
                "ARUARU_LLM_MLA_CALIBRATED set: wired PCA-calibrated MLA-style KV cache compression \
                 (head_dim={head_dim} -> d_c={d_c}, {reduction_percent:.1}% smaller per-token KV storage, \
                 calibrated on {} sample prompts). This avoids the degenerate repetition failure mode seen \
                 with the random-projection variant, but still degrades quality versus the uncompressed path \
                 (see open-cuda's CLAUDE.md 2026-08-08 HANDOFF for the actual measured generations).",
                sample_prompts.len()
            );
            true
        }
        Err(err) => {
            tracing::warn!("ARUARU_LLM_MLA_CALIBRATED set but enable_mla_kv_compression_calibrated failed ({err:#}); keeping full-precision KV cache");
            false
        }
    }
}

/// `wire_mla_kv_compression`(乱数射影)・`wire_mla_kv_compression_calibrated`
/// (PCA較正)のどちらを使うか判定して呼び分ける(2026-08-08追加)。
/// `ARUARU_LLM_MLA_CALIBRATED=1`が優先(両方同時にオンにしても意味が
/// 無い——`GptModel`側は`layer.mla`を1つしか保持できないため後勝ちに
/// なるだけ、混乱を避けるため明示的に排他にする)。
fn wire_mla_kv_compression_any(model: &mut GptModel, tokenizer: &GptTokenizer) -> bool {
    if mla_calibrated_enabled() {
        return wire_mla_kv_compression_calibrated(model, tokenizer);
    }
    wire_mla_kv_compression(model)
}

fn load_from_dir(dir: &std::path::Path) -> Result<LoadedGpt, String> {
    #[allow(unused_mut)]
    let mut model = GptModel::load(dir).map_err(|e| format!("failed to load GPT-2 weights from {dir:?}: {e:#}"))?;
    let tokenizer = GptTokenizer::load(dir).map_err(|e| format!("failed to load GPT-2 tokenizer from {dir:?}: {e:#}"))?;
    #[allow(unused_mut)]
    let mut matmul_spirv_wired = false;
    #[cfg(feature = "real-vulkan")]
    {
        matmul_spirv_wired = wire_matmul_spirv(&mut model);
        // softmax配線はmatmul配線とは独立(片方が失敗してももう片方は試みる)。
        // ただし実際にGPU常駐softmaxが効くのはmatmul_spirv_wiredも真の場合のみ
        // (opencuda_blas側がGEMM経路とsoftmax経路を常に一致させる設計のため)。
        let _softmax_wired = wire_softmax_spirv(&mut model);
        // flash-attention配線はopt-in(既定オフ)。有効化されると
        // GptModel側の設計によりsoftmax配線より優先される(上記doc参照)。
        let _flash_attention_wired = wire_flash_attention_spirv(&mut model);
    }
    // MLA KVキャッシュ圧縮配線はGPU非依存(CPU実行でも成立)のため
    // `real-vulkan` feature配下に置かない。既定offはopt-in(上記doc参照)。
    // 2026-08-08: 乱数射影版/PCA較正版のどちらを使うかは
    // `wire_mla_kv_compression_any`が`ARUARU_LLM_MLA_CALIBRATED`で判定する。
    let _mla_wired = wire_mla_kv_compression_any(&mut model, &tokenizer);
    // 階層的アクセラレーションの第2段(2026-08-23追加): SPIR-V(Vulkan)
    // 配線が成立しなかった場合に限り、D3D12 Compute(DXIL)への密GEMM
    // オフロードを試みる。Vulkanが使えているならそちらの方が対象範囲が
    // 広い(Attention/softmaxもGPU常駐にできる)ため、重複配線はしない。
    #[allow(unused_mut)]
    let mut matmul_dxil_offloaded = false;
    #[cfg(feature = "real-dx12")]
    {
        if !matmul_spirv_wired {
            matmul_dxil_offloaded = wire_matmul_dxil_offload(&mut model);
        }
    }
    Ok(LoadedGpt { model, tokenizer, dir: dir.to_path_buf(), matmul_spirv_wired, matmul_dxil_offloaded })
}

/// コンパイル済み`matmul.dxil`(`opencuda-directx`のリポジトリへコミット
/// 済みの成果物。`.spv`と違い事前コンパイルが不要なので`include_bytes!`で
/// 埋め込める)を`GptModel::set_matmul_dxil_offload`経由で全`Linear`層へ
/// 配線する(2026-08-23新設)。
///
/// `DirectXDevice::new`に失敗した場合(D3D12非対応環境・ドライバ無し等)は
/// サービスを落とさず`false`を返し、CPU実行のままにする(既存の
/// `real-vulkan`配線と同じ安全側フォールバック方針)。
#[cfg(feature = "real-dx12")]
const MATMUL_DXIL: &[u8] = include_bytes!("../../open-cuda/crates/opencuda-directx/shaders/matmul.dxil");

#[cfg(feature = "real-dx12")]
fn wire_matmul_dxil_offload(model: &mut GptModel) -> bool {
    match opencuda_directx::real::DirectXDevice::new(0) {
        Ok(device) => {
            let name = device.info().name.clone();
            let device: Arc<dyn GpuDevice> = device;
            match model.set_matmul_dxil_offload(device, MATMUL_DXIL.to_vec()) {
                Ok(()) => {
                    tracing::info!(
                        "real-dx12 feature enabled: dense GEMM offloaded to D3D12 device '{name}' \
                         ({} bytes of matmul.dxil, weights resident in VRAM). \
                         Attention/LayerNorm/GELU still run on the CPU device.",
                        MATMUL_DXIL.len()
                    );
                    true
                }
                Err(err) => {
                    tracing::warn!("real-dx12: failed to upload resident weights to '{name}' ({err:#}); dense GEMM stays on the CPU");
                    false
                }
            }
        }
        Err(err) => {
            tracing::warn!("real-dx12 feature enabled but DirectXDevice::new failed ({err}); dense GEMM stays on the CPU");
            false
        }
    }
}

/// 現在アクティブなモデルを返す。まだ一度もロードしていなければ
/// `default_model_dir()`から初回ロードを試み、成功すればアクティブとして
/// 保持する(既存のOnceLock方式と同じ「初回リクエスト時の遅延ロード」
/// 挙動を維持)。
fn active_or_load_default() -> Result<Arc<LoadedGpt>, String> {
    if let Some(loaded) = ACTIVE.read().unwrap().clone() {
        return Ok(loaded);
    }
    let dir = default_model_dir();
    let loaded = Arc::new(load_from_dir(&dir)?);
    *ACTIVE.write().unwrap() = Some(loaded.clone());
    Ok(loaded)
}

/// 起動時ウォームアップ用(コールドスタート対策、`scoring::warmup`と同じ
/// 設計思想)。失敗しても致命的ではない(初回リクエスト時に再試行される)。
pub fn warmup() -> Result<()> {
    match active_or_load_default() {
        Ok(_) => Ok(()),
        Err(e) => anyhow::bail!("{e}"),
    }
}

/// `dir`から新しいモデルを読み込み、**読み込みに成功した場合のみ**現在の
/// アクティブモデルを置き換える(プロセス再起動不要のホットスワップ)。
/// 失敗した場合、現在動作中のモデルはそのまま維持される(サービスを
/// 壊さない設計、呼び出し元にはエラーの詳細を返す)。
pub fn select_model(dir: PathBuf) -> Result<()> {
    let loaded = load_from_dir(&dir).map_err(|e| anyhow::anyhow!(e))?;
    *ACTIVE.write().unwrap() = Some(Arc::new(loaded));
    Ok(())
}

/// 現在アクティブなモデルの読み込み元ディレクトリ(診断用、まだ一度も
/// ロードしていなければ`None`——`active_or_load_default`を呼んでいない
/// ため、副作用〈遅延ロードの起動〉を持たない読み取り専用の問い合わせ)。
pub fn active_model_dir() -> Option<PathBuf> {
    ACTIVE.read().unwrap().as_ref().map(|l| l.dir.clone())
}

/// エンジン識別子(`engine`フィールド用、常に実装方式を正直に返す方針)。
/// 後方互換のため定数として残すが、`model_catalog`経由でモデルを
/// ホットスワップした後は実際のサイズと乖離するため、レスポンスには
/// [`engine_label`]の方を使うこと(2026-07-27修正——「お勧めLLM
/// ダウンロード」機能でgpt2-medium/large/xl等に切り替えた後も
/// "gpt2-124m-..."と表示され続けるのは不正直なため)。
pub const ENGINE_GPT2_GREEDY: &str = "gpt2-124m-greedy-decode-v0-open-cuda-llm-cpu";

/// 現在アクティブなモデルの実行経路(Vulkan/CPU)を示す接尾辞。
/// **2026-08-06修正**: 以前は`engine_label`が常に`-cpu`固定文字列を
/// 返しており、`real-vulkan` feature有効時にVulkanDevice経由で実際に
/// 生成していても`engine`ラベル上は判別できなかった(CLAUDE.md
/// 2026-08-05 HANDOFFで指摘済みの粗)。`device.supports_spirv()`
/// (呼び出し側が実際に選択したデバイス)と`matmul_spirv_wired`
/// (モデル側の配線が実際に成功したか)の**両方**が真の場合のみ
/// `-vulkan`を返す——配線が失敗していればCPU GEMMパスへ安全側
/// フォールバックしているため、その場合は正直に`-cpu`を返す。
fn dispatch_suffix(device: &Arc<dyn GpuDevice>, loaded: &LoadedGpt) -> &'static str {
    if device.supports_spirv() && loaded.matmul_spirv_wired {
        "-vulkan"
    } else if loaded.matmul_dxil_offloaded {
        // 密GEMMのみD3D12へオフロードされたハイブリッド構成
        // (Attention/LayerNorm等はCPU)。誇張しないよう`-directx`ではなく
        // `-directx-gemm`とし、GEMMだけであることをラベル自体で示す。
        "-directx-gemm"
    } else {
        "-cpu"
    }
}

/// 現在アクティブなモデルディレクトリ名・実際の実行経路(Vulkan/CPU)を
/// 反映したエンジン識別子。ディレクトリ名がわかればそれをそのまま
/// 埋め込み(例: `"gpt2-medium-greedy-decode-v0-open-cuda-llm-vulkan"`)、
/// 未ロード(まだ一度もモデルをロードしていない)なら`device`のみを見て
/// デフォルトの`ENGINE_GPT2_GREEDY`相当のラベルを返す。
/// 現在アクティブなモデルで、密GEMMがD3D12(DXIL)へオフロード済みか
/// (2026-08-23追加、`GET /v1/runtime`が「実際にどの段のアクセラレーション
/// が効いているか」を正直に報告するために使う)。モデル未ロード時は`false`。
pub fn matmul_dxil_offloaded() -> bool {
    ACTIVE.read().unwrap().as_ref().map(|l| l.matmul_dxil_offloaded).unwrap_or(false)
}

/// 現在アクティブなモデルで、SPIR-V(Vulkan)matmulが配線済みか(同上)。
pub fn matmul_spirv_wired() -> bool {
    ACTIVE.read().unwrap().as_ref().map(|l| l.matmul_spirv_wired).unwrap_or(false)
}

pub fn engine_label(device: &Arc<dyn GpuDevice>) -> String {
    match ACTIVE.read().unwrap().clone() {
        Some(loaded) => {
            let suffix = dispatch_suffix(device, &loaded);
            let name = loaded.dir.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
            format!("{name}-greedy-decode-v0-open-cuda-llm{suffix}")
        }
        None => {
            let suffix = if device.supports_spirv() { "-vulkan" } else { "-cpu" };
            format!("gpt2-124m-greedy-decode-v0-open-cuda-llm{suffix}")
        }
    }
}

/// 繰り返しペナルティの既定値(CTRL方式、`open-cuda-llm::GptModel::
/// generate_with_repetition_penalty`参照)。対話ファインチューニング無しの
/// 素のGPT-2貪欲デコードが同一文字列(例: "Student: Hello"の無限ループ)へ
/// 陥る既知の劣化モードへの根本対応——実GPT-2 124M重みでの検証
/// (`open-cuda-llm`側テスト`repetition_penalty_reduces_degenerate_loop_
/// on_real_gpt2_weights`)では、この値で反復ループが解消し文法的に自然な
/// 会話文へ変わることを確認済み。`ARUARU_LLM_REPETITION_PENALTY`環境変数で
/// 上書き可能(`1.0`にすると従来通りペナルティ無しの挙動に戻る)。
pub fn default_repetition_penalty() -> f32 {
    std::env::var("ARUARU_LLM_REPETITION_PENALTY")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.3)
}

/// `prompt`の続きを`max_new_tokens`個、貪欲デコード(argmax、サンプリング
/// 無し)+繰り返しペナルティ(`default_repetition_penalty()`)で生成する。
/// GPT-2実重み・実トークナイザが読み込めない場合はエラーを返す(黙って
/// 別経路にフォールバックしない、意図分類とは別軸のためbag-of-words的な
/// 代替は存在しない)。
pub fn generate(device: &Arc<dyn GpuDevice>, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let loaded = active_or_load_default().map_err(|e| anyhow::anyhow!(e))?;
    let prompt_ids = loaded.tokenizer.encode(prompt).context("open-cuda-llm tokenizer encode failed")?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");
    let generated_ids = loaded
        .model
        .generate_with_repetition_penalty(device, &prompt_ids, max_new_tokens, default_repetition_penalty())
        .context("GptModel::generate_with_repetition_penalty failed")?;
    loaded.tokenizer.decode(&generated_ids).context("open-cuda-llm tokenizer decode failed")
}

/// `open_cuda_llm::GptModel::generate_speculative`(DSpark/Leviathan et al.
/// 方式のロスレス投機的デコード、2026-08-17新設)を、現在アクティブな
/// モデルをターゲットとして呼ぶ薄いラッパー。`draft_id`は
/// `model_catalog::CATALOG`のいずれか(例: `"distilgpt2"`)——ダウンロード
/// 済みでない場合はエラーを返す(黙って別モデルへフォールバックしない、
/// 他のエンドポイントと同じ「サービスを止めないが偽装もしない」方針)。
///
/// **正直な開示・既定の`/v1/generate`とは別エンドポイントにした理由**:
/// `open-cuda-llm`側で実機計測したところ(CPU実行、ターゲット`gpt2`+
/// ドラフト`distilgpt2`、`draft_k=4`)、採用率80%と高かったにもかかわらず
/// **素の`generate()`より実際には遅かった**(`open-cuda-llm/src/lib.rs`の
/// `generate_speculative`のdocコメント参照)。CPU素朴GEMM実装では
/// ディスパッチ固定オーバーヘッドという「削減すべきコスト」自体がほぼ
/// 存在しないため、ドラフトモデルの追加計算コストが純増分になって
/// しまう。このため既定の`/v1/generate`の内部実装を置き換えるのではなく、
/// 明示的にオプトインする別エンドポイント(`POST /v1/generate-speculative`)
/// として提供する——`real-vulkan`環境(Vulkanディスパッチオーバーヘッドが
/// 支配的、本来の狙い)での速度検証は未実施のまま、次の増分として残す。
/// **繰り返しペナルティ・MLA圧縮モデルは未対応**(`GptModel::
/// generate_speculative`のドキュメント通り、MLA圧縮モデルは`ensure!`で
/// 拒否される)。
pub fn generate_speculative(
    device: &Arc<dyn GpuDevice>,
    draft_id: &str,
    prompt: &str,
    max_new_tokens: usize,
    draft_k: usize,
) -> Result<(String, open_cuda_llm::SpeculativeStats)> {
    let target = active_or_load_default().map_err(|e| anyhow::anyhow!(e))?;

    let entry = crate::model_catalog::find(draft_id).ok_or_else(|| anyhow::anyhow!("unknown draft model id: {draft_id}"))?;
    let draft_dir = crate::model_catalog::models_root().join(entry.id);
    anyhow::ensure!(draft_dir.join("model.safetensors").exists(), "draft model '{draft_id}' is not installed yet (POST /v1/models/install first)");
    let draft_model = GptModel::load(&draft_dir).map_err(|e| anyhow::anyhow!("failed to load draft model '{draft_id}' from {draft_dir:?}: {e:#}"))?;

    let prompt_ids = target.tokenizer.encode(prompt).context("open-cuda-llm tokenizer encode failed")?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");

    let (generated_ids, stats) = target
        .model
        .generate_speculative(device, &draft_model, &prompt_ids, max_new_tokens, draft_k)
        .context("GptModel::generate_speculative failed")?;
    let text = target.tokenizer.decode(&generated_ids).context("open-cuda-llm tokenizer decode failed")?;
    Ok((text, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;

    /// sibling repoの実GPT-2重み(`open-cuda`側で既にダウンロード・検証
    /// 済み、他の既存テスト・デフォルト設定と同じディレクトリ)を使い、
    /// `select_model`が実際に成功し、`active_model_dir()`が反映され、
    /// 切り替え後の`generate`が実際に動作することを確認する。実重みが
    /// この環境に無い場合はスキップする(ネットワーク到達不能環境でも
    /// `cargo test`全体を壊さないため)。
    #[test]
    fn select_model_succeeds_for_a_real_directory_and_updates_active_model_dir() {
        let dir = PathBuf::from("../open-cuda/crates/open-cuda-llm/models/gpt2");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping select_model test: real GPT-2 weights not present at {dir:?}");
            return;
        }
        select_model(dir.clone()).expect("select_model should succeed for a real, valid model directory");
        let active = active_model_dir().expect("active_model_dir should be Some after a successful select_model");
        assert_eq!(active, dir);

        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let text = generate(&device, "Hello", 2).expect("generate should succeed against the newly selected model");
        assert!(!text.is_empty(), "generated text should not be empty");
    }

    /// 検索結果活用の新旧プロンプト書式(2026-08-26改善)を、実GPT-2
    /// 124M重みで定性比較する。CIでは実行しない(`--ignored`指定時のみ)、
    /// 出力は`--nocapture`で目視確認する用途(自動アサーションでの
    /// 「活用できている/いない」判定はGPT-2の出力が決定的でも意味論的な
    /// 一致判定が難しいため行わない、正直な限界)。
    #[test]
    #[ignore]
    fn qualitative_compare_old_vs_new_search_prompt_format() {
        let dir = PathBuf::from("models/gpt2");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping qualitative comparison: real GPT-2 weights not present at {dir:?}");
            return;
        }
        select_model(dir).expect("select_model should succeed for a real, valid model directory");
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);

        let context = "1. Rust Programming Language: Rust is a systems programming language \
            focused on safety, speed, and concurrency.";
        let question = "What is Rust used for?";

        let old_format = format!("Reference information from a web search:\n{context}\n\n{question}");
        let new_format = crate::web_search::build_search_augmented_prompt(context, question);

        let old_output = generate(&device, &old_format, 40).expect("old-format generate should succeed");
        let new_output = generate(&device, &new_format, 40).expect("new-format generate should succeed");

        eprintln!("=== old format prompt ===\n{old_format}\n=== old format output ===\n{old_output}\n");
        eprintln!("=== new format prompt ===\n{new_format}\n=== new format output ===\n{new_output}\n");
    }

    #[test]
    fn select_model_fails_cleanly_for_a_nonexistent_directory() {
        let bogus = PathBuf::from(format!("/definitely/does/not/exist/{}", rand_suffix()));
        let result = select_model(bogus);
        assert!(result.is_err(), "select_model should return an error for a nonexistent directory, not panic");
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64).unwrap_or(0)
    }
}
