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

fn load_from_dir(dir: &std::path::Path) -> Result<LoadedGpt, String> {
    #[allow(unused_mut)]
    let mut model = GptModel::load(dir).map_err(|e| format!("failed to load GPT-2 weights from {dir:?}: {e:#}"))?;
    let tokenizer = GptTokenizer::load(dir).map_err(|e| format!("failed to load GPT-2 tokenizer from {dir:?}: {e:#}"))?;
    #[allow(unused_mut)]
    let mut matmul_spirv_wired = false;
    #[cfg(feature = "real-vulkan")]
    {
        matmul_spirv_wired = wire_matmul_spirv(&mut model);
    }
    Ok(LoadedGpt { model, tokenizer, dir: dir.to_path_buf(), matmul_spirv_wired })
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
    } else {
        "-cpu"
    }
}

/// 現在アクティブなモデルディレクトリ名・実際の実行経路(Vulkan/CPU)を
/// 反映したエンジン識別子。ディレクトリ名がわかればそれをそのまま
/// 埋め込み(例: `"gpt2-medium-greedy-decode-v0-open-cuda-llm-vulkan"`)、
/// 未ロード(まだ一度もモデルをロードしていない)なら`device`のみを見て
/// デフォルトの`ENGINE_GPT2_GREEDY`相当のラベルを返す。
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

/// `prompt`の続きを`max_new_tokens`個、貪欲デコード(argmax、サンプリング
/// 無し)で生成する。GPT-2実重み・実トークナイザが読み込めない場合はエラー
/// を返す(黙って別経路にフォールバックしない、意図分類とは別軸のため
/// bag-of-words的な代替は存在しない)。
pub fn generate(device: &Arc<dyn GpuDevice>, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let loaded = active_or_load_default().map_err(|e| anyhow::anyhow!(e))?;
    let prompt_ids = loaded.tokenizer.encode(prompt).context("open-cuda-llm tokenizer encode failed")?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");
    let generated_ids = loaded.model.generate(device, &prompt_ids, max_new_tokens).context("GptModel::generate failed")?;
    loaded.tokenizer.decode(&generated_ids).context("open-cuda-llm tokenizer decode failed")
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
