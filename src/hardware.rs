//! ハードウェア検出→推奨LLMサイズの判定(2026-07-27新規、ユーザー指示
//! 「open-directx・open-cuda・aruaru-llmの組み合わせで『お勧めLLMを
//! ダウンロード』ボタンで最適なLLMをダウンロードする機能を搭載して」
//! への対応)。
//!
//! ## 正直な開示(最重要、誇張しない)
//!
//! ここで行っているのは**精密な性能予測ではない**。「モデルサイズ
//! (パラメータ数×4バイト、fp32換算の概算)がVRAM容量に収まるか」という
//! 単純な比較に基づく簡易的な目安に過ぎない。実際に必要なメモリは
//! KVキャッシュ・アクティベーション・OS/ドライバのオーバーヘッド等で
//! 変動するため、本ヒューリスティックの推奨値ぎりぎりの構成では
//! 実際にはVRAM不足になる可能性がある。また現行の`open-cuda-llm`
//! (`GptModel::generate`)は依然CPU実行のみ(`open-cuda`側CLAUDE.mdの
//! 2026-07-26 HANDOFF「安易なGPU配線は逆に遅くなりうる」の結論通り、
//! 1トークンごとの逐次デコードではVulkan/DirectXディスパッチの固定
//! オーバーヘッドがCPU実行より不利になりうるため、意図的にGPU推論配線は
//! 行っていない)——ここでの「VRAM検出」は**推奨モデルサイズを選ぶための
//! 入力**であり、選ばれたモデルの生成処理自体は引き続きCPU
//! (`opencuda_cpu::CpuDevice`)で行われる。
//!
//! ## GPU検出の経路(open-directx/open-cudaとの連携)
//!
//! `open-cuda`が提供する2つの独立したGPU検出経路を両方試す:
//! - `opencuda-vulkan`(`hw-detect-vulkan` feature、`real-vulkan`経由)——
//!   `ash`によるVulkan物理デバイス列挙。クロスプラットフォーム
//!   (Windows/Linux/Android)。
//! - `opencuda-directx`(`hw-detect-directx` feature、`real-dx12`経由)——
//!   DXGIアダプタ列挙(`EnumAdapters1(0)`)。Windows専用。
//!
//! いずれのfeatureも既定では無効(opt-in、CPUのみの環境やクロス
//! コンパイル環境でVulkanローダー/Windows SDKへの依存を強制しないため、
//! `opencuda-vulkan`/`opencuda-directx`自身の既存の`real-vulkan`/
//! `real-dx12` feature設計方針と同じ)。**両方有効な場合はDirectXを優先し
//! Vulkanの結果をクロスチェックとしてログへ記録する**(どちらの経路の
//! 情報を使うかを明確にするため——ユーザー指示「ハードウェア検出ロジック
//! がどちらの経路からの情報を使うか明確にドキュメント化すること」への
//! 対応。2026-07-27、ユーザー指示によりVulkan優先→DirectX優先へ反転)。
//! 両方失敗、または両feature無効の場合はCPUのみとみなし、
//! 安全側(最小サイズ)にフォールバックする。

use serde::Serialize;

#[cfg(any(feature = "hw-detect-vulkan", feature = "hw-detect-directx"))]
use opencuda_core::GpuDevice as _;

/// 検出したハードウェア能力の要約(API/UIへそのまま返せる形)。
#[derive(Debug, Clone, Serialize)]
pub struct HardwareSummary {
    /// 実際にGPUを検出できたか。
    pub gpu_detected: bool,
    /// 検出に使われた経路("vulkan" / "directx" / "cpu-only-fallback")。
    pub detection_path: &'static str,
    pub gpu_name: Option<String>,
    pub vram_bytes: Option<u64>,
    /// Vulkan/DirectXの両方が有効な場合、クロスチェック結果が一致したか
    /// どうか(`None`は片方または両方が無効/失敗したためチェック不能)。
    pub cross_check_agreement: Option<bool>,
}

/// VRAM容量に応じた推奨モデルサイズの簡易ヒューリスティック。
/// **正直な開示**: モデルサイズ(パラメータ数)とVRAM容量の単純な比較に
/// 基づく目安であり、精密な性能予測ではない(モジュールdoc参照)。
///
/// ## 2026-08-11追記: 高級GPUの実在確認(日英Web検索で裏取り)
///
/// ユーザーから「RTX5950X」という製品名の言及があったが、これは実在
/// しない(「5950X」はAMD Ryzenの型番であり、NVIDIA RTXシリーズの
/// 命名規則とは異なる——ユーザーへ確認済みの誤り)。実在する2026年
/// 時点の高VRAM帯NVIDIA製品を日英Web検索で確認した:
/// **RTX 5090**(Blackwellアーキテクチャ、32GB GDDR7、
/// [Jarvis Labs](https://jarvislabs.ai/ai-faqs/nvidia-rtx-5090-specs))・
/// **RTX 6000 Ada**(48GB、
/// [getdeploying.com](https://getdeploying.com/gpus/nvidia-rtx-5090-vs-nvidia-rtx-6000-ada))・
/// **RTX PRO 6000 Blackwell**(96GB、ECC対応のワークステーション向け、
/// [tech.sportskeeda.com](https://tech.sportskeeda.com/gaming-news/nvidia-rtx-pro-6000-vs-rtx-5090))。
/// ## 2026-09-03追記: 高VRAM帯 GPU 3ベンダー分の実在確認(英日中 Web 検索)
///
/// ユーザーから「RTX 5900」「AMD/Intel 32GB」の言及があった。**「RTX 5900」は
/// 現行 Blackwell 世代に実在しない**(2003 年の GeForceFX 5900 のみ。現行
/// RTX 50 は 5090 / 5080 / 5070)。実在する 2026 年時点の高 VRAM 帯製品:
/// - **NVIDIA**: RTX 5090(Blackwell、**32GB** GDDR7)/ RTX 5080(16GB、
///   5080 Super は 24GB)/ RTX PRO 6000 Blackwell(**96GB**、ECC、
///   24,064 CUDA コア)。全 Blackwell → FP8 Tensor Core あり
///   ([NVIDIA RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/))。
/// - **AMD**: Radeon AI PRO R9700(RDNA4、**32GB** GDDR6、64 CU、128 個の
///   第2世代 AI アクセラレータ、**FP8 matrix 383 TFLOPS dense / 766 sparse**、
///   FP8/FP16/INT8 対応、$1,299、2025-07-23 発売
///   ([ServeTheHome](https://www.servethehome.com/amd-takes-aim-for-workstation-ai-market-with-radeon-ai-pro-r9700/)))。
/// - **Intel**: Arc Pro B60(Battlemage、**24GB** GDDR6。デュアル構成の
///   サードパーティ製で 48GB もあり)。
///
/// **ベンダー非依存**: これら 3 ベンダーの GPU はいずれも `open-cuda` の
/// **同一 Vulkan / SPIR-V 経路**で到達する(カード別コードは無い)。FP8 の
/// 実行は移植性の高い `VK_EXT_shader_float8` + `VK_KHR_cooperative_matrix`
/// 経路(NVIDIA は 2025-06、AMD は Adrenalin 25.10.2 以降の出荷ドライバで
/// 対応)を第一実装先とする。正本: `open-cuda/OmniGPU-Design.md` §8.5 / §11.6。
///
/// **正直な開示**: 現在のモデルカタログ(`model_catalog.rs`)最大が
/// `gpt2-xl`(1.5B、約6.43GB)に留まるため、これらの高VRAM帯GPUを
/// 実際に検出しても、本ヒューリスティックはカタログ最大の`gpt2-xl`を
/// 推奨するだけで、それ以上大きなモデルを新たに推奨することはない
/// (VRAMが余っても実際に活用できるより大きなモデルがカタログに
/// 存在しないため)——名前だけ挙げて実在しない性能向上を示唆しない
/// ようにする。カタログ拡張(より大きい実在モデルの追加)が先。
#[cfg_attr(not(test), allow(dead_code))] // 2026-09-05: recommend()がrecommend_at_precisionへ直接委譲するようになったため本体経路では未使用、テスト(F32委譲との一致確認)のためだけに残す
fn recommend_id_for_vram(vram_bytes: Option<u64>) -> &'static str {
    recommend_id_for_vram_at_precision(vram_bytes, InferencePrecision::F32)
}

/// **2026-09-03追記: 精度(F16/F32/F64/F128)を考慮したVRAM見積もり
/// (ユーザー指示「open-directx・open-cuda・aruaru-llmで今後32GB VRAM級の
/// NVIDIA/AMD/Intel GPUを前提に、F16/F32/F64、さらにF128まで見据えて
/// 開発する」への対応)**。
///
/// ## 何が変わるか
/// 従来の`recommend_id_for_vram`(既定`F32`へ委譲、後方互換)は
/// モジュールdoc冒頭で開示している通り「パラメータ数×4バイト
/// (fp32換算)がVRAM容量に収まるか」という単純比較だった。しかし
/// **同じVRAM予算でも推論精度がfp16ならパラメータあたり2バイト
/// で済み、fp32の約2倍のパラメータ数のモデルが収まる**——これは
/// 実際のデプロイで広く行われている最適化であり、無視すると
/// 「32GB級GPUなのにfp32換算でしかモデルサイズを見積もらない」
/// という過小評価になる。本関数はその精度依存性を明示的な
/// `InferencePrecision`引数で表現する。
///
/// ## バイト/パラメータの根拠(誇張しない)
/// - `F16`(半精度、2バイト/パラメータ): 実際に量子化/半精度推論で
///   広く使われる形式(`open-cuda`側`tensor_f32`がF16→f32変換に既に
///   対応済み、2026-09-02 HANDOFF参照)。
/// - `F32`(単精度、4バイト/パラメータ): 従来の既定・唯一の想定
///   だった精度。
/// - `F64`(倍精度、8バイト/パラメータ): GPUの推論用途では通常
///   使われない(学習の数値安定性検証等が主用途)が、`open-cuda`側の
///   KernelArg型システムがF16/F32/F64/F128を型として揃える方針
///   (companion agentが同時に`opencuda-core`/`opencuda-blas`へ実装中)
///   に合わせ、一貫性のため見積もり側にも用意する。
/// - `F128`(四倍精度、16バイト/パラメータ): **正直な開示(最重要)**
///   ——GPU上でF128のネイティブTensor Core/ALUサポートを持つハード
///   ウェアは(NVIDIA/AMD/Intel問わず)存在しない。ソフトウェア
///   エミュレーション(倍々精度合成等)でのみ実現可能で、実用上の
///   推論速度は壊滅的に遅くなる。この見積もり関数へ含めているのは
///   `open-cuda`側の型システム(KernelArg)がF128をソフトウェア実装
///   として持つことに合わせた**計算上の一貫性のためだけ**であり、
///   「F128でLLM推論するのが実用的」という主張は一切していない。
///
/// ## 依然として正直な限界(モジュールdoc冒頭を参照、変わらない)
/// パラメータ数×バイト/パラメータ、という単純比較のままであり、
/// KVキャッシュ・アクティベーション・OS/ドライバオーバーヘッドは
/// 考慮していない。精度が変わってもこの限界自体は変わらない。
fn recommend_id_for_vram_at_precision(
    vram_bytes: Option<u64>,
    precision: InferencePrecision,
) -> &'static str {
    const GB: u64 = 1024 * 1024 * 1024;
    // F32(4バイト/パラメータ)を基準に、他精度はバイト比で
    // 「見た目のVRAM容量」を換算する(容量を広げる/狭める側どちらも
    // 同じ式で表現できる: 換算後の容量 = 実VRAM * (4 / bytes_per_param))。
    let bytes_per_param = precision.bytes_per_param() as f64;
    let scale = 4.0 / bytes_per_param;

    match vram_bytes {
        None => "gpt2", // CPU実行のみ・検出不能 → 安全側の最小サイズ固定(精度に関わらず不変)
        Some(v) => {
            let scaled = (v as f64 * scale) as u64;
            match scaled {
                s if s < 2 * GB => "gpt2",
                s if s < 4 * GB => "gpt2-medium",
                s if s < 8 * GB => "gpt2-large",
                _ => "gpt2-xl", // カタログ最大(2026-08-11/09-03 HANDOFF参照、これ以上大きい実在モデルは未追加)
            }
        }
    }
}

/// 推論精度の想定(`recommend_id_for_vram_at_precision`向け)。
/// `open-cuda`側`opencuda-core`/`opencuda-blas`のKernelArg型システムが
/// 並行して同じ4種(F16/F32/F64/F128)を実装している(companion agent、
/// 別リポジトリ)ことに合わせた型——このenum自体はVRAM見積もり計算にしか
/// 使わず、実際の推論ディスパッチ(`generation.rs`)への配線は無い——
/// `open-cuda-llm::GptModel`の重みロード自体は依然F32(一部F16/BF16/
/// FP8→f32変換、2026-09-02 HANDOFF参照)前提で、この推奨サイズ計算が
/// 選んだ精度をロード時のdtypeとして実際に使う経路は無い(誇張しない)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InferencePrecision {
    F16,
    F32,
    F64,
    /// ソフトウェアエミュレーションのみ、GPU実行での実用性は無い
    /// (正直な開示、上記doc参照)。
    F128,
}

impl InferencePrecision {
    fn bytes_per_param(self) -> u32 {
        match self {
            InferencePrecision::F16 => 2,
            InferencePrecision::F32 => 4,
            InferencePrecision::F64 => 8,
            InferencePrecision::F128 => 16,
        }
    }

    /// `GET /v1/recommend?precision=f16`等のクエリ文字列から解釈する
    /// (大小無視、`f16`/`fp16`/`half`等の別名も受け付ける)。未知の値は
    /// `None`——呼び出し側が正直に`400`を返せるようにする(黙って
    /// F32へフォールバックしない、既存の「サービスを壊さない」設計とは
    /// 別の「入力ミスを隠さない」設計、2026-08-07の`/v1/generate`空入力
    /// 400化と同じ思想)。
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(InferencePrecision::F16),
            "f32" | "fp32" | "float" | "single" => Some(InferencePrecision::F32),
            "f64" | "fp64" | "double" => Some(InferencePrecision::F64),
            "f128" | "fp128" | "quad" => Some(InferencePrecision::F128),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            InferencePrecision::F16 => "f16",
            InferencePrecision::F32 => "f32",
            InferencePrecision::F64 => "f64",
            InferencePrecision::F128 => "f128",
        }
    }
}

#[cfg(feature = "hw-detect-vulkan")]
fn detect_via_vulkan() -> Option<(String, u64)> {
    match opencuda_vulkan::real::VulkanDevice::new(0) {
        Ok(dev) => {
            let info = dev.info();
            tracing::info!(name = %info.name, vram = info.total_memory, "hardware detection: vulkan path succeeded");
            Some((info.name.clone(), info.total_memory))
        }
        Err(e) => {
            tracing::info!("hardware detection: vulkan path failed (no GPU or driver issue): {e:#}");
            None
        }
    }
}

#[cfg(not(feature = "hw-detect-vulkan"))]
fn detect_via_vulkan() -> Option<(String, u64)> {
    None
}

#[cfg(feature = "hw-detect-directx")]
fn detect_via_directx() -> Option<(String, u64)> {
    match opencuda_directx::real::DirectXDevice::new(0) {
        Ok(dev) => {
            let info = dev.info();
            tracing::info!(name = %info.name, vram = info.total_memory, "hardware detection: directx path succeeded");
            Some((info.name.clone(), info.total_memory))
        }
        Err(e) => {
            tracing::info!("hardware detection: directx path failed (no GPU, non-Windows, or driver issue): {e:#}");
            None
        }
    }
}

#[cfg(not(feature = "hw-detect-directx"))]
fn detect_via_directx() -> Option<(String, u64)> {
    None
}

/// **将来構想・未実装(2026-08-17追記、ユーザー指示「将来NPU搭載機種で
/// あった場合のハードウェアアクセラレーターも考慮しておいて」への
/// 対応)**: 現状このモジュールが検出できるのはGPU(Vulkan/DirectX)と
/// CPUのみで、NPU(Android端末のNNAPI/Qualcomm QNN/Google Tensor
/// Edge TPU、Windows端末のDirectML NPU等)の検出・ディスパッチ経路は
/// `open-cuda`側にも一切実装されていない。実装するには、(a)
/// プラットフォームごとに異なるNPU APIへの個別バインディング
/// (Android: NNAPI経由が最も汎用的、iOS: Core ML、Windows: DirectML)、
/// (b) `opencuda-core::GpuDevice`トレイトをNPU実行に拡張する設計変更
/// (現状はGPU/CPU二値のディスパッチ前提)が必要な、規模の大きい増分と
/// なる。今回は「将来NPU搭載機種が増えた場合に見落とさないための
/// 記録」として本コメントを残すに留め、実装は行っていない——
/// 誇張せず、NPU対応は現状皆無であることを明記する。
///
/// ハードウェアを検出する。Vulkan経路を優先し、DirectX経路が(feature有効
/// なら)取れた場合はクロスチェックとしてログ比較する(モジュールdoc
/// 「GPU検出の経路」参照)。
pub fn detect() -> HardwareSummary {
    let vulkan = detect_via_vulkan();
    let directx = detect_via_directx();

    let cross_check_agreement = match (&vulkan, &directx) {
        (Some((_, v_vram)), Some((_, d_vram))) => {
            // 完全一致ではなく近似一致で判定(DXGIの`DedicatedVideoMemory`と
            // Vulkanの`VkPhysicalDeviceMemoryProperties`由来推定値は、
            // カウント対象(共有メモリ扱い等)がAPIごとに微妙に異なりうる
            // ため、10%程度の差は許容する)。
            let diff = v_vram.abs_diff(*d_vram);
            let tolerance = (*v_vram).max(*d_vram) / 10;
            let agree = diff <= tolerance;
            tracing::info!(vulkan_vram = v_vram, directx_vram = d_vram, agree, "cross-checked vulkan vs directx VRAM report");
            Some(agree)
        }
        _ => None,
    };

    // 2026-07-27追記(ユーザー指示: 「Vulkan環境を重視してましたが、今度は
    // 逆に、open-directxを重視するように変更して」): 優先順位を
    // DirectX→Vulkanへ反転した。クロスチェックの計算自体(上記)は
    // どちらを優先するかに関わらず対称なので変更不要。
    if let Some((name, vram)) = directx {
        return HardwareSummary { gpu_detected: true, detection_path: "directx", gpu_name: Some(name), vram_bytes: Some(vram), cross_check_agreement };
    }
    if let Some((name, vram)) = vulkan {
        return HardwareSummary { gpu_detected: true, detection_path: "vulkan", gpu_name: Some(name), vram_bytes: Some(vram), cross_check_agreement };
    }
    HardwareSummary { gpu_detected: false, detection_path: "cpu-only-fallback", gpu_name: None, vram_bytes: None, cross_check_agreement: None }
}

/// **NPU(Neural Processing Unit)自動検出(2026-08-19新規実装、ユーザー指示
/// 「NPUがPC側にあれば、それも自動検出して計算に使用する」への対応)**。
///
/// ## 正直な開示(最重要、誇張しない)
/// ここで実装したのは**検出のみ**である。Windows上で`Get-CimInstance
/// Win32_PnPEntity`(デバイスマネージャ相当の情報源)を呼び、デバイス名に
/// "NPU"・"Neural"・"AI Boost"(Intel NPUのデバイス名に含まれることが多い)・
/// "Hexagon"(Qualcomm NPU)のいずれかを含むデバイスが見つかれば「NPU検出」と
/// する簡易ヒューリスティック。**実際にこの開発機(2026-08-19時点)で
/// 実行したところ、該当デバイスは1件も見つからなかった
/// (`Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match
/// 'NPU|Neural' }`が空を返した)——このマシンにNPUは搭載されていない。**
/// NPU上で実際に計算を実行する処理(DirectML NPU推論等)は、対応SDKが
/// この環境に存在しないため実装していない。検出できた場合でも、
/// `idle_background_fold`のステップ処理自体は引き続きCPU
/// (`opencuda_cpu::CpuDevice`)上でのみ実行される——NPUが見つかった旨を
/// `AcceleratorInfo`へ記録するだけで、実際にNPUへディスパッチする経路は
/// 無い(このモジュールの他のGPU検出と同じ「検出はできるが本物の実行
/// パイプラインへは未配線」という設計上の限界を正直に明記する)。
#[cfg(target_os = "windows")]
pub fn detect_npu() -> Option<String> {
    use std::process::Command;
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'NPU|Neural|AI Boost|Hexagon' } | Select-Object -First 1 -ExpandProperty Name",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(not(target_os = "windows"))]
pub fn detect_npu() -> Option<String> {
    // 正直な開示: Windows以外(Linux VPS等)向けのNPU検出経路は未実装
    // (Android端末のNNAPI経由NPUは別経路、`CLAUDE.md`のHANDOFF参照)。
    None
}

/// USB接続されたAndroid端末の一覧を`adb devices`で検出する
/// (2026-08-19新規実装、ユーザー指示「使わなくなった複数のスマホを
/// USBで接続して...統合した計算リソースとして利用できるようにする」への
/// 対応の最小の一歩)。
///
/// ## 正直な開示(最重要)
/// これは「N台のスマホが接続されている」ことを検出してログ表示する
/// だけであり、検出したスマホへ実際に計算タスクを送る・NNAPI経由で
/// NPUを稼働させる、といった処理は一切実装していない
/// (`CLAUDE.md`のHANDOFF「USB接続スマホ活用」節参照)。`adb`コマンド
/// 自体がこの開発環境のPATH上に存在しない場合は、その旨を`Err`として
/// 正直に返す(黙って0台と偽装しない)。
pub fn detect_usb_android_devices() -> Result<Vec<String>, String> {
    use std::process::Command;
    let output = Command::new("adb")
        .arg("devices")
        .output()
        .map_err(|e| format!("adb command not available on this machine (adb devices failed to launch: {e})"))?;
    if !output.status.success() {
        return Err(format!(
            "adb devices exited with non-zero status: {:?}",
            output.status.code()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // 1行目は "List of devices attached" ヘッダー、以降 "<serial>\tdevice" の形式。
    let devices: Vec<String> = stdout
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let serial = line.split_whitespace().next()?;
            if line.ends_with("device") {
                Some(serial.to_string())
            } else {
                None
            }
        })
        .collect();
    Ok(devices)
}

/// 利用可能なアクセラレータ(CPU/GPU/NPU、PC/スマホ別)の一覧
/// (2026-08-19新設、`idle_background_fold`のスケジューラ・
/// `GET /v1/background-fold/status`から参照される)。
#[derive(Debug, Clone, Serialize)]
pub struct AcceleratorInventory {
    /// CPUは常に利用可能とみなす(このプロセス自体が動作している時点で
    /// 自明なため検出処理は行わない)。
    pub cpu_available: bool,
    pub gpu: HardwareSummary,
    /// NPUデバイス名(検出できた場合)。実行パイプラインへの配線は
    /// 無い旨、`disclosure`で明記する。
    pub npu_name: Option<String>,
    /// `adb devices`で検出したUSB接続Android端末のシリアル番号一覧。
    /// `adb`自体が使えない環境では`None`(検出不能、0台という意味では
    /// ない——この区別を保つため`Option`にしている)。
    pub usb_android_devices: Option<Vec<String>>,
    pub disclosure: &'static str,
}

pub fn detect_accelerators() -> AcceleratorInventory {
    let gpu = detect();
    let npu_name = detect_npu();
    let usb_android_devices = detect_usb_android_devices().ok();
    AcceleratorInventory {
        cpu_available: true,
        gpu,
        npu_name,
        usb_android_devices,
        disclosure: "これは利用可能なアクセラレータの『検出』結果に過ぎません。\
            NPU・USB接続スマホのCPU/GPU/NPUを実際の計算(Model Folding等)へ \
            ディスパッチする実行パイプラインは未実装です。実際の計算は引き続き \
            PC側のCPU(opencuda_cpu::CpuDevice)上でのみ実行されます。 / This only \
            reports what accelerators were detected. There is no execution pipeline \
            yet that dispatches actual computation to the NPU or to any USB-connected \
            phone's CPU/GPU/NPU. All real computation still runs on the PC's CPU only.",
    }
}

/// 推奨結果(API/UI向け)。
#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub hardware: HardwareSummary,
    pub recommended_model_id: &'static str,
    /// 見積もりに使った精度(2026-09-05新設、既定`f32`)。呼び出し側が
    /// `?precision=f16`等を指定した場合はそれが反映される——`recommend()`
    /// 経由(precision未指定)は常に`"f32"`。
    pub precision_used: &'static str,
    /// 精度を誇張しない旨の開示文言(常にレスポンスへ含める)。
    pub disclosure_ja: &'static str,
}

/// 精度未指定の既定呼び出し(常にF32換算、後方互換)。
pub fn recommend() -> Recommendation {
    recommend_at_precision(InferencePrecision::F32)
}

/// 精度考慮版(2026-09-05新設)。`InferencePrecision`(F16/F32/F64/F128)を
/// 指定してVRAM見積もりを行う——`recommend_id_for_vram_at_precision`
/// (2026-09-03新設)を、これまで内部関数のまま外部から呼び出す経路が
/// 存在しなかった不整合(READMEに書かれていた「未接続」ギャップ)を
/// 解消するために追加した公開エントリポイント。
pub fn recommend_at_precision(precision: InferencePrecision) -> Recommendation {
    let hardware = detect();
    let recommended_model_id = recommend_id_for_vram_at_precision(hardware.vram_bytes, precision);
    Recommendation {
        hardware,
        recommended_model_id,
        precision_used: precision.as_str(),
        disclosure_ja: "これはモデルサイズ(パラメータ数×精度ごとのバイト数概算)とVRAM容量の単純な\
            比較に基づく簡易的な目安であり、精密な性能予測ではありません。実際の必要メモリはKVキャッシュ・\
            アクティベーション等で変動します。生成処理自体は現状CPUで実行され、この見積もりが選んだ精度を\
            実際のロード時dtypeとして使う配線もまだありません\
            (GPU推論配線は逐次デコードではオーバーヘッドが支配的になりうるため見送っています)。",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommend_id_for_vram_thresholds() {
        assert_eq!(recommend_id_for_vram(None), "gpt2");
        assert_eq!(recommend_id_for_vram(Some(1024 * 1024 * 1024)), "gpt2"); // 1GB
        assert_eq!(recommend_id_for_vram(Some(3 * 1024 * 1024 * 1024)), "gpt2-medium"); // 3GB
        assert_eq!(recommend_id_for_vram(Some(6 * 1024 * 1024 * 1024)), "gpt2-large"); // 6GB
        assert_eq!(recommend_id_for_vram(Some(12 * 1024 * 1024 * 1024)), "gpt2-xl"); // 12GB
    }

    #[test]
    fn recommend_for_real_high_end_gpus_caps_at_catalog_max_without_overclaiming() {
        // 2026-08-11: 実在する高VRAM帯NVIDIA製品(日英Web検索で裏取り済み、
        // モジュールdoc参照)を実際のVRAM容量で検証する。いずれもカタログ
        // 最大のgpt2-xl(1.5B)止まりであり、それ以上大きなモデルを
        // 実在しないかのように推奨しないことを確認する。
        let rtx_5090_vram = 32 * 1024 * 1024 * 1024u64; // RTX 5090: 32GB GDDR7
        let rtx_6000_ada_vram = 48 * 1024 * 1024 * 1024u64; // RTX 6000 Ada: 48GB
        let rtx_pro_6000_blackwell_vram = 96 * 1024 * 1024 * 1024u64; // RTX PRO 6000 Blackwell: 96GB
        assert_eq!(recommend_id_for_vram(Some(rtx_5090_vram)), "gpt2-xl");
        assert_eq!(recommend_id_for_vram(Some(rtx_6000_ada_vram)), "gpt2-xl");
        assert_eq!(recommend_id_for_vram(Some(rtx_pro_6000_blackwell_vram)), "gpt2-xl");

        // 2026-09-03: 3ベンダー分(NVIDIA/AMD/Intel)の高VRAM帯 GPU も
        // 同じ Vulkan/SPIR-V 経路で扱う。VRAM に応じた推奨は現行カタログ
        // 上限(gpt2-xl)で頭打ちであることを、AMD/Intel でも確認する。
        let amd_r9700_vram = 32 * 1024 * 1024 * 1024u64; // AMD Radeon AI PRO R9700: 32GB GDDR6
        let intel_arc_pro_b60_vram = 24 * 1024 * 1024 * 1024u64; // Intel Arc Pro B60: 24GB
        let intel_arc_pro_b60_dual_vram = 48 * 1024 * 1024 * 1024u64; // dual-B60 (3rd-party): 48GB
        assert_eq!(recommend_id_for_vram(Some(amd_r9700_vram)), "gpt2-xl");
        assert_eq!(recommend_id_for_vram(Some(intel_arc_pro_b60_vram)), "gpt2-xl");
        assert_eq!(recommend_id_for_vram(Some(intel_arc_pro_b60_dual_vram)), "gpt2-xl");
    }

    #[test]
    fn recommend_without_any_gpu_feature_falls_back_to_cpu_and_smallest_model() {
        // hw-detect-vulkan/hw-detect-directxのいずれもfeature無効な
        // デフォルトビルドでは、detect()は常にcpu-only-fallbackを返す
        // (このテスト自体がデフォルトfeatureで実行される想定)。
        #[cfg(not(any(feature = "hw-detect-vulkan", feature = "hw-detect-directx")))]
        {
            let r = recommend();
            assert!(!r.hardware.gpu_detected);
            assert_eq!(r.hardware.detection_path, "cpu-only-fallback");
            assert_eq!(r.recommended_model_id, "gpt2");
        }
    }

    #[test]
    fn recommend_always_resolves_to_a_real_catalog_entry() {
        let r = recommend();
        assert!(crate::model_catalog::find(r.recommended_model_id).is_some());
    }

    #[test]
    fn recommend_at_precision_defaults_to_f32_matching_legacy_function() {
        // 2026-09-03: recommend_id_for_vramはF32へ委譲するだけの後方互換
        // ラッパーであることを確認(既存呼び出し元のシグネチャ・挙動を
        // 一切変えない設計)。
        for vram in [None, Some(1 * 1024 * 1024 * 1024), Some(3 * 1024 * 1024 * 1024),
                     Some(6 * 1024 * 1024 * 1024), Some(32 * 1024 * 1024 * 1024)] {
            assert_eq!(
                recommend_id_for_vram(vram),
                recommend_id_for_vram_at_precision(vram, InferencePrecision::F32)
            );
        }
    }

    #[test]
    fn f16_precision_recommends_a_larger_model_than_f32_for_the_same_vram() {
        // 2026-09-03: 32GB級カード(AMD R9700・RTX 5090等)を、fp32換算
        // ではなくfp16推論で使う想定なら、同じVRAM予算で約2倍の
        // パラメータ数のモデルが収まるはず——F16の方がF32以上のサイズを
        // 推奨することを確認する(実際にはカタログ上限gpt2-xlで頭打ちに
        // なるため、上限に達しない中間サイズの容量で比較する)。
        let vram_4gb = 4 * 1024 * 1024 * 1024u64; // ちょうどF32ではgpt2-medium/gpt2-large境界
        let f32_choice = recommend_id_for_vram_at_precision(Some(vram_4gb), InferencePrecision::F32);
        let f16_choice = recommend_id_for_vram_at_precision(Some(vram_4gb), InferencePrecision::F16);
        // F32: 4GB ちょうど → gpt2-large(4GB以上の枝)。
        assert_eq!(f32_choice, "gpt2-large");
        // F16: 換算後8GB相当 → gpt2-large(8GB未満)ではなくgpt2-xlへ進む
        // (8GB以上の枝、カタログ最大)。
        assert_eq!(f16_choice, "gpt2-xl");

        // より小さい予算でも同じ傾向(F16が同容量でF32以上のサイズを選ぶ)
        // であることを、複数のVRAM容量で確認する。
        for vram in [
            2 * 1024 * 1024 * 1024u64,
            3 * 1024 * 1024 * 1024u64,
            6 * 1024 * 1024 * 1024u64,
        ] {
            let f32_id = recommend_id_for_vram_at_precision(Some(vram), InferencePrecision::F32);
            let f16_id = recommend_id_for_vram_at_precision(Some(vram), InferencePrecision::F16);
            let rank = |id: &str| match id {
                "gpt2" => 0,
                "gpt2-medium" => 1,
                "gpt2-large" => 2,
                "gpt2-xl" => 3,
                _ => panic!("unexpected catalog id in test: {id}"),
            };
            assert!(
                rank(f16_id) >= rank(f32_id),
                "F16 should never recommend a smaller model than F32 for the same VRAM (vram={vram}, f32={f32_id}, f16={f16_id})"
            );
        }
    }

    #[test]
    fn f64_and_f128_precision_recommend_a_smaller_or_equal_model_than_f32() {
        // 2026-09-03: F64(8バイト/パラメータ)・F128(16バイト/パラメータ、
        // ソフトウェアエミュレーションのみ・実用性は無いと正直に開示済み)
        // は、同じVRAM予算でF32よりパラメータ数の少ないモデルしか収まら
        // ないはず——F32以下のサイズを推奨することを確認する。
        for vram in [
            2 * 1024 * 1024 * 1024u64,
            4 * 1024 * 1024 * 1024u64,
            8 * 1024 * 1024 * 1024u64,
            32 * 1024 * 1024 * 1024u64,
        ] {
            let rank = |id: &str| match id {
                "gpt2" => 0,
                "gpt2-medium" => 1,
                "gpt2-large" => 2,
                "gpt2-xl" => 3,
                _ => panic!("unexpected catalog id in test: {id}"),
            };
            let f32_id = recommend_id_for_vram_at_precision(Some(vram), InferencePrecision::F32);
            let f64_id = recommend_id_for_vram_at_precision(Some(vram), InferencePrecision::F64);
            let f128_id = recommend_id_for_vram_at_precision(Some(vram), InferencePrecision::F128);
            assert!(rank(f64_id) <= rank(f32_id));
            assert!(rank(f128_id) <= rank(f64_id));
        }
    }

    #[test]
    fn precision_choice_never_affects_the_none_vram_fallback() {
        // GPU検出不能・CPU実行のみの場合は、精度に関わらず常に安全側
        // (最小モデル)へ固定フォールバックすることを確認する。
        for precision in [
            InferencePrecision::F16,
            InferencePrecision::F32,
            InferencePrecision::F64,
            InferencePrecision::F128,
        ] {
            assert_eq!(recommend_id_for_vram_at_precision(None, precision), "gpt2");
        }
    }

    #[test]
    fn bytes_per_param_matches_the_disclosed_precision_semantics() {
        assert_eq!(InferencePrecision::F16.bytes_per_param(), 2);
        assert_eq!(InferencePrecision::F32.bytes_per_param(), 4);
        assert_eq!(InferencePrecision::F64.bytes_per_param(), 8);
        assert_eq!(InferencePrecision::F128.bytes_per_param(), 16);
    }
}
