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
/// **正直な開示**: 現在のモデルカタログ(`model_catalog.rs`)最大が
/// `gpt2-xl`(1.5B、約6.43GB)に留まるため、これらの高VRAM帯GPUを
/// 実際に検出しても、本ヒューリスティックはカタログ最大の`gpt2-xl`を
/// 推奨するだけで、それ以上大きなモデルを新たに推奨することはない
/// (VRAMが余っても実際に活用できるより大きなモデルがカタログに
/// 存在しないため)——名前だけ挙げて実在しない性能向上を示唆しない
/// ようにする。
fn recommend_id_for_vram(vram_bytes: Option<u64>) -> &'static str {
    const GB: u64 = 1024 * 1024 * 1024;
    match vram_bytes {
        None => "gpt2",                    // CPU実行のみ・検出不能 → 安全側の最小サイズ固定
        Some(v) if v < 2 * GB => "gpt2",        // VRAM 2GB未満 → 124M
        Some(v) if v < 4 * GB => "gpt2-medium",  // 2-4GB → 355M
        Some(v) if v < 8 * GB => "gpt2-large",   // 4-8GB → 774M
        Some(_) => "gpt2-xl",                    // 8GB以上(RTX 5090の32GB・RTX 6000 Adaの48GB・RTX PRO 6000 Blackwellの96GB等も含む) → カタログ最大の1.5B
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
    /// 精度を誇張しない旨の開示文言(常にレスポンスへ含める)。
    pub disclosure_ja: &'static str,
}

pub fn recommend() -> Recommendation {
    let hardware = detect();
    let recommended_model_id = recommend_id_for_vram(hardware.vram_bytes);
    Recommendation {
        hardware,
        recommended_model_id,
        disclosure_ja: "これはモデルサイズ(パラメータ数×4バイトのfp32概算)とVRAM容量の単純な比較に\
            基づく簡易的な目安であり、精密な性能予測ではありません。実際の必要メモリはKVキャッシュ・\
            アクティベーション等で変動します。生成処理自体は現状CPUで実行されます\
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
}
