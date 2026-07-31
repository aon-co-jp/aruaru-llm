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
fn recommend_id_for_vram(vram_bytes: Option<u64>) -> &'static str {
    const GB: u64 = 1024 * 1024 * 1024;
    match vram_bytes {
        None => "gpt2",                    // CPU実行のみ・検出不能 → 安全側の最小サイズ固定
        Some(v) if v < 2 * GB => "gpt2",        // VRAM 2GB未満 → 124M
        Some(v) if v < 4 * GB => "gpt2-medium",  // 2-4GB → 355M
        Some(v) if v < 8 * GB => "gpt2-large",   // 4-8GB → 774M
        Some(_) => "gpt2-xl",                    // 8GB以上 → 1.5B
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
