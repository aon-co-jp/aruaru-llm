//! この機械で動かせるLLMの一覧(選択用)と、ハードウェア変更の検知による
//! 自動導入(2026-11-19、ユーザー指示「グラフィックボードの交換や再インストールの
//! 際に、適したLLMを自動でインストールし、その機械のスペックで入れられるLLMの
//! 一覧も表示して選べるように」への対応)。
//!
//! 正直な開示: 「入る/入らない」は**目安**(ファイルサイズ×2をメモリ必要量と
//! みなし、搭載メモリの6割と比べるだけ)。現状の生成はCPUで行うため、判定には
//! GPUのVRAMではなく**搭載メモリ(RAM)**を使う。スマホ・タブレット・WEBサーバーでも
//! 同じ判定を使う。自動導入は「前回記録したハードウェアと今回が違うとき」だけで、
//! 初回起動では何もダウンロードしない(既存の配備を勝手に変えないため)。
//! 環境変数 `ARUARU_LLM_AUTO_MODEL=0` で自動導入を止められる。

use serde::Serialize;

use crate::{hardware, model_catalog};

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct FitEntry {
    pub family: &'static str,
    pub id: &'static str,
    pub display_name_ja: &'static str,
    pub display_name_en: &'static str,
    pub approx_size_mb: u32,
    pub license_note_ja: &'static str,
    /// この機械のメモリに入る見込みか(目安)。
    pub fits: bool,
    pub installed: bool,
    /// この機械へのお勧め(VRAMに基づく既存の推奨と一致するもの)。
    pub recommended: bool,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CompatibleResponse {
    pub hardware: hardware::HardwareSummary,
    pub ram_bytes: Option<u64>,
    pub models: Vec<FitEntry>,
    pub explanation: Vec<(&'static str, &'static str)>,
}

/// 搭載メモリ(バイト)。取れなければ`None`。Windows/Linux/Android/macOSに対応。
pub fn total_ram_bytes() -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = s.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemStatus {
            len: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page: u64,
            avail_page: u64,
            total_virt: u64,
            avail_virt: u64,
            avail_ext: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(buf: *mut MemStatus) -> i32;
        }
        let mut m = MemStatus {
            len: std::mem::size_of::<MemStatus>() as u32,
            load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page: 0,
            avail_page: 0,
            total_virt: 0,
            avail_virt: 0,
            avail_ext: 0,
        };
        // SAFETY: 構造体は仕様どおりのレイアウトで、lenを設定済み。
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            Some(m.total_phys)
        } else {
            None
        }
    }
    #[cfg(target_os = "macos")]
    {
        let o = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        String::from_utf8_lossy(&o.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", windows, target_os = "macos")))]
    {
        None
    }
}

/// 目安(日英独露のWeb調査で一致した「重み+KVキャッシュ+余裕15〜40%」に合わせた):
/// 必要量 = ファイルサイズ × 読み込み係数(GPT-2はf32のまま=1、Qwenはbf16→f32で=2) × 1.25 + 1GB、
/// これが搭載メモリの85%以内なら「入る」。メモリ不明なら小さいものだけ。
fn fits(family: &str, approx_size_mb: u32, ram: Option<u64>) -> bool {
    let factor = if family == "qwen" { 2.0 } else { 1.0 };
    let need_mb = approx_size_mb as f64 * factor * 1.25 + 1024.0;
    match ram {
        Some(r) => need_mb * 1024.0 * 1024.0 <= r as f64 * 0.85,
        None => need_mb <= 2200.0,
    }
}

pub fn compatible() -> CompatibleResponse {
    let rec = hardware::recommend();
    let ram = total_ram_bytes();
    let root = model_catalog::models_root();
    let gpt = model_catalog::installed_ids(&root);
    let qwen = model_catalog::installed_qwen_ids(&root);
    let mut models = Vec::new();
    for (family, list, inst) in [("gpt2", model_catalog::CATALOG, &gpt), ("qwen", model_catalog::QWEN_CATALOG, &qwen)] {
        for e in list {
            models.push(FitEntry {
                family,
                id: e.id,
                display_name_ja: e.display_name_ja,
                display_name_en: e.display_name_en,
                approx_size_mb: e.approx_size_mb,
                license_note_ja: e.license_note_ja,
                fits: fits(family, e.approx_size_mb, ram),
                installed: inst.contains(&e.id),
                recommended: e.id == rec.recommended_model_id,
            });
        }
    }
    CompatibleResponse { hardware: rec.hardware, ram_bytes: ram, models, explanation: EXPLANATION.to_vec() }
}

/// 10か国語の説明(AI翻訳、ネイティブ未確認)。
pub const EXPLANATION: &[(&str, &str)] = &[
    ("ja", "この機械のメモリに入るAIモデルの一覧です。「入る」は目安です。グラフィックボードを替えたり入れ直したりすると、自動で合うモデルを入れます(止めるには ARUARU_LLM_AUTO_MODEL=0)。"),
    ("en", "These are the AI models that should fit this machine's memory (a rough guide). If you swap the graphics card or reinstall, a suitable model is installed automatically (set ARUARU_LLM_AUTO_MODEL=0 to stop that)."),
    ("zh-CN", "这是适合本机内存的 AI 模型列表(仅为大致参考)。更换显卡或重新安装后,会自动安装合适的模型(设置 ARUARU_LLM_AUTO_MODEL=0 可关闭)。"),
    ("zh-TW", "這是適合本機記憶體的 AI 模型清單(僅供大致參考)。更換顯示卡或重新安裝後,會自動安裝合適的模型(設定 ARUARU_LLM_AUTO_MODEL=0 可關閉)。"),
    ("ko", "이 기기의 메모리에 들어갈 AI 모델 목록입니다(대략적인 기준). 그래픽 카드를 바꾸거나 다시 설치하면 맞는 모델을 자동으로 설치합니다(끄려면 ARUARU_LLM_AUTO_MODEL=0)."),
    ("de", "Das sind die KI-Modelle, die in den Speicher dieses Geräts passen sollten (grobe Richtlinie). Nach einem Grafikkartenwechsel oder einer Neuinstallation wird automatisch ein passendes Modell installiert (mit ARUARU_LLM_AUTO_MODEL=0 abschaltbar)."),
    ("fr", "Voici les modèles d'IA qui devraient tenir dans la mémoire de cet appareil (à titre indicatif). Si vous changez de carte graphique ou réinstallez, un modèle adapté est installé automatiquement (désactivable avec ARUARU_LLM_AUTO_MODEL=0)."),
    ("es", "Estos son los modelos de IA que deberían caber en la memoria de este equipo (orientativo). Si cambias la tarjeta gráfica o reinstalas, se instala automáticamente un modelo adecuado (se desactiva con ARUARU_LLM_AUTO_MODEL=0)."),
    ("pt-BR", "Estes são os modelos de IA que devem caber na memória deste aparelho (valor aproximado). Se você trocar a placa de vídeo ou reinstalar, um modelo adequado é instalado automaticamente (desative com ARUARU_LLM_AUTO_MODEL=0)."),
    ("ru", "Это список ИИ-моделей, которые должны поместиться в память этого устройства (ориентировочно). После замены видеокарты или переустановки подходящая модель устанавливается автоматически (отключить: ARUARU_LLM_AUTO_MODEL=0)."),
];

fn fingerprint() -> String {
    let h = hardware::detect();
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let ram_gb = total_ram_bytes().map(|b| b / (1024 * 1024 * 1024)).unwrap_or(0);
    format!("gpu={:?};vram={:?};ram_gb={};cores={}", h.gpu_name, h.vram_bytes, ram_gb, cores)
}

/// 起動時に呼ぶ。前回と違うハードウェアなら、お勧めモデルを背景で導入する。
pub fn spawn_hardware_change_check() {
    tokio::spawn(async {
        let root = model_catalog::models_root();
        let path = root.join(".hw-fingerprint");
        let now = tokio::task::spawn_blocking(fingerprint).await.unwrap_or_default();
        let prev = std::fs::read_to_string(&path).ok();
        if prev.as_deref() == Some(now.as_str()) {
            return;
        }
        let _ = std::fs::create_dir_all(&root);
        let _ = std::fs::write(&path, &now);
        let Some(prev) = prev else {
            tracing::info!("hardware fingerprint recorded (first run, nothing downloaded): {now}");
            return;
        };
        if std::env::var("ARUARU_LLM_AUTO_MODEL").map(|v| v == "0").unwrap_or(false) {
            tracing::info!("hardware changed ({prev} -> {now}); auto model install disabled");
            return;
        }
        let rec = hardware::recommend();
        match model_catalog::find(rec.recommended_model_id) {
            Some(entry) => {
                tracing::info!("hardware changed ({prev} -> {now}); installing recommended model {}", entry.id);
                if let Err(e) = model_catalog::install(entry, &root.join(entry.id)).await {
                    tracing::warn!("auto model install failed: {e:#}");
                } else {
                    tracing::info!("auto model install done: {} (select it with ModelsSelect to use it)", entry.id);
                }
            }
            None => tracing::warn!("recommended id {} not in catalog", rec.recommended_model_id),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_rule() {
        let gb = 1024u64 * 1024 * 1024;
        assert!(fits("qwen", 942, Some(4 * gb)));
        assert!(!fits("qwen", 6200, Some(8 * gb)));
        assert!(fits("qwen", 6200, Some(32 * gb)));
        assert!(fits("gpt2", 500, None) && !fits("qwen", 2900, None));
    }

    #[test]
    fn explains_in_ten_languages() {
        assert_eq!(EXPLANATION.len(), 10);
    }
}
