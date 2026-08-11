//! ネットワークトラフィックの意味的類似度分類(RS-SmartTCPの「AI侵入
//! 検知」プラグイン用、2026-08-11新設)。
//!
//! `security.rs`(RS-Guardの「AI二次判定」)と全く同じ`open-cuda-bert`
//! (multilingual-e5-small)埋め込み+コサイン類似度の仕組みを流用し、
//! RS-SmartTCP側が観測したトラフィック特徴量を短い自然文の説明へ変換した
//! ものを、「ポートスキャン/SYNフラッド/ブルートフォース/データ持ち出し/
//! 正常」の代表例と比較して、最も近いカテゴリと類似度スコアを返す。
//!
//! **正直な開示(最重要)**: これは**攻撃トラフィックの実データで訓練した
//! 専用の分類器ではなく、汎用の文埋め込みモデルによる意味的類似度の
//! ヒューリスティック**——`security.rs`と全く同じ限界を持つ。実際の
//! パケットバイト列を解析しているわけでもない(RS-SmartTCP側が既に
//! 数値特徴量から短い説明文を組み立てた上でここへ渡す設計)。
//! GPU/NPU実行については、このプロセスが使う`opencuda-bert`自体が
//! 既存の`hw-detect-vulkan`/`hw-detect-directx`+`real-vulkan` feature
//! (`src/hardware.rs`・`main.rs`のデバイス選択箇所参照)経由で
//! Vulkan/DirectXへディスパッチ可能な設計を共有しているため、新たな
//! GPU配線をこのモジュール専用に実装する必要はない——`scoring`/
//! `security`と同じデバイス(`Arc<dyn GpuDevice>`)をそのまま受け取る。

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use open_cuda_bert::cosine_similarity;
use opencuda_core::GpuDevice;

use crate::scoring::{embed, normalize};

pub struct TrafficCategory {
    pub name: &'static str,
    pub description: &'static str,
    examples: &'static [&'static str],
    pub suspicious: bool,
}

pub const CATEGORIES: &[TrafficCategory] = &[
    TrafficCategory {
        name: "port-scan",
        description: "Port scan: many distinct destination ports probed in a short time from one source.",
        examples: &[
            "one source IP connected to 40 different destination ports within 5 seconds",
            "sequential TCP SYN packets to ports 1 through 1024 with no completed handshake",
            "a single host is probing many ports on the target in rapid succession",
            "1つの送信元から短時間に多数の異なるポートへ接続を試みている",
            "5秒間に40個の異なるポートへ順番にSYNパケットを送っている",
        ],
        suspicious: true,
    },
    TrafficCategory {
        name: "syn-flood",
        description: "SYN flood: a very high rate of SYN packets to one port without completing the handshake, consistent with a denial-of-service attempt.",
        examples: &[
            "thousands of SYN packets per second to the same port with no corresponding ACK",
            "a flood of half-open TCP connections targeting one service",
            "extremely high packet rate of connection attempts overwhelming a single port",
            "同一ポートへ大量のSYNパケットが送られ続けハンドシェイクが完了しない",
            "1秒間に数千件の接続要求が同じサービスに集中している",
        ],
        suspicious: true,
    },
    TrafficCategory {
        name: "brute-force-login",
        description: "Brute-force login attempt: repeated authentication failures against the same login service in quick succession.",
        examples: &[
            "dozens of failed SSH login attempts against the same account within a minute",
            "repeated password authentication failures against the same login endpoint",
            "an automated script trying many passwords against one username in rapid succession",
            "同じアカウントに対して短時間に何度もログイン失敗が繰り返されている",
            "1分間に数十回のSSHログイン失敗が同じホストから記録されている",
        ],
        suspicious: true,
    },
    TrafficCategory {
        name: "data-exfiltration",
        description: "Data exfiltration: an unusually large, sustained outbound data transfer to an unfamiliar external destination.",
        examples: &[
            "a sustained large outbound transfer of several gigabytes to an unfamiliar external IP",
            "an internal host is uploading an unusually large amount of data to the internet at night",
            "outbound traffic volume to one destination is far above this host's normal baseline",
            "普段通信しない外部の宛先へ、通常より著しく大きなデータ量を送信し続けている",
            "深夜に社内端末から大量のデータが外部へアップロードされている",
        ],
        suspicious: true,
    },
    TrafficCategory {
        name: "normal",
        description: "Ordinary, unremarkable network traffic consistent with everyday usage.",
        examples: &[
            "a normal web browsing session with a handful of HTTPS connections to common sites",
            "steady low-volume traffic consistent with video streaming from a known service",
            "typical office traffic: email, a few web pages, and a cloud file sync",
            "普段通りのWeb閲覧やメール送受信で通信量・接続パターンに異常はない",
            "既知のサービスへの安定した通信で、量・頻度ともに普段と変わらない",
        ],
        suspicious: false,
    },
];

/// `security.rs`と同じ根拠(汎用埋め込みは無関係な文同士でも0.8前後の
/// 底上げ類似度が出る)に基づく保守的な閾値。
const SUSPICIOUS_THRESHOLD: f32 = 0.83;

static CATEGORY_EMBEDDINGS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();

fn category_embeddings(device: &Arc<dyn GpuDevice>) -> Result<&'static Vec<Vec<f32>>> {
    if let Some(e) = CATEGORY_EMBEDDINGS.get() {
        return Ok(e);
    }
    let mut embeddings = Vec::with_capacity(CATEGORIES.len());
    for category in CATEGORIES {
        let mut acc: Vec<f32> = Vec::new();
        for example in category.examples {
            let v = embed(device, example, false)?;
            if acc.is_empty() {
                acc = vec![0.0; v.len()];
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        normalize(&mut acc);
        embeddings.push(acc);
    }
    let _ = CATEGORY_EMBEDDINGS.set(embeddings);
    Ok(CATEGORY_EMBEDDINGS.get().expect("CATEGORY_EMBEDDINGS was just set"))
}

pub struct TrafficVerdict {
    pub label: &'static str,
    pub description: &'static str,
    pub score: f32,
    pub is_suspicious: bool,
}

/// 渡されたトラフィック特徴量の説明文を、最も近いカテゴリへ分類する。
pub fn classify_traffic(device: &Arc<dyn GpuDevice>, description: &str) -> Result<TrafficVerdict> {
    let cat_embeddings = category_embeddings(device)?;
    let query = embed(device, description, true)?;

    let mut best: Option<(usize, f32)> = None;
    for (i, ce) in cat_embeddings.iter().enumerate() {
        let sim = cosine_similarity(&query, ce);
        if best.map(|(_, bs)| sim > bs).unwrap_or(true) {
            best = Some((i, sim));
        }
    }

    let (idx, score) = best.expect("CATEGORIES is non-empty");
    let category = &CATEGORIES[idx];
    let is_suspicious = category.suspicious && score >= SUSPICIOUS_THRESHOLD;

    Ok(TrafficVerdict { label: category.name, description: category.description, score, is_suspicious })
}

/// 起動時ウォームアップ(モデルロード+カテゴリ代表ベクトル計算の前倒し)。
pub fn warmup(device: &Arc<dyn GpuDevice>) -> Result<()> {
    let _ = classify_traffic(device, "warmup")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;

    fn cpu_device() -> Arc<dyn GpuDevice> {
        CpuDevice::new(0)
    }

    #[test]
    fn classifies_port_scan_description_as_suspicious() {
        let device = cpu_device();
        let v = classify_traffic(&device, "one source IP probed 60 different destination ports within 3 seconds, no handshake completed").unwrap();
        assert!(v.is_suspicious, "expected suspicious, got label={} score={}", v.label, v.score);
    }

    #[test]
    fn classifies_syn_flood_description_as_suspicious() {
        let device = cpu_device();
        let v = classify_traffic(&device, "thousands of SYN packets per second hit the same port with no completed handshake").unwrap();
        assert!(v.is_suspicious, "expected suspicious, got label={} score={}", v.label, v.score);
    }

    #[test]
    fn classifies_ordinary_browsing_as_not_suspicious() {
        let device = cpu_device();
        let v = classify_traffic(&device, "a normal web browsing session with a handful of HTTPS connections to common sites").unwrap();
        assert!(!v.is_suspicious, "expected benign, got label={} score={}", v.label, v.score);
    }
}
