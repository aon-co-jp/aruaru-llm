//! CPU+GPUを同時並列に稼働させるためのラウンドロビンデバイスプール。
//!
//! これまでの`device: Arc<dyn GpuDevice>`は起動時に選んだ単一デバイス
//! (CPUまたはGPUのどちらか一方、Vulkan構築失敗時のみCPUへフォールバック)
//! を全リクエストが共有する設計だった。ユーザー指示「CPU+システム
//! メモリ+GPUを非同期ででも、同時に並列並行で動作させて」を受け、
//! 複数の同時並行リクエストがCPU/GPU双方へ分散し、両方が実際に
//! 並行稼働する設計へ変更する。
//!
//! **正直な設計上の限界**: 単一リクエストの計算そのもの(1回の
//! forward pass)をCPUとGPUに分割するテンソル並列化(モデル並列)は
//! 行っていない——1リクエストは開始時に選ばれた1つのデバイスの上で
//! 最初から最後まで実行される(`alloc`/`memcpy`/`launch_kernel`の
//! 一貫性を保つため、`DevicePtr`をデバイスをまたいで使い回すのは
//! 安全ではない)。ここで実現するのは「複数の同時リクエストを
//! CPU担当分とGPU担当分に振り分け、両方のハードウェアを同時に
//! 稼働させる」というリクエストレベルの並列化。

use opencuda_core::GpuDevice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct DevicePool {
    devices: Vec<Arc<dyn GpuDevice>>,
    next: AtomicUsize,
}

impl DevicePool {
    pub fn new(devices: Vec<Arc<dyn GpuDevice>>) -> Arc<Self> {
        assert!(!devices.is_empty(), "DevicePoolには最低1台のデバイスが必要");
        Arc::new(Self {
            devices,
            next: AtomicUsize::new(0),
        })
    }

    /// ラウンドロビンで次のデバイスを返す。デバイスが1台のみ(GPU不在等)
    /// の場合は常に同じデバイスを返す(既存の単一デバイス動作と同じ)。
    pub fn next_device(&self) -> Arc<dyn GpuDevice> {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.devices.len();
        Arc::clone(&self.devices[i])
    }

    pub fn device_names(&self) -> Vec<String> {
        self.devices.iter().map(|d| d.info().name.clone()).collect()
    }

    /// プールが保持している全デバイスへの参照(`GET /v1/runtime`が
    /// `info()`を読み出して「実際にどのバックエンドで動いているか」を
    /// 正直に報告するために使う、2026-08-22追加)。
    pub fn devices(&self) -> &[Arc<dyn GpuDevice>] {
        &self.devices
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;

    #[test]
    fn single_device_pool_always_returns_same_device() {
        let cpu: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let pool = DevicePool::new(vec![cpu]);
        let a = pool.next_device();
        let b = pool.next_device();
        assert_eq!(a.info().name, b.info().name);
        assert_eq!(pool.device_count(), 1);
    }

    #[test]
    fn two_device_pool_alternates_round_robin() {
        let cpu0: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let cpu1: Arc<dyn GpuDevice> = CpuDevice::new(1);
        let pool = DevicePool::new(vec![cpu0, cpu1]);
        let picks: Vec<usize> = (0..4).map(|_| pool.next_device().info().id).collect();
        assert_eq!(picks, vec![0, 1, 0, 1]);
    }
}
