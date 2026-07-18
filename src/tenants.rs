//! 「分身の術」構想: `aruaru-llm`は1インスタンスを複数ドメイン
//! (e-gov.info・aruaru-tokyo・karu.tokyo等)が共有し、ドメインを追加する
//! たびに新しい`aruaru-llm`プロセスを個別インストール・起動する必要が
//! 無いようにする(`open-web-server`/`open-easy-web`の
//! `appserver_registration.rs`と同じ設計思想、ユーザー指示2026-07-18)。
//!
//! 動的テナント登録用の管理API(`POST /admin/tenants`)を持ち、
//! `x-admin-token`ヘッダで簡易認証する(未設定の場合は誰でも登録可能
//! ——本番運用では`E_GOV_LLM_ADMIN_TOKEN`等の環境変数で必ず設定すること)。
//! 登録は`RwLock<HashMap>`によるプロセス内共有状態で、再起動なしに
//! 実行時追加・削除ができる(`open-web-server`の`TenantRegistry`と同じ
//! パターン)。
//!
//! `/v1/chat`は`tenant`が未登録でも応答は返す(このサービスの主目的は
//! 応答生成であり、テナント登録は利用状況の可視化・将来の課金/
//! レート制限のための構造という位置づけ。登録必須にして可用性を
//! 落とすことは避ける)。

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantInfo {
    /// 登録したドメイン(例: "e-gov.info")。
    pub host: String,
    /// このテナントが担当するサイトの表示名(任意)。
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Default)]
pub struct TenantRegistry {
    tenants: RwLock<HashMap<String, TenantInfo>>,
}

impl TenantRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, info: TenantInfo) {
        self.tenants.write().unwrap().insert(info.host.clone(), info);
    }

    pub fn remove(&self, host: &str) -> bool {
        self.tenants.write().unwrap().remove(host).is_some()
    }

    pub fn list(&self) -> Vec<TenantInfo> {
        self.tenants.read().unwrap().values().cloned().collect()
    }

    pub fn contains(&self, host: &str) -> bool {
        self.tenants.read().unwrap().contains_key(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_then_list_returns_the_tenant() {
        let registry = TenantRegistry::new();
        registry.register(TenantInfo { host: "e-gov.info".to_string(), label: Some("e-gov".to_string()) });
        let all = registry.list();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].host, "e-gov.info");
    }

    #[test]
    fn contains_reflects_registration_state() {
        let registry = TenantRegistry::new();
        assert!(!registry.contains("aruaru.tokyo"));
        registry.register(TenantInfo { host: "aruaru.tokyo".to_string(), label: None });
        assert!(registry.contains("aruaru.tokyo"));
    }

    #[test]
    fn remove_returns_false_for_unknown_host() {
        let registry = TenantRegistry::new();
        assert!(!registry.remove("nope.example.com"));
    }

    #[test]
    fn re_registering_same_host_overwrites_not_duplicates() {
        let registry = TenantRegistry::new();
        registry.register(TenantInfo { host: "karu.tokyo".to_string(), label: Some("old".to_string()) });
        registry.register(TenantInfo { host: "karu.tokyo".to_string(), label: Some("new".to_string()) });
        let all = registry.list();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].label.as_deref(), Some("new"));
    }
}
