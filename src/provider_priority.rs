//! Google検索・ChatGPT/DeepSeek/Gemini/Claudeを横断した「無料枠を優先で
//! 使い切り、順番に切り替える」機能(ユーザー指示: 「Google、ChatGPT/
//! DeepSeek/Gemini/Claudeは、無料枠を優先で使い切り順番に使用、に
//! チェックを付けられる様にして。Googleなどは、順番を入力したり、数字の
//! ラジオボタンを押すかのどちらかで優先の順番を変更可能にして」への対応)。
//!
//! ## 設計方針(正直な開示)
//!
//! - `web_search.rs`(Google Custom Search、検索専用)と`chat_providers.rs`
//!   (ChatGPT/DeepSeek/Gemini/Claude、チャット補完専用)は元々別物の
//!   API(用途が異なる)だが、ユーザーが「5つ横並びで優先順位を付けたい」
//!   と明示的に指示したため、この共通モジュールで**5つを同列に扱う
//!   優先順位リスト**のみを一元管理する。実際にその順序へ従って
//!   「無料枠を使い切ったら次へ」を行う処理(HTTPステータス429/quota
//!   系エラーの検知)は、各サービスの性質が異なるため個別
//!   (`web_search::search_in_priority_order`・
//!   `chat_providers::complete_in_priority_order`)に実装する——この
//!   モジュール自体は「順序」と「有効/無効フラグ」という状態のみを持つ。
//! - 「無料枠を使い切った」ことをAPI応答から機械的に確実に検知する方法は
//!   各社共通ではない(HTTP 429を返す場合もあれば、200 OKのままエラー
//!   本文で通知する場合もある)。このモジュールが提供するのは
//!   「あるプロバイダの呼び出しが失敗したら、優先順位の次のプロバイダへ
//!   自動的にフォールバックする」という汎用的な仕組みであり、失敗理由が
//!   本当に「無料枠を使い切ったから」なのか「他の一時的なエラーか」を
//!   判別する精度は保証しない(誇張しない)。
//! - `RUNTIME_CREDENTIALS`/`RUNTIME_KEYS`と同じくプロセスメモリ上にのみ
//!   保持し、ディスク永続化はしない(プロセス再起動で既定順序へ戻る)。

use std::sync::RwLock;

use serde::{Deserialize, Serialize};

/// 優先順位付けの対象となるサービス(Google検索+チャット系4社の計5つ)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PriorityService {
    GoogleSearch,
    Openai,
    Deepseek,
    Gemini,
    Claude,
}

impl PriorityService {
    fn default_order() -> Vec<PriorityService> {
        vec![PriorityService::GoogleSearch, PriorityService::Openai, PriorityService::Deepseek, PriorityService::Gemini, PriorityService::Claude]
    }
}

struct PriorityState {
    /// `true`の場合、呼び出し側は「設定された順序で、無料枠(または
    /// 単に呼び出し)に失敗したら次のサービスへ自動的に切り替える」
    /// 挙動を行う。`false`の場合は呼び出し側が指定した単一サービスの
    /// みを使う(既存の挙動、後方互換)。
    enabled: bool,
    order: Vec<PriorityService>,
}

static STATE: RwLock<Option<PriorityState>> = RwLock::new(None);

fn with_state<T>(f: impl FnOnce(&PriorityState) -> T) -> T {
    let guard = STATE.read().expect("provider priority state lock poisoned");
    match guard.as_ref() {
        Some(state) => f(state),
        None => f(&PriorityState { enabled: false, order: PriorityService::default_order() }),
    }
}

/// `POST /v1/settings/provider-priority`から呼ばれる、有効/無効+順序の
/// 一括設定。`order`に含まれないサービスは既定順序の末尾(重複除去済み)
/// へ自動的に補完する(利用者が一部だけ並べ替えても他のサービスが
/// 抜け落ちないようにするため)。
pub fn set_priority(enabled: bool, order: Vec<PriorityService>) {
    let mut normalized = Vec::new();
    for svc in order {
        if !normalized.contains(&svc) {
            normalized.push(svc);
        }
    }
    for svc in PriorityService::default_order() {
        if !normalized.contains(&svc) {
            normalized.push(svc);
        }
    }
    let mut guard = STATE.write().expect("provider priority state lock poisoned");
    *guard = Some(PriorityState { enabled, order: normalized });
}

/// 既定順序へリセットする(`DELETE /v1/settings/provider-priority`)。
pub fn reset_priority() {
    let mut guard = STATE.write().expect("provider priority state lock poisoned");
    *guard = None;
}

pub fn is_enabled() -> bool {
    with_state(|s| s.enabled)
}

pub fn current_order() -> Vec<PriorityService> {
    with_state(|s| s.order.clone())
}

#[derive(Debug, Serialize)]
pub struct PriorityStatus {
    pub enabled: bool,
    pub order: Vec<PriorityService>,
}

pub fn status() -> PriorityStatus {
    with_state(|s| PriorityStatus { enabled: s.enabled, order: s.order.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_disabled_with_default_order() {
        reset_priority();
        assert!(!is_enabled());
        assert_eq!(current_order(), PriorityService::default_order());
    }

    #[test]
    fn set_priority_normalizes_and_fills_missing_services() {
        set_priority(true, vec![PriorityService::Claude, PriorityService::GoogleSearch, PriorityService::Claude]);
        assert!(is_enabled());
        let order = current_order();
        assert_eq!(order[0], PriorityService::Claude);
        assert_eq!(order[1], PriorityService::GoogleSearch);
        assert_eq!(order.len(), 5);
        reset_priority();
    }
}
