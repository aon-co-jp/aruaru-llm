//! 共有基盤としての受付制御(2026-11-09、配備方針「共有基盤は1回だけ導入」)。
//!
//! 重い推論系の Connect 呼び出しに、(1)同時実行数の上限、(2)サイト(テナント)別の待ち行列と
//! ラウンドロビン配分(1サイトが独占しない)、(3)待ち行列の長さの上限(過負荷は即 429)、
//! (4)待ち時間の上限と実行時間の上限(超過は 503 / 504)を与える。
//! 実行時間超過の後も、バックグラウンドで走り続ける計算が終わるまで枠は返さない(枠の数え方を正直に保つ)。
//!
//! サイト識別は `x-tenant` ヘッダ、無ければ `x-forwarded-host`、無ければ `anonymous`。
//! 信頼境界: 前面の open-web-server が付ける前提。直接公開すると名前を偽装して公平配分を回避できる。

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use tokio::sync::oneshot;

pub struct Config {
    pub max_running: usize,
    pub max_queue_total: usize,
    pub max_queue_per_tenant: usize,
    pub queue_wait: Duration,
    pub exec_timeout: Duration,
}

impl Config {
    pub fn from_env() -> Self {
        fn n(k: &str, d: usize) -> usize {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).filter(|v| *v > 0).unwrap_or(d)
        }
        Config {
            max_running: n("ARUARU_LLM_MAX_CONCURRENT", 2),
            max_queue_total: n("ARUARU_LLM_MAX_QUEUE", 32),
            max_queue_per_tenant: n("ARUARU_LLM_MAX_QUEUE_PER_TENANT", 8),
            queue_wait: Duration::from_secs(n("ARUARU_LLM_QUEUE_WAIT_SECS", 30) as u64),
            exec_timeout: Duration::from_secs(n("ARUARU_LLM_EXEC_TIMEOUT_SECS", 120) as u64),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Reject {
    /// 待ち行列が満杯(サイト別または全体)。
    Overloaded,
    /// 待ち時間の上限を超えた。
    QueueTimeout,
}

struct Entry {
    id: u64,
    tx: oneshot::Sender<Permit>,
}

#[derive(Default)]
struct State {
    running: usize,
    waiting: usize,
    next_id: u64,
    queues: HashMap<String, VecDeque<Entry>>,
    order: VecDeque<String>,
}

pub struct Governor {
    cfg: Config,
    st: Mutex<State>,
}

pub struct Permit {
    gov: Arc<Governor>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        let granted = {
            let mut st = self.gov.st.lock().unwrap();
            st.running -= 1;
            self.gov.pick(&mut st)
        };
        for (tx, p) in granted {
            // 受け手が既に居なければ Permit が返り、Drop で枠が戻る。
            let _ = tx.send(p);
        }
    }
}

impl Governor {
    pub fn new(cfg: Config) -> Arc<Self> {
        Arc::new(Governor { cfg, st: Mutex::new(State::default()) })
    }

    /// 空き枠がある限り、サイトのラウンドロビンで待ち行列から取り出す(ロック内)。
    fn pick(self: &Arc<Self>, st: &mut State) -> Vec<(oneshot::Sender<Permit>, Permit)> {
        let mut out = Vec::new();
        while st.running < self.cfg.max_running {
            let Some(t) = st.order.pop_front() else { break };
            let Some(q) = st.queues.get_mut(&t) else { continue };
            let Some(e) = q.pop_front() else {
                st.queues.remove(&t);
                continue;
            };
            st.waiting -= 1;
            if q.is_empty() {
                st.queues.remove(&t);
            } else {
                st.order.push_back(t);
            }
            st.running += 1;
            out.push((e.tx, Permit { gov: Arc::clone(self) }));
        }
        out
    }

    pub async fn acquire(self: &Arc<Self>, tenant: &str) -> Result<Permit, Reject> {
        let (rx, id) = {
            let mut st = self.st.lock().unwrap();
            if st.running < self.cfg.max_running && st.waiting == 0 {
                st.running += 1;
                return Ok(Permit { gov: Arc::clone(self) });
            }
            let tq = st.queues.get(tenant).map_or(0, |q| q.len());
            if st.waiting >= self.cfg.max_queue_total || tq >= self.cfg.max_queue_per_tenant {
                return Err(Reject::Overloaded);
            }
            let (tx, rx) = oneshot::channel();
            let id = st.next_id;
            st.next_id += 1;
            if !st.queues.contains_key(tenant) {
                st.order.push_back(tenant.to_string());
            }
            st.queues.entry(tenant.to_string()).or_default().push_back(Entry { id, tx });
            st.waiting += 1;
            (rx, id)
        };
        match tokio::time::timeout(self.cfg.queue_wait, rx).await {
            Ok(Ok(p)) => Ok(p),
            _ => {
                // タイムアウト: まだ並んでいれば取り除く。既に付与済みなら受け取れなかっただけ(Drop で返る)。
                let mut st = self.st.lock().unwrap();
                if let Some(q) = st.queues.get_mut(tenant) {
                    if let Some(pos) = q.iter().position(|e| e.id == id) {
                        q.remove(pos);
                        st.waiting -= 1;
                        if st.queues.get(tenant).map_or(false, |q| q.is_empty()) {
                            st.queues.remove(tenant);
                            st.order.retain(|t| t != tenant);
                        }
                    }
                }
                Err(Reject::QueueTimeout)
            }
        }
    }

    pub fn exec_timeout(&self) -> Duration {
        self.cfg.exec_timeout
    }

    pub fn snapshot(&self) -> (usize, usize) {
        let st = self.st.lock().unwrap();
        (st.running, st.waiting)
    }
}

static GOV: OnceLock<Arc<Governor>> = OnceLock::new();

pub fn global() -> &'static Arc<Governor> {
    GOV.get_or_init(|| Governor::new(Config::from_env()))
}

/// Connect のメソッドのうち、重い計算を伴うもの(受付制御の対象)。
pub fn is_heavy(path: &str) -> bool {
    let Some(m) = path.strip_prefix(crate::connect::PREFIX) else { return false };
    matches!(
        m,
        "Generate" | "GenerateSpeculative" | "GenerateWithSearch" | "GenerateQwen" | "GenerateDeepseek"
            | "Translate" | "Transcribe" | "Ocr" | "OcrCorrect" | "OcrGlyphRank" | "OcrScan" | "OcrPropose" | "OcrFill"
    )
}

/// ヘッダからサイトを識別する(長さ・文字を制限)。
pub fn tenant_of(headers: &hyper::HeaderMap) -> String {
    let raw = ["x-tenant", "x-forwarded-host"]
        .iter()
        .find_map(|k| headers.get(*k).and_then(|v| v.to_str().ok()))
        .unwrap_or("anonymous");
    let s: String = raw.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':' | '_')).take(64).collect();
    if s.is_empty() { "anonymous".to_string() } else { s.to_ascii_lowercase() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gov(run: usize, q: usize, qt: usize, wait_ms: u64) -> Arc<Governor> {
        Governor::new(Config {
            max_running: run,
            max_queue_total: q,
            max_queue_per_tenant: qt,
            queue_wait: Duration::from_millis(wait_ms),
            exec_timeout: Duration::from_secs(5),
        })
    }

    #[tokio::test]
    async fn limits_running_and_rejects_when_queue_full() {
        let g = gov(1, 2, 1, 2000);
        let p = g.acquire("a").await.unwrap();
        let g2 = Arc::clone(&g);
        let w = tokio::spawn(async move { g2.acquire("a").await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(g.acquire("a").await.err(), Some(Reject::Overloaded)); // per-tenant 1
        drop(p);
        assert!(w.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn queue_wait_times_out_and_cleans_up() {
        let g = gov(1, 4, 4, 80);
        let _p = g.acquire("a").await.unwrap();
        assert_eq!(g.acquire("b").await.err(), Some(Reject::QueueTimeout));
        assert_eq!(g.snapshot(), (1, 0));
    }

    #[tokio::test]
    async fn round_robin_between_tenants() {
        let g = gov(1, 20, 10, 5000);
        let p = g.acquire("x").await.unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut hs = Vec::new();
        for (t, n) in [("a", 3), ("b", 1)] {
            for _ in 0..n {
                let (g, o) = (Arc::clone(&g), Arc::clone(&order));
                hs.push(tokio::spawn(async move {
                    let p = g.acquire(t).await.unwrap();
                    o.lock().unwrap().push(t);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    drop(p);
                }));
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        drop(p);
        for h in hs { h.await.unwrap(); }
        // a は先に3件並んだが、b の1件は a の2件目より前に処理される。
        let o = order.lock().unwrap().clone();
        assert_eq!(o[0], "a");
        assert_eq!(o[1], "b", "order={o:?}");
    }

    /// 負荷の実測: A が20件、B・C が各3件を同時に投げ、各サイトの完了時刻(全体開始から)を表示する。
    /// 1件100ms・同時2。FIFO なら B・C は最後(約1.0秒)になるが、サイト別配分なら早く終わる。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn load_fairness_measurement() {
        let g = gov(2, 100, 100, 30_000);
        let t0 = std::time::Instant::now();
        let mut hs = Vec::new();
        for (t, n) in [("site-a", 20), ("site-b", 3), ("site-c", 3)] {
            for _ in 0..n {
                let g = Arc::clone(&g);
                hs.push(tokio::spawn(async move {
                    let p = g.acquire(t).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    drop(p);
                    (t, t0.elapsed().as_millis())
                }));
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }
        let mut last: HashMap<&str, u128> = HashMap::new();
        for h in hs {
            let (t, ms) = h.await.unwrap();
            let e = last.entry(t).or_insert(0);
            *e = (*e).max(ms);
        }
        println!("LOADTEST last-finish ms: a={} b={} c={}", last["site-a"], last["site-b"], last["site-c"]);
        assert!(last["site-b"] * 10 < last["site-a"] * 7 && last["site-c"] * 10 < last["site-a"] * 7);
        assert_eq!(g.snapshot(), (0, 0));
    }
}
