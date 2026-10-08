//! モデル非依存の「賢さ」ストア(ユーザー指示 2026-10-05「aruaru-llmの賢くなる部分は、
//! 新しいLLMに入れ替えても消えずに引き継がれて残る様にして」)。
//!
//! ## 設計(正直な開示つき)
//! - GPT-2/Qwenの**重みは学習で書き換わらない**(ファインチューニング無し)。
//!   「賢くなる」実体は、ここに溜める**平文の知識**(接客方針・国別の話題・訂正済み回答など)と、
//!   `data/`配下のニュース/地理DBであり、どのモデルにも依存しない。
//! - 保存はプレーンなJSON(`ARUARU_LLM_KNOWLEDGE_DIR`、既定`data/knowledge`)。
//!   モデルのディレクトリ(`models/`)とは完全に別で、モデルの切替・追加・入替は
//!   このディレクトリへ一切触れない。埋め込みベクトル等のモデル固有物は**保存しない**
//!   (検索は単語の重なりで行う)ので、新しいLLMでもそのまま使える。
//! - モデル切替の直前に`snapshot()`で世代バックアップを残す(直近`KEEP_SNAPSHOTS`世代)。
//! - 生成時は`context_for(query)`の結果をプロンプトへ足す形で使う(守る保証は無い)。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const KEEP_SNAPSHOTS: usize = 10;
const MAX_TEXT_CHARS: usize = 2000;
static LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// 出典(URLや「ユーザー訂正」等)。不明なら空。
    #[serde(default)]
    pub source: String,
    pub created_unix: u64,
}

#[derive(Debug, Deserialize)]
pub struct AddRequest {
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: String,
}

pub fn dir() -> PathBuf {
    std::env::var("ARUARU_LLM_KNOWLEDGE_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("data/knowledge"))
}

fn file() -> PathBuf {
    dir().join("knowledge.json")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn load_unlocked() -> Vec<Entry> {
    std::fs::read_to_string(file()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

fn save_unlocked(entries: &[Entry]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir())?;
    let tmp = dir().join("knowledge.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(entries).map_err(std::io::Error::other)?)?;
    std::fs::rename(tmp, file()) // 書き込み途中の破損を避けるため原子的に置換
}

pub fn list() -> Vec<Entry> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    load_unlocked()
}

pub fn add(req: AddRequest) -> Result<Entry, String> {
    let text = req.text.trim().to_string();
    if text.is_empty() || text.chars().count() > MAX_TEXT_CHARS {
        return Err(format!("text must be 1-{MAX_TEXT_CHARS} characters"));
    }
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut all = load_unlocked();
    if let Some(dup) = all.iter().find(|e| e.text == text) {
        return Ok(dup.clone()); // 同じ内容は重複登録しない
    }
    let entry = Entry {
        id: all.iter().map(|e| e.id).max().unwrap_or(0) + 1,
        text,
        tags: req.tags,
        source: req.source,
        created_unix: now(),
    };
    all.push(entry.clone());
    save_unlocked(&all).map_err(|e| format!("failed to save knowledge: {e}"))?;
    Ok(entry)
}

/// 日本語・中国語・韓国語の文字か(ひらがな・カタカナ・漢字・ハングル・半角カナ)。
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xAC00..=0xD7AF | 0xFF66..=0xFF9F)
}

/// 検索用の語に分ける。英数字は空白・記号区切りの2文字以上の語、日本語などは空白で単語が
/// 区切られないため、連続するCJK文字を**2文字ずつ重ねた断片(バイグラム)**にして比べる
/// (2026-10-08: 日本語の知識がほぼ拾われなかった問題の対策。形態素解析器などの依存は増やさない)。
fn words(s: &str) -> HashSet<String> {
    let lower = s.to_lowercase();
    let mut out = HashSet::new();
    for token in lower.split(|c: char| !c.is_alphanumeric()) {
        let mut run: Vec<char> = Vec::new();
        let mut flush = |run: &mut Vec<char>, cjk: bool, out: &mut HashSet<String>| {
            if run.is_empty() {
                return;
            }
            if cjk {
                if run.len() == 1 {
                    return; // 1文字だけの助詞などは雑音になるので使わない
                }
                for w in run.windows(2) {
                    out.insert(w.iter().collect());
                }
            } else if run.len() >= 2 {
                out.insert(run.iter().collect());
            }
            run.clear();
        };
        let mut cur_cjk = false;
        for c in token.chars() {
            let c_cjk = is_cjk(c);
            if !run.is_empty() && c_cjk != cur_cjk {
                flush(&mut run, cur_cjk, &mut out);
            }
            cur_cjk = c_cjk;
            run.push(c);
        }
        flush(&mut run, cur_cjk, &mut out);
    }
    out
}

/// 同じ`text`が無いものだけ追加して、追加件数を返す(IDは各インスタンスで振り直す)。
pub fn merge(incoming: Vec<Entry>) -> usize {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut all = load_unlocked();
    let mut added = 0;
    for e in incoming {
        let text = e.text.trim().to_string();
        if text.is_empty() || text.chars().count() > MAX_TEXT_CHARS || all.iter().any(|x| x.text == text) {
            continue;
        }
        let id = all.iter().map(|x| x.id).max().unwrap_or(0) + 1;
        all.push(Entry { id, text, tags: e.tags, source: e.source, created_unix: e.created_unix });
        added += 1;
    }
    if added > 0 {
        let _ = save_unlocked(&all);
    }
    added
}

/// 同梱の種知識(接客技法の言い換え・先生キャラ方針)を取り込む。何度呼んでも重複しない。
pub fn seed() -> usize {
    serde_json::from_str::<Vec<Entry>>(include_str!("../data/knowledge_seed.json")).map(merge).unwrap_or(0)
}

/// 既定の取得元(公開WEB→GitHubの順)。`ARUARU_LLM_KNOWLEDGE_SYNC_URL`で1つに上書き、`off`で無効。
fn sync_urls() -> Vec<String> {
    match std::env::var("ARUARU_LLM_KNOWLEDGE_SYNC_URL") {
        Ok(v) if v.eq_ignore_ascii_case("off") => Vec::new(),
        Ok(v) if !v.trim().is_empty() => v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        _ => vec![
            "https://easy-web.tokyo/open-english/v1/public/knowledge/export".to_string(),
            "https://raw.githubusercontent.com/aon-co-jp/open-english/master/data/knowledge/knowledge.json".to_string(),
        ],
    }
}

/// 公開WEB(失敗時はGitHub)から知識を取得して手元へ統合する。アプリを入れ直した直後でも、
/// 次回起動時にここで知識が自動で戻る。取得できなくても手元の知識はそのまま使える。
pub async fn sync_from_remote() -> usize {
    let Ok(client) = reqwest::Client::builder().timeout(std::time::Duration::from_secs(20)).build() else {
        return 0;
    };
    for url in sync_urls() {
        let Ok(resp) = client.get(&url).send().await else { continue };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(body) = resp.text().await else { continue };
        // 形式は`{"items":[...]}`(APIの返り値)か、配列そのもの(GitHub上のファイル)のどちらでもよい
        let entries: Option<Vec<Entry>> = serde_json::from_str::<Vec<Entry>>(&body).ok().or_else(|| {
            serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("items").cloned())
                .and_then(|v| serde_json::from_value(v).ok())
        });
        if let Some(entries) = entries {
            let n = merge(entries);
            tracing::info!("knowledge sync from {url}: {n} new entr(ies) merged");
            return n;
        }
    }
    0
}

/// 起動時に種知識を入れ、以後6時間ごとに公開WEB/GitHubから最新の知識を取り込む。
pub fn spawn_background_sync() {
    seed();
    tokio::spawn(async {
        loop {
            sync_from_remote().await;
            tokio::time::sleep(std::time::Duration::from_secs(6 * 3600)).await;
        }
    });
}

/// 単語の重なり(タグ一致は加点)で上位`limit`件を返す。モデル非依存。
pub fn search(query: &str, limit: usize) -> Vec<Entry> {
    let q = words(query);
    if q.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, Entry)> = list()
        .into_iter()
        .map(|e| {
            let mut score = words(&e.text).intersection(&q).count();
            score += e.tags.iter().filter(|t| q.contains(&t.to_lowercase())).count() * 2;
            (score, e)
        })
        .filter(|(s, _)| *s > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.created_unix.cmp(&a.1.created_unix)));
    scored.into_iter().take(limit).map(|(_, e)| e).collect()
}

/// プロンプトへ足す短い文脈(該当無しなら`None`)。
pub fn context_for(query: &str) -> Option<String> {
    let hits = search(query, 3);
    if hits.is_empty() {
        return None;
    }
    Some(format!("Known facts: {}", hits.iter().map(|e| e.text.as_str()).collect::<Vec<_>>().join(" | ")))
}

/// 生成プロンプト全体から(最後の`Student:`以降、無ければ全体を検索語として)文脈を作る。
/// 既に`Known facts:`が入っている(persona経由)場合は二重に足さない。
pub fn context_for_prompt(prompt: &str) -> Option<String> {
    if prompt.contains("Known facts:") {
        return None;
    }
    let query = prompt.rsplit_once("Student:").map(|(_, s)| s).unwrap_or(prompt);
    context_for(query)
}

/// モデル切替の直前に呼ぶ世代バックアップ(失敗してもモデル切替は止めない)。
pub fn snapshot() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let src = file();
    if !src.exists() {
        return;
    }
    let bdir = dir().join("backups");
    if std::fs::create_dir_all(&bdir).is_err() {
        return;
    }
    let _ = std::fs::copy(&src, bdir.join(format!("knowledge-{}.json", now())));
    if let Ok(rd) = std::fs::read_dir(&bdir) {
        let mut files: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        files.sort();
        while files.len() > KEEP_SNAPSHOTS {
            let _ = std::fs::remove_file(files.remove(0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 保存先は環境変数(プロセス全体で共有)なので、保存先を触るテストは同時に走らせない。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn knowledge_survives_independent_of_models_dir() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir().join(format!("aruaru-knowledge-test-{}", now()));
        std::env::set_var("ARUARU_LLM_KNOWLEDGE_DIR", &tmp);
        add(AddRequest { text: "Guests love anime topics".into(), tags: vec!["anime".into()], source: "test".into() }).unwrap();
        snapshot();
        // 「モデル入替」を模して models/ を触っても、知識は読める
        assert_eq!(search("anime", 3).len(), 1);
        assert!(context_for("tell me about anime").is_some());
        assert!(tmp.join("backups").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn words_splits_japanese_into_bigrams_and_keeps_english_words() {
        let w = words("ドットインストールの料金はProgate 1480円");
        assert!(w.contains("ドッ") && w.contains("トイ") && w.contains("料金"), "{w:?}");
        assert!(w.contains("progate") && w.contains("1480"), "{w:?}");
        // 英数字の語と日本語の断片が混ざっても、境界でちゃんと切れる
        assert!(!w.iter().any(|x| x.contains('p') && x.chars().any(|c| is_cjk(c))), "{w:?}");
        // 1文字だけの日本語(助詞など)は雑音になるので入れない
        assert!(!words("の").contains("の"));
    }

    #[test]
    fn japanese_question_finds_japanese_fact() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = std::env::temp_dir().join(format!("aruaru-knowledge-ja-{}", now()));
        std::env::set_var("ARUARU_LLM_KNOWLEDGE_DIR", &tmp);
        add(AddRequest { text: "ドットインストールの月額プランは1480円（税込）。".into(), tags: vec![], source: "test".into() }).unwrap();
        add(AddRequest { text: "paizaラーニングの無料講座はPython入門を含む。".into(), tags: vec![], source: "test".into() }).unwrap();
        // 日本語の質問(語順・助詞が違っても)で、日本語の事実が拾える
        let hits = search("ドットインストールの料金を教えて", 3);
        assert_eq!(hits.first().map(|e| e.text.contains("ドットインストール")), Some(true), "{hits:?}");
        // 無関係な質問では拾わない
        assert!(search("天気予報", 3).is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
