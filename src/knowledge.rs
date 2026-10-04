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

fn words(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(str::to_string)
        .collect()
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

    #[test]
    fn knowledge_survives_independent_of_models_dir() {
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
}
