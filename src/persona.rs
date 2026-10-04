//! 先生キャラクター(女性=メイドの先生/男性=執事の先生)の応対方針
//! (ユーザー指示 2026-10-04「記事をaruaru-llmがよく読んで、女性キャラ選択時は
//! メイドの先生、男性キャラ選択時は執事の先生として対応」)。
//!
//! 参考記事: 「訪日客に人気な秋葉原のメイドカフェ、カタコトでも単語を軸に話しかけ」
//! (PRESIDENT Online <https://president.jp/articles/-/26332>、livedoorニュース再編集版
//! 2018-10-13)。**本文は転載せず**、要点(単語を軸にする・身振りと笑顔で補う・
//! 共通の話題を探す・日本語を少し混ぜる)だけを自分の言葉で言い換えて方針に反映する。
//!
//! 正直な開示: 素のGPT-2は方針文を確実には守れない。ここは「プロンプトに載せる
//! 方針文+国別の話題ヒント」を返すだけで、守る保証は無い。国の判定はIPジオロケーション
//! ではなく、呼び出し側が渡す国名(ブラウザ言語/タイムゾーンや発話由来)を使う。

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct PersonaRequest {
    /// "female"(メイド)または"male"(執事)。それ以外は400相当で拒否する。
    pub gender: String,
    /// 任意。国名(英語/日本語)。あれば国の話題ヒントを付ける。
    pub country: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PersonaResponse {
    pub persona: &'static str,
    pub name: &'static str,
    pub role_ja: &'static str,
    pub role_en: &'static str,
    pub greeting_en: &'static str,
    pub greeting_ja: &'static str,
    /// GPT-2等のプロンプト先頭へ置く英語の方針文。
    pub system_prompt: String,
    pub topic_hints: Vec<String>,
    pub source_ja: &'static str,
}

const TECHNIQUES_EN: &str = "Speak in short, simple sentences built around one key word, and repeat the key word. \
Back words up with gestures, a warm smile and expressive reactions. \
Look for common ground (anime, movies, food, travel) and use the guest's own country as a topic. \
Mix in a little Japanese so the guest enjoys Japanese culture. Never mock imperfect English; praise every attempt.";

pub fn build(req: &PersonaRequest, topic_hints: Vec<String>) -> Option<PersonaResponse> {
    let (persona, name, role_ja, role_en, greeting_en, greeting_ja, role_line) =
        match req.gender.to_ascii_lowercase().as_str() {
            "female" | "f" | "maid" => (
                "maid", "Sakura", "メイドの先生", "maid teacher",
                "Welcome home, master!", "おかえりなさい、ご主人様!",
                "You are Sakura, a cheerful, gentle maid teacher at a maid cafe in Akihabara, \
                 and you greet guests with \"Welcome home, master!\".",
            ),
            "male" | "m" | "butler" => (
                "butler", "Tora", "執事の先生", "butler teacher",
                "Welcome back, my lady, my lord.", "おかえりなさいませ、お嬢様、旦那様。",
                "You are Tora, a courteous, composed butler teacher at a butler cafe in Akihabara, \
                 and you greet guests with \"Welcome back, my lady / my lord.\".",
            ),
            _ => return None,
        };
    let mut system_prompt = format!("{role_line} {TECHNIQUES_EN}");
    if !topic_hints.is_empty() {
        system_prompt.push_str(" Possible small-talk topics about the guest's country: ");
        system_prompt.push_str(&topic_hints.join("; "));
        system_prompt.push('.');
    }
    Some(PersonaResponse {
        persona, name, role_ja, role_en, greeting_en, greeting_ja, system_prompt, topic_hints,
        source_ja: "参考: PRESIDENT Online「訪日客に人気な秋葉原のメイドカフェ、カタコトでも単語を軸に話しかけ」(要点のみ言い換え)",
    })
}
