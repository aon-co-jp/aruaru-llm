//! aruaru-llm 用の翻訳ワーカー(別プロセス)。M2M100(418M, INT8)を CTranslate2 で動かす。
//!
//! 受け口は Connect の単純呼び出し: `POST /aruaru_translate.v1.Translate/Translate`
//! 本文 `{"text":"..","target_lang":"ja","source_lang":"en"|null}` → `{"translation":"..","engine":".."}`。
//! `POST /aruaru_translate.v1.Translate/Health` は `{}` → `{"ok":true}`。
//! 127.0.0.1 にだけ待ち受ける(外へ出さない)。
//! 環境変数: `ARUARU_TRANSLATE_MODEL_DIR`(model.bin / shared_vocabulary.json / sentencepiece.bpe.model)、
//! `ARUARU_TRANSLATE_ADDR`(既定 127.0.0.1:4620)、`ARUARU_TRANSLATE_THREADS`(既定 4)。
use anyhow::{anyhow, Result};
use ct2rs::{Config, Tokenizer, TranslationOptions, Translator};
use sentencepiece::SentencePieceProcessor;
use std::sync::Mutex;

const MARK: char = '\u{1}';
const MAX_BODY: usize = 64 * 1024;
const MAX_TEXT_CHARS: usize = 4000;

/// 原文の言語を `Tokenizer::encode`(文字列1つしか渡せない)へ運ぶための包み方。
pub fn wrap_source(src: &str, text: &str) -> String {
    format!("{MARK}{src}{MARK}{text}")
}

pub fn unwrap_source(input: &str) -> (&str, &str) {
    input.strip_prefix(MARK).and_then(|r| r.split_once(MARK)).unwrap_or(("en", input))
}

/// 言語コードは小文字の英字2〜3文字だけ許す(トークン名 `__xx__` を作るため)。
pub fn valid_lang(c: &str) -> bool {
    (2..=3).contains(&c.len()) && c.bytes().all(|b| b.is_ascii_lowercase())
}

pub fn lang_token(c: &str) -> String {
    format!("__{c}__")
}

/// 出力のトークン列から言語タグ・終端記号を取り除く。
pub fn strip_special(tokens: Vec<String>) -> Vec<String> {
    tokens.into_iter().filter(|t| t != "</s>" && t != "<s>" && !(t.starts_with("__") && t.ends_with("__") && t.len() <= 7)).collect()
}

/// 文ごとに分ける(長い1文のまとめ訳で前半が落ちる/意味が逆になる問題の対策)。
/// 。！？!?改行は常に区切り、`.` は後ろが空白のときだけ区切る(小数点・略語を壊さない)。
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let end = match c {
            '。' | '！' | '？' | '!' | '?' | '\n' => true,
            '.' => chars.get(i + 1).map_or(true, |n| n.is_whitespace()),
            _ => false,
        };
        if end {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// 短い決まり文句(あいさつ)は M2M100 が誤訳しやすい(実測: 418M で "Good morning" → 「こんばんは」)。
/// 完全一致(大文字小文字・末尾の句読点は無視)したときだけ辞書の訳を使い、それ以外はモデルに任せる。
const FIXED: &[(&str, &str, &str, &str)] = &[
    ("en", "ja", "good morning", "おはようございます"),
    ("en", "ja", "good afternoon", "こんにちは"),
    ("en", "ja", "good evening", "こんばんは"),
    ("en", "ja", "good night", "おやすみなさい"),
    ("en", "ja", "hello", "こんにちは"),
    ("en", "ja", "thank you", "ありがとうございます"),
    ("en", "ja", "thanks", "ありがとう"),
    ("en", "ja", "goodbye", "さようなら"),
    ("en", "ja", "nice to meet you", "はじめまして"),
    ("ja", "en", "おはようございます", "Good morning"),
    ("ja", "en", "おはよう", "Good morning"),
    ("ja", "en", "こんにちは", "Hello"),
    ("ja", "en", "こんばんは", "Good evening"),
    ("ja", "en", "おやすみなさい", "Good night"),
    ("ja", "en", "ありがとうございます", "Thank you"),
    ("ja", "en", "ありがとう", "Thank you"),
    ("ja", "en", "さようなら", "Goodbye"),
    ("ja", "en", "はじめまして", "Nice to meet you"),
];

/// 文1つを決まり文句の辞書で引く。末尾の句読点は訳の後ろへそのまま付け直す。
pub fn fixed_phrase(source: &str, target: &str, sentence: &str) -> Option<String> {
    let t = sentence.trim();
    let core = t.trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | '。' | '！' | '？'));
    let tail = &t[core.len()..];
    let key = core.trim().to_lowercase();
    let (_, _, _, out) = FIXED.iter().find(|(s, g, k, _)| *s == source && *g == target && *k == key)?;
    let tail = if target == "ja" {
        tail.replace('.', "。").replace('!', "！").replace('?', "？")
    } else {
        tail.replace('。', ".").replace('！', "!").replace('？', "?")
    };
    Some(format!("{out}{tail}"))
}

struct M2mTokenizer {
    spm: SentencePieceProcessor,
}

impl Tokenizer for M2mTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<String>> {
        let (src, text) = unwrap_source(input);
        let mut toks = vec![lang_token(src)];
        toks.extend(self.spm.encode(text)?.into_iter().map(|p| p.piece));
        toks.push("</s>".to_string());
        Ok(toks)
    }

    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(self.spm.decode_pieces(&strip_special(tokens))?)
    }
}

fn json_reply(status: u16, body: serde_json::Value) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut r = tiny_http::Response::from_string(body.to_string()).with_status_code(status);
    r.add_header(tiny_http::Header::from_bytes("content-type", "application/json").unwrap());
    r
}

fn translate(tr: &Mutex<Translator<M2mTokenizer>>, v: &serde_json::Value) -> Result<String> {
    let text = v.get("text").and_then(|x| x.as_str()).ok_or_else(|| anyhow!("text is required"))?;
    let target = v.get("target_lang").and_then(|x| x.as_str()).ok_or_else(|| anyhow!("target_lang is required"))?;
    let source = v.get("source_lang").and_then(|x| x.as_str()).unwrap_or("en");
    if text.chars().count() > MAX_TEXT_CHARS {
        return Err(anyhow!("text is too long (max {MAX_TEXT_CHARS} chars)"));
    }
    if !valid_lang(target) || !valid_lang(source) {
        return Err(anyhow!("language codes must be 2-3 lowercase letters"));
    }
    let opts = TranslationOptions::<String, String> { beam_size: 2, max_decoding_length: 256, ..Default::default() };
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Err(anyhow!("empty translation"));
    }
    let mut results: Vec<Option<String>> = sentences.iter().map(|t| fixed_phrase(source, target, t)).collect();
    let todo: Vec<usize> = (0..sentences.len()).filter(|&i| results[i].is_none()).collect();
    if !todo.is_empty() {
        let inputs: Vec<String> = todo.iter().map(|&i| wrap_source(source, &sentences[i])).collect();
        let prefixes: Vec<Vec<String>> = todo.iter().map(|_| vec![lang_token(target)]).collect();
        let out = tr
            .lock()
            .map_err(|_| anyhow!("model lock poisoned"))?
            .translate_batch_with_target_prefix(&inputs, &prefixes, &opts, None)?;
        for (&i, (s, _)) in todo.iter().zip(out) {
            results[i] = Some(s);
        }
    }
    let sep = if matches!(target, "ja" | "zh") { "" } else { " " };
    let joined = results.into_iter().flatten().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(sep);
    if joined.is_empty() { Err(anyhow!("empty translation")) } else { Ok(joined) }
}

fn main() -> Result<()> {
    let dir = std::env::var("ARUARU_TRANSLATE_MODEL_DIR").map_err(|_| anyhow!("set ARUARU_TRANSLATE_MODEL_DIR"))?;
    let addr = std::env::var("ARUARU_TRANSLATE_ADDR").unwrap_or_else(|_| "127.0.0.1:4620".into());
    if !addr.starts_with("127.0.0.1:") && !addr.starts_with("[::1]:") {
        return Err(anyhow!("ARUARU_TRANSLATE_ADDR must be a loopback address"));
    }
    let threads: usize = std::env::var("ARUARU_TRANSLATE_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let spm = SentencePieceProcessor::open(std::path::Path::new(&dir).join("sentencepiece.bpe.model"))?;
    let cfg = Config { num_threads_per_replica: threads, ..Default::default() };
    let tr = Mutex::new(Translator::with_tokenizer(&dir, M2mTokenizer { spm }, &cfg)?);
    let server = tiny_http::Server::http(&addr).map_err(|e| anyhow!("bind {addr}: {e}"))?;
    eprintln!("translate-worker listening on {addr}");
    for mut req in server.incoming_requests() {
        let path = req.url().to_string();
        let resp = if req.method() != &tiny_http::Method::Post {
            json_reply(405, serde_json::json!({"error": "POST only"}))
        } else if path == "/aruaru_translate.v1.Translate/Health" {
            json_reply(200, serde_json::json!({"ok": true}))
        } else if path == "/aruaru_translate.v1.Translate/Translate" {
            let mut body = Vec::new();
            use std::io::Read;
            let _ = req.as_reader().take(MAX_BODY as u64 + 1).read_to_end(&mut body);
            if body.len() > MAX_BODY {
                json_reply(413, serde_json::json!({"error": "body too large"}))
            } else {
                match serde_json::from_slice::<serde_json::Value>(&body).map_err(|e| anyhow!("bad JSON: {e}")).and_then(|v| translate(&tr, &v)) {
                    Ok(t) => json_reply(200, serde_json::json!({"translation": t, "engine": "m2m100-418m-ctranslate2-int8"})),
                    Err(e) => json_reply(400, serde_json::json!({"error": e.to_string()})),
                }
            }
        } else {
            json_reply(404, serde_json::json!({"error": "unknown method"}))
        };
        let _ = req.respond(resp);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_round_trip() {
        let w = wrap_source("ja", "こんにちは");
        assert_eq!(unwrap_source(&w), ("ja", "こんにちは"));
        assert_eq!(unwrap_source("plain"), ("en", "plain"));
    }

    #[test]
    fn language_codes_are_restricted() {
        assert!(valid_lang("ja") && valid_lang("fil"));
        assert!(!valid_lang("J") && !valid_lang("ja-JP") && !valid_lang("日本") && !valid_lang(""));
    }

    #[test]
    fn sentences_are_split() {
        assert_eq!(split_sentences("Hello. How are you? Fine!"), vec!["Hello.", "How are you?", "Fine!"]);
        assert_eq!(split_sentences("今日は晴れ。散歩しよう！"), vec!["今日は晴れ。", "散歩しよう！"]);
        assert_eq!(split_sentences("Pi is 3.14 ok"), vec!["Pi is 3.14 ok"]);
        assert!(split_sentences("  \n ").is_empty());
    }

    #[test]
    fn fixed_phrases_override_known_mistranslations() {
        assert_eq!(fixed_phrase("en", "ja", "Good morning").as_deref(), Some("おはようございます"));
        assert_eq!(fixed_phrase("en", "ja", "good morning!").as_deref(), Some("おはようございます！"));
        assert_eq!(fixed_phrase("ja", "en", "おはようございます。").as_deref(), Some("Good morning."));
        assert_eq!(fixed_phrase("en", "ja", "Good morning, everyone"), None);
        assert_eq!(fixed_phrase("en", "fr", "Good morning"), None);
    }

    #[test]
    fn specials_are_stripped() {
        let t = ["__ja__", "▁今日", "</s>", "は"].iter().map(|s| s.to_string()).collect();
        assert_eq!(strip_special(t), vec!["▁今日".to_string(), "は".to_string()]);
    }
}
