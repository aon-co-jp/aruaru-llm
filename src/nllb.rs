//! 専用翻訳モデル(M2M100)による`POST /v1/translate`の品質改善。
//!
//! 2026-11-16: `rust-bert`/`tch`(libtorch)への依存をやめ、**別プロセスの翻訳ワーカー**を
//! Connect(`POST <URL>/aruaru_translate.v1.Translate/Translate`、JSON)で呼ぶ方式に置き換えた。
//! 理由: `rust-bert`は更新が止まり(0.23.0が最新、`tch ^0.17`固定)、推移的に`protobuf 2`・`zip 0.6`・
//! `zstd-safe 5`・`h2 0.3`の脆弱性(RUSTSEC-2024-0437/2026-0258/0367〜0369/0403)を抱えていた。
//! ワーカーは`tools/translate-worker`(CTranslate2、MIT)。URLは環境変数`ARUARU_TRANSLATE_URL`
//! (未設定なら無効=従来のGPT-2流用へフォールバック)。ワーカー停止・失敗時も同様にフォールバックする。

use std::time::Duration;

/// M2M100の言語コードへ。英語名・小文字・2〜3文字コードを受け付ける(未知の名前は`None`)。
pub fn lang_code(name: &str) -> Option<String> {
    let n = name.trim().to_ascii_lowercase();
    let code = match n.as_str() {
        "english" => "en",
        "japanese" | "jp" => "ja",
        "french" => "fr",
        "german" => "de",
        "spanish" => "es",
        "chinese" | "chinese_simplified" => "zh",
        "korean" => "ko",
        "russian" => "ru",
        "italian" => "it",
        "arabic" => "ar",
        "portuguese" => "pt",
        "vietnamese" => "vi",
        "thai" => "th",
        "indonesian" => "id",
        "hindi" => "hi",
        _ if (2..=3).contains(&n.len()) && n.chars().all(|c| c.is_ascii_lowercase()) => return Some(n),
        _ => return None,
    };
    Some(code.to_string())
}

/// ワーカーのURL(設定されていなければ`None`)。
pub fn worker_url() -> Option<String> {
    std::env::var("ARUARU_TRANSLATE_URL").ok().map(|s| s.trim().trim_end_matches('/').to_string()).filter(|s| !s.is_empty())
}

pub fn request_body(text: &str, target: &str, source: Option<&str>) -> String {
    serde_json::json!({"text": text, "target_lang": target, "source_lang": source}).to_string()
}

/// ワーカーで翻訳する。失敗は`Err`(呼び出し側がGPT-2流用へフォールバックする)。
pub async fn translate_with_nllb(text: &str, target_lang: &str, source_lang: Option<&str>) -> Result<String, String> {
    let url = worker_url().ok_or_else(|| "ARUARU_TRANSLATE_URL is not set".to_string())?;
    let target = lang_code(target_lang).ok_or_else(|| format!("unsupported target_lang: {target_lang}"))?;
    let source = source_lang.and_then(lang_code);
    let client = reqwest::Client::builder().timeout(Duration::from_secs(60)).build().map_err(|e| e.to_string())?;
    let resp = client
        .post(format!("{url}/aruaru_translate.v1.Translate/Translate"))
        .header("content-type", "application/json")
        .body(request_body(text, &target, source.as_deref()))
        .send()
        .await
        .map_err(|e| format!("translate worker unreachable: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("translate worker HTTP {}", resp.status()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("bad JSON from translate worker: {e}"))?;
    v.get("translation").and_then(|x| x.as_str()).map(|s| s.to_string()).filter(|s| !s.trim().is_empty()).ok_or_else(|| "translate worker returned no translation".to_string())
}

/// 翻訳ワーカーが設定されているか(レスポンスの注記の出し分けに使う)。
pub fn is_available() -> bool {
    worker_url().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_names_and_codes() {
        assert_eq!(lang_code("Japanese").as_deref(), Some("ja"));
        assert_eq!(lang_code("fr").as_deref(), Some("fr"));
        assert_eq!(lang_code("klingon-ish"), None);
    }

    #[test]
    fn request_body_is_valid_json() {
        let v: serde_json::Value = serde_json::from_str(&request_body("hi \"x\"", "ja", None)).unwrap();
        assert_eq!(v["text"], "hi \"x\"");
        assert!(v["source_lang"].is_null());
    }

    #[test]
    fn without_url_translation_is_unavailable() {
        std::env::remove_var("ARUARU_TRANSLATE_URL");
        assert!(!is_available());
    }
}
