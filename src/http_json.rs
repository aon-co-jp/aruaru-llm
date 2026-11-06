//! `reqwest` の `json` 機能の代わり(方針: JSON は RS-JSON の `parse_secure` と `sonic-rs` で読み書きする)。
//!
//! - 送信: `RequestBuilder::json(&body)` と同じ書き方で、`sonic-rs` で直列化して送る。
//! - 受信: `Response::json::<T>()` と同じ書き方で、本文を `RS-JSON` の `parse_secure`(サイズ・深さ・重複キーの上限つき)
//!   で読んでから `T` に変換する。外部のAPIから来る本文なので、巨大・極端に深い入れ子・重複キーは拒否する。
//!
//! 呼び出し側は `use crate::http_json::{RequestBuilderJson, ResponseJson};` を足して、
//! `.json_sonic(&body)`(送信)と `.json_secure().await`(受信)を使う。
//!
//! **メソッド名を `json`/`json` と同じにしてはいけない**(2026-11-06に判明した実害): `reqwest` の `json` 機能は、
//! 他の依存(RPoem のルーター・OpenTelemetry など)がフィーチャを足すと有効になり(フィーチャの統合)、
//! その固有メソッド `Response::json` / `RequestBuilder::json` がこのトレイトのメソッドより優先される。
//! 以前は同名だったため、この安全な受信(上限・重複キーの検査)は実際には一度も働かず、`serde_json` で
//! 素通しに読んでいた。名前を変えて、必ずここを通るようにした。

use serde::de::DeserializeOwned;
use serde::Serialize;

/// 受信する本文の上限。検索結果や生成結果は大きめなので、知識の取り込み(8 MiB・深さ16)よりゆるくする。
fn response_limits() -> rust_json::Limits {
    rust_json::Limits { max_bytes: 32 * 1024 * 1024, max_depth: 64, reject_duplicate_keys: true }
}

#[derive(Debug)]
pub struct JsonHttpError(pub String);

impl std::fmt::Display for JsonHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JsonHttpError {}

pub trait RequestBuilderJson {
    /// `application/json` で本文を送る(`reqwest` の `json` 機能の代わり)。
    fn json_sonic<T: Serialize + ?Sized>(self, body: &T) -> Self;
}

impl RequestBuilderJson for reqwest::RequestBuilder {
    fn json_sonic<T: Serialize + ?Sized>(self, body: &T) -> Self {
        match sonic_rs::to_vec(body) {
            Ok(bytes) => self.header(reqwest::header::CONTENT_TYPE, "application/json").body(bytes),
            // 直列化できないのはプログラムの誤り。空の本文のまま送ると原因が分かりにくいので、
            // 送信時にエラーになるよう、不正なヘッダーを付けて失敗させる。
            Err(e) => self.header("x-json-serialize-error", format!("\n{e}")),
        }
    }
}

pub trait ResponseJson {
    /// 本文を `RS-JSON` の `parse_secure` で読んで `T` にする(`reqwest` の `json` 機能の代わり)。
    fn json_secure<T: DeserializeOwned>(self) -> impl std::future::Future<Output = Result<T, JsonHttpError>>;
}

impl ResponseJson for reqwest::Response {
    async fn json_secure<T: DeserializeOwned>(self) -> Result<T, JsonHttpError> {
        let bytes = self.bytes().await.map_err(|e| JsonHttpError(format!("failed to read response body: {e}")))?;
        let text = std::str::from_utf8(&bytes).map_err(|e| JsonHttpError(format!("response is not UTF-8: {e}")))?;
        // 先に安全モード(SIMD版)で上限・深さ・重複キーを検査し、通ったら型へ直接読む(中間の `serde_json::Value` を作らない)。
        rust_json::parse_secure_sonic(text, &response_limits()).map_err(|e| JsonHttpError(format!("rejected response JSON: {e}")))?;
        sonic_rs::from_str(text).map_err(|e| JsonHttpError(format!("unexpected response shape: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_reject_duplicate_keys_and_deep_nesting() {
        let l = response_limits();
        assert!(rust_json::parse_secure_sonic(r#"{"a":1,"a":2}"#, &l).is_err());
        let deep = format!("{}1{}", "[".repeat(80), "]".repeat(80));
        assert!(rust_json::parse_secure_sonic(&deep, &l).is_err());
        assert!(rust_json::parse_secure_sonic(r#"{"a":[1,2,{"b":null}]}"#, &l).is_ok());
    }

    #[test]
    fn request_body_is_serialized_with_sonic_rs() {
        let rb = reqwest::Client::new().post("http://127.0.0.1:9/x").json_sonic(&serde_json::json!({"k": "値", "n": [1, 2]}));
        let req = rb.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "application/json");
        let body = req.body().unwrap().as_bytes().unwrap();
        let v: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(v["k"], "値");
        assert_eq!(v["n"][1], 2);
    }
}
