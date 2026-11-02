//! `reqwest` の `json` 機能の代わり(方針: JSON は RS-JSON の `parse_secure` と `sonic-rs` で読み書きする)。
//!
//! - 送信: `RequestBuilder::json(&body)` と同じ書き方で、`sonic-rs` で直列化して送る。
//! - 受信: `Response::json::<T>()` と同じ書き方で、本文を `RS-JSON` の `parse_secure`(サイズ・深さ・重複キーの上限つき)
//!   で読んでから `T` に変換する。外部のAPIから来る本文なので、巨大・極端に深い入れ子・重複キーは拒否する。
//!
//! 呼び出し側は `use crate::http_json::{RequestBuilderJson, ResponseJson};` を足すだけで、
//! 従来の `.json(&body)` と `.json().await` をそのまま使える。

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
    fn json<T: Serialize + ?Sized>(self, body: &T) -> Self;
}

impl RequestBuilderJson for reqwest::RequestBuilder {
    fn json<T: Serialize + ?Sized>(self, body: &T) -> Self {
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
    fn json<T: DeserializeOwned>(self) -> impl std::future::Future<Output = Result<T, JsonHttpError>>;
}

impl ResponseJson for reqwest::Response {
    async fn json<T: DeserializeOwned>(self) -> Result<T, JsonHttpError> {
        let bytes = self.bytes().await.map_err(|e| JsonHttpError(format!("failed to read response body: {e}")))?;
        let text = std::str::from_utf8(&bytes).map_err(|e| JsonHttpError(format!("response is not UTF-8: {e}")))?;
        let value = rust_json::parse_secure(text, &response_limits()).map_err(|e| JsonHttpError(format!("rejected response JSON: {e}")))?;
        serde_json::from_value(value).map_err(|e| JsonHttpError(format!("unexpected response shape: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_reject_duplicate_keys_and_deep_nesting() {
        let l = response_limits();
        assert!(rust_json::parse_secure(r#"{"a":1,"a":2}"#, &l).is_err());
        let deep = format!("{}1{}", "[".repeat(80), "]".repeat(80));
        assert!(rust_json::parse_secure(&deep, &l).is_err());
        assert!(rust_json::parse_secure(r#"{"a":[1,2,{"b":null}]}"#, &l).is_ok());
    }

    #[test]
    fn request_body_is_serialized_with_sonic_rs() {
        let rb = reqwest::Client::new().post("http://127.0.0.1:9/x").json(&serde_json::json!({"k": "値", "n": [1, 2]}));
        let req = rb.build().unwrap();
        assert_eq!(req.headers().get("content-type").unwrap(), "application/json");
        let body = req.body().unwrap().as_bytes().unwrap();
        let v: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(v["k"], "値");
        assert_eq!(v["n"][1], 2);
    }
}
