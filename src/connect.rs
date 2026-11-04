//! Connect の「単純呼び出し(unary)」で受ける API の入口(2026-11-04新設、REST 全廃方針)。
//!
//! 形式は `POST /aruaru_llm.v1.Llm/<Method>`、本文は JSON(`content-type: application/json`)。
//! すべて POST。以前 GET だった呼び出し(引数は URL のクエリ)は、JSON 本文の各項目をクエリに直して
//! 既存のハンドラへ渡す(`cx_get`)。以前 POST だったものは本文をそのまま渡す(`cx_post`)。
//! エラーは Connect の形式 `{"code": "...", "message": "..."}`(HTTP ステータスは Connect 仕様の対応表)。
//! JSON の読み込みは RS-JSON の `parse_secure`(サイズ・深さ・重複キーの上限つき)。
//!
//! REST のまま残すのは、規格や配布の都合が形を決めるものだけ: `/healthz`、`/`(案内ページ)、
//! `/v1/ocr/fonts/:name`(フォントの静的ファイル配信)、`/api/docs`(型駆動の文書)。

use http_body_util::BodyExt;
use hyper::StatusCode;
use open_runo_poem_compat::hyper_compat::{json_response, Params};
use open_runo_poem_compat::{handler_fn, post, Handler, MethodRouter, Request, Response};
use std::collections::HashMap;

pub const PREFIX: &str = "/aruaru_llm.v1.Llm/";
/// 引数だけの呼び出し(以前の GET)の本文の上限。
const QUERY_BODY_MAX: usize = 64 * 1024;

fn code_of(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 422 => "invalid_argument",
        401 => "unauthenticated",
        403 => "permission_denied",
        404 => "not_found",
        405 | 501 => "unimplemented",
        409 => "aborted",
        412 => "failed_precondition",
        413 | 429 => "resource_exhausted",
        502 | 503 | 504 => "unavailable",
        s if s >= 500 => "internal",
        _ => "unknown",
    }
}

pub fn error(status: StatusCode, message: &str) -> Response {
    json_response(status, &serde_json::json!({"code": code_of(status), "message": message, "error": message}))
}

/// 旧形式のエラー本文(`{"error": ".."}` など)を Connect の `{"code","message"}` に直す。成功はそのまま通す。
async fn normalise(resp: Response) -> Response {
    if resp.status().as_u16() < 400 {
        return resp;
    }
    let status = resp.status();
    let (_, body) = resp.into_parts();
    let bytes = match body.collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => return error(status, "error"),
    };
    let text = String::from_utf8_lossy(&bytes).to_string();
    let limits = rust_json::Limits { max_bytes: 1 << 20, max_depth: 32, reject_duplicate_keys: false };
    let message = rust_json::parse_secure(&text, &limits)
        .ok()
        .and_then(|v| {
            v.get("message").or_else(|| v.get("error")).and_then(|m| m.as_str().map(str::to_string))
        })
        .unwrap_or(text);
    error(status, &message)
}

fn pct_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// JSON 本文(オブジェクト)を読み、(クエリ文字列, 項目) を返す。空の本文は `{}` と同じ。
async fn read_args(req: &mut Request) -> Result<HashMap<String, String>, Response> {
    let mut buf = Vec::new();
    let body = req.body_mut();
    loop {
        match body.frame().await {
            None => break,
            Some(Ok(f)) => {
                if let Some(d) = f.data_ref() {
                    buf.extend_from_slice(d);
                    if buf.len() > QUERY_BODY_MAX {
                        return Err(error(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"));
                    }
                }
            }
            Some(Err(_)) => return Err(error(StatusCode::BAD_REQUEST, "could not read the request body")),
        }
    }
    let text = String::from_utf8(buf).map_err(|_| error(StatusCode::BAD_REQUEST, "body is not UTF-8"))?;
    if text.trim().is_empty() {
        return Ok(HashMap::new());
    }
    let limits = rust_json::Limits { max_bytes: QUERY_BODY_MAX, max_depth: 8, reject_duplicate_keys: true };
    let value = rust_json::parse_secure(&text, &limits)
        .map_err(|e| error(StatusCode::BAD_REQUEST, &format!("invalid JSON: {e}")))?;
    let obj = value.as_object().ok_or_else(|| error(StatusCode::BAD_REQUEST, "body must be a JSON object"))?;
    let mut out = HashMap::new();
    for (k, v) in obj {
        let s = match v {
            serde_json::Value::Null => continue,
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => return Err(error(StatusCode::BAD_REQUEST, &format!("field `{k}` must be a string, number or boolean"))),
        };
        out.insert(k.clone(), s);
    }
    Ok(out)
}

fn ensure_post_json(req: &Request) -> Option<Response> {
    if req.method() != hyper::Method::POST {
        return Some(error(StatusCode::METHOD_NOT_ALLOWED, "Connect calls must be POST"));
    }
    None
}

/// 以前 POST だったもの(本文をそのままハンドラへ)。
pub fn cx_post(h: Handler) -> MethodRouter {
    post(handler_fn(move |req: Request, p: Params| {
        let h = h.clone();
        async move { normalise(h(req, p).await).await }
    }))
}

/// 以前 GET だったもの(JSON 本文の項目をクエリに直してハンドラへ)。
/// `path_param` があれば、その項目を `:param` として渡す(以前のパス引数)。
pub fn cx_get(h: Handler, path_param: Option<&'static str>) -> MethodRouter {
    post(handler_fn(move |mut req: Request, mut p: Params| {
        let h = h.clone();
        async move {
            if let Some(r) = ensure_post_json(&req) {
                return r;
            }
            let args = match read_args(&mut req).await {
                Ok(a) => a,
                Err(r) => return r,
            };
            let mut args = args;
            if let Some(name) = path_param {
                match args.remove(name) {
                    Some(v) if !v.is_empty() => {
                        p.0.insert(name.to_string(), v);
                    }
                    _ => return error(StatusCode::BAD_REQUEST, &format!("`{name}` is required")),
                }
            }
            let mut keys: Vec<_> = args.keys().cloned().collect();
            keys.sort();
            let q: Vec<String> = keys.iter().map(|k| format!("{}={}", pct_encode(k), pct_encode(&args[k]))).collect();
            let uri = if q.is_empty() { "/".to_string() } else { format!("/?{}", q.join("&")) };
            if let Ok(u) = uri.parse() {
                *req.uri_mut() = u;
            }
            *req.method_mut() = hyper::Method::GET;
            normalise(h(req, p).await).await
        }
    }))
}

/// `/v1/ocr/prepare` → `OcrPrepare` のように、旧パスから Connect のメソッド名を作る。
pub fn path(method: &str) -> String {
    format!("{PREFIX}{method}")
}
