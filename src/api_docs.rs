//! 型付きのAPI宣言(RPoemのopen-runo-openapi)。`/api/docs`・`/api/openapi.json`・
//! Cosmo互換SDL・Tauriマニフェスト・TSクライアントの元になる。
//!
//! リクエスト本文・クエリの型は、実際のハンドラが読む構造体そのもの
//! (`super::*Request`)。レスポンスはハンドラごとにアドホックな`json!`なので、
//! 共通の外枠(`ok`/`error`+追加フィールド)として宣言する。

use super::*;
use open_runo_poem_compat::openapi::{Operation as Op, Spec};
use schemars::JsonSchema;
use serde::Serialize;

/// 多くのエンドポイントが返すJSONの外枠。`ok`と`error`以外は
/// エンドポイントごとに異なる追加フィールドとして付く。
#[derive(Serialize, JsonSchema)]
struct Envelope {
    ok: Option<bool>,
    error: Option<String>,
    #[serde(flatten)]
    data: serde_json::Map<String, serde_json::Value>,
}

fn get(path: &str, summary: &str, tag: &str) -> Op {
    Op::get(path).summary(summary).tag(tag).public().ok::<Envelope>("OK")
}

fn get_t<T: JsonSchema>(path: &str, summary: &str, tag: &str) -> Op {
    Op::get(path).summary(summary).tag(tag).public().ok::<T>("OK")
}

fn post_t<T: JsonSchema>(path: &str, summary: &str, tag: &str) -> Op {
    Op::post(path).summary(summary).tag(tag).public().ok::<T>("OK").response::<ErrorBody>(400, "入力不正")
}

#[derive(Serialize, JsonSchema)]
struct ErrorBody {
    error: String,
}

/// `GET /v1/media-search`
#[derive(JsonSchema)]
#[allow(dead_code)]
struct MediaSearchQuery {
    /// 検索語(必須)
    q: String,
    /// 件数(既定10)
    n: Option<u8>,
}

/// `GET /v1/recommend`
#[derive(JsonSchema)]
#[allow(dead_code)]
struct RecommendQuery {
    /// 推論精度(省略時はF32)
    precision: Option<String>,
}

/// `GET /v1/news/for`
#[derive(JsonSchema)]
#[allow(dead_code)]
struct NewsForQuery {
    /// 国(必須)
    country: String,
}

/// `GET /v1/knowledge/search`
#[derive(JsonSchema)]
#[allow(dead_code)]
struct KnowledgeSearchQuery {
    /// 検索語(省略時は全件)
    q: Option<String>,
}

fn post(path: &str, summary: &str, tag: &str) -> Op {
    Op::post(path).summary(summary).tag(tag).public().ok::<Envelope>("OK").response::<Envelope>(400, "入力不正")
}

pub fn spec() -> Spec {
    let s = Spec::new("aruaru-llm", env!("CARGO_PKG_VERSION"))
        .description("契約不要のAI推論サーバー。本文・クエリの型は実装の構造体、レスポンスは共通の外枠(ok/error+追加フィールド)。");
    // --- 生成・チャット ---
    let s = s
        .op(post_t::<ChatResponse>("/v1/chat", "チャット(会話)", "generate").body::<ChatRequest>())
        .op(post_t::<GenerateResponse>("/v1/generate", "文章生成", "generate").body::<GenerateRequest>())
        .op(post_t::<GenerateSpeculativeResponse>("/v1/generate-speculative", "投機的デコードによる文章生成", "generate").body::<GenerateSpeculativeRequest>())
        .op(post_t::<GenerateWithSearchResponse>("/v1/generate-with-search", "検索結果つきの文章生成", "generate").body::<GenerateWithSearchRequest>())
        .op(post_t::<GenerateQwenResponse>("/v1/generate-qwen", "Qwenで文章生成", "generate").body::<GenerateQwenRequest>())
        .op(post_t::<GenerateDeepseekResponse>("/v1/generate-deepseek", "DeepSeekで文章生成", "generate").body::<GenerateDeepseekRequest>())
        .op(post_t::<TranslateResponse>("/v1/translate", "翻訳(NLLB)", "generate").body::<TranslateRequest>())
        .op(post_t::<TranscribeResponse>("/v1/transcribe", "音声の文字起こし", "generate").body::<TranscribeRequest>())
        .op(post_t::<RerankResponse>("/v1/rerank", "検索結果の再ランク", "generate").body::<RerankRequest>())
        .op(post_t::<crate::persona::PersonaResponse>("/v1/persona/prompt", "ペルソナのプロンプトを作る", "generate").body::<crate::persona::PersonaRequest>());
    // --- 分類・セキュリティ ---
    let s = s
        .op(post_t::<ClassifyResponse>("/v1/classify", "汎用の分類", "classify").body::<ClassifyRequest>())
        .op(post_t::<ClassifySecurityResponse>("/v1/classify-security", "セキュリティ分類", "classify").body::<ClassifySecurityRequest>())
        .op(post_t::<ClassifyTrafficResponse>("/v1/security/classify-traffic", "通信の分類(侵入検知)", "classify").body::<ClassifyTrafficRequest>())
        .op(post_t::<ClassifyPiiResponse>("/v1/classify-pii", "個人情報の分類", "classify").body::<ClassifyPiiRequest>());
    // --- OCR ---
    let s = s
        .op(post_t::<OcrResponse>("/v1/ocr", "OCR(画像から文字)", "ocr").body::<OcrRequest>())
        .op(get_t::<OcrLanguagesResponse>("/v1/ocr/languages", "OCRの対応言語", "ocr"))
        .op(post_t::<OcrPrepareResponse>("/v1/ocr/prepare", "OCRの言語データを準備", "ocr").body::<OcrPrepareRequest>())
        .op(post_t::<OcrCorrectResponse>("/v1/ocr/correct", "OCR結果の補正", "ocr").body::<OcrCorrectRequest>())
        .op(get_t::<CharsetResponse>("/v1/ocr/charset", "頻出順の文字集合", "ocr").query::<OcrCharsetQuery>())
        .op(post_t::<OcrGlyphRankResponse>("/v1/ocr/glyph-rank", "字形による候補の順位付け", "ocr").body::<OcrGlyphRankRequest>())
        .op(post_t::<OcrScanResponse>("/v1/ocr/scan", "不自然な文字の走査", "ocr").body::<OcrScanRequest>())
        .op(post_t::<OcrProposeResponse>("/v1/ocr/propose", "候補の提案", "ocr").body::<OcrProposeRequest>())
        .op(post_t::<OcrFillResponse>("/v1/ocr/fill", "文脈による補完", "ocr").body::<OcrFillRequest>())
        .op(get("/v1/ocr/fonts/:name", "OCR用フォントの取得", "ocr"));
    // --- モデル管理 ---
    let s = s
        .op(get_t::<CatalogResponse>("/v1/models/catalog", "モデルカタログ", "models"))
        .op(post_t::<crate::cache_optimizer::CacheOptimizationResult>("/v1/models/optimize-cache", "キャッシュ最適化", "models").body::<OptimizeCacheRequest>())
        .op(post_t::<InstallModelResponse>("/v1/models/install", "モデルを導入", "models").body::<InstallModelRequest>())
        .op(post_t::<SelectModelResponse>("/v1/models/select", "モデルを選択", "models").body::<SelectModelRequest>())
        .op(post_t::<LayerRedundancyResponse>("/v1/models/layer-redundancy", "層の冗長性を調べる", "models").body::<LayerRedundancyRequest>())
        .op(post_t::<FoldLayersResponse>("/v1/models/fold-layers", "層を畳む", "models").body::<FoldLayersRequest>())
        .op(get_t::<CatalogResponse>("/v1/qwen/catalog", "Qwenのカタログ", "models"))
        .op(post_t::<InstallModelResponse>("/v1/qwen/install", "Qwenを導入", "models").body::<InstallModelRequest>())
        .op(post_t::<SelectModelResponse>("/v1/qwen/select", "Qwenを選択", "models").body::<SelectModelRequest>())
        .op(get_t::<DeepseekStatusResponse>("/v1/deepseek/status", "DeepSeekの状態", "models"))
        .op(post_t::<SelectModelResponse>("/v1/deepseek/select", "DeepSeekのモデルを選択", "models").body::<SelectDeepseekModelRequest>())
        .op(get_t::<crate::hardware::Recommendation>("/v1/recommend", "環境に合う推奨モデル", "models").query::<RecommendQuery>())
        .op(post_t::<RecommendAndDownloadResponse>("/v1/recommend-and-download", "推奨モデルを取得", "models"))
        .op(post_t::<StepModelResponse>("/v1/download-larger", "より大きいモデルを取得", "models"))
        .op(post_t::<StepModelResponse>("/v1/download-smaller", "より小さいモデルを取得", "models"));
    // --- 検索・提供元の設定 ---
    let s = s
        .op(post_t::<GoogleSearchSettingsStatusResponse>("/v1/settings/google-search", "Google検索の設定", "settings").body::<GoogleSearchSettingsRequest>())
        .op(post_t::<GithubSearchSettingsStatusResponse>("/v1/settings/github-search", "GitHub検索の設定", "settings").body::<GithubSearchSettingsRequest>())
        .op(post_t::<YoutubeSearchSettingsStatusResponse>("/v1/settings/youtube-search", "YouTube検索の設定", "settings").body::<YoutubeSearchSettingsRequest>())
        .op(post_t::<ChatProviderSettingsStatusResponse>("/v1/settings/chat-providers", "チャット提供元の設定", "settings").body::<ChatProviderSettingsRequest>())
        .op(post_t::<crate::provider_priority::PriorityStatus>("/v1/settings/provider-priority", "提供元の優先順位", "settings").body::<ProviderPrioritySettingsRequest>())
        .op(get_t::<GoogleSearchSettingsStatusResponse>("/v1/settings/google-search", "Google検索の設定状況", "settings"))
        .op(Op::delete("/v1/settings/google-search").summary("Google検索の設定を消す").tag("settings").public().ok::<GoogleSearchSettingsStatusResponse>("OK"))
        .op(get_t::<GithubSearchSettingsStatusResponse>("/v1/settings/github-search", "GitHub検索の設定状況", "settings"))
        .op(Op::delete("/v1/settings/github-search").summary("GitHub検索の設定を消す").tag("settings").public().ok::<GithubSearchSettingsStatusResponse>("OK"))
        .op(get_t::<YoutubeSearchSettingsStatusResponse>("/v1/settings/youtube-search", "YouTube検索の設定状況", "settings"))
        .op(Op::delete("/v1/settings/youtube-search").summary("YouTube検索の設定を消す").tag("settings").public().ok::<YoutubeSearchSettingsStatusResponse>("OK"))
        .op(get_t::<ChatProviderSettingsStatusResponse>("/v1/settings/chat-providers", "チャット提供元の設定状況", "settings"))
        .op(Op::delete("/v1/settings/chat-providers").summary("チャット提供元の設定を消す").tag("settings").public().ok::<ChatProviderSettingsStatusResponse>("OK"))
        .op(get_t::<crate::provider_priority::PriorityStatus>("/v1/settings/provider-priority", "提供元の優先順位の状況", "settings"))
        .op(Op::delete("/v1/settings/provider-priority").summary("提供元の優先順位を初期に戻す").tag("settings").public().ok::<crate::provider_priority::PriorityStatus>("OK"))
        .op(post_t::<ChatProviderCompleteResponse>("/v1/chat-providers/complete", "提供元1つで補完", "providers").body::<ChatProviderCompleteRequest>())
        .op(post_t::<ChatProviderCompleteMultiResponse>("/v1/chat-providers/complete-multi", "複数の提供元で補完", "providers").body::<ChatProviderCompleteMultiRequest>())
        .op(post_t::<ChatProviderCompletePriorityResponse>("/v1/chat-providers/complete-priority", "優先順位つきで補完", "providers").body::<ChatProviderCompletePriorityRequest>())
        .op(get_t::<crate::chat_providers::HybridStatus>("/v1/chat-providers/active", "有効なチャット提供元", "providers"))
        .op(post("/v1/search/raw", "検索の生結果", "search").body::<SearchRawRequest>())
        .op(get("/v1/media-search", "画像・動画の検索", "search").query::<MediaSearchQuery>())
        .op(get_t::<crate::google_search_fallback_health::FallbackHealth>("/v1/search/fallback-status", "検索フォールバックの状態", "search"));
    // --- ニュース・知識・地理 ---
    let s = s
        .op(post_t::<crate::news_geo::NewsDb>("/v1/news/refresh", "ニュースを更新", "news"))
        .op(get_t::<crate::news_geo::NewsDb>("/v1/news/latest", "最新ニュース", "news"))
        .op(get_t::<crate::news_geo::NewsDb>("/v1/news/for", "国別ニュース", "news").query::<NewsForQuery>())
        .op(post_t::<PruneResponse>("/v1/news/prune-archive", "ニュースアーカイブの整理", "news"))
        .op(post_t::<crate::news_geo::AiNewsDb>("/v1/news/ai-refresh", "AIニュースを更新", "news"))
        .op(get_t::<crate::news_geo::AiNewsDb>("/v1/news/ai-latest", "最新のAIニュース", "news"))
        .op(post_t::<crate::knowledge::Entry>("/v1/knowledge/add", "知識を追加", "knowledge").body::<crate::knowledge::AddRequest>())
        .op(get_t::<KnowledgeItems>("/v1/knowledge/search", "知識を検索", "knowledge").query::<KnowledgeSearchQuery>())
        .op(get_t::<crate::geo_content::GeoRandomResponse>("/v1/geo/random", "ランダムな地点", "geo"))
        .op(get_t::<crate::geo_content::FujiInfoResponse>("/v1/geo/fuji", "富士山の情報", "geo"))
        .op(post_t::<GeoToursResponse>("/v1/geo/tours", "ツアーの提案", "geo").body::<GeoToursRequest>())
        .op(post_t::<crate::geo_content::GeoLookupResponse>("/v1/geo/lookup", "地点の検索", "geo").body::<GeoLookupRequest>())
        .op(post_t::<ReferralsCheckResponse>("/v1/referrals/check", "紹介の確認", "geo").body::<ReferralsCheckRequest>());
    // --- 運用 ---
    s.op(get("/", "トップ画面(HTML)", "system"))
        .op(get("/healthz", "生存確認", "system"))
        .op(get_t::<RuntimeInfoResponse>("/v1/runtime", "実行基盤の情報", "system"))
        .op(get("/v1/accelerators", "利用できるアクセラレータ", "system"))
        .op(get_t::<crate::idle_background_fold::FoldProgress>("/v1/background-fold/status", "バックグラウンド折り畳みの状態", "system"))
        .op(get_t::<crate::phone_task::PhoneTask>("/v1/background-fold/task", "バックグラウンド折り畳みのタスク取得", "system"))
        .op(post_t::<crate::phone_task::TaskResultRecord>("/v1/background-fold/task-result", "バックグラウンド折り畳みの結果送信", "system").body::<crate::phone_task::TaskResultRequest>())
        .op(post_t::<OkText>("/admin/tenants", "テナントを登録(管理)", "admin").body::<crate::tenants::TenantInfo>())
        .op(get_t::<Vec<crate::tenants::TenantInfo>>("/admin/tenants", "テナント一覧(管理)", "admin"))
        .op(Op::delete("/admin/tenants/:host").summary("テナントを削除(管理)").tag("admin").public().ok::<Envelope>("OK"))
}

#[cfg(test)]
mod tests {
    /// 宣言と、`.at("/...")`で登録された実ルートの食い違いを検出する
    /// (main.rsのソースを走査。少なくとも1メソッドの宣言が無いパスを失敗にする)。
    #[test]
    fn spec_is_lint_clean_and_covers_registered_routes() {
        let s = super::spec();
        assert!(s.lint().is_empty(), "{:?}", s.lint());
        let mut undeclared = Vec::new();
        let mut rest = include_str!("main.rs");
        while let Some(i) = rest.find(".at(") {
            rest = &rest[i + 4..];
            if let Some(p) = rest.trim_start().strip_prefix('"') {
                if let Some(end) = p.find('"') {
                    let path = &p[..end];
                    let none = ["GET", "POST", "DELETE"].iter().all(|m| !s.undocumented([(*m, path)]).is_empty());
                    if path.starts_with('/') && none {
                        undeclared.push(path.to_string());
                    }
                }
            }
        }
        assert!(undeclared.is_empty(), "未宣言のルート: {undeclared:?}");
    }
}

// ---- ハンドラが`json!`で組み立てる形を写した型(返却形を変えたらここも直す) ----

#[derive(Serialize, JsonSchema)]
struct RerankResponse {
    scores: Vec<f32>,
    engine: String,
}
#[derive(Serialize, JsonSchema)]
struct PruneResponse {
    archived_countries: Vec<String>,
}
#[derive(Serialize, JsonSchema)]
struct KnowledgeItems {
    count: u64,
    items: Vec<crate::knowledge::Entry>,
}
#[derive(Serialize, JsonSchema)]
struct CharsetResponse {
    chars: String,
}
#[derive(Serialize, JsonSchema)]
struct OcrPrepareResponse {
    languages: Vec<String>,
    quality: String,
    fetched_languages: Vec<String>,
}

/// プレーンテキスト`ok`を返すエンドポイントの、JSON表現上の代用。
type OkText = String;
