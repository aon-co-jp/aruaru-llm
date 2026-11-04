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
    Op::post(path).summary(summary).tag(tag).public().ok::<Envelope>("OK")
}

fn get_t<T: JsonSchema>(path: &str, summary: &str, tag: &str) -> Op {
    Op::post(path).summary(summary).tag(tag).public().ok::<T>("OK")
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
        .op(post_t::<ChatResponse>("/aruaru_llm.v1.Llm/Chat", "チャット(会話)", "generate").body::<ChatRequest>())
        .op(post_t::<GenerateResponse>("/aruaru_llm.v1.Llm/Generate", "文章生成", "generate").body::<GenerateRequest>())
        .op(post_t::<GenerateSpeculativeResponse>("/aruaru_llm.v1.Llm/GenerateSpeculative", "投機的デコードによる文章生成", "generate").body::<GenerateSpeculativeRequest>())
        .op(post_t::<GenerateWithSearchResponse>("/aruaru_llm.v1.Llm/GenerateWithSearch", "検索結果つきの文章生成", "generate").body::<GenerateWithSearchRequest>())
        .op(post_t::<GenerateQwenResponse>("/aruaru_llm.v1.Llm/GenerateQwen", "Qwenで文章生成", "generate").body::<GenerateQwenRequest>())
        .op(post_t::<GenerateDeepseekResponse>("/aruaru_llm.v1.Llm/GenerateDeepseek", "DeepSeekで文章生成", "generate").body::<GenerateDeepseekRequest>())
        .op(post_t::<TranslateResponse>("/aruaru_llm.v1.Llm/Translate", "翻訳(NLLB)", "generate").body::<TranslateRequest>())
        .op(post_t::<TranscribeResponse>("/aruaru_llm.v1.Llm/Transcribe", "音声の文字起こし", "generate").body::<TranscribeRequest>())
        .op(post_t::<RerankResponse>("/aruaru_llm.v1.Llm/Rerank", "検索結果の再ランク", "generate").body::<RerankRequest>())
        .op(post_t::<crate::persona::PersonaResponse>("/aruaru_llm.v1.Llm/PersonaPrompt", "ペルソナのプロンプトを作る", "generate").body::<crate::persona::PersonaRequest>());
    // --- 分類・セキュリティ ---
    let s = s
        .op(post_t::<ClassifyResponse>("/aruaru_llm.v1.Llm/Classify", "汎用の分類", "classify").body::<ClassifyRequest>())
        .op(post_t::<ClassifySecurityResponse>("/aruaru_llm.v1.Llm/ClassifySecurity", "セキュリティ分類", "classify").body::<ClassifySecurityRequest>())
        .op(post_t::<ClassifyTrafficResponse>("/aruaru_llm.v1.Llm/SecurityClassifyTraffic", "通信の分類(侵入検知)", "classify").body::<ClassifyTrafficRequest>())
        .op(post_t::<ClassifyPiiResponse>("/aruaru_llm.v1.Llm/ClassifyPii", "個人情報の分類", "classify").body::<ClassifyPiiRequest>());
    // --- OCR ---
    let s = s
        .op(post_t::<OcrResponse>("/aruaru_llm.v1.Llm/Ocr", "OCR(画像から文字)", "ocr").body::<OcrRequest>())
        .op(get_t::<OcrLanguagesResponse>("/aruaru_llm.v1.Llm/OcrLanguages", "OCRの対応言語", "ocr"))
        .op(post_t::<OcrPrepareResponse>("/aruaru_llm.v1.Llm/OcrPrepare", "OCRの言語データを準備", "ocr").body::<OcrPrepareRequest>())
        .op(post_t::<OcrCorrectResponse>("/aruaru_llm.v1.Llm/OcrCorrect", "OCR結果の補正", "ocr").body::<OcrCorrectRequest>())
        .op(get_t::<CharsetResponse>("/aruaru_llm.v1.Llm/OcrCharset", "頻出順の文字集合", "ocr").query::<OcrCharsetQuery>())
        .op(post_t::<OcrGlyphRankResponse>("/aruaru_llm.v1.Llm/OcrGlyphRank", "字形による候補の順位付け", "ocr").body::<OcrGlyphRankRequest>())
        .op(post_t::<OcrScanResponse>("/aruaru_llm.v1.Llm/OcrScan", "不自然な文字の走査", "ocr").body::<OcrScanRequest>())
        .op(post_t::<OcrProposeResponse>("/aruaru_llm.v1.Llm/OcrPropose", "候補の提案", "ocr").body::<OcrProposeRequest>())
        .op(post_t::<OcrFillResponse>("/aruaru_llm.v1.Llm/OcrFill", "文脈による補完", "ocr").body::<OcrFillRequest>())
        .op(get("/v1/ocr/fonts/:name", "OCR用フォントの取得", "ocr"));
    // --- モデル管理 ---
    let s = s
        .op(get_t::<CatalogResponse>("/aruaru_llm.v1.Llm/ModelsCatalog", "モデルカタログ", "models"))
        .op(post_t::<crate::cache_optimizer::CacheOptimizationResult>("/aruaru_llm.v1.Llm/ModelsOptimizeCache", "キャッシュ最適化", "models").body::<OptimizeCacheRequest>())
        .op(post_t::<InstallModelResponse>("/aruaru_llm.v1.Llm/ModelsInstall", "モデルを導入", "models").body::<InstallModelRequest>())
        .op(post_t::<SelectModelResponse>("/aruaru_llm.v1.Llm/ModelsSelect", "モデルを選択", "models").body::<SelectModelRequest>())
        .op(post_t::<LayerRedundancyResponse>("/aruaru_llm.v1.Llm/ModelsLayerRedundancy", "層の冗長性を調べる", "models").body::<LayerRedundancyRequest>())
        .op(post_t::<FoldLayersResponse>("/aruaru_llm.v1.Llm/ModelsFoldLayers", "層を畳む", "models").body::<FoldLayersRequest>())
        .op(get_t::<CatalogResponse>("/aruaru_llm.v1.Llm/QwenCatalog", "Qwenのカタログ", "models"))
        .op(post_t::<InstallModelResponse>("/aruaru_llm.v1.Llm/QwenInstall", "Qwenを導入", "models").body::<InstallModelRequest>())
        .op(post_t::<SelectModelResponse>("/aruaru_llm.v1.Llm/QwenSelect", "Qwenを選択", "models").body::<SelectModelRequest>())
        .op(get_t::<DeepseekStatusResponse>("/aruaru_llm.v1.Llm/DeepseekStatus", "DeepSeekの状態", "models"))
        .op(post_t::<SelectModelResponse>("/aruaru_llm.v1.Llm/DeepseekSelect", "DeepSeekのモデルを選択", "models").body::<SelectDeepseekModelRequest>())
        .op(get_t::<crate::hardware::Recommendation>("/aruaru_llm.v1.Llm/Recommend", "環境に合う推奨モデル", "models").query::<RecommendQuery>())
        .op(post_t::<RecommendAndDownloadResponse>("/aruaru_llm.v1.Llm/RecommendAndDownload", "推奨モデルを取得", "models"))
        .op(post_t::<StepModelResponse>("/aruaru_llm.v1.Llm/DownloadLarger", "より大きいモデルを取得", "models"))
        .op(post_t::<StepModelResponse>("/aruaru_llm.v1.Llm/DownloadSmaller", "より小さいモデルを取得", "models"));
    // --- 検索・提供元の設定 ---
    let s = s
        .op(post_t::<GoogleSearchSettingsStatusResponse>("/aruaru_llm.v1.Llm/SetGoogleSearchSettings", "Google検索の設定", "settings").body::<GoogleSearchSettingsRequest>())
        .op(post_t::<GithubSearchSettingsStatusResponse>("/aruaru_llm.v1.Llm/SetGithubSearchSettings", "GitHub検索の設定", "settings").body::<GithubSearchSettingsRequest>())
        .op(post_t::<YoutubeSearchSettingsStatusResponse>("/aruaru_llm.v1.Llm/SetYoutubeSearchSettings", "YouTube検索の設定", "settings").body::<YoutubeSearchSettingsRequest>())
        .op(post_t::<ChatProviderSettingsStatusResponse>("/aruaru_llm.v1.Llm/SetChatProviderSettings", "チャット提供元の設定", "settings").body::<ChatProviderSettingsRequest>())
        .op(post_t::<crate::provider_priority::PriorityStatus>("/aruaru_llm.v1.Llm/SetProviderPrioritySettings", "提供元の優先順位", "settings").body::<ProviderPrioritySettingsRequest>())
        .op(get_t::<GoogleSearchSettingsStatusResponse>("/aruaru_llm.v1.Llm/GetGoogleSearchSettings", "Google検索の設定状況", "settings"))
        .op(Op::post("/aruaru_llm.v1.Llm/ClearGoogleSearchSettings").summary("Google検索の設定を消す").tag("settings").public().ok::<GoogleSearchSettingsStatusResponse>("OK"))
        .op(get_t::<GithubSearchSettingsStatusResponse>("/aruaru_llm.v1.Llm/GetGithubSearchSettings", "GitHub検索の設定状況", "settings"))
        .op(Op::post("/aruaru_llm.v1.Llm/ClearGithubSearchSettings").summary("GitHub検索の設定を消す").tag("settings").public().ok::<GithubSearchSettingsStatusResponse>("OK"))
        .op(get_t::<YoutubeSearchSettingsStatusResponse>("/aruaru_llm.v1.Llm/GetYoutubeSearchSettings", "YouTube検索の設定状況", "settings"))
        .op(Op::post("/aruaru_llm.v1.Llm/ClearYoutubeSearchSettings").summary("YouTube検索の設定を消す").tag("settings").public().ok::<YoutubeSearchSettingsStatusResponse>("OK"))
        .op(get_t::<ChatProviderSettingsStatusResponse>("/aruaru_llm.v1.Llm/GetChatProviderSettings", "チャット提供元の設定状況", "settings"))
        .op(Op::post("/aruaru_llm.v1.Llm/ClearChatProviderSettings").summary("チャット提供元の設定を消す").tag("settings").public().ok::<ChatProviderSettingsStatusResponse>("OK"))
        .op(get_t::<crate::provider_priority::PriorityStatus>("/aruaru_llm.v1.Llm/GetProviderPrioritySettings", "提供元の優先順位の状況", "settings"))
        .op(Op::post("/aruaru_llm.v1.Llm/ResetProviderPrioritySettings").summary("提供元の優先順位を初期に戻す").tag("settings").public().ok::<crate::provider_priority::PriorityStatus>("OK"))
        .op(post_t::<ChatProviderCompleteResponse>("/aruaru_llm.v1.Llm/ChatProvidersComplete", "提供元1つで補完", "providers").body::<ChatProviderCompleteRequest>())
        .op(post_t::<ChatProviderCompleteMultiResponse>("/aruaru_llm.v1.Llm/ChatProvidersCompleteMulti", "複数の提供元で補完", "providers").body::<ChatProviderCompleteMultiRequest>())
        .op(post_t::<ChatProviderCompletePriorityResponse>("/aruaru_llm.v1.Llm/ChatProvidersCompletePriority", "優先順位つきで補完", "providers").body::<ChatProviderCompletePriorityRequest>())
        .op(get_t::<crate::chat_providers::HybridStatus>("/aruaru_llm.v1.Llm/ChatProvidersActive", "有効なチャット提供元", "providers"))
        .op(post("/aruaru_llm.v1.Llm/SearchRaw", "検索の生結果", "search").body::<SearchRawRequest>())
        .op(get("/aruaru_llm.v1.Llm/MediaSearch", "画像・動画の検索", "search").query::<MediaSearchQuery>())
        .op(get_t::<crate::google_search_fallback_health::FallbackHealth>("/aruaru_llm.v1.Llm/SearchFallbackStatus", "検索フォールバックの状態", "search"));
    // --- ニュース・知識・地理 ---
    let s = s
        .op(post_t::<crate::news_geo::NewsDb>("/aruaru_llm.v1.Llm/NewsRefresh", "ニュースを更新", "news"))
        .op(get_t::<crate::news_geo::NewsDb>("/aruaru_llm.v1.Llm/NewsLatest", "最新ニュース", "news"))
        .op(get_t::<crate::news_geo::NewsDb>("/aruaru_llm.v1.Llm/NewsFor", "国別ニュース", "news").query::<NewsForQuery>())
        .op(post_t::<PruneResponse>("/aruaru_llm.v1.Llm/NewsPruneArchive", "ニュースアーカイブの整理", "news"))
        .op(post_t::<crate::news_geo::AiNewsDb>("/aruaru_llm.v1.Llm/NewsAiRefresh", "AIニュースを更新", "news"))
        .op(get_t::<crate::news_geo::AiNewsDb>("/aruaru_llm.v1.Llm/NewsAiLatest", "最新のAIニュース", "news"))
        .op(post_t::<crate::knowledge::Entry>("/aruaru_llm.v1.Llm/KnowledgeAdd", "知識を追加", "knowledge").body::<crate::knowledge::AddRequest>())
        .op(get_t::<KnowledgeItems>("/aruaru_llm.v1.Llm/KnowledgeSearch", "知識を検索", "knowledge").query::<KnowledgeSearchQuery>())
        .op(get_t::<crate::geo_content::GeoRandomResponse>("/aruaru_llm.v1.Llm/GeoRandom", "ランダムな地点", "geo"))
        .op(get_t::<crate::geo_content::FujiInfoResponse>("/aruaru_llm.v1.Llm/GeoFuji", "富士山の情報", "geo"))
        .op(post_t::<GeoToursResponse>("/aruaru_llm.v1.Llm/GeoTours", "ツアーの提案", "geo").body::<GeoToursRequest>())
        .op(post_t::<crate::geo_content::GeoLookupResponse>("/aruaru_llm.v1.Llm/GeoLookup", "地点の検索", "geo").body::<GeoLookupRequest>())
        .op(post_t::<ReferralsCheckResponse>("/aruaru_llm.v1.Llm/ReferralsCheck", "紹介の確認", "geo").body::<ReferralsCheckRequest>());
    // --- 運用 ---
    s.op(Op::get("/").summary("トップ画面(HTML)").tag("system").public().response_text(200, "HTML", "text/html"))
        .op(Op::get("/healthz").summary("生存確認").tag("system").public().response_text(200, "ok", "text/plain"))
        .op(get_t::<RuntimeInfoResponse>("/aruaru_llm.v1.Llm/Runtime", "実行基盤の情報", "system"))
        .op(get("/aruaru_llm.v1.Llm/Accelerators", "利用できるアクセラレータ", "system"))
        .op(get_t::<crate::idle_background_fold::FoldProgress>("/aruaru_llm.v1.Llm/BackgroundFoldStatus", "バックグラウンド折り畳みの状態", "system"))
        .op(get_t::<crate::phone_task::PhoneTask>("/aruaru_llm.v1.Llm/BackgroundFoldTask", "バックグラウンド折り畳みのタスク取得", "system"))
        .op(post_t::<crate::phone_task::TaskResultRecord>("/aruaru_llm.v1.Llm/BackgroundFoldTaskResult", "バックグラウンド折り畳みの結果送信", "system").body::<crate::phone_task::TaskResultRequest>())
        .op(Op::post("/aruaru_llm.v1.Llm/RegisterTenant").summary("テナントを登録(管理)").tag("admin").public().body::<crate::tenants::TenantInfo>().response_text(200, "ok", "text/plain").response::<ErrorBody>(400, "入力不正"))
        .op(get_t::<Vec<crate::tenants::TenantInfo>>("/aruaru_llm.v1.Llm/ListTenants", "テナント一覧(管理)", "admin"))
        .op(Op::post("/aruaru_llm.v1.Llm/RemoveTenant").summary("テナントを削除(管理)").tag("admin").public().response_text(200, "ok", "text/plain"))
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
        let mut rest = include_str!("main.rs");
        while let Some(i) = rest.find("connect::path(\"") {
            rest = &rest[i + 15..];
            if let Some(end) = rest.find('"') {
                let path = format!("/aruaru_llm.v1.Llm/{}", &rest[..end]);
                if !s.undocumented([("POST", path.as_str())]).is_empty() {
                    undeclared.push(path);
                }
            }
        }
        assert!(undeclared.is_empty(), "未宣言のルート: {undeclared:?}");
    }
}

// ---- ハンドラが`json!`で組み立てる形を写した型(返却形を変えたらここも直す) ----

#[derive(Serialize, JsonSchema)]
pub(crate) struct RerankResponse {
    pub(crate) scores: Vec<f32>,
    pub(crate) engine: String,
}
#[derive(Serialize, JsonSchema)]
pub(crate) struct PruneResponse {
    pub(crate) archived_countries: usize,
}
#[derive(Serialize, JsonSchema)]
pub(crate) struct KnowledgeItems {
    pub(crate) count: u64,
    pub(crate) items: Vec<crate::knowledge::Entry>,
}
#[derive(Serialize, JsonSchema)]
pub(crate) struct CharsetResponse {
    pub(crate) chars: String,
}
#[derive(Serialize, JsonSchema)]
pub(crate) struct OcrPrepareResponse {
    pub(crate) languages: Vec<String>,
    pub(crate) quality: String,
    pub(crate) fetched_languages: Vec<String>,
}

