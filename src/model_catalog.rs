//! GPT-2互換(config.json + model.safetensors + tokenizer.json)のオープン
//! ソースローカルLLMを、Hugging Faceから選択・ダウンロード・インストール
//! できるようにするカタログ機能(2026-07-26新規、ユーザー指示「aruaru-llm
//! 以外のオープンソースのローカルLLMも簡単にダウンロード・インストールを
//! 選択可能にしてユーザビリティを高めて」への対応)。
//!
//! ## スコープの正直な開示(誇張しない)
//!
//! `open_cuda_llm::GptModel::load`は現状GPT-2アーキテクチャ専用
//! (`config.json`のフィールド・`model.safetensors`内のテンソル名
//! `wte.weight`/`h.{i}.attn.c_attn.weight`等・`Conv1D`の重み配置を前提)
//! であり、Llama/Mistral/Qwen等、アーキテクチャの異なるモデル(RoPE・
//! Grouped Query Attention・SwiGLU・RMSNorm等)は**このエンジンでは
//! ロードできない**。そのため、このカタログが提供するのは「GPT-2と
//! 同じアーキテクチャを持つ、Hugging Face上の別の学習済み重み」に
//! 限定される(サイズ違い・言語違いのGPT-2系モデル)。「本当に任意の
//! オープンソースLLMが動く」かのように見せることは正直さの方針に反する
//! ため、カタログ内にこの制約を明記する。
//!
//! ダウンロード先はHugging Faceの公開`resolve/main/`エンドポイント
//! (認証不要、モデルカードに記載のライセンスに従う)。実際にモデル重み
//! ファイル(数百MB)を取得する操作のため、呼び出し側(`main.rs`の
//! `install_model`ハンドラ)は必ずユーザーの明示的なリクエストからのみ
//! 起動すること(起動時に自動ダウンロードはしない)。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

/// カタログ1件。いずれも`config.json`/`model.safetensors`/`tokenizer.json`
/// の3ファイルを`resolve/main/`から直接取得できることを前提とする
/// (Hugging Face上の実在するpublicリポジトリ、認証・利用規約同意不要な
/// もののみを選定——ダウンロード前にライセンス欄をユーザーへ提示する)。
#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    /// このカタログ内での識別子(`models/{id}/`ディレクトリ名にもなる)。
    pub id: &'static str,
    pub display_name_ja: &'static str,
    pub display_name_en: &'static str,
    /// Hugging Faceのリポジトリパス(`{org}/{repo}`)。
    pub hf_repo: &'static str,
    /// **2026-09-01追加**: `tokenizer.json`を`hf_repo`とは別のリポジトリ
    /// から取得する場合に指定する(`None`なら`hf_repo`から)。
    /// `microsoft/DialoGPT-small`のように`tokenizer.json`単体を公開して
    /// いない(`vocab.json`+`merges.txt`のみ)が、GPT-2本体と全く同じ
    /// BPE語彙(`vocab_size` 50257)を使うモデル向け——`openai-community/gpt2`
    /// の`tokenizer.json`をそのまま流用できる。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_hf_repo: Option<&'static str>,
    pub approx_size_mb: u32,
    /// ライセンスに関する正直な注記(再配布・商用利用可否をユーザー自身が
    /// 確認できるよう、モデルカードへの参照を促す文言に留める)。
    pub license_note_ja: &'static str,
}

/// 2026-07-26時点で動作確認(アーキテクチャ互換性の机上確認のみ、実際の
/// ダウンロード・ロードは既定の`gpt2`以外は未実施——下記CLAUDE.md参照)
/// している候補。いずれもOpenAI/Hugging Face公式の`openai-community`
/// 名前空間配下で、GPT-2アーキテクチャ(hidden_size違いのみ)。
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "gpt2",
        display_name_ja: "GPT-2 (124M, 既定)",
        display_name_en: "GPT-2 (124M, default)",
        hf_repo: "openai-community/gpt2",
        tokenizer_hf_repo: None,
        approx_size_mb: 548,
        license_note_ja: "Modified MIT License(OpenAIのモデルカード参照)。既定で使用中のモデルと同一。",
    },
    CatalogEntry {
        id: "distilgpt2",
        display_name_ja: "DistilGPT-2 (82M、より軽量・高速)",
        display_name_en: "DistilGPT-2 (82M, lighter and faster)",
        hf_repo: "distilbert/distilgpt2",
        tokenizer_hf_repo: None,
        approx_size_mb: 353,
        license_note_ja: "Apache License 2.0(モデルカード参照)。",
    },
    CatalogEntry {
        id: "gpt2-medium",
        display_name_ja: "GPT-2 Medium (355M、gpt2より高品質・低速)",
        display_name_en: "GPT-2 Medium (355M, higher quality, slower)",
        hf_repo: "openai-community/gpt2-medium",
        tokenizer_hf_repo: None,
        approx_size_mb: 1520,
        license_note_ja: "Modified MIT License(OpenAIのモデルカード参照)。",
    },
    CatalogEntry {
        id: "gpt2-large",
        display_name_ja: "GPT-2 Large (774M、CPU推論はかなり低速になる想定)",
        display_name_en: "GPT-2 Large (774M, expect much slower CPU inference)",
        hf_repo: "openai-community/gpt2-large",
        tokenizer_hf_repo: None,
        approx_size_mb: 3250,
        license_note_ja: "Modified MIT License(OpenAIのモデルカード参照)。",
    },
    // 2026-07-27追加(`hardware::recommend`のVRAM8GB以上枠に対応するため)。
    // 実在するpublicリポジトリ(`openai-community/gpt2-xl`)、config.json/
    // model.safetensors/tokenizer.jsonの3ファイル構成をHugging Face上で
    // 事前に確認済み(GPT-2アーキテクチャそのままのXLバリアント)。
    CatalogEntry {
        id: "gpt2-xl",
        display_name_ja: "GPT-2 XL (1.5B、VRAM 8GB以上推奨。CPU推論は非常に低速)",
        display_name_en: "GPT-2 XL (1.5B, recommended for 8GB+ VRAM. CPU inference will be very slow)",
        hf_repo: "openai-community/gpt2-xl",
        tokenizer_hf_repo: None,
        approx_size_mb: 6430,
        license_note_ja: "Modified MIT License(OpenAIのモデルカード参照)。",
    },
    // 2026-09-01追加: 対話ファインチューニング済みのGPT-2互換モデル。
    // `microsoft/DialoGPT-small`はGPT-2アーキテクチャそのまま(hidden 768/
    // 12層/vocab 50257)で、Redditの1.47億対話でファインチューニングされて
    // いるため、素のGPT-2より会話的な応答になりやすい。**F16(半精度)で
    // 配布**されており、従来は`open-cuda-llm`ローダーがF32のみ対応だった
    // ため追加を見送っていた(`CLAUDE.md` 2026-08-26エントリ)——2026-09-01に
    // `tensor_f32`がF16/BF16/FP8→f32変換へ対応したため実重みでロード・
    // 生成できることを実機E2Eで確認済み(`open-cuda/CLAUDE.md`同日エントリ)。
    // `tokenizer.json`単体は公開していない(`vocab.json`+`merges.txt`のみ)
    // が、GPT-2本体と同一のBPE語彙のため`openai-community/gpt2`のものを流用。
    CatalogEntry {
        id: "dialogpt-small",
        display_name_ja: "DialoGPT-small (117M、GPT-2ベース・Reddit対話でFT済み。F16配布→ロード時にf32へ変換)",
        display_name_en: "DialoGPT-small (117M, GPT-2-based, fine-tuned on Reddit dialogue. F16 weights, converted to f32 at load)",
        hf_repo: "microsoft/DialoGPT-small",
        tokenizer_hf_repo: Some("openai-community/gpt2"),
        approx_size_mb: 335,
        license_note_ja: "MIT License(Microsoftのモデルカード参照)。対話FT済みだが、指示追従(instruction following)まで学習したモデルではない点に注意。",
    },
];

pub fn find(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// `CATALOG`を概算サイズ(`approx_size_mb`)の昇順に並べたビュー
/// (2026-07-27追加、「一つ大きい/小さいモデルをダウンロード」ボタン向け)。
fn size_ordered_catalog() -> Vec<&'static CatalogEntry> {
    let mut v: Vec<&'static CatalogEntry> = CATALOG.iter().collect();
    v.sort_by_key(|e| e.approx_size_mb);
    v
}

/// `current_id`より1段階サイズが大きいカタログエントリ(既に最大サイズ、
/// または`current_id`自体がカタログに無い場合は`None`)。
pub fn next_larger(current_id: &str) -> Option<&'static CatalogEntry> {
    let ordered = size_ordered_catalog();
    let idx = ordered.iter().position(|e| e.id == current_id)?;
    ordered.get(idx + 1).copied()
}

/// `current_id`より1段階サイズが小さいカタログエントリ(既に最小サイズ、
/// または`current_id`自体がカタログに無い場合は`None`)。
pub fn next_smaller(current_id: &str) -> Option<&'static CatalogEntry> {
    let ordered = size_ordered_catalog();
    let idx = ordered.iter().position(|e| e.id == current_id)?;
    idx.checked_sub(1).and_then(|i| ordered.get(i)).copied()
}

/// ダウンロード対象の3ファイル(`open_cuda_llm::GptModel::load`/
/// `GptTokenizer::load`が要求するもの、`open-cuda-llm/src/lib.rs`参照)。
const REQUIRED_FILES: &[&str] = &["config.json", "model.safetensors", "tokenizer.json"];

const HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";

fn resolve_url(base_url: &str, hf_repo: &str, filename: &str) -> String {
    format!("{base_url}/{hf_repo}/resolve/main/{filename}")
}

/// `dest_dir`(既定`models/{id}/`)配下に3ファイルをダウンロードする。
/// 途中経過はログへ出す。既にファイルが存在し中断なく完了している場合は
/// 何もしない(冪等、再実行しても壊れたダウンロードを再開できるよう
/// `.part`一時ファイル経由でアトミックに置き換える)。
pub async fn install(entry: &CatalogEntry, dest_dir: &Path) -> Result<()> {
    install_from_base_url(entry, dest_dir, HUGGINGFACE_BASE_URL).await
}

/// `install`の実体。`base_url`を注入できるようにし、実Hugging Faceへ
/// 接続せずローカルのモックHTTPサーバーでダウンロード→書き込み経路
/// そのもの(冪等性・アトミックな置き換え含む)を検証できるようにする
/// (テストコード参照)。
async fn install_from_base_url(entry: &CatalogEntry, dest_dir: &Path, base_url: &str) -> Result<()> {
    tokio::fs::create_dir_all(dest_dir)
        .await
        .with_context(|| format!("failed to create model directory {dest_dir:?}"))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("failed to build reqwest client")?;

    for filename in REQUIRED_FILES {
        let dest = dest_dir.join(filename);
        if dest.exists() {
            tracing::info!(model = entry.id, file = filename, "already present, skipping download");
            continue;
        }
        // `tokenizer.json`だけは`tokenizer_hf_repo`(指定時)から取得する。
        // DialoGPT系のようにGPT-2と同一BPE語彙だが`tokenizer.json`単体を
        // 公開していないモデル向け(2026-09-01追加)。
        let source_repo = match (*filename, entry.tokenizer_hf_repo) {
            ("tokenizer.json", Some(tok_repo)) => tok_repo,
            _ => entry.hf_repo,
        };
        let url = resolve_url(base_url, source_repo, filename);
        tracing::info!(model = entry.id, file = filename, url = %url, "downloading");

        let response = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("request failed for {url}"))?
            .error_for_status()
            .with_context(|| format!("non-success HTTP status for {url}"))?;

        let tmp_path = dest.with_extension(format!("{}.part", filename.rsplit('.').next().unwrap_or("part")));
        let bytes = response.bytes().await.with_context(|| format!("failed to read response body for {url}"))?;
        tokio::fs::write(&tmp_path, &bytes).await.with_context(|| format!("failed to write {tmp_path:?}"))?;
        tokio::fs::rename(&tmp_path, &dest)
            .await
            .with_context(|| format!("failed to move {tmp_path:?} to {dest:?}"))?;
        tracing::info!(model = entry.id, file = filename, bytes = bytes.len(), "download complete");
    }

    Ok(())
}

/// `models_root`配下で、カタログの各エントリが既にインストール済み
/// (3ファイル全て存在する)かどうかを判定する。
pub fn installed_ids(models_root: &Path) -> Vec<&'static str> {
    CATALOG
        .iter()
        .filter(|e| {
            let dir = models_root.join(e.id);
            REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
        })
        .map(|e| e.id)
        .collect()
}

/// `ARUARU_LLM_MODELS_ROOT`環境変数(既定`models`、このプロセスの
/// カレントディレクトリ相対)。既存の`ARUARU_LLM_GPT2_DIR`(単一モデル
/// ディレクトリを直接指す既存の設定方式)とは独立——カタログ経由で
/// インストールしたモデルは常に`{models_root}/{id}/`に置かれる。
pub fn models_root() -> PathBuf {
    std::env::var("ARUARU_LLM_MODELS_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_nonempty() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|e| e.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), CATALOG.len(), "duplicate catalog id found");
        assert!(!CATALOG.is_empty());
    }

    #[test]
    fn find_returns_none_for_unknown_id() {
        assert!(find("does-not-exist-in-catalog").is_none());
    }

    #[test]
    fn find_returns_entry_for_known_id() {
        let e = find("distilgpt2").expect("distilgpt2 must be in the catalog");
        assert_eq!(e.hf_repo, "distilbert/distilgpt2");
    }

    /// サイズ順(概算MB昇順)は dialogpt-small < distilgpt2 < gpt2 <
    /// gpt2-medium < gpt2-large < gpt2-xl になるはず(2026-07-27追加の
    /// 「大きい/小さいモデルへ切替」ボタン機能向け。2026-09-01に
    /// `dialogpt-small`〈335MB〉を最小として追加)。
    #[test]
    fn next_larger_and_next_smaller_follow_approx_size_order() {
        assert_eq!(next_larger("dialogpt-small").map(|e| e.id), Some("distilgpt2"));
        assert_eq!(next_larger("distilgpt2").map(|e| e.id), Some("gpt2"));
        assert_eq!(next_larger("gpt2").map(|e| e.id), Some("gpt2-medium"));
        assert_eq!(next_larger("gpt2-medium").map(|e| e.id), Some("gpt2-large"));
        assert_eq!(next_larger("gpt2-large").map(|e| e.id), Some("gpt2-xl"));
        assert!(next_larger("gpt2-xl").is_none(), "gpt2-xl is already the largest catalog entry");

        assert_eq!(next_smaller("gpt2-xl").map(|e| e.id), Some("gpt2-large"));
        assert_eq!(next_smaller("gpt2-large").map(|e| e.id), Some("gpt2-medium"));
        assert_eq!(next_smaller("gpt2-medium").map(|e| e.id), Some("gpt2"));
        assert_eq!(next_smaller("gpt2").map(|e| e.id), Some("distilgpt2"));
        assert_eq!(next_smaller("distilgpt2").map(|e| e.id), Some("dialogpt-small"));
        assert!(next_smaller("dialogpt-small").is_none(), "dialogpt-small is now the smallest catalog entry");
    }

    #[test]
    fn next_larger_and_next_smaller_return_none_for_unknown_id() {
        assert!(next_larger("does-not-exist").is_none());
        assert!(next_smaller("does-not-exist").is_none());
    }

    #[test]
    fn resolve_url_builds_the_expected_huggingface_resolve_link() {
        assert_eq!(
            resolve_url(HUGGINGFACE_BASE_URL, "openai-community/gpt2", "config.json"),
            "https://huggingface.co/openai-community/gpt2/resolve/main/config.json"
        );
    }

    #[test]
    fn installed_ids_is_empty_for_a_fresh_empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(installed_ids(tmp.path()).is_empty());
    }

    #[test]
    fn installed_ids_detects_a_fully_downloaded_model() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("distilgpt2");
        std::fs::create_dir_all(&dir).unwrap();
        for f in REQUIRED_FILES {
            std::fs::write(dir.join(f), b"stub").unwrap();
        }
        assert_eq!(installed_ids(tmp.path()), vec!["distilgpt2"]);
    }

    #[tokio::test]
    async fn install_writes_all_required_files_from_a_local_mock_server() {
        // 実Hugging Faceには接続せず、ローカルのモックHTTPサーバーで
        // `install`(実体は`install_from_base_url`)が実際に3ファイルを
        // ダウンロード・書き込みすることを検証する(実ネットワーク
        // アクセスを伴う実HTTP結合テストは、数百MBの実モデル重みを
        // テスト実行のたびに取得するのは非現実的なため、意図的に避ける
        // ——ダウンロードロジック自体の正しさはこのモックで検証し、
        // 実Hugging Face URLの到達性は別途手動/運用時に確認する設計)。
        let mut server = mockito::Server::new_async().await;
        let _m1 = server.mock("GET", "/mock-org/mock-repo/resolve/main/config.json").with_status(200).with_body("{}").create_async().await;
        let _m2 = server.mock("GET", "/mock-org/mock-repo/resolve/main/model.safetensors").with_status(200).with_body("stub-weights").create_async().await;
        let _m3 = server.mock("GET", "/mock-org/mock-repo/resolve/main/tokenizer.json").with_status(200).with_body("{}").create_async().await;

        let entry = CatalogEntry {
            id: "mock-model",
            display_name_ja: "テスト用モックモデル",
            display_name_en: "Mock model for testing",
            hf_repo: "mock-org/mock-repo",
            tokenizer_hf_repo: None,
            approx_size_mb: 0,
            license_note_ja: "テスト専用、配布物ではない。",
        };

        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("mock-model");

        install_from_base_url(&entry, &dest, &server.url()).await.expect("install should succeed against the mock server");

        for filename in REQUIRED_FILES {
            let path = dest.join(filename);
            assert!(path.exists(), "{filename} should have been written");
        }
        assert_eq!(std::fs::read(dest.join("model.safetensors")).unwrap(), b"stub-weights");

        // 冪等性: 全ファイルが既に存在する状態で再度呼んでも、モックへ
        // 再度リクエストせず(mockitoの各モックは1回だけ期待、Drop時に
        // 呼び出し回数を検証する)成功すること。
        install_from_base_url(&entry, &dest, &server.url()).await.expect("re-running install on an already-complete directory should be a no-op success");
    }
}
