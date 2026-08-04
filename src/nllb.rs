//! オープンソース専用翻訳モデル(M2M100、`rust-bert`crate経由)による
//! 高品質翻訳(2026-08-04追加、ユーザー指示「翻訳精度が低ければ
//! オープンソースの翻訳システムを組み込んで」)。
//!
//! `POST /v1/translate`が従来使っていたGPT-2流用実装(`generation.rs`)は
//! 実HTTP検証の結果、指示追従学習を一切受けていないため実際の翻訳文を
//! 生成できず実用に耐えないと判明した(2026-08-04 CLAUDE.md HANDOFF
//! 参照)。この既知の限界を解消するため、専用に事前学習・ファイン
//! チューニングされた翻訳モデル(M2M100、100言語間の直接翻訳に対応、
//! Facebook/Metaによるオープンソース〈MIT〉モデル)を`rust-bert`経由で
//! 利用する。
//!
//! **正直な開示(アーキテクチャ上の妥協)**: `rust-bert`は`tch`
//! (libtorch、PyTorchのC++ライブラリ)への依存が必須であり、このリポジトリ
//! ・`open-cuda`エコシステムが他の全モデル(GPT-2・BERT・Whisper相当)で
//! 貫いてきた「手作りRust実装+safetensors直接ロード、重量級MLフレーム
//! ワーク非依存」という方針からは意図的に外れる。この妥協は
//! `nllb-translate` Cargo feature(既定オフ)の背後に隔離し、未指定時は
//! 既存のGPT-2流用実装がそのまま動作する(この変更による既定ビルドへの
//! 影響はゼロ)。将来、翻訳専用の軽量Rustネイティブ実装(NLLB-200
//! distilled等をsafetensorsから手作りロード)へ置き換える余地は残して
//! ある——今回はまず「実際に機能する翻訳」を素早く提供することを優先した
//! 判断。

#[cfg(feature = "nllb-translate")]
use rust_bert::pipelines::translation::{Language, TranslationModel, TranslationModelBuilder};
#[cfg(feature = "nllb-translate")]
use std::sync::Mutex;

/// M2M100モデルの遅延初期化+スレッド間共有(初回リクエストでのみモデルを
/// ロードする、既存の`generation.rs`のGPT-2重みロードと同じ「初回起動
/// コストを許容しウォームアップ後は高速」という設計方針を踏襲)。
#[cfg(feature = "nllb-translate")]
static MODEL: std::sync::OnceLock<Mutex<Result<TranslationModel, String>>> = std::sync::OnceLock::new();

#[cfg(feature = "nllb-translate")]
fn parse_language(name: &str) -> Option<Language> {
    // M2M100がサポートする100言語のうち、実用上よく使われる言語名の
    // 表記ゆれ(英語名・小文字化)を吸収する簡易マッピング(全100言語の
    // 網羅は行わない、正直な開示——`rust_bert::pipelines::translation::
    // Language`が持つ既知のvariant名と一致しない場合は`None`を返し、
    // 呼び出し側で明確なエラーを返す)。
    match name.to_ascii_lowercase().as_str() {
        "english" | "en" => Some(Language::English),
        "japanese" | "ja" | "jp" => Some(Language::Japanese),
        "french" | "fr" => Some(Language::French),
        "german" | "de" => Some(Language::German),
        "spanish" | "es" => Some(Language::Spanish),
        "chinese" | "zh" | "chinese_simplified" => Some(Language::ChineseMandarin),
        "korean" | "ko" => Some(Language::Korean),
        "russian" | "ru" => Some(Language::Russian),
        "italian" | "it" => Some(Language::Italian),
        "arabic" | "ar" => Some(Language::Arabic),
        _ => None,
    }
}

/// M2M100モデルで実際に翻訳する。モデル未ロードなら初回呼び出し時に
/// ロードする(ダウンロード先: `rust-bert`既定のキャッシュディレクトリ、
/// 初回はモデル重みのダウンロードが発生するため数十秒〜数分かかりうる)。
#[cfg(feature = "nllb-translate")]
pub fn translate_with_nllb(text: &str, target_lang: &str, source_lang: Option<&str>) -> Result<String, String> {
    let target = parse_language(target_lang).ok_or_else(|| format!("unsupported target_lang for nllb-translate: {target_lang}"))?;
    let source = source_lang.and_then(parse_language);

    let cell = MODEL.get_or_init(|| {
        let model = TranslationModelBuilder::new().with_source_languages(vec![]).with_target_languages(vec![]).create_model();
        Mutex::new(model.map_err(|e| format!("failed to load M2M100 translation model: {e}")))
    });
    let guard = cell.lock().map_err(|_| "translation model lock poisoned".to_string())?;
    let model = guard.as_ref().map_err(|e| e.clone())?;
    let source_slice = source.map(|s| vec![s]).unwrap_or_default();
    let output = model.translate(&[text], source_slice.first().copied(), target).map_err(|e| format!("translation failed: {e}"))?;
    output.into_iter().next().ok_or_else(|| "translation model returned no output".to_string())
}

/// `nllb-translate` feature無効時のフォールバック(常にエラーを返し、
/// 呼び出し側〈`main.rs::translate`〉が既存のGPT-2流用実装へフォール
/// バックする)。
#[cfg(not(feature = "nllb-translate"))]
pub fn translate_with_nllb(_text: &str, _target_lang: &str, _source_lang: Option<&str>) -> Result<String, String> {
    Err("nllb-translate feature not enabled at build time".to_string())
}

/// このビルドでM2M100翻訳が利用可能かどうか(featureフラグの状態を
/// 実行時に問い合わせるためのヘルパー、レスポンスの`engine`フィールドの
/// 出し分けに使う)。
pub fn is_available() -> bool {
    cfg!(feature = "nllb-translate")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_reflects_the_compiled_feature_flag() {
        // このテスト自体は`nllb-translate`feature無しでビルドされるのが
        // 既定のCI/開発フローのため、既定ビルドでは`false`を期待する
        // (feature有効ビルド時にこのテストを実行する場合は`true`になる、
        // 環境依存のテストであることを正直に明記)。
        assert_eq!(is_available(), cfg!(feature = "nllb-translate"));
    }

    #[cfg(not(feature = "nllb-translate"))]
    #[test]
    fn translate_with_nllb_reports_unavailable_when_feature_disabled() {
        let result = translate_with_nllb("hello", "japanese", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not enabled"));
    }
}
