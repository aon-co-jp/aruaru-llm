//! 地理・観光・名物データベース(2026-08-11新設)。
//!
//! ユーザー指示「open-englishの自己紹介トレーニングが『Where are you
//! from? I'm from Australia. I love kangaroo & koala.』程度しか対応
//! できなかったのを、日本全国・都道府県別、アメリカは州別、世界中の
//! 首都名・観光名所・名物料理・お土産のDATABASEを作成して」への対応。
//!
//! ## 正直な開示(最重要)
//!
//! - **DUAL DB**: `aruaru-db`(pgwireプロトコル、実質Postgres互換)に
//!   接続する。`aruaru-db`自身が`DUAL_DATABASE_URL`経由で本物のPostgreSQL
//!   へミラーする冗長化機能(`aruaru-dist::dual_database::
//!   DualDatabaseMirror`)を既に持っているため、このクライアント側は
//!   単一のPostgres接続(`ARUARU_LLM_GEO_DATABASE_URL`)を張るだけでよい
//!   ——DUAL DB化自体を本クレートで再実装する必要はない。
//! - **DB未接続時のフォールバック**: `ARUARU_LLM_GEO_DATABASE_URL`が
//!   未設定、またはaruaru-dbへの接続に失敗した場合、本クレートに
//!   埋め込んだ静的データ(`data/geo_content.json`、`include_str!`で
//!   バイナリへ埋め込み)からそのまま応答する——サービスを止めない設計
//!   (既存の`/v1/chat`のbag-of-wordsフォールバックと同じ思想)。
//!   `source`フィールドで実際にどちらを使ったか(`"aruaru-db"`または
//!   `"embedded-json-fallback"`)を常に正直に返す。
//! - **カバレッジの限界**: 日本47都道府県・米国50州は全件収録済み。
//!   世界の首都は主要60ヶ国分のみ(国連加盟196ヶ国全ては未収録、今後
//!   拡張予定)。誇張せず`data/geo_content.json`の`_disclosure_ja`/
//!   `_disclosure_en`に明記している。
//! - **DB同期(seed)は起動時にベストエフォートで実行**: 接続に成功した
//!   場合のみ`CREATE TABLE IF NOT EXISTS`+`INSERT ... ON CONFLICT DO
//!   NOTHING`でaruaru-dbへ投入する(冪等、既存データを上書きしない)。
//!   このセッションでは実際に稼働中のaruaru-dbインスタンスに対する
//!   接続検証は未実施(環境にaruaru-db実行中のプロセスが無いため)——
//!   ビルド成功・埋め込みJSONフォールバック経路の単体テストまでを
//!   検証済みとし、実DB接続の検証は次回に持ち越すことを正直に記録する。

use std::sync::OnceLock;

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

const EMBEDDED_JSON: &str = include_str!("../data/geo_content.json");

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JapanPrefecture {
    pub name_ja: String,
    pub name_en: String,
    pub landmark_ja: String,
    pub landmark_en: String,
    pub food_ja: String,
    pub food_en: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UsState {
    pub name_en: String,
    pub name_ja: String,
    pub landmark_en: String,
    pub landmark_ja: String,
    pub food_en: String,
    pub food_ja: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorldCapital {
    pub country_en: String,
    pub country_ja: String,
    pub capital_en: String,
    pub capital_ja: String,
    pub landmark_en: String,
    pub landmark_ja: String,
    pub food_en: String,
    pub food_ja: String,
    pub souvenir_en: String,
    pub souvenir_ja: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FujiMountainHut {
    pub name_ja: String,
    pub name_en: String,
    pub station_ja: String,
    pub station_en: String,
    pub season_ja: String,
    pub season_en: String,
    pub phone: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FujiTransportOption {
    pub name_ja: String,
    pub name_en: String,
    pub note_ja: String,
    pub note_en: String,
    pub url: String,
}

/// 登山用品(スキーウェア・ヘルメット・登山靴)のレンタル/購入店。
/// `FujiTransportOption`と構造が同じだが、意味的に別カテゴリのため
/// 別の型として定義する(呼び出し側APIレスポンスのフィールド名を
/// `gear_shops`として明確に分ける)。
pub type FujiGearShop = FujiTransportOption;

#[derive(Debug, Deserialize)]
struct GeoDataset {
    #[serde(rename = "_disclosure_ja")]
    #[allow(dead_code)]
    disclosure_ja: String,
    #[serde(rename = "_disclosure_en")]
    #[allow(dead_code)]
    disclosure_en: String,
    japan_prefectures: Vec<JapanPrefecture>,
    us_states: Vec<UsState>,
    world_capitals: Vec<WorldCapital>,
    fuji_safety_ja: String,
    fuji_safety_en: String,
    fuji_source_ja: String,
    fuji_source_en: String,
    fuji_transport_reservations: Vec<FujiTransportOption>,
    fuji_gear_shops: Vec<FujiGearShop>,
    fuji_mountain_huts: Vec<FujiMountainHut>,
}

fn dataset() -> &'static GeoDataset {
    static DATASET: OnceLock<GeoDataset> = OnceLock::new();
    DATASET.get_or_init(|| {
        // 生文字列のJSONパース自体はRust-JSON(`../Rust-JSON`)の
        // `parse_strict`を使う(ユーザー指示「JSONよりRS-JSONを使って」)。
        // 型付きへの変換(`serde_json::from_value`)はserde側に委ねる
        // ——Rust-JSON自身が`serde_json::Value`を値モデルとして使う設計
        // のため(クレートのモジュールdoc参照)。
        let value = rust_json::parse_strict(EMBEDDED_JSON)
            .expect("data/geo_content.json must be valid strict JSON");
        serde_json::from_value(value)
            .expect("data/geo_content.json must match GeoDataset shape")
    })
}

/// aruaru-db(pgwire)接続文字列。未設定なら埋め込みJSONのみを使う。
fn database_url() -> Option<String> {
    std::env::var("ARUARU_LLM_GEO_DATABASE_URL").ok().filter(|s| !s.trim().is_empty())
}

#[derive(Debug, Clone, Serialize)]
pub struct GeoRandomResponse {
    pub prefecture: JapanPrefecture,
    pub state: UsState,
    pub capital: WorldCapital,
    pub source: String,
    pub disclosure_ja: String,
    pub disclosure_en: String,
}

const DISCLOSURE_JA: &str = "収録範囲: 日本47都道府県+米国50州は全件、世界の首都は主要60ヶ国分(国連加盟196ヶ国全てではない、今後拡張予定)。";
const DISCLOSURE_EN: &str = "Coverage: all 47 Japanese prefectures and 50 US states; world capitals cover ~60 major countries only (not all 196 UN member states yet, expansion planned).";

/// aruaru-dbへの接続を試み、成功すれば`(prefecture, state, capital)`を
/// ランダムに1件ずつ返す。失敗・未設定時は`None`(呼び出し側が埋め込み
/// JSONへフォールバックする)。
async fn try_random_from_db() -> Option<(JapanPrefecture, UsState, WorldCapital)> {
    let url = database_url()?;
    let (client, connection) = tokio_postgres::connect(&url, tokio_postgres::NoTls).await.ok()?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::warn!("aruaru-db geo connection error: {e}");
        }
    });

    let pref_row = client
        .query_opt("SELECT name_ja, name_en, landmark_ja, landmark_en, food_ja, food_en FROM geo_japan_prefectures ORDER BY random() LIMIT 1", &[])
        .await
        .ok()??;
    let state_row = client
        .query_opt("SELECT name_en, name_ja, landmark_en, landmark_ja, food_en, food_ja FROM geo_us_states ORDER BY random() LIMIT 1", &[])
        .await
        .ok()??;
    let capital_row = client
        .query_opt("SELECT country_en, country_ja, capital_en, capital_ja, landmark_en, landmark_ja, food_en, food_ja, souvenir_en, souvenir_ja FROM geo_world_capitals ORDER BY random() LIMIT 1", &[])
        .await
        .ok()??;

    Some((
        JapanPrefecture {
            name_ja: pref_row.get(0),
            name_en: pref_row.get(1),
            landmark_ja: pref_row.get(2),
            landmark_en: pref_row.get(3),
            food_ja: pref_row.get(4),
            food_en: pref_row.get(5),
        },
        UsState {
            name_en: state_row.get(0),
            name_ja: state_row.get(1),
            landmark_en: state_row.get(2),
            landmark_ja: state_row.get(3),
            food_en: state_row.get(4),
            food_ja: state_row.get(5),
        },
        WorldCapital {
            country_en: capital_row.get(0),
            country_ja: capital_row.get(1),
            capital_en: capital_row.get(2),
            capital_ja: capital_row.get(3),
            landmark_en: capital_row.get(4),
            landmark_ja: capital_row.get(5),
            food_en: capital_row.get(6),
            food_ja: capital_row.get(7),
            souvenir_en: capital_row.get(8),
            souvenir_ja: capital_row.get(9),
        },
    ))
}

fn random_from_embedded() -> (JapanPrefecture, UsState, WorldCapital) {
    let d = dataset();
    let mut rng = rand::thread_rng();
    (
        d.japan_prefectures.choose(&mut rng).cloned().expect("japan_prefectures must not be empty"),
        d.us_states.choose(&mut rng).cloned().expect("us_states must not be empty"),
        d.world_capitals.choose(&mut rng).cloned().expect("world_capitals must not be empty"),
    )
}

/// 日本の都道府県1件・米国の州1件・世界の首都1件をランダムに返す
/// (open-englishの自己紹介トレーニング用、`POST /v1/geo/random`が呼ぶ)。
pub async fn random_entry() -> GeoRandomResponse {
    if let Some((prefecture, state, capital)) = try_random_from_db().await {
        return GeoRandomResponse {
            prefecture,
            state,
            capital,
            source: "aruaru-db".to_string(),
            disclosure_ja: DISCLOSURE_JA.to_string(),
            disclosure_en: DISCLOSURE_EN.to_string(),
        };
    }
    let (prefecture, state, capital) = random_from_embedded();
    GeoRandomResponse {
        prefecture,
        state,
        capital,
        source: "embedded-json-fallback".to_string(),
        disclosure_ja: DISCLOSURE_JA.to_string(),
        disclosure_en: DISCLOSURE_EN.to_string(),
    }
}

/// aruaru-dbへテーブル作成+シード投入をベストエフォートで行う
/// (接続できない・未設定なら何もせず正常に戻る、起動を妨げない設計)。
pub async fn seed_database_if_configured() {
    let Some(url) = database_url() else {
        tracing::info!("ARUARU_LLM_GEO_DATABASE_URL not set: geo content will use the embedded JSON dataset only");
        return;
    };
    let (client, connection) = match tokio_postgres::connect(&url, tokio_postgres::NoTls).await {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!("failed to connect to aruaru-db for geo seed: {e} (falling back to embedded JSON dataset)");
            return;
        }
    };
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::warn!("aruaru-db geo seed connection error: {e}");
        }
    });

    let ddl = [
        "CREATE TABLE IF NOT EXISTS geo_japan_prefectures (name_en TEXT PRIMARY KEY, name_ja TEXT, landmark_ja TEXT, landmark_en TEXT, food_ja TEXT, food_en TEXT)",
        "CREATE TABLE IF NOT EXISTS geo_us_states (name_en TEXT PRIMARY KEY, name_ja TEXT, landmark_en TEXT, landmark_ja TEXT, food_en TEXT, food_ja TEXT)",
        "CREATE TABLE IF NOT EXISTS geo_world_capitals (country_en TEXT PRIMARY KEY, country_ja TEXT, capital_en TEXT, capital_ja TEXT, landmark_en TEXT, landmark_ja TEXT, food_en TEXT, food_ja TEXT, souvenir_en TEXT, souvenir_ja TEXT)",
        "CREATE TABLE IF NOT EXISTS geo_fuji_mountain_huts (name_en TEXT PRIMARY KEY, name_ja TEXT, station_ja TEXT, station_en TEXT, season_ja TEXT, season_en TEXT, phone TEXT, url TEXT)",
    ];
    for stmt in ddl {
        if let Err(e) = client.execute(stmt, &[]).await {
            tracing::warn!("aruaru-db geo seed DDL failed ({stmt}): {e}");
            return;
        }
    }

    let d = dataset();
    for p in &d.japan_prefectures {
        let _ = client.execute(
            "INSERT INTO geo_japan_prefectures (name_en, name_ja, landmark_ja, landmark_en, food_ja, food_en) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (name_en) DO NOTHING",
            &[&p.name_en, &p.name_ja, &p.landmark_ja, &p.landmark_en, &p.food_ja, &p.food_en],
        ).await;
    }
    for s in &d.us_states {
        let _ = client.execute(
            "INSERT INTO geo_us_states (name_en, name_ja, landmark_en, landmark_ja, food_en, food_ja) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (name_en) DO NOTHING",
            &[&s.name_en, &s.name_ja, &s.landmark_en, &s.landmark_ja, &s.food_en, &s.food_ja],
        ).await;
    }
    for c in &d.world_capitals {
        let _ = client.execute(
            "INSERT INTO geo_world_capitals (country_en, country_ja, capital_en, capital_ja, landmark_en, landmark_ja, food_en, food_ja, souvenir_en, souvenir_ja) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT (country_en) DO NOTHING",
            &[&c.country_en, &c.country_ja, &c.capital_en, &c.capital_ja, &c.landmark_en, &c.landmark_ja, &c.food_en, &c.food_ja, &c.souvenir_en, &c.souvenir_ja],
        ).await;
    }
    tracing::info!(
        "aruaru-db geo seed complete: {} prefectures, {} states, {} capitals (idempotent upsert)",
        d.japan_prefectures.len(),
        d.us_states.len(),
        d.world_capitals.len()
    );
}

#[derive(Debug, Clone, Serialize)]
pub struct GeoLookupResponse {
    pub found: bool,
    pub capital: Option<WorldCapital>,
    pub source: String,
}

/// 「今度オーストラリアに旅行/出張の予定がある」のような発話向けの
/// 国名検索(2026-08-11新設、ユーザー指示「今度どこどこの国に観光や
/// お仕事で行く予定があるんだけど、の様なフレーズにもDBで対応して」)。
/// **部分一致で検索する**(実機テストで発覚した実バグの修正: 当初は
/// クエリ全体と国名の完全一致のみを見ていたため、"I FROM JAPAN."の
/// ような文全体を渡すと一致しなかった)——`open-english`側は発話文を
/// そのまま渡す設計のため、文中に国名が含まれていれば一致させる。
/// 見つからない場合は`found: false`を正直に返す(黙って無関係な結果を
/// 返さない、既存の「graceful degradation, never silent」方針)。
pub async fn lookup_country(query: &str) -> GeoLookupResponse {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        return GeoLookupResponse { found: false, capital: None, source: "invalid-empty-query".to_string() };
    }

    if let Some(url) = database_url() {
        if let Ok((client, connection)) = tokio_postgres::connect(&url, tokio_postgres::NoTls).await {
            tokio::spawn(async move {
                if let Err(e) = connection.await {
                    tracing::warn!("aruaru-db geo lookup connection error: {e}");
                }
            });
            let pattern = format!("%{needle}%");
            if let Ok(Some(row)) = client
                .query_opt(
                    "SELECT country_en, country_ja, capital_en, capital_ja, landmark_en, landmark_ja, food_en, food_ja, souvenir_en, souvenir_ja FROM geo_world_capitals WHERE lower(country_en) LIKE $1 OR $2 LIKE '%' || country_ja || '%' LIMIT 1",
                    &[&pattern, &query.trim()],
                )
                .await
            {
                return GeoLookupResponse {
                    found: true,
                    capital: Some(WorldCapital {
                        country_en: row.get(0),
                        country_ja: row.get(1),
                        capital_en: row.get(2),
                        capital_ja: row.get(3),
                        landmark_en: row.get(4),
                        landmark_ja: row.get(5),
                        food_en: row.get(6),
                        food_ja: row.get(7),
                        souvenir_en: row.get(8),
                        souvenir_ja: row.get(9),
                    }),
                    source: "aruaru-db".to_string(),
                };
            }
        }
    }

    let d = dataset();
    let trimmed = query.trim();
    let found = d
        .world_capitals
        .iter()
        .find(|c| needle.contains(&c.country_en.to_lowercase()) || trimmed.contains(&c.country_ja))
        .cloned();
    GeoLookupResponse {
        found: found.is_some(),
        capital: found,
        source: "embedded-json-fallback".to_string(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FujiInfoResponse {
    pub safety_ja: String,
    pub safety_en: String,
    pub source_ja: String,
    pub source_en: String,
    pub transport_reservations: Vec<FujiTransportOption>,
    pub gear_shops: Vec<FujiGearShop>,
    pub mountain_huts: Vec<FujiMountainHut>,
}

/// 富士山の話題が出た際に案内する安全上の注意+山小屋(吉田口ルート)の
/// 予約先一覧(2026-08-11新設、ユーザー指示「富士山は危険な山ですので、
/// 1年中冬の様に寒いので上下スキーウェアを着て登山して、途中の山小屋を
/// 必ず安全な登山の為に予約して一泊されてから登山して下さいませ」+
/// 「https://mtfuji.jpn.org/availablelist.php ここのHP中身はCOPYして
/// DB化してから富士山の話題が出たら日本語と英語で紹介して」)。
/// **正直な開示**: 営業期間・電話番号は`mtfuji.jpn.org/availablelist.php`
/// (2026-08-11時点)から収集した静的データであり、山小屋の営業状況は
/// 毎年変わる——このレスポンスの`source_ja`/`source_en`フィールドで
/// 常に出典元と「利用前に直接確認すること」を明記する(既存の
/// 「正直な開示」方針、誇張しない)。
pub fn fuji_info() -> FujiInfoResponse {
    let d = dataset();
    FujiInfoResponse {
        safety_ja: d.fuji_safety_ja.clone(),
        safety_en: d.fuji_safety_en.clone(),
        source_ja: d.fuji_source_ja.clone(),
        source_en: d.fuji_source_en.clone(),
        transport_reservations: d.fuji_transport_reservations.clone(),
        gear_shops: d.fuji_gear_shops.clone(),
        mountain_huts: d.fuji_mountain_huts.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_dataset_parses_and_has_expected_counts() {
        let d = dataset();
        assert_eq!(d.japan_prefectures.len(), 47, "must cover all 47 Japanese prefectures");
        assert_eq!(d.us_states.len(), 50, "must cover all 50 US states");
        assert!(d.world_capitals.len() >= 50, "should have a substantial set of world capitals");
        assert!(!d.fuji_mountain_huts.is_empty(), "must have at least one Mt. Fuji mountain hut entry");
    }

    #[test]
    fn fuji_info_includes_safety_advisory_and_huts() {
        let info = fuji_info();
        assert!(info.safety_ja.contains("危険"));
        assert!(info.safety_en.to_lowercase().contains("dangerous"));
        assert!(info.mountain_huts.iter().any(|h| h.name_en == "Sato-goya"));
    }

    #[tokio::test]
    async fn random_entry_falls_back_to_embedded_when_db_not_configured() {
        std::env::remove_var("ARUARU_LLM_GEO_DATABASE_URL");
        let resp = random_entry().await;
        assert_eq!(resp.source, "embedded-json-fallback");
        assert!(!resp.prefecture.name_en.is_empty());
        assert!(!resp.state.name_en.is_empty());
        assert!(!resp.capital.country_en.is_empty());
    }

    #[tokio::test]
    async fn lookup_country_matches_english_and_japanese_names() {
        std::env::remove_var("ARUARU_LLM_GEO_DATABASE_URL");
        let en = lookup_country("australia").await;
        assert!(en.found);
        assert_eq!(en.capital.unwrap().capital_en, "Canberra");

        let ja = lookup_country("日本").await;
        assert!(ja.found);
        assert_eq!(ja.capital.unwrap().capital_en, "Tokyo");
    }

    #[tokio::test]
    async fn lookup_country_matches_country_name_embedded_in_a_sentence() {
        // 実機テストで実際に見つかったバグの回帰テスト: open-english側は
        // ユーザー発話文全体("I FROM JAPAN.")をそのまま渡すため、国名が
        // 文中の一部であっても一致させる必要がある。
        std::env::remove_var("ARUARU_LLM_GEO_DATABASE_URL");
        let resp = lookup_country("I FROM　JAPAN.").await;
        assert!(resp.found, "expected a sentence containing a country name to match");
        assert_eq!(resp.capital.unwrap().capital_en, "Tokyo");
    }

    #[tokio::test]
    async fn lookup_country_returns_not_found_for_unknown_country() {
        std::env::remove_var("ARUARU_LLM_GEO_DATABASE_URL");
        let resp = lookup_country("Narnia").await;
        assert!(!resp.found);
        assert!(resp.capital.is_none());
    }
}
