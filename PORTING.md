# PORTING.md — aruaru-llm を他プロジェクトへお引越しする際のガイド

> 🎯 **移植時の前提(2026-08-29)**: aruaru-dbはRPoemとのSETで初めて
> 「REST API不要・Cosmo有料版互換」の価値が成立する(正本:
> aruaru-db/CLAUDE.md冒頭)。aruaru-llmはopen-cudaとのSETで独立した
> AI推論層のため、この方針の直接の対象外だが、`src/tenants.rs`拡張時は
> 念頭に置くこと。

> **2026-07-25 更新**: 開発方針ファイル(`CLAUDE.md`)の見出しを
> 「設計思想＆開発方針＆開発環境ルール」へ改名しました
> (設計思想・開発方針・開発環境ルールを明確に区別)。移設先でも
> `CLAUDE.md`の内容を必ず確認してください。


## 0. Google検索APIキーのリクエスト単位オーバーライド(2026-08-25新設)

`POST /v1/generate-with-search`は任意フィールド`google_search_api_key`/
`google_search_cx`を受け付ける。両方指定された場合、プロセス全体で
共有されるグローバル設定(環境変数/`POST /v1/settings/google-search`)には
一切触れず、そのリクエスト限りの認証情報として`web_search::
search_with_credentials()`を使う。**複数の訪問者が同一インスタンスを
共有するデプロイ(VPS等)では、この経路を使うことで訪問者間のAPIキー・
クォータの意図しない共有・消費を防げる**——移植先でも同様のマルチ
テナント公開シナリオがあれば、この設計をそのまま踏襲すること。

## 0b. ブラウザ内AI実行(WASM+WebGPU)構想(2026-08-25追記、計画段階)

RPoemの`PORTING.md`/`CLAUDE.md`(2026-08-25エントリ)に技術検証結果
(`wgpu`のwasm32ビルド成功)・段階的導入計画を記載済み。第1段階では
このリポジトリのGPT-2/distilgpt2生成ロジック(現状CPU実行)を
`wasm32-unknown-unknown`向けにビルドできるCargo featureを追加する
想定——移植先でこの機能を先行実装する場合はRPoem側のCLAUDE.mdを
必ず先に確認すること。

## 1. open-cuda連携パターン(SET構成)

`src/scoring.rs`に、open-cudaの`GpuDevice`実行パイプライン
(`alloc_buffer`→`copy_from_host`→`launch_kernel`→`synchronize`→
`copy_to_host`)を実際に呼び出すパターンをまとめている。

移植手順:
1. `Cargo.toml`に、移植先から見た相対パスで`opencuda-core`/
   `opencuda-cpu`をpath依存として追加する(本リポジトリとopen-cudaが
   同じ親ディレクトリ配下にある前提。`../open-cuda/crates/...`)。
2. `src/scoring.rs`の`elementwise_multiply_via_opencuda`関数
   (open-cudaの`examples/vector_add`と同一の安全性根拠を持つ最小
   カーネル実行パターン)をそのままコピーし、用途に応じてカーネルの
   演算内容(乗算→加算等)を書き換える。
3. 依存先(open-cuda)のデフォルトfeatureは`winfsp_backend`/`gpu_accel`
   だが、`opencuda-core`/`opencuda-cpu`単体はこれらのfeatureに依存
   しないため、追加のSDK(WinFsp/dxc等)は不要。

## 2. ルールベース意図分類(将来の実LLM差し替え前提の設計)

`INTENTS`定数(キーワード・応答文の組)と、`best_intent()`関数の
シグネチャ(`&str` → `Option<&Intent>`)を維持したまま、内部実装だけを
実際のLLM呼び出しに差し替えられるようにしてある。`engine`フィールドに
常に実装方式を正直に返すことで、呼び出し側が「本物のAIかどうか」を
判別できるようにする設計は、他プロジェクトへ移植する際にも踏襲すること。

## 3. HTTP API層

**2026-07-31更新**: `main.rs`は本家`poem`クレートではなく`RPoem`
(`open-runo-poem-compat`、path依存)を使用する。`Data<T>`抽出子が無いため、
共有状態(`Arc<dyn GpuDevice>`・`Arc<TenantRegistry>`)はハンドラ登録時の
クロージャで`Arc::clone`をキャプチャするパターンに置き換わっている
(デバイスをリクエストごとに再生成しない、という設計意図自体は不変)。
他プロジェクトへ移植する際は、`open-runo-poem-compat`の
`Route::new().at(path, get(handler_fn(...)))`+`handler_fn`+
`Json::from_body(req).await`+`PathParams::from(params)`という組み合わせを
テンプレートとして使えばよい(本リポジトリの`src/main.rs`がそのまま
参考実装になる)。

## 3.5. 空入力の入力検証パターン(2026-08-06追加)

`POST /v1/generate`/`POST /v1/translate`が空の`prompt`/`text`を受け
取った場合、以前はトークナイザ内部の「0トークン」エラーが本物のバック
エンド障害と区別できず、誤解を招く`503 Service Unavailable`を返して
いた実バグがあった。`main.rs`のハンドラ冒頭で明示的な入力検証
(`prompt must not be empty`等、`400 Bad Request`)を追加して解消。
他プロジェクトのAPIハンドラでも、「内部コンポーネントのエラーを
そのまま外部エラーコードへ透過させていないか」(特に空入力・境界値)を
確認する価値がある——本件と同型のバグが`/v1/chat`・
`/v1/classify-security`にも潜んでいないかは未監査(次回課題)。

## 4. 「分身の術」テナント登録パターン(`open-web-server`と共通)

`src/tenants.rs`の`TenantRegistry`(`RwLock<HashMap<String, TenantInfo>>`)
+ `main.rs`の`POST /admin/tenants`・`GET /admin/tenants`・
`DELETE /admin/tenants/:host`(`x-admin-token`ヘッダ簡易認証)は、
「1インスタンスを複数ドメインが共有し、ドメインごとの個別インストールを
不要にする」という`open-web-server`/`open-easy-web`と同じ設計思想の
最小実装。他プロジェクトへ移植する際は、この3ファイル
(`tenants.rs`本体、`main.rs`の管理ハンドラ、`check_admin_token`)を
そのままコピーし、`TenantInfo`のフィールドだけ用途に応じて拡張すること。

## 5. 本格的な生成能力(`opencuda-llm::GptModel`、2026-07-25追加)

`src/generation.rs`に、GPT-2 124M実重み(`openai-community/gpt2`)を
`OnceLock`でプロセス内キャッシュしつつロード・貪欲デコードするパターンを
まとめている(`opencuda-bert::BertModel::load`と同じ設計思想)。

移植手順:
1. `Cargo.toml`に`opencuda-llm = { path = "../open-cuda/crates/opencuda-llm" }`
   をpath依存として追加する(本リポジトリとopen-cudaが同じ親ディレクトリ
   配下にある前提、他の`opencuda-*`依存と同じsibling pathパターン)。
2. `src/generation.rs`をそのままコピーする。`model_dir()`のデフォルトパス
   (`../open-cuda/crates/opencuda-llm/models/gpt2`)と環境変数名
   (`ARUARU_LLM_GPT2_DIR`)は移植先の事情に合わせて変更してよい。
3. `main.rs`に`POST /v1/generate`ハンドラを追加する(`GenerateRequest`/
   `GenerateResponse`/`GenerateErrorResponse`ごとコピー可能)。
4. **重要(誇大表示の回避)**: `disclosure`フィールド(GPT-2 124Mが小型・
   2019年モデルであり最新商用LLMと同等でないことを明記)は、レスポンス
   から省略しないこと。`engine`フィールドに実装方式
   (`gpt2-124m-greedy-decode-v0-opencuda-llm-cpu`)を常に正直に返す設計も
   踏襲すること。
5. **意図分類(`/v1/chat`)と生成(`/v1/generate`)を無理に統合しない**
   ——役割が異なる(前者は軽量・高速な定型応答振り分け、後者は本格的だが
   重い自由文生成)ため、別エンドポイントとして両方提供するのがこの
   エコシステムの設計方針。

## 6. ハードウェア検出→推奨LLMサイズ→自動ダウンロード(2026-07-27追加)

`src/hardware.rs`(VRAM容量→推奨モデルサイズの簡易ヒューリスティック、
`open-cuda`/`open-directx`のGPU検出結果をどちらの経路から取るか)と、
`main.rs`の`GET /v1/recommend`・`POST /v1/recommend-and-download`・
`GET /`(`static/index.html`の最小UI)を新設した。

移植手順:
1. `Cargo.toml`に`opencuda-vulkan`/`opencuda-directx`をoptional
   path依存として追加し、`hw-detect-vulkan`/`hw-detect-directx`
   feature(既定無効)を定義する(`hw-detect-vulkan = ["dep:opencuda-vulkan",
   "opencuda-vulkan/real-vulkan"]`のように、上流クレート自身のopt-in
   feature〈`real-vulkan`/`real-dx12`〉へ連鎖させる)。**重要**: これらの
   featureを既定で有効にしないこと——Android等クロスコンパイル環境や
   CI環境でVulkanローダー/Windows SDKへの依存を強制しないため
   (`opencuda-vulkan`/`opencuda-directx`自身の既存の設計方針と同じ)。
2. `src/hardware.rs`をそのままコピーする。VRAM閾値
   (`recommend_id_for_vram`)は`model_catalog::CATALOG`のサイズ構成に
   合わせて調整すること。
3. Vulkan/DirectXを両方有効にした場合、片方を優先しもう片方はクロス
   チェック(`cross_check_agreement`フィールド)として扱う設計を維持
   すること——「どちらの経路の情報を実際に使っているか」を常に
   レスポンスへ明記する(誇大表示回避、`detection_path`フィールド)。
4. `main.rs`に`GET /v1/recommend`(検出のみ)・
   `POST /v1/recommend-and-download`(検出→ダウンロード→ホットスワップ
   切り替えまで一括)ハンドラを追加する。切り替え失敗時は現在動作中の
   モデルを維持すること(`generation::select_model`と同じ「失敗しても
   サービスを壊さない」設計を踏襲)。
5. **正直な開示を省略しないこと**: VRAM容量とモデルサイズの単純比較に
   過ぎず精密な性能予測ではない旨(`hardware.rs`モジュールdoc参照)を、
   レスポンスの`disclosure_ja`フィールドとUI双方に必ず表示すること。
6. UIを追加する場合、Tauri/Node.js/TypeScript等の重量フレームワークを
   導入せず、`static/index.html`(`include_str!`でRustバイナリへ埋め込み、
   `poem`から`text/html`で配信)のような最小構成に留めること
   (過剰実装を避ける、このエコシステム共通の設計方針)。

## 7. 翻訳プラグイン(`nllb-translate` feature、2026-08-04追加)

`POST /v1/translate`のGPT-2流用実装は実用に耐えないと実HTTP検証で判明
したため、`rust-bert`(M2M100)によるオープンソース専用翻訳モデルを
Cargo featureの着脱式プラグインとして追加した(`src/nllb.rs`)。

移植手順:
1. `Cargo.toml`に`rust-bert = { version = "0.23", optional = true }`・
   `tch = { version = "0.17", optional = true }`を追加し、
   `[features]`に`nllb-translate = ["dep:rust-bert", "dep:tch"]`
   (既定オフ)を追加する。
2. `src/nllb.rs`をそのままコピーする(`#[cfg(feature = "nllb-translate")]`
   で完全に分岐しており、移植先のCargo.tomlに同名featureを用意すれば
   無変更で動く設計)。
3. 翻訳エンドポイントのハンドラで、まず`nllb::translate_with_nllb(...)`
   を試み、`Err`の場合のみ既存の生成実装(GPT-2等)へフォールバックする
   構成にする。
4. **正直な開示・移植時の注意**: `rust-bert`は`tch`(libtorch、
   PyTorchのC++ライブラリ)への依存が必須で、このエコシステムの他の
   モデル(GPT-2・BERT・Whisper相当)が貫く「手作りRust実装+
   safetensors直接ロード」方針から意図的に外れる大きな依存。移植先が
   このビルド時間・依存グラフの増加を許容できるか、着手前に判断
   すること。`nllb-translate` feature未指定であれば依存は一切ビルドに
   含まれないため、既定では影響ゼロ。

## 8. 実推論のVulkanディスパッチ(`real-vulkan` feature、2026-08-04追加、未完成につき移植非推奨)

`main()`のデバイス選択を`opencuda_vulkan::real::VulkanDevice`へ
切り替えるオプトインfeature。移植手順自体は`hw-detect-vulkan`と同型
(`Cargo.toml`に`opencuda-vulkan`をoptional path依存として追加、
`real-vulkan = ["dep:opencuda-vulkan", "opencuda-vulkan/real-vulkan"]`、
`main()`を`#[cfg(feature = "real-vulkan")]`で分岐、構築失敗時はCPUへ
フォールバック)だが、**現時点では実際のGEMMディスパッチが機能しない
既知バグが`open-cuda`側`open-cuda-llm`クレートにある**(`Linear::forward`
が`matmul.spv`を`sgemm`へ渡していないため`GemmPath::VulkanGeneric`選択時に
即座にエラー、詳細はREADME.md「実推論ディスパッチ先としてのVulkan」節・
`CLAUDE.md`HANDOFF参照)。**このfeatureパターン自体を他プロジェクトへ
移植するのは、`open-cuda`側の修正が完了し実機で速度改善が確認できてから
にすること**(現状は「配線したが動かない」状態を複製するだけになる)。

## 繰り返しペナルティ(2026-08-10新設、`open-cuda`側`GptModel::
generate_with_repetition_penalty`)

`src/generation.rs::generate`は、対話ファインチューニング無しの素の
GPT-2貪欲デコードが陥る既知の劣化モード(同一文字列の無限ループ)への
根本対応として、`open-cuda`側の`GptModel::generate_with_repetition_
penalty`(CTRL方式、penalty>1.0で既に登場したトークンのlogitを弱める)を
既定`1.3`で呼ぶ。移植手順:

1. `open-cuda`側`crates/open-cuda-llm`が`generate_with_repetition_
   penalty`(および後方互換ラッパー`generate`)を持つことを確認する
   (2026-08-10以降のコミットに存在)。
2. `src/generation.rs::default_repetition_penalty()`
   (`ARUARU_LLM_REPETITION_PENALTY`環境変数、既定`1.3`)と、`generate()`
   内の呼び出し(`model.generate_with_repetition_penalty(device,
   &prompt_ids, max_new_tokens, default_repetition_penalty())`)を
   そのままコピーする。
3. `penalty=1.0`にすると既存の`generate()`(ペナルティ無し)と完全に
   同一の出力になる(`open-cuda`側テスト
   `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`の
   `via_generate == no_penalty`アサーションで裏付け済み)ため、移植先で
   挙動を変えたくない場合はこの値に設定すればよい。

## 9. 音声認識(`POST /v1/transcribe`、whisper.cpp CLI、2026-08-29追加)

`POST /v1/transcribe`(`src/transcribe.rs`)を他プロジェクトへ持ち込む
場合の要点。正本は `open-english/docs/SPEECH_RECOGNITION_REDESIGN.md`
§P2-β。

1. **`whisper-rs` を直接リンクしないこと**(最重要)。`whisper-rs-sys` は
   Windows(MSVC)で bindgen が glibc 固有型を生成して破綻する既知
   ブロッカーがあり、`0.16.0` でも `WHISPER_DONT_GENERATE_BINDINGS=1`
   でも解消しない(issue 2026-04-21、公式 fix 未提供)。代わりに
   **whisper.cpp の公式リリース同梱プレビルド CLI(`whisper-cli` / 旧
   `main`)を子プロセス起動**する(`Db::backup_postgres_via_pg_dump` が
   `pg_dump` を、`component_update` が `Expand-Archive` を、Android 連携が
   `adb` を子プロセスで呼ぶのと同じパターン)。C++ リンク・bindgen を
   完全回避できるため **Cargo feature は不要**——`src/transcribe.rs` は
   常にコンパイルされ、`is_available()` = `cli_present() && model_present()`
   の**実行時**判定で可否を出す。
2. 実装(`src/transcribe.rs`、~250 行、外部 crate 追加なし): 16kHz mono
   f32 PCM → 最小 WAV を手書き(44 バイトヘッダ + i16 サンプル)→
   `whisper-cli -m <model> -f <wav> -l <lang|auto> -oj -of <prefix> -nt
   -np -t <n>` を `std::process::Command` で起動 → `<prefix>.json` を
   `serde_json::Value` で緩くパース(`transcription[].text` 連結、
   `result.language`)。壁時計上限(既定 300s、`*_TIMEOUT_SECS`)超過で
   `child.kill()`。スクラッチは `std::env::temp_dir()` 下の一意サブ
   ディレクトリ(`tempfile` crate を実行時依存に加えない)。
3. パス解決: `ARUARU_LLM_WHISPER_CLI`(既定 `<crate>/models/whisper/
   whisper-cli[.exe]`、`main[.exe]` もフォールバック)、
   `ARUARU_LLM_WHISPER_MODEL`(既定 `.../ggml-base.bin`)。どちらも
   リポジトリ非同梱。無ければ `503` + 入手先(whisper.cpp releases)を案内。
4. 入力は **16kHz mono f32 PCM の LE バイト列を base64 化**したもの
   (呼び出し側=ブラウザが `OfflineAudioContext` で 16kHz へリサンプル
   済みの `Float32Array` をそのまま送る想定)。`sample_rate ≠ 16000` /
   base64 不正 / 4 の倍数でない / 10 分超(~38MB)はいずれも `400`。
5. 重い子プロセス処理は `tokio::task::spawn_blocking` へ逃がす
   (`generate` ハンドラと同じ)。
6. `GET /v1/runtime` に `whisper` 段(`available` / `backend` /
   `cli_path` / `cli_present` / `model_path` / `model_present` /
   `detail`)を追加し、CLI とモデルの実在を**正直に**見せる。両方
   true のときだけ実際に書き起こせる。

## 注意事項

- 本プロジェクトは「LLM」を名乗り、2026-07-25以降`/v1/generate`で実際の
  GPT-2 124M自己回帰生成が可能になったが、GPT-2 124M自体は小型・2019年
  モデルであり最新商用LLM(GPT-4等)と同等の性能ではない旨を、移植先でも
  必ず明記すること(誇大表示の回避、このエコシステム共通の「正直な開示」
  規約)。`/v1/chat`(意図分類)は引き続きルールベース+エンコーダの
  意味的類似度分類であり、こちらもニューラル対話生成そのものではない
  ことを混同しないこと。
