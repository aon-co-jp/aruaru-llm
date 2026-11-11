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


## -1. Model Folding(2026-09-01新設)——依存先`open-cuda`側`open-cuda-llm`クレートの新規API

`POST /v1/models/fold-layers`(3モード: 独立閾値/連続ブロック探索/線形
アダプタ)・`POST /v1/models/layer-redundancy`の実体は`open-cuda`側
`open-cuda-llm::GptModel`の`analyze_layer_redundancy`/
`prune_redundant_layers`/`find_best_layer_block_to_remove`/
`remove_layer_block`/`fold_block_with_linear_adapter`にある
(このリポジトリは薄いHTTPラッパー`generation.rs`のみ)。移植先でも
`open-cuda`をsibling path依存として持ってくる際は、この一式が
`crates/open-cuda-llm/src/lib.rs`に含まれることを確認すること。
「DeepSeekの折りたたみ理論」は実在しない技術である旨・実装した
代替手法(ShortGPT/Gromov et al./SHIFT-LLM/SlimLLM着想)の詳細は
`CLAUDE.md`のHANDOFF(2026-09-01)を必ず読むこと——移植先で同じ質問
(「DeepSeekの折りたたみを実装して」)を受けた場合、同じ調査を
繰り返す前にまずこの記録を参照すること。

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

## 追記(2026-09-21): アクセラレーター検出APIの移植ポイント / Accelerator detection API porting notes

**日本語**: `/v1/accelerators`は`hardware::detect_accelerators()`(Windows: PowerShellのCIMでNPU名、adbでUSB接続Android)+論理コア数。Linux/VPSではNPU検出は未実装(`None`)。CPU命令は`open_cpu::inventory()`(パス依存`../open-cpu`)。

**English**: `/v1/accelerators` wraps `hardware::detect_accelerators()` (Windows: NPU name via PowerShell/CIM, USB Android via adb) plus the logical core count. NPU detection is not implemented on Linux/VPS (`None`). CPU features come from `open_cpu::inventory()` (path dependency `../open-cpu`).

## HANDOFF追記(2026-09-22、本日後半のまとめ・多言語) / HANDOFF addendum (2026-09-22, later today, multilingual)

**日本語**: 本日追加した機能: (1) 管理者専用のAI優先順位番号付け(#1-3)、(2) 四つの9パズルの自動採点(演算子・括弧の全角半角/日本語表記ゆれを正規化、括弧忘れのみ半分正解、÷9丸ごと忘れは不正解)、
(3) 自分のAPIキー設定時はWEB版共有AIと併用しない(自分の鍵を優先・排他利用)、(4) タイ語・ベトナム語・フィリピン語・ミャンマー語・クルド語・トルコ語・ロマンシュ語を学習言語に追加(文化・敬語の豆知識つき)、
(5) 学びたい言語・応答言語・レベルの保存(PC/スマホ/タブレット共通)、(6) 話題ガイドにGoogle画像検索リンク追加、(7) 文字入力/音声入力/翻訳/AI自身の知識に自信が無い時のGoogle検索自動裏取り(3つの理由を利用者へ開示)、
(8) 3〜4ヶ国語同時ハイブリッド表示(hybrid3/hybrid4)、(9) 「今日のニュースは?」が公開WEB版で常に失敗していた不具合修正(`/v1/public/news/for?country=...`、質問言語で国を判定)、
(10) ニュースダイジェストの国別DATABASE化(TTL3時間、`data/news_by_country.json`)+日付記録(検索日時・記事取得日時、`retrieved_at_unix`)、(11) 8日以上前のニュースを生きているDBから追い出しMarkdownへアーカイブする仕組み
(`prune_and_archive_stale_news`、GitHubへの実際のpushは今回未実装——常時稼働プロセスへ書き込み資格情報を持たせないための意図的な判断、実際のpushは今後Claude Code経由で手動実施)、
(12) 「もっと簡単に」「難しい」発言でレベルを自動的に1段下げる機能+ネイティブ表現アドバイスの指示、(13) 「最新の情報が欲しい」ニュアンス検出でGoogle検索・GitHub調査を自動的に有効化。
**未着手**: 実際のGitHubアーカイブpush、GitHub Actionsやスケジュールタスクとしての定期実行の自動化(`news_prune_archive`エンドポイントは実装済みだが、呼び出しの自動化・cron化は未着手)。

**English**: Features added today: (1) admin-only AI priority numbering (#1-3), (2) automatic grading for the four-nines puzzle, (3) own API key now excludes the shared web AI (exclusive use), (4) added Thai/Vietnamese/Filipino/Burmese/Kurdish/Turkish/Romansh as learn languages (with cultural tips),
(5) persisted learn-target/reply-lang/level across PC/phone/tablet, (6) Google Images links in topic guides, (7) automatic Google-search grounding when text/voice/translation/AI-knowledge confidence is low (with a visible reason), (8) 3/4-language simultaneous hybrid mode,
(9) fixed "today's news" always failing on the public web version, now picks the country from the question's language, (10) per-country news digest database (3h TTL) with retrieval timestamps on each item, (11) archiving of news older than 8 days out of the live DB into a local Markdown file
(actual GitHub push not yet automated — deliberately not giving the always-running server process write credentials; the real push will be done manually via Claude Code), (12) auto level step-down on "too difficult/simpler please", (13) auto-enabling Google/GitHub search on "latest info" phrasing.
**Not done yet**: actually pushing the archive to GitHub, and scheduling/cron automation for `news_prune_archive`.

## HANDOFF追記(2026-09-22、多言語版) / Multilingual handoff addendum

**简体中文**: 今天新增的功能:(1)管理员专用的AI优先级编号(#1-3),(2)"四个9"数学谜题自动评分(标准化全角/半角运算符及日语表达,只忘记括号记为半对,完全忘记÷9记为错误),
(3)设置自己的API密钥后不再与网页版共享AI并用(优先且排他使用自己的密钥),(4)新增泰语、越南语、菲律宾语、缅甸语、库尔德语、土耳其语、罗曼什语作为学习语言(附文化及敬语小知识),
(5)学习语言/回复语言/等级设置可跨PC、手机、平板保存,(6)话题指南中新增Google图片搜索链接,(7)当文字输入/语音输入/翻译/AI自身知识置信度低时自动通过Google搜索核实(并向用户说明三种触发原因),
(8)支持3~4种语言同时混合显示(hybrid3/hybrid4),(9)修复了公开网页版"今天的新闻"功能一直失败的问题(新增`/v1/public/news/for?country=...`,根据提问语言判断国家),
(10)按国家分类的新闻摘要数据库(TTL 3小时,`data/news_by_country.json`)并记录检索/发布日期(`retrieved_at_unix`),(11)将8天以上的旧新闻从活跃数据库中清除并归档为Markdown文件
(实际推送到GitHub的部分本次尚未实现——刻意不给常驻运行的服务器进程赋予GitHub写入权限,实际推送将来通过Claude Code手动完成),
(12)当用户说"简单一点"或"太难了"时自动降低一级难度,并附带母语者表达建议,(13)检测到"想要最新信息"的语气时自动启用Google搜索与GitHub调查。
**尚未完成**: 实际推送新闻归档到GitHub、以及`news_prune_archive`的定时/自动化执行。

**繁體中文**: 今天新增的功能:(1)管理員專用的AI優先順序編號(#1-3),(2)「四個9」數學謎題自動評分(標準化全形/半形運算子及日語表達,只忘記括號記為半對,完全忘記÷9記為錯誤),
(3)設定自己的API金鑰後不再與網頁版共用AI並用(優先且排他使用自己的金鑰),(4)新增泰語、越南語、菲律賓語、緬甸語、庫德語、土耳其語、羅曼什語作為學習語言(附文化及敬語小知識),
(5)學習語言/回覆語言/等級設定可跨PC、手機、平板保存,(6)話題指南中新增Google圖片搜尋連結,(7)當文字輸入/語音輸入/翻譯/AI自身知識信心度低時自動透過Google搜尋核實(並向使用者說明三種觸發原因),
(8)支援3~4種語言同時混合顯示(hybrid3/hybrid4),(9)修復了公開網頁版「今天的新聞」功能一直失敗的問題(新增`/v1/public/news/for?country=...`,依提問語言判斷國家),
(10)依國家分類的新聞摘要資料庫(TTL 3小時,`data/news_by_country.json`)並記錄檢索/發布日期(`retrieved_at_unix`),(11)將8天以上的舊新聞從活躍資料庫中清除並歸檔為Markdown檔案
(實際推送到GitHub的部分本次尚未實作——刻意不給常駐運行的伺服器行程賦予GitHub寫入權限,實際推送將來透過Claude Code手動完成),
(12)當使用者說「簡單一點」或「太難了」時自動降低一級難度,並附帶母語者表達建議,(13)偵測到「想要最新資訊」的語氣時自動啟用Google搜尋與GitHub調查。
**尚未完成**: 實際推送新聞歸檔到GitHub、以及`news_prune_archive`的排程/自動化執行。

**Français**: Fonctionnalités ajoutées aujourd'hui : (1) numérotation de priorité IA réservée aux admins (#1-3), (2) notation automatique du puzzle des « quatre 9 » (normalisation des opérateurs pleine/demi-chasse et des expressions japonaises ; oubli des parenthèses seul = à moitié correct, oubli complet de ÷9 = incorrect),
(3) l'utilisation de sa propre clé API exclut désormais l'IA partagée du site web (usage exclusif de sa propre clé), (4) ajout du thaï, vietnamien, philippin, birman, kurde, turc et romanche comme langues d'apprentissage (avec notes culturelles),
(5) persistance des préférences (langue cible, langue de réponse, niveau) sur PC/mobile/tablette, (6) liens de recherche d'images Google dans les guides thématiques, (7) recherche Google automatique lorsque la confiance dans la saisie texte/vocale/traduction/connaissance de l'IA est faible (raison affichée à l'utilisateur),
(8) mode hybride 3/4 langues simultané (hybrid3/hybrid4), (9) correction du bug où « les nouvelles du jour » échouaient toujours sur le site public (nouvelle route `/v1/public/news/for?country=...`, pays déterminé par la langue de la question),
(10) base de données d'actualités par pays (TTL 3h, `data/news_by_country.json`) avec horodatage de récupération (`retrieved_at_unix`), (11) archivage des actualités de plus de 8 jours vers un fichier Markdown local
(le push GitHub réel n'est pas encore automatisé — décision délibérée de ne pas donner d'identifiants d'écriture GitHub au processus serveur permanent ; le push sera fait manuellement via Claude Code), (12) rétrogradation automatique du niveau sur demande explicite (« plus simple », « trop difficile ») avec conseils de formulation naturelle,
(13) activation automatique de la recherche Google/GitHub sur détection d'une intention « informations les plus récentes ».
**Non terminé** : push réel de l'archive vers GitHub, automatisation planifiée de `news_prune_archive`.

**Deutsch**: Heute hinzugefügte Funktionen: (1) nur für Admins sichtbare KI-Prioritätsnummerierung (#1-3), (2) automatische Bewertung des „Vier-Neunen"-Rätsels (Normalisierung von Voll-/Halbbreiten-Operatoren und japanischen Ausdrücken; nur vergessene Klammern = halb richtig, komplett vergessenes ÷9 = falsch),
(3) eigener API-Schlüssel schließt die gemeinsame Web-KI nun aus (exklusive Nutzung des eigenen Schlüssels), (4) Thailändisch, Vietnamesisch, Filipino, Birmanisch, Kurdisch, Türkisch und Rätoromanisch als Lernsprachen hinzugefügt (mit kulturellen Hinweisen),
(5) Speicherung von Zielsprache/Antwortsprache/Niveau geräteübergreifend (PC/Handy/Tablet), (6) Google-Bildersuche-Links in Themenführern, (7) automatische Google-Suchabsicherung bei geringer Konfidenz bei Text-/Spracheingabe/Übersetzung/KI-Wissen (Grund wird dem Nutzer angezeigt),
(8) gleichzeitiger 3/4-Sprachen-Hybridmodus (hybrid3/hybrid4), (9) Fehlerbehebung: „heutige Nachrichten" funktionierten auf der öffentlichen Website nie (neue Route `/v1/public/news/for?country=...`, Land wird anhand der Fragesprache bestimmt),
(10) länderspezifische Nachrichten-Datenbank (TTL 3h, `data/news_by_country.json`) mit Abrufzeitstempel (`retrieved_at_unix`), (11) Archivierung von Nachrichten älter als 8 Tage aus der Live-DB in eine lokale Markdown-Datei
(tatsächlicher GitHub-Push noch nicht automatisiert — bewusste Entscheidung, dem dauerhaft laufenden Serverprozess keine GitHub-Schreibrechte zu geben; der eigentliche Push erfolgt später manuell über Claude Code), (12) automatische Niveau-Herabstufung bei „einfacher bitte"/„zu schwer" mit Hinweisen zu natürlicher Ausdrucksweise,
(13) automatische Aktivierung der Google-/GitHub-Suche bei erkannter „aktuellste Informationen"-Absicht.
**Noch nicht erledigt**: tatsächlicher Push des Archivs zu GitHub, geplante/automatisierte Ausführung von `news_prune_archive`.

**ภาษาไทย**: ฟีเจอร์ที่เพิ่มวันนี้: (1) การจัดลำดับความสำคัญ AI แบบใส่หมายเลข (#1-3) สำหรับผู้ดูแลระบบเท่านั้น (2) การให้คะแนนอัตโนมัติสำหรับปริศนา "เลข 9 สี่ตัว" (ปรับเครื่องหมายเต็ม/ครึ่งความกว้างและคำภาษาญี่ปุ่นให้เท่ากัน ลืมเฉพาะวงเล็บ = ถูกครึ่งหนึ่ง ลืม ÷9 ทั้งหมด = ผิด)
(3) เมื่อตั้งค่าคีย์ API ของตนเองแล้ว จะไม่ใช้ AI ที่แชร์บนเว็บร่วมด้วย (ใช้คีย์ของตนเองแบบเอกสิทธิ์) (4) เพิ่มภาษาไทย เวียดนาม ฟิลิปปินส์ พม่า เคิร์ด ตุรกี และโรมันช์ เป็นภาษาที่เรียนได้ (พร้อมข้อมูลวัฒนธรรม)
(5) บันทึกการตั้งค่าภาษาที่เรียน/ภาษาตอบกลับ/ระดับ ข้ามอุปกรณ์ PC/มือถือ/แท็บเล็ต (6) ลิงก์ค้นหารูปภาพ Google ในคู่มือหัวข้อ (7) เปิดใช้การค้นหา Google อัตโนมัติเมื่อความมั่นใจในการพิมพ์/เสียง/การแปล/ความรู้ของ AI ต่ำ (แสดงเหตุผลให้ผู้ใช้เห็น)
(8) โหมดผสมหลายภาษาพร้อมกัน 3-4 ภาษา (hybrid3/hybrid4) (9) แก้ไขบั๊กที่ฟีเจอร์ "ข่าววันนี้" ใช้งานไม่ได้เลยบนเว็บสาธารณะ (เพิ่มเส้นทาง `/v1/public/news/for?country=...` เลือกประเทศตามภาษาของคำถาม)
(10) ฐานข้อมูลข่าวสรุปรายประเทศ (TTL 3 ชั่วโมง `data/news_by_country.json`) พร้อมบันทึกเวลาดึงข้อมูล (`retrieved_at_unix`) (11) เก็บถาวรข่าวที่เก่ากว่า 8 วันออกจากฐานข้อมูลไปเป็นไฟล์ Markdown
(ยังไม่ได้ทำการ push ไปยัง GitHub จริง — เป็นการตัดสินใจโดยเจตนาที่จะไม่ให้สิทธิ์เขียน GitHub แก่โพรเซสเซิร์ฟเวอร์ที่ทำงานตลอดเวลา การ push จริงจะทำผ่าน Claude Code ในภายหลัง) (12) ลดระดับความยากอัตโนมัติเมื่อผู้ใช้พูดว่า "ง่ายกว่านี้" หรือ "ยากเกินไป" พร้อมคำแนะนำสำนวนแบบเจ้าของภาษา
(13) เปิดใช้การค้นหา Google/GitHub อัตโนมัติเมื่อตรวจพบความต้องการ "ข้อมูลล่าสุด"
**ยังไม่เสร็จ**: การ push คลังข่าวจริงไปยัง GitHub และระบบอัตโนมัติ/ตั้งเวลาสำหรับ `news_prune_archive`

## v1.3.0(2026-10-10): かすれ補正・字形照合・AI文脈補完の拡張 / v1.3.0: faded-text rescue, glyph matching, extended AI context completion

### 日本語
- OCR の補完・字形照合の API を追加: `POST /v1/ocr/scan` / `propose` / `fill`(`return_all` / `use_web` / `lm_candidates`)/ `glyph-rank`、`GET /v1/ocr/charset?n=`。いずれも言語モデル(`/v1/qwen/install` → `/v1/qwen/select`)が必要(無いとき 503。`glyph-rank` は `extra_chars` だけでも動く)。
- 実装: `src/ocr_fill.rs`(走査・提案・採点・検索・連続 1〜4 文字。採点は最大 8 候補・後ろ文脈 8 文字)、`src/ocr_glyph.rs`(字形の照合 = `opencuda_blas::sgemm`、共有クレート `RPoem/open-runo-glyph`)、`src/qwen_generation.rs`(`score_texts` / `top_next_texts` / `scan_text` / `common_chars`)、`src/web_search.rs`(`snippets_via_aruaru_search`)。
- 検索: `ARUARU_LLM_SEARCH_URL`(既定 `http://127.0.0.1:4610` = `aruaru-search`)。試験は `cargo test --bin aruaru-llm ocr_`(20 件)、速度は `cargo test --release --bin aruaru-llm bench_sgemm -- --ignored --nocapture`(字形 1 升 約 14 ms)。
- 測定: 補完はこれまでの試験で「害はないが上積みも無く遅い」(1 か所 10〜20 秒、Qwen2.5-1.5B・CPU)。既定オフ・実験的。詳細は `pdf-r2l-rs/PORTING.md`。
- **VPS 本番**: `/v1/ocr`・`/v1/ocr/fill` は反映済み。v1.3.0 の追加分(`/scan`・`/propose`・`/glyph-rank`・`/charset`)も 2026-10-10 に反映(Qwen 未導入のため補完系は 503 が正常。VPS は 4 コア・RAM 3.6GB)。バックアップ: `/root/aruaru-llm.bin.bak-20261010`。トークンは `/root/aruaru-llm/.env.ocr`(git 管理外)。

### English
- Added OCR completion / glyph-matching APIs: `POST /v1/ocr/scan` / `propose` / `fill` (`return_all` / `use_web` / `lm_candidates`) / `glyph-rank`, `GET /v1/ocr/charset?n=`. All need a language model (`/v1/qwen/install` then `/v1/qwen/select`; 503 without one — `glyph-rank` works with just `extra_chars`).
- Code: `src/ocr_fill.rs` (scan / propose / score / search / 1–4 character spans; scoring uses at most 8 candidates and 8 characters of following context), `src/ocr_glyph.rs` (glyph matching = `opencuda_blas::sgemm`, shared crate `RPoem/open-runo-glyph`), `src/qwen_generation.rs` (`score_texts` / `top_next_texts` / `scan_text` / `common_chars`), `src/web_search.rs` (`snippets_via_aruaru_search`).
- Search: `ARUARU_LLM_SEARCH_URL` (default `http://127.0.0.1:4610` = `aruaru-search`). Tests: `cargo test --bin aruaru-llm ocr_` (20 tests); speed: `cargo test --release --bin aruaru-llm bench_sgemm -- --ignored --nocapture` (~14 ms per glyph cell).
- Measurements: in our tests completion does no harm but adds nothing and is slow (10–20 s per spot, Qwen2.5-1.5B on CPU). Off by default, experimental. See `pdf-r2l-rs/PORTING.md`.
- **Production VPS**: `/v1/ocr` and `/v1/ocr/fill` were deployed earlier; the v1.3.0 additions (`/scan`, `/propose`, `/glyph-rank`, `/charset`) were deployed on 2026-10-10 (503 is expected without a model; the VPS has 4 cores / 3.6 GB RAM). Backup: `/root/aruaru-llm.bin.bak-20261010`. The token is in `/root/aruaru-llm/.env.ocr` (not in git).

## お引越し手順(VPS再構成) / Migration runbook (rebuilding the VPS)

**日本語**
1. 必要物: `git` `jq` `curl` と Rust(`~/.cargo/bin`)。兄弟ディレクトリに依存リポジトリ(`/root/RPoem`→`/root/repository/RPoem`、`/root/open-cuda`→`/root/repository/open-cuda`)。
2. `git clone https://github.com/aon-co-jp/aruaru-llm /root/aruaru-llm` → `cargo build --release`。
3. 秘密を別経路で復元(Gitに無い): `/root/aruaru-llm/.env.google-search` `.env.providers` `.env.ocr`、`~/.git-credentials`(open-englishへpushできるPAT)。
4. `aruaru-llm.service`(WorkingDirectory=`/root/aruaru-llm`、ExecStart=`target/release/aruaru-llm`、`ARUARU_LLM_BIND=127.0.0.1:4600`)を設置して起動。`deploy/install-knowledge-sync.sh`で知識退避タイマーを設置。
5. **知識は自動で戻る**: 起動直後のログに`knowledge sync from ...: N new entr(ies) merged`が出る。`curl localhost:4600/v1/knowledge/search`で件数を確認。
6. 戻らないもの(別途): ニュースの生きているDB(`data/news_by_country.json`、見出しは`open-english-news-archive`に退避済み)、モデルの重み(再ダウンロード)、秘密。
確認済み(2026-10-06): GitHubのみからの復元、WEB停止時のGitHub切替、タイマー発火によるpush。VPS丸ごとの再構成そのものの通し試験は未実施。

**English**
1. Needs `git`, `jq`, `curl`, Rust, and sibling dependency repos (`/root/RPoem`, `/root/open-cuda`).
2. `git clone https://github.com/aon-co-jp/aruaru-llm /root/aruaru-llm`, then `cargo build --release`.
3. Restore secrets out of band (not in Git): `.env.google-search`, `.env.providers`, `.env.ocr`, `~/.git-credentials` (a PAT that can push to open-english).
4. Install `aruaru-llm.service` (WorkingDirectory `/root/aruaru-llm`, ExecStart `target/release/aruaru-llm`, `ARUARU_LLM_BIND=127.0.0.1:4600`) and start it; run `deploy/install-knowledge-sync.sh` for the backup timer.
5. **Knowledge restores itself**: look for `knowledge sync from ...: N new entr(ies) merged` in the log; check the count with `curl localhost:4600/v1/knowledge/search`.
6. Not restored automatically: the live news DB (headlines are archived in `open-english-news-archive`), model weights (re-download), secrets.
Verified (2026-10-06): restore from GitHub only, GitHub fallback when the web source fails, push fired by the timer. A full from-scratch VPS rebuild has **not** been rehearsed.

## 知識の日本語検索の強化・AI学習コメントの学習とGitHub自動バックアップ(2026-10-08)
- **日本語検索**: `knowledge.rs`の`words()`が、連続するCJK文字を2文字ずつの断片(バイグラム)で比べるようにした(英数字の語は従来どおり)。以前は日本語の文が1語に潰れ、ほぼ拾えなかった。テスト3件合格(`cargo test knowledge::`)。
- **学習**: `audiocafe.tokyo`のPHPが`POST /v1/knowledge/add`で、検証済みの価格・AIの誤りの訂正・「料金は推測で書かない」ルールを教える。AIが書いた価格欄は24件中23件が実価格と不一致だったため。
- **GitHub自動バックアップ**: `knowledge-sync-push`(6時間ごと+PHPクロール直後)が`open-english/data/knowledge/knowledge.json`へ保存。起動時に`sync_from_remote`で復元。空のストアでの復元テスト合格(19件を18秒)。
- **お引越し**: 知識はGitHubから自動復元される。復元されないのは`.env.providers`等のAPIキーだけ(手動で再配置)。systemdの実物は`audiocafe-tokyo`の`deploy/vps/`。
- 注意: 知識は公開リポジトリに保存されるため、公開してよい内容だけを入れる。

## Better Japanese knowledge search, AI-comment learning and automatic GitHub backup (2026-10-08)
- **Japanese search**: `words()` in `knowledge.rs` now compares runs of CJK characters as overlapping 2-character fragments (bigrams); alphanumeric words behave as before. Previously a Japanese sentence collapsed into one token and was almost never matched. 3 tests pass (`cargo test knowledge::`).
- **Learning**: the PHP of `audiocafe.tokyo` teaches verified prices, corrections of the AI's mistakes and the rule "never guess prices" via `POST /v1/knowledge/add` (23 of 24 AI-written prices disagreed with the real ones).
- **Automatic GitHub backup**: `knowledge-sync-push` (every 6 h and right after the PHP crawl) saves to `open-english/data/knowledge/knowledge.json`; `sync_from_remote` restores it at start-up. Restore test on an empty store passed (19 entries in 18 s).
- **Moving servers**: knowledge restores itself from GitHub. Only the API keys (`.env.providers` etc.) must be placed by hand. The real systemd files are in `deploy/vps/` of `audiocafe-tokyo`.
- Note: knowledge is stored in a public repository, so only put publishable content in it.

## JSON処理の使用方針: `parse_secure_sonic`(速い型+安全)と`sonic-rs`は使える所では全て使う(2026-10-14、ユーザー指示)
正本は[`RS-JSON`](https://github.com/aon-co-jp/RS-JSON)(`README.md`・`PORTING.md`)。

**方針**: JSONを読む所は、使える所では**全て** `rust-json`(RS-JSON)の `parse_secure_sonic`(`fast`機能、結果は`sonic_rs::Value`)を使う。
`sonic-rs` を直接使える所(型付き構造体への読み込みなど)も同様。使えない所は `parse_secure`(結果は`serde_json::Value`)を使う。
外部から来る信頼できない入力(HTTP本文・ネットワーク越しのファイル・アップロード)は、**必ずサイズ・深さ・重複キーの上限つき**で読む。

**実測(7 MB・4万件)**: `serde_json` 200 ms / `parse_secure` 217 ms / **`parse_secure_sonic` 83 ms(約2.4倍速い、防御込み)** / `sonic-rs`そのまま 33 ms(約6倍、防御なし)。
`serde_json::Value`に読む限りSIMDでもほぼ速くならない(188 ms)。**速さの源は、結果を連続した領域に詰め込む値の型**。

**使える所 / 使えない所**
- ✅ 読み取り中心で、JSONを大きく・頻繁に読む所(キャッシュJSON、設定、APIの応答、ログ、インデックス)。
- ✅ 外部からの入力の入口(速さより防御が目的。`parse_secure`でも可)。
- ⚠️ `serde_json::Value`を関数間で受け渡している既存コードは、型を`sonic_rs::Value`に直す書き換えが要る(`JsonValueTrait`/`JsonContainerTrait`のメソッド)。書き換えずに`serde_json::Value`へ読むと速くならない。
- ❌ WASM(`rust_json::light`、依存ゼロ)、PHP/JavaScript側のJSON処理(Rustの部品は使えない)。
- 注意: 依存先の`sonic-rs`は内部で`unsafe`を使う(自前コードに`unsafe`は無い)。小さなJSON(数KB)では体感差が出ない。速度目的の書き換えは、実測で効く所から。

**導入**: `Cargo.toml`に `rust-json = { path = "../RS-JSON", features = ["fast"] }`。使用例は`RS-JSON/README.md`。

**このリポジトリでの適用状況**: `knowledge::sync_from_remote`(外部取り込み)は`parse_secure`に適用済み。ほかは未適用(方針のみ)。次の対象: 検索インデックス・ロード時のJSON。

## JSON handling policy: use `parse_secure_sonic` (fast type + secure) and `sonic-rs` everywhere they can be used (2026-10-14, user instruction)
The source of truth is [`RS-JSON`](https://github.com/aon-co-jp/RS-JSON) (`README.md`, `PORTING.md`).

**Policy**: wherever JSON is read, use `parse_secure_sonic` from `rust-json` (RS-JSON, the `fast` feature; the result is `sonic_rs::Value`) **everywhere it can be used**.
Use `sonic-rs` directly where possible too (e.g. reading into typed structs). Where it cannot be used, use `parse_secure` (result: `serde_json::Value`).
Untrusted input from outside (HTTP bodies, files over the network, uploads) **must always be read with size / depth / duplicate-key limits**.

**Measured (7 MB, 40,000 records)**: `serde_json` 200 ms / `parse_secure` 217 ms / **`parse_secure_sonic` 83 ms (about 2.4x faster, with the defences)** / raw `sonic-rs` 33 ms (about 6x, no defences).
Reading into `serde_json::Value` gives almost no speed-up even with SIMD (188 ms). **The speed comes from the value type that packs the result into contiguous memory.**

**Where it can / cannot be used**
- ✅ Read-heavy code that parses large or frequent JSON (cache JSON, config, API responses, logs, indexes).
- ✅ Entrances for outside input (the goal there is defence rather than speed; `parse_secure` is fine).
- ⚠️ Existing code that passes `serde_json::Value` between functions needs its type changed to `sonic_rs::Value` (methods from `JsonValueTrait` / `JsonContainerTrait`). Reading into `serde_json::Value` without that change is not faster.
- ❌ WASM (`rust_json::light`, zero dependencies) and the JSON handling on the PHP / JavaScript side (Rust components cannot be used there).
- Note: the `sonic-rs` dependency uses `unsafe` internally (our own code has none). For small JSON (a few KB) the difference is not noticeable; do speed-motivated rewrites where measurements show a benefit.

**Adoption**: in `Cargo.toml`, `rust-json = { path = "../RS-JSON", features = ["fast"] }`. Example in `RS-JSON/README.md`.

**Status in this repository**: `knowledge::sync_from_remote` (outside import) already uses `parse_secure`. Everything else is not yet applied (policy only). Next targets: search indexes and JSON loaded at start-up.

## API 方針: REST 全廃・gRPC / Connect へ (2026-11-04)

### 日本語
- **REST は全廃する。新しい API は REST で作らない。** ブラウザや `curl` から呼ぶ API は **Connect の単純呼び出し**(`POST /<パッケージ>.<Service>/<Method>`、`content-type: application/json`、本文は JSON)、サービス間は **gRPC / gRPC-web**(RPoem の管理API `open_runo.admin.v1.Admin` と同じ流儀)。ストリーミングは、必要になってから。
- **既存の REST は、触るときに順に移す。** 移したら旧 REST の経路は消す。移行が終わるまで残るが、**新規には追加しない**。
- **JSON は RS-JSON に統一**する(読み込みは `from_str_secure` / `parse_secure` でサイズ・深さ・重複キーの上限つき、書き出しは `rust_json::fast`)。ブラウザの WASM に `prost` は入れない(サイズが増えるため)。
- **例外(自分たちの API ではなく、相手の規格が決めているもの)**: ACME、OAuth/OIDC のコールバック、第三者(GitHub など)からの Webhook、ヘルスチェック、静的ファイル配信、OTLP など。これらは規格どおりの HTTP を受ける。
- 参考実装: `RPoem`(`open_runo.admin.v1.Admin`)、`open-easy-web`(`server/src/connect.rs`、`proto/open_easy_web/admin/v1/admin.proto`)。
- ※ このリポジトリのコードが既に方針どおりかどうかは、各リポジトリの現状による。この節は方針の記録であり、コードの変更を意味しない。

### English
- **REST is being abolished. New APIs are not built as REST.** APIs called from browsers or `curl` use **Connect unary calls** (`POST /<package>.<Service>/<Method>`, `content-type: application/json`, JSON body); service-to-service calls use **gRPC / gRPC-web** (the same style as RPoem's admin API `open_runo.admin.v1.Admin`). Streaming only when it is actually needed.
- **Existing REST endpoints are migrated one by one, when they are touched.** The old REST route is removed once migrated. Until migration is finished they remain, but **no new ones are added**.
- **JSON is unified on RS-JSON**: reading with `from_str_secure` / `parse_secure` (size, depth and duplicate-key limits), writing with `rust_json::fast`. `prost` is not put into browser WASM (it would grow the binary).
- **Exceptions (shapes dictated by someone else's standard, not our own API)**: ACME, OAuth/OIDC callbacks, webhooks from third parties (e.g. GitHub), health checks, static file serving, OTLP, and so on. These accept plain HTTP as the standard requires.
- Reference implementations: `RPoem` (`open_runo.admin.v1.Admin`) and `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Note: whether the code in a given repository already follows this policy depends on that repository. This section records the policy; it does not mean the code has been changed.

### Deutsch (KI-Übersetzung, redaktionell überarbeitet; noch nicht von Muttersprachlern geprüft)
- **REST wird vollständig abgeschafft; neue APIs entstehen nicht mehr als REST.** APIs, die aus dem Browser oder per `curl` aufgerufen werden, verwenden **unäre Connect-Aufrufe** (`POST /<Paket>.<Service>/<Methode>`, `content-type: application/json`, JSON-Body). Zwischen Diensten kommen **gRPC bzw. gRPC-web** zum Einsatz, im selben Stil wie die Admin-API von RPoem (`open_runo.admin.v1.Admin`). Streaming wird erst ergänzt, wenn es tatsächlich gebraucht wird.
- **Bestehende REST-Endpunkte werden schrittweise umgestellt, und zwar immer dann, wenn man ohnehin an ihnen arbeitet.** Sobald ein Endpunkt umgestellt ist, wird die alte REST-Route entfernt. Bis zum Abschluss der Migration bleiben die übrigen bestehen, es werden aber **keine neuen mehr hinzugefügt**.
- **JSON wird einheitlich über RS-JSON verarbeitet**: Gelesen wird mit `from_str_secure` bzw. `parse_secure` (mit Obergrenzen für Größe, Verschachtelungstiefe und doppelte Schlüssel), geschrieben mit `rust_json::fast`. `prost` wird nicht in das Browser-WASM aufgenommen, da es die Datei vergrößern würde.
- **Ausnahmen (Schnittstellen, deren Form ein fremder Standard vorgibt und die nicht zu unserer eigenen API gehören)**: ACME, OAuth/OIDC-Callbacks, Webhooks von Drittanbietern (z. B. GitHub), Health-Checks, die Auslieferung statischer Dateien, OTLP und Ähnliches. Diese nehmen, wie vom jeweiligen Standard verlangt, gewöhnliches HTTP entgegen.
- Referenzimplementierungen: `RPoem` (`open_runo.admin.v1.Admin`) und `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Hinweis: Ob der Code eines einzelnen Repositorys diese Richtlinie bereits erfüllt, hängt vom jeweiligen Repository ab. Dieser Abschnitt dokumentiert lediglich die Richtlinie; er besagt nicht, dass der Code schon angepasst wurde.

### Русский (перевод выполнен с помощью ИИ и отредактирован; носителями языка пока не проверен)
- **От REST мы полностью отказываемся; новые API больше не создаются в стиле REST.** API, к которым обращаются из браузера или через `curl`, используют **унарные вызовы Connect** (`POST /<пакет>.<Service>/<Method>`, `content-type: application/json`, тело запроса — JSON). Для взаимодействия между сервисами применяются **gRPC и gRPC-web** — в том же стиле, что и административный API RPoem (`open_runo.admin.v1.Admin`). Потоковая передача данных добавляется лишь тогда, когда в ней действительно возникает необходимость.
- **Существующие REST-эндпоинты переводятся поэтапно — по мере того, как с ними приходится работать.** После перевода старый REST-маршрут удаляется. Пока миграция не завершена, оставшиеся эндпоинты сохраняются, но **новые не добавляются**.
- **Работа с JSON унифицирована на основе RS-JSON**: чтение — через `from_str_secure` или `parse_secure` (с ограничениями на размер, глубину вложенности и повторяющиеся ключи), запись — через `rust_json::fast`. `prost` в браузерный WASM не включается, поскольку это увеличило бы размер файла.
- **Исключения (интерфейсы, форму которых определяет чужой стандарт, а не наш собственный API)**: ACME, обратные вызовы OAuth/OIDC, вебхуки сторонних сервисов (например, GitHub), проверки работоспособности, раздача статических файлов, OTLP и тому подобное. Такие интерфейсы принимают обычный HTTP, как того требует соответствующий стандарт.
- Эталонные реализации: `RPoem` (`open_runo.admin.v1.Admin`) и `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Примечание: соответствует ли код конкретного репозитория этой политике, зависит от самого репозитория. Данный раздел лишь фиксирует политику и не означает, что код уже изменён.

## 本番の一括切り替え手順書 (Connect移行, 2026-11-05)
前提: 呼び出し元が `connect-migration` に対応済み(`CLAUDE.md` の一覧と、引き継ぎメモの表で確認)。新旧の両対応は作らない(恒久化を避ける)。切り替え窓は数分。
1. **事前確認**: 各呼び出し元の master/main に Connect 対応コミットがあること(`gh api repos/aon-co-jp/<repo>/commits`)。VPS の呼び出し元サービスは、入れ替え用バイナリ/ファイルを先に VPS へ置いておく(まだ起動しない)。
2. **ビルド**: WSL で `cargo build --release`(VPS ではビルドしない)。`objdump -T` で glibc 2.39 以下を確認。
3. **バックアップ**: VPS の `/root/repository/aruaru-llm` の現バイナリを `.bak-YYYYMMDD` に退避(`systemctl cat` は秘密を伏せて見る)。
4. **入れ替え(順に、各段で確認)**: aruaru-llm を scp→置換→`systemctl restart aruaru-llm.service`→`curl -s -X POST localhost:4600/aruaru_llm.v1.Llm/Runtime -d '{}'` が 200 → すぐに呼び出し元(aruaru-search、audiocafe-tokyo-rust、open-english/maid-cafe-english、version-control)を順に入れ替えて再起動。
5. **確認**: 旧経路 `curl -o /dev/null -w '%{http_code}' localhost:4600/v1/chat -X POST -d '{}'` が 404、各サービスの画面/ログで aruaru-llm 呼び出しが 200。
6. **ロールバック**: 4 の途中で失敗したら、`.bak` を戻して該当サービスを再起動(aruaru-llm を戻す場合は、既に入れ替えた呼び出し元も一つ前に戻す)。
7. 成功後に `connect-migration` を main へマージ(強制pushしない)。クライアント配布物(aruaru-IME・pdf-r2l-rs・Android)は旧版が壊れるので、更新の案内を出す。
## API 方針の補足: 用語の整理と使い分け (2026-11-04)

### 日本語
**1つに統一するものではなく、別の層の話です。** 「REST の代わり」になるのは Connect・gRPC・GraphQL だけで、WASM は画面を作る技術です。

| 用語 | 何か | REST の代わりか |
|---|---|---|
| REST | URL と HTTP メソッドで操作を表す API の作り方 | 全廃する側 |
| **Connect** | `POST /サービス名/メソッド名` に JSON を送る形式。`curl` でもブラウザの `fetch` でも呼べる | はい(**採用**) |
| **gRPC** | 同じ「サービス/メソッド」の考え方を、バイナリ(protobuf)と HTTP/2 で行う形式。速く、ストリーミングも得意 | はい(**サービス同士に採用**) |
| gRPC-web | gRPC のブラウザ向けの変形版 | はい(必要なら) |
| GraphQL | 欲しい項目を問い合わせる形式 | 一部だけ(**問い合わせ(読み取り)専用**) |
| WASM | Rust をブラウザの中で動かす仕組み | いいえ(画面を作る技術) |

- **ブラウザの画面や `curl` から呼ぶ API は Connect**(POST、JSON を RS-JSON で読み書き)。ブラウザの WASM は素の gRPC を話せず、Connect なら `fetch` のままで、`prost` を足さずに済む(画面のサイズも増えない)。
- **サービス同士の通信(サーバー間、重い処理、ストリーミング)は gRPC。** 多くは Connect で足りるので、ストリーミングが必要になった所だけ gRPC に上げる。
- **GraphQL は、まとめて問い合わせたいときの読み取り専用。** 書き込み(操作)には使わない。
- **WASM は画面を作る技術として、そのまま使う。** 画面から呼ぶ API が Connect になっただけ。
- 外部の規格が形を決めるもの(ACME、OAuth のコールバック、第三者の Webhook、ヘルスチェック、静的ファイル、OTLP)は、規格どおりの HTTP のまま。
- **迷ったら Connect。**

### English
**These are not alternatives to unify; they sit at different layers.** Only Connect, gRPC and GraphQL can replace REST; WASM is a way to build the UI.

| Term | What it is | Replaces REST? |
|---|---|---|
| REST | API style that expresses operations through URLs and HTTP methods | The thing being abolished |
| **Connect** | `POST /Service/Method` with a JSON body; callable from `curl` and from a browser `fetch` | Yes (**adopted**) |
| **gRPC** | The same service/method idea over binary (protobuf) and HTTP/2; fast, good at streaming | Yes (**for service-to-service**) |
| gRPC-web | A browser-friendly variant of gRPC | Yes (if needed) |
| GraphQL | A query language: ask for exactly the fields you want | Partly (**read-only queries**) |
| WASM | Runs Rust inside the browser | No (a UI technology) |

- **APIs called from browser screens or `curl` use Connect** (POST, JSON read and written with RS-JSON). Browser WASM cannot speak plain gRPC; with Connect it stays a `fetch`, and `prost` is not needed (the binary does not grow).
- **Service-to-service calls (server to server, heavy work, streaming) use gRPC.** Connect is enough in most places; move a call to gRPC only where streaming is actually needed.
- **GraphQL is read-only**, for queries that gather several pieces of data. It is not used for writes (operations).
- **WASM stays as the way to build the UI.** The API it calls simply became Connect.
- Interfaces whose shape an outside standard dictates (ACME, OAuth callbacks, third-party webhooks, health checks, static files, OTLP) stay plain HTTP.
- **When in doubt, use Connect.**

### Deutsch (KI-Übersetzung, sprachlich überarbeitet; noch nicht von Muttersprachlern geprüft)
**Es geht nicht darum, eine einzige Technik auszuwählen – die Begriffe liegen auf unterschiedlichen Ebenen.** Als Ersatz für REST kommen nur Connect, gRPC und GraphQL in Frage; WASM dagegen ist eine Technik, mit der die Benutzeroberfläche gebaut wird.

| Begriff | Was ist das? | Ersatz für REST? |
|---|---|---|
| REST | Ein API-Stil, bei dem Operationen über URLs und HTTP-Methoden ausgedrückt werden | wird abgeschafft |
| **Connect** | `POST /Dienst/Methode` mit JSON im Body; aufrufbar mit `curl` ebenso wie mit `fetch` im Browser | ja (**gewählt**) |
| **gRPC** | Dasselbe Prinzip aus Dienst und Methode, aber über binäres protobuf und HTTP/2; schnell und gut für Streaming geeignet | ja (**zwischen Diensten**) |
| gRPC-web | Eine für den Browser angepasste Variante von gRPC | ja (bei Bedarf) |
| GraphQL | Abfragesprache, mit der man genau die benötigten Felder anfordert | teilweise (**nur für lesende Abfragen**) |
| WASM | Führt Rust-Code direkt im Browser aus | nein (Technik für die Oberfläche) |

- **APIs, die von Browseroberflächen oder per `curl` aufgerufen werden, verwenden Connect** (POST; JSON wird mit RS-JSON gelesen und geschrieben). WASM im Browser kann kein reines gRPC sprechen. Mit Connect bleibt es bei einem gewöhnlichen `fetch`, und `prost` ist nicht nötig – die WASM-Datei wird dadurch nicht größer.
- **Für die Kommunikation zwischen Diensten (Server zu Server, rechenintensive Aufgaben, Streaming) wird gRPC eingesetzt.** In den meisten Fällen reicht Connect aus; auf gRPC wird nur dort umgestellt, wo Streaming tatsächlich gebraucht wird.
- **GraphQL dient ausschließlich lesenden Abfragen**, wenn mehrere Informationen gebündelt abgefragt werden sollen. Für Schreibzugriffe (Operationen) wird es nicht verwendet.
- **WASM bleibt die Technik für die Oberfläche.** Geändert hat sich lediglich, dass die aufgerufene API nun auf Connect beruht.
- Schnittstellen, deren Form durch einen externen Standard festgelegt ist (ACME, OAuth-Callbacks, Webhooks von Drittanbietern, Health-Checks, statische Dateien, OTLP), bleiben gewöhnliches HTTP.
- **Im Zweifel: Connect.**

### Русский (перевод выполнен с помощью ИИ и литературно отредактирован; носителями языка пока не проверен)
**Это не варианты, из которых нужно выбрать один, а технологии разных уровней.** Заменить REST могут только Connect, gRPC и GraphQL; WASM же — это способ создания пользовательского интерфейса.

| Термин | Что это | Заменяет REST? |
|---|---|---|
| REST | Стиль построения API, при котором операции выражаются через URL и HTTP-методы | отменяется |
| **Connect** | `POST /Сервис/Метод` с телом в формате JSON; вызывается как из `curl`, так и из браузерного `fetch` | да (**выбран**) |
| **gRPC** | Тот же принцип «сервис — метод», но поверх двоичного формата protobuf и HTTP/2; работает быстро и хорошо подходит для потоковой передачи | да (**для связи между сервисами**) |
| gRPC-web | Адаптированный для браузера вариант gRPC | да (при необходимости) |
| GraphQL | Язык запросов: клиент запрашивает ровно те поля, которые ему нужны | частично (**только запросы на чтение**) |
| WASM | Позволяет выполнять код на Rust прямо в браузере | нет (технология для интерфейса) |

- **API, которые вызываются из браузерных интерфейсов или через `curl`, строятся на Connect** (POST; JSON читается и записывается через RS-JSON). Браузерный WASM не умеет работать с «чистым» gRPC. С Connect достаточно обычного `fetch`, а `prost` не требуется — размер WASM-файла не увеличивается.
- **Для обмена между сервисами (сервер — сервер, ресурсоёмкие задачи, потоковая передача) применяется gRPC.** В большинстве случаев достаточно Connect; на gRPC переходят только там, где потоковая передача действительно необходима.
- **GraphQL используется только для чтения**, когда нужно получить сразу несколько связанных данных. Для записи (операций) он не применяется.
- **WASM по-прежнему остаётся технологией для интерфейса.** Изменилось лишь то, что вызываемый им API теперь построен на Connect.
- Интерфейсы, форма которых задана внешним стандартом (ACME, обратные вызовы OAuth, вебхуки сторонних сервисов, проверки работоспособности, статические файлы, OTLP), остаются обычным HTTP.
- **Если есть сомнения — выбирайте Connect.**

### 本番切り替え 実施記録 (2026-11-05)
- aruaru-llm(connect-migration)と呼び出し元6サービス(aruaru-search〈VPS現行+呼び出しのみConnect〉・audiocafe-tokyo-rust・e-gov-server・open-english(+demo)・realdata-pro・open-easy-web)を一括入れ替え。PHP(audiocafe aruaru/index.php)と3スクリプトも更新。
- 確認: Runtime 200・旧`/v1/runtime` 404・全サービスactive。初回は起動待ち不足でスクリプトが自動ロールバックし、起動待ちを足して再実施し成功。
- バックアップは VPS の `/root/cutover-20261105/bak`。aruaru-ime-daemon(/opt)は未入れ替え。

## 本番切り替えの実施記録 (2026-11-05)

- aruaru-llm(`connect-migration` をWSLでリリースビルド、glibc 2.39)を VPS に反映。同時に realdata-pro・aruaru-jobs・version-control のバイナリ、audiocafe-tokyo(PHP, ff-pull)、diet_crawl.py、スクリプト3本を入れ替え。バックアップは各ファイルの `.bak-20261105`。
- 初回は起動待ち(ウォームアップ)が8秒では足りず自動ロールバック→待ちを最大120秒に伸ばして再実行し成功。Runtime 200・旧 `/v1/chat` 404・healthz 200・KnowledgeSearch 200、各サービス active、直近3分のエラー0。
- main へのマージは未実施(別途判断)。

## 2026-11-06 依存更新(version-control通知) / Dependency update

**日本語**: 直接依存を更新: `reqwest` 0.12→0.13(feature `rustls-tls`→`rustls`)、`rand` 0.8→0.10(`thread_rng`→`rng()`、`SliceRandom`→`IndexedRandom`)、`sonic-rs` 0.3→0.5、`base64` 0.22→0.23、`sha2` 0.10→0.11。`zstd-sys` は 2.1.2 へ(RUSTSEC-2026-0410 解消)。`cargo build`・`cargo test`(183件)通過。
**直せないもの(上流待ち)**: `h2` 0.3.27・`protobuf` 2.28・`zip` 0.6.6・`zstd-safe` 5 は、すべて**オプション機能 `nllb-translate`** の `rust-bert` 0.23.0(crates.io の最新版)が固定している推移的依存(`cached-path`→`reqwest` 0.11→`h2` 0.3、`rust_tokenizers`→`protobuf` 2、`zip` 0.6)で、`Cargo.toml` からは上げられない。既定ビルド(`default = []`)と本番VPSのバイナリには含まれない。解消するには `rust-bert` の新版を待つか、`nllb-translate` を別実装へ置き換える。`sonic-rs` 0.3.17 が `Cargo.lock` に残るのは兄弟リポジトリ(`open-cuda`・`open-web-server`)の固定で、各リポジトリ側で更新する。
**English**: Updated direct deps (reqwest 0.13, rand 0.10, sonic-rs 0.5, base64 0.23, sha2 0.11) and `zstd-sys` 2.1.2; build and 183 tests pass. h2 0.3 / protobuf 2 / zip 0.6 / zstd-safe 5 are pinned by `rust-bert` 0.23.0 (latest on crates.io) behind the optional `nllb-translate` feature; they are not in the default build or the production binary and cannot be bumped from this `Cargo.toml`.

## JSON の読み込みを RS-JSON の安全モード+sonic-rs に統一(2026-11-06)
方針(`F:\runo\CLAUDE.md`「JSON処理の使用方針」)どおり、信頼できない入力は `rust_json::parse_secure_sonic`(SIMD版・サイズ/深さ/重複キーの上限つき)で検査してから型へ読む。
- **重大な発見と修正**: `http_json` の安全な受信(`ResponseJson::json`)は、**実際には一度も働いていなかった**。`reqwest` の `json` 機能は他の依存(RPoem のルーター・OpenTelemetry)がフィーチャを足すと有効になり(フィーチャの統合)、固有メソッド `Response::json` がトレイトの同名メソッドより優先され、`serde_json` で素通しに読まれていた(`JsonHttpError is never constructed` の警告が手掛かり)。メソッド名を **`json_secure()`(受信)/`json_sonic(&body)`(送信)** に変え、全呼び出し(29か所:chat_providers・web_search・github_search・youtube_search・news_geo・self_update・google_search_fallback_health)を置き換えた。これで外部 API の応答は必ず上限・重複キー検査を通る。
- 受信は「検査(`parse_secure_sonic`)→ 型へ直接(`sonic_rs::from_str`)」で、中間の `serde_json::Value` を作らない。知識の取り込み(`knowledge::parse_remote_entries`)・埋め込みの地理データ(`geo_content`)・ニュースのディスク保存(`news_feed`)・OCR の manifest(`ocr_data`)・Connect 入口(`connect.rs`)も同様。`rust_json` が返す値は sonic-rs 0.3 の型なので、トレイトは `rust_json::fast::sonic_rs` から使う。
- 外から来る HTTP 本文(`Json::from_body`)は RPoem の `open-runo-poem-compat`→`secure_json` 側で処理(重複キー・深さ・16 MiB の検査つき、RPoem の記録を参照)。
- **実測(前後)**: 知識の取り込み 約3.7 MB・2万件(`parse_remote_entries`)が **約70 ms → 約39 ms(約1.8倍)**。`cargo test --release bench_parse_remote_entries -- --ignored --nocapture` で再現。
- 試験: `cargo test` 全通過(183件、モデルのある環境。モデルの無い新規 clone では `scoring`/`security`/`intrusion_detection` の14件がモデル不足で失敗する=従来どおり)。

**VPS 反映(2026-11-06)**: WSL でリリースビルド(`cargo build --release --locked`、既定フィーチャ)→ `objdump -T` で最大 GLIBC_2.39(VPS は 2.39)を確認 → scp → `/root/aruaru-llm/target/release/aruaru-llm.bak-20261106` へバックアップ → 入れ替え → `systemctl restart aruaru-llm` → `/healthz`・`/`・`Generate` が 200 を確認(失敗時は自動で戻す手順)。VPS 上の clone(未コミットの変更あり)には触っていない。戻すには `cp -p aruaru-llm.bak-20261106 aruaru-llm && systemctl restart aruaru-llm`。

**モデルの無い環境の試験(2026-11-06)**: 埋め込みモデル(`models/multilingual-e5-small/`、git に入れていない)を要る14件の試験(`scoring`・`security`・`intrusion_detection`)は、モデルが無いと失敗していた。`scoring::embedding_model_available()` と `skip_without_embedding_model!()` を足し、モデルが無い環境(新規 clone・CI)ではスキップ(理由を標準エラーに出す)、有る環境では従来どおり実行する。確認: `ARUARU_LLM_EMBED_MODEL_DIR=/nonexistent` でもモデルありでも `scoring::`/`security::`/`intrusion_detection::` の15件が全通過。

## 共有基盤としての受付制御(2026-11-09、配備方針「共有基盤は1回だけ導入」)
- 重い推論系のConnect呼び出し(Generate/GenerateSpeculative/GenerateWithSearch/GenerateQwen/GenerateDeepseek/Translate/Transcribe/Ocr系)に`src/admission.rs`の受付制御を追加。同時実行上限(既定2)、サイト別の待ち行列+ラウンドロビン配分、行列長の上限(全体32・サイト別8、超過は即429 `resource_exhausted`)、待ち時間上限30秒(503 `unavailable`)、実行時間上限120秒(504)。時間切れ後も計算が終わるまで枠は返さない。エラーはConnect形式`{code,message}`。
- サイト識別は`x-tenant`ヘッダ→`x-forwarded-host`→`anonymous`。前面のopen-web-serverが付ける前提(直接公開だと偽装で公平配分を回避できる)。環境変数: `ARUARU_LLM_MAX_CONCURRENT`/`_MAX_QUEUE`/`_MAX_QUEUE_PER_TENANT`/`_QUEUE_WAIT_SECS`/`_EXEC_TIMEOUT_SECS`。
- 調査結果: プロセスは1つ(VPS 127.0.0.1:4600)で全サイト共有、重い処理は元から`spawn_blocking`、モデルの二重ロードなし。Chat/Classify等の軽い呼び出しは受付制御の対象外。サイトごとに同梱・個別起動している箇所は見つからなかった。
- 実測(WSL Ubuntu、GPT-2実推論、同時2・行列6・サイト別4・待ち20秒・実行60秒): 単発1件が60秒でタイムアウト(504。推論自体が遅い)。12件同時(A8/B2/C2)では、行列超過が約2.4〜2.7秒で429、行列待ちが約22.5秒で503。公平性は推論が遅すぎて実機では測れず、`admission`の単体測定(同時2・1件100ms、A20件/B3件/C3件を同時投入)で全完了A=1400ms、B=767ms、C=867ms(FIFOなら全て約1400ms)。VPS本番(2026-11-09反映、旧バイナリ`target/release/aruaru-llm.before-20261109adm`)で3サイト同時の2トークン生成は3件とも200。

### Shared-base admission control (2026-11-09)
Heavy Connect calls (Generate*/Translate/Transcribe/Ocr*) pass `src/admission.rs`: max concurrency (default 2), per-site queues served round-robin, queue caps (32 total / 8 per site, overflow = immediate 429), queue wait 30 s (503), execution cap 120 s (504; the slot is freed only when the computation really ends). Site id: `x-tenant` > `x-forwarded-host` > `anonymous` (set by the front open-web-server). Env: `ARUARU_LLM_MAX_CONCURRENT`, `_MAX_QUEUE`, `_MAX_QUEUE_PER_TENANT`, `_QUEUE_WAIT_SECS`, `_EXEC_TIMEOUT_SECS`. Measured: real GPT-2 inference is too slow in WSL for a fairness run (a single call hit the 60 s cap); overflow gave 429 in ~2.5 s, queue timeout 503 in ~22.5 s; the unit-level load test (2 slots, 100 ms jobs, A=20/B=3/C=3) finished A=1400 ms, B=767 ms, C=867 ms vs ~1400 ms under FIFO. Deployed to the VPS 2026-11-09 (3 simultaneous sites: all 200).

## 依存更新の記録: RUSTSEC-2026-0258(h2 0.3.27)(2026-11-11、version-control 通知への対応)
- `h2 0.4.19` → `0.4.20` へ更新(`cargo update -p h2@0.4.19`、`cargo check --locked` 成功)。
- **`h2 0.3.27`(`RUSTSEC-2026-0258`、修正は `>=0.4.16`)は上流が未対応で直せない**: 経路は `rust-bert 0.23.0`(crates.io の最新) → `cached-path ^0.6` → `reqwest 0.11` → `hyper 0.14` → `h2 0.3`。`cached-path` 自体は 0.10.1 まであるが、`rust-bert` が `^0.6` に固定している。
- **影響範囲**: この経路は任意フィーチャ `nllb-translate` のみ(既定ビルド・VPS 本番の `--locked` 既定フィーチャには入らない)。`nllb-translate` を有効にして配備する場合は、この脆弱性(空の DATA フレームの無制限受け入れ=DoS)を承知の上で使うこと。
- 次の手: `rust-bert` の新版(`cached-path` 更新)を待つ。それまで version-control の通知は残る。

### Dependency note: RUSTSEC-2026-0258 (h2 0.3.27) (2026-11-11)
`h2 0.4.19 → 0.4.20` updated. `h2 0.3.27` cannot be fixed here: it comes only via the optional `nllb-translate` feature (`rust-bert 0.23.0`, the latest → `cached-path ^0.6` → `reqwest 0.11` → `hyper 0.14` → `h2 0.3`). Default builds and the VPS production build are unaffected. Wait for a `rust-bert` release that updates `cached-path`.
