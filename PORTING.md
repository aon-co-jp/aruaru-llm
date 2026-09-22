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
