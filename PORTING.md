# PORTING.md — aruaru-llm を他プロジェクトへお引越しする際のガイド

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

`main.rs`の`poem::Route` + `Data<Arc<dyn GpuDevice>>`による依存性注入
パターン。デバイスをリクエストごとに再生成せず、起動時に1回だけ生成して
`app.data(device)`で共有する。

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

## 注意事項

- 本プロジェクトは「LLM」を名乗り、2026-07-25以降`/v1/generate`で実際の
  GPT-2 124M自己回帰生成が可能になったが、GPT-2 124M自体は小型・2019年
  モデルであり最新商用LLM(GPT-4等)と同等の性能ではない旨を、移植先でも
  必ず明記すること(誇大表示の回避、このエコシステム共通の「正直な開示」
  規約)。`/v1/chat`(意図分類)は引き続きルールベース+エンコーダの
  意味的類似度分類であり、こちらもニューラル対話生成そのものではない
  ことを混同しないこと。
