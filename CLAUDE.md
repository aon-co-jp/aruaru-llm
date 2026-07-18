# 開発方針＆開発環境ルール(aruaru-llm)

作業ドライブは`F:\open-runo`。この節は[`open-raid-z`](https://github.com/aon-co-jp/open-raid-z)の
`CLAUDE.md`を正本とし、各プロジェクトへコピーして同期する方針に準じる。
GitHubリポジトリ: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm)。

> ⚠️ **正直な開示(最重要)**: このリポジトリ名は「LLM」を冠しているが、
> **v0.1.0時点では実際のニューラルネットワーク推論を一切行わない**。
> 中身はキーワードマッチングによる**ルールベースの応答プレースホルダー**
> であり、`e-gov.info`の`src/chat_commerce.rs`にあった同種のロジックを、
> 複数プロジェクトから再利用できる独立サービスとして切り出したもの。
> 「AI」「LLM」を名乗る以上、この限界を隠さず常に明記すること。

## このプロジェクトの役割

`aruaru`エコシステム(aruaru-tokyo・aruaru-db・e-gov.info・karu.tokyo等)
共通の「AIチャットコマース」応答ロジックを提供する、独立したHTTPサービス。
各サイトがそれぞれ個別にチャット応答ロジックを実装するのではなく、この
サービスにHTTP経由で問い合わせる構成にすることで、将来実際のLLM推論に
差し替える際の変更箇所を1箇所に集約する。

### なぜ今すぐ本物のLLM推論を実装しないか

[`open-cuda`](https://github.com/aon-co-jp/open-cuda)(このエコシステムの
GPUランタイム)の現状(2026-07-18調査時点)は、CPUバックエンドと実Vulkan
経由のvector_add/matmulまでは実装済みだが、**LLM推論に不可欠な
Attention機構・行列積(GEMM)の実装は`opencuda-blas`クレートにおいて
明示的に「Phase 3」として先送りされたスタブ**(`bail!("not yet
implemented (Phase 3)")`)の段階。加えて、Qwen3-14B等の実モデル重みの
入手・ライセンス確認、推論に必要なVRAM容量の確保、量子化(int4等、これも
同様にスタブ)といった前提条件が未整備。これらが揃うまで、本物の
ニューラルLLM推論をこのリポジトリに実装することは時期尚早と判断する。

### 現状の実装(v0.1.0、ルールベースプレースホルダー)

- キーワードマッチングによる意図分類(申請/購入/与信/不動産等のカテゴリ)
- 各カテゴリに対応した定型応答文
- 将来、本物のLLM推論(または外部LLM APIの薄いラッパー)に差し替える際、
  **HTTP APIの入出力契約(`POST /v1/chat` → `{"reply": "...", "engine":
  "..."}`)は変えずに内部実装だけ差し替えられる**ように設計する。
  `engine`フィールドには常に現在の実装方式(`"rule-based-v0"`等)を
  正直に返し、呼び出し側が「本物のAIかどうか」を判別できるようにする。

## 技術スタック

`e-gov.info`と同じ方針(2026-07-18更新のPoem判断基準に基づく): 単純な
HTTPサービスとして`poem`クレートを直接利用する。DB非依存・1バイナリ完結。

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "e-gov.info"(任意)}` を
  受け取り `{"reply": "...", "engine": "rule-based-v0-opencuda-cpu",
  "matched_intent": "..."}` を返す。`tenant`は未登録でも応答は返す
  (可用性を落とさないため)。
- `POST /admin/tenants` — テナント(呼び出し元ドメイン)を動的登録する
  (`{"host": "...", "label": "..."}`)。`x-admin-token`ヘッダ認証
  (`E_GOV_LLM_ADMIN_TOKEN`環境変数で設定、未設定時は無認証)。
- `GET /admin/tenants` — 登録済みテナント一覧。
- `DELETE /admin/tenants/:host` — テナント登録解除。
- `GET /healthz` — ヘルスチェック。

## 「分身の術」構成(2026-07-18追記、正本はopen-raid-z参照)

`open-web-server`と同じ設計思想により、**このサービスは1インスタンスを
複数ドメイン(e-gov.info・aruaru-tokyo・karu.tokyo等)が共有する**。
ドメインを追加するたびに新しい`aruaru-llm`プロセスを個別インストール・
起動する必要はない——`src/tenants.rs`の`TenantRegistry`(`RwLock`による
プロセス内共有状態、再起動不要で実行時追加・削除可能)と、上記
`/admin/tenants`系APIがこれを実現する。**管理は`open-easy-web`側から
行う想定**(`open-easy-web/server/src/appserver_registration.rs`を拡張し、
この`/admin/tenants`APIを呼び出す統合は未着手、次回以降の実装対象)。

マルチCPU/マルチコア/マルチスレッド対応: `#[tokio::main]`は既定の
multi_threadフレーバー(`current_thread`への固定なし)。CPU計算
(bag-of-wordsスコアリング)は`opencuda_cpu::CpuDevice`が
`std::thread::available_parallelism()`で検出した全論理コアへ
`rayon`経由で並列ディスパッチする。

## 関連プロジェクト

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — 将来の実推論バックエンド候補(GPUランタイム、現状はPhase 1-2のみ実装済み)。SET構成の相方
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — 本サービスの最初の呼び出し元(`src/chat_commerce.rs`のロジックをここに集約する想定)。「分身の術」構成の最初のテナント候補
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — 本サービスの管理(テナント登録・削除)を行う想定の管理ツール(統合は未着手)
- [aruaru-tokyo](https://github.com/aon-co-jp/aruaru-tokyo-server) — 将来の呼び出し元候補
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — 開発ルールの正本

## HANDOFF

- **2026-07-18 「分身の術」構成(共有インスタンス化)を実装、ビルド未検証で
  中断**: ユーザー指示「open-cudaとaruaru-llmとRPoemやopen-cosmoや
  open-raid-zやaruaru-dbなどは...ドメイン毎にインストールする必要は
  ないように」「管理はopen-easy-webで行なうように」に基づき、
  `src/tenants.rs`(`TenantRegistry`)を新設し、`main.rs`に
  `POST /admin/tenants`・`GET /admin/tenants`・
  `DELETE /admin/tenants/:host`(`x-admin-token`ヘッダ認証)を配線、
  `/v1/chat`に任意の`tenant`フィールドを追加した。**このパスでは
  `cargo build`/`cargo test`による実検証が完了していない**
  (直前にビルド中断・プロセスロックが発生し、ユーザー指示で
  ビルド確認自体を中止してドキュメント整備を優先したため)。
  次回以降: (1) `cargo build`/`cargo test`で実際に動くことを確認する
  (特に`poem::Route`の`.at()`に`post(...).get(...)`のメソッドチェーンが
  正しいAPIかどうか、`Path<String>`抽出子の使い方に誤りがないか等、
  未検証のまま残っている点に注意)、(2) `open-easy-web`側から実際に
  この`/admin/tenants`APIを呼び出す統合(`appserver_registration.rs`の
  拡張)、(3) `open-cuda`・`RPoem`・`RCosmo`・`open-raid-z`・`aruaru-db`
  への同パターン展開は本リポジトリの範囲外(それぞれのリポジトリで
  対応、`open-raid-z/CLAUDE.md`に引き継ぎ事項として記録済み)。

- **2026-07-18 新規作成**: ユーザー指示により、`e-gov.info`の
  `chat_commerce.rs`と同等のルールベース応答ロジックを、独立したHTTP
  サービスとして新規プロジェクト化。実LLM推論は`open-cuda`側の
  Phase 3(BLAS/Attention)完成待ちであることを明記。次回以降:
  (1) `e-gov.info`側を、ローカルの`chat_commerce.rs`直接呼び出しから
  この`aruaru-llm`へのHTTP問い合わせに置き換えるかどうかの検討、
  (2) `open-cuda`のPhase 3進捗の定期確認、(3) 実LLM連携時のモデル
  選定・ライセンス・VRAM要件の調査。
- **2026-07-18 open-cudaとのSET構成を実装(コード上の実連携)**:
  ユーザー指示「open-cudaとSETでaruaru-llmも実装して」に基づき、
  `Cargo.toml`に`opencuda-core`/`opencuda-cpu`をpath依存として追加し、
  `src/scoring.rs`で実際にopen-cudaの`GpuDevice`実行パイプライン
  (`alloc_buffer`→`copy_from_host`→`launch_kernel`→`synchronize`→
  `copy_to_host`、`examples/vector_add`と同一パターン)を呼び出す設計に
  変更した。具体的には、ユーザー発話と各インテントの固定語彙
  bag-of-wordsベクトルを組み立て、加算ではなく**要素積カーネル**を
  `opencuda_cpu::CpuDevice`上で実行し、その結果をホスト側で合計して
  ドット積スコア(intent分類のスコアリング)とする。これは
  Cargo依存だけの見せかけの連携ではなく、`/v1/chat`へのリクエストごとに
  実際に`launch_kernel`が呼ばれる。**ただし正直に言えば、これは本物の
  ニューラル推論(埋め込み+Attention等)ではなく、固定語彙への
  bag-of-wordsドット積という極めて単純なベクトル演算**であり、
  「LLM」という名前が示唆するものとの乖離を`scoring.rs`冒頭にも
  明記した。次回以降: open-cudaの`opencuda-blas`(Phase 3、GEMM/
  Attention)が実装され次第、この単純なドット積スコアリングを実際の
  埋め込みベクトル類似度計算に置き換える余地がある。
