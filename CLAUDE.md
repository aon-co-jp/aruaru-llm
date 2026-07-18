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

- `POST /v1/chat` — `{"message": "..."}` を受け取り `{"reply": "...",
  "engine": "rule-based-v0"}` を返す。
- `GET /healthz` — ヘルスチェック。

## 関連プロジェクト

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — 将来の実推論バックエンド候補(GPUランタイム、現状はPhase 1-2のみ実装済み)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — 本サービスの最初の呼び出し元(`src/chat_commerce.rs`のロジックをここに集約する想定)
- [aruaru-tokyo](https://github.com/aon-co-jp/aruaru-tokyo-server) — 将来の呼び出し元候補
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — 開発ルールの正本

## HANDOFF

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
