# aruaru-llm

**開発開始日: 2026-07-18**(このリポジトリのGitHub作成日)

Python向けのAIライブラリとLLMをRust向けの書き直しを開始しました。
ベースになったのは、このaruaru-llm(開発途中)＋
[open-cuda](https://github.com/aon-co-jp/open-cuda)(Windows＋MAC＋LINUX互換
＆ INTEL＋AMD＋nVIDIA互換を開発途中です)。

`aruaru`エコシステム(aruaru-tokyo・aruaru-db・e-gov.info・karu.tokyo等)
共通の「AIチャットコマース」応答サービス。各サイトが個別にチャット応答
ロジックを持つのではなく、このHTTPサービスに問い合わせる構成にすることで、
将来実際のLLM推論に差し替える際の変更箇所を1箇所に集約する。

> ⚠️ **正直な開示(最重要)**: リポジトリ名は「LLM」を冠しているが、
> v0.1.0時点では実際のニューラルネットワーク推論を一切行わない。中身は
> 固定語彙に対する**bag-of-wordsドット積**による、単純なルールベースの
> 意図分類。詳細・理由は [CLAUDE.md](CLAUDE.md) を参照。

## open-cudaとのSET構成

[`open-cuda`](https://github.com/aon-co-jp/open-cuda)(このエコシステムの
GPUランタイム)の`opencuda-core`/`opencuda-cpu`をpath依存として使い、
`/v1/chat`へのリクエストごとに実際に`GpuDevice::launch_kernel`を呼び出す
(bag-of-wordsベクトルの要素積カーネル実行)。Cargo依存だけの見せかけの
連携ではなく、実行時に本当にopen-cudaのカーネル実行パイプラインを通る。

ただし、これは本物のニューラルLLM推論ではない。open-cuda側の
`opencuda-blas`(GEMM/Attention)は現状「Phase 3」として明示的に未実装
(スタブ)のため、実際の埋め込み類似度計算・Transformer推論は今後の課題。

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(任意)}` → `{"reply": "...", "engine":
  "...", "matched_intent": "..."}`
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` — テナント登録管理(`x-admin-token`ヘッダ認証)
- `GET /healthz` — ヘルスチェック

## 「分身の術」構成

`open-web-server`と同じ設計思想で、1インスタンスを複数ドメインが共有する
(ドメインごとの個別インストール不要)。管理は[open-easy-web](https://github.com/aon-co-jp/open-easy-web)
側から行う想定(統合は未着手)。詳細は[CLAUDE.md](CLAUDE.md)を参照。

## 技術スタック

Rust + [Poem](https://github.com/poem-web/poem) + [open-cuda](https://github.com/aon-co-jp/open-cuda)。
DB非依存・1バイナリ完結。

詳細な設計思想は [CLAUDE.md](CLAUDE.md) を、他プロジェクトへの移植手順は
[PORTING.md](PORTING.md) を参照してください。

## 関連プロジェクト

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPUランタイム(SET構成の相方)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — 最初の呼び出し元想定
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — 開発ルールの正本
