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

> ⚠️ **正直な開示(最重要、2026-07-22更新)**: リポジトリ名は「LLM」を
> 冠しているが、**自己回帰デコーダによる対話文生成はまだ実装していない**。
> 2026-07-21以降、`open-cuda`の`opencuda-bert`クレート
> (multilingual-e5-small、MIT、100言語対応)による実際の文埋め込み+
> コサイン類似度分類へ移行済み(旧: 固定語彙bag-of-wordsドット積)。
> 意味理解の質は大きく向上したが、これは検索・分類向けのエンコーダで
> あり、文章を生成する能力ではない。詳細・理由は [CLAUDE.md](CLAUDE.md)
> を参照。

## open-cudaとのSET構成

[`open-cuda`](https://github.com/aon-co-jp/open-cuda)(このエコシステムの
GPUランタイム)の`opencuda-core`/`opencuda-cpu`/`opencuda-blas`/
`opencuda-bert`をpath依存として使う。`/v1/chat`へのリクエストごとに、
`opencuda-bert`がmultilingual-e5-smallのforward passを実行してメッセージを
埋め込みベクトルへ変換し(内部で`opencuda-blas`のGEMM/Attentionカーネルを
実際に呼び出す)、各インテント代表文の埋め込み(起動時に一度計算しキャッシュ)
とのコサイン類似度で意図分類する。Cargo依存だけの見せかけの連携ではなく、
実行時に本当にopen-cudaの演算パイプラインを通る(2026-07-22、実際に
サーバーを起動し`POST /v1/chat`への応答を確認して検証済み)。

ただし、これは本物のニューラルLLM推論(対話文生成)ではない。エンコーダの
forward passのみで、自己回帰デコーダは未実装。GPU専用の高速パス
(`GemmPath::CuBlas`/`RocBlas`/`OneMkl`)も引き続きスタブのまま(CPU/
Vulkan汎用パスは実装済み)。詳細はopen-cuda側の`CLAUDE.md`のHANDOFF節を
参照。

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
