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

> ⚠️ **正直な開示(最重要、2026-07-25更新)**: 2026-07-25、`open-cuda`の
> `opencuda-llm`クレート(GPT-2 124M実学習済み重み、`openai-community/gpt2`)
> を統合し、`POST /v1/generate`で**実際に自己回帰的な文章生成が可能**に
> なった(自己回帰デコーダ未実装、という従来の記述は本エンドポイントに
> 限り解消)。ただし**GPT-2 124Mは2019年発表の小型モデルであり、GPT-4等
> 最新の商用LLMと同等の性能・知識は無い**。あくまで「外部LLM API契約
> 不要の自己完結型AI」としての実証段階。生成テキストは文法的には自然な
> 英語になることが多いが、意味的に正確とは限らない(幻覚しうる)。
> `/v1/chat`(意図分類、`opencuda-bert`のエンコーダによる文埋め込み+
> コサイン類似度、2026-07-21〜)は引き続き軽量・高速な定型応答振り分け
> 専用で、生成とは役割分担している(無理に統合しない設計)。詳細・理由は
> [CLAUDE.md](CLAUDE.md)を参照。

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

**2026-07-25追記(可用性フォールバック)**: `models/multilingual-e5-small/`
(470MB超)が未取得・ロード失敗の環境でもサービスを完全停止させないよう、
`scoring::classify`が自動的に旧bag-of-wordsドット積(`src/bow_fallback.rs`)
へフォールバックするようにした。`/v1/chat`の`engine`フィールドには実際に
使われた経路(`embedding-cosine-v0-opencuda-bert-cpu`または
`bow-dotproduct-v0-opencuda-cpu-fallback`)を常に正直に返す——分類精度は
フォールバック時に明確に下がる(意味理解ではなくキーワード一致のため)。

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(任意)}` → `{"reply": "...", "engine":
  "...", "matched_intent": "..."}`(意図分類、軽量・高速な定型応答)
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens": 16(任意、既定16、上限128), "tenant": "..."(任意)}`
  → `{"completion": "...", "engine": "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`
  (GPT-2 124M実重みによる自己回帰生成、本格的だが重い。プロンプトは英語推奨——
  GPT-2のBPE語彙は英語中心のため。実験例:
  `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out of your"`)
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` — テナント登録管理(`x-admin-token`ヘッダ認証)
- `GET /healthz` — ヘルスチェック

### 意図分類 vs 生成、どちらを使うか

`/v1/chat`(意図分類)と`/v1/generate`(生成)は目的が異なるため、あえて
統合していない: `/v1/chat`は定型応答への振り分け専用で軽量・高速
(埋め込みモデルのforward passのみ)、`/v1/generate`はGPT-2 124M
(548MBの重み)を使う本格的だが重い自由文生成。用途に応じて使い分けること。

## 「分身の術」構成

`open-web-server`と同じ設計思想で、1インスタンスを複数ドメインが共有する
(ドメインごとの個別インストール不要)。管理は[open-easy-web](https://github.com/aon-co-jp/open-easy-web)
側から行う想定(統合は未着手)。詳細は[CLAUDE.md](CLAUDE.md)を参照。

## 技術スタック

Rust + [Poem](https://github.com/poem-web/poem) + [open-cuda](https://github.com/aon-co-jp/open-cuda)。
DB非依存・1バイナリ完結。

詳細な設計思想は [CLAUDE.md](CLAUDE.md) を、他プロジェクトへの移植手順は
[PORTING.md](PORTING.md) を参照してください。

## インストール

2026-07-23、`install.sh`(Linux、systemdサービス登録)・`install.ps1`
(Windows、サービス登録案内)・`.github/workflows/release.yml`(タグ
push時にLinux x86_64・Windows x86_64向けバイナリを自動ビルドし
[GitHub Releases](https://github.com/aon-co-jp/aruaru-llm/releases)へ
添付)の3点セットを追加した。**正直な開示**: 起動には470MB超の
`multilingual-e5-small`モデル重み(Hugging Face配布、MIT)を別途
取得する必要がある(ライセンス上の理由でインストーラーに同梱していない、
`install.sh`/`install.ps1`にダウンロード手順を記載)。ビルドは
`../open-cuda`へのsibling path依存があるため、ソースからビルドする
場合は`open-cuda`を隣接ディレクトリへcloneしておくこと(CIでは
`release.yml`が自動でclone)。**2026-07-25追記**: `/v1/generate`
(GPT-2 124M生成)を使うには、さらに`../open-cuda/crates/opencuda-llm/models/gpt2/`
配下に`config.json`/`model.safetensors`(548MB)/`tokenizer.json`
(`openai-community/gpt2`、Hugging Face)が必要(`ARUARU_LLM_GPT2_DIR`
環境変数でパス変更可)。無い場合`/v1/generate`のみ503を返すが、
`/v1/chat`等の既存機能は引き続き正常動作する(可用性優先の設計、
`bow_fallback`と同じ思想)。

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## 関連プロジェクト

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPUランタイム(SET構成の相方)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — 最初の呼び出し元想定
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — 開発ルールの正本
