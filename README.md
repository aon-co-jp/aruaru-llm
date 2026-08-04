# aruaru-llm

> **2026-07-25 更新**: 開発方針ファイル(`CLAUDE.md`)の見出しを
> 「開発方針＆開発環境ルール」から「設計思想＆開発方針＆開発環境ルール」
> へ改名しました。プロジェクトの設計思想(何を大事にしているか)・
> 開発方針(どう進めるか)・開発環境ルール(具体的な運用規約)を明確に
> 区別して記載しています。詳細は`CLAUDE.md`を参照してください。


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
- `GET /v1/models/catalog` — インストール可能なGPT-2アーキテクチャ互換モデル
  (`gpt2`/`distilgpt2`/`gpt2-medium`/`gpt2-large`/`gpt2-xl`〈2026-07-27追加〉)
  の一覧・インストール済みID・現在使用中のモデルディレクトリを返す
  (2026-07-26追加)。
- `POST /v1/models/install` — `{"id": "distilgpt2"}`のようにカタログから
  選択したモデルをHugging Faceからダウンロードする(2026-07-27追加)。
- `POST /v1/models/select` — `{"id": "distilgpt2"}`でダウンロード済みの
  モデルへ**プロセス再起動無しで**切り替える(2026-07-27追加、読み込みに
  成功した場合のみ切り替わる設計——失敗時は現在動作中のモデルを維持)。
  正直な開示: このカタログ・切り替え機能はGPT-2アーキテクチャ互換モデル
  限定(Llama/Mistral/Qwen等、アーキテクチャの異なるモデルは現行エンジン
  ではロードできない)。詳細は`CLAUDE.md`参照。
- `POST /v1/translate`(2026-08-04追加) — `{"text": "...", "target_lang":
  "Japanese", "source_lang": "..."(任意), "tenant": "..."(任意)}` →
  `{"translation": "...", "engine": "...", "disclosure": "..."}`。
  **翻訳プラグイン(`nllb-translate` feature)がインストールされていれば**
  専用翻訳モデル(M2M100、`rust-bert`経由)で高品質な翻訳を行い、
  **インストールされていなければ**GPT-2流用の簡易実装(実用に耐えない
  ことが実HTTP検証で判明済み、常にその旨を`disclosure`に明記)へ自動
  フォールバックする。詳細は下記「翻訳プラグイン」節参照。
- `GET /v1/recommend`（2026-07-27追加） — `open-cuda`(Vulkan)/`open-directx`
  (DXGI)経由のハードウェア検出(VRAM容量)から推奨モデルサイズを算出する
  (ダウンロードは行わない)。`{"hardware": {"gpu_detected":bool,
  "detection_path":"vulkan"|"directx"|"cpu-only-fallback",
  "gpu_name":"...","vram_bytes":...,"cross_check_agreement":bool|null},
  "recommended_model_id":"gpt2-medium","disclosure_ja":"..."}`のような形。
- `POST /v1/recommend-and-download`（2026-07-27追加、「お勧めLLMを
  ダウンロード」ボタンの受け口） — ハードウェア検出→推奨サイズ算出→
  未ダウンロードならHugging Faceから取得(既にあれば再取得しない)→
  取得後は自動的に`/v1/generate`用へホットスワップ切り替え、まで一括で
  行う。`{"recommendation": {...}, "already_installed":bool,
  "switched_to_recommended":bool, "message_ja":"..."}`を返す。
- `GET /` （2026-07-27追加） — 「お勧めLLMをダウンロード」ボタン1つ+
  進捗表示+切り替え後の生成テスト導線を持つ、最小限の静的HTML UI
  (`static/index.html`、フレームワーク不使用)。
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` — テナント登録管理(`x-admin-token`ヘッダ認証)
- `GET /healthz` — ヘルスチェック

### ハードウェア検出→推奨LLMサイズ（2026-07-27新設）

`open-directx`(DXGIアダプタ列挙)・`open-cuda`(`opencuda-vulkan`の
Vulkan物理デバイス列挙)いずれかの実GPU検出結果(VRAM容量)を使い、
GPT-2ファミリーの複数サイズ(124M/355M/774M/1.5B)から推奨サイズを選ぶ
簡易ヒューリスティックを`src/hardware.rs`に実装した。VRAM 2GB未満→124M、
2-4GB→355M、4-8GB→774M、8GB以上→1.5B、GPU検出不能・CPUのみ→124M固定
(安全側フォールバック)。**正直な開示**: これはモデルサイズ(パラメータ数×
4バイトのfp32概算)とVRAM容量の単純な比較に基づく目安であり、精密な
性能予測ではない(KVキャッシュ・アクティベーション等の実消費は含まない)。

GPU検出は既定では無効（`hw-detect-vulkan`/`hw-detect-directx`という
opt-in Cargo feature、CPUのみの環境やクロスコンパイル環境でVulkan
ローダー/Windows SDKへの依存を強制しないため）。有効にした場合、Vulkan
経路を優先し、両方有効ならDXGI(DirectX)側の結果をクロスチェックとして
ログへ記録する(`cross_check_agreement`フィールド)。**実機検証済み**:
このマシン(NVIDIA GeForce GT 730)で`--features hw-detect-vulkan`を
有効にして実行したところ、`vram_bytes=2104819712`を取得——これは
`open-cuda`側`CLAUDE.md`のHANDOFF(DXGI経由の同一実機での実測値)と
完全一致する値であり、Vulkan/DirectX両経路が同一ハードウェアに対し
同じVRAM量を報告することを確認した。

```
cargo run --release --features hw-detect-vulkan
# または Windows専用: --features hw-detect-directx
```

### 意図分類 vs 生成、どちらを使うか

`/v1/chat`(意図分類)と`/v1/generate`(生成)は目的が異なるため、あえて
統合していない: `/v1/chat`は定型応答への振り分け専用で軽量・高速
(埋め込みモデルのforward passのみ)、`/v1/generate`はGPT-2 124M
(548MBの重み)を使う本格的だが重い自由文生成。用途に応じて使い分けること。

## 翻訳プラグイン(`nllb-translate` feature、2026-08-04追加)

`POST /v1/translate`のGPT-2流用実装は実HTTP検証の結果、実際の翻訳文を
生成できず実用に耐えないと判明した(`CLAUDE.md`のHANDOFF参照)。この
問題を解消するオープンソースの専用翻訳モデル(M2M100、`rust-bert`crate
経由)を、**Cargo featureによる着脱式プラグイン**として提供する
(ユーザー指示「翻訳部分だけプラグインという形にして、必要な人だけ
インストール/アンインストールできるように」への対応)。

- **インストール(有効化)**:
  ```bash
  cargo build --release --features nllb-translate
  ```
  この場合のみ`rust-bert`+`tch`(libtorch、PyTorchのC++ライブラリ)への
  依存がビルドに含まれ、`POST /v1/translate`が実際に機能する翻訳文を
  返すようになる。初回リクエスト時にM2M100モデルの重みを自動ダウンロード
  する(数百MB、`rust-bert`既定のキャッシュディレクトリへ保存)。
- **アンインストール(既定状態)**:
  ```bash
  cargo build --release
  ```
  (featureフラグを付けない、既定)。この場合`rust-bert`/`tch`は
  ビルドに一切含まれず、バイナリサイズ・依存グラフへの影響はゼロ。
  `POST /v1/translate`自体は404にはならず、GPT-2流用の簡易実装へ
  自動フォールバックする(`disclosure`フィールドに実用に耐えない旨と
  プラグインの入れ方を明記して返す)。
- **正直な開示**: これは実行時に着脱できる真の意味での「プラグイン」
  (動的ライブラリロード等)ではなく、**ビルド時にCargo featureで
  組み込むか組み込まないかを選ぶ**という意味でのプラグイン方式。
  `rust-bert`は`tch`(libtorch)への依存が必須で、このエコシステムが
  他の全モデル(GPT-2・BERT・Whisper相当)で貫いてきた「手作りRust実装
  +safetensors直接ロード、重量級MLフレームワーク非依存」という方針
  からは意図的に外れる——この妥協は`nllb-translate` feature配下に
  隔離することで、必要な人だけが依存の増加を受け入れる設計にした。
  起動時ログ(`translation plugin: ENABLED`/`not installed`)で現在の
  状態を確認できる。

## 「分身の術」構成

`open-web-server`と同じ設計思想で、1インスタンスを複数ドメインが共有する
(ドメインごとの個別インストール不要)。管理は[open-easy-web](https://github.com/aon-co-jp/open-easy-web)
側から行う想定(統合は未着手)。詳細は[CLAUDE.md](CLAUDE.md)を参照。

## 技術スタック

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)(`open-runo-poem-compat`、
本家[Poem](https://github.com/poem-web/poem)クレートには依存せず、
Poem互換のAPI形状をtokio/hyper直接実装で提供するファサード、
2026-07-31移行) + [open-cuda](https://github.com/aon-co-jp/open-cuda)。
DB非依存・1バイナリ完結。RustはもちろんPython等どの言語からでも
HTTP経由で利用できる(`opencuda-bert`/`opencuda-llm`/`opencuda-whisper`
というPython製AIライブラリ〈Transformers/vLLM/Whisper〉のRust移植を、
このサービスがHTTPサーバーとして公開する窓口)。

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
