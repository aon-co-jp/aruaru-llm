# aruaru-llm

*日本語*: これは原文です ·
*English*: [README-English.md](README-English.md) ·
*Other languages*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **最近の更新(2026-08-26続き)**: `/v1/generate-with-search`の
> プロンプト組み立てを、検索結果を単純連結するだけの旧書式から
> `"Search results: ...\nQuestion: {prompt}\nAnswer:"`というQA形式
> (`web_search::build_search_augmented_prompt`)へ変更した。GPT-2の
> 事前学習コーパスに多いQ&A形式のパターン補完に乗せることで、検索
> 結果を踏まえた応答が出やすくなることを狙った改善の試みであり、
> **確実に検索結果を活用するようになったことを保証するものではない**
> (GPT-2/distilgpt2は指示追従のファインチューニングを受けていないため)。
> instruction-tunedモデルへの切替・要約段階の追加は調査未完了のため
> 見送り、次回セッションの課題とした。**この改善はプロンプト文字列の
> 組み立て方の変更のみであり、open-directx/open-cuda(GPU推論の
> 「速度」基盤)には一切手を加えていない**——応答の「精度」とGPU
> 推論の「速度」は引き続き別の話である。詳細は
> [CLAUDE.md](CLAUDE.md)の2026-08-26追記(続き4)参照。
>
> 📌 **最近の更新(2026-08-26)**: Windows用インストーラー
> `aruaru-llm-installer.exe`を新設(管理者権限のPowerShellを手動で
> 立ち上げる必要なし)。open-cuda/open-directxのVulkan/DirectXバック
> エンドを組み込んだGPU版を任意でインストール可能(既定は未チェック
> ——実測でCPU版より遅いGPUがあることが分かっているため)、インストール
> 中に推奨モデルサイズを検出しYes/No/Cancelで選べるダイアログ付き。
> 詳細は[CLAUDE.md](CLAUDE.md)の2026-08-26エントリ参照。
>
> 📌 **更新(2026-08-25)**: (1) Google検索APIキーをリクエスト単位で
> 指定できるようにした(`POST /v1/generate-with-search`の任意フィールド
> `google_search_api_key`/`google_search_cx`)——複数の訪問者が同じ
> インスタンスを共有するデプロイ(VPS等)で、ある訪問者のキー設定が
> 他の訪問者の検索へ影響しないようにするための修正。(2) ブラウザ内AI
> 実行(WASM+WebGPU)構想の技術検証・段階的導入計画を策定(未実装、計画
> 段階)。いずれも詳細は[CLAUDE.md](CLAUDE.md)の2026-08-25エントリ参照。
>
> 📌 **更新(2026-08-10)**: `open-cuda`側`GptModel::
> generate_with_repetition_penalty`(CTRL方式の繰り返しペナルティ)を
> `/v1/generate`へ配線し、**既定で有効化**した(`ARUARU_LLM_REPETITION_
> PENALTY`環境変数、既定値`1.3`、`1.0`にすると従来のペナルティ無し挙動)。
> 対話ファインチューニング無しの素のGPT-2貪欲デコードが陥る既知の劣化
> モード(同一文字列の無限ループ、例: `open-english`利用中のユーザー報告
> 「しつこく繰り返すバグ」——"Student: Hello"を延々繰り返す)への根本対応。
> `open-cuda`側の実GPT-2 124M重みでのテストで、ペナルティ無しでは実際に
> ループへ陥ること・`penalty=1.3`で実際にループが解消し文法的に自然な
> 会話文へ変わることを確認済み。実HTTPでも
> `POST /v1/generate {"prompt":"...Student: Hello\nTrainer:",
> "max_new_tokens":24}` → `"I'm sorry for the delay in your appointment
> but it's not too late to get back on track! Thank you so"`のように、
> 反復なしの応答が返ることを確認した。`penalty=1.0`時は`generate()`
> (既存API)と完全に同一の出力になるため既存の他テストへの回帰は無い。
> 詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Wired `open-cuda`'s new `GptModel::generate_with_repetition_
> penalty` (CTRL-style repetition penalty) into `/v1/generate`, **enabled
> by default** (`ARUARU_LLM_REPETITION_PENALTY` env var, default `1.3`;
> set to `1.0` to restore the old no-penalty behavior). This directly
> addresses a known GPT-2 base-model degeneracy — endless repetition of
> the same string (e.g. a user reported it looping "Student: Hello"
> forever while using `open-english`) — since the base model has no
> dialogue fine-tuning. Verified on real GPT-2 124M weights on the
> `open-cuda` side: without the penalty the loop actually reproduces;
> with `penalty=1.3` it stops and produces grammatically natural
> conversational text instead. Also verified via a live HTTP request. At
> `penalty=1.0` the output is byte-identical to the existing `generate()`
> API, so no other tests regress. See [CLAUDE.md](CLAUDE.md) for details.

> 📌 **最近の更新(2026-08-08)**: `open-cuda`側で実装・実機検証済みだった
> DeepSeek-V3風MLA(KVキャッシュ低ランク圧縮)を`/v1/generate`へ
> オプトイン配線(既定off)した。`ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`で
> 有効化、GPT-2 124Mでhead_dim=64→d_c=16(75%削減)。**正直な開示**:
> 射影行列が乱数初期化(未学習)のため非可逆圧縮であり、実際に生成品質を
> 目視で明確に劣化させることを実測で確認した(既定offとした理由)。
> 追加で、`open-cuda`側にPCA較正版(`enable_mla_kv_compression_calibrated`、
> 実サンプル文の活性化統計から主成分分析で射影基底を求める)も新設された
> ため、こちらも`ARUARU_LLM_MLA_CALIBRATED=1`(`ARUARU_LLM_ENABLE_MLA_KV_
> COMPRESSION=1`と併用)でオプトイン配線した。乱数射影版の反復破綻
> (例: "...and point of the government"の無限ループ)は回避するが、
> **非圧縮版と比較すればなお明確に品質が劣化しており**、こちらも既定offの
> ままとした。詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Wired an opt-in (default-off) MLA-style KV cache compression
> path (`open-cuda`'s DeepSeek-V3-inspired implementation) into
> `/v1/generate` via `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`. For GPT-2
> 124M this is head_dim=64 -> d_c=16 (75% smaller per-token KV storage).
> **Honest disclosure**: the projection matrices are randomly initialized
> (untrained), so this is lossy, and measured testing confirmed it
> visibly degrades generation quality — which is why it defaults off.
> `open-cuda` has since added a PCA-calibrated variant
> (`enable_mla_kv_compression_calibrated`, basis derived from real
> activation statistics via PCA instead of random init), wired here via
> `ARUARU_LLM_MLA_CALIBRATED=1` (used together with `ARUARU_LLM_ENABLE_MLA_
> KV_COMPRESSION=1`). It avoids the random variant's degenerate repetition
> loops, but quality is still clearly worse than the uncompressed path, so
> it also defaults off. See [CLAUDE.md](CLAUDE.md) for details.

> 📌 保留タスク(2026-08-06): 東芝SBM・DeepSeek技術の組み込み構想あり。詳細は[CLAUDE.md](CLAUDE.md)参照。

> 📌 **最近の更新(2026-08-07)**: `/v1/chat`・`/v1/classify-security`が
> 以前`/v1/generate`・`/v1/translate`で見つかった「空入力→503誤判定」
> バグの影響を受けていないか実バイナリ・実HTTPで検証。両エンドポイント
> とも空入力で200(フォールバック応答/benign判定)を正しく返すことを
> 確認、コード変更は不要と判明した。詳細は[CLAUDE.md](CLAUDE.md)参照。
>
> *English*: Verified via a real running binary + live HTTP requests that
> `/v1/chat` and `/v1/classify-security` do **not** suffer the
> "empty input → 503" bug previously fixed for `/v1/generate` and
> `/v1/translate` — both correctly return 200 for empty input. No code
> change was needed.

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
  (GPT-2 124M実重みによる自己回帰生成、本格的だが重い。**繰り返しペナルティ
  既定値`1.3`**〈`ARUARU_LLM_REPETITION_PENALTY`で上書き可、`1.0`で無効化〉
  で同一文字列の無限ループを防止。プロンプトは英語推奨——
  GPT-2のBPE語彙は英語中心のため。実験例:
  `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out of your"`)。
  **2026-08-07修正**: `prompt`が空文字列(または空白のみ)の場合は
  `400 Bad Request`(`{"error": "prompt must not be empty", "engine": "..."}`)
  を即座に返すようにした——以前はトークナイザが0トークンにエンコード
  した後の内部エラーが`503 Service Unavailable`としてそのまま返っており、
  呼び出し側から見て「サーバー障害」なのか「自分の入力が不正」なのか
  区別できず不便だったための修正(実HTTPで`400`が返ることを検証済み)。
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
  **2026-08-07修正**: `text`または`target_lang`が空(空白のみ含む)の
  場合は`400 Bad Request`を即座に返す(`/v1/generate`と同じ理由、
  実HTTPで検証済み)。
- `POST /v1/transcribe`(2026-08-29追加) — `{"pcm_f32_base64": "...",
  "sample_rate": 16000, "language": "auto"(任意), "prompt": "直前の文脈"(任意), "tenant": "..."(任意)}` →
  `{"transcript": "...", "language": "...", "engine": "...", "disclosure": "..."}`。
  `pcm_f32_base64`は16kHz mono の f32 PCM をリトルエンディアンバイト列に
  して base64 化したもの(`open-english`の`blobToPcm16k()`が生成する形式)。
  **whisper.cpp のプレビルド CLI(`whisper-cli`)と GGML Whisper モデルが
  実在すれば**それを子プロセス起動して書き起こし、**無ければ**`503` +
  入手先(whisper.cpp releases)を案内するエラーを返す。詳細は下記
  「音声認識」節参照。`sample_rate`が16000以外・base64不正・10分超の音声は
  いずれも`400`。
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

### 実推論ディスパッチ先としてのVulkan(`real-vulkan` feature、2026-08-04新設、既知の未解決課題あり)

上記のハードウェア検出(`hw-detect-vulkan`)とは別軸で、`/v1/generate`の
実際の推論計算そのものをGPU(Vulkan)へディスパッチするオプトイン
feature`real-vulkan`を追加した。既定では無効(CPUのみ)、有効時は
`main()`のデバイス選択が`opencuda_vulkan::real::VulkanDevice`に切り替わる
(構築失敗時はCPUへ自動フォールバックし、サービスを壊さない設計)。

```
cargo run --release --features real-vulkan
```

**正直な開示(重要、未解決の既知バグ)**: 実機(NVIDIA GeForce GT 730)で
検証したところ、デバイス選択自体は正しく機能する(起動ログに
`OpenCUDA Vulkan Device (NVIDIA GeForce GT 730)`が出る)が、
`POST /v1/generate`への実リクエストは**約0.2秒で即座にエラー失敗する**。
原因は`open-cuda`側`open-cuda-llm`クレートの`Linear::forward`が
`opencuda_blas::sgemm`へ`spirv`引数を常に`None`で渡しており、
`GemmPath::VulkanGeneric`選択時に必須のコンパイル済みシェーダバイト列
(`matmul.spv`)が渡っていないため。これは「配線しても遅い」という
以前の設計上の懸念より根本的な、単純に**機能しない**という結果であり、
性能比較のベンチマークはまだ実施できていない。修正には`open-cuda`
リポジトリ側で`Linear::forward`に`matmul.spv`を配線する変更が必要
(詳細・次にすべきことは`CLAUDE.md`のHANDOFF 2026-08-04エントリ参照)。

### DeepSeek「Engram」風KVキャッシュ/重みオフロードの検討結果(2026-08-08、見送り)

DeepSeekの技術のうち、静的な知識(KVキャッシュや重みの一部)をVRAMから
システムRAMへ退避し必要時に再ロードする「Engram」的な手法を、GT 730の
ような小VRAM GPU向けに実装できないか検討した。**実コード読解の結果、
実装を見送った**——理由は「難しいから」ではなく「このリポジトリが依存
する`open-cuda`側の推論経路には、そもそも退避すべき"VRAM常駐状態"が
存在しない」ため。`opencuda-blas`のGEMM/Attention/softmax(`sgemm`
経由でVulkanにディスパッチする箇所すべて)は、呼び出しのたびに
`ScopedAlloc`(`opencuda-blas/src/lib.rs`)というRAIIガードで
VRAMバッファをalloc→host→device転送→計算→device→host転送→即freeする
設計になっており、呼び出しが終わるとVRAM上には何も残らない。GPT-2の
重み(`GptModel`の`word_embeddings`/各層の`Linear`)もKVキャッシュ
(`open-cuda-llm::KvCacheHead`の`k`/`v`/`k_latent`/`v_latent`)も、
実体は最初から最後まで`Vec<f32>`としてシステムRAM上に存在し続けている
(GPU実行時〈`--features real-vulkan`〉でも同じ)。つまりこのアーキテク
チャは、意図した設計ではないものの結果として既に「常時システムRAM
常駐、GPUは演算のたびの一時利用のみ」という、Engramが目指す状態に
近い形になっている。LRUエビクション等の追加機構を実装しても退避対象が
無いため、意味のある効果は測定しようがない(誇張しない開示)。詳細・
読んだコード箇所は`CLAUDE.md`のHANDOFF 2026-08-08エントリ参照。

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

## 音声認識(`POST /v1/transcribe`、whisper.cpp CLI、2026-08-29追加)

`open-english`の音声認識は、ブラウザの Web Speech API 頼みで精度が
低かった。ブラウザ内 Whisper(transformers.js、P2-α)を追加したが、
端末の GPU/NPU/CPU 性能に大きく左右され、large 級モデルはブラウザには
重すぎる。P2-β として、**利用者が自分の PC で起動している aruaru-llm に
whisper.cpp でより大きな Whisper モデルの書き起こしを任せる**経路を
`POST /v1/transcribe` として用意する(正本:
`open-english/docs/SPEECH_RECOGNITION_REDESIGN.md` P2-β)。

**実装方式(2026-08-29 方針変更)**: 当初は `whisper-rs`(whisper.cpp の
Rust バインディング)を直接リンクする設計だったが、`whisper-rs-sys` は
**Windows(MSVC)で bindgen が破綻する既知ブロッカー**があり
(`whisper-rs 0.16.0` でも `WHISPER_DONT_GENERATE_BINDINGS=1` でも解消
しない、issue 2026-04-21)、open-english の主対象が Windows なので不成立。
代わりに **whisper.cpp の公式リリース同梱プレビルド CLI(`whisper-cli` /
旧 `main`)を子プロセス起動**する(`pg_dump` / `Expand-Archive` / `adb` を
子プロセスで呼ぶのと同じパターン。C++ リンク・bindgen を完全回避、GPU
バックエンドはプレビルド CLI 側で選ばれたものがそのまま使われる)。
**Cargo feature は不要**——コンパイル時依存が無いため、既定ビルドに
そのまま含まれる。

- **有効化**: 2 つのファイルを置くだけ(ビルド不要)。
  1. whisper.cpp の[プレビルド `whisper-cli`](https://github.com/ggml-org/whisper.cpp/releases)を
     `<crate>/models/whisper/whisper-cli`(Windows は `whisper-cli.exe`)へ、
     または `ARUARU_LLM_WHISPER_CLI` でパス指定。
  2. GGML モデル(`ggml-base.bin` は軽量・そこそこ、`ggml-large-v3-turbo` が
     最高品質。いずれも非同梱)を `<crate>/models/whisper/ggml-base.bin` へ、
     または `ARUARU_LLM_WHISPER_MODEL` でパス指定。
- **未配置時**: `POST /v1/transcribe` は `503` + 入手先を案内するエラー。
- **状態確認**: `GET /v1/runtime` の `whisper` フィールド
  (`available` / `backend` / `cli_path` / `cli_present` / `model_path` /
  `model_present` / `detail`)。
- **調整**: `ARUARU_LLM_WHISPER_TIMEOUT_SECS`(既定 300)で子プロセスの
  壁時計上限。
- **正直な開示・検証状況**: `cargo build --release` 成功、
  `cargo test --release` **100 件全 green**(新規 `transcribe` テスト 7 件
  = WAV RIFF ヘッダ生成・whisper-cli JSON パース・CLI/モデル不在時の
  エラー・env 上書き)。**実 `whisper-cli` + 実 GGML モデルでの書き起こし
  E2E は、この開発環境に両方が無いため未検証**——次周、プレビルド CLI と
  `ggml-base.bin` を用意して `POST /v1/transcribe` を実 HTTP で検証する。

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
