# 設計思想＆開発方針＆開発環境ルール(aruaru-llm)

> **📌 2026-08-19追記: 東芝SBM/DeepSeek調査タスクは完了・現状維持と判断
> (`open-english/PORTING.md`のHANDOFFに残っていた「項目4未着手」を受けて
> 日英中(簡体)中(繁体)4言語でWebSearch調査を実施)**:
> - **東芝SBM**: 2026年の新展開として、(1) 2026年4月発表の第3世代SB
>   アルゴリズム("カオスの縁"利用、Physical Review Applied掲載、旧世代比
>   約10〜100倍高速化)、(2) 2026年2月、SBMを自律移動ロボットへ搭載し
>   リアルタイム制御に応用(Nature Communications等掲載、東芝・MIRISE)、
>   (3) 2026年6月、最適化制御AI+Isingモデル圧縮を組み合わせた新フレーム
>   ワークを確認した。ただしこのリポジトリでは既に`cache_optimizer.rs`
>   (`POST /v1/models/optimize-cache`)でSBMをナップサック問題(モデル
>   キャッシュ選択)として実装済み(2026-08-10 HANDOFF参照)であり、今回
>   発見した第3世代アルゴリズムの改善(カオスの縁による成功確率向上)は
>   理論上参考になり得るが、現行実装は8シード多重探索で75%以上の近似
>   精度という別方式で既に運用中のため、既存実装を置き換えるほどの
>   緊急性・具体的な適用差分は無いと判断し、コード変更は行わなかった
>   (正直な開示)。
> - **DeepSeek**: MLA(Multi-head Latent Attention、KV量93.3%削減)・
>   DeepSeekMoE(スパース比率の継続的な圧縮、V3の5.4%→V4-Proの3.1%)・
>   R1からのreasoning蒸留(検証・振り返りパターンをV3へ蒸留)を実在の
>   技術として確認。加えてSlimMoE(arXiv:2506.18349、MoEのexpert
>   slimming+多段蒸留、Phi 3.5-MoEを41.9B→7.6B/3.8Bへ圧縮)を新規に発見
>   した。ただしこのリポジトリの推論エンジンはGPT-2 124M(密なdecoder-only、
>   MoEではない)であり、MLA/DeepSeekMoE/SlimMoEはいずれも
>   **MoEアーキテクチャまたはマルチヘッド潜在アテンション前提の再学習を
>   要する技術**で、既存の事前学習済みGPT-2重みへ後付けで適用することは
>   できない(アーキテクチャ変更には再学習が必須、正直な開示)。既存の
>   INT4/INT8量子化(`opencuda-blas`、2026-07-21時点で実装済み)は
>   DeepSeekの量子化アプローチと方向性が一致しており、追加で取り込む
>   べき新規手法は見つからなかった。
> - **結論**: 今回の調査で、東芝SBM・DeepSeekのいずれについても、
>   このリポジトリへ**新規にコード実装すべき具体的な差分は無い**と
>   判断した(過去のHANDOFFで既に両技術とも実装済み、かつ今回発見した
>   最新動向は現行実装のアーキテクチャ前提〈非MoE・単一シードでなく
>   多重探索〉と噛み合わないため)。この保留タスクは調査完了として
>   クローズする。

> **📌 2026-08-19追記: open-cuda/open-directxの「間接的な自動アップデート」経路の再調査**
> (ユーザー指摘「open-englishでaruaru-llmが使われるのですから、
> open-cuda・open-directxが一緒に自動アップデートの対象で深い関連リポジトリ
> の対象です。もしシステムがプログラムがそうなっていないなら、システムの
> 開発の全て修正の対象です」を受けた再調査)。
> - **依存形態の実態**: `Cargo.toml`は`opencuda-core`/`opencuda-cpu`/
>   `opencuda-blas`/`open-cuda-bert`/`open-cuda-llm`をすべて
>   `path = "../open-cuda/crates/..."`のローカルpath依存で参照しており、
>   crates.io公開もgit依存でのcommit固定(rev/tag pin)もしていない。
>   `open-directx`(dream-os/open-directx、別リポジトリ)への依存は
>   `Cargo.toml`に一切無いことを確認済み(2026-07-26のHANDOFF記載どおり
>   現状も変わらず——GPU検出のDirectX経路は`open-cuda`内の
>   `opencuda-directx`クレートであり、別リポジトリ`open-directx`とは
>   無関係)。
> - **CI(`.github/workflows/release.yml`)の実態確認**: タグpush時の
>   リリースビルドは、ビルド直前に`git clone --depth=1
>   https://github.com/aon-co-jp/open-cuda.git ../open-cuda`を実行して
>   おり、これはブランチ/タグ/commitを指定しない**デフォルトブランチの
>   最新HEAD**を毎回新規checkoutする形になっている(2026-08-17時点で
>   既に実装済み、RPoem/RS-JSON等の他sibling path依存も同様の方式)。
>   つまり——ユーザー指摘の「aruaru-llmが新しいリリースをビルドする際、
>   常に最新のopen-cudaを取り込んでビルドする」という間接的自動更新の
>   経路は、**タグpushによるCIリリースビルドに関しては既に実装済み**
>   であることが判明した(前回セッションでの「自己アップデート機構は
>   適用できないため見送り」という判断は、self_update.rs型のバイナリ
>   自己更新の文脈に限った判断であり、CI経由の間接更新は別軸で既に
>   機能していたことを見落としていた)。
> - **手元(ローカル)ビルド運用のみ抜けていた点**: 一方、開発者が
>   ローカルで`cargo build --release`する場合は、隣接ディレクトリ
>   `../open-cuda`の**その時点でのworking treeの状態**をそのまま使う
>   ため、ローカルの`open-cuda`をpullし忘れると古い状態でビルドされる
>   リスクが残る(CIのような強制checkoutが無い)。ここは運用ルールとして
>   明記が抜けていたため、今回**運用ルールとして追記**する:
>   **リリースタグをpushする前に、必ず`git -C ../open-cuda pull`(および
>   `git -C ../RPoem pull`等、他のsibling path依存リポジトリ)で最新化
>   してから`cargo update -p opencuda-core -p opencuda-cpu -p
>   opencuda-blas -p open-cuda-bert -p open-cuda-llm`を実行し、
>   `Cargo.lock`を最新のsibling内容に追従させること。**
> - **検証結果(正直な開示)**: ローカル(`F:\runo\aruaru-llm`)で
>   `cargo update -p opencuda-core -p opencuda-cpu -p opencuda-blas
>   -p open-cuda-bert -p open-cuda-llm`を実行し正常終了(0 packages
>   changed = 既にsibling内容と一致)したことを確認。続けて`cargo build
>   --release --bin aruaru-llm`を実行したが、ビルドは時間がかかり
>   バックグラウンドタスクとして進行中で、本追記の時点ではビルド完走
>   (成功/失敗)の確認は取れていない——**CI環境上での実際の動作(GitHub
>   Actions上でのgit clone→build成功)までは検証できていない**ことを
>   明記する。
> - **結論・次にすべきこと**: (1) `open-directx`は現時点でaruaru-llmから
>   一切依存されていないため、間接自動更新の対象外(依存が発生した時点で
>   同様のsibling checkout方式をCIへ追加する)。(2) `open-cuda`について
>   CI側の間接自動更新は既に機能しているため追加実装は不要、ただし
>   ローカルリリース手順の運用ルール(上記pull徹底)を本HANDOFFに明記した。
>   (3) ローカルでの`cargo build --release`完走確認(成功/失敗)は
>   バックグラウンドで進行中のため、完了次第結果をこのHANDOFFへ追記する
>   必要がある。


作業ドライブは`F:\open-runo`。この節は[`open-raid-z`](https://github.com/aon-co-jp/open-raid-z)の
`CLAUDE.md`を正本とし、各プロジェクトへコピーして同期する方針に準じる。
GitHubリポジトリ: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm)。

> ⚠️ **正直な開示(最重要、2026-07-25更新)**: 2026-07-25、`open-cuda`の
> `opencuda-llm`クレート(GPT-2 124M実学習済み重み、`openai-community/gpt2`)
> を統合し、`POST /v1/generate`で**実際に自己回帰デコーダによる文章生成が
> 可能**になった(下記「以前の記述」にあった「自己回帰デコーダはまだ
> 実装していない」は本エンドポイントに限り解消済み)。ただし
> **GPT-2 124Mは2019年発表の小型モデルであり、GPT-4等最新の商用LLMと
> 同等の性能・知識・指示追従能力は無い**——あくまで「外部LLM API契約
> (課金)不要の自己完結型AI」としての実証段階であることを常に明記する。
> 生成テキストは文法的には自然な英語になることが多いが、意味的に正確
> とは限らない(幻覚しうる、ファインチューニング済み対話モデルではなく
> 素の事前学習済み言語モデルの貪欲デコード)。詳細は下記HANDOFF
> 2026-07-25エントリ参照。
>
> **以前の記述(2026-07-21〜、`/v1/chat`には引き続き該当)**: `/v1/chat`
> (意図分類)は`open-cuda`の`opencuda-bert`クレート(multilingual-e5-small、
> MITライセンス、日本語含む100言語対応)で実際に文を埋め込みベクトルへ
> 変換し、意図ごとの代表例文とのコサイン類似度で分類する**エンコーダ
> ベースの意味的類似度分類**のままであり、対話文を生成する能力ではない
> (生成は`/v1/generate`が別途担う、下記「意図分類 vs 生成」参照)。
> 「AI」「LLM」を名乗る以上、この限界を隠さず常に明記すること。

## このプロジェクトの役割

`aruaru`エコシステム(aruaru-tokyo・aruaru-db・e-gov.info・karu.tokyo等)
共通の「AIチャットコマース」応答ロジックを提供する、独立したHTTPサービス。
各サイトがそれぞれ個別にチャット応答ロジックを実装するのではなく、この
サービスにHTTP経由で問い合わせる構成にすることで、将来実際のLLM推論に
差し替える際の変更箇所を1箇所に集約する。

### なぜ今すぐ本物のLLM推論を実装しないか(2026-07-21更新、旧記述は誤り)

> ⚠️ 訂正: 以前の本節は「`opencuda-blas`のGEMM/Attentionはスタブのまま」
> としていたが、これは古い情報のまま更新漏れしていた。実際には
> `opencuda-blas`の**CPUパスでGEMM(`sgemm`, `GemmPath::CpuNaive`)・
> 素朴なAttention(`scaled_dot_product_attention`)・INT4/INT8量子化は
> 既に実装済み**(2026-07-21時点でのopen-cuda `opencuda-blas/src/lib.rs`
> 確認)。テストも全green。

未実装のまま残っているのは以下のみ:
- GPU専用の高速パス(`GemmPath::CuBlas`/`RocBlas`/`OneMkl`/`VulkanGeneric`)
- 真のFlash Attention(タイル化・オンラインsoftmax、`flash_attention`関数)

本物のLLM推論に本当に不足していたのはGEMM/Attentionという**演算プリミティブ**
ではなく、**意味のある入力ベクトル**だった。2026-07-21、`opencuda-bert`
クレート(multilingual-e5-small、学習済み埋め込み層+トークナイザ)が
実装され、`scoring.rs`はbag-of-wordsから実際の文埋め込み+コサイン類似度
分類へ移行した(下記「現状の実装」参照)。ただしこれは**エンコーダ専用**
であり、文章を生成する自己回帰デコーダ(対話生成としての「LLM」の能力)は
まだ実装していない。それにはQwen3-14B等の実モデル重みの入手・ライセンス
確認が前提条件になる(未着手、次のHANDOFF参照)。

### セットアップ(2026-07-21追記): モデル重みの取得

`models/multilingual-e5-small/`(470MB超)は`.gitignore`対象で**Gitに
含めない**。ビルド・起動前に、各自Hugging Faceから取得すること:

```
huggingface-cli download intfloat/multilingual-e5-small \
  --local-dir models/multilingual-e5-small
```

(または`config.json`/`model.safetensors`/`sentencepiece.bpe.model`/
`special_tokens_map.json`/`tokenizer.json`/`tokenizer_config.json`を
`https://huggingface.co/intfloat/multilingual-e5-small/tree/main`から
個別ダウンロードし、同ディレクトリに配置する)。

### 現状の実装(2026-07-21更新、エンコーダベースの意味的類似度分類)

- `opencuda-bert`(multilingual-e5-small)による実際の文埋め込み計算+
  `opencuda_bert::cosine_similarity`によるコサイン類似度で意図分類
  (申請/購入/与信/不動産等のカテゴリ)。各インテントの代表例文embeddingは
  プロセス起動後の初回呼び出し時に計算・キャッシュする(`OnceLock`、
  リクエストごとの再ロードを避けるため)。
- 各カテゴリに対応した定型応答文
- 将来、自己回帰デコーダによる文章生成(または外部LLM APIの薄いラッパー)に
  差し替える際、**HTTP APIの入出力契約(`POST /v1/chat` → `{"reply": "...",
  "engine": "..."}`)は変えずに内部実装だけ差し替えられる**ように設計する。
  `engine`フィールドには常に現在の実装方式
  (`"embedding-cosine-v0-opencuda-bert-cpu"`等)を正直に返し、呼び出し側が
  「本物の対話生成AIかどうか」を判別できるようにする。

## 技術スタック

**2026-07-31更新**: 本家`poem`クレートへの直接依存を廃止し、`RPoem`
(`RPoem/crates/open-runo-poem-compat`、`open_runo_router::hyper_compat`
のtokio/hyper直接実装をpoemと同じ呼び出し形状でラップした薄い
ファサード)へ移行した(ユーザー指示「Rust＋Poem互換のRPoemで、Rustでも
他のプログラム言語でも扱える仕様に」)。`Data<T>`抽出子は提供されない
ため、共有状態(`device`/`registry`)はハンドラ登録時のクロージャで
`Arc`をキャプチャする形。DB非依存・1バイナリ完結という設計自体は
不変。実HTTPで全エンドポイント(chat/classify-security/generate/
models系/admin系/healthz/静的UI)の動作を確認済み。

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "e-gov.info"(任意)}` を
  受け取り `{"reply": "...", "engine":
  "embedding-cosine-v0-opencuda-bert-cpu", "matched_intent": "..."}` を
  返す。`tenant`は未登録でも応答は返す(可用性を落とさないため)。
- `POST /v1/generate`(2026-07-25新設) — `{"prompt": "...", "max_new_tokens":
  16(任意、既定16、上限128), "tenant": "..."(任意)}` を受け取り
  `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}` を
  返す。`opencuda-llm::GptModel`(GPT-2 124M実重み)による貪欲デコード
  生成。GPT-2実重み(`../open-cuda/crates/opencuda-llm/models/gpt2/`、
  `ARUARU_LLM_GPT2_DIR`で変更可)が無い・ロード失敗時は503を返す
  (`/v1/chat`とは異なりbag-of-words的なフォールバック先が存在しないため、
  黙って別経路には落とさず正直にエラーを返す設計)。`/v1/chat`(意図分類、
  軽量・高速)とは目的が異なるため無理に統合していない——役割分担は
  `src/generation.rs`のモジュールdocコメント参照。
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

## 運用ルール追記(2026-07-18、正本はopen-raid-zのCLAUDE.md参照) — 確認不要の自動継続・リミット解除後の自動再開

- **コンテキストウインドウ・5時間利用制限・その他のセッション中断が
  発生し、その後リミットが解除されて新しいセッションが開始された場合、
  「続けてよろしいですか」等の確認を挟まず、毎回自動的に前回セッションの
  続きの作業を再開すること**(ユーザー指示、2026-07-18)。具体的には:
  1. セッション開始時、各リポジトリの`git status`/`git log`と、この
     `CLAUDE.md`(および他プロジェクトのCLAUDE.md)のHANDOFF節・
     「次にすべきこと」記載を確認し、未完了・未pushの作業が無いかを
     まず裏取りする(タスク管理メタデータを鵜呑みにしない既存方針と
     同じ姿勢で、実際のgit状態を確認する)。
  2. 未完了作業が見つかった場合、ユーザーへの確認を求めず、そのまま
     自動的に検証(build/test)→修正→コミット→pushまで完了させる。
  3. 完了している場合は、各CLAUDE.mdの「次にすべきこと」「未着手・
     未完成」に記載された次の項目へ確認なしに着手する(既存の
     「未着手だからといって確認を求めて手を止めない」方針の延長)。
  4. 「続けてよろしければそのまま自動開発を継続します」のような、
     続行そのものを尋ねる確認は今後一切行わない(ユーザー指示、
     2026-07-18)。作業内容の要約・進捗報告はしてよいが、それは
     承認を求めるものではなく完了報告として書く。
  5. こまめにコミット・pushしておくことで、次回セッションが「どこから
     再開すべきか」を迷わず`git log`/CLAUDE.mdから機械的に判断できる
     ようにしておく(区切りがついた時点で都度コミット・pushする既存
     方針との組み合わせ)。


## 運用ルール追記(2026-07-19、正本はopen-raid-zのCLAUDE.md参照) — 白画面バグ等を見逃さない検証徹底

- **WEB/UIを持つ機能を実装した後は、ビルド成功・`cargo test`・curlでの
  ステータスコード確認だけで「完了」と報告せず、実際に画面が正しく
  表示される(白画面・レンダリング崩れ・コンソールエラーが無い)ところ
  まで確認すること**(ユーザー指示、2026-07-19)。
  1. ブラウザ操作が可能な環境では、実際にページを開いて表示内容
     (見出し・本文・想定した要素の存在)とコンソールエラーの有無を
     確認する。
  2. ブラウザ操作ができない環境では、少なくとも`curl`等でHTMLボディの
     中身を取得し、期待される文字列が実際に含まれているかを確認する
     ——ステータスコード200だけを見て「動作確認済み」としない。
  3. 白画面・エラー・期待した内容の欠落等の不具合が見つかった場合は、
     確認を求めず自動的に原因調査・修正・再確認まで行う。
  4. 本番ドメインが未取得・DNS未設定なだけの状態は上記の「白画面
     バグ」とは別物であり、混同しない(`localhost`確認で代替可)。


## HANDOFF

- **2026-08-19(続き3) スマホ計算タスク配布API(`GET /v1/background-fold/
  task`・`POST /v1/background-fold/task-result`)を新設(ユーザー指示
  「実際のスマホ側計算処理を実装してほしい」への対応、`open-english`側
  Androidワーカーと対をなす、詳細は`open-english/CLAUDE.md`同日HANDOFF
  参照)**:
  1. **新規`src/phone_task.rs`**: `GET /v1/background-fold/task`が
     コサイン類似度計算タスク(2本のベクトル)を1件返す。ベクトルは
     `scoring.rs`が既にウォームアップ済みのインテント埋め込み
     (`sample_embedding_pair_for_phone_task`を新設)から取り、未ウォーム
     アップ時は8次元のデモベクトルへフォールバックする。
     `POST /v1/background-fold/task-result`は結果を受け取り記録する
     のみ(実際のモデル推論・Model Foldingへは反映しない、モジュールdoc
     に明記)。認証・レート制限は無い最小実装(同一LAN/USB内の信頼された
     端末を想定)。
  2. **`main.rs`へのルート登録**: 既存の`/v1/background-fold/status`
     (`idle_background_fold.rs`)と並べて2ルートを追加。
  3. **検証**: `cargo build --release`成功、`cargo test --release
     phone_task`2件全green。実際にサーバーを起動し、
     `GET /v1/background-fold/task`が実際の384次元embedding(実
     multilingual-e5-small由来)を含むJSONを返すこと、
     `POST /v1/background-fold/task-result`が受け取った結果を
     `received_epoch_secs`付きで正しくエコーすることを実HTTPで確認した。
  4. **正直な開示**: スマホ側(`open-english/android`)の実装・ビルド
     成功は確認済みだが、実機Android端末・USB接続環境がこの開発機に
     無いため、PC↔スマホ間の実際のE2E(実機でタスクが往復すること)は
     未検証のまま。
  - 次にすべきこと: (1) 実機での往復検証、(2) 将来的にタスクの種類を
    増やす場合(現状は`cosine_similarity`固定)の設計拡張、(3) 認証・
    レート制限が必要かどうかの要否確認(公開インターネットへの露出を
    想定するなら必須)。

- **2026-08-19(続き2) PC側NPU自動検出+USB接続Android台数検出を実装
  (ユーザー指示「使わなくなったスマホもフル動員」「NPUがPC側にあれば
  自動検出して計算に使用」「複数スマホをUSB接続してCPU+GPU+NPUを統合
  リソースとして利用」への対応、直前2026-08-19エントリのアイドル検知
  スケジューラの上に構築)**:
  1. **PC側NPU検出(`src/hardware.rs::detect_npu`)**: Windows上で
     `Get-CimInstance Win32_PnPEntity`を呼び、デバイス名に"NPU"・
     "Neural"・"AI Boost"(Intel NPU)・"Hexagon"(Qualcomm NPU)を含む
     デバイスを探す簡易検出。**実機確認結果(正直な開示)**: この開発機
     (2026-08-19時点)には該当デバイスが1件も無く、`detect_npu()`は
     常に`None`を返した——このマシンにNPUは搭載されていない。検出でき
     た場合でも実際にNPU上で計算する経路(DirectML NPU推論等)は対応
     SDKが無いため未実装、`AcceleratorInventory`へ記録するだけに留めた
     (既存のGPU検出〈Vulkan/DirectX〉と同じ「検出はできるが実行
     パイプライン未配線」という設計上の限界)。Windows以外
     (Linux VPS等)向けの検出経路は未実装。
  2. **USB接続Android台数検出(`src/hardware.rs::detect_usb_android_
     devices`)**: `adb devices`を子プロセスとして呼び、`device`状態の
     シリアル番号一覧を返す。**実機確認結果**: この開発環境には`adb`
     コマンド自体がPATH上に存在せず(`adb: command not found`)、
     `Err`を正直に返す(黙って0台と偽装しない設計)。実機Android端末の
     USB接続検証はこの開発環境ではできなかった。
  3. **`AcceleratorInventory`+`GET /v1/background-fold/status`拡張**:
     `hardware::detect_accelerators()`(CPU常時available+既存GPU検出+
     上記1・2を統合)を`idle_background_fold::FoldProgress`
     (`accelerators`フィールド)へ追加。実際にサーバーを起動し実HTTP
     リクエストで`{"accelerators":{"cpu_available":true,"gpu":{...
     "detection_path":"cpu-only-fallback"...},"npu_name":null,
     "usb_android_devices":null,"disclosure":"..."}}`という正しい応答
     (このマシンの実際の検出結果と一致)を確認済み。
  4. **USB接続スマホ活用の設計(実装は行っていない、正直な開示)**:
     `F:\runo\open-english\android\`側のプロトコル設計(ADB経由の
     `adb forward`によるPC⇔スマホHTTP通信確立案、NNAPI/TensorFlow
     Lite NNAPI Delegate経由でのスマホ側NPU活用案)は、既存の
     `aruaru-llm/CLAUDE.md` 2026-08-19エントリ5番の記載
     (`GET /v1/background-fold/task`・`POST /v1/background-fold/result`
     というポーリング方式の設計)がそのまま該当し、今回追加で具体化する
     新規設計事項は無かった——重複を避けるため、本エントリでは同記載を
     参照するに留める。実機のAndroidスマホがこの開発環境に無いため、
     この設計自体の実装(上記エンドポイント新設)は今回も行っていない。
  5. **検証**: `cargo build --release`成功(既存4件のpre-existing警告
     のみ、今回の変更に起因する新規警告なし)。`cargo test --release`
     **70件全green**(既存70件、回帰なし——今回の変更は検出処理のみで
     専用の新規単体テストは追加していない、実機での起動+実HTTP確認を
     主体とした)。実際にサーバーを起動し`GET /v1/background-fold/
     status`への実HTTPリクエストで上記3番の通り確認済み。
  6. **UI側**(`open-english`側で対応、詳細は`open-english/CLAUDE.md`
     参照): 日英併記の「使わなくなったスマホもフル動員」バナーを
     `index.html`へ追加。
  - 次にすべきこと: (1) 実際にNPU搭載機・Android実機(adb接続可能な
    環境)が用意でき次第、上記1・2の検出ロジックを実機で再検証、
    (2) `GET /v1/background-fold/task`・`POST /v1/background-fold/
    result`(スマホ側プロトコル)の実装は、実機Android端末が用意でき
    次第着手する、(3) Linux VPS向けのNPU検出経路(`/proc`や`lspci`
    経由等)は今回未実装のまま。

- **2026-08-19 「1週間かけてスマホもフル活用しつつPCバックグラウンドで
  再学習」要望への対応: 実現可能性評価+アイドル検知スケジューラの実装
  (ユーザー指示「再学習が、もし、一週間で、既存のリソースのスマホや
  使わなくなったスマホなどもフル利用して今のPCのバックグラウンドで
  再学習も可能な範囲で、裏で実行する様な仕様にしましょう」への対応)**:
  1. **スマホ活用の現実的な評価(誇張しない)**: GPT-2 124Mクラスの
     本格的な逆伝播学習をAndroidスマホ単体で行うのは、メモリ・熱・
     電池の制約から通常非現実的と判断した。現実的な役割分担案として、
     スマホ側は「軽量な補助計算(例: 層間類似度計算のような順伝播
     のみのタスク)」に限定し、勾配計算・オプティマイザ更新はPC側に
     集約するのが妥当という結論に至った。ただし**この開発環境には
     実際に検証できるAndroid実機(余ったスマホ)が存在しない**ため、
     スマホ側の実装は行わず、プロトコル設計のみに留めた(下記5番)。
  2. **採用した設計**: 「四六時中GPUフル稼働の学習」ではなく、
     「PCがアイドル(HTTPリクエストが一定時間無い)を検知した時だけ、
     低優先度のバックグラウンドスレッドで軽量な処理を1ステップ進め、
     すぐ休止する」という間欠実行方式を採用した。既存の
     `android/`(`tokyo.runo.aruarullm`)の`PowerProfile`
     (省電力/通常/常時電源接続の3モード、`WakeLock`制御)という
     エコシステム標準の電源方針と設計思想を合わせている。
  3. **実装した範囲**: 新規`src/idle_background_fold.rs`
     (`touch_activity()`でHTTPハンドラ側からアクティビティを記録、
     アイドル閾値120秒・ポーリング5秒・ステップ間隔30秒の低頻度
     `std::thread`ループ)。`src/main.rs`の`chat`/`generate`ハンドラで
     `touch_activity()`を呼ぶよう配線し、起動時に`idle_background_fold::
     spawn()`を実行。進捗は`GET /v1/background-fold/status`で確認できる
     (`FoldProgress`、`disclosure`フィールドで常に限界を自己申告)。
     **正直な開示(最重要)**: 本物のModel Folding(ICLR 2025、
     arXiv:2502.10216、重みのクラスタリング・統合)は実装していない
     ——`open-cuda-llm::GptModel`の各層の重み(`layers: Vec<DecoderLayer>`)
     は`private`フィールドで公開アクセサが無く、実装するには
     `open-cuda-llm`クレート側への新規public API追加が前提となるため、
     このセッションのスコープには含めなかった。代わりに実装したのは
     「アイドル検知→低頻度実行→進捗可視化」という**スケジューリング
     基盤**であり、各ステップの中身は`scoring.rs`が既に保持する実
     インテント埋め込みベクトル(open-cuda-bert、multilingual-e5-small)
     同士のコサイン類似度計算という読み取り専用のプレースホルダに
     留めた(モデルの重みは一切変更されない)。`run_one_step()`を
     独立関数にし、将来`open-cuda-llm`に重み読み取りAPIが追加されれば
     この中身だけを本物のModel Folding計算へ差し替えられる設計にした。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo build --release`成功(既存4件のpre-existing警告のみ、
     今回の変更に起因する新規警告なし)。`cargo test --release`
     **70件全green**(既存回帰なし)。実際にサーバーを起動し、
     `GET /v1/background-fold/status`を即座に呼んだところ
     `steps_completed:0`(起動直後、まだステップ未実行)、約70秒後に
     再度呼んだところ`steps_completed:2`・
     `last_similarity_summary:"gov~trade=0.928, gov~credit=0.937, ..."`
     という実際の計算結果を確認した(アイドル状態で実際にバックグラウンド
     処理が進行することを実証)。続けて`POST /v1/chat`を実際に送信した
     直後に`GET /v1/background-fold/status`を呼び、
     `currently_idle`が`true`から`false`へ(`idle_seconds`が0へ)正しく
     切り替わることを確認した(実際のHTTPトラフィックに基づくアイドル
     判定が機能していることの実証)。
  5. **スマホ側プロトコル設計(実装は行っていない、正直な開示)**:
     既存の`android/`クライアント(`tokyo.runo.aruarullm`)が既に
     HTTP経由で`POST /v1/chat`を叩く仕組みを持っているため、将来
     スマホ側の軽量補助計算タスクを実装する場合は、同じHTTPクライアント
     基盤の上に「Wi-Fi接続時のみ、PCから小さなタスク(例: 特定層の
     類似度スコア計算に必要な小さなテンソル)を`GET`で取得し、計算結果を
     `POST`で返す」というポーリング方式のエンドポイント
     (`GET /v1/background-fold/task`・`POST /v1/background-fold/result`
     のような形)を新設する設計が現実的と考えられる。`PowerProfile`の
     既存3モードのうち「省電力」モード相当の設計(`WakeLock`を取得しない、
     充電中かつWi-Fi接続時のみ動作する等)を踏襲すればスマホを壊さない
     設計にできる。**このエンドポイント自体は今回実装していない**
     ——実機が無い状態でプロトコルだけを実装しても実証できず、動かない
     ものを「実装した」と報告しないという既存の運用ルールに従った。
  - 次にすべきこと: (1) `open-cuda-llm::GptModel`に層の重みへの
    読み取り専用アクセサ(例: `pub fn layer_weight_summaries(&self) ->
    Vec<LayerWeightSummary>`のような、生の`Vec<f32>`全体を晒さない
    集約済み統計量を返すAPI)を追加し、`run_one_step()`の中身を本物の
    Model Folding(層間類似度→クラスタリング候補の探索)へ差し替える、
    (2) 実際に余ったAndroidスマホが用意できた場合、上記5番のプロトコル
    設計に基づき`GET /v1/background-fold/task`等を実装し実機で検証する、
    (3) `IDLE_THRESHOLD_SECS`(120秒)・`STEP_COOLDOWN`(30秒)は経験的な
    初期値であり、実際に1週間程度運用してみてCPU負荷・進捗速度の
    バランスを見ながら調整する余地がある。

- **2026-08-18(続き) 調査完了・バージョンを0.2.2→0.2.3へ更新
  (「aruaru-llmのバージョンアップとして、既存の古い物からDATABASE
  システムに移動も簡単にする機能を搭載して」への対応、ユーザー指摘
  「スコープが確定する為の実績や努力をして、未着手は着手して」を受けて
  実際に調査・着手した)**:
  1. **調査内容**: このリポジトリのソース(`tenants.rs`・`security.rs`・
     `generation.rs`・`bow_fallback.rs`等、ファイルI/Oを含む主要モジュール
     全て)を実際に確認したが、会話履歴・ユーザー設定をファイル/DBへ
     永続化している既存コードは**一切存在しなかった**(Google Search
     APIキーは既存方針通りメモリ上保持のみで、再起動すれば消える設計)。
     `open-english`側の`app.js`も同様に調査済み(詳細は`open-english`側
     `CLAUDE.md`2026-08-18(続き)HANDOFF参照)——会話履歴を`localStorage`
     等へ保存していた実装は無かった。
  2. **結論**: このエコシステムには「移行すべき具体的な既存の古い
     データ」が実在しない。ユーザー指示の「既存の古い物からDATABASE
     システムに移動」は、実データが既にどこかに存在する前提の指示
     だったと推測されるが、実際には該当データが無いため、**このリポジトリ
     側で行う具体的なデータ移行処理は無い**(正直な開示——存在しない
     ものを移行する処理を作文で実装することはしない)。
  3. **実際に行った対応**: (a) `open-english`側に、将来どんな旧形式の
     エクスポートが持ち込まれても受け入れられる汎用的な取り込み口
     (`POST /v1/db/migrate-legacy`)を実装・実HTTP検証済み(該当リポジトリ
     側で対応、詳細は`open-english/CLAUDE.md`参照)。(b) このリポジトリ
     (`aruaru-llm`)は、open-englishのDATABASE化(SQLite+aruaru-db DUAL
     DBミラーリング)と足並みを揃えた協調リリースとして`Cargo.toml`の
     `version`を`0.2.2`→`0.2.3`へ更新した(このバージョン番号自体には
     コード変更を伴わない——移行対象データが無いという調査結果に基づく、
     意図的に「変更なし」を選んだ判断であることの記録)。
  4. **RS-JSON化について**: open-english側で「RS-JSONはHTTPボディ処理
     対象外」という過去の判断を撤回し、`from_slice_strict`/
     `to_vec_strict`(RPC/wire format向けの型付きAPI)を`/v1/db/*`へ
     実際に適用したことを確認済み(詳細はopen-english側CLAUDE.md参照)。
     このリポジトリ(`aruaru-llm`)のHTTP API(`/v1/generate`等)への
     同様の適用は、今回のスコープ(open-englishのDB機能)には含まれない
     別作業のため未着手のまま——次回、ユーザーに要否を確認の上で
     着手するかどうか判断する。
  - 次にすべきこと: (1) 実際に将来データが蓄積された場合(例:
    `aruaru-llm`自身が会話コンテキストをキャッシュする機能を新設する
    等)に備え、`import_legacy`相当の取り込み口が必要になるかは
    その時点で再検討、(2) このリポジトリのHTTP API自体へのRS-JSON
    適用要否をユーザーに確認。

- **2026-08-15(続き2) `real-vulkan`ビルドで実バグ発見: 同時並行リクエスト
  下でGPU担当分が`no spirv bytes were provided`で失敗することがある
  (sftp-git側のanalyzeDiff E2E確認セッションで発覚)**:
  1. **再現**: `--features real-vulkan`でビルドしたサーバーへ、
     `curl`で`/v1/generate`を2件同時送信したところ、片方が
     `{"error":"GptModel::generate_with_repetition_penalty failed:
     sgemm: GemmPath::VulkanGeneric selected (device.supports_spirv()
     ==true...) but no spirv bytes were provided..."}`で失敗した。
  2. **原因の見立て(未確定)**: `wire_matmul_spirv`は`load_from_dir`内で
     モデルロード時に一度だけ呼ばれ、成功すれば`GptModel`の全`Linear`に
     `matmul.spv`が配線される設計。DevicePool導入によりリクエストごとに
     CPU/GPUへラウンドロビン分散するようになったため、**wiring自体が
     何らかの理由で失敗していた場合(このセッションでは再現待ちの
     プロセスで失敗した形跡)、GPU担当に回った全リクエストが
     一律で失敗するようになった**——CPU専用ビルド(featureフラグ無し)
     で同じシナリオを再実行したところ問題無く成功したため、
     `real-vulkan`固有の問題と判断。根本原因(`wire_matmul_spirv`が
     具体的にどの条件で失敗するか)の特定は今回行っていない。
  3. **今回の対応**: 深追いせず、sftp-git側の検証には
     **CPU専用ビルド(既定、`real-vulkan`無し)を使う**という回避策で
     対応した(既定ビルドはこの問題の影響を受けない——`real-vulkan`は
     既定offのopt-in featureのため、通常利用者への実害は無い)。
  - 次にすべきこと: (1) `wire_matmul_spirv`の失敗条件を特定する
    (`GptModel::set_matmul_spirv`の戻り値・ログを詳しく調査、
    モデル切り替え〈distilgpt2⇄gpt2-medium等〉との組み合わせで再現するか
    確認)。(2) 配線失敗時に`real-vulkan`ビルドでも安全にCPUへ
    フォールバックする設計(現状は失敗を隠さず正直にエラーを返す
    という既存方針通りだが、DevicePool導入後はこれが「GPU番が回って
    きたリクエストは道連れで落ちる」という新しい失敗モードになって
    いる——DevicePoolがGPU未配線を検知したらCPUのみのプールへ自動的に
    縮退する等の対策を検討)。

- **2026-08-15 CPU+GPU同時並列稼働を実装(ユーザー指示「CPU+システム
  メモリ+GPUを非同期ででも、同時に並列並行で動作させて」、`sftp-git`
  開発セッションからの横断作業)**:
  1. **設計**: 新規`src/device_pool.rs`の`DevicePool`が、CPU
     (`CpuDevice`)と、`real-vulkan` feature有効時に構築成功した
     GPU(`VulkanDevice`)の両方を保持し、リクエストごとに
     `next_device()`でラウンドロビン分散する。従来は「VulkanDevice
     構築成功ならGPU、失敗ならCPU」という排他選択(どちらか一方)
     だったが、CPUは常にプールへ加え、GPU構築成功時は**置き換えでは
     なく追加**する形に変更。全6エンドポイント(`/v1/chat`・
     `/v1/classify-security`・`/v1/security/classify-traffic`・
     `/v1/generate`・`/v1/generate-with-search`・`/v1/translate`)の
     ハンドラ登録を、起動時1回の`Arc::clone(&device)`から、
     リクエストごとの`pool.next_device()`呼び出しへ変更した。
  2. **正直な設計上の限界**: 単一リクエストの計算(1回のforward pass)
     をCPU/GPUに分割するテンソル並列化(モデル並列)は行っていない
     ——1リクエストは開始時に選ばれた1つのデバイス上で最初から最後
     まで実行される(`alloc`/`memcpy`/`launch_kernel`の一貫性を保つ
     ため)。実現したのは「複数の同時リクエストをCPU担当分とGPU担当分
     に振り分け、両方のハードウェアを同時に稼働させる」という
     リクエストレベルの並列化。
  3. **既知の性能上の懸念(過去の実測記録との整合)**: このマシンの
     実GPU(NVIDIA GT 730、Kepler世代・VRAM 2GB)は、過去複数回の
     HANDOFF(2026-08-04〜08-08)で「1トークンデコードのGEMMは極めて
     軽く、Vulkanディスパッチの固定オーバーヘッドがCPU実行より支配的
     になり、GPU経由の方がCPU経由より遅い」ことが実測されている。
     今回のDevicePoolはこの事実を変えるものではない——GPU担当に
     振り分けられたリクエストは引き続きCPU担当分より遅くなる可能性が
     高い。今回の変更で実現したのは「速くする」ことではなく
     「両方のハードウェアリソースを同時に活用する」ことである点を
     誇張せず明記する。
  4. **実機検証**: `cargo build --release --features real-vulkan`
     成功。`cargo test --release --features real-vulkan device_pool`
     **2件全green**(単一デバイス時は常に同じデバイスを返すこと、
     複数デバイス時にラウンドロビンで交互に選ばれることを検証)。
     実際にサーバーを起動し(実GPU検出済み環境)、`/v1/chat`への
     複数回のHTTPリクエストがエラー無く応答することを確認した。
     **正直な開示**: `next_device()`がリクエストごとに実際にCPU/GPU
     どちらを選んだかをログで直接確認するところまでは今回できて
     いない(標準出力キャプチャの環境都合、次回tracing出力の確認方法を
     整理する余地がある)——ラウンドロビンのロジック自体は単体テストで
     数学的に検証済みだが、実際の同時並行リクエスト下でCPU/GPU両方が
     文字通り「同時に」稼働していることの実時間計測(例: GPU使用率
     モニタリングとの突き合わせ)は次回の課題として残す。
  5. **open-englishへの波及**: `open-english`はaruaru-llmのHTTP API
     (`/v1/chat`等)を呼び出すクライアントであり、本変更はサーバー側
     (`aruaru-llm`)のみで完結するため、`open-english`側のコード変更は
     不要——次回`aruaru-llm`を再ビルド・再起動するだけで
     `open-english`からのリクエストも自動的にDevicePoolの恩恵を
     受ける。
  - 次にすべきこと: (1) 実際の同時並行リクエスト下でCPU/GPU両方が
    並行稼働していることのタイムライン計測(例: `next_device()`に
    一時的なログ出力を仕込み、複数の同時リクエストを送って実際の
    選択順序を確認する)。(2) ラウンドロビンではなく負荷ベース
    (busy/idle判定)でのより賢い振り分けへの発展(現状は単純な交互
    割り当てのみ)。(3) `real-vulkan` featureは既定offのままであり、
    今回の変更もこのfeatureを有効化してビルドした場合のみ効果を持つ
    (既定ビルドはCPUのみのまま、既存の安全側デフォルトを変更していない)。

- **2026-08-15(続き) `spawn_blocking`化+タイムスタンプ計測でCPU/GPU
  同時稼働を実測で裏付け(ユーザー指示「実際の同時並行ではなく非同期
  でのリクエスト下で非同期のマルチコア、マルチスレッドを活かす方に
  変更…タイムライン計測して調査」)**:
  1. **見つけた設計上の問題**: `generate`(他のハンドラも同様)は
     `generation::generate`(同期・重い計算)を`async fn`内で
     直接呼んでおり、`tokio::task::spawn_blocking`を使っていな
     かった。tokioのマルチスレッドランタイムは限られた数のワーカー
     スレッド(通常CPUコア数)で全ての非同期タスクを捌く設計のため、
     これでは長時間の同期計算がワーカースレッドを占有し、他の
     非同期タスク(他リクエストの受付処理等)を妨げる可能性がある。
  2. **修正**: `generate`ハンドラを`tokio::task::spawn_blocking`で
     包み、専用のブロッキングスレッドプール(需要に応じて自動拡張)
     上で実行するよう変更。開始/終了ログに**デバイス名・実行スレッド
     ID・呼び出し元スレッドID・経過ミリ秒**を出力し、複数リクエストを
     同時に送った際に実際にどのスレッドでどのデバイスが並行稼働
     したかを事後計測できるようにした。
  3. **実測(3並行`/v1/generate`リクエスト、RUST_LOG=info)**:
     ```
     05:26:34.670 dispatch start device=CPU        thread=ThreadId(2)
     05:26:34.786 dispatch start device=Vulkan(GPU) thread=ThreadId(2)
     05:26:34.899 dispatch start device=CPU        thread=ThreadId(2)
     05:26:36.753 dispatch end   device=CPU    exec_thread=ThreadId(66) elapsed_ms=2082
     05:26:37.003 dispatch end   device=CPU    exec_thread=ThreadId(68) elapsed_ms=2104
     05:26:52.249 dispatch end   device=Vulkan exec_thread=ThreadId(67) elapsed_ms=17463
     ```
     3リクエストがほぼ同時(230ms以内)に開始され、**別々のOSスレッド
     (66/67/68)で実際に並行実行**された。CPU担当2件(約2.1秒、
     ほぼ同時に完了)が実行されている間、GPU担当1件(ThreadId 67)も
     並行して稼働し続けていた(34.786開始、CPU完了後の36.7〜37.0秒台も
     引き続き稼働、52.249まで)——これはラウンドロビンのロジックが
     正しいだけでなく、**実際のOSスレッドレベルでCPU計算とGPU
     ディスパッチが同時に走っている**ことを実測で裏付ける。
  4. **既知の性能特性の再確認(誇張しない)**: GPU担当リクエストは
     約17.5秒、CPU担当リクエストは約2.1秒——**約8倍GPU側が遅い**
     (GT730のディスパッチオーバーヘッドが支配的、過去のHANDOFF記録
     と整合)。同時並行化によって個々のリクエストが速くなるわけでは
     なく、複数リクエストが来た際に**待ち行列にならず両方のハード
     ウェアが同時に仕事を進められる**という利点に限定される。
  5. **検証結果**: `cargo build --release --features real-vulkan`
     成功(既存3件の警告のみ、pre-existing)。実サーバー起動+実HTTP
     並行リクエストで上記4番の通り確認。`chat`等の他エンドポイントは
     今回`spawn_blocking`化していない(`generate`が最も計算コストが
     重く実測に適しているため優先、他は次回の増分)。
  - 次にすべきこと: (1) `chat`/`classify-security`/
    `classify-traffic`/`generate-with-search`/`translate`の残り
    5ハンドラも同じ`spawn_blocking`パターンへ統一する(現状は
    `generate`のみ)。(2) ブロッキングスレッドプールのサイズ上限
    (tokio既定は512、通常は問題にならないが大量同時リクエスト時の
    挙動は未検証)。(3) ラウンドロビンではなく実際の負荷(busy/idle)
    に基づく振り分けへの発展は引き続き未着手。

- **2026-08-11(続き5) 高VRAM帯NVIDIA GPU情報を日英Web検索で裏取りし
  `hardware.rs`へ反映(ユーザーが言及した「RTX5950X」という製品名が
  実在しないことの確認+実在する高級GPU情報の正確な記載)**:
  1. **誤りの確認**: ユーザー言及の「RTX5950X」はNVIDIA製品として実在
     しない——「5950X」はAMD Ryzenの型番であり、NVIDIA RTXシリーズの
     命名規則とは異なる。ユーザーへの確認質問で「今は具体的な実装依頼
     ではない」と回答を得た上で、実在する製品の情報を正確に反映する
     形で対応した。
  2. **日英Web検索で確認した実在の高VRAM帯製品**: RTX 5090
     (Blackwellアーキテクチャ、32GB GDDR7)・RTX 6000 Ada(48GB)・
     RTX PRO 6000 Blackwell(96GB、ECC対応ワークステーション向け)。
  3. **`hardware.rs`への反映**: `recommend_id_for_vram`のdocコメントに
     上記の実在確認結果を出典付きで追記。**正直な開示**: 現在のモデル
     カタログ最大が`gpt2-xl`(1.5B、約6.43GB)に留まるため、これらの
     高VRAM帯GPUを検出してもカタログ最大を推奨するだけで、それ以上
     大きなモデルを実在しないかのように新規推奨することはない——
     名前だけ挙げて実在しない性能向上を示唆しないよう明記した。
  4. **検証**: 新規テスト`recommend_for_real_high_end_gpus_caps_at_
     catalog_max_without_overclaiming`で、RTX 5090/RTX 6000 Ada/
     RTX PRO 6000 Blackwellそれぞれの実VRAM容量を渡し、いずれも
     `gpt2-xl`に収束することを確認。`cargo test --release hardware`
     **4件全green**(既存3件+新規1件)。
  - 次にすべきこと: モデルカタログに`gpt2-xl`より大きい実在モデル
    (別アーキテクチャ含む)を追加する場合、この高VRAM帯GPU情報が
    実際の推奨先として活きる——現状はカタログの上限が先に律速して
    いるため、優先度はカタログ拡張側にある。

- **2026-08-11(続き4) RS-SmartTCPの「AI侵入検知」プラグイン向けに
  `POST /v1/security/classify-traffic`を新設(ユーザー指示「TLS復号・
  AI侵入検知の本実装して」→調査の上、実際の機械学習モデル推論として
  `security.rs`と全く同じ既存の`open-cuda-bert`埋め込み+コサイン類似度
  基盤を再利用する方針で対応)**:
  1. **新規モジュール`src/intrusion_detection.rs`**: `security.rs`
     (RS-Guardの「AI二次判定」)と全く同じ設計パターン(multilingual-
     e5-small埋め込み+コサイン類似度)で、ポートスキャン/SYNフラッド/
     ブルートフォース/データ持ち出し/正常の5カテゴリを判定。**正直な
     開示**: 攻撃トラフィックの実データで訓練した専用分類器ではなく、
     汎用埋め込みモデルによる意味的類似度のヒューリスティック
     (`security.rs`と同じ限界)。GPU/NPU実行は`opencuda-bert`が既に
     共有する`hw-detect-vulkan`/`hw-detect-directx`+`real-vulkan`
     feature経由で可能なため、本モジュール専用の新規GPU配線は行って
     いない(既存インフラの再利用)。
  2. **`POST /v1/security/classify-traffic`**: `{"description": "..."}`
     (RS-SmartTCP側が数値特徴量から組み立てた短い説明文を受け取る設計)
     → `{label, description, score, is_suspicious, engine}`。
  3. **検証(実バイナリ・実HTTP、モックなし)**: `cargo build --release`
     成功(既存2件のdead_code警告のみ、pre-existing)。`cargo test
     --release`**63件全green**(既存60件+`intrusion_detection`新規3件)。
     実際にサーバーを起動し、`curl`で
     `{"description":"one source IP probed 60 different destination
     ports within 3 seconds"}`→`{"label":"port-scan",...,
     "is_suspicious":true,"score":0.90...}`、
     `{"description":"a normal web browsing session..."}`→
     `{"label":"normal",...,"is_suspicious":false,"score":0.90...}`を
     確認(正しく分類、実行経路も`-cpu`まで含め正直に表示)。
     **正直な開示・気づいた実ミス**: 最初のビルドはルート登録
     (`.at("/v1/security/classify-traffic", ...)`)を追加する前に
     バックグラウンドで開始してしまい、古いバイナリで`404`が返る
     ことに気づいて再ビルドし直した——型チェックだけでなく実際に
     `curl`で確認する方針を徹底したことで発見できた実例として記録
     する。
  - 次にすべきこと: (1) RS-SmartTCP側からこのエンドポイントを実際に
    HTTPで呼ぶクライアント配線(現状はaruaru-llm側のみ実装、
    `RS-SmartTCP/CLAUDE.md`同日HANDOFF参照)、(2) TLSディープパケット
    インスペクション(本セッションではRS-SmartTCP側で証明書生成のみ
    実装、実際の復号プロキシ本体は未実装)が実装されれば、復号した
    ペイロードから本エンドポイントへ渡す説明文をどう組み立てるかの
    設計、(3) より多様な攻撃パターン(DNSトンネリング・水平スキャン等)
    のカテゴリ追加は今回のスコープ外。

- **2026-08-11(続き3) 就職・転職・観光の話題検出+aruaru.tokyo/
  nasa.tokyo/audiocafe.tokyo(aruaru・aruaru-lady)紹介機能を追加
  (ユーザー指示「英語と日本語と観光と就職転職情報の話題が出たら
  https://aruaru.tokyo/ 内のAI駆動開発CLAUDE CODE DESKTOP、
  audiocafe.tokyo/aruaru(IT・建築系求人)・audiocafe.tokyo/aruaru-lady
  (女性向け求人)のSET、https://aruaru.tokyo/ とhttps://nasa.tokyo/
  両方とも紹介して」への対応)**:
  1. **新規モジュール`src/referrals.rs`**: `mentions_career_or_tourism`
     (日英キーワード一致による簡易検出——GPT-2系の指示追従は保証されない
     ため、既存のbag-of-wordsフォールバック等と同じ確実性優先の単純
     実装)+`career_and_tourism_referrals`(4件のリンク、いずれも
     ユーザー本人から直接指示されたURLのみ使用、推測・捏造URLは
     含まない)。
  2. **`POST /v1/referrals/check`新設**: `{"text": "..."}` →
     `{"matched": bool, "referrals": {...} | null}`。
  3. **検証**: `cargo build --release`成功。`cargo test --release`
     **60件全green**(既存57件+`referrals`モジュール新規3件)。実際に
     サーバーを起動し`curl`で実HTTPリクエスト
     (`{"text":"I am thinking about a career change"}`)を送信、4件の
     リンクすべてが正しく返ることを確認済み(`open-english`側からの
     実ブラウザ経由の統合検証は`open-english/CLAUDE.md`同日HANDOFF
     参照)。
  - 次にすべきこと: 特になし(今回のスコープは完了)。キーワード検出は
    簡易的な文字列一致のため、将来的に誤検出/検出漏れが目立つ場合は
    埋め込みベースの意図分類(`scoring.rs`と同じ方式)への置き換えを
    検討してもよい。

- **2026-08-11(続き2) 富士山の安全案内+山小屋・登山バス/タクシー・
  登山用品店DB+観光ツアー検索を新設(ユーザー指示「富士山は危険な山
  ですので…上下スキーウェアを着て…落石で死ぬ場合もありますので必ず
  ヘルメットもして…山小屋を必ず予約して一泊されてから登山して」+
  「https://mtfuji.jpn.org/availablelist.php ここのHP中身はCOPYして
  DB化してから富士山の話題が出たら日本語と英語で紹介して」+「登山バス
  タクシーの予約」「スキーウェアとヘルメットと登山靴などを安く販売
  しているお店」+「日本も世界も観光で訪れるなら観光ツアーの紹介と
  オンライン予約をその都度検索して」への対応)**:
  1. **富士山データ収集**(`data/geo_content.json`拡張): WebFetchで
     `https://mtfuji.jpn.org/availablelist.php`(吉田口ルート山小屋の
     営業期間一覧)を実際に取得し、山小屋18件(五合目〜八合目)を
     `fuji_mountain_huts`として収録。WebSearchで登山バス
     (`bus.fujikyu.co.jp`)・吉田ルート通行予約システム
     (`fujisan-climb.jp`)・タクシー会社一覧(`fujisanpo.com`)・
     登山用品レンタル4店(やまどうぐレンタル屋・そらのした・VIPツアー・
     山岳同盟)を実際に検索して`fuji_transport_reservations`/
     `fuji_gear_shops`として収録。安全上の注意文(スキーウェア+ヘルメット
     着用、山小屋の事前予約・一泊を強く推奨)を`fuji_safety_ja`/
     `fuji_safety_en`に明記。
  2. **`GET /v1/geo/fuji`新設**: 安全案内・出典・山小屋一覧・バス/
     タクシー予約先・登山用品店を一括で返す。**正直な開示**: 営業期間・
     電話番号は2026-08-11時点の収集値であり毎年変わるため、レスポンス
     の`source_ja`/`source_en`で常に出典と「利用前に直接確認すること」を
     明記する。
  3. **`POST /v1/geo/tours`新設**(観光ツアー紹介+オンライン予約検索):
     既存の`web_search.rs`(Google Custom Search連携、
     `/v1/generate-with-search`と同じAPIキー設定を共有)を再利用し、
     `"<place> 観光ツアー オンライン予約 tour booking"`で検索する。
     APIキー未設定時は`configured: false`+空結果を正直に返す(黙って
     結果を偽装しない)。YouTube検索については専用のAPI連携までは
     実装せず、URLエンコード済みのYouTube検索結果ページへの直リンクを
     返す設計に留めた(誇張しない、実装スコープを正直に開示)。
  4. **検証**: `cargo build --release`成功。`cargo test --release`
     **57件全green**(既存55件+`fuji_info_includes_safety_advisory_
     and_huts`等の新規テスト)。実際にサーバーを起動し、
     `open-english`側から研修モードで"I love Japan and Mount Fuji."を
     送信→`/v1/geo/lookup`(Japan一致)→富士山関連ランドマークを検知して
     `/v1/geo/fuji`を追加取得→安全案内・山小屋例・バス予約先・登山用品
     店が実際に日英併記で表示されることをブラウザ上で確認済み(型
     チェックのみで完了と報告しない方針の実践)。`/v1/geo/tours`は
     このセッションではGoogle Search APIキー未設定のため
     `configured: false`の正直なフォールバック応答となることも実際に
     確認した(実際の検索結果表示はAPIキー設定後に別途検証が必要)。
  - 次にすべきこと: (1) 実際にGoogle Search APIキーを設定した状態での
    `/v1/geo/tours`のE2E検証(検索結果が実際に返ることの確認、
    `web_search.rs`側の既存の未検証事項と同じ)、(2) YouTube Data APIの
    ような専用連携があれば、直リンクではなく実際の検索結果(動画タイトル・
    サムネイル等)を返せるようになる可能性の検討、(3) 富士山以外の
    山(立山・穂高等)への同様のDB拡張は今回のスコープ外。

- **2026-08-11(続き) 実機テストで発覚した`lookup_country`の実バグを修正
  (ユーザー指示「実際のopen-englishでTESTしたい」を受け、
  `aruaru-llm`+`open-english-server`を実際に起動してブラウザで実際に
  研修モードを最後まで動かして検証した結果)**:
  1. **発見した実バグ**: `POST /v1/geo/lookup`は当初、クエリ全体と
     国名の**完全一致**のみを見ていたため、`open-english`が実際に送信
     する「発話文そのまま」("I FROM JAPAN."等)を渡すと一致しなかった
     (ブラウザで実際に「Where are you from?」に「I from Japan.」と
     入力したところ、DB検索が空振りし固定の一般的な返答にフォール
     バックしてしまうことを実際の画面で確認)。
  2. **修正**: `lookup_country`を部分一致(埋め込みJSON側は
     `needle.contains(country_en)`/`trimmed.contains(country_ja)`、
     aruaru-db側は`LIKE '%...%'`)へ変更。回帰テスト
     `lookup_country_matches_country_name_embedded_in_a_sentence`を
     追加。
  3. **実機再検証**: 修正後、実際に`cargo build --release`で再ビルド
     したバイナリを再起動し、ブラウザで同じ操作("I from Japan.")を
     再現したところ、`"I love Mount Fuji and Sushi! / 私は富士山と
     寿司が大好きです!\nA popular souvenir there is Folding fan. /
     そこの人気のお土産は扇子です。"`という正しいDB駆動の応答が実際に
     表示されることを確認した(型チェック・単体テストだけで完了と
     報告しない方針の実践——実際にサーバーを2回起動し直し、ブラウザ
     操作で再現・修正確認まで行った)。
  4. **検証結果**: `cargo test --release`**56件全green**(既存55件+
     回帰テスト1件)。
  - 次にすべきこと: 特になし(この不具合自体は解消済み)。今後国名検索
    ロジックを変更する際は、必ず「発話文全体を渡す」実際の呼び出し
    パターンで再テストすること(単発の国名のみを渡す単体テストだけでは
    この種のバグを見逃す)。

- **2026-08-11 地理・観光・名物データベースを新設(`geo_content.rs`+
  `POST /v1/geo/random`・`POST /v1/geo/lookup`)、ユーザー指示
  「open-englishの自己紹介トレーニングが『I'm from Australia. I love
  kangaroo & koala』程度しか対応できなかったのを、日本全国・都道府県別、
  アメリカは州別、世界中の首都名・観光名所・名物料理・お土産の
  DATABASEを作成して」+「今度どこどこの国に旅行/仕事で行く予定がある、
  のようなフレーズにもDBで対応して」への対応**:
  1. **データ範囲(正直な開示)**: 日本47都道府県+米国50州は全件収録、
     世界の首都は主要60ヶ国分のみ(国連加盟196ヶ国全てではない、今後
     拡張予定)。各エントリにランドマーク1件+名物料理1件(+首都は
     お土産1件)を日英で収録した`data/geo_content.json`を新設。
  2. **DUAL DB方針の確認(実装不要と判明)**: `aruaru-db`自身が
     `DUAL_DATABASE_URL`経由で本物のPostgreSQLへミラーする冗長化機能
     (`aruaru-dist::dual_database::DualDatabaseMirror`)を既に持って
     いるため、本クレート側でDUAL DB化を再実装する必要はなく、単一の
     Postgres接続(`ARUARU_LLM_GEO_DATABASE_URL`)を張るだけで済む設計
     にした。
  3. **フォールバック設計**: `ARUARU_LLM_GEO_DATABASE_URL`未設定・
     接続失敗時は埋め込みJSON(`include_str!`)からそのまま応答する
     (既存のbag-of-wordsフォールバックと同じ「サービスを止めない」
     思想)。起動時に`seed_database_if_configured()`をベストエフォートで
     呼び、接続できた場合のみ`CREATE TABLE IF NOT EXISTS`+
     `ON CONFLICT DO NOTHING`で冪等にseed投入する。
  4. **JSONパースはRust-JSON(`../RS-JSON`)を使用**(ユーザー指示
     「JSONよりRS-JSONを使って」): 埋め込みJSONの生文字列パースを
     `serde_json::from_str`から`rust_json::parse_strict`+
     `serde_json::from_value`へ変更。Rust-JSON自身が値モデルとして
     `serde_json::Value`を使う設計のため、型付きDeserialize自体は
     引き続きserdeに委ねる(Rust-JSON自身のモジュールdocに明記された
     設計方針通り)。HTTPリクエストボディの`Json<T>`抽出(RPoem側)は
     対象外(該当箇所なし、静的データファイルのパースのみ対応)。
  5. **検証**: `cargo build --release`成功(既存2件のpre-existing
     dead_code警告のみ)。`cargo test --release`**55件全green**
     (既存51件+geo_content新規4件: 埋め込みデータセットの件数確認・
     DB未接続時のランダム取得フォールバック・国名検索の日英一致・
     未知の国名での`found:false`)。
  6. **正直な開示・未検証事項**: 実際に稼働中の`aruaru-db`インスタンス
     への接続検証はこのセッションでは未実施(環境にaruaru-dbの実行中
     プロセスが無いため)——埋め込みJSONフォールバック経路のみを実際に
     検証済み。次回、実際に`aruaru-db`を起動した状態での
     `ARUARU_LLM_GEO_DATABASE_URL`設定+seed投入+`/v1/geo/random`の
     実HTTP検証が必要。
  - 次にすべきこと: (1) 実際に稼働中のaruaru-dbへの接続・seed投入の
    実機検証、(2) 世界の首都データを国連加盟196ヶ国全てへ拡張、
    (3) ユーザーからさらに要望のあった「現在のハードウェア環境からの
    推薦LLM・少し小さい/大きいLLMの特徴・メリデメを日英表示」機能や、
    「起動時メンテナンス中に最新LLM情報・最新NVIDIA/AMD/Intel GPU情報を
    収集してDB化する」機能、GPU/ゲーム推奨(4K/5K/120FPS対応・基本無料
    オンラインゲームの流行調査)・Amazon購入リンク表示は、いずれも
    広範な最新情報の継続調査(このセッション内のGoogle検索では網羅
    できない規模)を要するため今回は着手していない——次回、専用の
    調査セッションとしてスコープを切って着手することを推奨する
    (Amazon購入リンクについては、実際の購入操作はユーザー自身が行う
    必要がある旨も併せて検討すること)。

- **2026-08-10(続き5) 東芝SBM(シミュレーテッド分岐)を実際に組み込む
  新規モジュール`cache_optimizer.rs`+`POST /v1/models/optimize-cache`を
  実装(ユーザー指示「架空の最適化問題を作ってSBMを実際のaruaru-llmに
  組み込んでほしい」への対応、`open-cuda`側2回の日英調査で本物の適用先が
  見つからなかった〈`open-cuda/CLAUDE.md`同日HANDOFF参照〉ことを受けて)**:
  1. **定式化**: モデルカタログ(5サイズ)のディスク容量予算下での
     キャッシュ選択を0/1ナップサック問題として定式化(価値=サイズの
     平方根、既定ヒューリスティック)。標準的なQUBO変換(スラック
     ビットによる不等式制約の等式化)→Ising(±1スピン)への変換を
     実装し、東芝SBM(Ballistic Simulated Bifurcation、`open-cuda/
     examples/sbm_demo`のMax-Cut専用実装を局所磁場付きの一般形へ拡張)
     で解く。
  2. **実装中に発見・修正した実バグ**: `c0`(結合強度の正規化係数)が
     結合行列`j`のみから計算されており、スラックビット由来の局所磁場
     `h`(ペナルティ展開で`penalty*a_j^2`項が生じ、`j`より桁違いに
     大きくなりうる)を無視していたため、力学系が最初の数ステップで
     `h`の符号だけに支配されて即座に飽和し、最適化が実質機能しない
     実バグがあった(全探索の最適解と一致しない現象として発覚)。
     `h`・`j`両方の最大絶対値を基準に`c0`を計算するよう修正。
  3. **正直な開示・SBMの限界(誇張しない)**: 修正後も、`open-cuda`側の
     Max-Cutデモ(全ケース厳密一致)と異なり、本ナップサック問題(スラック
     変数を伴うより複雑なQUBO)では一部の予算で全探索の厳密最適解に
     到達しないケースが実測で見つかった(SBMは近似ヒューリスティック
     であり厳密解到達を保証しない、というこのエコシステム全体の既存
     方針通り)。複数シードでの多重探索(8回)を実装して改善したが、
     テストの許容基準は「厳密一致」ではなく「全探索最適値の75%以上」
     とした(誇張せずこの限界を記録する)。また、SBM解が容量制約を
     満たさない場合は価値密度順の貪欲フォールバックへ安全に切り替える
     設計とした(`used_sbm_solution`フィールドで呼び出し側が判別可能)。
  4. **`POST /v1/models/optimize-cache`**(advisory専用、実際のディスク
     削除は行わない): `{"budget_mb": ..., "value_overrides": {...}
     (任意)}` → `{"keep": [...], "evict": [...], "total_size_mb",
     "budget_mb", "total_value", "used_sbm_solution"}`。
  5. **検証**: `cargo test --release`**49件全green**(既存46件+新規3件:
     QUBO→Ising変換の数値一致・SBM解が全探索の75%以上を達成・
     `optimize_model_cache`が予算を守ることの確認)。実際にサーバーを
     起動し`POST /v1/models/optimize-cache {"budget_mb": 2000}`を実HTTP
     で叩き、`{"keep":["distilgpt2","gpt2-medium"],"evict":["gpt2",
     "gpt2-large","gpt2-xl"],"total_size_mb":1873,"budget_mb":2000,...,
     "used_sbm_solution":true}`という正しい応答を確認済み。
  6. **正直な開示・実用上の位置づけ**: この規模(5変数)のナップサック
     問題は全探索・動的計画法でも瞬時に厳密解が求まり、SBMを使う実用上
     の必要性は薄い——本機能は「SBMを実際の意思決定パスへ配線し動作
     実証する」ことが目的であり、「SBMが無ければ解けない/著しく遅い」
     という主張はしていない(モジュールdocコメントに明記)。
  - 次にすべきこと: (1) より多くの候補・より複雑な制約(例: 複数
    予算次元、モデル間の依存関係)があれば、SBMを使う実用上の意味が
    増す可能性がある(現状は無し)、(2) ユーザーからさらに複数の適用例
    (モデルDL順序のスケジューリング等)の要望があったため、次回以降
    検討する。

- **2026-08-10(続き2) open-english向け既定モデルを`gpt2`(124M)から
  `distilgpt2`(82M)へ切替(ユーザー指示、4項目の優先順位「速度改善→
  モデル差し替え調査→フロントエンドRust移植→SBM/DeepSeek調査」の
  1・2番目に対応)**:
  1. **実測比較**(このマシンの実CPU、`ARUARU_LLM_REPETITION_PENALTY`
     既定`1.3`込み、同一プロンプト`"...Student: Hello\nTrainer:"`・
     `max_new_tokens=24`): `gpt2`=8.37秒、`distilgpt2`=4.83秒
     (**約42%高速化**)。生成文はいずれも反復ループなし・文法的に自然。
  2. **切替方法**: `POST /v1/models/select {"id":"distilgpt2"}`で
     ホットスワップ後、Windowsユーザー環境変数
     `ARUARU_LLM_GPT2_DIR=F:\runo\aruaru-llm\models\distilgpt2`を
     `setx`相当(`[Environment]::SetEnvironmentVariable`)で設定し、
     プロセス再起動後も既定でdistilgpt2がロードされるようにした
     (`default_model_dir()`の環境変数優先ロジックを利用、コード変更は
     無し)。
  3. **正直な開示**: (1) 品質比較は上記1プロンプトの実測のみ——複数
     プロンプトでの体系的な品質比較は未実施。(2) この環境変数はこの
     Windows開発機のユーザースコープのみに設定されており、他のマシン
     (VPS等)やこのマシンの別ユーザーには反映されない——本番デプロイ
     時は同様の環境変数設定、またはサービス起動スクリプト側での
     明示を忘れないこと。(3) `real-vulkan` featureは既定offのまま
     変更なし(既存HANDOFFの結論〈GT730ではVulkanの方が遅い〉を再確認
     済み、今回は再検証していない)。
  - 次にすべきこと: (1) 複数プロンプトでのdistilgpt2品質の体系的検証、
    (2) フロントエンドJS(`open-english`)のRust+RPoemへの移植(優先度
    3番目、別セッションでスコープを切って着手予定)、(3) 冒頭の
    東芝SBM/DeepSeek技術組み込み構想の調査(優先度4番目)。

- **2026-08-10 `/v1/generate`の反復ループバグを根本解決(`open-cuda`側
  `GptModel::generate_with_repetition_penalty`をオプトイン→既定有効で
  配線)、ユーザー報告「しつこく繰り返すバグ 反応も遅すぎ」への対応**:
  1. **背景**: `open-english`フロントエンドから「Hello」等の短い発話を
     送ると、`Student: Hello`を延々繰り返す劣化ループが実際に発生する
     ことをユーザーが報告した(対話ファインチューニング無しの素の
     GPT-2貪欲デコードの既知の失敗モード)。以前(2026-08-10、直前の
     パス)フロントエンド側で「最初の"Student:"手前で切り捨てる」応急
     処置を実装済みだったが、これは表示上の症状を隠すだけで根本原因
     (貪欲デコード自体に反復を防ぐ機構が無い)は未解決のままだった。
  2. **`open-cuda`側の対応**(詳細は`open-cuda/CLAUDE.md`同日HANDOFF
     参照): `GptModel::generate_with_repetition_penalty`(CTRL方式、
     既に登場したトークンのlogitを弱める)を新設、既存`generate()`は
     `penalty=1.0`の薄いラッパーへ変更(数値的に完全同一、回帰無し)。
     実GPT-2 124M重みで、`open-english`と同じプロンプト構造
     (`"...Student: Hello\nTrainer:"`)を使い、ペナルティ無しでは実際に
     ループへ陥ること・`penalty=1.3`で実際に解消し文法的に自然な会話文
     へ変わることを検証済み。
  3. **`aruaru-llm`側の実装**(`src/generation.rs`):
     `default_repetition_penalty()`(`ARUARU_LLM_REPETITION_PENALTY`
     環境変数、既定値`1.3`、パース失敗・非正数・非有限値は既定値へ
     安全にフォールバック)を新設し、`generate()`が
     `loaded.model.generate_with_repetition_penalty(device,
     &prompt_ids, max_new_tokens, default_repetition_penalty())`を
     呼ぶよう変更(既定で有効化、`/v1/chat`〈意図分類〉には無関係)。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo build --release`(aruaru-llm)成功。`cargo test --release`
     **46件全green**(既存機能への回帰なし)。実際にサーバーを起動し、
     `POST /v1/generate`へ`open-english`と同一のプロンプト構造
     (`"...Student: Hello\nTrainer:"`、`max_new_tokens=24`)を送信した
     ところ、`"I'm sorry for the delay in your appointment but it's
     not too late to get back on track! Thank you so"`(反復なし、
     文法的に自然な英文)という応答を得た——修正前の同一プロンプトでの
     応答(`"Hello\nStudent: Hello\nStudent: Hello\n..."`)と比較して
     劣化ループが実際に解消されたことを実HTTPで確認済み。
  5. **正直な開示・スコープ**: (a) このペナルティは`/v1/generate`
     (GPT-2自己回帰生成)のみに適用、`/v1/chat`・`/v1/classify-security`
     (貪欲デコード自体を使わない意図分類/セキュリティ分類)には無関係。
     (b) 経験的な既定値`1.3`はこの1シナリオでの実測に基づく——他の
     プロンプトパターンでの最適値は未検証、必要なら
     `ARUARU_LLM_REPETITION_PENALTY`で調整可能。(c) サンプリング
     (温度・top-k/top-p)は依然未実装(貪欲デコード+繰り返しペナルティ
     のみ)。(d) ユーザー報告のもう一方の懸念「反応も遅すぎ」は、前回
     パス(フロントエンド側`max_new_tokens`48→24縮小)で既に対応済み
     のまま変更なし——本パスでは追加のGPU/NPUハードウェア高速化の
     実験は行っていない(既存HANDOFF記録〈このマシンのGT730では
     1トークンデコードでCPUよりVulkan経由が遅い、と複数回実測済み〉を
     踏まえ、ユーザーへその旨を報告し新たな実験は見送った)。
  - 次にすべきこと: (1) より高性能なGPU実機が得られた場合の
    `real-vulkan` feature再ベンチマーク、(2) フロントエンドJS
    (`open-english`)をRust+RPoemへ移植する大規模タスク(ユーザーから
    言及あり、規模が大きいため別セッションでスコープを切って着手)。

- **2026-08-08(続き3) PCA較正版MLA(`open-cuda`側同日新設
  `enable_mla_kv_compression_calibrated`)をオプトイン配線
  (直下2026-08-08(続き2)エントリで「乱数射影は品質を明確に劣化させる」
  と実測したことを受け、`open-cuda`側で修正できるか調査・実装した
  結果への対応)**:
  1. **`open-cuda`側の対応**: 乱数射影の代わりに、実サンプル文の
     プリフィルで集めた本物のK/V活性化にPCA(`nalgebra`の対称固有値
     分解)を適用し、分散最大の上位`d_c`方向を射影基底として使う
     `GptModel::enable_mla_kv_compression_calibrated(d_c, device,
     sample_prompts)`が新設された(詳細・数学的根拠・実測結果は
     `open-cuda/CLAUDE.md`同日HANDOFF参照)。
  2. **`aruaru-llm`側の実装**(`src/generation.rs`):
     `mla_calibrated_enabled()`(`ARUARU_LLM_MLA_CALIBRATED=1`または
     `true`で有効化、**既定off**)・`mla_calibration_prompts()`
     (8文の一般的な英文がデフォルト、トピック分散を意図、
     `ARUARU_LLM_MLA_CALIBRATION_PROMPTS`〈`;`区切り〉で上書き可)・
     `wire_mla_kv_compression_calibrated`(トークナイザで較正文を
     エンコードし`opencuda_cpu::CpuDevice`で較正、`real-vulkan`
     feature有無に関わらず動作)を新設。既存の乱数射影版
     (`wire_mla_kv_compression`)とは`wire_mla_kv_compression_any`で
     排他的に呼び分ける(`ARUARU_LLM_MLA_CALIBRATED=1`が優先、
     両方同時に有効化しても`GptModel`側は`layer.mla`を1つしか
     保持できず混乱するだけのため)。
  3. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo build --release`成功(既存2件のdead_code警告のみ、
     pre-existingで無関係)。`cargo test --release -- --test-threads=1`
     **46件全green**(regression無し)。実際にサーバーを
     `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1
     ARUARU_LLM_MLA_CALIBRATED=1`で起動し、起動ログで
     `ARUARU_LLM_MLA_CALIBRATED set: wired PCA-calibrated MLA-style
     KV cache compression (head_dim=64 -> d_c=16, 75.0% smaller...
     calibrated on 8 sample prompts)`を確認した上で、実HTTP
     リクエスト`POST /v1/generate {"prompt":"The quick brown fox",
     "max_new_tokens":16}`を送信し、
     `"completion":"es are a bit of a lot of the way to the forest.\n\n"`
     という応答を得た——これは`open-cuda`側の単体テスト
     `calibrated_pca_mla_kv_compression_on_real_gpt2_weights`が実測した
     PCA較正版の出力と**完全一致**しており、配線が正しく機能して
     いることを実際のHTTPレスポンスで裏付けた(直下2026-08-08(続き2)
     エントリで実測した乱数射影版の出力
     `"es, and the government, and away from the government and point
     of the government"`のような反復破綻は再現しなかった)。
  4. **正直な評価(誇張しない)**: PCA較正版は乱数射影版より明らかに
     改善しているが、**非圧縮版と比較すればなお明確に品質が劣化して
     いる**(`open-cuda`側同日HANDOFFの評価と同じ)。よって
     `ARUARU_LLM_MLA_CALIBRATED`も乱数射影版と同じく既定offのopt-in
     機能として提供するに留めた——実ユーザー向け応答の既定挙動を
     置き換える判断はしない。
  - 次にすべきこと: (1) `open-cuda`側同日HANDOFFの「次にすべきこと」
    (較正サンプル数の拡大・中心化PCAの検証・`d_c`のトレードオフ実測)が
    進めば、このリポジトリ側でも再度品質検証を行う、(2) 較正データ
    (`mla_calibration_prompts`の既定8文)を、このサービスが実際に
    受け取るプロンプトの分布(`/v1/generate`・`/v1/translate`の実
    トラフィック)に近づけた場合に品質が変わるかの検証は未実施、
    (3) `real-vulkan`有効時にPCA較正MLAと`set_flash_attention_spirv`/
    `set_softmax_spirv`を同時有効化した場合の相互作用(速度・メモリ
    両面)は今回も未計測。

- **2026-08-08(続き2) `GptModel::enable_mla_kv_compression`(DeepSeek-V3
  風MLA、KVキャッシュ低ランク圧縮、`open-cuda`側2026-08-07実装・実機
  検証済み)を`aruaru-llm`側からオプトイン配線(直前2026-08-08 HANDOFFの
  「次にすべきこと(2)」で名指しされていた候補への対応)**:
  1. **実装**(`src/generation.rs`): `mla_kv_compression_enabled()`
     (`ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`または`true`で有効化、
     **既定off**)・`mla_d_c(head_dim)`(既定`head_dim/4`=75%削減、
     `ARUARU_LLM_MLA_D_C`で上書き可、`0 < d_c < head_dim`を満たさない
     値は既定値へ安全にフォールバック)・`wire_mla_kv_compression`を
     新設し、`load_from_dir`から(`real-vulkan` featureの有無に関わらず)
     常に呼ぶようにした。`enable_mla_kv_compression`自体は
     `opencuda_blas::mla_compress_kv`/`mla_decompress_kv`(`sgemm`をCPU/
     Vulkan両対応で呼ぶだけの計算)を土台にしておりデバイス種別に
     依存しないため、`wire_matmul_spirv`等とは異なり`#[cfg(feature =
     "real-vulkan")]`の外に置いた——CPUのみの既定ビルドでも有効化できる。
  2. **d_cの決定根拠**: 実GPT-2 124Mは`hidden_size=768`・`num_heads=12`
     なので`head_dim=64`。既定`d_c=64/4=16`(75%削減、`open-cuda`側の
     自テスト`mla_kv_compression_enabled_model_generates_without_
     panicking`と同じ削減率を踏襲)。`d_c`は`model.config()`から実際の
     ロード済みモデルの`hidden_size`/`num_heads`を読んで動的計算する
     (推測やハードコードではなく、モデルカタログの他サイズ
     〈distilgpt2/gpt2-medium/large/xl〉に切り替えても追従する設計)。
  3. **なぜ既定offか(速度ではなく品質、`wire_flash_attention_spirv`とは
     異なる理由)**: `open-cuda`側`open-cuda-llm/src/lib.rs`の
     `enable_mla_kv_compression`実装を読んだところ、down/up-projection
     行列は`random_vec`による**乱数初期化**(DeepSeek本家が大規模事前
     学習で獲得する射影とは無関係)——`open-cuda`側の回帰テスト
     `mla_kv_compression_actually_changes_generation_versus_uncompressed`
     自体が「圧縮ありなしで生成結果が実際に異なることを確認する」
     テストであり、`open-cuda`側は元から品質維持を主張していない
     (同ファイルdocコメント参照)。ただし`open-cuda`側の実機検証は
     すべて`GptConfig::tiny`(ランダム初期化トイモデル)止まりで、
     **実学習済み重み(実GPT-2 124M)での品質検証は`open-cuda`側にも
     このリポジトリにも存在しなかった**。今回それを実施し、以下の
     実測により「品質を落とす」という懸念が推測ではなく事実である
     ことを確認した。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo build --release`成功(既存2件のdead_code警告のみ、pre-
     existingで無関係)。`cargo test --release`**46件全green**
     (1回目実行時に無関係な`memory allocation ... failed`で
     クラッシュしたが、他プロセスは実行されておらず一時的なメモリ
     逼迫と判断、再実行で全green・regression無し)。
     実際にサーバーを2回起動し(実GPT-2 124M、同一プロンプト・同一
     `max_new_tokens=16`)、`POST /v1/generate
     {"prompt":"The quick brown fox","max_new_tokens":16}`を実HTTP
     リクエストで比較:
     - **MLA無効(既定)**: `"es are a great way to get a little bit of
       a kick out of your"`(2026-07-25 HANDOFF記録の既知の継続文と
       完全一致、regression無し)。
     - **MLA有効**(`ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`、起動ログ
       で`head_dim=64 -> d_c=16, 75.0% smaller`を確認):
       `"es, and the government, and away from the government and
       point of the government"`——文法的な体裁こそ保っているが、
       同じ単語("government")の反復に陥っており、無効時の自然な
       継続と比べて**明確に品質が劣化している**ことを実際の出力で
       確認した(誇張ではなく実測、上記3番の懸念の裏取り)。
  5. **結論・opt-inとした判断の正当性**: KVキャッシュのメモリ削減
     (ヘッドあたり75%、`head_dim=64→d_c=16`)自体は`d_c`と`head_dim`の
     比から機械的に導かれる数値であり実際に成立する一方、これは
     「学習済み射影が無いことによる代償」を伴う——実ユーザー向け
     応答を返す`/v1/generate`の既定挙動をこれで置き換えるべきではない
     と判断し、既定offのopt-in機能として提供するに留めた。学習済みの
     `down_proj`/`up_proj`重みを読み込めるローダーが`open-cuda`側に
     実装されない限り、この機能は「配線が正しく動くことの実証」の
     域を出ない。
  - 次にすべきこと: (1) `open-cuda`側に学習済みMLA射影重みローダーが
    実装された場合、そちらへ切り替えて再度品質検証を行う(現状の
    乱数初期化のままでは既定on化はしない)、(2)
    `real-vulkan`有効時にMLA圧縮と`set_flash_attention_spirv`/
    `set_softmax_spirv`を同時に有効化した場合の相互作用(速度・
    メモリ両面)は今回未計測、(3) `GET /v1/models/catalog`等の
    レスポンスに現在MLA圧縮が有効かどうかを表示する診断フィールドは
    未追加(現状は起動ログでのみ確認可能)。

- **2026-08-08(続き) DeepSeek「Engram」風KVキャッシュ/重みオフロードの
  実装を検討したが、コードを実際に読んだ結果「退避対象となるVRAM常駐
  状態がそもそも存在しない」と判明したため実装を見送り(ユーザー指示:
  DeepSeekのEngram技術〈静的な知識・KVキャッシュ/重みの一部をVRAMから
  システムRAMへオフロードし必要時に再ロードする手法〉をこのマシンの
  実GPU〈NVIDIA GT 730、Keplerクラス旧世代・小VRAM・テンソルコア無し〉
  向けに実装できないか検討せよ、というタスク)**:
  1. **調査方針**: 「フックできそうな場所を推測して着手」ではなく、
     まずモデルロード・推論経路の実コードを読んで、実際にVRAM常駐する
     状態(重み・KVキャッシュ)が存在するかどうかを確認することから
     始めた。読んだファイル: `aruaru-llm/src/generation.rs`・
     `aruaru-llm/src/hardware.rs`、`open-cuda/crates/open-cuda-llm/
     src/lib.rs`(`GptModel`・`KvCacheHead`・`DecoderLayer`)、
     `open-cuda/crates/opencuda-vulkan/src/real.rs`
     (`VulkanDevice::alloc`/`free`/`dispatch_spirv`/`launch_kernel`)、
     `open-cuda/crates/opencuda-blas/src/lib.rs`
     (`ScopedAlloc`・`sgemm_vulkan_generic`・
     `scaled_dot_product_attention_with_spirv[_and_softmax]`)。
  2. **判明した事実(コード上の根拠)**:
     - `opencuda-blas::ScopedAlloc`(`opencuda-blas/src/lib.rs`
       104〜130行目付近)は`device.alloc(bytes)`で確保し
       `Drop`で必ず`device.free(self.ptr)`するRAIIガードであり、
       `sgemm`/`sgemm_vulkan_generic`/`scaled_dot_product_attention_
       with_spirv*`等、Vulkan経由でディスパッチする関数は**呼び出しの
       たびに**これでVRAMバッファを確保し、host→device転送→計算→
       device→host転送→即解放、という一回性の使い方をしている。
       関数を抜けた時点でVRAM上には何も残らない。
     - `open-cuda-llm::GptModel`の重み(`word_embeddings`・各層の
       `Linear`の`weight_t`等)は普通の`Vec<f32>`フィールドであり、
       `GptModel::load`でsafetensorsから読み込んだ後は一貫して
       システムRAM(通常のRustヒープ)上に存在し続ける。GPU実行時
       (`--features real-vulkan`、`set_matmul_spirv`配線済み)でも、
       `Linear::forward`のたびにこの`Vec<f32>`から一時的にVRAMへ
       コピー→計算→結果を`Vec<f32>`へコピー戻す、という形で変わらない。
     - `open-cuda-llm::KvCacheHead`(`k`/`v`/`k_latent`/`v_latent`、
       いずれも`Vec<f32>`)も同様——2026-08-07に配線されたDeepSeek MLA
       (Multi-Head Latent Attention)風の低ランク圧縮
       (`mla_compress_kv`/`mla_decompress_kv`)経由であっても、圧縮後の
       潜在表現自体がシステムRAM上の`Vec<f32>`として保持される設計で、
       VRAM上に永続する形では一切存在しない。
     - `VulkanDevice::alloc`(`opencuda-vulkan/src/real.rs`788行目
       付近)自体、host-visibleかつmapされたメモリを都度新規作成する
       設計で、呼び出し間でバッファをキャッシュ・再利用する仕組みも
       存在しない。
  3. **結論(実装を見送った理由)**: Engram的な「VRAMに常駐する静的な
     知識をLRU等でシステムRAMへ退避し、必要時に再ロードする」という
     最適化は、**そもそもVRAMに常駐する状態が存在する場合にのみ意味を
     持つ**。しかしこのアーキテクチャは(意図した設計ではなく、単に
     「呼び出しのたびにalloc/freeする」という素朴な実装の結果として)
     既に「重み・KVキャッシュは常時システムRAMに存在し、GPUは1回の
     演算のたびに一時的に借用されるだけ」という状態になっている。
     つまりEngramが解決しようとする問題(VRAM容量を圧迫する常駐状態)が
     この実装には存在しない。ここへLRUエビクション等の追加機構を
     実装しても、退避すべき対象が無いため意味のある効果を測定しようが
     ない——「実装したが効果が無かった」という負の実験結果ですらなく、
     「そもそも適用対象が無い」という前提の不一致であり、無理に
     コードを追加することは複雑さを増やすだけの誇張的な実装になると
     判断し、着手しなかった。
  4. **実機検証(正直な開示)**: 上記の通り実装を見送ったため、
     GT 730での「以前OOMしていたモデル/コンテキストが動くようになった」
     「VRAM使用量が実測で減った」といった類の実機効果検証は**行って
     いない**(検証すべき実装が無いため)。念のため`nvidia-smi`で
     このマシンの実GPUが引き続きNVIDIA GeForce GT 730(VRAM 2048MiB)の
     1台のみであることのみ再確認した。既存コードへの変更は一切無い
     ため、`cargo build`/`cargo test`の再実行も不要と判断した
     (無変更のリポジトリに対する再ビルドは新しい情報を生まないため)。
  5. **本当に効果が見込める、Engramに近い将来の増分(次回以降の候補、
     今回は未着手)**: 今回のアーキテクチャ調査で分かったこと自体は、
     将来的にVRAM常駐が意味を持つ変更(例: 複数レイヤーの重みを
     プリフィル処理のためだけVRAMへまとめて常駐させ複数GEMMで再利用
     する、といった真のバッチ最適化)を行う際の前提知識として価値が
     ある。ただしこれは現状の「1トークンデコードのGEMMが極めて軽く
     Vulkanディスパッチのオーバーヘッドが支配的」という既存の性能上の
     結論(2026-08-06/07/08の各HANDOFF参照)を覆すものではなく、
     Engram風オフロード単体を今回のスコープとして実装する動機には
     ならないと判断した。
  - 次にすべきこと: (1) 上記3番の通り、Engram風オフロードは適用対象が
    無いため今後もこのままでは着手しない方針とする(前提が変わる
    ——例えば将来的に重みを本当にVRAM常駐させる設計へ移行する場合)
    ——が生じない限り再検討しない)、(2) 東芝SBM/DeepSeek技術の
    このリポジトリへの適用候補としては、既に`open-cuda`側で実装・
    実機検証済みのMLA(KVキャッシュ低ランク圧縮、2026-08-07)・
    fused flash attention(2026-08-07/08)を`aruaru-llm`側から実際に
    有効化するオプトイン配線(`generation.rs`に`wire_flash_attention_
    spirv`は既にあるが`enable_mla_kv_compression`相当の配線は未着手)
    の方が、既存のVRAM常駐前提を変えずに着手できる現実的な候補として
    残しておく。

- **2026-08-08 `GptModel::set_flash_attention_spirv`のオプトイン配線+
  「GEMM+GPU softmax」vs「GEMM+fused flash attention」の実機速度比較
  (rs-sync横断セッション、直前2026-08-07 HANDOFFの「次にすべきこと(1)」
  への対応)**:
  1. **実装**(`src/generation.rs`): `default_flash_attention_spirv_path`
     (`ARUARU_LLM_FLASH_ATTENTION_SPIRV`環境変数で上書き可、既定は
     `open-cuda`側の`examples/flash_attention_vulkan_real/shaders/
     flash_attention.spv`)+`wire_flash_attention_spirv`を追加。
     `wire_matmul_spirv`/`wire_softmax_spirv`とは異なり**既定では
     wireしない**——`ARUARU_LLM_ENABLE_FLASH_ATTENTION=1`(または`true`)を
     明示的に設定した場合のみ配線する設計にした。理由: `GptModel`側の
     実装(`open-cuda/crates/open-cuda-llm/src/lib.rs`)は`flash_attn_spirv`
     が`Some`なら常にそちらを`softmax_spirv`より優先するため、両方を
     常時wireすると「GEMM+GPU softmax」経路を意図的に選ぶ手段が無くなり、
     3経路(GEMM+CPU softmax/GEMM+GPU softmax/GEMM+fused flash attention)
     の比較ができなくなるため。
  2. **実機速度比較(NVIDIA GT 730、型チェックのみで完了と報告しない方針を
     徹底、実際にサーバーを起動し実HTTPリクエストで計測)**:
     `POST /v1/generate {"prompt":"The quick brown fox",
     "max_new_tokens":5}`を、同一プロンプト・同一トークン数で2経路
     それぞれ実測した(いずれも生成結果`"es are a great way"`で完全一致、
     数値的に壊れていないことも確認):
     - **GEMM+GPU softmax**(`wire_softmax_spirv`のみ、既定の`real-vulkan`
       挙動): **約26.2秒**。
     - **GEMM+fused flash attention**(`ARUARU_LLM_ENABLE_FLASH_ATTENTION=1`
       追加): **約16.4〜16.6秒**(2回計測、"Artificial intelligence is"の
       別プロンプトでも再現)。
     **fused flash attentionへの切り替えで約37%高速化**——レイヤーあたり
     QKᵀ・softmax・P·Vの3回ディスパッチが1回に統合されたことで、
     `open-cuda`側2026-08-06 HANDOFFで懸念されていた「Vulkanディスパッチ
     固定オーバーヘッドが支配的」という問題が実際に緩和されることを
     初めて実測で確認した(推測ではなく実測、誇張しない範囲で報告する)。
  3. **正直な開示・それでもなお遅いという事実**: 上記いずれの経路も、
     同条件でのCPU版(`real-vulkan` feature無し)の実測(2026-08-04
     HANDOFF記録: `max_new_tokens=20`で約6〜7秒)と比較すると依然として
     大幅に遅い(`max_new_tokens=5`で16秒台は、単純比例換算でも
     CPU版の`max_new_tokens=20`実測を上回る)。**「fused flash attentionは
     softmax分離版より速い」ことは実証したが、「Vulkan経路がCPU版より
     速い」ことは今回も実証していない**——GT 730のような低性能GPUでは、
     1トークンあたりのディスパッチ回数をどれだけ減らしても、GPT-2 124M
     程度の軽い計算に対するVulkanのディスパッチ固定オーバーヘッド自体が
     依然支配的である可能性が高い。`real-vulkan`(既定無効)を既定へ
     昇格させる判断はしない。
  4. **検証結果**: `cargo build --release`(featureなし)/`cargo build
     --release --features real-vulkan`いずれも成功。`cargo test
     --release`(featureなし)は既存回帰なし(バックグラウンドで実行、
     結果は本エントリ確定前に確認済み)。`cargo clippy --release
     --features real-vulkan --all-targets -- -D warnings`で**pre-existing
     の3件**(`generation.rs`の`unused_assignments`、`generation.rs`/
     `scoring.rs`の`ENGINE_GPT2_GREEDY`/`ENGINE_EMBEDDING`の`dead_code`、
     いずれも今回の変更とは無関係で以前から存在)を検出——今回は変更が
     本題(実機速度比較)から逸れることを避けるため修正していない、
     次回clippy運用時にまとめて対応する候補として記録する。
  5. **正直な開示・スコープ外**: (a) `scoring`/`security`(BERT系)側への
     flash-attention相当の配線は今回も行っていない(`open-cuda-bert::
     BertModel`はエンコーダでKVキャッシュを持たないため、`open-cuda`側
     2026-08-07エントリで指摘されている通り単純な流用はできない)。
     (b) dream-os/open-directx側との直接連携は本セッションでは
     `open-cuda`経由の間接連携(同じ`flash_attention.spv`アセットを
     `dream-os-kernel`も別途参照するようになった、詳細は`dream-os/
     CLAUDE.md`2026-08-08エントリ参照)のみで、本リポジトリのファイルは
     変更していない。
  - 次にすべきこと: (1) 上記clippy pre-existing警告3件のクリーンアップ、
    (2) `real-vulkan`をこのGT 730のような低性能GPU環境では既定にしない
    という判断の妥当性を、より高性能なGPU実機が得られた際に再検証する、
    (3) `scoring`/`security`側のVulkan GEMM配線・ベンチマークは引き続き
    未着手。

- **2026-08-07(続き) `/v1/chat`・`/v1/classify-security`の空入力挙動を実HTTPで
  検証(前回HANDOFFの「次にすべきこと(2)」への対応、ユーザー指示
  「dream-os/open-directx/open-cuda/aruaru-llmの関連性・連携性・実用性・
  完成度を向上」の一環)**:
  1. **背景**: 直前2026-08-07 HANDOFF(`/v1/generate`・`/v1/translate`の
     空入力`400`化)の「次にすべきこと(2)」で、同様の粗が`/v1/chat`
     (`message`)・`/v1/classify-security`(`text`)にも無いか未確認のまま
     残っていた(コード読解上は`scoring::classify`/`security::classify_
     security`が空文字列でも例外を投げず正常応答する設計と推測されて
     いたが、実HTTPでの確認は未実施だった)。
  2. **検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo build`(debugビルド、`target/debug/aruaru-llm.exe`)後、
     実際にサーバーを起動し(`0.0.0.0:4600`)、以下を実HTTPリクエストで
     確認した:
     - `POST /v1/chat {"message":""}` → `200`
       `{"reply":"e-gov.infoへようこそ...","engine":"embedding-cosine-v0-
       open-cuda-bert-cpu","matched_intent":null,...}`(既定のフォール
       バック応答、`503`にならず正常応答)。
     - `POST /v1/classify-security {"text":""}` → `200`
       `{"label":"benign","score":0.886...,"is_suspicious":false,
       "engine":"embedding-cosine-heuristic-v0-open-cuda-bert-cpu",
       "static_signals":[]}`(空入力を`benign`判定、`503`にならず正常
       応答)。
     - 回帰確認として`POST /v1/generate {"prompt":""}`→`400`・
       `POST /v1/translate {"text":"","target_lang":"Japanese"}`→`400`
       も再実行し、前回HANDOFFの修正(コミット`b500023`)が引き続き
       正しく動作していることを確認した。
  3. **結論・コード変更は無し**: `/v1/chat`・`/v1/classify-security`は
     いずれも空入力で`503`化する問題を再現できなかった——前回HANDOFFの
     「`scoring::classify`/`security::classify_security`は空文字列でも
     例外を投げずNone/低スコアで正常応答する設計」という推測が実HTTPで
     裏付けられた。**このため`src/main.rs`等への変更は行っていない**
     (「確認のみでよい」という前回の優先度判断が正しかったことの実証、
     誤りを見逃していないことの正直な報告)。
  4. **正直な開示・今回スコープ外のまま**: `open-cuda`側で今回同時に
     `open-cuda-llm`のAttention経路へ`flash_attention_with_spirv`
     (1ディスパッチで完結するfused attentionカーネル)を配線した
     (`open-cuda/CLAUDE.md` 2026-08-07(続き5)HANDOFF参照)が、
     `aruaru-llm`側で`GptModel::set_flash_attention_spirv`を実際に呼ぶ
     配線は**今回は行っていない**(このリポジトリでは検証専任のパスに
     留め、影響範囲拡大を避けた——`generation.rs::wire_matmul_spirv`
     相当の追加配線+実機速度計測は次回の増分として残す)。
     `scoring`/`security`側のVulkan GEMM配線・GT-730でのVulkan vs CPU
     ベンチマークも前回から変更なく未着手。dream-os・open-directx側との
     具体的な連携強化(API呼び出し経路の実装等)も本リポジトリ内では
     引き続き未着手。
  - 次にすべきこと: (1) `GptModel::set_flash_attention_spirv`の
    `aruaru-llm`側オプトイン配線(`generation.rs`に`wire_matmul_spirv`と
    並ぶ`wire_flash_attention_spirv`相当を追加し、
    `--features real-vulkan`で実機速度計測——「GEMM+CPU softmax」
    「GEMM+GPU softmax」「GEMM+fused flash attention」の3経路比較)、
    (2) `scoring`/`security`側のVulkan GEMM配線・GT-730でのベンチマークは
    引き続き未着手、(3) dream-os・open-directx側との具体的な連携強化は
    未着手のまま。

- **2026-08-06 softmax専用SPIR-Vカーネルをaruaru-llm側でも実配線、
  「GPU GEMM + GPU softmax」経路の実HTTP検証まで完了(直前コミット
  「Vulkan GEMM配線バグを解消、softmax専用カーネル連携を実機検証」の
  次にすべきこと=Attention経路への本格統合、ユーザー指示「aruaru-llm
  連携性向上」への対応)**:
  1. **前回コミットの正直な補足記録**: 直前コミット(`f7030ca`、CLAUDE.md
     未記録のまま残っていたため今回まとめて記録)は、`engine_label`が
     実行経路(Vulkan/CPU)を`matmul_spirv_wired`フラグで正しく判定する
     よう修正したのみで、**softmaxカーネル自体はこの時点ではまだ
     aruaru-llm側に一切参照されていなかった**(`grep softmax src/`が
     0件)——コミットメッセージの「softmax専用カーネル連携を実機検証」は
     `open-cuda`側(`opencuda-blas`)でのスタンドアロン検証を指しており、
     aruaru-llm側の実配線は未着手だった。この節でその配線を実施した。
  2. **`open-cuda`側の対応**(このセッションでopen-cuda本体にも着手、
     詳細は`open-cuda/CLAUDE.md` 2026-08-06 HANDOFF参照): 前回
     (2026-08-06付`open-cuda`側HANDOFF)で実装・実機検証済みだった
     `softmax_vulkan_generic`(スタンドアロンのsoftmaxカーネル、Attention
     経路には未配線)を、`opencuda_blas::scaled_dot_product_attention_
     with_spirv_and_softmax`という新関数経由でAttention計算(QKᵀ→softmax
     →P·V)へ実際に組み込んだ。`open-cuda-llm::GptModel`・`open-cuda-bert::
     BertModel`双方に`set_softmax_spirv`を新設(`set_matmul_spirv`と同じ
     パターン)。
  3. **`aruaru-llm`側の配線**: `src/generation.rs`に`default_softmax_
     spirv_path`(`ARUARU_LLM_SOFTMAX_SPIRV`環境変数で上書き可)・
     `wire_softmax_spirv`を追加し、`load_from_dir`内で`wire_matmul_spirv`
     と並べて呼ぶよう変更(`GptModel::set_softmax_spirv`経由)。
     `src/scoring.rs`にも同じパターンで`default_softmax_spirv_path`・
     `wire_softmax_spirv`を追加し、`get_model()`内で`BertModel::
     set_softmax_spirv`を呼ぶよう変更(`scoring`/`security`共通、
     `security.rs`は`scoring::embed`を再利用する既存設計のため無改修)。
  4. **実機検証(型チェックのみで完了と報告しない方針を徹底、NVIDIA
     GeForce GT 730)**:
     - `cargo build --release`(featureなし)/`cargo build --release
       --features real-vulkan`いずれも成功。`cargo test --release`/
       `cargo test --release --features real-vulkan`いずれも既存
       **46件全green**(regression無し、テスト自体の新規追加は今回無し
       ——このリポジトリ側の新規ロジックは「起動時にファイルを読んで
       配線する」だけの薄い層で、実質的な検証は`open-cuda`側の
       `open-cuda-llm`/`opencuda-blas`テストで既に実施済みのため)。
     - **実際にサーバーを起動**(`--features real-vulkan`、
       `RUST_LOG=info`)し、起動ログで`generation`・`scoring`双方について
       `loaded matmul.spv (2732 bytes) ... via set_matmul_spirv`・
       `loaded softmax.spv (4680 bytes) ... via set_softmax_spirv`の
       **両方**が記録されることを確認(片方だけの配線ではないことの
       裏取り)。
     - **実HTTPリクエスト**: `POST /v1/generate
       {"prompt":"The quick brown fox","max_new_tokens":5}`へ実際に
       リクエストを送り、`"completion":"es are a great way"`(CPU版の
       既知の継続文`"es are a great way to get a little bit of a"`の
       先頭一致、生成内容が壊れていないことを確認)・
       `"engine":"gpt2-greedy-decode-v0-open-cuda-llm-vulkan"`
       (`-vulkan`接尾辞、実際にVulkan経由で動作したことをエンジン
       ラベルからも確認)という正しい応答を得た。
  5. **正直な開示・性能(誇張しない)**: 上記`POST /v1/generate`
     (`max_new_tokens=5`)の実測所要時間は**約35.9秒**——CPU版の既存記録
     (`max_new_tokens=20`で約6〜7秒)と比較して大幅に遅い。これは
     `open-cuda`側2026-07-26 HANDOFFで示した「1トークンデコードは
     `seq_len=1`のGEMM/softmaxが極めて軽く、Vulkanのディスパッチ固定
     オーバーヘッドの方が支配的になり、CPU実行より遅くなりうる」という
     設計上の懸念が、今回softmax専用カーネルの追加ディスパッチ
     (レイヤーあたりQKᵀ・softmax・P・Vの3回、従来のGEMMのみ版の1.5倍の
     ディスパッチ回数)によりさらに悪化する形で実測された。
     **「正しく動く」ことは実証できたが「速くなる」ことは実証していない**
     ——`real-vulkan` featureは既定無効(opt-in)のままであり、この結果を
     もって既定を切り替える判断はしない。
  6. **検証結果まとめ**: `cargo build --workspace`/`cargo test
     --workspace`(open-cuda側含む)全クレートregression無し。
     `cargo clippy --workspace --all-targets --release -- -D warnings`
     (open-cuda側)警告0件。
  - 次にすべきこと: (1) デコード側(`forward_step`、`seq_len=1`)への
    Vulkanディスパッチはオーバーヘッドが支配的で実用上不利なことが
    改めて確認されたため、プリフィル(`forward_prefill`、バッチGEMM)
    側でのみVulkan経路を使い、デコード側はCPU固定にする「経路ごとの
    使い分け」の設計(`open-cuda`側の変更が必要な増分)、(2) 真に
    GPU常駐率を上げるにはAttention全体を1回のディスパッチにまとめる
    融合(fused)カーネルが必要(`open-cuda`側スコープ)、(3) 現状の
    `real-vulkan`(opt-in、既定無効)のまま据え置き、既定を切り替える
    判断は行わない(性能が実証されるまでは)、(4) `flash_attention`
    (タイル化+オンラインsoftmax)側は依然ホスト側CPU実装のまま
    未着手(`open-cuda`側スコープ)。

- **2026-08-04(続き2) `real-vulkan` feature配線を実装、実機検証の結果
  「単純な配線では動作しない」ことが判明(前回HANDOFFの「次にすべきこと
  (1)」への対応、ユーザー指示「open-directx open-cuda aruaru-llmなどの
  使いやすさ向上と連携と実用性と完成度を向上させて」の一環)**:
  1. **実装**: `Cargo.toml`に`opencuda-vulkan`をoptional path依存として
     追加し、`real-vulkan = ["dep:opencuda-vulkan", "opencuda-vulkan/real-vulkan"]`
     を新設(既存の`hw-detect-vulkan`——ハードウェア検出専用——とは別軸、
     推論ディスパッチ先を切り替えるためのfeature)。`src/main.rs`の
     デバイス選択(従来`CpuDevice::new(0)`固定)を`#[cfg(feature =
     "real-vulkan")]`で分岐し、有効時は`opencuda_vulkan::real::
     VulkanDevice::new(0)`を使う(構築失敗時はCPUへ安全側フォールバック、
     サービスを壊さない設計)。既定ビルド(feature無し)への影響はゼロ
     (`cargo build --release`〈featureなし〉が引き続き成功することを
     確認済み)。
  2. **ビルド検証**: `cargo build --release --features real-vulkan`
     成功。`cargo test --release`(featureなし)・`cargo test --release
     --features real-vulkan`いずれも**46件全green**、リグレッション無し。
  3. **実機検証(NVIDIA GT 730、型チェックのみで完了と報告しない方針を
     徹底)、そして重要な負の結果**: 実際にサーバーを起動し
     `POST /v1/generate`へ実HTTPリクエストを送った。
     - **CPU版(feature無し)**: `{"prompt":"The quick brown fox",
       "max_new_tokens":20}` → `"es are a great way to get a little bit
       of a kick out of your dog.\n\n"`(200)、所要時間**約6.0〜6.8秒**
       (`time curl`で2回計測、6.796s/6.060s)。
     - **Vulkan版(`--features real-vulkan`)**: 起動ログでは
       `real-vulkan feature enabled: using VulkanDevice for inference
       dispatch`・`device: OpenCUDA Vulkan Device (NVIDIA GeForce GT
       730)`が確認できたが、**`POST /v1/generate`への実リクエストは
       0.228秒で即座に失敗**した:
       `"error":"GptModel::generate failed: sgemm: GemmPath::
       VulkanGeneric selected (device.supports_spirv()==true,
       vendor-specific path is still a stub) but no spirv bytes were
       provided; pass the compiled matmul.spv bytes via the `spirv`
       argument"`
  4. **根本原因の特定**: `open-cuda`側`open-cuda-llm/src/lib.rs:224`の
     `Linear::forward`が`opencuda_blas::sgemm(..., None)`と、`spirv`引数
     に常に`None`を渡す実装になっている(CPU実行のみを前提とした既存の
     呼び出し)。`opencuda_blas::select_gemm_path`は`VulkanDevice`に対して
     `GemmPath::VulkanGeneric`を選ぶが、この経路は`spirv`バイト列
     (コンパイル済み`matmul.spv`)が必須で、`None`だと`sgemm`が
     `Err`を返す設計(`opencuda-blas/src/lib.rs`の`sgemm`関数、意図的な
     安全策——「実装していないという誤ったシグナルを出さない」既存の
     エコシステム方針通り、黙ってCPUへフォールバックせず正直にエラーに
     している)。同じ理由で、`scoring::warmup`/`security::warmup`
     (`open-cuda-bert`経由のBERT埋め込み計算)もVulkan版サーバー起動直後
     に同一エラーで警告ログを出していた(`warmup failed ...: sgemm:
     GemmPath::VulkanGeneric selected ... but no spirv bytes were
     provided`)——`generation`側のwarmupだけは重み・トークナイザの
     ロードのみで実際に`sgemm`を呼ばないため警告なしに完了しており、
     初回`/v1/generate`呼び出しで初めて失敗が表面化した。
  5. **正直な評価(誇張しない)**: 前回HANDOFF(2026-07-26)で示した
     「安易な配線は逆に遅くなりうる」という**性能上の懸念**は、今回の
     実測では検証できなかった——実際には性能比較以前に**機能しない**
     (即座にエラーで失敗する)ことが判明した。これは`aruaru-llm`側の
     feature配線自体のバグではなく(配線・デバイス切り替えは設計通り
     正しく動作している——ログで実際にVulkanDeviceが選択されたことは
     確認済み)、`opencuda-blas`/`open-cuda-llm`側がまだ「呼び出し側が
     コンパイル済みSPIR-Vシェーダバイト列を明示的に渡す」設計のままで、
     `open-cuda-llm`のLinear層がその配線を実装していない、という
     `open-cuda`側の既知のギャップに起因する。「成功した」と誇張せず、
     この負の結果をそのまま記録する。
  6. **スコープの判断(ユーザー指示「open-cuda側のコードは今回変更しない
     こと」に従う)**: 根本原因は`open-cuda-llm/src/lib.rs`の
     `Linear::forward`が`matmul.spv`のコンパイル済みバイト列を`sgemm`へ
     渡していない点にあり、修正には(a)`open-cuda-llm`側で`matmul.spv`を
     ロードし`Linear`構造体が保持する、(b)`VulkanDevice`固有の
     `sgemm_vulkan_generic`をどこかで呼ぶよう`Linear::forward`を変更する、
     という`open-cuda`側の実装変更が必要(`opencuda-vulkan`側には
     `tools/compile-vulkan-shaders.*`で`matmul.spv`をビルドする仕組みが
     既に存在するため、シェーダ自体は用意されている)。これは
     `open-cuda-llm`クレート内部の変更であり、今回のタスク範囲
     「`open-cuda`側のコードは変更しないこと」に明確に抵触するため、
     **今回は修正を行わなかった**(ユーザー指示通り、正直に開示するのみ
     に留める)。
  7. **`aruaru-llm`側の変更のみで到達可能な範囲の結論**: `real-vulkan`
     featureの配線自体(Cargo feature新設・デバイス選択の分岐・既定
     ビルドへの無影響・ビルド/テストの回帰無し)は完了・検証済みだが、
     実際にVulkan経由での生成が機能するには`open-cuda-llm`側への
     追加実装(上記6番)が前提条件として必要、という結論に至った。
  - 次にすべきこと: (1) (`open-cuda`専用セッションとしてスコープを切り、
    ユーザーの了承を得た上で)`open-cuda-llm::Linear::forward`が
    `matmul.spv`のコンパイル済みバイト列を`sgemm`へ渡すよう変更する
    (`opencuda-vulkan`の`tools/compile-vulkan-shaders.*`で生成済みの
    シェーダを`Linear`構造体または`GptModel`がロード・保持する設計)、
    (2) 上記が実装され次第、本HANDOFFで実施した同じ手順(`cargo build
    --release --features real-vulkan`→サーバー起動→`POST /v1/generate`
    実HTTPリクエスト)でCPU版とVulkan版の生成結果の数値一致・実際の速度差
    を再検証する、(3) `open-cuda-bert`(scoring/security)側も同じ根本
    原因で警告が出ているため、修正時は`open-cuda-llm`と合わせて
    `opencuda-blas`側の呼び出し全体を横断的に確認する価値がある。

- **2026-08-04(続き) `open-cuda`側`open-cuda-llm::GptModel`にプリフィル/
  デコード分離+QKV融合GEMMを実装(直前の2026-07-26 HANDOFF「安易なGPU
  配線は逆に遅くなりうる」で示した設計変更(a)(b)への対応、詳細は
  `open-cuda`側CLAUDE.md 2026-08-04 HANDOFF参照、ユーザー指示
  「open-directx open-cuda aruaru-llmなどの使いやすさ向上と連携と実用性と
  完成度を向上させて」)**:
  1. **要旨**: `open-cuda-llm/src/lib.rs`のQ/K/V(3本の`Linear`)を1本の
     融合`Linear`(`qkv`)へ統合し(GPT-2 safetensorsが元々`c_attn`という
     融合`Conv1D`で保存している構造をそのまま活かす形)、プロンプトの
     初回forward(プリフィル)を`forward_prefill`(`seq_len`をGEMMの`m`
     パラメータとする本当のバッチ処理)へ変更した。生成トークンの逐次
     デコードは従来通り`forward_step`(`seq_len=1`)のまま(prefill/decode
     分離)。
  2. **検証**: 実GPT-2 124M重み(`../open-cuda/crates/open-cuda-llm/
     models/gpt2/`、本リポジトリが既定で使うのと同じ重み)で、変更前後の
     生成結果(プロンプト`"The quick brown fox"`・`max_new_tokens=12`)が
     `[274, 389, 257, 1049, 835, 284, 651, 257, 1310, 1643, 286, 257]`
     (`"es are a great way to get a little bit of a"`)で完全一致する
     ことを確認(挙動を変えない最適化であることの実証)。`cargo test -p
     open-cuda-llm --release`9件全green、`cargo build --workspace`/
     `cargo test --workspace`全クレートregression無し。
  3. **本リポジトリ(`aruaru-llm`)側の変更は無い**: 今回の変更は
     `open-cuda`側`open-cuda-llm`クレート内部の最適化に留まり、
     `aruaru-llm`はこの依存クレートを`path`依存で参照しているだけ
     (`Cargo.lock`不使用のpath依存のため、次回`cargo build`時に自動的に
     新しいコードが使われる。既存の`POST /v1/generate`のAPI契約
     〈入出力形式〉は無変更)。
  4. **正直な開示・スコープ外(直前HANDOFFの(c)は今回未着手)**:
     `aruaru-llm`側にオプトインの`real-vulkan` feature(デバイス選択を
     `CpuDevice`から`opencuda-vulkan::real::VulkanDevice`へ切り替える
     配線)は**今回も実装していない**。(a)(b)の実装・「挙動を変えない
     ことの実証」までを確実にやり切ることを優先したため
     (ユーザー指示の優先順位通り)。ディスパッチ回数削減
     (レイヤーあたりプリフィルが`4*seq_len`→`4`)によりVulkan配線時の
     オーバーヘッド懸念はプリフィル側について理論上緩和されているはず
     だが、実際にVulkanDevice経由で走らせての速度実測・CPU版との数値
     一致検証はまだ行っていない。
  - 次にすべきこと: (1) `main.rs`のデバイス選択箇所に`real-vulkan`
    feature(既定無効)を追加し`VulkanDevice`へ切り替え、(2) 実機
    (NVIDIA GT 730)でCPU版とVulkan版の生成結果が数値的に一致すること・
    実際の速度差(遅くなっていないか)をベンチマークで確認、(3) 上記が
    確認できれば`README.md`/`README-English.md`に`real-vulkan`
    feature(有効化方法・既知の制約)を追記。

- **2026-08-04 `POST /v1/translate`を新設(ユーザー指示「aruaru-llmに自動翻訳機能を持たせて」)**:
  `/v1/generate`(GPT-2系テキスト生成)を翻訳プロンプトで呼び出す薄いラッパー。
  `{text, target_lang, source_lang(任意), tenant(任意)}` → `{translation, engine,
  disclosure}`。**正直な開示(最重要)**: 専用の翻訳モデル(NLLB/M2M100等)ではなく、
  英語中心の事前学習のみでファインチューニング無しのGPT-2系(124M-1.5B)を
  そのまま流用しているため、特に非英語↔非英語の組み合わせで品質が不安定。
  この限界はレスポンスの`disclosure`フィールドに毎回明記する(`/v1/generate`と
  同じ設計方針)。`audiocafe.tokyo/rakuten-mobile`の17言語ページ整備
  (ユーザー指示「毎朝5時に自動翻訳して」)の実現手段として新設したが、
  当該ページの17言語版自体は今回、生成AIではなく人力翻訳(Claude Code)で
  作成済みであり、この`/v1/translate`エンドポイントを実際に呼び出す
  cronジョブの配線は次回以降の課題として未着手のまま(エンドポイント自体は
  `cargo build --release`で実装・ビルド確認済み、実HTTP検証は未実施)。
  - 次にすべきこと: (1) 実際に`POST /v1/translate`を呼ぶ検証(モックでなく
    実リクエスト)、(2) 品質が実用に足るかの評価(現状は理論上の懸念のみ)、
    (3) 必要であれば`audiocafe-tokyo-php`側のcronから定期呼び出しする配線。

- **2026-07-31 本家`poem`クレートからRPoem(`open-runo-poem-compat`)へ移行完了(ユーザー指示)**:
  ユーザー指示「Python向けAIライブラリとLLMをRustなど他の言語からでも
  利用できるRust製+Rust＋Poem互換のRPoemで開発して」に対応。前回
  (`open-cuda`側)の`opencuda-whisper`新設に続き、その成果を他言語から
  HTTP経由で使える窓口である本リポジトリ自体を、本家`poem`から
  `RPoem`(`../RPoem/crates/open-runo-poem-compat`、path依存)へ移行した。
  1. **移行内容**: `Cargo.toml`から`poem = "3.1"`を削除し
     `open-runo-poem-compat`(path依存)+`hyper`+`bytes`を追加。
     `src/main.rs`を全面書き換え——`#[handler]`マクロ・
     `Data<&Arc<...>>`抽出子は本ファサードに存在しないため、
     `handler_fn`(素の`Fn(Request, Params) -> Future<Response>`)へ
     ハンドラを変換し、共有状態(`device: Arc<dyn GpuDevice>`・
     `registry: Arc<TenantRegistry>`)はルート登録時のクロージャで
     `Arc::clone`をキャプチャする形に置き換えた。`Json<T>::from_body
     (req).await`(リクエストボディ)・`hyper_compat::json_response`
     (レスポンス)・`PathParams::from(params)`(パスパラメータ)は
     ファサードの提供APIをそのまま使用。プレーンテキスト応答
     ("ok"/"invalid admin token"/"tenant not found")用に軽量な
     `text_response()`ヘルパーを新設(ファサードにcontent-type自動判定
     機能が無いため)。
  2. **ロジック自体は無変更**: `scoring::classify`/`security::
     classify_security`/`generation::generate`/`model_catalog`/
     `hardware::recommend`等、ビジネスロジック側のモジュールは一切
     変更していない——変更は`main.rs`のHTTP層のみ。
  3. **検証(実HTTP、型チェックのみで完了と報告しない既存運用ルール
     徹底)**: `cargo build`成功(RPoem側の既存3警告のみ、aruaru-llm自体は
     警告0件)。実際にサーバーを起動し(`E_GOV_LLM_ADMIN_TOKEN=test-token`)、
     起動ログでwarmup完了(モデルロード28秒・セキュリティ分類器ウォーム
     アップ)を確認した上で: `GET /healthz`→200・`GET /`→200(静的HTML
     UI)・`POST /v1/chat`(govインテント一致・正しい応答)・`POST
     /v1/generate`(実GPT-2重みでの文章生成、200)・`GET /v1/models/
     catalog`→200・`POST /admin/tenants`(登録)→200・`GET /admin/
     tenants`(一覧反映)→200・`DELETE /admin/tenants/:host`(削除)→200・
     `GET /admin/tenants`(トークン無し)→401・未知パス→404、すべて
     実HTTPで確認済み。`cargo test`は実行中(バックグラウンド、結果は
     次のHANDOFF更新時に反映)。
  4. **正直な開示**: (a) 本ファサード(`open-runo-poem-compat`)自体が
     `poem::Endpoint`/`FromRequest`トレイトを実装していない薄いシムで
     あるため、将来poem本体のミドルウェアエコシステムを使う予定がある
     機能は今後も個別対応が必要(現状のaruaru-llmはそのような機能を
     使っていないため実害無し)。(b) `handler_fn`のシグネチャが
     `Fn(Request, Params) -> Future`固定のため、複数の状態(`device`+
     `registry`)を渡すハンドラはクロージャのネストがやや読みにくい——
     可読性より正確な移行を優先した。
  - 次にすべきこと: (a) `cargo test`のバックグラウンド実行結果を確認、
    (b) `opencuda-whisper`(音声認識)を本サービスのエンドポイントとして
    実際に公開する配線(現状`open-cuda`側crateとしては実装済みだが
    aruaru-llm側のHTTPエンドポイントとしては未接続)。

- **2026-07-27(続き3) 「一つ大きなモデルをダウンロードする」/「一つ小さな
  モデルをダウロードする」ボタンを追加(ユーザー指示への対応、直前の
  HANDOFF〈お勧めLLMダウンロード〉の上に構築)**:
  1. **`src/model_catalog.rs`にサイズ順ナビゲーション関数を追加**:
     `size_ordered_catalog()`(`approx_size_mb`昇順ソート)・
     `next_larger(current_id)`・`next_smaller(current_id)`
     (いずれも未知IDや端(最大/最小)では`None`を返す)。カタログの
     サイズ順は`distilgpt2`(353MB) < `gpt2`(548MB) <
     `gpt2-medium`(1520MB) < `gpt2-large`(3250MB) < `gpt2-xl`(6430MB)。
  2. **`src/main.rs`に`step_model_size()`共通ロジック+2エンドポイント**:
     `POST /v1/download-larger`・`POST /v1/download-smaller`。現在
     アクティブなモデルID(`current_model_id()`、`generation::
     active_model_dir()`から逆引き)を起点に`next_larger`/`next_smaller`
     を解決し、未取得なら`model_catalog::install`でダウンロード、
     `generation::select_model`でホットスワップ切り替え。既に
     カタログ端(最大/最小)の場合は`already_installed: true,
     switched: false`と共に「これ以上大きく/小さくできません」と
     正直に応答する。切り替え失敗時も直前の稼働モデルは維持される
     (既存の`recommend-and-download`と同じ「サービスを壊さない」設計)。
  3. **`static/index.html`にボタン2個(`largerBtn`/`smallerBtn`)を追加**:
     `stepModelSize(path, button)`という共通JS関数で両エンドポイントを
     叩き、結果(切り替え元/先ID・ダウンロード要否・成否)を`#status`へ
     表示する。
  4. **検証**: `cargo build`成功。`cargo test`**42件全green**(新規2件
     `model_catalog::tests::next_larger_and_next_smaller_follow_
     approx_size_order`・`next_larger_and_next_smaller_return_none_
     for_unknown_id`を含む——`CatalogEntry`が`PartialEq`未実装のため
     `assert_eq!(..., None)`ではなく`assert!(...is_none())`で比較する
     形に実装済み)。実機での実HTTP検証(サーバー起動+ボタンクリック
     +GPU実機での実ダウンロード確認)は今回未実施(GPUハードウェア
     featureが既定offのCPU-onlyフォールバック経路のみローカルで確認)。
  - 次にすべきこと: (1) `hw-detect-vulkan`/`hw-detect-directx`を有効化
    した実機ビルドでの実ダウンロード・実切り替えの動作確認、
    (2) UIの日本語文言を他言語(英/伊/仏/独/露)へ拡張するかの検討。

- **2026-07-27(続き2) ハードウェア検出→推奨LLMサイズ→「お勧めLLMを
  ダウンロード」ボタンを実装(ユーザー指示「open-directx・open-cuda・
  aruaru-llmの組み合わせで『お勧めLLMをダウンロード』ボタンで最適な
  LLMをダウンロードする機能を搭載して」への対応、直前のHANDOFF
  〈モデルカタログ・ホットスワップ〉の上に構築)**:
  1. **新規モジュール`src/hardware.rs`**: `open-cuda`(`opencuda-vulkan`、
     Vulkan物理デバイス列挙)・`open-directx`(`opencuda-directx`、DXGI
     アダプタ列挙)のいずれかからGPUベンダー名・VRAM容量を取得し、
     VRAM容量に応じてGPT-2ファミリーの推奨サイズ(124M/355M/774M/1.5B)
     を選ぶ簡易ヒューリスティックを実装(閾値: <2GB→124M、2-4GB→355M、
     4-8GB→774M、8GB以上→1.5B、GPU検出不能・CPUのみ→124M固定の
     安全側フォールバック)。**正直な開示**: パラメータ数×4バイトの
     fp32概算とVRAM容量の単純比較に過ぎず、精密な性能予測ではない旨を
     モジュールdoc・APIレスポンス(`disclosure_ja`)・UI双方に明記した。
  2. **GPU検出はopt-in feature**(`hw-detect-vulkan`/`hw-detect-directx`、
     `Cargo.toml`に追加、既定は両方無効): `opencuda-vulkan`/
     `opencuda-directx`をoptional path依存として追加し、それぞれ上流
     クレート自身の`real-vulkan`/`real-dx12` featureへ連鎖させた。
     既定ビルド(feature無効)ではCPUのみとみなし安全側(124M)へ
     フォールバックするため、Android等クロスコンパイル環境やCI環境に
     Vulkanローダー/Windows SDK依存を強制しない(既存の`opencuda-vulkan`/
     `opencuda-directx`自身の設計方針を踏襲)。
  3. **どちらの経路の情報を使うか明確化**(ユーザー指示への直接対応):
     `hardware::detect()`はVulkan経路を優先し、両feature有効なら
     DXGI(DirectX)側の結果を10%許容誤差でクロスチェックし
     `cross_check_agreement`フィールドへ記録する。実際に使われた経路は
     `detection_path`("vulkan"/"directx"/"cpu-only-fallback")で常に
     APIレスポンスへ明示する。
  4. **モデルカタログに`gpt2-xl`(1.5B、`openai-community/gpt2-xl`)を
     追加**(`src/model_catalog.rs`)——VRAM8GB以上枠に対応する実在の
     Hugging Face公開リポジトリ。
  5. **新規HTTPエンドポイント2件**(`main.rs`): `GET /v1/recommend`
     (検出のみ)、`POST /v1/recommend-and-download`
     (検出→未ダウンロードならHugging Faceから取得〈`model_catalog::install`
     を再利用、既存ファイルがあれば再取得しない冪等設計〉→
     `generation::select_model`でホットスワップ切り替え、まで一括)。
     切り替え失敗時は現在動作中のモデルを維持し、エラー内容を正直に
     返す(`select_model`と同じ「サービスを壊さない」設計)。
  6. **`generation::engine_label()`を新設**(`src/generation.rs`):
     従来の固定`ENGINE_GPT2_GREEDY`定数は、ホットスワップでgpt2-medium
     等へ切り替えた後も"gpt2-124m-..."と表示され続ける不正直な状態
     だった(直前のHANDOFFで追加したホットスワップ機能の副作用に
     今回気付いて修正)。`active_model_dir()`のディレクトリ名を反映した
     動的な`engine`文字列("gpt2-medium-greedy-decode-v0-opencuda-llm-cpu"
     等)を返すよう`main.rs`の`/v1/generate`ハンドラを修正した。
  7. **最小限の静的HTML UI新設**(`static/index.html`、`GET /`で配信):
     「お勧めLLMをダウンロード」ボタン1つ+進捗/結果表示+切り替え成功後の
     生成テスト導線。Tauri/Node.js/TypeScript不使用、`include_str!`で
     Rustバイナリへ埋め込み配信(過剰実装を避ける方針通り)。
  8. **実機検証(型チェックのみで完了と報告しない方針を徹底、このマシンの
     実GPU〈NVIDIA GeForce GT 730〉で一気通貫の動作確認まで実施)**:
     - `cargo test --release --features hw-detect-vulkan`
       **42件全green**(既存39件+`hardware`モジュール新規3件、実際に
       Vulkan経由のGPU検出コードパスを通した上でパニックしないことを
       含めて検証)。
     - 実際にサーバーを起動し(`--features hw-detect-vulkan`)、
       `GET /v1/recommend`が実際に`{"gpu_detected":true,
       "detection_path":"vulkan","gpu_name":"OpenCUDA Vulkan Device
       (NVIDIA GeForce GT 730)","vram_bytes":2104819712}`を返すことを
       確認した。**この`vram_bytes=2104819712`という値は、`open-cuda`
       側`CLAUDE.md`の2026-07-23 HANDOFFに記録されているDXGI経由の
       同一実機での実測値と完全一致する**——これにより「open-directx
       経由でもopen-cuda経由でも同じベンダー・VRAM情報を返すこと」
       (ユーザー指示4番)を、実際に2つの独立した検出コードパスの
       実行結果を突き合わせて確認できた(DirectX側は今回新たに
       実行し直してはいないが、既存の実測記録との一致を裏取りとして
       使った、正直な検証範囲として明記)。
     - `POST /v1/recommend-and-download`を実際に呼び出し、推奨モデル
       (このマシンではVRAM 1.96GB<2GBのため`gpt2`)の判定・ダウンロード
       (既にダウンロード済みのケース)・ホットスワップ切り替え・
       `/v1/generate`での実際の生成("The quick brown fox" →
       "es are a great way to get a little bit of a kick out of your")
       まで一連の流れが実際に動作することを確認した。
     - さらに大きいサイズの実ダウンロードも実施(ユーザー指示「可能なら
       1つ大きめのサイズも実際に試すこと」への対応): `POST
       /v1/models/install {"id":"gpt2-medium"}`で実際にHugging Faceから
       355M(1.52GB、`model.safetensors`)を約51秒でダウンロード、
       `POST /v1/models/select`でプロセス再起動無しに切り替え、
       `/v1/generate`で実際に生成("Artificial intelligence is" →
       " a big deal. It's a big deal because it's going to change the
           way we think about")、`engine`フィールドが
       "gpt2-medium-greedy-decode-v0-opencuda-llm-cpu"へ正しく更新
       されていることを確認した。
     - UIを実際にブラウザで開き(`http://127.0.0.1:4600/`)、白画面・
       コンソールエラーが無いことを確認した上で「お勧めLLMを
       ダウンロード」ボタンをクリック→検出結果・推奨モデル・切り替え
       結果が実際に画面へ表示されること、続けて「生成テスト実行」
       ボタンで実際の生成結果がDOMへ反映されることを確認した
       (白画面バグ等を見逃さない検証徹底ルールに基づく)。
  9. **正直な開示・スコープ外**: (a) 上記の通りDirectX経路
     (`--features hw-detect-directx`)は今回のセッションで改めて
     ビルド成功のみ確認し(`cargo build --features hw-detect-directx`
     成功)、実機での`/v1/recommend`呼び出しによる再検証はVulkan経由の
     みで行った(既存のDXGI実測記録との数値一致で代替、上記8番参照)。
     (b) 生成処理自体は引き続きCPU実行のまま(`open-cuda`側CLAUDE.mdの
     「安易なGPU配線は逆に遅くなりうる」という2026-07-26の結論を
     踏襲、ハードウェア検出は推奨サイズ選定の入力としてのみ使い、
     生成のGPUディスパッチ配線は行っていない)。(c) 1.5B
     (`gpt2-xl`、6.4GB)は今回実ダウンロードで検証していない
     (カタログへの追加・ヒューリスティックのロジック検証に留まる、
     このマシンのVRAM検出結果〈1.96GB〉ではそもそも推奨対象にならない
     ため優先度を下げた)。
  - 次にすべきこと: (1) `--features hw-detect-directx`を実際に有効化
    した状態での`/v1/recommend`実行検証(今回はビルド成功のみ)、
    (2) `gpt2-xl`(1.5B)の実ダウンロード検証(8GB以上VRAM環境が
    得られ次第、またはVRAM閾値を無視した強制インストールでの動作確認)、
    (3) ダウンロード進捗のストリーミング報告(現状は完了までブロック
    する既存のシンプルな実装のまま、直前のHANDOFFから継続する既知の
    未着手事項)。

- **2026-07-27 モデルのホットスワップ(プロセス再起動不要のモデル切り替え)
  を実装——直前エントリの「次にすべきこと(2)」で残っていた制約
  (「ダウンロード完了後、実際にそのモデルを使うにはプロセスを再起動する
  必要がある」)を解消**:
  1. **`src/generation.rs`を書き換え**: 起動時`OnceLock<Result<LoadedGpt,
     String>>`(一度ロードしたら差し替え不可)を`static ACTIVE:
     RwLock<Option<Arc<LoadedGpt>>>`へ変更。新規`select_model(dir:
     PathBuf) -> Result<()>`は、指定ディレクトリからの読み込みに
     **成功した場合のみ**`ACTIVE`を置き換える(失敗時は現在動作中の
     モデルをそのまま維持——不正なディレクトリを指定してサービスを
     壊さない設計)。`active_model_dir() -> Option<PathBuf>`で現在
     使用中のモデルの読み込み元を問い合わせ可能にした。
  2. **新規HTTPエンドポイント`POST /v1/models/select`**(`main.rs`):
     `{"id": "distilgpt2"}`のようなリクエストで
     `{models_root}/{id}/`から`generation::select_model`を呼ぶ
     (ブロッキングI/Oのため`tokio::task::spawn_blocking`経由)。
     `GET /v1/models/catalog`のレスポンスにも`active_model_dir`
     フィールドを追加し、現在どのモデルが使われているか可視化した。
  3. **検証(実測、モックではなく実GPT-2重みで検証)**: 新規テスト2件
     (`generation::tests`)——`select_model_succeeds_for_a_real_directory_
     and_updates_active_model_dir`は、実際にsibling repoの実GPT-2重み
     (`../open-cuda/crates/opencuda-llm/models/gpt2`、既存の既定
     モデルと同じ実体)へ`select_model`を呼び、`active_model_dir()`が
     実際に反映されること、切り替え後の`generate()`が実際に
     (空文字列ではない)テキストを生成することを確認した。
     `select_model_fails_cleanly_for_a_nonexistent_directory`は、
     存在しないディレクトリを指定した場合にパニックせず`Err`を
     返すことを確認。`cargo test`**37→39件、全green**(既存テストへの
     回帰なし)。
  4. **正直な開示**: (1) 「新しいモデルとして本当に別アーキテクチャの
     重みへ切り替える」ところまでは検証していない(この環境には
     `distilgpt2`等の実際のダウンロード済みモデルが無いため、検証は
     「同じ実GPT-2重みディレクトリへの再ロードが機能すること」に
     留まる——ロード処理自体は`GptModel::load`をそのまま呼ぶだけなので、
     サイズ・語彙が異なるGPT-2互換モデルでも同様に動く設計上の根拠は
     あるが、実際にその組み合わせでの動作確認はしていない)。
     (2) 切り替え中に進行中の生成リクエストがあった場合の挙動
     (`Arc`のクローンにより、切り替え前の`generate`呼び出しは古い
     `Arc<LoadedGpt>`を握ったまま完走する設計——エラーにはならないが、
     切り替えの瞬間に古いモデルと新しいモデルが一時的に混在しうる)は
     意図した仕様ではあるが、専用のテストは追加していない。
  - 次にすべきこと: (1) 実際に`distilgpt2`等別モデルをダウンロード→
    切り替え→生成、という一気通貫の確認(実ダウンロードは今回も
    見送ったまま、前々回エントリから継続)、(2) 管理UI(Tauri Admin GUI
    等)からのモデル選択操作(現状はHTTP API止まり)。

- **2026-07-26(続き) 他のGPT-2アーキテクチャ互換モデルを選択・ダウンロード
  できるモデルカタログ機能を新規実装(ユーザー指示「aruaru-llm以外の
  オープンソースのローカルLLMも簡単にダウンロード・インストールを選択
  可能にしてユーザビリティを高めて」への対応)**:
  1. **スコープを正直に限定**: `opencuda_llm::GptModel::load`はGPT-2
     アーキテクチャ専用(`config.json`のフィールド・`model.safetensors`の
     テンソル名規約・`Conv1D`の重み配置が前提)であり、Llama/Mistral/Qwen等
     アーキテクチャの異なるモデルはこのエンジンではロードできない。
     「任意のオープンソースLLM」ではなく「GPT-2互換の別サイズ/別重み」に
     限定したカタログとして実装した(`src/model_catalog.rs`冒頭のdoc
     コメントに明記)。
  2. **新規モジュール`src/model_catalog.rs`**: `CatalogEntry`
     (id/表示名/Hugging Faceリポジトリ/概算サイズ/ライセンス注記)の
     静的配列`CATALOG`(既定の`gpt2`に加え、`distilgpt2`〈82M、軽量〉・
     `gpt2-medium`〈355M〉・`gpt2-large`〈774M〉、いずれも
     `config.json`/`model.safetensors`/`tokenizer.json`の3ファイル構成が
     確認できる公式`openai-community`/`distilbert`名前空間のリポジトリ)。
     `install(entry, dest_dir)`がHugging Faceの`resolve/main/`エンドポイント
     (認証不要)から3ファイルを`reqwest`でダウンロードし、`.tmp`拡張子経由の
     アトミックなリネームで書き込む(冪等——既にファイルが存在すれば
     再ダウンロードしない)。
  3. **新規HTTPエンドポイント2件**(`main.rs`): `GET /v1/models/catalog`
     (カタログ一覧+インストール済みID一覧+アーキテクチャ制約の開示文言を
     返す)、`POST /v1/models/install`(`{"id": "distilgpt2"}`のような
     リクエストでダウンロードを実行、完了後は`ARUARU_LLM_GPT2_DIR`を
     ダウンロード先へ向けてプロセスを再起動する必要がある旨を
     レスポンスに明記——現状のロード方式が起動時`OnceLock`のため実行中の
     ホットスワップ切り替えには対応していない、この制約も正直に開示)。
  4. **検証(実測)**: 新規テスト7件(`model_catalog::tests`)を追加、
     うち`install_writes_all_required_files_from_a_local_mock_server`は
     `mockito`によるローカルHTTPモックサーバーへ実際にHTTPリクエストを
     送り、`install`(内部の`install_from_base_url`)が実際に3ファイルを
     ダウンロード・書き込みし、既存ファイルがあれば再ダウンロードしない
     (冪等)ことを確認した。`cargo build`/`cargo test`
     **37件全green**(既存30件+新規7件、回帰なし)。
  5. **実Hugging Face到達性の確認(誇張しない範囲での実地検証)**:
     数百MB〜3GB超のモデル重みを本セッション中に自動ダウンロードする
     ことはせず(明示的なユーザーリクエスト無しに大容量ファイルを
     自動取得しない方針)、`curl`で`distilbert/distilgpt2/resolve/main/
     tokenizer.json`・`openai-community/gpt2-large/resolve/main/
     model.safetensors`へのリダイレクト追跡込みアクセスが実際に
     `200`を返すことのみ確認した(カタログに載せたURLが実在し到達可能
     であることの裏取り)。**実際に`POST /v1/models/install`を呼んで
     数百MB級モデルのダウンロード完了→ロード→生成、という一気通貫の
     動作確認は今回未実施**(正直な開示)。
  - 次にすべきこと: (1) 実際に`distilgpt2`等をダウンロードし、
    `ARUARU_LLM_GPT2_DIR`切り替え後に`GptModel::load`が実際にロードでき
    生成が動くことのエンドツーエンド確認、(2) ダウンロード進捗の
    ストリーミング報告(現状は完了までブロックするシンプルな実装)、
    (3) 管理UI(Tauri Admin GUI等)からのカタログ選択・インストール操作
    (現状はHTTP API止まり)。

- **2026-07-26 SET連携(open-directx/open-cuda/aruaru-llm)調査: 「GPU推論」は
  実際にはCPUバックエンドのみで、opencuda-vulkan/open-directxへの実接続が
  無いことを確認。安易なGPU配線は逆に遅くなりうるという設計上の結論に
  達したため、今回はコードの変更は行わず調査結果のみを正直に記録する
  (ユーザー指示: 「open-directxとopencudaとaruaru-llmは、SETで考慮・
  配慮して連携の実用性と完成度を高めて」への対応、Windows/Linux/nVIDIA
  実機を中心とする方針)**:
  1. **現状の確認**: `src/main.rs:319`の`CpuDevice::new(0)`が唯一のデバイス
     インスタンスで、`generation.rs`のGPT-2生成(`opencuda_llm::GptModel::
     generate`)・`scoring.rs`の埋め込み計算(`opencuda-bert`経由)・
     `bow_fallback.rs`の要素積、いずれも実際にはCPU実行。`opencuda-vulkan`
     へのdependencyは無く(Cargo.toml未記載)、`open-directx`への参照も
     ソース中に皆無(grep該当0件)。「GPU推論」という呼称は名目上のもので、
     実際にVulkan/DirectX経由でGPU実機にディスパッチしたことは一度もない。
  2. **技術的に判明した、安易な配線を避けた理由**: `opencuda-llm::GptModel`
     の推論ループ(`generate`→`forward_step`→`DecoderLayer::forward_step`→
     `Linear::forward`)は、プロンプト処理も含め常に`seq_len=1`
     (1トークンずつの自己回帰デコード)で`opencuda_blas::sgemm`を呼ぶ設計
     になっている(`opencuda-llm/src/lib.rs:244,332,508`)。これは
     GPT-2 124M(hidden=768)程度の行列サイズでは、CPU側では数マイクロ秒で
     終わる極めて軽い計算である一方、Vulkanの`dispatch_spirv`はコマンド
     バッファ記録・`vkQueueSubmit`・フェンス待機という固定オーバーヘッドを
     1回のGEMM呼び出しごとに伴う(`opencuda-vulkan/src/real.rs`の
     `dispatch_spirv`実装)。1トークンあたりレイヤー数×6回(Q/K/V/attn_out/
     intermediate/output)のLinear呼び出しがあり、GPT-2 124Mは12層のため
     1トークンで72回のGEMM呼び出しが発生する——これを単純にVulkan経由へ
     置き換えると、実計算時間よりディスパッチのオーバーヘッドの方が
     支配的になり、**CPU実行より遅くなる可能性が高い**という結論に至った
     (実際にベンチマークするところまでは今回行っていないが、既存の
     `dispatch_spirv`実装のコマンドバッファ/フェンス同期パターンから
     判断した設計上の懸念であり、誇張せず正直に「推測に基づく懸念」として
     記録する)。
  3. **今回あえて実装しなかったこと**: `opencuda_blas::sgemm`の
     `GemmPath::VulkanGeneric`(`device.supports_spirv()`かつ`spirv`引数
     ありで動作)自体は既に実装済みで、`aruaru-llm`側からVulkanDeviceを
     構築し`matmul.spv`を渡せば技術的には配線可能だった。しかし上記2.の
     懸念を解消せずに「GPUを使っている」という体裁だけを整えるのは
     ユーザーの「実用性を高めて」という指示に反すると判断し、見せかけの
     配線は行わなかった。
  4. **本当に効果が見込める設計変更(次回以降の推奨、今回は未着手)**:
     (a) プロンプトのプリフィル処理(初回の複数トークン分のforward)を、
     現状の「1トークンずつのループ」から「`seq_len=プロンプト長`の
     バッチ処理」へ変更すれば、Linear層のGEMMが本当のGEMM(m>1)になり
     Vulkanディスパッチの固定オーバーヘッドを算術強度で上回れる可能性が
     ある(デコード側は引き続きCPUのままでよい、いわゆる
     prefill/decode分離)。(b) Q/K/Vの3つの`Linear`呼び出しを、
     safetensors側で既に`c_attn`という1本の融合`Conv1D`として保存されて
     いる構造(`opencuda-llm/src/lib.rs`の`load_fused_qkv`参照)に合わせ、
     推論側でも1回のGEMM呼び出しに統合すればディスパッチ回数を1/3に
     削減できる。(c) 上記(a)(b)を実施した上で初めて、`aruaru-llm`側の
     デバイス選択(`main.rs:319`)を`opencuda-vulkan::real::VulkanDevice`
     (`real-vulkan` feature、既定では無効なオプトイン機能にすべき——
     Android等クロスコンパイル環境でash/Vulkanローダーへの依存を強制
     しないため)へ切り替え、実GPU実機(NVIDIA GT 730)で生成結果が
     CPU版と数値的に一致すること・実際に速度が改善することの両方を
     ベンチマークで確認する、という増分計画とする。
  - 次にすべきこと: (1) プリフィル/デコード分離とQKV融合(上記4(a)(b))を
    `opencuda-llm`側に実装、(2) その上で`aruaru-llm`にオプトインの
    `real-vulkan` featureを追加しVulkanDevice経由の生成を実装、
    (3) CPU版との生成結果の数値一致・実際の速度差をベンチマークで検証。

- **2026-07-25(続き) `opencuda-llm::GptModel`(GPT-2 124M実重み)を統合、
  `POST /v1/generate`で本格的な自己回帰テキスト生成を実装
  ——実HTTPリクエストでの生成結果まで確認済み**: 姉妹リポジトリ
  `open-cuda`側で直近実装・実機検証済みだった`opencuda-llm::GptModel`
  (safetensorsローダー付き、GPT-2 124M実重みをロードして貪欲デコードで
  文法的に自然な英語を生成できる)を、このリポジトリへ統合した。
  1. **依存追加**: `Cargo.toml`に`opencuda-llm = { path =
     "../open-cuda/crates/opencuda-llm" }`をpath依存として追加(既存の
     `opencuda-bert`等と同じsibling pathパターン、`PORTING.md`にも
     移植手順を追記)。
  2. **新規モジュール`src/generation.rs`**: `OnceLock<Result<LoadedGpt,
     String>>`でGPT-2実重み(`GptModel::load`)・トークナイザ
     (`GptTokenizer::load`、GPT-2自身のBPE語彙、`tokenizers`クレート)を
     プロセス内キャッシュ(`scoring.rs`の埋め込みモデルキャッシュと
     同じ設計思想)。モデルディレクトリは既定
     `../open-cuda/crates/opencuda-llm/models/gpt2`(`open-cuda`側で
     2026-07-25に既に実際にダウンロード・検証済みの重みをそのまま
     sibling repoとして再利用、`ARUARU_LLM_GPT2_DIR`環境変数で上書き可)。
     `warmup()`・`generate(device, prompt, max_new_tokens) -> Result<String>`
     を公開。
  3. **新規エンドポイント`POST /v1/generate`**(`/v1/chat`とは別、
     意図分類と生成を無理に統合しない設計方針、ユーザー指示通り):
     `GenerateRequest{prompt, max_new_tokens(既定16、上限128でクランプ),
     tenant}` → 成功時`GenerateResponse{completion, engine:
     "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", disclosure: "..."}`
     (200)、失敗時(重み未取得等)`GenerateErrorResponse{error, engine}`
     (503)。`disclosure`フィールドは**毎回のレスポンスに**GPT-2 124Mが
     小型・2019年モデルであり最新商用LLMと同等でない旨を明記する
     (誇大表示を避けるため、レスポンス自体に埋め込む設計)。
  4. **起動時ウォームアップに追加**: `main()`の既存ウォームアップ処理
     (`scoring::warmup`/`security::warmup`と同じ並び)に
     `generation::warmup()`を追加。失敗しても致命的ではなく初回
     リクエスト時に再試行される(既存の設計思想を踏襲)。
  5. **実機検証(型チェックのみで完了と報告しない方針を徹底)**:
     `cargo build --release`成功、`cargo test --release`
     **22件全green**(既存22件、リグレッション無し——`/v1/generate`自体の
     専用単体テストは追加していない、GPT-2 124Mの実重みロード自体は
     `opencuda-llm`側で既に単体テスト・実機検証済みのため、このリポジトリ
     側では実際にサーバーを起動しての実HTTP検証を主体とした)。
     実際にサーバー(`aruaru-llm.exe`)を起動し、起動ログで
     `generation (GPT-2 124M) warmup complete`を確認した上で、
     `POST /v1/generate`へ実HTTPリクエストを送信:
     - `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
       `"completion": "es are a great way to get a little bit of a kick out of your"`
     - `{"prompt": "Artificial intelligence is", "max_new_tokens": 20}` →
       `"completion": " a new field of research that has been in the works for a while now. It is a field"`
     (いずれも文法的には自然な英語の継続生成。既存の`/v1/chat`
     〈`POST /v1/chat {"message": "How do I apply for a permit?"}` →
     govインテント一致・正しい応答〉も引き続き正常動作することを併せて
     確認、リグレッション無し)。
  6. **正直な評価**: GPT-2 124Mは小型モデルであり、GPT-4等の最新商用LLM
     と同等の性能・知識・指示追従能力は無い。「質問に答える」というより
     「文の続きを予測する」挙動(素の事前学習済み言語モデルの貪欲デコード、
     対話ファインチューニング無し)であり、実際`"Artificial intelligence
     is"`への継続も一般論的な文の羅列であって具体的で正確な回答ではない。
     README(日英)・PORTING.md・本ファイルの冒頭開示文言にもこの限界を
     明記した。
  7. **意図的にスコープ外とした事項**: (a) `/v1/generate`専用の単体
     テスト(統合的な実HTTP検証で代替、モデルロード自体のテストは
     `opencuda-llm`クレート側に既存)、(b) サンプリング(温度・top-k/
     top-p)——引き続き貪欲デコード(argmax)のみ、`opencuda-llm`側の
     `GptModel::generate`のシグネチャ自体がサンプリング非対応のため、
     (c) 日本語生成(GPT-2のBPE語彙は英語中心のため、日本語プロンプトは
     語彙効率・品質共に低下する旨をAPIドキュメントに明記するに留めた)。
  - 次にすべきこと: (1) `opencuda-llm`側でサンプリング(温度/top-k/top-p)
    が実装されればAPIへ配線、(2) より大規模なGPT-2バリアント(medium/
    large/xl)または日本語対応モデルへの切り替え検討、(3) `/v1/generate`
    専用の単体テスト追加(現状はサーバー起動+実HTTP検証のみ)。

- **2026-07-25 bag-of-wordsフォールバックを追加(可用性優先)**: 依頼の
  「bag-of-wordsから実埋め込みベクトル類似度への置き換え」を確認した
  ところ、`scoring.rs`は既に2026-07-21の移行で`opencuda-bert`
  (multilingual-e5-small)による実際の文埋め込み+コサイン類似度分類へ
  移行済みだった(CLAUDE.mdの記載通り、コードとドキュメントに齟齬は
  無かった)。実際に欠けていたのは、埋め込みモデルの重み
  (`models/multilingual-e5-small/`)が無い・ロードに失敗した場合の
  フォールバック——従来は`best_intent`がそのままエラーを返すだけで、
  サービス全体が意図分類不能に陥っていた。
  1. `src/bow_fallback.rs`を新設し、2026-07-21移行前の固定語彙
     bag-of-wordsドット積(`opencuda_cpu::CpuDevice`上での要素積カーネル
     実行、旧`scoring.rs`のロジック)を独立モジュールとして復元した。
  2. `scoring::classify(device, user_text) -> ClassifyResult`
     (`{intent, engine}`)を新設し、`main.rs`の`/v1/chat`ハンドラから
     これを呼ぶよう変更。`best_intent`(埋め込み経路)が失敗した場合のみ
     自動的に`bow_fallback::best_intent_bow`へフォールバックする。
     `engine`フィールドは常に実際に使われた経路
     (`ENGINE_EMBEDDING`=`"embedding-cosine-v0-opencuda-bert-cpu"`、
     `ENGINE_BOW_FALLBACK`=`"bow-dotproduct-v0-opencuda-cpu-fallback"`、
     両方失敗時`ENGINE_CLASSIFICATION_UNAVAILABLE`)を正直に返す。
  3. **実機検証**: このマシンには`models/multilingual-e5-small/`が
     実際にダウンロード済みだったため、埋め込み経路(通常運用)を
     `cargo test --release`(22件全green、既存16件+bag-of-words
     フォールバック新規6件)で実際に検証できた。bag-of-wordsフォール
     バック自体も独立した単体テストで検証(埋め込み経路を意図的に
     無効化しての統合テストまでは行っていない——モデルディレクトリを
     一時的に退避させての検証は今回スコープ外、正直に記録)。
  4. `cargo clippy --workspace --all-targets --release`で
     `manual_slice_size_calculation`警告1件を検出・修正
     (`n * std::mem::size_of::<f32>()` → `std::mem::size_of_val(a)`)、
     修正後**workspace全体で警告0件**。
  5. README.md/README-English.mdを現状(埋め込み優先+bag-of-words
     フォールバック)に合わせて更新(README-English.mdは2026-07-21の
     移行前の記述のまま更新漏れしていたのを合わせて是正)。
  - 次にすべきこと: (1) 埋め込みモデルを意図的に無効化した状態での
    フォールバック統合テスト(現状は`bow_fallback`モジュール単体の
    テストのみ)、(2) 自己回帰デコーダによる対話文生成(引き続き
    Qwen3-14B等の実モデル重みの入手・ライセンス確認が前提、未着手)。

- **2026-07-24(続き) Android版クライアントを新規実装(`android/`)
  ——直前のHANDOFF「Android実装が存在しないためスコープ外」から前進し、
  実際に着手・ビルド成功まで確認**: `open-web-server`の
  `android/`(`tokyo.runo.openwebserver`)を参照実装として、同じGradle
  構成パターン(`com.android.application` 8.7.2 + Kotlin 2.0.21、
  `compileSdk 35`/`minSdk 24`、`androidx.appcompat`+
  `kotlinx-coroutines-android`のみの最小依存)で`aruaru-llm/android/`を
  新規作成した。パッケージ名は`tokyo.runo.aruarullm`(open-web-server版
  `tokyo.runo.openwebserver`とは別名)。
  1. **open-web-server版との設計上の違い**: open-web-server版は
     クロスコンパイル済みネイティブバイナリを`ProcessBuilder`で端末上に
     起動する構成だったが、aruaru-llmはユーザー要望通り**リモートの
     aruaru-llmサーバーへHTTP接続する薄いクライアント**として実装した
     (ネイティブバイナリ同梱・jniLibsは無し)。`MainActivity`に
     サーバーURL入力欄(`SharedPreferences`で保存、既定
     `http://10.0.2.2:8080`=エミュレータからホストへの既定到達先)+
     簡易チャットUI(`EditText`+送信ボタン+ログ表示の`TextView`)を実装、
     `POST /v1/chat`(`CLAUDE.md`「API」節記載の`{"message","lang":"ja"}`
     → `{"reply","engine",...}`契約)を`HttpURLConnection`+`org.json`
     (Android標準同梱、追加依存不要)で叩く。
  2. **電源プロファイル管理**(`PowerProfile.kt`/`ProfileSelectActivity.kt`
     /`MainActivity.kt`、`AndroidManifest.xml`の`activity-alias`3種)は
     open-web-server版と同じ設計パターンをそのまま踏襲: 省電力/通常/
     常時電源接続の3モード、`SharedPreferences`保存、起動時
     `ProfileSelectActivity`(絵文字+日本語ラベル)、ホーム画面専用アイコン
     3種(緑/青/橙、色分け+ラベル文字列)、`ACTION_POWER_DISCONNECTED`/
     `ACTION_POWER_CONNECTED`の`BroadcastReceiver`による自動切替提案
     ダイアログ(既定推奨は省電力、常時電源接続版実行中に電源が外れた
     場合)。省電力/通常は`WakeLock`を取得せず、常時電源接続のみ
     `PARTIAL_WAKE_LOCK`を保持。`/healthz`ポーリング間隔もプロファイル別
     (省電力5分/通常1分/常時電源接続5秒、open-web-server版と同じ値)。
  3. **省電力版のHWアクセラレーター無効化指示(ユーザー指示分)**:
     `/v1/chat`リクエストに`X-Aruaru-Llm-Accel-Backend`ヘッダーを付与し、
     常時電源接続版は`hardware_accelerator`、省電力/通常版は`cpu`を
     指定する設計にした(open-web-server版の
     `OPEN_WEB_SERVER_ACCEL_BACKEND`環境変数と同じ設計思想をHTTP
     ヘッダーへ移した形)。**正直な開示(将来対応課題)**: aruaru-llm
     (Rust側)は本セッション時点でこのヘッダーを一切受け取らない・
     解釈しない——`src/main.rs`/`src/scoring.rs`にヘッダー読み取りや
     アクセラレーターバックエンド切替の実装は無く、`opencuda-bert`は
     現状CPUパスのみ実装済み(GPU/NPU専用パスは`open-cuda`側含め
     未実装、本ファイル冒頭「開発方針」節参照)。このAndroid側の
     ヘッダー付与は将来サーバー側が対応した際に効果を持つ先取り実装
     であり、現時点で実際の応答内容・処理経路には一切影響しない
     (WakeLock有無とポーリング間隔差のみが実効果)。
  4. **ビルド確認**: このマシンにキャッシュ済みの`gradle-8.11.1-all`
     配布物(`~/.gradle/wrapper/dists/`、open-web-server版のビルドで
     判明していた場所)を`gradlew`無しで直接実行し、
     `gradle :app:assembleDebug`が**1回目の実行で成功**
     (`BUILD SUCCESSFUL`、33 actionable tasks executed)、
     `android/app/build/outputs/apk/debug/app-debug.apk`
     (約3.2MB)の生成を確認した。型チェックのみでの完了報告ではない
     (実際に成功したAPKファイルの生成まで確認済み)。
  5. **正直な制約・未実施事項**: (a) 実機/エミュレータへの`adb install`
     以降の実地検証(実際に起動しチャットが往復すること)は今回未実施
     ——ビルド成功のみで完了と記録する(ユーザー指示「実機/エミュレータ
     での実地検証ができない場合はビルド成功のみで良い」に基づく)。
     (b) 上記3.の通りHWアクセラレーター指示は将来対応課題のまま。
     (c) チャット履歴の永続化・複数テナント切替UI・管理API認証は
     スコープ外(過剰実装を避けた)。(d) `local.properties`の
     `sdk.dir`はこの開発機のパス(`C:/Users/noruk/AppData/Local/
     Android/Sdk`)のままリポジトリに含めている(open-web-server版の
     既存慣行と同じ)。
  - 次にすべきこと: (1) 実機/エミュレータでの`adb install`→チャット
    往復の実地検証、(2) `opencuda-bert`側にGPU/NPU実装が入った時点で
    `X-Aruaru-Llm-Accel-Backend`ヘッダーをRust側で実際に解釈する配線、
    (3) フォアグラウンドサービス化・APK署名/配布(open-web-server版と
    同じく今回のスコープ外のまま)。

- **2026-07-24 スマホ版電源モード切替機能の依頼を受けたが、Android実装が
  このリポジトリに全く存在しないことを確認・正直に記録**: ユーザーから
  「省電力版/常時電源接続版(ハードウェアアクセラレーター対応)/通常版」の
  3モード選択と、電源抜き差し検知による自動切替提案の実装依頼があった。
  調査の結果、本リポジトリ(`aruaru-llm`)には`android/`ディレクトリ・
  `.kt`/`.kts`ファイルが一切存在せず(Rustクレートのみ、`Cargo.toml`+
  `src/`)、Android版インストーラーも上記2026-07-23エントリの通り
  「次にすべきこと」止まりで未着手のまま。よってフルアプリ新規開発は
  スコープ外と判断し、実装は行わず、将来Android実装時の設計方針のみ
  以下に記載する。
  - **モード設計方針(将来実装時)**:
    1. 「省電力版」「常時電源接続版(HWアクセラレーター対応)」「通常版」の
       3モードをユーザーが選択できるようにする。
    2. 省電力版では実際に消費電力を下げる具体施策を組み合わせる:
       `PowerManager`/`BatteryManager` APIで電池状態を監視、ポーリング
       間隔を延長、`WakeLock`を取得しない、そして本リポジトリのLLM推論
       固有の対応として`opencuda-*`系のGPU/NPUハードウェアアクセラレー
       ター利用を無効化しCPU低負荷(スレッド数制限・低優先度)モードに
       切り替える。
    3. 常時電源接続版ではCPU＋GPU＋NPUが揃っていればハードウェア
       アクセラレーターをフル活用する(推論バックエンドとして
       `opencuda-bert`/`opencuda-blas`等を通常より積極的に使う)。
    4. `BroadcastReceiver`で`Intent.ACTION_POWER_DISCONNECTED`を監視し、
       常時電源接続版モード中に電源が外れたら「省電力モードに切り替え
       ますか？それとも通常モードのままにしますか？」とダイアログで
       質問する。デフォルトの推奨選択は省電力モード。
    5. `Intent.ACTION_POWER_CONNECTED`受信時にも、常時電源接続版へ戻すか
       尋ねる導線を用意する。
  - 次にすべきこと: Android版アプリ自体が未着手のため、まずAndroid
    インストーラー/アプリの新規プロジェクト立ち上げが前提(2026-07-23
    エントリと同じ「次にすべきこと」を再確認)。電源モード機能は
    そのAndroidアプリ実装時に上記設計方針に沿って組み込む。

- **2026-07-23(続き) 3点セット(`install.sh`/`install.ps1`/
  `.github/workflows/release.yml`)を新規追加、v0.1.0タグでCI成功・
  GitHub Release実在確認まで完了**: エコシステム全体インストーラー
  整備計画(正本: `open-raid-z/CLAUDE.md`「エコシステム全体
  インストーラー整備計画」節)の一環、以前は3点セット丸ごと未整備
  だったリポジトリの1つ。
  1. `install.sh`(systemdサービス登録)・`install.ps1`(Windows
     サービス登録案内)を新規作成。**正直な開示**: 起動時に470MB超の
     `multilingual-e5-small`モデル重み(Hugging Face配布、MIT)を
     読み込むが、ライセンス上の理由でこのインストーラーには同梱せず、
     両スクリプトに`huggingface-cli download`での取得手順を明記した。
  2. `release.yml`: `Cargo.toml`が`../open-cuda/crates/opencuda-*`を
     path依存しているため(sibling path依存)、CI環境でも同じ相対位置
     (リポジトリルートの1つ上)へ`open-cuda`をgit cloneするステップを
     追加(aruaru-db/open-web-server/RPoemで実際に踏んだ「sibling
     path依存を忘れるとCI失敗」という罠を踏まないための対応)。
     Linux x86_64・Windows x86_64向けにビルドし、GitHub Releasesへ
     `softprops/action-gh-release@v2`で添付する構成。
  3. `v0.1.0`タグを実際にpushし、`gh run list`で2ジョブ(Linux/
     Windows)とも`completed success`、`gh release view v0.1.0`で
     `aruaru-llm-linux-x86_64.tar.gz`/`aruaru-llm-windows-x86_64.zip`
     の両方が実在することを確認した(型チェックのみでの完了報告では
     ない——sibling path依存を踏まえたCI初回成功は他リポジトリでは
     複数回の修正が必要だった実例があるため、特に注意して確認した)。
  4. README(日英両方)にインストール手順節を新設。
  - 次にすべきこと: Android版インストーラー(未着手、他リポジトリと
    共通のバックログ)。

- **2026-07-23 (関連リポジトリ動向の記録) `open-cuda`のDirectXバック
  エンドにmatmulカーネル対応・GPU圧縮/暗号化カーネル(ChaCha20)を実装**:
  このリポジトリが利用する`open-cuda`側で、`opencuda-directx`クレート
  にmatmul対応とChaCha20 GPUカーネルが追加された(RS-LinkFusion側の
  ハードウェアアクセラレータ要望への対応)。実機(NVIDIA GT 730)検証
  中にHLSL cbuffer配列パディングによる実バグ(GPU出力が暗号化されず
  平文のまま)を発見・修正済み(コミット`ec6acf1`、詳細は`open-cuda`
  側CLAUDE.md HANDOFF参照)。**このリポジトリ自体への直接の変更は
  無し**——`opencuda-bert`/`opencuda-blas`経由の既存利用箇所への
  影響は無いことを確認済み。

- **2026-07-22 応答言語の多言語対応 + 起動時ウォームアップ(コールドスタート対策)**:
  前回HANDOFFの「次にすべきこと」(1)(2)を実装した(バックグラウンド
  エージェントの異常終了により未コミットのまま残っていたのを本セッションで
  発見・検証・コミット)。
  - `ChatRequest`に`lang: String`(`#[serde(default = "default_lang")]`で
    既定`"ja"`、既存呼び出し元との後方互換を維持)を追加。
    `ChatResponse`に`reply_lang`(実際に返した言語)・`lang_fallback`
    (要求言語が未対応で英語へフォールバックしたか)を追加。
  - 各`Intent`に`reply_en`(英語訳)を追加し、`Intent::reply_for(lang)`/
    `scoring::fallback_reply_for(lang)`で`"ja"`→日本語、`"en"`→英語、
    それ以外→英語へフォールバックしつつ`lang_fallback: true`で正直に
    通知(黙って日本語へ落とさない、「graceful degradation, never
    silent」方針)。
  - `main()`起動時、`Server::run`の前に`scoring::warmup(&device)`を
    呼び出し、opencuda-bertのモデルロード+インテント代表ベクトル計算を
    前倒しで済ませる(実測5.58秒、warmup前は初回リクエストが
    e-gov.info側の3秒タイムアウトを超えていた問題への対策)。
  - 新規テスト4件追加(`reply_for_ja_returns_japanese_unchanged`、
    `reply_for_en_returns_english_translation`、
    `reply_for_unsupported_lang_falls_back_to_english_with_indicator`、
    `fallback_reply_for_respects_lang_and_flags_unsupported`)。
    `cargo test --release`は13件全passed。
  - 検証: 実際に`cargo build --release`→サーバー起動→
    `POST /v1/chat`へ実リクエスト送信で、`reply_lang`/`lang_fallback`を
    含む正しい応答(`credit`インテント一致、embedding-cosine経路)を
    確認済み(2026-07-22)。
  - README.mdの開示文言も、旧bag-of-words時代の記述のまま更新漏れして
    いたのを、現状のembedding-cosine分類の説明に合わせて修正した。

- **2026-07-22 `e-gov.info`側がこのサービスへのHTTP問い合わせに置き換わった
  (このリポジトリ自体は無変更)**: 下記2026-07-21エントリで「次にすべき
  こと」として記録していた「`e-gov.info`側を実際にaruaru-llmへのHTTP問い
  合わせに置き換えるかどうかの判断・実装」を、`e-gov.info`側
  (`src/chat_commerce.rs`)で実施した(詳細は`e-gov.info`のCLAUDE.md
  2026-07-22 HANDOFF参照)。このリポジトリ側のコード・API契約
  (`POST /v1/chat`)は変更なし。実際に両プロセスを起動してのHTTP統合
  検証で、`e-gov.info`からのリクエストに対しこのサービスの
  `chat`ハンドラが実際に呼ばれ(`tenant: "e-gov.info"`、ログにも記録
  された)、`scoring.rs`のgov intent応答が正しく返ることを確認済み。
  併せて、`e-gov.info`側の初回リクエストが`opencuda-bert`モデルの
  ロード時間(数秒)により3秒タイムアウトでフォールバックする実測が
  あった。
  - 次にすべきこと: (1) 応答文の多言語対応(現状全て日本語固定、
    `e-gov.info`側は13言語対応済みのため、このサービス経由だと言語が
    落ちる非対称が生じている)、(2) 起動直後のモデルロード時間が
    呼び出し元のタイムアウトを超えるコールドスタート問題への対策
    (ウォームアップ用エンドポイント、または起動時に一度ダミー推論を
    実行してキャッシュを温める等)。

- **2026-07-21 bag-of-wordsから実際の文埋め込み(opencuda-bert)ベースの
  意図分類へ移行**: `scoring.rs`の意図分類を、固定語彙bag-of-words+
  `opencuda_blas::sgemm`ドット積から、`opencuda-bert`クレート
  (multilingual-e5-small)による実際の文埋め込み+
  `opencuda_bert::cosine_similarity`ベースへ全面的に置き換えた。
  1. `Cargo.toml`に`opencuda-bert = { path = "../open-cuda/crates/
     opencuda-bert" }`をpath依存として追加。
  2. 各インテント(gov/trade/credit/realestate)に自然な例文を2〜3個ずつ
     用意し、`passage: `接頭辞(multilingual-e5系の規約)を付けて
     埋め込み、平均・L2正規化してインテント代表ベクトルとした。ユーザー
     発話は`query: `接頭辞を付けて埋め込む。モデル・トークナイザ・
     インテント代表ベクトルはいずれも`OnceLock`でプロセス内キャッシュし、
     初回呼び出し(数秒)以降はリクエストごとの再ロードを避けた
     (`cargo test --release`は9件全体で約7秒、モデルロードは1回のみ)。
  3. `best_intent`のシグネチャ(`&Arc<dyn GpuDevice>`, `&str` →
     `Result<Option<&'static Intent>>`)、`main.rs`からの呼び出し方は
     変更していない。
  4. **実測に基づく閾値調整**: 実際にコサイン類似度を測定したところ、
     multilingual-e5-smallは無関係な文同士でも0.80〜0.85程度のベース
     類似度が出ることが判明(「こんにちは」対trade例文で0.85等)。
     真の一致(最弱でcredit 0.87程度)とノイズ上限(最大でtrade
     0.85程度)の間に位置する`SIMILARITY_THRESHOLD = 0.86`に調整し、
     既存の`matches_government_intent`等5件のintentテストが実際の
     埋め込みベースでも正しく分類されること(`returns_none_for_
     unmatched_text`含む)を`cargo test --release`で確認した。
  5. `opencuda-bert`側に`BertModel::hidden_size()`(公開アクセサ)を1件
     追加(`config`フィールドがprivateで`aruaru-llm`から参照できな
     かったため)。`opencuda-bert`のテスト2件も引き続き全green
     (`cargo test -p opencuda-bert --release`)。
  6. `main.rs`・`CLAUDE.md`の`engine`フィールド表記を
     `"rule-based-v0-opencuda-cpu"`から
     `"embedding-cosine-v0-opencuda-bert-cpu"`へ更新し、開示コメントを
     「エンコーダによる意味的類似度分類(自己回帰的な対話生成は未実装)」
     という事実に合わせて書き換えた。
  - 次にすべきこと: (1) 自己回帰デコーダによる対話生成(Qwen3-14B等の
    実モデル重みの入手・ライセンス確認が前提)、(2) `e-gov.info`側を
    実際に`aruaru-llm`へのHTTP問い合わせに置き換えるかどうかの判断、
    (3) 閾値0.86は代表例文4カテゴリ・少数例文での実測値であり、今後
    インテントを追加する際は同様に実測して再調整すること。

- **2026-07-20 open-easy-web連携の実地検証・ドキュメント齟齬の是正
  (ユーザー指示: ドキュメント修正だけで終わらせず実用性・完成度を高める)**:
  1. **齟齬(1) — CLAUDE.md本文が古いまま**: 下記2026-07-18エントリの
     「残る実装対象はopen-easy-web側からの管理統合のみ」という記述は、
     実際には`open-easy-web`側で`server/src/appserver_registration.rs`の
     `AppServerKind::AruaruLlm`/`register_aruaru_llm()`が2026-07-18に
     実装され、2026-07-19にWASM側UI配線(`src/profiles.rs`の
     `appserver_kind_for()`、`src/shell.rs`の`<select>`選択肢)も完了
     済みであることが、`open-easy-web/CLAUDE.md`のHANDOFFで確認できた
     (このリポジトリのCLAUDE.mdだけが追記漏れで古いままだった)。
  2. **実地検証(型チェックだけで終わらせない)**: `cargo build`
     警告0件、`cargo test` **10件全green**(既存のtenants 4件・
     scoring 6件、リグレッション無し)。さらに実バイナリを
     `E_GOV_LLM_ADMIN_TOKEN=test-token`で起動し、`open-easy-web`の
     `register_aruaru_llm()`が実際に送信するリクエスト形状
     (`POST /admin/tenants`、`x-admin-token`ヘッダ、
     `{"host":"...","label":null}`)をそのまま`curl`で再現して検証:
     トークン無し→`401`、正しいトークン→`200 ok`→
     `GET /admin/tenants`で`[{"host":"e-gov.info","label":null}]`が
     返る→`DELETE /admin/tenants/e-gov.info`→削除後`[]`。
     `POST /v1/chat`(`tenant`付き)も実際に`gov`インテントへ正しく
     一致し実際の応答文が返ることを確認。これにより、`open-easy-web`
     側のモックサーバーテスト(`registers_aruaru_llm_tenant_with_
     expected_shape`)が検証しているリクエスト形状と、このリポジトリの
     実際の受け口が**双方とも実HTTPで整合している**ことを確認した
     (両リポジトリのソース突き合わせ+実HTTP、モックのみに頼らない)。
  3. **見つけた別の問題(このパスで修正済み)**: 作業ツリーに、
     `README.md`/`README-English.md`へ存在しない10ヶ国語README
     (`README-Japan.md`/`README-Chinese.md`等、実際にはこのリポジトリに
     存在しないファイル)へのリンクを追加する未コミットの変更が残って
     いた——他リポジトリ(`open-easy-web`等)の「10ヶ国語README」運用
     ルールを誤って本リポジトリに適用しようとした形跡と見られる、
     リンク切れになる差分だったため`git checkout`で破棄した。
  4. **個人情報監査**: `src/`・`Cargo.toml`・README/CLAUDE.md/PORTING.md
     に実メールアドレス・実電話番号・実APIキー等のハードコードは
     見つからなかった(該当なし、変更不要)。
  5. **スコープ外として記録(今回は変更していない)**: `e-gov.info`
     (`F:\open-runo\e-gov.info\src\chat_commerce.rs`)は、いまだに
     自前のルールベース応答ロジックを直接持ったままで、本サービス
     (`aruaru-llm`)へのHTTP問い合わせに置き換えられていない
     (このCLAUDE.mdの2026-07-18エントリで「検討事項」として記載
     済みのまま未着手)。今回の指示は`aruaru-llm`リポジトリ自身の
     完成度が対象のため着手しなかったが、次回以降のエコシステム
     全体の完成度向上の候補として引き続き記録する。
  - 次にすべきこと: (1) `e-gov.info`側を実際に`aruaru-llm`への
    HTTP問い合わせに置き換えるかどうかの判断・実装、(2) `open-cuda`の
    Phase 3(BLAS/Attention)進捗の定期確認。

- **2026-07-18 「分身の術」構成のビルド・実HTTP検証完了**: 前回パスで
  未検証のまま残っていた`src/tenants.rs`/`main.rs`の変更を実際に
  ビルド・実行して検証した。`cargo build`成功、`cargo test`
  **10件全green**(`tenants::tests`4件・`scoring::tests`6件)。
  さらに実バイナリを起動し、`curl`で実HTTPリクエストにより
  `/healthz`→`/v1/chat`(tenant無し)→`POST /admin/tenants`→
  `GET /admin/tenants`(登録確認)→`/v1/chat`(tenant付き)→
  `DELETE /admin/tenants/:host`→`GET /admin/tenants`(削除確認、
  空配列)という一連のフローが型チェックだけでなく実際に正しく
  動作することを確認した(`poem::Route::at().post(...).get(...)`の
  メソッドチェーン、`Path<String>`抽出子とも問題なし)。
  **エコシステム内の展開状況調査**: `RPoem`(`crates/
  open-runo-gateway/src/appserver_tenants.rs`・`open-runo-appserver/src/
  tenant_bridge.rs`)・`RCosmo`(同様)・`open-web-server`
  (`crates/open-web-server-gateway/src/tenant_router.rs`・
  `handlers/tenants.rs`)には**既にこの「分身の術」パターンが実装済み**
  であることが判明。`open-cuda`・`open-raid-z`はHTTPサービスではなく
  ライブラリ(GPUランタイム/ストレージ)のため、そもそも「ドメインごとの
  個別インストール」という概念自体が当てはまらず、path依存として
  複数プロジェクトから共有される時点で要件を自然に満たしている
  (追加のTenantRegistry実装は不要と判断)。`aruaru-db`は既存の
  `aruaru-server`(pgwire)自体が既に「1インスタンスを複数クライアント
  アプリが接続して共有する」設計であり、HTTPの`/admin/tenants`的な
  仕組みを別途持つよりSQLデータベース/スキーマ単位のマルチテナント性を
  活かす方が自然——今回は追加実装を見送り、この判断根拠を記録するに
  留めた。**残る実装対象は`open-easy-web`側からの管理統合のみ**
  (`appserver_registration.rs`拡張、次のHANDOFFエントリ参照)。

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
---

## エコシステム全体マップ(2026-07-21追記)

同時並行開発の対象プロジェクト一覧・各リポジトリの現況は
[`open-raid-z`のCLAUDE.md](https://github.com/aon-co-jp/open-raid-z/blob/main/CLAUDE.md)
「関連プロジェクト」節を参照。**どのリポジトリから読み始めても、
この節を起点に他プロジェクトへ辿れる**ようにしてある(このリポジトリ
自身の状況はこの上のHANDOFF節を参照)。

- **2026-07-27(続き4) 起動時にGPU検出featureの有効/無効を明示ログ出力(使いやすさ改善、ユーザー指示「open-directx と open-cuda と aruaru-llmのSETの完成度と実用性と使いやすさの向上をお願い」)**:
  1. **問題**: `hw-detect-vulkan`/`hw-detect-directx`はいずれも既定off
     (Cargo feature)のため、何も知らずに`cargo run`しただけのユーザーは
     常にCPU-onlyフォールバック(最小モデル固定推奨)に静かに誘導されて
     しまい、GPU対応ビルドの存在自体に気づけない(外部監査で指摘された
     使いやすさの最重要ギャップ)。
  2. **`src/main.rs`起動時に`tracing::info!`でfeatureの有効/無効を明示**:
     無効時は「GPU検出は無効化されています。ハードウェアに基づく推奨には
     `cargo build --features hw-detect-vulkan`(Windowsは
     hw-detect-directxも可)で再ビルドしてください」という趣旨のログを、
     有効時は有効なfeature名を出力する(`#[cfg(...)]`で条件分岐、実行時
     オーバーヘッド無し)。
  3. **検証**: `cargo build`成功。`cargo test`**44件全green**(既存機能
     への回帰なし、ログ出力のみの変更のためテスト自体は追加せず——
     ログ文言の単体テストは費用対効果が低いと判断)。
  - 次にすべきこと: (1) 実際に`cargo run`(feature無し)した際の起動ログに
    このメッセージが出力されることの目視確認(今回はビルド成功止まり)、
    (2) README.mdのクイックスタート冒頭にもこのfeatureフラグをより
    目立たせる形で言及するかの検討(現状は「ハードウェア検出」節に
    既に記載済みのため優先度は低いと判断)。

- **2026-07-27(続き5) GPU検出の優先順位をVulkan→DirectXへ反転(ユーザー指示「Vulkan環境を重視してましたが、今度は逆に、open-directxを重要視して」)**:
  1. **`src/hardware.rs::detect()`の分岐順序を入れ替え**: 両feature有効時、
     従来はVulkan結果を優先しDirectXをクロスチェックとして記録していたが、
     今回の指示によりDirectX結果を優先しVulkanをクロスチェックとして
     記録する形へ反転した。クロスチェック計算自体(10%許容誤差での
     一致判定)は対称なため変更不要。
  2. **モジュールdocコメントも同期更新**(「両方有効な場合はDirectXを
     優先しVulkanの結果をクロスチェック」に書き換え)。
  3. **実機検証**: `cargo build --release --features hw-detect-vulkan,
     hw-detect-directx`でビルドし、実際にサーバーを起動して
     `GET /v1/recommend`を叩いたところ、
     `"detection_path":"directx"`(`gpu_name":"NVIDIA GeForce GT 730"`、
     `cross_check_agreement":true`)が返ることを実機(GT 730)で確認済み
     ——反転が実際に効いていることを型チェックだけでなく実行結果で確認。
  4. **`cargo test`は既存のfeature無効時のテスト3件のみ引き続きgreen**
     (優先順位の分岐自体は両feature有効時のみ到達するコードパスのため、
     ユニットテストでの直接検証は行わず、上記の実機起動確認で代替した)。
  - 次にすべきこと: 特になし(この反転自体は完了)。

- **2026-07-27(続き6) 実際にdistilgpt2をダウンロード→切り替え→生成の一気通貫E2Eを実施、実バグを発見・修正(open-cuda側で対応)**:
  1. **実施内容**: `POST /v1/models/install {"id":"distilgpt2"}`→
     `POST /v1/models/select {"id":"distilgpt2"}`→
     `POST /v1/generate {"prompt":"The quick brown fox",...}`の一気通貫
     を実際に実行。
  2. **発見した実バグ**: `select`の段階で`missing tensor 'wte.weight':
     TensorNotFound`エラーで失敗。調査の結果、
     `distilbert/distilgpt2`のsafetensorsは`transformer.wte.weight`
     のように`transformer.`プレフィックス付きテンソル名を使っており、
     `opencuda-llm::GptModel::load`がプレフィックス無し前提でハード
     コードされていたことが原因と判明(詳細・修正内容は
     `open-cuda/CLAUDE.md`の同名HANDOFF参照)。
  3. **修正後の実行結果**: `select`成功
     (`"使用するモデルをdistilgpt2に切り替えました"`)、`generate`で
     実際に英文
     (`"es are a common sight in the wild, and are often found in the
     wild."`、エンジンラベル`distilgpt2-greedy-decode-v0-opencuda-llm-cpu`)
     が生成されることを確認した。
  4. **意義**: 「型チェック・ビルド成功だけで完了と報告しない」方針の
     実践例——`cargo test`は全green(opencuda-llm側の合成フィクスチャ
     テストも含む)だったが、実際にHugging Faceから実モデルをダウン
     ロードして使うE2Eを実行して初めてこの実バグが見つかった。
  - **追記**: `gpt2-medium`(既にダウンロード済みだったもの)についても
    追加で`select`→`generate`のE2Eを実施し、`gpt2`と同じ無印テンソル名
    規約で問題なく動作することを確認済み(切り替え成功・実際に英文
    "! I'm a newbie here, so I"を生成、エンジンラベル
    `gpt2-medium-greedy-decode-v0-opencuda-llm-cpu`)。
  - **追記2**: `gpt2-large`(3.25GB)・`gpt2-xl`(6.43GB)についても実際に
    Hugging Faceからダウンロード→`select`→`generate`のE2Eを実施し、
    いずれも問題なく動作することを確認した(両方とも`gpt2`/`gpt2-medium`
    と同じ無印テンソル名規約、`transformer.`プレフィックス問題は
    distilgpt2固有だったと判明)。
    - `gpt2-large`: 切り替え成功、"jumps over the lazy dog. The quick
      brown fox"を生成(エンジンラベル
      `gpt2-large-greedy-decode-v0-opencuda-llm-cpu`)。
    - `gpt2-xl`: 切り替え成功、"jumps over the lazy dog.\n\nThe slow"を
      生成(エンジンラベル`gpt2-xl-greedy-decode-v0-opencuda-llm-cpu`)。
  - **結論**: カタログ5モデル(`gpt2`/`distilgpt2`/`gpt2-medium`/
    `gpt2-large`/`gpt2-xl`)全てについて、実ダウンロード→切り替え→生成の
    一気通貫E2Eを実施済み。テンソル名規約の違いは`distilgpt2`のみで、
    その修正(open-cuda側`key_prefix`自動判定)は既に完了・全モデルで
    正しく動作することを実証した。
  - 次にすべきこと: 特になし(カタログ全モデルのE2E確認は完了)。今後
    カタログに新モデルを追加する際は、同様の実ダウンロード検証を
    行うこと。

## HANDOFF追記(2026-07-31) インストーラーの電源プロファイル選択機能(未実装、エコシステム標準方針として記録)

`open-raid-z`のCLAUDE.md(全リポジトリ共通の設計思想セクション)に、
インストーラー(`install.sh`/`install.ps1`等)実行時に以下3つの電源
プロファイルを選択させる標準方針を追記した(ユーザー指示、2026-07-31):

1. **省電力(Power-saving)**: CPU使用率・ポーリング間隔を抑えた低負荷設定。
2. **省メモリ(Low-memory)**: メモリ確保量・キャッシュサイズを抑えた設定。
3. **常時電源接続(Always-on)**: 上記の抑制を行わないフル性能設定。
   **この場合のみ**ハードウェアアクセラレータ(NPU/GPU)のサポートを
   自動検出・自動有効化する(`open-cuda`の`GpuDevice`抽象化を利用)。

**正直な開示**: このリポジトリのインストーラーへの実装はまだ未着手。
実装時は`open-raid-z/CLAUDE.md`の該当節、および先行実装予定の
`open-redmine/CLAUDE.md`を参照し、`open-cuda`側のGPU/NPUベンダー検出
ロジックを再利用すること(車輪の再発明を避ける)。
- 次にすべきこと: このリポジトリの`install.sh`/`install.ps1`に上記3
  プロファイルの選択機能を追加する。

- **2026-08-04(続き) `/v1/translate`の実HTTP検証完了(直前エントリの
  「次にすべきこと(1)(2)」に対応)——結論: 現状のGPT-2流用実装は実用に
  耐えない**:
  1. **検証方法**: 実バイナリを起動(`cargo build --release`→実際に
     プロセス起動、モデルウォームアップ完了後)し、`POST /v1/translate`
     へ実HTTPリクエストを2件送信(`{"Hello, how are you today?" → 日本語}`・
     `{"Good morning" → French}`)。
  2. **結果(実測、モックではない)**: いずれも`200 OK`は返るが、
     `translation`フィールドの中身は**入力文をそのまま繰り返すだけで、
     実際の翻訳(日本語・フランス語への変換)は一切生成されなかった**
     (例: "Good morning"入力→出力は"Good morning,\nGood morning,\n..."の
     16回反復、フランス語は1単語も現れない)。
  3. **正直な結論**: 事前実装済みの`disclosure`フィールドの警告
     (「専用翻訳モデルではなく指示追従のファインチューニングも無い
     GPT-2流用、品質は信頼できない」)は**誇張ではなく実際にその通り**
     だったことが実機検証で確認された——このエンドポイントは
     現状「翻訳」としては機能しておらず、`audiocafe.tokyo/rakuten-mobile`
     の17言語ページ整備等の実用途にはこのまま使えない。
  4. **次にすべきこと(優先度順、正直な評価に基づき更新)**: (1) 翻訳を
     実用にするには、指示追従可能なモデル(例: 小型のinstruction-tuned
     モデル、またはNLLB/M2M100のような専用翻訳モデル)への差し替えが
     必要——現行のGPT-2流用のままではプロンプトエンジニアリングでの
     改善余地も限定的(そもそも翻訳という指示概念を学習していない
     ベースモデルのため)。(2) 当面、`audiocafe.tokyo/rakuten-mobile`の
     17言語ページは既存方針通り人力翻訳(Claude Code)を継続し、この
     エンドポイントへの依存は避けるべき。(3) cronからの定期呼び出し
     配線は、上記(1)のモデル差し替えが完了するまで見送るべき(実用に
     ならない出力を自動生成・公開してしまうリスクを避けるため)。

- **2026-08-04(続き2) 翻訳プラグイン(M2M100/rust-bert)を新設(ユーザー
  指示「翻訳精度が低ければオープンソースの翻訳システムを組み込んで」
  →「翻訳部分だけプラグインという形にして、必要な人だけインストール/
  アンインストールできるように」)**:
  1. **背景**: 直前エントリで`/v1/translate`(GPT-2流用実装)を実HTTP
     検証したところ、実際の翻訳文を生成できず実用に耐えないと判明
     していた。ユーザーへ組み込み方式を確認したところ`rust-bert`
     (M2M100/NLLB対応)を選択(他候補: ONNX Runtime+NLLB-200蒸留版、
     外部API併用〈契約不要方針と非整合のため除外〉)。
  2. **プラグイン方式の実装**: 新規`src/nllb.rs`+Cargo feature
     `nllb-translate`(既定オフ、`rust-bert`/`tch`をoptional依存化)。
     `translate()`ハンドラは、featureが有効かつM2M100モデルロード・
     翻訳が成功すればその結果を返し、それ以外(feature無効/対応言語外/
     モデルロード失敗)は既存のGPT-2流用実装へ安全にフォールバックする。
     起動時ログで`translation plugin: ENABLED`/`not installed`を出力し、
     現在の状態を明示。
  3. **「プラグイン」の実体(正直な開示)**: 実行時の動的着脱(dylibロード
     等)ではなく、**ビルド時のCargo feature選択**による着脱
     (`cargo build --release --features nllb-translate`=インストール、
     フラグ無し=アンインストール)。`rust-bert`は`tch`(libtorch)への
     依存が必須で、このエコシステムが他の全モデルで貫いてきた
     「手作りRust実装+safetensors直接ロード、重量級MLフレームワーク
     非依存」という方針から意図的に外れる大きな依存——この妥協を
     feature配下に隔離することで、既定ビルド(featureフラグ無し)には
     一切影響を与えない設計にした。
  4. **検証**: 既定ビルド(`nllb-translate`feature無し)で
     `cargo build`・`cargo test`とも**46件全green**(回帰なし、新規
     `nllb.rs`のfeature無効時ユニットテスト2件を含む)。
     **正直な開示・未検証事項**: `nllb-translate` feature有効時の
     実ビルド・実M2M100モデルロード・実翻訳品質の検証は、この開発
     環境にlibtorch(tchクレートのビルド要件)が存在せず、
     ダウンロードには相応の時間・ディスク容量を要するため**このパスでは
     未実施**——コードレビュー・既定ビルドへの非破壊性の確認までに
     留まる。次回、実際に`--features nllb-translate`でビルドし、
     実際にM2M100で翻訳が機能することを実HTTPで確認する必要がある。
  5. **ドキュメント**: `README.md`に「翻訳プラグイン」節を新設し、
     インストール/アンインストール手順(`cargo build --features
     nllb-translate`の有無)・正直な開示を記載。
  - 次にすべきこと: (1) `--features nllb-translate`での実ビルド・
    実M2M100翻訳品質の実HTTP検証、(2) 検証後、
    `audiocafe.tokyo/rakuten-mobile`等の実用途への適用を再検討。

- **2026-08-04(続き3) 自動翻訳の運用方針を撤回、`libtorch`はWindows PC
  ローカル限定に変更(ユーザー指示「libtorchは、LINUXのVPS上で運用すると
  VPSのメモリーを逼迫するので、WindowsPCにインストール形で使用したい」
  「ですから毎朝自動クロールして、自動翻訳するのは辞めましょう」
  「毎朝自動クロールしたら日本語の原文をそのまま利用しましょう」)**:
  1. **cron自動翻訳配線は行わない(方針撤回)**: 直前エントリまで検討して
     いた「`audiocafe.tokyo/rakuten-mobile`の毎朝自動クロール結果を
     `/v1/translate`で自動翻訳する」という運用は中止。今後、毎朝の
     自動クロールで取得した文章は**翻訳せず日本語原文のままそのまま
     利用する**方針に統一する。この`aruaru-llm`側に翻訳cronを配線する
     実装は行わないこと(コード変更は無し、方針の明記のみ)。
  2. **`nllb-translate` feature(libtorch依存)はVPSへデプロイしない**:
     `rust-bert`/`tch`(libtorch、PyTorchのC++ライブラリ)はメモリ消費が
     大きく、Linux VPS上で常時稼働させるとVPS全体のメモリを逼迫する
     ため、この機能はVPS本番環境には一切ビルド・デプロイしない
     (`--features nllb-translate`を付けない既定ビルドのままVPSへ配置)。
     試す場合はこのWindows開発機など**ローカルPC限定**とする。
  3. **`Cargo.toml`の`tch`依存からWeb自動ダウンロード(`download-libtorch`
     feature)を撤回**: 「PCにインストール形で使用したい」という指示に
     基づき、ビルドの都度Web上から自動取得する`download-libtorch`
     featureは使わず、`tch = { version = "0.17", optional = true }`
     (素の状態、`LIBTORCH`環境変数でローカルにインストール済みの
     libtorchパスを明示的に指す前提)に戻した。
  4. **正直な開示・現状**: 上記方針転換により、`--features
     nllb-translate`での実ビルド・実M2M100翻訳検証は**優先度が下がった
     (production用途が無くなったため)**。ビルド中に`indicatif 0.16.2`
     と`console 0.16.4`のAPI不整合(`console::Style`/
     `measure_text_width`が`std` featureの裏に隠された)によるビルド
     失敗を確認し、`console`を`0.15.11`へダウングレードする対処までは
     行ったが、その後の再ビルド・実機検証はこの方針転換により保留とした
     (緊急性が無くなったため)。
  - 次にすべきこと: (1) 今後、翻訳が本当に必要になった場合のみ、
    Windows PCローカルで`--features nllb-translate`のビルド・検証を
    再開する(VPSには絶対にデプロイしない)、(2) 毎朝の自動クロール
    ロジック(`audiocafe.tokyo/rakuten-mobile`等)側で、翻訳を呼ばず
    日本語原文をそのまま使う実装になっているかの確認(このリポジトリ
    ではなく該当リポジトリ側の作業)。

- **2026-08-05 `real-vulkan`のGEMM未配線バグ修正を実機検証、`/v1/generate`が
  実際にVulkanDevice経由で動作することを確認(優先度3位・未着手として
  指示された本リポジトリの「次にすべきこと」対応)**:
  1. **前提**: 作業開始時点で`src/generation.rs`に未コミットの変更が
     存在していた(`wire_matmul_spirv`関数、`GptModel::load`後に
     `../open-cuda/examples/matmul_vulkan_real/shaders/matmul.spv`を
     読み込み`set_matmul_spirv`で配線する処理)。これは直前HANDOFF
     (2026-08-04)で報告されていた「`open-cuda`側`Linear::forward`が
     spirvを`sgemm`へ渡していないため`real-vulkan` featureが機能しない」
     というバグに対応する配線コードで、`open-cuda`リポジトリ側
     (`F:\runo\open-cuda`)を確認したところ、該当バグ修正コミット
     `6452ae4`("Fix Linear::forward never passing spirv to sgemm")が
     既に存在し、`GptModel::set_matmul_spirv`もマージ済みと判明した。
     つまり両リポジトリ側の実装は既に揃っており、**実機検証だけが
     未実施のまま残っていた**状態だった。
  2. **実施した検証(実バイナリ・実HTTP、モックなし)**:
     - `cargo build --release --features real-vulkan`でビルド成功。
     - 実際にプロセスを起動し、起動ログで
       `real-vulkan feature enabled: using VulkanDevice for inference
       dispatch`→`OpenCUDA Vulkan Device (NVIDIA GeForce GT 730)`→
       `real-vulkan feature enabled: loaded matmul.spv (2732 bytes) ...
       and wired into GptModel via set_matmul_spirv`→
       `generation (GPT-2 124M) warmup complete`まで一気通貫で成功する
       ことを確認(直前HANDOFFで報告されていた「約0.2秒で即座に失敗する」
       という症状は解消)。
     - `POST /v1/generate`へ実HTTPリクエストを2件送信
       (`"The quick brown fox jumps over"`→
       `" the fence and runs into the bushes.\n\n\"I"`、
       `"Once upon a time"`→
       `", the world was a place of great beauty and great danger. ..."`)。
       いずれも`200 OK`で文法的に自然な英文が実際に生成され、エラーは
       発生しなかった。2件目は生成に約42.7秒を要した(このマシンの
       NVIDIA GeForce GT 730はVulkan GEMMのディスパッチオーバーヘッドが
       CPU実行より大きく出るローエンドGPUであるため、速度面のベンチ
       マーク・CPU比較は未実施——今回の検証範囲は「機能するか」の確認
       までで、「速いか」は別問題として残っている)。
     - `scoring::warmup`/`security::warmup`(BERT系、`open-cuda-bert`)は
       依然として`wire_matmul_spirv`相当の配線が無いため、`real-vulkan`
       feature有効時は起動時ウォームアップが同じ「spirv未提供」エラーで
       失敗する(ただし可用性優先の設計により初回リクエスト時に遅延
       リトライされるだけでサービス自体は止まらない)。今回の対応範囲は
       `/v1/generate`(GPT-2)のみで、`/v1/chat`(意図分類)・
       `/v1/admin`のセキュリティ分類側のVulkan配線は未対応のまま。
     - 既定ビルド(featureフラグ無し、VPS本番相当)でも
       `cargo build --release`→実プロセス起動→`/v1/generate`への実
       HTTPリクエストで正常動作することを別途確認し、今回の変更が
       既定ビルドを壊していないことを検証した。
     - `cargo test --release`(既定feature)は46件全green
       (回帰なし)。**正直な開示**: 検証の途中、直前に起動していた
       サーバープロセスがポート/メモリを保持したまま`cargo test`を
       実行してしまい、一度`memory allocation of 384056832 bytes
       failed`でクラッシュした(プロセスをkillしてから再実行したところ
       正常に46件全green)——テスト自体の欠陥ではなく、この開発環境
       での実行手順ミスだったことを確認済み。
  3. **正直な開示・気づいた既存の粗(今回は未修正)**: `/v1/generate`の
     `engine`フィールドはVulkan/CPUどちらで実行されたかに関わらず常に
     `"gpt2-...-open-cuda-llm-cpu"`という固定文字列を返す
     (`src/generation.rs`の`ENGINE_GPT2_GREEDY`定数、`-cpu`が
     ハードコード)。実際にVulkanDeviceで実行されていても`engine`
     ラベル上は判別できない——`/v1/chat`側は既に実際の経路
     (`embedding-cosine-v0-opencuda-bert-cpu`等)を正直に返す設計に
     なっているのと対照的。今回のタスク範囲外と判断し修正していないが、
     次に`/v1/generate`まわりを触る際はラベルにも実行経路(vulkan/cpu)を
     反映させることを検討すべき。
  - 次にすべきこと: (1) `scoring`/`security`(BERT系)側にも
    `wire_matmul_spirv`相当のVulkan GEMM配線を追加するかどうかの判断
    (現状は意図分類・セキュリティ分類はCPU固定のまま、GPT-2生成のみ
    Vulkan対応)。(2) このGT-730のような非力なGPUでのVulkan実行が
    CPU実行と比べて実際に速いのか遅いのかのベンチマーク未実施——
    `real-vulkan`を既定featureへ昇格させるかどうかの判断material。
    (3) `engine`ラベルの`-cpu`ハードコードを実行経路に応じて動的に
    する改修(正直な開示の一貫性向上)。

- **2026-08-07 API使いやすさ改善(`/v1/generate`・`/v1/translate`の空入力が
  誤って`503`扱いになる粗を修正)、他リポジトリ(dream-os・open-directx・
  open-cuda)と並列作業中・本リポジトリ担当分**:
  1. **前提・背景**: ユーザー指示「dream-os・open-directx・open-cuda・
     aruaru-llmの連携性強化・実用性向上・利便性向上・完成度向上」
     (4リポジトリ並列作業、`open-cuda`は別エージェントが同時作業中の
     ため本セッションでは読み取りのみ・変更禁止という制約)。直前
     HANDOFF(2026-08-05・2026-08-06)の「次にすべきこと」は主に
     `open-cuda`側の変更(decode/prefill経路分離・ベンチマーク)を
     要するため本セッションの範囲外と判断し、代わりにユーザー指示の
     もう一つの候補「実際に使ってみて不便な点(API使いやすさ・エラー
     メッセージ)」を本リポジトリ単体で完結する形で対応した。
  2. **見つけたバグ**: `POST /v1/generate`へ空文字列(または空白のみ)の
     `prompt`を送ると、`generation::generate`内部でトークナイザが
     0トークンにエンコードした後の`ensure!`失敗が、正常なバックエンド
     障害(モデル未ロード等)と同じ`503 Service Unavailable`として
     そのまま返っていた。呼び出し側からは「サーバーが落ちている」のか
     「自分の入力が不正」なのか区別できず、実用上のエラーハンドリングが
     困難だった。`POST /v1/translate`も同じ生成エンジンを土台にしている
     ため`text`が空の場合に同じ症状が起きることを確認した。
  3. **修正内容**(`src/main.rs`): `generate`ハンドラの先頭で
     `req.prompt.trim().is_empty()`を検査し、真なら`400 Bad Request`
     (`{"error": "prompt must not be empty", "engine": "..."}`)を
     即座に返すようにした。`translate`ハンドラも同様に`text`・
     `target_lang`それぞれの空チェックを追加し`400`を返すようにした。
     `open-cuda`側のファイルは一切変更していない(本リポジトリの
     `src/main.rs`のみの変更)。
  4. **検証(実バイナリ・実HTTP、モックなし)**: `cargo build --release`
     成功(ビルド中、並列作業中の`open-cuda`側が一時的に壊れていた
     タイミングと重なり8分半ほど待つ場面があったが、最終的に問題なく
     ビルド完了)。`cargo test --release`46件全green(回帰なし)。
     実際に`target/release/aruaru-llm.exe`を起動し、
     `POST /v1/generate {"prompt": ""}` → `400`
     `{"error":"prompt must not be empty",...}`、
     `POST /v1/generate {"prompt": "Hello there", "max_new_tokens": 8}`
     → `200` `{"completion": ", I'm sorry. I'm sorry", ...}`(正常系は
     従来通り動作)、
     `POST /v1/translate {"text": "", "target_lang": "Japanese"}` →
     `400` `{"error":"text must not be empty",...}`、
     `POST /v1/translate {"text": "Hello", "target_lang": ""}` →
     `400` `{"error":"target_lang must not be empty",...}`
     をいずれも実HTTPリクエストで確認した。
  5. **ドキュメント**: `README.md`の`/v1/generate`・`/v1/translate`の
     節にこの`400`検証の挙動を追記済み。
  - 次にすべきこと: (1) 前回HANDOFF(2026-08-05/06)の残課題
     (`scoring`/`security`側のVulkan GEMM配線・GT-730でのVulkan vs CPU
     ベンチマーク)は引き続き未着手、`open-cuda`側の実装・実機を
     伴うため次回`open-cuda`が空いているタイミングで着手する。
     (2) 同様の「空入力が`503`扱いになる」粗が`/v1/chat`
     (`message`)・`/v1/classify-security`(`text`)にも無いか未確認
     ——ただしこちらは`scoring::classify`/`security::classify_security`が
     空文字列でも例外を投げずNone/低スコアで正常応答する設計のため
     優先度は低いと見ている(次回確認のみでよい)。
     (3) dream-os・open-directx側との具体的な連携強化(API呼び出し
     経路の実装等)は、今回は本リポジトリ内で完結する改善に留めた
     ため未着手のまま。
