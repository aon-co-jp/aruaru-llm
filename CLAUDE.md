# 開発方針＆開発環境ルール(aruaru-llm)

作業ドライブは`F:\open-runo`。この節は[`open-raid-z`](https://github.com/aon-co-jp/open-raid-z)の
`CLAUDE.md`を正本とし、各プロジェクトへコピーして同期する方針に準じる。
GitHubリポジトリ: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm)。

> ⚠️ **正直な開示(最重要、2026-07-21更新)**: このリポジトリ名は「LLM」を
> 冠しているが、**自己回帰デコーダによる文章生成(対話生成としての
> 「LLM」の能力)はまだ実装していない**。2026-07-21以降は、`open-cuda`の
> `opencuda-bert`クレート(multilingual-e5-small、MITライセンス、日本語
> 含む100言語対応)で実際に文を埋め込みベクトルへ変換し、意図ごとの
> 代表例文とのコサイン類似度で分類する**エンコーダベースの意味的類似度
> 分類**へ移行した(旧: 固定語彙へのbag-of-wordsドット積による単純な
> キーワードマッチング)。意味理解の質は大きく向上したが、これは検索・
> 分類向けのエンコーダであり、対話文を生成する能力ではない。
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

`e-gov.info`と同じ方針(2026-07-18更新のPoem判断基準に基づく): 単純な
HTTPサービスとして`poem`クレートを直接利用する。DB非依存・1バイナリ完結。

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "e-gov.info"(任意)}` を
  受け取り `{"reply": "...", "engine":
  "embedding-cosine-v0-opencuda-bert-cpu", "matched_intent": "..."}` を
  返す。`tenant`は未登録でも応答は返す(可用性を落とさないため)。
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
