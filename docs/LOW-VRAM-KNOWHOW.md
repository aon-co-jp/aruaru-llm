# VRAMが無い/小さいPC・VPSで、AI(LLM)を安定して速く動かすノウハウ / Know-how for running LLMs on machines with no or little VRAM

2026-11-20。`aruaru-llm`・`open-cuda`で実測した内容(32GB RAM・GPUなしCPU、4コアVPS想定)。**効かなかったことも残す。** / Measured in `aruaru-llm` and `open-cuda` (32 GB RAM, CPU only; 4-core VPS in mind). **Failures are recorded too.**

## 1. 効いたもの / What worked
| 技術 / Technique | 効果 / Effect | 場所 / Where |
|---|---|---|
| 分割重み(`model.safetensors.index.json`)の自動取得・読込 / Sharded weights | 3B(2分割6GB)・7B(4分割15GB)が扱える。単一ファイルが404のときだけ分割取得、ストリーム書き込みでRAMを食わない / 3B and 7B work; streamed to disk | `aruaru-llm/src/model_catalog.rs`, `open-cuda-llm/qwen_arch.rs` |
| mmap読込 / mmap loading | 生ファイルをRAMへコピーしない(ピーク半減) / no raw copy in RAM | `qwen_arch.rs` `map_file` |
| FP8重み(E4M3、行ごとスケール、LUT復元、列ブロック融合GEMM) / FP8 weights | メモリ約1/4。1.5Bでppl 7.08→7.27。7Bが32GB機に載る / ~1/4 memory; 7B fits | `opencuda-blas` `sgemm_fp8_weight`, `QwenModel::load_fp8` |
| 混合精度(`QWEN_KEEP_F32="1,24,26"`) / Mixed precision | 壊れやすい層だけf32: ppl 7.267→7.096、速さ・メモリはほぼ同じ / near-lossless | `qwen_arch.rs` |
| FP8の高速化 / FP8 speed | 量子化の二分探索化+並列化、融合GEMM、スケールの巻き上げ: 読込340→21秒、生成1067→27.8秒(f32は25秒) / load 340→21 s, gen 1067→27.8 s | `opencuda-blas` |
| 小さい行列積の直列化 / Serial small GEMM | 1トークンずつの生成でrayon並列がかえって約1秒/語の遅延を生んだ。小さい行列は直列に / rayon on tiny GEMMs cost ~1 s/token | open-cuda `480718f` |
| プロンプト長の上限(64トークン)・知識文脈を短く / Cap prompt and context | VPSの4コアでGenerateが150秒超→8トークン6秒、48トークン22.8秒 / 150 s+ → 6 s | aruaru-llm `4805563` |
| 同時実行上限・サイト別公平キュー・タイムアウト(429/503/504) / Admission control | 共有基盤1つを多数のサイトで安全に共有 / safe sharing | aruaru-llm(VPS反映済み) |

## 2. 効かなかったもの(試して分かった) / What did not work
- **後付けMLA(PCA圧縮)**: KV25%減でppl 7.1→12.7、50%減で崩壊。MLAは学習時から入れる設計でないと効かない。 / Post-hoc MLA breaks quality; it must be trained in.
- **学び直し無しの層スキップ(折り畳み)**: 最大約14%速いが、別の文で4〜50%悪化。1つの文で選ぶと過学習する。 / Layer skipping without retraining: up to ~14% faster but 4-50% worse on held-out text.
- **焼きなまし・SB(東芝/富士通)**: 28層の小さな問題では貪欲法と差が出ない。組合せが爆発する問題(層×エキスパート×ビット数)で再挑戦する。LLM推論そのものを速める道具ではない。 / No edge over greedy at 28 layers; retry on exploding search spaces.
- **FP8のSIMD追加**: 1語ずつの生成はメモリ速度で決まり、FP8は既にf32の約1.1倍の時間。見込みが小さく未実施(未検証)。 / Not tried; decode is memory-bound.

## 3. VRAMなし・小VRAMの定石 / Rules of thumb
1. 重みはmmapで読み、必要ならFP8へ。大事な層(最初・最後・感度の高い層)だけ元の精度で残す。 / mmap, FP8, keep sensitive layers full precision.
2. 小さい行列積は並列化しない。並列化は大きい行列だけ。 / Do not parallelize tiny GEMMs.
3. プロンプトと文脈を短く上限し、必ずタイムアウトを付ける(失敗しても本体が止まらない設計)。 / Cap prompt/context; always set timeouts.
4. 共有基盤は1回だけ入れ、サイトごとに入れない。同時要求は上限+公平キューでさばく。 / Install the shared base once per VPS; cap concurrency with fair queues.
5. 重い測定・デバッグは手元PC(WSL)で。VPSは配備と確認だけ(負荷試験で本番を遅くしない)。 / Heavy tests on the local PC, not the VPS.
6. 数字は必ず実測して残す。CI成功は動作確認ではない。 / Measure; CI green is not proof.

## 4. 次の候補(未了) / Next
- 層・エキスパートごとのビット配分の自動決定(線形/整数計画 GEMQ・BitsMoE 型 → 大きなモデルで焼きなまし/SB)。
- MoE(DeepSeek)の実チェックポイントでの遅延読み込み通し(`deepseek_arch.rs`は実装済み、実物は未確認)。
- 7BのFP8を高速カーネルで再測定(約8秒/語は旧カーネルの値)。
- 量子化(4/8ビット整数)・KV量子化の検討、`pl2pl llm setup`での分割モデル選択の通し確認。

## 5. どのリポジトリで何に使うか(展開・応用・再設計の入口) / Where to apply (rollout map)
| リポジトリ / Repo | 使う技術 / Apply | 状態 / Status |
|---|---|---|
| aruaru-llm | 分割重み・FP8・混合精度・プロンプト上限・同時実行上限 / all | 実装済み(v0.2.6)。FP8を既定の選択肢にする設計は未了 / FP8 as a selectable default: open |
| open-cuda / open-cpu | FP8 GEMM・小行列直列化・mmap。VRAM無しの既定経路(CPU)を最優先に保つ / CPU path first | 実装済み。4/8bit整数・KV量子化は未了 |
| pl2pl | `llm setup`でRAMに合わせて3B/1.5B/0.5Bを選ぶ(分割モデル対応版を取得)。FP8/混合精度の選択を追加する / pick model by RAM | 3B候補は既にあり。FP8選択は未了 |
| open-english / maid-cafe-english / maid-cafe-tokyo | 答えられないときの補完や要約は、短い文脈・上限・タイムアウト必須。VPSで重い推論を直接回さない / short context, timeouts, no heavy VPS inference | 方針。補完機能は未着手 |
| pdf-r2l-rs / aruaru-IME / open-media | 端末側の推論は小さいモデル+FP8/mmap。メモリが足りなければ自動で小さいモデルへ下げる / auto-downgrade | 方針(各リポジトリで実装時に参照) |
| version-control / aruaru-search | LLM要約は`llm_timeout_secs`内に収まる高速な推論先へ。収まらなければ要約を省いて動作継続 / skip summaries on timeout | 実装済み(VPSは遅いので要約は実用外) |
| 共通 / All | 重い試験は手元PC(WSL)、VPSは配備と確認だけ。数字は実測して記録 / measure locally | 常時 |
再設計の優先順 / Redesign order: (1) 起動時にRAM/VRAMを調べ、モデル・精度(f32/FP8/混合)・同時実行数を自動選択する共通部品(`aruaru-llm`の`ModelsCompatible`を拡張)、(2) 4/8bit整数とKV量子化、(3) MoE遅延読み込みの実チェックポイント通し、(4) 大きなモデルでのビット配分の自動探索(焼きなまし/SB)。

## 6. 精度の階段(FP8〜大きな桁、整数)と、端末ごとの使い分け / Precision ladder and device tiers (2026-11-20)

### 6.1 正直な見立て / Honest assessment
- **メモリが足りない環境(VRAM無しのVPS・PC・タブレット・スマホ)で効くのは「桁を減らす」方向**: FP8(実測済み)→ INT8 → INT4(4bit)。FP16/BF16は「FP8にすると品質が落ちる層」の保護先として使う。 / On memory-poor devices the gain comes from **fewer** bits: FP8 (measured) -> INT8 -> INT4; FP16/BF16 protect layers that FP8 hurts.
- **FP32は基準(今のf32経路)。FP64・FP128・FP256…FP4096は推論の品質をほとんど上げず、メモリと時間が桁違いに増える**(FP64で重みが2倍、FP128で4倍…。ハードウェア非対応でソフトウェア演算になり数十〜数千倍遅い)。そのため**推論の本番経路には使わない**。 / FP32 is the baseline. FP64 and wider do not improve inference quality noticeably but cost 2x-100x+ memory/time (software-emulated beyond FP64), so **never in production inference**.
- **広い桁が役に立つ所**: (1) 試験の「正解(オラクル)」: 同じ行列積をFP64/FP128で計算し、FP8/INT4/f32との誤差を測る。(2) 総和(累積)だけを広くする(FP8の重みでも累積はf32、長い文脈ではKahan/ペア和)。(3) 数値の不安定な所(softmax・RMSNormの分散)のみ必要時にFP64。 / Wide formats are useful as **test oracles**, for **wide accumulation only**, and for unstable spots (softmax, norm variance) when needed.
- 「大きなLLMでも活かす」=重みを小さい桁で持ち、**層ごとに桁を変える**(混合精度)方向で広げる。大きいモデルほどFP8/INT4の恩恵が大きい(7Bが32GB機に載った実績)。 / For big LLMs the lever is per-layer bit width, not wider floats.

### 6.2 共通の「精度の型」にする設計(未了・再設計) / Design: one precision enum (open)
`Precision = F64Ref | F32 | F16 | BF16 | Fp8E4M3 | Fp8E5M2 | Int8 | Int4`。層ごと・エキスパートごとに1つ選ぶ。`F64Ref`は試験専用(本番では選べない)。選び方は、(a) 感度測定(層を1つずつ低精度にしてppl変化を測る)→(b) 貪欲法/整数計画で「メモリ予算の中で品質最大」→(c) 大きなモデルでは焼きなまし/SB。実装は`QWEN_KEEP_F32`(済み)を一般化する形で、`open-cuda-llm`と`opencuda-blas`に置く。

### 6.3 端末ごとの目安 / Device tiers (measured only for the first two rows)
| 端末 / Device | 目安 / Guide | 状態 / Status |
|---|---|---|
| VRAM無しVPS(4コア・3.6GB) | 0.5B〜1.5B、FP8、プロンプト64トークン、同時実行1〜2、タイムアウト必須 / small models, FP8, caps | 実測済み |
| VRAM無しPC(32GB) | 〜3Bはf32/FP8、7BはFP8(約15GB)で動作(遅い) / up to 7B in FP8 | 実測済み |
| タブレット・スマホ | 0.5B〜1.5BのINT4/INT8が現実的。メモリが足りなければ自動で小さいモデルへ / INT4/INT8 small models, auto-downgrade | **未実装・未測定**(Androidは`librporea_ffi.so`の起動のみ実機確認) |
| 共有基盤VPS | 1回だけ設置、サイトは呼ぶだけ、公平キュー / install once | 実装済み |

### 6.4 次に作る順(確認なしで進める) / Next steps
1. `Precision`型と、層ごとの精度表(JSON、RS-JSONで読み書き)を`open-cuda-llm`に追加。`QWEN_KEEP_F32`をその一例にする。
2. INT8(行ごとスケール)・INT4(ブロックスケール)重みのGEMMをCPUで実装し、FP8と同じ手順(ppl・速さ・メモリ)で実測。
3. `F64Ref`オラクルを試験に追加(小さい行列で、FP8/INT8/INT4の誤差の上限を自動検査)。
4. 起動時にRAM/VRAMを調べ、モデル・精度表・同時実行数を自動選択(`ModelsCompatible`を拡張)。
5. スマホ(Android)で0.5Bを実機測定。

## 実測追記 2026-11-20: INT8/INT4 重み / Measured: INT8/INT4 weights
日本語: Qwen2.5-1.5B・CPUのみ。迷い度(評価文/別の文)・生成秒: f32 7.08/–/25、FP8 7.27/12.17/29.8、INT8 7.37/12.94/27.2、INT4(128グループ) 10.55/15.37/29.0。INT4を1行1グループにすると26.4に悪化。除算を畳み込む前のINT8は81秒だった。メモリは重みがINT8で約1/4、INT4で約1/8。`QWEN_PRECISION="int4,1:f32"`で層ごとに指定(open-cuda 1b1f1b4)。3B/7Bは未測定。
English: Qwen2.5-1.5B, CPU only. Perplexity (eval / held-out) and generation seconds: f32 7.08/–/25, FP8 7.27/12.17/29.8, INT8 7.37/12.94/27.2, INT4 (128-element groups) 10.55/15.37/29.0. One group per row made INT4 much worse (26.4). INT8 took 81 s before the per-element division was folded away. Weight memory is about 1/4 (INT8) and 1/8 (INT4). Set per layer with `QWEN_PRECISION="int4,1:f32"` (open-cuda 1b1f1b4). 3B/7B not measured yet.

## 2026-11-20 追記: 起動時の精度の自動選択 / Startup precision auto-selection
日本語: 環境変数 `ARUARU_LLM_PRECISION` を追加(`qwen_generation.rs`)。未設定=従来どおりf32(挙動不変)。`auto`=総RAMの8割を予算に、`open-cuda-llm`の`PrecisionTable::auto_for_memory`でf32→全層INT8→一部INT4を選ぶ(重みはbf16配布が多いのでf32換算はファイル合計の2倍と近似)。`int8,1:f32`形式なら指定どおり。コンパイルは確認済みだが、**実サーバーでの通し確認は未了**。KVキャッシュ量子化は未着手。
English: New env var `ARUARU_LLM_PRECISION` (`qwen_generation.rs`). Unset = f32 as before (no behaviour change). `auto` uses 80% of total RAM as the budget and lets `PrecisionTable::auto_for_memory` pick f32, all-INT8 or partly INT4 (f32 size is approximated as twice the file total, since weights usually ship as bf16). A table like `int8,1:f32` is used as given. It compiles, but **an end-to-end run on a real server is not done**. KV-cache quantization is not started.
