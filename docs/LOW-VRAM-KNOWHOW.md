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
