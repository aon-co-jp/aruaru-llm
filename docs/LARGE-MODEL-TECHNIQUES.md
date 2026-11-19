# 大きなAIを小さなパソコンで動かす技術の調査 / Techniques for running big LLMs on small machines

調査日 / Date: 2026-11-19〜20。日英の検索(Google)と GitHub で確認した。**未確認の主張は「未検証」と書く。** / Found by English and Japanese web and GitHub searches. **Unverified claims are marked as such.**
判定の基準 / Verdict question: 「CPUだけ・メモリ32GBのPCで効くか」 / "Does it help CPU-only inference on a 32 GB PC?"

| 技術 / Technique | 何をするか / What it does | 判定 / Verdict | 出典 / Sources |
|---|---|---|---|
| 東芝 Simulated Bifurcation(SB) | 組合せ最適化(イジング/QUBO)を速く解く。LLM推論そのものは速くしない。 / Fast solver for combinatorial optimization (Ising/QUBO); does not speed up LLM inference itself. | 直接は効かない。「どの層を飛ばすか・何ビットにするか」を選ぶ道具になり得るが、当リポジトリの実測(1.5B、28層)では貪欲法と大差なし。 / Not directly. Could pick layers/bits; in our 28-layer test it did not beat greedy. | [Azure Quantum blog](https://azure.microsoft.com/en-us/blog/quantum/?p=9831), [SB library docs](https://simulated-bifurcation-algorithm.readthedocs.io/en/latest/) |
| 富士通 Digital Annealer | QUBO形式の最適化を専用/クラウドで解く。 / Solves QUBO problems (service/hardware). | LLMの量子化・層配置での事例は見つからなかった。 / No LLM quantization/placement use found. | [富士通 技術資料](https://www.fujitsu.com/jp/documents/about/research/techintro/3rd-g-da.pdf) |
| Tree-Ising | LLMのBF16重みの離散的な修復をQUBOにして、焼きなましやSBで解く。論文の自己申告値(未検証)。 / Casts discrete repair of BF16 weights as QUBO solved by annealing or SB; self-reported numbers (unverified). | 推論の高速化ではなく重みの編集。実装は未確認。 / Weight editing, not speed-up; no implementation checked. | [hyper.ai 2611.14210](https://hyper.ai/en/papers/2611.14210) |
| Model Folding(ICLR 2025) | 似たニューロンをまとめて、データ無しでモデルを小さくする。 / Merges similar neurons, data-free. | LLM(LLaMA-7B)での結果は論文側の主張。LLM経路が校正データを使うかは**未検証**。当リポジトリでは未実装。 / LLM results are the paper's claim; whether the LLM path uses calibration data is **unverified**. Not implemented here. | [arXiv 2502.19159 系列(Sliding-Window Merging)](https://arxiv.org/abs/2502.19159), [model-folding-universal](https://gh.r-mj.com/nanguoyu/model-folding-universal) |
| Layer Collapse / GPTailor | 層をまとめる・切り貼りして深さを減らす。 / Merge or stitch layers to reduce depth. | 学び直し無しだと効果が小さい(当リポジトリの層スキップ実測: 最大約14%高速、別の文で4〜50%悪化)。 / Small gain without retraining (our skip test: up to ~14% faster, 4-50% worse on held-out text). | [arXiv 2402.11187](https://arxiv.org/abs/2402.11187), [arXiv 2506.20480](https://arxiv.org/abs/2506.20480) |
| DeepSeek MLA | KVキャッシュを低ランクに圧縮。学習時から組み込む設計。 / Low-rank KV-cache compression, designed in at training time. | **後付けは不可**(当リポジトリ実測: KV25%減でppl 7.1→12.7、50%減で崩壊)。長い文脈で効く。 / **Cannot be added afterwards** (ours: 25% less KV gave ppl 7.1→12.7; 50% broke). Helps long contexts. | [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) |
| DeepSeek MoE(共有+細粒度エキスパート) | 1トークンに使うエキスパートを一部だけにする。 / Uses only some experts per token. | 遅延読み込みと相性が良い(`open-cuda` の `deepseek_arch.rs` に実装済み、実在チェックポイントでの通しは未確認)。 / Fits lazy loading (implemented in `open-cuda`; real-checkpoint run unverified). | [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) |
| FP8(DeepSeek-V3、E4M3、128x128ブロック) | 重み・活性を1バイトで持つ。論文では主に学習用。 / 1-byte weights/activations; mainly training in the paper. | 推論での重みFP8は有効: メモリ約1/4、1.5Bでppl 7.08→7.27、7Bが32GB機に載った。ただしCPUではf32より約2.8倍遅い。 / Weight-only FP8 works: ~1/4 memory, ppl 7.08→7.27 on 1.5B, 7B fits a 32 GB PC; ~2.8x slower than f32 on CPU. | [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437), [hardware co-design paper](https://arxiv.org/pdf/2505.09343) |
| 混合精度ビット配分(GEMQ, BitsMoE) | エキスパート・層ごとにビット数を線形/整数計画で決める(QUBOではない)。 / Per-expert/layer bit allocation by LP/ILP (not QUBO). | 有望。次の実装候補。 / Promising; next implementation candidate. | [GEMQ 2605.23078](https://arxiv.org/abs/2605.23078), [BitsMoE 2606.00079](https://huggingface.co/papers/2606.00079) |
| mmap・遅延読み込み | ファイル全体をRAMへ載せず、必要な所だけOSが読む。 / Let the OS page in only needed parts. | 有効(当リポジトリで実装、3B/7Bの読み込みで確認)。 / Works (implemented here; checked with 3B/7B). | `open-cuda` の `qwen_arch.rs` |

## 結論 / Conclusion
- **効いたもの / What worked**: 分割重み対応、mmap、FP8重み(メモリ1/4)。 / sharded weights, mmap, FP8 weights (1/4 memory).
- **効かなかったもの / What did not**: 後付けMLA、学び直し無しの層スキップ。 / post-hoc MLA, layer skipping without retraining.
- **次に試す / Next**: 混合精度ビット配分、FP8のCPU高速化。 / mixed-precision bit allocation, faster FP8 on CPU.
