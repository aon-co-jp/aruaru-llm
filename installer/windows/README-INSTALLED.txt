aruaru-llm — installed / インストール完了

------------------------------------------------------------------
English
------------------------------------------------------------------
aruaru-llm is a self-contained AI backend (no external LLM API
contract required). It listens on http://127.0.0.1:4600 by default.

GPU acceleration (open-cuda + open-directx "SET"):
  This installer can optionally install a GPU-accelerated build of
  aruaru-llm that has open-cuda's Vulkan backend (and, on Windows,
  its DirectX 12 backend) compiled in as a single combined binary —
  this is what "using aruaru-llm together with open-cuda/
  open-directx" means in practice: they are Rust libraries linked
  directly into this one executable, not separate apps to install.

  HONEST DISCLOSURE: enabling GPU acceleration does NOT always mean
  "faster". On this project's own development machine (an older,
  low-end NVIDIA GT 730), the GPU-accelerated build was measured to
  be SLOWER than the default CPU-only build for GPT-2-class
  inference (dispatch overhead dominates for such small compute
  jobs). Whether it helps on YOUR GPU depends on the model — modern
  mid/high-end GPUs are much more likely to benefit. Use the
  in-app "Recommend LLM" hardware check after installing to see
  what this app detects on your machine, and feel free to try both
  builds.

Model weights: GPT-2/distilgpt2 weights (hundreds of MB) are NOT
bundled (license/size reasons). To get them, no manual commands or
API calls are needed — just run aruaru-llm.exe, then open
http://127.0.0.1:4600/ in your browser and click the
"🧠 Recommend LLM" button; it detects your hardware and downloads a
matching model automatically.

------------------------------------------------------------------
日本語
------------------------------------------------------------------
aruaru-llmは外部LLM APIとの契約が不要な自己完結型AIバックエンドです。
既定で http://127.0.0.1:4600 で待ち受けます。

GPUアクセラレーション(open-cuda + open-directxの「SET」構成):
  このインストーラーでは、任意でopen-cudaのVulkanバックエンド
  (Windowsではさらに内蔵のDirectX 12バックエンドも)を1本の実行
  ファイルへコンパイルして組み込んだ「GPUアクセラレーション版」の
  aruaru-llmを選んでインストールできます——「aruaru-llmをopen-cuda/
  open-directxと一緒に使う」とは、実際にはこれらがRustのライブラリ
  としてこの1本のexeへ直接リンクされている、という意味です(別々の
  アプリとして個別インストールするものではありません)。

  正直な開示: GPUアクセラレーションを有効にしても「必ず速くなる」
  わけではありません。この開発プロジェクト自身の開発機(古い
  ローエンドのNVIDIA GT 730)での実測では、GPT-2クラスの推論に
  ついてはGPUアクセラレーション版の方が既定のCPU版より**遅く**
  なることが確認されています(このような小さな計算にはGPU
  ディスパッチのオーバーヘッドの方が支配的になるため)。お使いの
  GPUで効果があるかは機種次第です——中〜上位クラスのGPUほど恩恵を
  受けやすい傾向にあります。インストール後、アプリ内の「おすすめ
  LLM」機能でお使いの端末の検出結果を確認し、両方のビルドを
  実際に試してみることをお勧めします。

モデル重み: GPT-2/distilgpt2の重み(数百MB)はライセンス・サイズ上の
理由からこのインストーラーには同梱されていません。取得に難しい手順や
コマンド入力は不要です——aruaru-llm.exeを起動したまま、ブラウザで
http://127.0.0.1:4600/ を開き、「🧠 おすすめLLM」ボタンを押すだけで、
お使いの端末に合ったモデルを自動的に検出・ダウンロードします。
