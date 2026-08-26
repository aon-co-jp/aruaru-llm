# このフォルダについて / About this folder

`aruaru-llm.iss` はこのインストーラーを**作るための** [Inno Setup](https://jrsoftware.org/isinfo.php)
ビルドスクリプトです。`aruaru-llm-installer.exe` はこのフォルダ内に実体として置いてあります
(ユーザー指示により、ローカルビルドしたバイナリを直接コミット)。CPU版・GPU版
(open-cuda Vulkan/DirectXバックエンド組み込み)の両方が同梱されており、GPU版は
インストール時の`installgpu`タスク(既定は未チェック)で選択できます。

**⬇ 今すぐダウンロード**: [aruaru-llm-installer.exe](aruaru-llm-installer.exe)

**正直な開示**: このファイルはビルド成果物であり、`aruaru-llm.iss`やソースコードを変更しても
自動的には更新されません(手動での再ビルド・再コミットが必要)。[常に最新版が欲しい場合は
GitHub Releasesも参照してください。](https://github.com/aon-co-jp/aruaru-llm/releases/latest)

---

`aruaru-llm.iss` is the [Inno Setup](https://jrsoftware.org/isinfo.php) build script used to
**produce** this installer. `aruaru-llm-installer.exe` itself is committed directly into this
folder (per explicit user instruction, a locally-built binary is committed as-is). Both the
CPU build and the GPU build (open-cuda Vulkan/DirectX backends) are bundled inside; the GPU
variant can be selected via the `installgpu` task at install time (unchecked by default).

**Honest disclosure**: this file is a build artifact. It does not update automatically when
`aruaru-llm.iss` or the source code changes (a manual rebuild + recommit is required). [If you
always want the latest version, also check GitHub Releases.](https://github.com/aon-co-jp/aruaru-llm/releases/latest)
