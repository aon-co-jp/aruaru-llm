# aruaru-llm

> 🎯 **aruaru-db × RPoem SET policy (2026-08-29)**: `aruaru-db`'s
> `CLAUDE.md` opening note explains that aruaru-db only delivers "no
> REST API needed, compatible with WunderGraph Cosmo's paid
> Enterprise tier" when paired (SET) with RPoem. This repo
> (aruaru-llm) is instead paired with `open-cuda` as an independent
> AI inference layer (see the "no-contract independent AI" section)
> and is a separate concern from that REST/API-key work — but any
> future expansion of its own dynamic tenant registration
> (`src/tenants.rs`) should still ask "does this strengthen the
> aruaru-db+RPoem SET value?" before adding REST alternatives.

*日本語*: [README.md](README.md) ·
*English*: this is the original ·
*Other languages*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **Recent update (2026-08-26)**: Changed the `/v1/generate-with-search`
> prompt assembly from plain concatenation to a QA-style format
> (`web_search::build_search_augmented_prompt`: `"Search results:
> ...\nQuestion: {prompt}\nAnswer:"`). The idea is that GPT-2's
> pretraining corpus contains a lot of Q&A-formatted text, so this
> pattern may make greedy decoding more likely to continue in a way
> that actually uses the preceding search context — **this is an
> attempted improvement, not a guarantee** that the model now reliably
> uses search results (GPT-2/distilgpt2 have no instruction-tuning).
> Switching to an instruction-tuned model and adding a summarization
> step were investigated but not completed this session, so both are
> deferred. **This change only touches prompt string assembly** —
> open-directx/open-cuda (the GPU inference *speed* layer) were not
> touched, since they don't affect response *accuracy*. See
> [CLAUDE.md](CLAUDE.md)'s 2026-08-26 entry for details.
>
> 📌 **Recent update (2026-08-10)**: Wired `open-cuda`'s new `GptModel::
> generate_with_repetition_penalty` (CTRL-style repetition penalty) into
> `/v1/generate`, **enabled by default** (`ARUARU_LLM_REPETITION_PENALTY`
> env var, default `1.3`; set to `1.0` to restore the old no-penalty
> behavior). This directly addresses a known GPT-2 base-model
> degeneracy — endless repetition of the same string (e.g. a user
> reported it looping "Student: Hello" forever while using
> `open-english`) — since the base model has no dialogue fine-tuning.
> Verified on real GPT-2 124M weights on the `open-cuda` side: without
> the penalty the loop actually reproduces; with `penalty=1.3` it stops
> and produces grammatically natural conversational text instead. Also
> verified via a live HTTP request:
> `POST /v1/generate {"prompt":"...Student: Hello\nTrainer:",
> "max_new_tokens":24}` returned a non-repeating reply. At `penalty=1.0`
> the output is byte-identical to the existing `generate()` API, so no
> other tests regress. See [CLAUDE.md](CLAUDE.md) for details.
>
> *日本語*: `open-cuda`側`GptModel::generate_with_repetition_penalty`
> (CTRL方式の繰り返しペナルティ)を`/v1/generate`へ配線し、既定で有効化
> した(`ARUARU_LLM_REPETITION_PENALTY`環境変数、既定値`1.3`)。対話
> ファインチューニング無しの素のGPT-2貪欲デコードが陥る既知の劣化モード
> (同一文字列の無限ループ)への根本対応。詳細は[CLAUDE.md](CLAUDE.md)参照。

> 📌 **Recent update (2026-08-08)**: Wired an opt-in (default-off)
> MLA-style KV cache compression path (`open-cuda`'s DeepSeek-V3-inspired
> implementation, `GptModel::enable_mla_kv_compression`) into
> `/v1/generate` via `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`. For GPT-2
> 124M this is head_dim=64 -> d_c=16 (75% smaller per-token KV storage),
> computed from the actual loaded model's config, not a device-specific
> feature (works on CPU as well as `real-vulkan`). **Honest disclosure**:
> the down/up-projection matrices are randomly initialized (untrained),
> so this is lossy compression, and a real end-to-end test against the
> real GPT-2 124M weights showed visibly degraded/more repetitive output
> compared to the uncompressed path — which is why it defaults off. See
> [CLAUDE.md](CLAUDE.md) for the measured before/after generations.

> 📌 Pending task (2026-08-06): a plan exists to incorporate Toshiba SBM / DeepSeek techniques. See [CLAUDE.md](CLAUDE.md) for details.

> 📌 **Recent update (2026-08-07)**: verified via a real running binary +
> live HTTP requests that `/v1/chat` and `/v1/classify-security` do
> **not** suffer the "empty input → 503" bug previously fixed for
> `/v1/generate` and `/v1/translate` — both correctly return 200 for
> empty input. No code change was needed. See [CLAUDE.md](CLAUDE.md) for
> details.

> **Updated 2026-07-25**: The dev-policy file (`CLAUDE.md`) heading was
> renamed from "Development Policy & Dev Environment Rules" to
> "Design Philosophy & Development Policy & Dev Environment Rules",
> to more clearly separate the project's design philosophy (what we
> value), development policy (how we work), and dev environment rules
> (concrete operational conventions). See `CLAUDE.md` for details.


A shared "AI chat commerce" response service for the `aruaru` ecosystem
(aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo, etc). Instead of each
site implementing its own chat-reply logic, they call this single HTTP
service — centralizing the one place that needs to change when real LLM
inference is eventually wired in.

> ⚠️ **Honest disclosure (important, updated 2026-07-25)**: as of
> 2026-07-25 this service integrates `open-cuda`'s `opencuda-llm` crate
> (real trained GPT-2 124M weights, `openai-community/gpt2`), so
> `POST /v1/generate` now performs **actual autoregressive text
> generation** — the "no autoregressive generation" claim below no longer
> applies to that endpoint. That said, **GPT-2 124M is a small, 2019-era
> model and is not comparable to modern commercial LLMs like GPT-4** in
> capability or knowledge. This is a demonstration that self-contained
> generation works without an external LLM API contract, not a claim of
> state-of-the-art quality — output is often grammatically fluent English
> but is not guaranteed to be factually accurate (it can hallucinate).
> `POST /v1/chat` (intent classification via `opencuda-bert`'s sentence
> embeddings + cosine similarity, since 2026-07-21) remains a separate,
> lightweight, fast path for canned replies — intentionally not merged
> with generation. See [CLAUDE.md](CLAUDE.md) for details and rationale.

## Paired ("SET") with open-cuda

Depends on [`open-cuda`](https://github.com/aon-co-jp/open-cuda)'s
`opencuda-core`/`opencuda-cpu`/`opencuda-blas`/`opencuda-bert` crates via a
path dependency. On every `/v1/chat` request, `opencuda-bert` runs
multilingual-e5-small's forward pass (calling into `opencuda-blas`'s real
GEMM/Attention kernels on `opencuda_cpu::CpuDevice`) to embed the message,
then compares it via cosine similarity against each intent's cached
representative embedding. This is a real runtime call through open-cuda's
compute pipeline, not just a `Cargo.toml` reference — verified by actually
starting the server and exercising `POST /v1/chat`.

That said, this is not real neural LLM inference (dialogue generation) —
only the encoder forward pass; the autoregressive decoder remains
unimplemented. GPU-specific fast paths (`GemmPath::CuBlas`/`RocBlas`/
`OneMkl`) are still stubbed (CPU and generic-Vulkan paths are implemented).
See open-cuda's `CLAUDE.md` HANDOFF section for details.

**2026-07-25 update (availability fallback)**: if
`models/multilingual-e5-small/` (470MB+) is missing or fails to load, this
service now automatically falls back to the original bag-of-words dot
product (`src/bow_fallback.rs`) instead of failing requests outright. The
`/v1/chat` response's `engine` field always honestly reports which path was
actually used (`embedding-cosine-v0-opencuda-bert-cpu` or
`bow-dotproduct-v0-opencuda-cpu-fallback`) — classification quality is
noticeably lower on the fallback path (keyword matching, not semantic
understanding).

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(optional)}` → `{"reply": "...", "engine":
  "...", "matched_intent": "..."}` (intent classification, lightweight/fast canned replies)
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens": 16(optional, default 16, capped at 128), "tenant": "..."(optional)}`
  → `{"completion": "...", "engine": "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`
  (real autoregressive generation via GPT-2 124M weights — heavier but genuine.
  **Repetition penalty defaults to `1.3`** — `ARUARU_LLM_REPETITION_PENALTY`
  env var to override, `1.0` disables it — to prevent endless-repetition
  loops. English prompts
  recommended, since GPT-2's BPE vocabulary is trained mostly on English text. Example, verified
  end-to-end over real HTTP: `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out of your"`)
- `GET /v1/models/catalog` — GPT-2-compatible models available to install
  (`gpt2`/`distilgpt2`/`gpt2-medium`/`gpt2-large`/`gpt2-xl`, the last added
  2026-07-27), which ones are already installed, and the currently active
  model directory.
- `POST /v1/models/install` / `POST /v1/models/select` — download a catalog
  model from Hugging Face, and hot-swap the active model without a process
  restart.
- `GET /v1/recommend` (added 2026-07-27) — detects hardware (VRAM) via
  `open-cuda` (Vulkan) or `open-directx` (DXGI) and returns a recommended
  GPT-2-family model size, without downloading anything.
- `POST /v1/recommend-and-download` (added 2026-07-27, backs the
  "Download recommended LLM" button) — detects hardware → picks a
  recommended size → downloads it from Hugging Face if not already present
  (idempotent) → hot-swaps `/v1/generate` to use it. Returns
  `{"recommendation": {...}, "already_installed":bool,
  "switched_to_recommended":bool, "message_ja":"..."}`.
- `GET /` (added 2026-07-27) — minimal static HTML UI (`static/index.html`,
  no framework) with one "Download recommended LLM" button, progress
  display, and a generation-test panel once switched.
- `POST /v1/transcribe` (added 2026-08-29) — `{"pcm_f32_base64": "...",
  "sample_rate": 16000, "language": "auto"(optional), "tenant": "..."(optional)}`
  → `{"transcript": "...", "language": "...", "engine": "...", "disclosure": "..."}`.
  `pcm_f32_base64` is 16 kHz mono f32 PCM as a little-endian byte array,
  base64-encoded (the format open-english's `blobToPcm16k()` produces).
  Speech recognition via whisper.cpp — spawns a **prebuilt `whisper-cli`
  subprocess** (no Cargo feature; drop the CLI binary and a GGML model in
  place, or set `ARUARU_LLM_WHISPER_CLI` / `ARUARU_LLM_WHISPER_MODEL`).
  Without them it returns `503` + a link to whisper.cpp releases.
  `sample_rate ≠ 16000` / bad base64 / audio over 10 minutes all return
  `400`. See the "Speech recognition" section below.
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` — tenant registration management (`x-admin-token` header auth)
- `GET /healthz` — health check
- `GET /v1/runtime` — reports which compute backend is actually in use;
  includes a `whisper` block (`available` / `backend` / `cli_path` /
  `cli_present` / `model_path` / `model_present` / `detail`) added
  2026-08-29 for `POST /v1/transcribe`.

### Hardware detection → recommended LLM size (added 2026-07-27)

`src/hardware.rs` implements a simple heuristic that picks a GPT-2-family
size (124M/355M/774M/1.5B) from detected VRAM: <2GB → 124M, 2-4GB → 355M,
4-8GB → 774M, 8GB+ → 1.5B; undetectable GPU / CPU-only → 124M (safe
fallback). **Honest disclosure**: this is a rough size-vs-VRAM comparison
(parameter count × 4 bytes, fp32 estimate), not a precise performance
model — it ignores KV-cache and activation memory.

GPU detection is opt-in via Cargo features `hw-detect-vulkan` /
`hw-detect-directx` (disabled by default, so CPU-only or cross-compiled
builds are not forced to depend on the Vulkan loader / Windows SDK). When
enabled, Vulkan is preferred; if both features are enabled, the DXGI
(DirectX) result is cross-checked against the Vulkan result and logged
(`cross_check_agreement`). **Verified on real hardware**: running with
`--features hw-detect-vulkan` on this machine's NVIDIA GeForce GT 730
reported `vram_bytes=2104819712` — exactly matching the value previously
recorded via DXGI in `open-cuda`'s CLAUDE.md, confirming both detection
paths agree on this GPU.

### DeepSeek-"Engram"-style KV-cache/weight offload: investigated and declined (2026-08-08)

We investigated whether DeepSeek's "Engram"-style technique — evicting
static knowledge (KV-cache or weight shards) from VRAM to system RAM and
reloading on demand — could help this service run on small-VRAM GPUs like
the GT 730. **After reading the actual code, we declined to implement
it** — not because it is hard, but because the `open-cuda` inference path
this repo depends on has no VRAM-resident state to evict in the first
place. Every GEMM/Attention/softmax dispatch in `opencuda-blas` (i.e.
every `sgemm` call that goes to Vulkan) allocates a VRAM buffer through a
`ScopedAlloc` RAII guard (`opencuda-blas/src/lib.rs`), copies host→device,
computes, copies device→host, and frees it immediately — nothing remains
in VRAM once the call returns. Both the GPT-2 weights (`GptModel`'s
`word_embeddings` and per-layer `Linear`s) and the KV cache
(`open-cuda-llm::KvCacheHead`'s `k`/`v`/`k_latent`/`v_latent`) are plain
`Vec<f32>` that live in system RAM for their entire lifetime, even when
running with `--features real-vulkan`. In other words, this architecture
already ends up — by accident of its design, not by intent — in the state
Engram aims for: state stays resident in system RAM at all times, and the
GPU is only ever touched transiently per operation. Adding an LRU
eviction layer on top would have nothing to evict, so there would be no
measurable effect to report (we are not going to claim a benefit we
cannot measure). See the 2026-08-08 CLAUDE.md HANDOFF entry for the exact
code paths read.

### Classification vs. generation — which to use

`/v1/chat` (classification) and `/v1/generate` (generation) serve different
purposes and are deliberately not merged: `/v1/chat` only routes to canned
replies and is lightweight/fast (a single embedding forward pass);
`/v1/generate` runs the full GPT-2 124M model (548MB of weights) and is
heavier but produces genuine free-form text. Pick whichever fits the use
case.

## Speech recognition (`POST /v1/transcribe`, whisper.cpp CLI, added 2026-08-29)

open-english's speech recognition was originally limited to the browser's
Web Speech API. In-browser Whisper (transformers.js, "P2-α") was added,
but is bounded by the user's device GPU/NPU/CPU. P2-β lets the user's own
PC — which is already running `aruaru-llm` — do the transcription with a
larger Whisper model via `POST /v1/transcribe`. Canonical record:
`open-english/docs/SPEECH_RECOGNITION_REDESIGN.md` (§P2-β).

**Implementation (2026-08-29 pivot)**: the original plan was to link
`whisper-rs` (Rust bindings to whisper.cpp) directly, but `whisper-rs-sys`
has a **known blocker on Windows/MSVC** — bindgen emits glibc-specific
types and the `whisper_full_params` size assertion underflows
(`1_usize - 264_usize overflow`); re-research confirmed this reproduces on
`whisper-rs 0.16.0` and is **not** fixed by
`WHISPER_DONT_GENERATE_BINDINGS=1` (upstream issue 2026-04-21, no fix
yet). Since open-english targets Windows, the direct-link approach can't
work. Instead we **spawn whisper.cpp's official prebuilt CLI
(`whisper-cli` / legacy `main`) as a subprocess** — the same pattern this
codebase already uses for `pg_dump` / `Expand-Archive` / `adb`. No C++
link, no bindgen, and the CLI's chosen GPU backend is used as-is. **No
Cargo feature** — with no compile-time dependency it's just part of the
default build.

- **Enable**: drop two files in place (no rebuild).
  1. whisper.cpp's [prebuilt `whisper-cli`](https://github.com/ggml-org/whisper.cpp/releases)
     at `<crate>/models/whisper/whisper-cli` (`whisper-cli.exe` on
     Windows), or point `ARUARU_LLM_WHISPER_CLI` at it.
  2. a GGML model (`ggml-base.bin` = fast/modest, `ggml-large-v3-turbo` =
     best; neither is bundled) at `<crate>/models/whisper/ggml-base.bin`,
     or `ARUARU_LLM_WHISPER_MODEL`.
- **Without them**: `POST /v1/transcribe` returns `503` + a link to
  whisper.cpp releases.
- **Status**: `GET /v1/runtime` → `whisper` block (`available`,
  `cli_present`, `model_present`, `cli_path`, `model_path`, `detail`).
- **Tuning**: `ARUARU_LLM_WHISPER_TIMEOUT_SECS` (default 300) caps the
  subprocess wall-clock.
- **Honest disclosure / verification**: `cargo build --release` succeeds;
  `cargo test --release` — **100 passed** (7 new `transcribe` tests: WAV
  RIFF header, whisper-cli JSON parsing, missing-CLI/model errors, env
  overrides). **End-to-end with a real `whisper-cli` + real GGML model is
  not yet verified** (this dev box has neither) — next iteration will set
  both up and exercise `POST /v1/transcribe` over real HTTP.

## "Shadow clone" (分身の術) architecture

Following the same design as `open-web-server`: a single running instance
is shared by multiple domains, with no per-domain install required.
Management is intended to happen from [open-easy-web](https://github.com/aon-co-jp/open-easy-web)
(that integration is not yet wired up). See [CLAUDE.md](CLAUDE.md) for details.

## Tech stack

Rust + [RPoem](https://github.com/aon-co-jp/RPoem) (`open-runo-poem-compat`,
a Poem-API-compatible facade implemented directly on tokio/hyper — no
dependency on the real [Poem](https://github.com/poem-web/poem) crate,
migrated 2026-07-31) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
No DB dependency, single self-contained binary. Usable from Rust or any
other language over plain HTTP (this service is the HTTP-serving front
door for Python-AI-library Rust ports — `opencuda-bert`/`opencuda-llm`/
`opencuda-whisper`, i.e. Transformers/vLLM/Whisper equivalents).

See [CLAUDE.md](CLAUDE.md) for the design philosophy and
[PORTING.md](PORTING.md) for how to port these patterns elsewhere.

## Install

As of 2026-07-23, `install.sh` (Linux, registers a systemd service),
`install.ps1` (Windows, prints Windows service registration steps), and
`.github/workflows/release.yml` (builds Linux x86_64 / Windows x86_64
binaries on every tag push and attaches them to
[GitHub Releases](https://github.com/aon-co-jp/aruaru-llm/releases)) were
added. **Honest disclosure**: at startup this binary needs the 470MB+
`multilingual-e5-small` model weights (Hugging Face, MIT license) fetched
separately — not bundled with the installer for licensing reasons; see
`install.sh`/`install.ps1` for the download command. The build has a
sibling path dependency on `../open-cuda`, so building from source requires
cloning `open-cuda` into an adjacent directory (CI does this
automatically via `release.yml`). **Added 2026-07-25**: `/v1/generate`
(GPT-2 124M generation) additionally requires `config.json` /
`model.safetensors` (548MB) / `tokenizer.json`
(`openai-community/gpt2`, from Hugging Face) under
`../open-cuda/crates/opencuda-llm/models/gpt2/` (override the path with the
`ARUARU_LLM_GPT2_DIR` env var). If missing, only `/v1/generate` returns 503
— `/v1/chat` and the rest of the service keep working normally
(availability-first design, same philosophy as `bow_fallback`).

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## Related projects

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPU runtime (the SET pairing)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — first intended caller
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — canonical dev-policy source


---

## Update 2026-08-23 — `GET /v1/runtime` now reports the CPU SIMD path

CPU inference GEMM is SIMD-dispatched through `opencuda-blas`, but there
was no way to see *which combination of instruction sets* was actually
selected. The `/v1/runtime` response now includes a `cpu_simd` object,
with detection provided by the shared
[`open-cpu`](https://github.com/aon-co-jp/open-cpu) crate. Paths are
chosen by **combinations** of instruction sets (AVX2+FMA3,
AVX-512F+BW+VNNI, …), not by single flags.

Verified by actually starting the server and running
`curl http://127.0.0.1:4600/v1/runtime` on the development machine
(Ryzen 9 3950X, Zen 2):

```json
"cpu_simd": {
  "features": "avx2+fma3+sse2",
  "isa_profile": "avx2+fma3",
  "avx2_fma_path": true,
  "vnni_path": false,
  "avx512_opt_in": false
}
```

**Honest disclosure**: this endpoint only reports; it accelerates nothing.
The actual speedup is the earlier CPU inference SIMD work (AVX2+FMA3,
measured 3.34x). This change aligned that path's selection criteria with
`open-cpu`'s combination checks and made them visible. **The int8 VNNI
path is still unverified on real hardware** — this CPU does not have VNNI.
