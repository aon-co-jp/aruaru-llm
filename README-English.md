# aruaru-llm

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
  (real autoregressive generation via GPT-2 124M weights — heavier but genuine. English prompts
  recommended, since GPT-2's BPE vocabulary is trained mostly on English text. Example, verified
  end-to-end over real HTTP: `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out of your"`)
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` — tenant registration management (`x-admin-token` header auth)
- `GET /healthz` — health check

### Classification vs. generation — which to use

`/v1/chat` (classification) and `/v1/generate` (generation) serve different
purposes and are deliberately not merged: `/v1/chat` only routes to canned
replies and is lightweight/fast (a single embedding forward pass);
`/v1/generate` runs the full GPT-2 124M model (548MB of weights) and is
heavier but produces genuine free-form text. Pick whichever fits the use
case.

## "Shadow clone" (分身の術) architecture

Following the same design as `open-web-server`: a single running instance
is shared by multiple domains, with no per-domain install required.
Management is intended to happen from [open-easy-web](https://github.com/aon-co-jp/open-easy-web)
(that integration is not yet wired up). See [CLAUDE.md](CLAUDE.md) for details.

## Tech stack

Rust + [Poem](https://github.com/poem-web/poem) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
No DB dependency, single self-contained binary.

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
