# aruaru-llm

A shared "AI chat commerce" response service for the `aruaru` ecosystem
(aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo, etc). Instead of each
site implementing its own chat-reply logic, they call this single HTTP
service — centralizing the one place that needs to change when real LLM
inference is eventually wired in.

> ⚠️ **Honest disclosure (important, updated 2026-07-25)**: despite the
> "LLM" name, this service does **not** perform autoregressive dialogue
> generation. Since 2026-07-21 it classifies intent by computing a real
> sentence embedding with open-cuda's `opencuda-bert` crate
> (multilingual-e5-small, MIT license, 100 languages including Japanese)
> and scoring cosine similarity against representative example sentences
> per intent — a genuine improvement over the earlier fixed-vocabulary
> bag-of-words dot product, but still an **encoder-only semantic
> similarity classifier**, not a text-generation capability. See
> [CLAUDE.md](CLAUDE.md) for details and rationale.

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
  "...", "matched_intent": "..."}`
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` — tenant registration management (`x-admin-token` header auth)
- `GET /healthz` — health check

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
automatically via `release.yml`).

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## Related projects

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPU runtime (the SET pairing)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — first intended caller
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — canonical dev-policy source
