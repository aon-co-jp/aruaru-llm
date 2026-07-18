# aruaru-llm

A shared "AI chat commerce" response service for the `aruaru` ecosystem
(aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo, etc). Instead of each
site implementing its own chat-reply logic, they call this single HTTP
service — centralizing the one place that needs to change when real LLM
inference is eventually wired in.

> ⚠️ **Honest disclosure (important)**: despite the "LLM" name, v0.1.0
> does **not** perform any real neural-network inference. It's a simple
> rule-based intent classifier using a **bag-of-words dot product** over a
> fixed vocabulary. See [CLAUDE.md](CLAUDE.md) for details and rationale.

## Paired ("SET") with open-cuda

Depends on [`open-cuda`](https://github.com/aon-co-jp/open-cuda)'s
`opencuda-core`/`opencuda-cpu` crates via a path dependency, and actually
calls `GpuDevice::launch_kernel` on every `/v1/chat` request (an
elementwise-multiply kernel over the bag-of-words vectors). This is a real
runtime call through open-cuda's kernel-execution pipeline, not just a
`Cargo.toml` reference.

That said, this is not real neural LLM inference — open-cuda's
`opencuda-blas` crate (GEMM/Attention) is still explicitly stubbed out as
"Phase 3", so actual embedding-similarity or transformer inference remains
future work.

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

## Related projects

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPU runtime (the SET pairing)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — first intended caller
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — canonical dev-policy source
