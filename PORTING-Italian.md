# PORTING.md — Guida al porting di aruaru-llm (versione condensata)

> **Nota**: questa è una traduzione condensata. La guida tecnica
> completa con dettagli di codice e insidie rimane disponibile solo in
> giapponese in [PORTING.md](PORTING.md) — consultarla prima di
> adottare effettivamente un pattern.

Riepilogo dei pattern di implementazione riutilizzabili di questo
progetto, nel caso vengano portati in un altro progetto:

1. **Pattern di accoppiamento con open-cuda (configurazione SET)**:
   dipendenza di percorso su `opencuda-core`/`opencuda-cpu`; invoca
   la vera esecuzione di kernel GPU (`alloc_buffer`→`copy_from_host`→
   `launch_kernel`→`synchronize`→`copy_to_host`).
2. **Classificazione di intenti basata su regole, progettata per una
   futura sostituzione con un vero LLM**: mantenere il campo `engine`
   e riportare sempre onestamente quale implementazione è stata
   effettivamente usata.
3. **Livello API HTTP tramite RPoem** (`open-runo-poem-compat`) invece
   del vero crate `poem` — nessun estrattore `Data<T>`, lo stato
   condiviso viene catturato tramite closure `Arc::clone`.
4. **Pattern di validazione input vuoti** (2026-08-06): `400 Bad
   Request` esplicito invece di far trapelare errori interni del
   tokenizer come un fuorviante `503`.
5. **Pattern di registrazione tenant "clone ombra"** (condiviso con
   `open-web-server`): `TenantRegistry` + endpoint `/admin/tenants`.
6. **Capacità di generazione reale tramite `opencuda-llm::GptModel`**:
   pesi GPT-2 124M — il campo `disclosure` non deve mai essere
   omesso, `/v1/chat` e `/v1/generate` non devono essere uniti.
7. **Rilevamento hardware → dimensione LLM consigliata → download
   automatico** (feature opt-in `hw-detect-vulkan`/`hw-detect-directx`,
   pattern di cross-check, divulgazione onesta che si tratta solo di
   un'euristica approssimativa dimensione-vs-VRAM).
8. **Pattern plugin di traduzione** (feature `nllb-translate`): una
   dipendenza pesante e opzionale `rust-bert`/`tch`, isolata dietro
   una feature Cargo, disattivata di default.
9. **Feature di dispatch `real-vulkan`** — **Nota**: non ancora
   consigliata per il porting altrove, a causa di un bug noto e non
   risolto (`Linear::forward` non collega i byte SPIR-V di
   `matmul.spv` a `sgemm`, causando il fallimento immediato di
   `GemmPath::VulkanGeneric`).
10. **Pattern di penalità di ripetizione**
    (`generate_with_repetition_penalty`, default `1.3`, sovrascrivibile
    tramite variabile d'ambiente).

**Avvertenza importante**: GPT-2 124M è piccolo e risale al 2019 — non
paragonabile a LLM commerciali moderni. `/v1/chat` rimane basato su
regole + classificazione di similarità tramite encoder, non generazione
di dialogo neurale. Ciò va divulgato anche in ogni destinazione di
porting.

---

Altre lingue: [日本語 (originale, dettagli completi)](PORTING.md) ·
[Deutsch](PORTING-German.md) · [Français](PORTING-French.md) ·
[Русский](PORTING-Russian.md) · [Українська](PORTING-Ukrainian.md) ·
[עברית](PORTING-Hebrew.md) · [فارسی](PORTING-Persian.md) · [العربية](PORTING-Arabic.md)
