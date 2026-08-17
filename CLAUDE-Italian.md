# Filosofia di design & politica di sviluppo & regole dell'ambiente di sviluppo (aruaru-llm)

> **Nota**: questa è una traduzione condensata dello stato attuale. Il
> registro storico dettagliato HANDOFF (decine di voci) rimane
> disponibile solo in giapponese in [CLAUDE.md](CLAUDE.md), per brevità
> — consultarlo per i dettagli sessione per sessione.

Repository GitHub: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm).

## Ruolo di questo progetto

Un servizio HTTP condiviso e indipendente che fornisce la logica di
risposta "AI chat commerce" per l'ecosistema `aruaru` (aruaru-tokyo,
aruaru-db, e-gov.info, karu.tokyo, ecc). Invece che ogni sito
implementi la propria logica di risposta chat, tutti interrogano
questo unico servizio via HTTP — centralizzando così l'unico punto da
modificare quando in futuro verrà collegata una vera inferenza LLM.

## Divulgazione onesta (importante)

Dal 2026-07-25 `/v1/generate` utilizza il crate `opencuda-llm` di
`open-cuda` (pesi reali addestrati GPT-2 124M,
`openai-community/gpt2`) per una **vera generazione di testo
autoregressiva**. Tuttavia GPT-2 124M è un modello piccolo del 2019 e
non è paragonabile a LLM commerciali moderni come GPT-4, né per
capacità né per conoscenza. `/v1/chat` (classificazione di intenti)
resta separato: `opencuda-bert` (multilingual-e5-small) calcola veri
embedding di frasi e classifica tramite similarità coseno con vettori
rappresentativi di intento — una **classificazione di similarità
semantica basata su encoder**, non generazione di dialogo. Le due
capacità non sono deliberatamente unificate.

## Superficie API attuale

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(opzionale)}` →
  `{"reply": "...", "engine": "embedding-cosine-v0-opencuda-bert-cpu",
  "matched_intent": "..."}`.
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(opzionale, default 16, max 128), "tenant": "..."(opzionale)}` →
  `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`.
  Se i pesi reali di GPT-2 mancano, restituisce onestamente `503`
  (nessun fallback silenzioso come per `/v1/chat`).
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — gestione dinamica dei tenant
  (autenticazione tramite header `x-admin-token`).
- `GET /healthz` — controllo di stato.

### Nuovo: `POST /v1/generate-speculative` (aggiunto il 2026-08-17, commit `8f08900`)

Decodifica speculativa senza perdita in stile DSpark tramite
`open-cuda-llm::GptModel::generate_speculative`, **opt-in** (NON
sostituisce il `/v1/generate` predefinito). Accetta un `draft_id` che
indica un modello del catalogo già scaricato (ad es. `"distilgpt2"`)
come modello bozza. **Divulgazione onesta critica**: su esecuzione CPU
in `open-cuda` è stato misurato che questo percorso è **più lento**
del semplice `generate()` anche con un tasso di accettazione dell'80%
— perché la GEMM ingenua su CPU ha quasi nessun overhead di dispatch
da eliminare, quindi il calcolo aggiuntivo del modello bozza risulta
in una perdita netta su CPU. La verifica della velocità sotto
`real-vulkan` (dove l'overhead di dispatch domina — il caso d'uso
realmente previsto) non è ancora stata effettuata. Divulgato inoltre:
la penalità di ripetizione e i modelli compressi con MLA non sono
supportati da questo percorso speculativo.

## Stack tecnologico

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`, una facciata compatibile con l'API Poem
implementata direttamente su tokio/hyper, dal 2026-07-31 al posto del
vero crate [Poem](https://github.com/poem-web/poem) — nessun
estrattore `Data<T>`, lo stato condiviso viene catturato tramite
closure `Arc::clone` alla registrazione delle rotte) +
[open-cuda](https://github.com/aon-co-jp/open-cuda). Nessuna
dipendenza da DB, un unico binario autonomo.

## Architettura "clone ombra" (分身の術)

Come `open-web-server`: un'unica istanza in esecuzione è condivisa da
più domini, senza necessità di installazione per dominio
(`TenantRegistry` in `src/tenants.rs`, registrazione a runtime senza
riavvio tramite le API `/admin/tenants`). La gestione è prevista da
[open-easy-web](https://github.com/aon-co-jp/open-easy-web)
(integrazione non ancora collegata).

## Progetti correlati

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — runtime GPU, il compagno del SET
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — primo chiamante
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — gestione prevista
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — fonte canonica delle regole di sviluppo

---

Altre lingue: [日本語 (originale, con cronologia HANDOFF completa)](CLAUDE.md) ·
[Deutsch](CLAUDE-German.md) · [Français](CLAUDE-French.md) ·
[Русский](CLAUDE-Russian.md) · [Українська](CLAUDE-Ukrainian.md) ·
[עברית](CLAUDE-Hebrew.md) · [فارسی](CLAUDE-Persian.md) · [العربية](CLAUDE-Arabic.md)
