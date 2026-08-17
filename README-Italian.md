# aruaru-llm

*日本語*: [README.md](README.md) ·
*English*: [README-English.md](README-English.md) ·
*Other languages*: [Deutsch](README-German.md) · [Français](README-French.md) ·
[Русский](README-Russian.md) · [Українська](README-Ukrainian.md) ·
[עברית](README-Hebrew.md) · [فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **Ultimo aggiornamento (2026-08-10)**: collegata la nuova
> `GptModel::generate_with_repetition_penalty` di `open-cuda` (penalità di
> ripetizione in stile CTRL) a `/v1/generate`, **abilitata di default**
> (variabile d'ambiente `ARUARU_LLM_REPETITION_PENALTY`, valore predefinito
> `1.3`; impostare `1.0` per ripristinare il vecchio comportamento senza
> penalità). Risolve un noto degrado del modello base GPT-2 — ripetizione
> infinita della stessa stringa — poiché il modello base non ha fine-tuning
> di dialogo. Verificato su pesi reali GPT-2 124M lato `open-cuda`: senza
> penalità il loop si riproduce davvero; con `penalty=1.3` si interrompe e
> produce testo conversazionale grammaticalmente naturale. Con
> `penalty=1.0` l'output è identico byte per byte a `generate()`, quindi
> nessun altro test regredisce. Dettagli in [CLAUDE.md](CLAUDE.md) (solo
> giapponese).

> 📌 **Aggiornamento (2026-08-08)**: aggiunto un percorso opt-in
> (disattivato di default) di compressione della cache KV in stile MLA
> (implementazione di `open-cuda` ispirata a DeepSeek-V3) in `/v1/generate`
> via `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`. Per GPT-2 124M:
> head_dim=64 -> d_c=16 (75% in meno di memoria KV per token). **Divulgazione
> onesta**: le matrici di proiezione sono inizializzate casualmente (non
> addestrate), quindi la compressione è lossy — test reali hanno mostrato
> una qualità di generazione visibilmente peggiore, motivo per cui resta
> disattivata di default. Esiste anche una variante calibrata con PCA
> (`ARUARU_LLM_MLA_CALIBRATED=1`) che evita i loop di ripetizione degenerati
> della variante casuale, ma è comunque chiaramente peggiore del percorso
> non compresso — anch'essa resta disattivata di default.

> 📌 Attività in sospeso (2026-08-06): esiste un piano per integrare le
> tecniche Toshiba SBM e DeepSeek. Dettagli in [CLAUDE.md](CLAUDE.md).

> 📌 **Aggiornamento (2026-08-07)**: verificato via richieste HTTP reali che
> `/v1/chat` e `/v1/classify-security` **non** soffrono del bug "input vuoto
> → 503" precedentemente corretto per `/v1/generate` e `/v1/translate` —
> entrambi restituiscono correttamente 200 per input vuoto. Nessuna modifica
> al codice necessaria.

Un servizio di risposta condiviso "AI chat commerce" per l'ecosistema
`aruaru` (aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo, ecc). Invece che
ogni sito implementi la propria logica di risposta chat, tutti chiamano
questo unico servizio HTTP — centralizzando così l'unico punto da modificare
quando in futuro verrà collegata una vera inferenza LLM.

> ⚠️ **Divulgazione onesta (importante, aggiornata al 2026-07-25)**: dal
> 2026-07-25 questo servizio integra il crate `opencuda-llm` di `open-cuda`
> (pesi reali addestrati GPT-2 124M, `openai-community/gpt2`), quindi
> `POST /v1/generate` ora esegue **vera generazione di testo autoregressiva**.
> Tuttavia **GPT-2 124M è un modello piccolo del 2019 e non è paragonabile
> a LLM commerciali moderni come GPT-4** per capacità o conoscenza. Questa è
> una dimostrazione che la generazione autonoma funziona senza un contratto
> API LLM esterno, non un'affermazione di qualità allo stato dell'arte —
> l'output è spesso inglese grammaticalmente fluente ma non garantito
> fattualmente accurato (può allucinare). `POST /v1/chat` (classificazione
> di intenti via embedding di frasi `opencuda-bert` + similarità coseno, dal
> 2026-07-21) resta un percorso separato, leggero e veloce per risposte
> predefinite — deliberatamente non unito alla generazione. Dettagli e
> motivazioni in [CLAUDE.md](CLAUDE.md).

## Accoppiato ("SET") con open-cuda

Dipende dai crate `opencuda-core`/`opencuda-cpu`/`opencuda-blas`/
`opencuda-bert` di [`open-cuda`](https://github.com/aon-co-jp/open-cuda)
tramite dipendenza di percorso. Ad ogni richiesta `/v1/chat`,
`opencuda-bert` esegue il forward pass di multilingual-e5-small (chiamando
i veri kernel GEMM/Attention di `opencuda-blas` su `opencuda_cpu::CpuDevice`)
per incorporare il messaggio, poi lo confronta via similarità coseno con
l'embedding rappresentativo (in cache) di ciascun intento. È una vera
chiamata a runtime attraverso la pipeline di calcolo di open-cuda, non solo
un riferimento in `Cargo.toml` — verificato avviando davvero il server ed
eseguendo `POST /v1/chat`.

Detto ciò, non è vera inferenza LLM neurale (generazione di dialogo) — solo
il forward pass dell'encoder; il decoder autoregressivo resta non
implementato. I percorsi rapidi specifici per GPU (`GemmPath::CuBlas`/
`RocBlas`/`OneMkl`) restano stub (i percorsi CPU e Vulkan generico sono
implementati). Dettagli nella sezione HANDOFF del `CLAUDE.md` di open-cuda.

**Aggiornamento 2026-07-25 (fallback di disponibilità)**: se
`models/multilingual-e5-small/` (470MB+) manca o non si carica, questo
servizio ora ricade automaticamente sul prodotto scalare bag-of-words
originale (`src/bow_fallback.rs`) invece di far fallire le richieste. Il
campo `engine` della risposta `/v1/chat` riporta sempre onestamente quale
percorso è stato effettivamente usato (`embedding-cosine-v0-opencuda-bert-cpu`
o `bow-dotproduct-v0-opencuda-cpu-fallback`) — la qualità di classificazione
è nettamente più bassa nel percorso di fallback (corrispondenza di parole
chiave, non comprensione semantica).

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(opzionale)}` →
  `{"reply": "...", "engine": "...", "matched_intent": "..."}`
  (classificazione di intenti, risposte predefinite leggere/veloci)
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens": 16(opzionale,
  default 16, max 128), "tenant": "..."(opzionale)}` →
  `{"completion": "...", "engine": "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu",
  "disclosure": "..."}` (vera generazione autoregressiva con pesi GPT-2
  124M — più pesante ma genuina. **Penalità di ripetizione predefinita
  `1.3`** — variabile `ARUARU_LLM_REPETITION_PENALTY` per sovrascrivere,
  `1.0` la disattiva — per prevenire loop di ripetizione infinita. Prompt in
  inglese consigliati, poiché il vocabolario BPE di GPT-2 è addestrato
  principalmente su testo inglese. Esempio, verificato end-to-end via HTTP
  reale: `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out of
  your"`)
- `GET /v1/models/catalog` — modelli compatibili GPT-2 disponibili
  (`gpt2`/`distilgpt2`/`gpt2-medium`/`gpt2-large`/`gpt2-xl`, l'ultimo
  aggiunto il 2026-07-27), quali sono già installati, e la directory del
  modello attualmente attivo.
- `POST /v1/models/install` / `POST /v1/models/select` — scarica un modello
  del catalogo da Hugging Face, o effettua l'hot-swap del modello attivo
  senza riavviare il processo.
- `GET /v1/recommend` (aggiunto il 2026-07-27) — rileva l'hardware (VRAM)
  via `open-cuda` (Vulkan) o `open-directx` (DXGI) e restituisce una
  dimensione consigliata della famiglia GPT-2, senza scaricare nulla.
- `POST /v1/recommend-and-download` (aggiunto il 2026-07-27, dietro il
  pulsante "Download recommended LLM") — rileva l'hardware → sceglie una
  dimensione consigliata → la scarica da Hugging Face se non presente
  (idempotente) → effettua l'hot-swap di `/v1/generate` su di essa.
- `GET /` (aggiunto il 2026-07-27) — UI HTML statica minima
  (`static/index.html`, nessun framework).
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` —
  gestione della registrazione tenant (autenticazione tramite header
  `x-admin-token`)
- `GET /healthz` — controllo di stato

### Rilevamento hardware → dimensione LLM consigliata (aggiunto il 2026-07-27)

`src/hardware.rs` implementa un'euristica semplice che sceglie una
dimensione della famiglia GPT-2 (124M/355M/774M/1.5B) dalla VRAM rilevata:
<2GB → 124M, 2-4GB → 355M, 4-8GB → 774M, 8GB+ → 1.5B; GPU non rilevabile /
solo CPU → 124M (fallback sicuro). **Divulgazione onesta**: è un confronto
approssimativo dimensione-vs-VRAM (numero di parametri × 4 byte, stima
fp32), non un modello preciso di prestazioni — ignora la memoria di cache KV
e delle attivazioni.

Il rilevamento GPU è opt-in tramite i feature Cargo `hw-detect-vulkan` /
`hw-detect-directx` (disattivati di default). Se abilitato, Vulkan è
preferito; se entrambi sono abilitati, il risultato DXGI (DirectX) viene
incrociato con quello Vulkan e registrato (`cross_check_agreement`).
**Verificato su hardware reale**: eseguendo con `--features hw-detect-vulkan`
sulla NVIDIA GeForce GT 730 di questa macchina è stato riportato
`vram_bytes=2104819712` — esattamente il valore registrato in precedenza via
DXGI nel `CLAUDE.md` di `open-cuda`, confermando che entrambi i percorsi di
rilevamento concordano.

### Offload KV-cache/pesi in stile "Engram" di DeepSeek: valutato e scartato (2026-08-08)

È stato valutato se la tecnica "Engram" di DeepSeek — evacuare conoscenza
statica (cache KV o parti di pesi) dalla VRAM alla RAM di sistema e
ricaricarla su richiesta — potesse aiutare questo servizio su GPU con poca
VRAM come la GT 730. **Dopo aver letto il codice reale, l'implementazione è
stata scartata** — non perché difficile, ma perché il percorso di inferenza
di `open-cuda` da cui dipende questo repo non ha stato residente in VRAM da
evacuare: ogni dispatch GEMM/Attention/softmax in `opencuda-blas` alloca un
buffer VRAM, copia, calcola, copia indietro e libera immediatamente — nulla
resta in VRAM. Sia i pesi GPT-2 che la cache KV vivono come `Vec<f32>` nella
RAM di sistema dall'inizio alla fine, anche con `--features real-vulkan`.
Questa architettura si trova quindi già, per caso, vicina allo stato a cui
mira Engram.

### Classificazione vs. generazione — quale usare

`/v1/chat` (classificazione) e `/v1/generate` (generazione) hanno scopi
diversi e non sono deliberatamente unificati: `/v1/chat` instrada solo verso
risposte predefinite ed è leggero/veloce (un singolo forward pass di
embedding); `/v1/generate` esegue l'intero modello GPT-2 124M (548MB di
pesi) ed è più pesante ma produce testo libero genuino.

## Architettura "clone ombra" (分身の術)

Segue lo stesso design di `open-web-server`: un'unica istanza in esecuzione
è condivisa da più domini, senza necessità di installazione per dominio. La
gestione è prevista da
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) (integrazione
non ancora collegata). Dettagli in [CLAUDE.md](CLAUDE.md).

## Stack tecnologico

Rust + [RPoem](https://github.com/aon-co-jp/RPoem) (`open-runo-poem-compat`,
una facciata compatibile con l'API Poem implementata direttamente su
tokio/hyper — nessuna dipendenza dal vero crate
[Poem](https://github.com/poem-web/poem), migrato il 2026-07-31) +
[open-cuda](https://github.com/aon-co-jp/open-cuda). Nessuna dipendenza da
DB, un unico binario autonomo. Utilizzabile da Rust o da qualsiasi altro
linguaggio via HTTP semplice (questo servizio è la porta HTTP per i port
Rust di librerie AI Python — `opencuda-bert`/`opencuda-llm`/
`opencuda-whisper`, equivalenti di Transformers/vLLM/Whisper).

Vedi [CLAUDE.md](CLAUDE.md) (solo giapponese) per la filosofia di design e
[PORTING.md](PORTING.md) (solo giapponese) per come portare questi pattern
altrove.

## Installazione

Dal 2026-07-23 sono presenti `install.sh` (Linux, registra un servizio
systemd), `install.ps1` (Windows, mostra i passi di registrazione del
servizio Windows) e `.github/workflows/release.yml` (compila binari Linux
x86_64 / Windows x86_64 ad ogni push di tag e li allega a
[GitHub Releases](https://github.com/aon-co-jp/aruaru-llm/releases)).
**Divulgazione onesta**: all'avvio questo binario richiede i pesi del
modello `multilingual-e5-small` (470MB+, Hugging Face, licenza MIT) da
scaricare separatamente — non inclusi nell'installer per motivi di licenza.
La build ha una dipendenza di percorso "sibling" da `../open-cuda`, quindi
compilare dal sorgente richiede di clonare `open-cuda` in una directory
adiacente (la CI lo fa automaticamente via `release.yml`). **Dal
2026-07-25**: `/v1/generate` (generazione GPT-2 124M) richiede inoltre
`config.json`/`model.safetensors` (548MB)/`tokenizer.json`
(`openai-community/gpt2`, da Hugging Face) sotto
`../open-cuda/crates/opencuda-llm/models/gpt2/` (percorso sovrascrivibile
con `ARUARU_LLM_GPT2_DIR`). Se mancante, solo `/v1/generate` restituisce
503 — `/v1/chat` e il resto del servizio continuano a funzionare
normalmente (design che privilegia la disponibilità, stessa filosofia di
`bow_fallback`).

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## Progetti correlati

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — runtime GPU (il compagno del SET)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — primo chiamante previsto
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — fonte canonica delle regole di sviluppo
