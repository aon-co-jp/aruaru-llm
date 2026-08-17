# aruaru-llm

*日本語*: [README.md](README.md) ·
*English*: [README-English.md](README-English.md) ·
*Other languages*: [Italiano](README-Italian.md) · [Français](README-French.md) ·
[Русский](README-Russian.md) · [Українська](README-Ukrainian.md) ·
[עברית](README-Hebrew.md) · [فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **Letztes Update (2026-08-10)**: `open-cuda`s neues `GptModel::
> generate_with_repetition_penalty` (Wiederholungsstrafe im CTRL-Stil) wurde
> in `/v1/generate` eingebunden und ist **standardmäßig aktiviert**
> (Umgebungsvariable `ARUARU_LLM_REPETITION_PENALTY`, Standardwert `1.3`;
> `1.0` stellt das alte Verhalten ohne Strafe wieder her). Dies behebt einen
> bekannten Degenerationsmodus des GPT-2-Basismodells — endlose Wiederholung
> derselben Zeichenkette —, da das Basismodell kein Dialog-Finetuning hat.
> Verifiziert mit echten GPT-2-124M-Gewichten auf `open-cuda`-Seite: ohne
> die Strafe tritt die Schleife tatsächlich auf; mit `penalty=1.3` stoppt sie
> und erzeugt grammatikalisch natürlichen Konversationstext. Bei
> `penalty=1.0` ist die Ausgabe byte-identisch zur bestehenden `generate()`
> API, sodass keine anderen Tests regressieren. Details siehe
> [CLAUDE.md](CLAUDE.md) (nur Japanisch).

> 📌 **Update (2026-08-08)**: Ein Opt-in (standardmäßig deaktivierter)
> MLA-artiger KV-Cache-Kompressionspfad (`open-cuda`s an DeepSeek-V3
> angelehnte Implementierung) wurde über `ARUARU_LLM_ENABLE_MLA_KV_
> COMPRESSION=1` in `/v1/generate` eingebunden. Für GPT-2 124M bedeutet das
> head_dim=64 -> d_c=16 (75% weniger KV-Speicher pro Token). **Ehrliche
> Offenlegung**: Die Projektionsmatrizen sind zufällig initialisiert
> (untrainiert), daher verlustbehaftet — gemessene Tests zeigten sichtbar
> schlechtere Generierungsqualität, weshalb dies standardmäßig deaktiviert
> bleibt. Ein PCA-kalibrierter Nachfolger existiert ebenfalls
> (`ARUARU_LLM_MLA_CALIBRATED=1`), vermeidet die degenerierten
> Wiederholungsschleifen der Zufallsvariante, ist aber weiterhin klar
> schlechter als der unkomprimierte Pfad — auch dieser bleibt standardmäßig
> deaktiviert.

> 📌 Offene Aufgabe (2026-08-06): Es gibt Pläne, Toshiba-SBM- und
> DeepSeek-Techniken einzubinden. Details siehe [CLAUDE.md](CLAUDE.md).

> 📌 **Update (2026-08-07)**: Über echte HTTP-Anfragen verifiziert, dass
> `/v1/chat` und `/v1/classify-security` **nicht** von dem zuvor für
> `/v1/generate` und `/v1/translate` behobenen "leere Eingabe → 503"-Bug
> betroffen sind — beide geben für leere Eingaben korrekt 200 zurück. Keine
> Codeänderung erforderlich.

Ein gemeinsamer "AI-Chat-Commerce"-Antwortdienst für das `aruaru`-Ökosystem
(aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo usw.). Statt dass jede Site
ihre eigene Chat-Antwortlogik implementiert, rufen sie diesen einen
HTTP-Dienst auf — so bleibt die Stelle, die bei einer künftigen Umstellung
auf echte LLM-Inferenz geändert werden muss, an einem einzigen Ort
zentralisiert.

> ⚠️ **Ehrliche Offenlegung (wichtig, Stand 2026-07-25)**: Seit 2026-07-25
> integriert dieser Dienst `open-cuda`s `opencuda-llm`-Crate (echte trainierte
> GPT-2-124M-Gewichte, `openai-community/gpt2`), sodass `POST /v1/generate`
> jetzt **echte autoregressive Textgenerierung** durchführt. Allerdings ist
> **GPT-2 124M ein kleines Modell aus dem Jahr 2019 und nicht mit modernen
> kommerziellen LLMs wie GPT-4 vergleichbar** — weder in Fähigkeit noch in
> Wissen. Dies demonstriert, dass eigenständige Generierung ohne externen
> LLM-API-Vertrag funktioniert, nicht dass die Qualität State-of-the-Art
> wäre — die Ausgabe ist oft grammatikalisch flüssiges Englisch, aber nicht
> garantiert faktisch korrekt (Halluzinationen möglich). `POST /v1/chat`
> (Intent-Klassifikation via `opencuda-bert`-Satz-Embeddings + Kosinus-
> Ähnlichkeit, seit 2026-07-21) bleibt ein separater, leichtgewichtiger,
> schneller Pfad für vorgefertigte Antworten — bewusst nicht mit der
> Generierung zusammengeführt. Details und Begründung siehe
> [CLAUDE.md](CLAUDE.md).

## Gepaart ("SET") mit open-cuda

Hängt über eine Pfadabhängigkeit von [`open-cuda`](https://github.com/aon-co-jp/open-cuda)s
`opencuda-core`/`opencuda-cpu`/`opencuda-blas`/`opencuda-bert`-Crates ab. Bei
jeder `/v1/chat`-Anfrage führt `opencuda-bert` den Forward-Pass von
multilingual-e5-small aus (ruft dabei tatsächlich `opencuda-blas`s echte
GEMM/Attention-Kernel auf `opencuda_cpu::CpuDevice` auf), um die Nachricht
zu embedden, und vergleicht sie dann per Kosinus-Ähnlichkeit mit dem
gecachten repräsentativen Embedding jeder Intention. Dies ist ein echter
Laufzeitaufruf durch die Compute-Pipeline von open-cuda, keine bloße
`Cargo.toml`-Referenz — verifiziert durch tatsächliches Starten des Servers
und Ausführen von `POST /v1/chat`.

Dennoch ist dies keine echte neuronale LLM-Inferenz (Dialoggenerierung) —
nur der Encoder-Forward-Pass; der autoregressive Decoder ist nicht
implementiert. GPU-spezifische Fast-Paths (`GemmPath::CuBlas`/`RocBlas`/
`OneMkl`) sind weiterhin Stubs (CPU- und generische Vulkan-Pfade sind
implementiert). Details siehe HANDOFF-Abschnitt von open-cudas `CLAUDE.md`.

**Update 2026-07-25 (Verfügbarkeits-Fallback)**: Fehlt oder scheitert das
Laden von `models/multilingual-e5-small/` (470MB+), fällt dieser Dienst
jetzt automatisch auf das ursprüngliche Bag-of-Words-Punktprodukt
(`src/bow_fallback.rs`) zurück, statt Anfragen komplett scheitern zu lassen.
Das `engine`-Feld der `/v1/chat`-Antwort meldet immer ehrlich, welcher Pfad
tatsächlich verwendet wurde (`embedding-cosine-v0-opencuda-bert-cpu` oder
`bow-dotproduct-v0-opencuda-cpu-fallback`) — die Klassifikationsqualität ist
im Fallback-Pfad spürbar niedriger (Keyword-Matching statt semantisches
Verständnis).

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(optional)}` →
  `{"reply": "...", "engine": "...", "matched_intent": "..."}`
  (Intent-Klassifikation, leichtgewichtige/schnelle vorgefertigte Antworten)
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens": 16(optional,
  Standard 16, max. 128), "tenant": "..."(optional)}` →
  `{"completion": "...", "engine": "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu",
  "disclosure": "..."}` (echte autoregressive Generierung via GPT-2-124M-
  Gewichten — schwerer, aber echt. **Wiederholungsstrafe standardmäßig
  `1.3`** — überschreibbar via `ARUARU_LLM_REPETITION_PENALTY`, `1.0`
  deaktiviert sie — um Endlos-Wiederholungsschleifen zu verhindern.
  Englische Prompts empfohlen, da GPT-2s BPE-Vokabular überwiegend auf
  englischem Text trainiert ist. Beispiel, End-to-End über echtes HTTP
  verifiziert: `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out of
  your"`)
- `GET /v1/models/catalog` — verfügbare GPT-2-kompatible Modelle
  (`gpt2`/`distilgpt2`/`gpt2-medium`/`gpt2-large`/`gpt2-xl`, letzteres seit
  2026-07-27), bereits installierte, und das aktuell aktive Modellverzeichnis.
- `POST /v1/models/install` / `POST /v1/models/select` — ein Katalogmodell
  von Hugging Face herunterladen bzw. das aktive Modell ohne Prozess-
  neustart hot-swappen.
- `GET /v1/recommend` (seit 2026-07-27) — erkennt Hardware (VRAM) via
  `open-cuda` (Vulkan) oder `open-directx` (DXGI) und gibt eine empfohlene
  GPT-2-Familiengröße zurück, ohne etwas herunterzuladen.
- `POST /v1/recommend-and-download` (seit 2026-07-27, hinter dem "Download
  recommended LLM"-Button) — Hardware-Erkennung → empfohlene Größe wählen →
  von Hugging Face herunterladen falls nicht vorhanden (idempotent) →
  `/v1/generate` darauf hot-swappen.
- `GET /` (seit 2026-07-27) — minimale statische HTML-UI
  (`static/index.html`, kein Framework).
- `POST /admin/tenants` / `GET /admin/tenants` / `DELETE /admin/tenants/:host` —
  Tenant-Verwaltung (`x-admin-token`-Header-Authentifizierung)
- `GET /healthz` — Health-Check

### Hardware-Erkennung → empfohlene LLM-Größe (seit 2026-07-27)

`src/hardware.rs` implementiert eine einfache Heuristik, die aus erkanntem
VRAM eine GPT-2-Familiengröße wählt (124M/355M/774M/1.5B): <2GB → 124M,
2-4GB → 355M, 4-8GB → 774M, 8GB+ → 1.5B; nicht erkennbare GPU / nur CPU →
124M (sicherer Fallback). **Ehrliche Offenlegung**: Dies ist ein grober
Vergleich von Größe und VRAM (Parameteranzahl × 4 Byte, fp32-Schätzung),
kein präzises Performance-Modell — KV-Cache und Aktivierungsspeicher werden
ignoriert.

GPU-Erkennung ist über die Cargo-Features `hw-detect-vulkan` /
`hw-detect-directx` opt-in (standardmäßig deaktiviert). Bei Aktivierung wird
Vulkan bevorzugt; sind beide Features aktiv, wird das DXGI(DirectX)-Ergebnis
gegen Vulkan quergecheckt und geloggt (`cross_check_agreement`). **Auf
echter Hardware verifiziert**: Mit `--features hw-detect-vulkan` auf der
NVIDIA GeForce GT 730 dieser Maschine wurde `vram_bytes=2104819712`
gemeldet — exakt der zuvor via DXGI in `open-cuda`s `CLAUDE.md` erfasste
Wert, was bestätigt, dass beide Erkennungspfade übereinstimmen.

### DeepSeek-"Engram"-artiges KV-Cache/Gewichts-Offloading: untersucht und abgelehnt (2026-08-08)

Untersucht wurde, ob DeepSeeks "Engram"-artige Technik — Verdrängung
statischen Wissens (KV-Cache oder Gewichtsteile) aus dem VRAM ins System-RAM
und bedarfsweises Neuladen — diesem Dienst helfen könnte, auf kleinen
VRAM-GPUs wie der GT 730 zu laufen. **Nach Lesen des tatsächlichen Codes
wurde die Implementierung abgelehnt** — nicht weil es schwierig wäre,
sondern weil der von diesem Repo genutzte `open-cuda`-Inferenzpfad von
vornherein keinen VRAM-residenten Zustand zum Verdrängen hat: Jeder
GEMM/Attention/Softmax-Dispatch in `opencuda-blas` alloziert einen
VRAM-Puffer, kopiert, berechnet, kopiert zurück und gibt sofort frei —
nichts bleibt im VRAM. Sowohl die GPT-2-Gewichte als auch der KV-Cache
liegen von Anfang bis Ende als `Vec<f32>` im System-RAM, auch bei
`--features real-vulkan`. Diese Architektur befindet sich also — unbeab-
sichtigt — bereits nahe dem Zustand, den Engram anstrebt.

### Klassifikation vs. Generierung — was wann nutzen

`/v1/chat` (Klassifikation) und `/v1/generate` (Generierung) dienen
unterschiedlichen Zwecken und wurden bewusst nicht zusammengeführt:
`/v1/chat` routet nur zu vorgefertigten Antworten und ist leichtgewichtig/
schnell (ein einzelner Embedding-Forward-Pass); `/v1/generate` führt das
volle GPT-2-124M-Modell (548MB Gewichte) aus und ist schwerer, produziert
aber echten freien Text.

## "Schattenklon" ("分身の術")-Architektur

Nach demselben Design wie `open-web-server`: Eine einzelne laufende Instanz
wird von mehreren Domains gemeinsam genutzt, ohne dass eine Installation pro
Domain erforderlich ist. Die Verwaltung soll von
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) aus erfolgen
(diese Integration ist noch nicht verdrahtet). Details siehe
[CLAUDE.md](CLAUDE.md).

## Tech-Stack

Rust + [RPoem](https://github.com/aon-co-jp/RPoem) (`open-runo-poem-compat`,
eine Poem-API-kompatible Fassade direkt auf tokio/hyper implementiert — keine
Abhängigkeit vom echten [Poem](https://github.com/poem-web/poem)-Crate,
migriert am 2026-07-31) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
Keine DB-Abhängigkeit, eine einzige eigenständige Binärdatei. Nutzbar von
Rust oder jeder anderen Sprache über einfaches HTTP (dieser Dienst ist die
HTTP-Servierfront für Rust-Ports von Python-KI-Bibliotheken —
`opencuda-bert`/`opencuda-llm`/`opencuda-whisper`, d.h. Transformers/vLLM/
Whisper-Äquivalente).

Siehe [CLAUDE.md](CLAUDE.md) (nur Japanisch) für die Designphilosophie und
[PORTING.md](PORTING.md) (nur Japanisch) dafür, wie diese Muster anderswo
portiert werden.

## Installation

Seit 2026-07-23 gibt es `install.sh` (Linux, registriert einen
systemd-Dienst), `install.ps1` (Windows, zeigt Windows-Dienst-
Registrierungsschritte an) und `.github/workflows/release.yml` (baut bei
jedem Tag-Push Linux-x86_64/Windows-x86_64-Binärdateien und hängt sie an
[GitHub Releases](https://github.com/aon-co-jp/aruaru-llm/releases) an).
**Ehrliche Offenlegung**: Beim Start benötigt diese Binärdatei die 470MB+
großen `multilingual-e5-small`-Modellgewichte (Hugging Face, MIT-Lizenz)
separat abgerufen — aus Lizenzgründen nicht im Installer gebündelt. Der
Build hat eine Sibling-Path-Abhängigkeit von `../open-cuda`, daher muss beim
Bauen aus dem Quellcode `open-cuda` in ein Nachbarverzeichnis geklont werden
(CI tut dies automatisch via `release.yml`). **Seit 2026-07-25**: `/v1/generate`
(GPT-2-124M-Generierung) benötigt zusätzlich `config.json`/
`model.safetensors` (548MB)/`tokenizer.json` (`openai-community/gpt2`, von
Hugging Face) unter `../open-cuda/crates/opencuda-llm/models/gpt2/`
(überschreibbar via `ARUARU_LLM_GPT2_DIR`). Fehlt dies, gibt nur
`/v1/generate` 503 zurück — `/v1/chat` und der Rest des Dienstes funktionieren
weiterhin normal (Verfügbarkeit-zuerst-Design, dieselbe Philosophie wie
`bow_fallback`).

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## Verwandte Projekte

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPU-Runtime (das SET-Pendant)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — erster vorgesehener Aufrufer
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — kanonische Quelle der Entwicklungsrichtlinien
