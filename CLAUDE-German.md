# Designphilosophie & Entwicklungspolitik & Entwicklungsumgebungsregeln (aruaru-llm)

> **Hinweis**: Dies ist eine kondensierte Übersetzung des aktuellen
> Zustands. Das ausführliche historische HANDOFF-Änderungsprotokoll
> (Dutzende von Einträgen) bleibt aus Gründen der Kürze nur auf
> Japanisch in [CLAUDE.md](CLAUDE.md) verfügbar — siehe dort für
> Details zu einzelnen Sitzungen.

GitHub-Repo: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm).

## Rolle dieses Projekts

Ein gemeinsamer, eigenständiger HTTP-Dienst, der die "KI-Chat-Commerce"-
Antwortlogik für das `aruaru`-Ökosystem bereitstellt (aruaru-tokyo,
aruaru-db, e-gov.info, karu.tokyo usw.). Statt dass jede Site ihre
eigene Chat-Antwortlogik implementiert, fragen alle diesen einen
Dienst per HTTP ab — so bleibt die Stelle, die bei einer künftigen
Umstellung auf echte LLM-Inferenz geändert werden muss, an einem
einzigen Ort zentralisiert.

## Ehrliche Offenlegung (wichtig)

`/v1/generate` nutzt seit 2026-07-25 `open-cuda`s `opencuda-llm`-Crate
(echte trainierte GPT-2-124M-Gewichte, `openai-community/gpt2`) für
**echte autoregressive Textgenerierung**. GPT-2 124M ist jedoch ein
kleines Modell aus dem Jahr 2019 und nicht mit modernen kommerziellen
LLMs wie GPT-4 vergleichbar — weder in Fähigkeit noch in Wissen. `/v1/chat`
(Intent-Klassifikation) bleibt separat: `opencuda-bert` (multilingual-
e5-small) berechnet echte Satz-Embeddings und klassifiziert per
Kosinus-Ähnlichkeit mit repräsentativen Intent-Vektoren — eine
**Encoder-basierte semantische Ähnlichkeitsklassifikation**, keine
Dialoggenerierung. Beide Fähigkeiten werden absichtlich nicht
zusammengeführt.

## Aktuelle API-Oberfläche

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(optional)}` →
  `{"reply": "...", "engine": "embedding-cosine-v0-opencuda-bert-cpu",
  "matched_intent": "..."}`.
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(optional, Standard 16, max. 128), "tenant": "..."(optional)}` →
  `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`.
  Fehlen die echten GPT-2-Gewichte, wird ehrlich `503` zurückgegeben
  (kein stiller Fallback wie bei `/v1/chat`).
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — dynamische Tenant-Verwaltung
  (`x-admin-token`-Header-Authentifizierung).
- `GET /healthz` — Health-Check.

### Neu: `POST /v1/generate-speculative` (hinzugefügt 2026-08-17, Commit `8f08900`)

Verlustfreie spekulative Dekodierung im DSpark-Stil über
`open-cuda-llm::GptModel::generate_speculative`, **opt-in** (ersetzt
NICHT das Standard-`/v1/generate`). Nimmt eine `draft_id` entgegen, die
ein bereits heruntergeladenes Katalogmodell (z. B. `"distilgpt2"`) als
Entwurfsmodell benennt. **Kritische ehrliche Offenlegung**: Auf
CPU-Ausführung in `open-cuda` wurde gemessen, dass dieser Pfad selbst
bei 80% Akzeptanzrate **langsamer** ist als das einfache `generate()`
— weil naive CPU-GEMM kaum Dispatch-Overhead hat, den man eliminieren
könnte, sodass die zusätzliche Rechenlast des Entwurfsmodells auf CPU
einen Nettoverlust bedeutet. Eine Geschwindigkeitsverifikation unter
`real-vulkan` (wo Dispatch-Overhead dominiert — der eigentlich
beabsichtigte Anwendungsfall) wurde noch nicht durchgeführt. Ebenfalls
offengelegt: Wiederholungsstrafe (repetition penalty) und MLA-
komprimierte Modelle werden von diesem spekulativen Pfad nicht
unterstützt.

## Tech-Stack

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`, eine Poem-API-kompatible Fassade direkt auf
tokio/hyper implementiert, seit 2026-07-31 statt des echten
[Poem](https://github.com/poem-web/poem)-Crates — kein `Data<T>`-
Extraktor, gemeinsamer Zustand wird per Closure-`Arc::clone` bei der
Routenregistrierung erfasst) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
Keine DB-Abhängigkeit, eine einzige eigenständige Binärdatei.

## "Schattenklon"-Architektur (分身の術)

Wie `open-web-server`: Eine einzelne laufende Instanz wird von
mehreren Domains gemeinsam genutzt, ohne dass eine Installation pro
Domain erforderlich ist (`src/tenants.rs`s `TenantRegistry`,
Laufzeit-Registrierung ohne Neustart über die `/admin/tenants`-APIs).
Die Verwaltung soll künftig von
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) aus
erfolgen (Integration noch nicht verdrahtet).

## Verwandte Projekte

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — GPU-Runtime, das SET-Pendant
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — erster Aufrufer
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — geplante Verwaltung
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — kanonische Quelle der Entwicklungsrichtlinien

---

Weitere Sprachen: [日本語 (Original, mit vollständigem HANDOFF-Verlauf)](CLAUDE.md) ·
[Italiano](CLAUDE-Italian.md) · [Français](CLAUDE-French.md) ·
[Русский](CLAUDE-Russian.md) · [Українська](CLAUDE-Ukrainian.md) ·
[עברית](CLAUDE-Hebrew.md) · [فارسی](CLAUDE-Persian.md) · [العربية](CLAUDE-Arabic.md)

## 21.11.2026 Automatische Genauigkeitswahl und geringer Speicherbedarf (open-cuda, aruaru-llm)
- **Genauigkeit der Gewichte**: Auf einem Rechner mit 32 GB und nur CPU sind INT8-Gewichte die praktische Wahl (etwa viermal schneller als FP8; ein 7B-Modell braucht ungefähr 2 Sekunden pro Token). INT4 nur in den mittleren Schichten ist möglich, war bei anderem Text aber etwa 17 % schlechter. FP8 ist langsam und daher nicht der Standard.
- **Automatische Wahl**: `PrecisionTable::auto_for_memory` nimmt in der Reihenfolge f32, alles INT8, INT4 in den mittleren Schichten die erste Tabelle, die in 70 % des freien Arbeitsspeichers passt. In `aruaru-llm` steuert man das über die Umgebungsvariable `ARUARU_LLM_PRECISION` (nicht gesetzt = f32, `auto` oder eine Tabelle wie `int8,1:f32`).
- **INT8-KV-Cache**: `QWEN_KV_INT8=1` simuliert nur die Rundung; die Perplexität wurde etwa 1 % schlechter. Der Speicherbedarf sinkt dadurch noch nicht.
- **Offen**: den KV-Cache tatsächlich als INT8 speichern und `auto` auf einem echten Server vollständig durchlaufen lassen. Die Übersetzungen stammen von einer KI und wurden noch nicht von Muttersprachlern geprüft.

## 21.11.2026 Nachtrag: KV-Cache wird wirklich als INT8 gespeichert
- Mit `QWEN_KV_INT8=1` wird der KV-Cache jetzt **tatsächlich** als INT8 mit einem Skalierungswert pro Kopf und Token gespeichert (`open-cuda` 2ba4b5f). Bei Qwen2.5-1.5B stieg die Perplexität von 7,079 auf 7,159 (etwa +1 %); bei anderem Text sank sie von 12,1 auf 11,8 (im Rauschen). Der KV-Speicher beträgt etwa ein Viertel von f32 (ein Test mit head_dim 64 zeigt mehr als dreifache Verkleinerung).
- Offen: ein vollständiger Lauf von `ARUARU_LLM_PRECISION=auto` auf einem echten Server. Die Übersetzungen stammen von einer KI und wurden noch nicht von Muttersprachlern geprüft.
