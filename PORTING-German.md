# PORTING.md — Leitfaden zur Portierung von aruaru-llm (Kurzfassung)

> **Hinweis**: Dies ist eine kondensierte Übersetzung. Die vollständige
> technische Anleitung mit Code-Details und Fallstricken bleibt nur auf
> Japanisch in [PORTING.md](PORTING.md) verfügbar — dort nachschlagen,
> bevor ein Muster tatsächlich übernommen wird.

Zusammenfassung der wiederverwendbaren Implementierungsmuster aus
diesem Projekt, falls sie in ein anderes Projekt portiert werden:

1. **open-cuda-Kopplungsmuster (SET-Konfiguration)**: Pfadabhängigkeit
   auf `opencuda-core`/`opencuda-cpu`; ruft echte GPU-Kernel-
   Ausführung auf (`alloc_buffer`→`copy_from_host`→`launch_kernel`→
   `synchronize`→`copy_to_host`).
2. **Regelbasierte Intent-Klassifikation, ausgelegt für einen
   künftigen Austausch gegen ein echtes LLM**: das `engine`-Feld
   beibehalten und darin immer ehrlich melden, welche Implementierung
   tatsächlich verwendet wurde.
3. **HTTP-API-Schicht über RPoem** (`open-runo-poem-compat`) statt des
   echten `poem`-Crates — kein `Data<T>`-Extraktor, gemeinsamer
   Zustand wird per Closure-`Arc::clone` erfasst.
4. **Muster zur Validierung leerer Eingaben** (2026-08-06): explizites
   `400 Bad Request` statt interne Tokenizer-Fehler als irreführendes
   `503` durchsickern zu lassen.
5. **"Schattenklon"-Tenant-Registrierungsmuster** (gemeinsam mit
   `open-web-server`): `TenantRegistry` + `/admin/tenants`-Endpunkte.
6. **Echte Generierungsfähigkeit via `opencuda-llm::GptModel`**:
   GPT-2-124M-Gewichte — das `disclosure`-Feld darf niemals
   ausgelassen werden, `/v1/chat` und `/v1/generate` dürfen nicht
   zusammengeführt werden.
7. **Hardware-Erkennung → empfohlene LLM-Größe → Auto-Download**
   (`hw-detect-vulkan`/`hw-detect-directx` opt-in Features,
   Cross-Check-Muster, ehrliche Offenlegung, dass dies nur eine grobe
   Größe-vs-VRAM-Heuristik ist).
8. **Übersetzungs-Plugin-Muster** (`nllb-translate`-Feature): eine
   optionale, schwere `rust-bert`/`tch`-Abhängigkeit, isoliert hinter
   einem Cargo-Feature, standardmäßig aus.
9. **`real-vulkan`-Dispatch-Feature** — **Hinweis**: für eine
   Portierung anderswo noch nicht empfohlen, wegen eines bekannten,
   ungelösten Bugs (`Linear::forward` verdrahtet die `matmul.spv`-
   SPIR-V-Bytes nicht mit `sgemm`, wodurch `GemmPath::VulkanGeneric`
   sofort fehlschlägt).
10. **Wiederholungsstrafe-Muster** (`generate_with_repetition_penalty`,
    Standard `1.3`, per Umgebungsvariable überschreibbar).

**Wichtiger Vorbehalt**: GPT-2 124M ist klein und stammt aus 2019 —
nicht vergleichbar mit modernen kommerziellen LLMs. `/v1/chat` bleibt
regelbasiert + Encoder-Ähnlichkeitsklassifikation, keine neuronale
Dialoggenerierung. Dies muss auch an jedem Portierungsziel offengelegt
werden.

---

Weitere Sprachen: [日本語 (Original, vollständige Details)](PORTING.md) ·
[Italiano](PORTING-Italian.md) · [Français](PORTING-French.md) ·
[Русский](PORTING-Russian.md) · [Українська](PORTING-Ukrainian.md) ·
[עברית](PORTING-Hebrew.md) · [فارسی](PORTING-Persian.md) · [العربية](PORTING-Arabic.md)
