# aruaru-llm

*日本語*: [README.md](README.md) ·
*English*: [README-English.md](README-English.md) ·
*Other languages*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Русский](README-Russian.md) · [Українська](README-Ukrainian.md) ·
[עברית](README-Hebrew.md) · [فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **Dernière mise à jour (2026-08-10)** : le nouveau
> `GptModel::generate_with_repetition_penalty` d'`open-cuda` (pénalité
> de répétition façon CTRL) a été branché sur `/v1/generate`, et est
> **activé par défaut** (variable d'environnement
> `ARUARU_LLM_REPETITION_PENALTY`, valeur par défaut `1.3` ; mettre
> `1.0` pour restaurer l'ancien comportement sans pénalité). Cela
> corrige un mode de dégénérescence connu du modèle de base GPT-2 —
> une répétition infinie de la même chaîne — puisque le modèle de base
> n'a pas de fine-tuning de dialogue. Vérifié sur les poids réels
> GPT-2 124M côté `open-cuda` : sans la pénalité, la boucle se
> reproduit effectivement ; avec `penalty=1.3`, elle s'arrête et
> produit un texte conversationnel grammaticalement naturel. Avec
> `penalty=1.0`, la sortie est identique octet pour octet à l'ancien
> `generate()`, donc aucun autre test ne régresse. Voir
> [CLAUDE.md](CLAUDE.md) (japonais uniquement) pour les détails.

> 📌 **Mise à jour (2026-08-08)** : ajout d'un chemin opt-in
> (désactivé par défaut) de compression du cache KV façon MLA
> (implémentation d'`open-cuda` inspirée de DeepSeek-V3) dans
> `/v1/generate` via `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`. Pour
> GPT-2 124M : head_dim=64 -> d_c=16 (75% de stockage KV en moins par
> token). **Divulgation honnête** : les matrices de projection sont
> initialisées aléatoirement (non entraînées), donc la compression est
> avec perte — des tests réels ont montré une qualité de génération
> visiblement dégradée, c'est pourquoi cette option reste désactivée
> par défaut. Une variante calibrée par PCA existe également
> (`ARUARU_LLM_MLA_CALIBRATED=1`), qui évite les boucles de répétition
> dégénérées de la variante aléatoire, mais reste clairement inférieure
> au chemin non compressé — elle aussi reste désactivée par défaut.

> 📌 Tâche en attente (2026-08-06) : il existe un projet d'intégrer les
> techniques Toshiba SBM et DeepSeek. Voir [CLAUDE.md](CLAUDE.md) pour
> les détails.

> 📌 **Mise à jour (2026-08-07)** : vérifié via de vraies requêtes HTTP
> que `/v1/chat` et `/v1/classify-security` **ne** souffrent **pas**
> du bug "entrée vide → 503" précédemment corrigé pour `/v1/generate`
> et `/v1/translate` — les deux renvoient correctement 200 pour une
> entrée vide. Aucune modification de code nécessaire.

Un service de réponse partagé "AI chat commerce" pour l'écosystème
`aruaru` (aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo, etc). Au lieu
que chaque site implémente sa propre logique de réponse de chat, ils
appellent tous ce service HTTP unique — centralisant ainsi l'unique
endroit à modifier lorsqu'une véritable inférence LLM sera un jour
branchée.

> ⚠️ **Divulgation honnête (important, mise à jour le 2026-07-25)** :
> depuis le 2026-07-25, ce service intègre le crate `opencuda-llm`
> d'`open-cuda` (poids réels entraînés GPT-2 124M,
> `openai-community/gpt2`), donc `POST /v1/generate` effectue
> désormais une **véritable génération de texte autorégressive**.
> Toutefois, **GPT-2 124M est un petit modèle de 2019 et n'est pas
> comparable aux LLM commerciaux modernes comme GPT-4** en capacité ou
> en connaissances. Ceci démontre que la génération autonome fonctionne
> sans contrat d'API LLM externe, pas une affirmation de qualité à la
> pointe de l'état de l'art — la sortie est souvent un anglais
> grammaticalement fluide mais n'est pas garantie factuellement exacte
> (elle peut halluciner). `POST /v1/chat` (classification d'intention
> via des embeddings de phrases `opencuda-bert` + similarité cosinus,
> depuis le 2026-07-21) reste un chemin séparé, léger et rapide pour
> des réponses toutes faites — délibérément non fusionné avec la
> génération. Voir [CLAUDE.md](CLAUDE.md) pour les détails et la
> justification.

## Apparié ("SET") avec open-cuda

Dépend, via une dépendance de chemin, des crates
`opencuda-core`/`opencuda-cpu`/`opencuda-blas`/`opencuda-bert` d'
[`open-cuda`](https://github.com/aon-co-jp/open-cuda). À chaque
requête `/v1/chat`, `opencuda-bert` exécute la passe avant de
multilingual-e5-small (en appelant réellement les vrais noyaux
GEMM/Attention d'`opencuda-blas` sur `opencuda_cpu::CpuDevice`) pour
encoder le message, puis le compare via similarité cosinus à
l'embedding représentatif (en cache) de chaque intention. Il s'agit
d'un véritable appel d'exécution à travers le pipeline de calcul
d'open-cuda, pas une simple référence dans `Cargo.toml` — vérifié en
démarrant réellement le serveur et en exécutant `POST /v1/chat`.

Cela dit, ce n'est pas une véritable inférence LLM neuronale
(génération de dialogue) — seulement la passe avant de l'encodeur ; le
décodeur autorégressif reste non implémenté. Les chemins rapides
spécifiques au GPU (`GemmPath::CuBlas`/`RocBlas`/`OneMkl`) restent des
stubs (les chemins CPU et Vulkan générique sont implémentés). Voir la
section HANDOFF du `CLAUDE.md` d'open-cuda pour les détails.

**Mise à jour 2026-07-25 (repli de disponibilité)** : si
`models/multilingual-e5-small/` (470 Mo+) est manquant ou échoue à se
charger, ce service se replie désormais automatiquement sur le produit
scalaire bag-of-words original (`src/bow_fallback.rs`) au lieu de
faire échouer les requêtes purement et simplement. Le champ `engine`
de la réponse `/v1/chat` indique toujours honnêtement quel chemin a
réellement été utilisé (`embedding-cosine-v0-opencuda-bert-cpu` ou
`bow-dotproduct-v0-opencuda-cpu-fallback`) — la qualité de
classification est nettement plus basse sur le chemin de repli
(correspondance de mots-clés, pas de compréhension sémantique).

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(optionnel)}` →
  `{"reply": "...", "engine": "...", "matched_intent": "..."}`
  (classification d'intention, réponses toutes faites légères/rapides)
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(optionnel, défaut 16, plafonné à 128), "tenant":
  "..."(optionnel)}` → `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`
  (véritable génération autorégressive via les poids GPT-2 124M — plus
  lourd mais authentique. **La pénalité de répétition est fixée par
  défaut à `1.3`** — variable `ARUARU_LLM_REPETITION_PENALTY` pour la
  modifier, `1.0` la désactive — pour éviter les boucles de répétition
  infinie. Prompts en anglais recommandés, car le vocabulaire BPE de
  GPT-2 est entraîné principalement sur du texte anglais. Exemple,
  vérifié de bout en bout via une vraie requête HTTP :
  `{"prompt": "The quick brown fox", "max_new_tokens": 16}` →
  `"completion": "es are a great way to get a little bit of a kick out
  of your"`)
- `GET /v1/models/catalog` — modèles compatibles GPT-2 disponibles à
  l'installation (`gpt2`/`distilgpt2`/`gpt2-medium`/`gpt2-large`/
  `gpt2-xl`, ce dernier ajouté le 2026-07-27), lesquels sont déjà
  installés, et le répertoire du modèle actuellement actif.
- `POST /v1/models/install` / `POST /v1/models/select` — télécharger un
  modèle du catalogue depuis Hugging Face, et permuter à chaud le
  modèle actif sans redémarrer le processus.
- `GET /v1/recommend` (ajouté le 2026-07-27) — détecte le matériel
  (VRAM) via `open-cuda` (Vulkan) ou `open-directx` (DXGI) et renvoie
  une taille de modèle recommandée de la famille GPT-2, sans rien
  télécharger.
- `POST /v1/recommend-and-download` (ajouté le 2026-07-27, derrière le
  bouton "Download recommended LLM") — détecte le matériel → choisit
  une taille recommandée → la télécharge depuis Hugging Face si elle
  n'est pas déjà présente (idempotent) → permute `/v1/generate` à chaud
  pour l'utiliser. Renvoie `{"recommendation": {...},
  "already_installed":bool, "switched_to_recommended":bool,
  "message_ja":"..."}`.
- `GET /` (ajouté le 2026-07-27) — interface HTML statique minimale
  (`static/index.html`, sans framework) avec un bouton "Download
  recommended LLM", un affichage de progression, et un panneau de test
  de génération une fois le modèle changé.
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — gestion de l'enregistrement des
  locataires (authentification par en-tête `x-admin-token`)
- `GET /healthz` — contrôle de santé

### Détection matérielle → taille de LLM recommandée (ajouté le 2026-07-27)

`src/hardware.rs` implémente une heuristique simple qui choisit une
taille de la famille GPT-2 (124M/355M/774M/1.5B) à partir de la VRAM
détectée : <2Go → 124M, 2-4Go → 355M, 4-8Go → 774M, 8Go+ → 1.5B ; GPU
non détectable / CPU seul → 124M (repli sûr). **Divulgation honnête** :
il s'agit d'une comparaison approximative taille-vs-VRAM (nombre de
paramètres × 4 octets, estimation fp32), pas d'un modèle de
performance précis — elle ignore la mémoire du cache KV et des
activations.

La détection GPU est opt-in via les features Cargo `hw-detect-vulkan` /
`hw-detect-directx` (désactivées par défaut, pour que les builds CPU
uniquement ou cross-compilés ne soient pas forcés de dépendre du
chargeur Vulkan / du SDK Windows). Lorsqu'elle est activée, Vulkan est
préféré ; si les deux features sont activées, le résultat DXGI
(DirectX) est recoupé avec le résultat Vulkan et journalisé
(`cross_check_agreement`). **Vérifié sur du matériel réel** : en
exécutant avec `--features hw-detect-vulkan` sur la NVIDIA GeForce
GT 730 de cette machine, `vram_bytes=2104819712` a été rapporté —
correspondant exactement à la valeur enregistrée précédemment via DXGI
dans le `CLAUDE.md` d'`open-cuda`, confirmant que les deux chemins de
détection s'accordent sur ce GPU.

### Déchargement du cache KV/des poids façon "Engram" de DeepSeek : étudié et écarté (2026-08-08)

Nous avons étudié si la technique "Engram" de DeepSeek — évincer des
connaissances statiques (cache KV ou fragments de poids) de la VRAM
vers la RAM système et les recharger à la demande — pourrait aider ce
service à fonctionner sur des GPU à faible VRAM comme le GT 730.
**Après avoir lu le code réel, nous avons renoncé à l'implémenter** —
non pas parce que c'est difficile, mais parce que le chemin
d'inférence d'`open-cuda` dont dépend ce dépôt n'a d'abord aucun état
résident en VRAM à évincer. Chaque dispatch GEMM/Attention/softmax
dans `opencuda-blas` (c'est-à-dire chaque appel `sgemm` allant vers
Vulkan) alloue un tampon VRAM via un garde RAII `ScopedAlloc`
(`opencuda-blas/src/lib.rs`), copie hôte→appareil, calcule, copie
appareil→hôte, et libère immédiatement — rien ne reste en VRAM une
fois l'appel terminé. Aussi bien les poids GPT-2 (`word_embeddings` de
`GptModel` et les `Linear` de chaque couche) que le cache KV
(`k`/`v`/`k_latent`/`v_latent` de `open-cuda-llm::KvCacheHead`) sont de
simples `Vec<f32>` qui vivent en RAM système pendant toute leur durée
de vie, même en exécutant avec `--features real-vulkan`. Autrement
dit, cette architecture se retrouve déjà — par accident de conception,
pas par intention — dans l'état que vise Engram : l'état reste résident
en RAM système en permanence, et le GPU n'est touché que
transitoirement, par opération. Ajouter une couche d'éviction LRU par-
dessus n'aurait rien à évincer, donc il n'y aurait aucun effet
mesurable à rapporter (nous n'allons pas revendiquer un bénéfice que
nous ne pouvons pas mesurer). Voir l'entrée HANDOFF du 2026-08-08 dans
CLAUDE.md pour les chemins de code exacts lus.

### Classification vs. génération — laquelle utiliser

`/v1/chat` (classification) et `/v1/generate` (génération) servent des
objectifs différents et ne sont délibérément pas fusionnés : `/v1/chat`
ne fait que router vers des réponses toutes faites et est
léger/rapide (une seule passe avant d'embedding) ; `/v1/generate`
exécute le modèle GPT-2 124M complet (548 Mo de poids) et est plus
lourd mais produit du texte libre authentique. Choisissez celui qui
convient au cas d'usage.

## Architecture "clone d'ombre" (分身の術)

Suivant la même conception qu'`open-web-server` : une seule instance en
cours d'exécution est partagée par plusieurs domaines, sans installation
par domaine requise. La gestion est censée se faire depuis
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) (cette
intégration n'est pas encore branchée). Voir [CLAUDE.md](CLAUDE.md)
pour les détails.

## Stack technique

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`, une façade compatible avec l'API Poem
implémentée directement sur tokio/hyper — aucune dépendance envers le
vrai crate [Poem](https://github.com/poem-web/poem), migré le
2026-07-31) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
Aucune dépendance à une base de données, un unique binaire autonome.
Utilisable depuis Rust ou tout autre langage via du HTTP simple (ce
service est la porte d'entrée HTTP pour les ports Rust de
bibliothèques d'IA Python — `opencuda-bert`/`opencuda-llm`/
`opencuda-whisper`, c'est-à-dire les équivalents de
Transformers/vLLM/Whisper).

Voir [CLAUDE.md](CLAUDE.md) (japonais uniquement) pour la philosophie
de conception et [PORTING.md](PORTING.md) (japonais uniquement) pour
savoir comment porter ces patterns ailleurs.

## Installation

Depuis le 2026-07-23, `install.sh` (Linux, enregistre un service
systemd), `install.ps1` (Windows, affiche les étapes d'enregistrement
du service Windows), et `.github/workflows/release.yml` (construit des
binaires Linux x86_64 / Windows x86_64 à chaque push de tag et les
attache aux [GitHub Releases](https://github.com/aon-co-jp/aruaru-llm/releases))
ont été ajoutés. **Divulgation honnête** : au démarrage, ce binaire a
besoin des poids du modèle `multilingual-e5-small` (470 Mo+, Hugging
Face, licence MIT) récupérés séparément — non fournis avec
l'installateur pour des raisons de licence ; voir `install.sh`/
`install.ps1` pour la commande de téléchargement. Le build a une
dépendance de chemin sibling sur `../open-cuda`, donc compiler depuis
les sources nécessite de cloner `open-cuda` dans un répertoire
adjacent (la CI le fait automatiquement via `release.yml`). **Ajouté
le 2026-07-25** : `/v1/generate` (génération GPT-2 124M) nécessite en
plus `config.json` / `model.safetensors` (548 Mo) / `tokenizer.json`
(`openai-community/gpt2`, depuis Hugging Face) sous
`../open-cuda/crates/opencuda-llm/models/gpt2/` (remplacer le chemin
avec la variable d'environnement `ARUARU_LLM_GPT2_DIR`). S'ils
manquent, seul `/v1/generate` renvoie 503 — `/v1/chat` et le reste du
service continuent de fonctionner normalement (conception privilégiant
la disponibilité, même philosophie que `bow_fallback`).

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## Projets liés

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — runtime GPU (le partenaire du SET)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — premier appelant prévu
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — source canonique de la politique de développement
