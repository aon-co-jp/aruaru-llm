# Philosophie de conception & politique de développement & règles d'environnement de développement (aruaru-llm)

> **Remarque** : ceci est une traduction condensée de l'état actuel. Le
> journal historique détaillé HANDOFF (des dizaines d'entrées) reste
> disponible uniquement en japonais dans [CLAUDE.md](CLAUDE.md), pour
> des raisons de concision — s'y référer pour les détails session par
> session.

Dépôt GitHub : [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm).

## Rôle de ce projet

Un service HTTP partagé et autonome qui fournit la logique de réponse
« AI chat commerce » pour l'écosystème `aruaru` (aruaru-tokyo,
aruaru-db, e-gov.info, karu.tokyo, etc.). Plutôt que chaque site
implémente sa propre logique de réponse de chat, tous interrogent ce
service unique via HTTP — centralisant ainsi l'unique endroit à
modifier lorsqu'une véritable inférence LLM sera un jour intégrée.

## Divulgation honnête (important)

Depuis le 2026-07-25, `/v1/generate` utilise le crate `opencuda-llm`
d'`open-cuda` (poids réels entraînés GPT-2 124M,
`openai-community/gpt2`) pour une **véritable génération de texte
autorégressive**. Cependant, GPT-2 124M est un petit modèle de 2019 et
n'est pas comparable aux LLM commerciaux modernes comme GPT-4, ni en
capacité ni en connaissances. `/v1/chat` (classification d'intention)
reste séparé : `opencuda-bert` (multilingual-e5-small) calcule de
véritables embeddings de phrases et classe par similarité cosinus avec
des vecteurs d'intention représentatifs — une **classification de
similarité sémantique basée sur un encodeur**, pas une génération de
dialogue. Les deux capacités ne sont délibérément pas fusionnées.

## Surface API actuelle

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(optionnel)}` →
  `{"reply": "...", "engine": "embedding-cosine-v0-opencuda-bert-cpu",
  "matched_intent": "..."}`.
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(optionnel, défaut 16, max 128), "tenant": "..."(optionnel)}` →
  `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`.
  Si les poids réels de GPT-2 sont absents, retourne honnêtement `503`
  (pas de repli silencieux comme pour `/v1/chat`).
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — gestion dynamique des tenants
  (authentification par en-tête `x-admin-token`).
- `GET /healthz` — vérification de l'état de santé.

### Nouveau : `POST /v1/generate-speculative` (ajouté le 2026-08-17, commit `8f08900`)

Décodage spéculatif sans perte façon DSpark via
`open-cuda-llm::GptModel::generate_speculative`, **opt-in** (ne
remplace PAS le `/v1/generate` par défaut). Accepte un `draft_id`
désignant un modèle du catalogue déjà téléchargé (par ex.
`"distilgpt2"`) comme modèle brouillon. **Divulgation honnête
critique** : en exécution CPU dans `open-cuda`, il a été mesuré que ce
chemin est **plus lent** que le simple `generate()`, même avec un taux
d'acceptation de 80 % — car le GEMM CPU naïf n'a presque aucun
surcoût de dispatch à éliminer, de sorte que le calcul supplémentaire
du modèle brouillon représente une perte nette sur CPU. La
vérification de vitesse sous `real-vulkan` (où le surcoût de dispatch
domine — le cas d'usage réellement visé) n'a pas encore été effectuée.
Également divulgué : la pénalité de répétition et les modèles
compressés MLA ne sont pas pris en charge par ce chemin spéculatif.

## Pile technologique

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`, une façade compatible avec l'API Poem
implémentée directement sur tokio/hyper, depuis le 2026-07-31 au lieu
du véritable crate [Poem](https://github.com/poem-web/poem) — pas
d'extracteur `Data<T>`, l'état partagé est capturé via une closure
`Arc::clone` lors de l'enregistrement des routes) +
[open-cuda](https://github.com/aon-co-jp/open-cuda). Aucune dépendance
à une base de données, un seul binaire autonome.

## Architecture « clone d'ombre » (分身の術)

Comme `open-web-server` : une seule instance en cours d'exécution est
partagée par plusieurs domaines, sans installation nécessaire par
domaine (`TenantRegistry` dans `src/tenants.rs`, enregistrement à
l'exécution sans redémarrage via les API `/admin/tenants`). La gestion
est prévue depuis
[open-easy-web](https://github.com/aon-co-jp/open-easy-web)
(intégration pas encore câblée).

## Projets liés

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — runtime GPU, le pendant du SET
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — premier appelant
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — gestion prévue
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — source canonique des règles de développement

---

Autres langues : [日本語 (original, avec l'historique HANDOFF complet)](CLAUDE.md) ·
[Deutsch](CLAUDE-German.md) · [Italiano](CLAUDE-Italian.md) ·
[Русский](CLAUDE-Russian.md) · [Українська](CLAUDE-Ukrainian.md) ·
[עברית](CLAUDE-Hebrew.md) · [فارسی](CLAUDE-Persian.md) · [العربية](CLAUDE-Arabic.md)
