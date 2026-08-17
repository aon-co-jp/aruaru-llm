# PORTING.md — Guide de portage d'aruaru-llm (version condensée)

> **Remarque** : ceci est une traduction condensée. Le guide technique
> complet avec les détails du code et les pièges reste disponible
> uniquement en japonais dans [PORTING.md](PORTING.md) — s'y référer
> avant d'adopter réellement un pattern.

Résumé des patterns d'implémentation réutilisables de ce projet, au
cas où ils seraient portés vers un autre projet :

1. **Pattern de couplage avec open-cuda (configuration SET)** :
   dépendance de chemin sur `opencuda-core`/`opencuda-cpu` ; invoque
   une véritable exécution de noyau GPU (`alloc_buffer`→
   `copy_from_host`→`launch_kernel`→`synchronize`→`copy_to_host`).
2. **Classification d'intention basée sur des règles, conçue pour un
   futur remplacement par un vrai LLM** : conserver le champ `engine`
   et toujours y rapporter honnêtement quelle implémentation a
   réellement été utilisée.
3. **Couche API HTTP via RPoem** (`open-runo-poem-compat`) au lieu du
   véritable crate `poem` — pas d'extracteur `Data<T>`, l'état partagé
   est capturé via une closure `Arc::clone`.
4. **Pattern de validation des entrées vides** (2026-08-06) : `400 Bad
   Request` explicite au lieu de laisser filtrer des erreurs internes
   du tokenizer sous forme de `503` trompeur.
5. **Pattern d'enregistrement de tenants « clone d'ombre »** (partagé
   avec `open-web-server`) : `TenantRegistry` + endpoints
   `/admin/tenants`.
6. **Véritable capacité de génération via `opencuda-llm::GptModel`** :
   poids GPT-2 124M — le champ `disclosure` ne doit jamais être omis,
   `/v1/chat` et `/v1/generate` ne doivent pas être fusionnés.
7. **Détection matérielle → taille LLM recommandée → téléchargement
   automatique** (features opt-in `hw-detect-vulkan`/
   `hw-detect-directx`, pattern de vérification croisée, divulgation
   honnête qu'il ne s'agit que d'une heuristique approximative
   taille-vs-VRAM).
8. **Pattern de plugin de traduction** (feature `nllb-translate`) :
   une dépendance lourde optionnelle `rust-bert`/`tch`, isolée
   derrière une feature Cargo, désactivée par défaut.
9. **Feature de dispatch `real-vulkan`** — **Remarque** : pas encore
   recommandée pour un portage ailleurs, en raison d'un bug connu et
   non résolu (`Linear::forward` ne connecte pas les octets SPIR-V de
   `matmul.spv` à `sgemm`, faisant échouer immédiatement
   `GemmPath::VulkanGeneric`).
10. **Pattern de pénalité de répétition**
    (`generate_with_repetition_penalty`, défaut `1.3`, remplaçable
    via une variable d'environnement).

**Mise en garde importante** : GPT-2 124M est petit et date de 2019 —
non comparable aux LLM commerciaux modernes. `/v1/chat` reste basé sur
des règles + classification de similarité par encodeur, pas de la
génération de dialogue neuronale. Cela doit également être divulgué à
chaque destination de portage.

---

Autres langues : [日本語 (original, détails complets)](PORTING.md) ·
[Deutsch](PORTING-German.md) · [Italiano](PORTING-Italian.md) ·
[Русский](PORTING-Russian.md) · [Українська](PORTING-Ukrainian.md) ·
[עברית](PORTING-Hebrew.md) · [فارسی](PORTING-Persian.md) · [العربية](PORTING-Arabic.md)
