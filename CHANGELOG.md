# Changelog — kabena

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

## [2.1.0] — 2026-07-05

Saut de version public : **1.2.0 → 2.1.0**. La 2.0.0 fut un jalon interne
(refonte v2) jamais publié sur GitHub ni déployé ; ses changements sont
inclus et remplacés ici.

### Ajouté
- **Stratégie v3 canonique** (preprint, Définition 1) : tirage à mélange
  défensif sur tout M_K (p_i = α/k + (1−α)·ε_i/Σε) + repondération
  Horvitz-Thompson auto-normalisée. Estimateur (quasi) sans biais —
  Lemmas 2-3 du preprint ; résout l'échec en déséquilibre extrême
  (AUC 0,9991 vs 0,53 pour v2, à 28,4 % d'économie identique).
- **API 2 lignes** : `kb = Kabena()` puis `active, w = kb.select(losses)`.
  Trois paramètres exposés (N, strategy, seed) ; K auto-calibré (percentile),
  α en avancé. Contrat de sortie identique pour v1/v2/v3 → bascule transparente.
- **Garde-fou v2→v3** (`strategy="v2"` + y déséquilibré < 5 % → warning et
  bascule v3 ; `.force()` pour assumer). Contre-indications du preprint §6.
- **Intégrations** : `integrations.sklearn` (helper SGDClassifier),
  `integrations.torch` (`KabenaTorch.reduce`), `integrations.keras`
  (callback `KabenaKeras` + sample_weight), `integrations.huggingface`
  (`KabenaTrainer`). Importables sans leurs dépendances.
- **validation/** : scripts reproduisant les tables du preprint (seedés,
  `QUICK=1` pour smoke test), `run_all.py` — dont `table4_rivals.py`
  (rivaux à budget égal : Focal, OHEM-style, IS global ; IC 95 % et tests
  de permutation appariés sign-flip) et `table5_sensitivity.py`
  (grille K-percentile × N), ajoutés lors de la révision peer-review.
- **experiments/playground.py** : bac à sable CLI des gains ET des limites
  (régimes standard/fraude/bruit, sweep α et N, garde-fou observable).
- **tutorials/** : 4 familles (classic_ml, mlp, cnn, transformer_nlp) ×
  3 niveaux (notebook, script .py, MLOps train+serve+Dockerfile) ;
  DL couvert sur PyTorch, TensorFlow/Keras et Hugging Face.
- **families/** : point d'extension documenté pour KNN/arbres/forêts
  (docs/EXTENDING.md) — quais réservés, non implémentés.

### Changé (rupture vs 1.2.0)
- `Kabena` (classe) devient l'API primaire ; `select` retourne
  `(active, weights)` — les poids sont nécessaires à v3.
- v2 (2.0.0 interne) reste disponible comme **mode régularisé optionnel**
  avec zone de validité stricte (signal minoritaire ≥ 5 %, bruit ≤ 25 %,
  N ≤ 0,5) — s'effondre mesurément hors zone (0,386 à 40 % de bruit).

### Inchangé / rétro-compatibilité
- `kabena_filter(...)` et `kabena_safe(...)` conservés avec défaut
  `strategy='v1'` : le code 1.x fonctionne sans modification.

### Périmètre validé (honnêteté L4)
- Exécuté et testé dans l'environnement de release : cœur NumPy,
  intégration sklearn, validation/, playground, MLOps classic_ml (13/13 tests).
- Vérifié syntaxiquement mais NON exécuté (GPU/deps absents) : intégrations
  et tutoriels torch / tensorflow / huggingface. Retours bienvenus via Issues.

## [1.2.0] — version publique antérieure (v1 uniquement)
