# kabena-ml

**K-ABENA** — K-Adaptive Backpropagation with Error-based N-exclusion Algorithm

> *"Gradient intelligent : ignorer les petites erreurs pour apprendre plus vite."*
> — YekoElite University × NeuroSoft IA

[![PyPI version](https://badge.fury.io/py/kabena-ml.svg)](https://pypi.org/project/kabena-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

---

## Installation en 30 secondes

```bash
pip install kabena-ml                # core (NumPy uniquement)
pip install kabena-ml[torch]         # + PyTorch
pip install kabena-ml[tf]            # + TensorFlow
pip install kabena-ml[all]           # tout inclus
```

## Coût d'adoption : 2 lignes de code

**Avant K-ABENA (SGD standard) :**
```python
loss = criterion(model(X), y)
loss.backward()
```

**Après K-ABENA (+2 lignes) :**
```python
losses = criterion(model(X), y, reduction="none")  # ligne 1 modifiée
mask   = kabena_filter(losses, K=0.25)              # ligne 2 ajoutée
losses[mask].mean().backward()                      # ligne 3 inchangée
```

C'est tout. Votre architecture, votre optimiseur, votre pipeline — rien ne change.

---

## Principe

K-ABENA filtre les observations "mineures" (erreur ≤ K) avant la backward pass.

| Paramètre | Rôle | Valeur par défaut |
|-----------|------|-------------------|
| `K` | Seuil absolu — erreurs ≤ K sont "mineures" | auto (calibrate_K) |
| `N` | Proportion de mineures **conservées** (0 = toutes exclues) | 0.0 |

```
N=0  → mode agressif  : toutes les mineures exclues (gain maximal)
N=1  → mode standard  : aucune exclue (= SGD classique)
N=0.4 → mode modéré  : 40% des mineures conservées (régularisation)
```

---

## Tutoriels

| Niveau | Format | Usage |
|--------|--------|-------|
| [Niveau 1](tutorials/niveau1_notebook/) | Jupyter Notebook | Exploration rapide, cours |
| [Niveau 2](tutorials/niveau2_scripts/)  | Scripts `.py`    | Projets complets |
| [Niveau 3](tutorials/niveau3_mlops/)   | MLOps (MLflow + Hydra) | Production |

14 notebooks au total, dont 2 dédiés à la robustesse au bruit avec couverture
PyTorch + TensorFlow/Keras + **HuggingFace `Trainer`** (notebooks 13–14).

---

## Algorithmes couverts

- ✅ Régression linéaire / Ridge / Lasso (scikit-learn)
- ✅ Classification logistique (scikit-learn)
- ✅ SVM linéaire SGD (scikit-learn)
- ✅ XGBoost / Gradient Boosting (sklearn)
- ✅ MLP multicouche (scikit-learn + PyTorch + TensorFlow)
- ✅ CNN — vision (PyTorch + TensorFlow/Keras)
- ✅ Transformer — NLP classification (PyTorch + HuggingFace Trainer)
- ✅ Données déséquilibrées — K-ABENA Stratifié

---

## Robustesse au bruit (v1.1.0)

K-ABENA présente un effet d'auto-protection empirique sous bruit de labels : l'avantage
sur SGD standard s'**amplifie** avec le niveau de bruit (+1.7 pts à 0% de bruit, +8.6 pts
à 30–40%). Détails complets, mécanisme et limites : Chapitre 16B du livre ·
[`calibrate_K_noisy()`](kabena_ml/core/filter.py) ·
[notebooks 13–14](tutorials/niveau1_notebook/).

> ⚠️ **Limite L10 — à lire avant usage en production.** Les valeurs par défaut de
> `calibrate_K_noisy()` (`noise_factor=2.0`, `q_cap=25.0`) ont été observées sur
> **deux configurations seulement** (CIFAR-10, SST-2) — ce n'est **pas** une constante
> universelle validée, contrairement à `N=0.3` (validé sur 8 datasets). **Constat
> additionnel vérifié** : avec ces valeurs par défaut, le plafond `q_cap=25%` est
> atteint dès 7.5% de bruit estimé — au-delà, toute variation de bruit produit
> exactement le même K. Toujours valider `noise_factor` via une grille
> `[1.0, 1.5, 2.0, 2.5, 3.0]` sur votre propre dataset avec bruit connu avant de
> déployer en production.

```python
from kabena_ml import calibrate_K_noisy

# Calibrage adapté au bruit estimé (30% de labels corrompus)
K = calibrate_K_noisy(losses_epoch1, estimated_noise_pct=0.30)
```

---

## Auteur

**M. Bonbhel** — YekoElite University × NeuroSoft IA  
bonbhel@gmail.com | +1 418 271 0819  
GitHub : [Bonbhel/kabena-ml](https://github.com/Bonbhel/kabena-ml)

---

*Référence : Bonbhel, J.-F. (2026). "Gradient intelligent : la théorie K-ABENA pour un Machine Learning plus efficace." YekoElite University.*
