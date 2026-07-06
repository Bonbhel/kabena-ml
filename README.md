# kabena — K-ABENA (v2.1.0)

[![PyPI](https://img.shields.io/pypi/v/kabena.svg)](https://pypi.org/project/kabena/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/kabena/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Bonbhel/kabena-ml/blob/main/LICENSE)
[![Preprint](https://img.shields.io/badge/preprint-PDF-red.svg)](https://github.com/Bonbhel/kabena-ml/blob/main/paper/KABENA_v3_preprint.pdf)

**Même modèle, ~28 % de calcul en moins.** K-ABENA exclut à chaque époque une
partie des exemples déjà appris et fait *voter chaque exemple conservé au nom
des dispensés* (repondération Horvitz-Thompson) : le gradient reste
design-unbiased — preuves et mesures dans le
[preprint](https://github.com/Bonbhel/kabena-ml/blob/main/paper/KABENA_v3_preprint.pdf).

## La promesse : 2 lignes

```python
from kabena import Kabena
kb = Kabena()                                   # 1. défauts du preprint (v3, N=0.3)

for epoch in range(E):
    losses = per_sample_loss(model, X, y)
    active, w = kb.select(losses, y=y)          # 2. masque + poids HT
    model.fit(X[active], y[active], sample_weight=w[active])

print(kb.last_gain_)                            # fraction de calcul économisée
```

PyTorch : `KabenaTorch().reduce(losses)` · Keras : callback `KabenaKeras` ·
Hugging Face : `KabenaTrainer`.

## Trois paramètres, pas un de plus

| Paramètre | Défaut | Rôle |
|---|---|---|
| `N` | `0.3` | proportion de mineures conservées (budget) |
| `strategy` | `"auto"` | `auto`(=v3) / `v3` / `v2` / `v1` — bascule transparente |
| `seed` | `None` | reproductibilité du tirage |

`v2` = mode régularisé optionnel (garde-fou automatique hors zone de
validité) · `v1` = compatibilité 1.x.

## Tutoriels — 4 familles × 3 niveaux

| Famille | Niveau 1 (notebook) | Niveau 2 (script) | Niveau 3 (MLOps) |
|---|---|---|---|
| ML classique (logistique, SVM, softmax) | [notebook](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/classic_ml/niveau1_notebook.ipynb) | [script](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/classic_ml/niveau2_script.py) | [train + serve + Docker](https://github.com/Bonbhel/kabena-ml/tree/main/tutorials/classic_ml/niveau3_mlops) |
| MLP / ANN (PyTorch **et** TF/Keras) | [notebook](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/mlp/niveau1_notebook.ipynb) | [PyTorch](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/mlp/niveau2_script_pytorch.py) · [TensorFlow](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/mlp/niveau2_script_tensorflow.py) | [pipeline TorchScript](https://github.com/Bonbhel/kabena-ml/tree/main/tutorials/mlp/niveau3_mlops) |
| CNN (PyTorch **et** TF/Keras) | [notebook](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/cnn/niveau1_notebook.ipynb) | [PyTorch](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/cnn/niveau2_script_pytorch.py) · [TensorFlow](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/cnn/niveau2_script_tensorflow.py) | [SavedModel / TF Serving](https://github.com/Bonbhel/kabena-ml/tree/main/tutorials/cnn/niveau3_mlops) |
| Transformer / NLP (Hugging Face) | [notebook](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/transformer_nlp/niveau1_notebook.ipynb) | [DistilBERT](https://github.com/Bonbhel/kabena-ml/blob/main/tutorials/transformer_nlp/niveau2_script_huggingface.py) | [fine-tune + serve](https://github.com/Bonbhel/kabena-ml/tree/main/tutorials/transformer_nlp/niveau3_mlops) |

## Reproduire le preprint (Reproducibility)

Chaque table du papier a son script seedé dans
[`validation/`](https://github.com/Bonbhel/kabena-ml/tree/main/validation) :

| Table du preprint | Script |
|---|---|
| Table 1 — jeux réels, parité à coût réduit | [`table1_real_datasets.py`](https://github.com/Bonbhel/kabena-ml/blob/main/validation/table1_real_datasets.py) |
| Table 2 — déséquilibre extrême 0,17 % | [`table2_extreme_imbalance.py`](https://github.com/Bonbhel/kabena-ml/blob/main/validation/table2_extreme_imbalance.py) |
| Table 3 — moments du gradient, bruit | [`table3_moments_noise.py`](https://github.com/Bonbhel/kabena-ml/blob/main/validation/table3_moments_noise.py) |
| Table 4 — rivaux à budget égal (tests de permutation) | [`table4_rivals.py`](https://github.com/Bonbhel/kabena-ml/blob/main/validation/table4_rivals.py) |
| §Sensitivity — grille K × N | [`table5_sensitivity.py`](https://github.com/Bonbhel/kabena-ml/blob/main/validation/table5_sensitivity.py) |

```bash
python validation/run_all.py          # les 5 tables (QUICK=1 pour un smoke test <2 min)
python experiments/playground.py --regime fraude --strategy v3   # bac à sable
```

## Installation

```bash
pip install kabena                     # coeur (numpy seul)
pip install "kabena[sklearn]"          # + helpers scikit-learn
pip install "kabena[torch]"            # + intégration PyTorch
pip install "kabena[tensorflow]"       # + intégration Keras
pip install "kabena[huggingface]"      # + KabenaTrainer
```

Licence MIT — © Jean-François Bonbhel ·
[Changelog](https://github.com/Bonbhel/kabena-ml/blob/main/CHANGELOG.md) ·
[Issues](https://github.com/Bonbhel/kabena-ml/issues)
