# Tutoriels K-ABENA — Guide des niveaux

## Philosophie : coût syntaxique minimal

K-ABENA s'intègre dans votre code existant **sans le réécrire**.  
La différence entre "pas de K-ABENA" et "avec K-ABENA" est mesurable en nombre de lignes modifiées.

---

## Niveau 1 — Jupyter Notebook (exploration)

**Audience :** étudiants, chercheurs, exploration rapide  
**Format :** `.ipynb` — copier-coller dans Colab, Kaggle, JupyterLab

| Fichier | Algorithme | Lignes ajoutées |
|---------|-----------|-----------------|
| `01_regression_lineaire.ipynb` | Ridge regression (sklearn) | +3 lignes |
| `02_classification_logistique.ipynb` | LogisticRegression (sklearn) | +2 lignes |
| `03_mlp_pytorch.ipynb` | MLP multicouche (PyTorch) | +2 lignes |
| `04_cnn_vision_pytorch.ipynb` | CNN CIFAR-10 (PyTorch) | +2 lignes |
| `05_nlp_tensorflow.ipynb` | NLP classification (TF/Keras) | +1 ligne (callback) |

### Lancer un notebook

```bash
pip install kabena-ml[all] jupyter
jupyter notebook tutorials/niveau1_notebook/
```

---

## Niveau 2 — Scripts Python (projets complets)

**Audience :** développeurs, data scientists, projets réels  
**Format :** `.py` avec argparse — prêts pour la ligne de commande

### sklearn_project.py
```bash
# Régression
python sklearn_project.py --task regression --K 0.20 --N 0.3 --plot

# Classification
python sklearn_project.py --task classification --K 0.20 --grid

# Données déséquilibrées
python sklearn_project.py --task imbalanced --K 0.20 --N 0.4
```

### pytorch_project.py
```bash
# MLP avec auto-calibrage de K
python pytorch_project.py --model mlp --epochs 100 --plot

# CNN CIFAR-10
python pytorch_project.py --model cnn --K 0.30 --N 0.4

# XGBoost avec objectif K-ABENA
python pytorch_project.py --model xgboost --K 0.10
```

### tensorflow_project.py
```bash
# Keras callback (+1 ligne)
python tensorflow_project.py --mode callback --K 0.25 --N 0.3

# GradientTape (contrôle total)
python tensorflow_project.py --mode tape --K 0.25 --epochs 50 --plot
```

---

## Niveau 3 — MLOps (Hydra + MLflow)

**Audience :** MLOps engineers, production, expérimentation systématique  
**Format :** Hydra config + MLflow tracking + DVC (optionnel)

### Installation
```bash
pip install kabena-ml[mlops]
```

### Usage
```bash
cd tutorials/niveau3_mlops/

# Entraînement avec config par défaut
python train_mlops.py

# Override de paramètres K-ABENA
python train_mlops.py kabena.K=0.25 kabena.N=0.3

# Changer le modèle et le dataset
python train_mlops.py model=logistic dataset=housing training.epochs=200

# Multi-run : exploration automatique de la grille K × N
python train_mlops.py --multirun kabena.K=0.10,0.20,0.30 kabena.N=0.0,0.3,0.6
```

### Suivi MLflow
```bash
# Démarrer l'interface MLflow
mlflow ui --backend-store-uri ./mlruns

# Ouvrir http://localhost:5000
```

### Structure des configs
```
conf/
├── config.yaml          ← config principale
├── model/
│   ├── mlp.yaml         ← MLP PyTorch
│   └── logistic.yaml    ← LogisticRegression sklearn
└── dataset/
    ├── classification.yaml
    └── housing.yaml
```

---

## Tableau récapitulatif — coût d'adoption

| Framework | Avant K-ABENA | Après K-ABENA | Lignes ajoutées |
|-----------|--------------|---------------|-----------------|
| sklearn | `model.fit(X, y)` | `KabenaWrapper(model).fit(X, y)` | +1 |
| PyTorch boucle | `loss = F(logits, y)` | `losses = F(..., reduction="none"); mask = kabena_filter_torch(losses, K)` | +2 |
| PyTorch trainer | boucle manuelle | `KabenaTrainer(model, cfg).fit(X, y)` | +2 |
| TF/Keras | `model.fit(X, y, ...)` | `model.fit(..., callbacks=[KabenaCallback(K)])` | +1 |
| TF GradientTape | `L.backward()` | `mask = trainer.filter(losses); L[mask].mean()...` | +2 |
