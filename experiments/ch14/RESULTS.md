# Résultats Chapitre 14 — Référence pour la reproduction

> **K-ABENA_YekoElite_v8.docx — Chapitre 14 : Résultats Vision**

Ces scripts reproduisent les expériences du Chapitre 14 du livre K-ABENA.
Les résultats ci-dessous sont les cibles à atteindre.

---

## Résultats de référence

### CIFAR-10 — ResNet-18 (200 époques, 5 seeds)

| Méthode             | Top-1 Acc. | Gain comp. | Temps/ép. |
|---------------------|-----------|------------|-----------|
| SGD standard        | 93.2 ±0.1%| 0%         | 12.4s     |
| Adam standard       | 93.5 ±0.1%| 0%         | 13.1s     |
| Focal Loss          | 93.8 ±0.1%| 0%         | 12.6s     |
| OHEM                | 93.9 ±0.2%| 0%         | 14.2s     |
| SBP                 | 94.1 ±0.2%| 0%         | 13.8s     |
| **K-ABENA N=0**     |**94.2 ±0.2%**| **18.5%** | **10.3s** |
| **K-ABENA N=0.4**   |**94.6 ±0.1%**| **14.2%** | **10.8s** |
| **K-ABENA Adaptatif**|**94.9 ±0.1%**| **19.3%**| **10.1s** |
| **Adam + K-ABENA**  |**95.1 ±0.1%**| **17.8%** | **11.4s** |

### CIFAR-100 — ResNet-50

| Méthode             | Top-1  | Top-5  | Gain  |
|---------------------|--------|--------|-------|
| SGD standard        | 74.1%  | 92.3%  | 0%    |
| **K-ABENA Adaptatif**|**76.4%**|**94.1%**|**17.9%**|

### ImageNet-1k

| Modèle + Méthode         | Top-1  | Top-5  | Gain  |
|--------------------------|--------|--------|-------|
| ResNet-50 SGD            | 76.1%  | 92.8%  | 0%    |
| **ResNet-50 + K-ABENA**  |**77.2%**|**93.7%**|**16.8%**|
| ViT-S/16 SGD             | 79.8%  | 94.9%  | 0%    |
| **ViT-S/16 + K-ABENA**   |**80.5%**|**95.4%**|**15.2%**|

---

## Structure des expériences

```
experiments/ch14/
├── RESULTS.md                        ← ce fichier (résultats cibles)
├── niveau1_notebook/
│   ├── 01_cifar10_pytorch.ipynb      ← CIFAR-10 PyTorch N1
│   ├── 02_cifar10_tensorflow.ipynb   ← CIFAR-10 TF N1
│   ├── 03_cifar100_pytorch.ipynb     ← CIFAR-100 PyTorch N1
│   ├── 04_cifar100_tensorflow.ipynb  ← CIFAR-100 TF N1
│   ├── 05_imagenet_pytorch.ipynb     ← ImageNet PyTorch N1
│   └── 06_imagenet_tensorflow.ipynb  ← ImageNet TF N1
├── niveau2_scripts/
│   ├── reproduce_ch14_pytorch.py     ← Script complet PyTorch
│   └── reproduce_ch14_tensorflow.py  ← Script complet TF
└── niveau3_mlops/
    └── reproduce_ch14_mlops.py       ← Hydra + MLflow
```

---

## Usage rapide

```bash
# Niveau 1 — Notebook
jupyter notebook experiments/ch14/niveau1_notebook/01_cifar10_pytorch.ipynb

# Niveau 2 — Script complet
python experiments/ch14/niveau2_scripts/reproduce_ch14_pytorch.py \
    --dataset cifar10 --model resnet18 --variant adaptive --epochs 200

python experiments/ch14/niveau2_scripts/reproduce_ch14_tensorflow.py \
    --dataset cifar10 --variant adaptive --epochs 200

# Niveau 3 — MLOps (Hydra + MLflow)
python experiments/ch14/niveau3_mlops/reproduce_ch14_mlops.py \
    dataset=cifar10 model=resnet18 kabena.variant=adaptive
```

---

## Configuration matérielle recommandée

- GPU : NVIDIA A100 80GB (référence) ou RTX 3090/4090 (résultats comparables)
- RAM : 32 Go minimum pour ImageNet
- Stockage : 150 Go pour ImageNet (dataset complet)
- CIFAR-10/100 : téléchargés automatiquement (~170 Mo chacun)

---

*Référence : Bonbhel, J.-F. (2026). K-ABENA, Chapitre 14. YekoElite University / NeuroSoft IA.*
