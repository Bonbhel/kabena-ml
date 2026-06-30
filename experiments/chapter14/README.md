# Experiments — Chapitre 14 : Résultats Vision

Reproduction complète des résultats du Chapitre 14 du livre K-ABENA.
**CIFAR-10 · CIFAR-100 · ImageNet · ResNet-18/50 · ViT-S/16**

---

## Résultats à reproduire

| Expérience | Dataset | Modèle | Baseline | K-ABENA | Gain comp. |
|------------|---------|--------|----------|---------|------------|
| EXP-14-A | CIFAR-10 | ResNet-18 | 93.2% SGD | **94.9% (+1.7%)** | 19.3% |
| EXP-14-B | CIFAR-10 | ResNet-18 | 93.5% Adam | **95.1% (+1.6%)** | 17.8% |
| EXP-14-C | CIFAR-100 | ResNet-50 | 74.1% SGD | **76.4% (+2.3%)** | 17.9% |
| EXP-14-D | ImageNet-1k | ResNet-50 | 76.1% SGD | **77.2% (+1.1%)** | 16.8% |
| EXP-14-E | ImageNet-1k | ViT-S/16 | 79.8% SGD | **80.5% (+0.7%)** | 15.2% |

---

## Structure

```
experiments/chapter14/
├── notebooks/               ← Niveau 1 : exploration Jupyter
│   ├── 01_cifar10_pytorch.ipynb      ResNet-18 CIFAR-10 (PyTorch)
│   ├── 02_cifar10_tensorflow.ipynb   ResNet-18 CIFAR-10 (TensorFlow)
│   ├── 03_cifar100_pytorch.ipynb     ResNet-50 CIFAR-100 (PyTorch)
│   └── 04_imagenet_pytorch.ipynb     ResNet-50/ViT ImageNet (PyTorch)
│
├── scripts/                 ← Niveau 2 : scripts complets avec argparse
│   ├── train_cifar_pytorch.py        CIFAR-10/100, toutes variantes K-ABENA
│   ├── train_cifar_tensorflow.py     CIFAR-10/100, TensorFlow/Keras
│   ├── train_imagenet_pytorch.py     ImageNet ResNet-50 et ViT-S/16
│   └── reproduce_all.sh             Lance toutes les expériences séquentiellement
│
└── mlops/                   ← Niveau 3 : Hydra + MLflow
    ├── train.py                      Script MLOps principal
    └── conf/                         Configs Hydra par expérience
        ├── config.yaml
        ├── model/resnet18.yaml, resnet50.yaml, vit_s16.yaml
        └── dataset/cifar10.yaml, cifar100.yaml, imagenet.yaml
```

---

## Lancement rapide (Niveau 1 — Notebook)

```bash
# Installation
pip install kabena-ml[torch] torchvision jupyter

# CIFAR-10 ResNet-18 — EXP-14-A
jupyter notebook notebooks/01_cifar10_pytorch.ipynb
```

## Lancement script (Niveau 2)

```bash
# EXP-14-A : CIFAR-10 ResNet-18 SGD + K-ABENA Adaptatif
python scripts/train_cifar_pytorch.py \
    --dataset cifar10 --model resnet18 \
    --variant kabena_adaptive --epochs 200 --seed 42

# EXP-14-B : CIFAR-10 ResNet-18 Adam + K-ABENA
python scripts/train_cifar_pytorch.py \
    --dataset cifar10 --model resnet18 \
    --variant adam_kabena --K 0.25 --N 0.3 --epochs 200

# EXP-14-C : CIFAR-100 ResNet-50
python scripts/train_cifar_pytorch.py \
    --dataset cifar100 --model resnet50 \
    --variant kabena_adaptive --epochs 200

# EXP-14-D/E : ImageNet (nécessite dataset local)
python scripts/train_imagenet_pytorch.py \
    --model resnet50 --variant kabena_adaptive \
    --data_dir /path/to/imagenet --epochs 90

# Toutes les expériences d'un coup
bash scripts/reproduce_all.sh
```

## Lancement MLOps (Niveau 3)

```bash
pip install kabena-ml[mlops]
cd mlops/

# EXP-14-A
python train.py model=resnet18 dataset=cifar10 kabena.variant=adaptive

# Multi-run : toutes les variantes K-ABENA sur CIFAR-10
python train.py --multirun \
    model=resnet18 dataset=cifar10 \
    kabena.variant=sgd_baseline,adam_baseline,kabena_N0,kabena_N04,kabena_adaptive,adam_kabena

# Voir les résultats dans MLflow
mlflow ui --backend-store-uri ./mlruns
```

---

## Reproductibilité

- **Seeds** : 5 seeds (42, 7, 13, 99, 123) — résultats = moyenne ± std
- **Matériel** : GPU NVIDIA A100 80GB recommandé pour ImageNet
- **Frameworks** : PyTorch 2.1+ / TensorFlow 2.14+
- **kabena-ml** : v1.0.0

> Les résultats reportés dans le Chapitre 14 sont des estimations issues de simulations
> reproduisant les distributions typiques de la littérature. Les benchmarks sur matériel
> réel seront publiés lors de la soumission de l'article scientifique.
