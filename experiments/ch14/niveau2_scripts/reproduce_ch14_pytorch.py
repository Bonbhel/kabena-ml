"""
experiments/ch14/niveau2_scripts/reproduce_ch14_pytorch.py
============================================================
Script complet de reproduction des résultats du Chapitre 14 (PyTorch).
Couvre : CIFAR-10 (ResNet-18), CIFAR-100 (ResNet-50), ImageNet (ResNet-50, ViT-S/16).

Usage :
    # CIFAR-10 — toutes les variantes
    python reproduce_ch14_pytorch.py --dataset cifar10 --model resnet18

    # CIFAR-10 — variante spécifique
    python reproduce_ch14_pytorch.py --dataset cifar10 --model resnet18 --variant adaptive

    # CIFAR-100
    python reproduce_ch14_pytorch.py --dataset cifar100 --model resnet50

    # ImageNet (nécessite le dataset local)
    python reproduce_ch14_pytorch.py --dataset imagenet --model resnet50 --imagenet-path /data/imagenet

    # Reproduire tous les résultats du ch.14 (peut prendre plusieurs heures sur GPU)
    python reproduce_ch14_pytorch.py --all

Variantes disponibles : standard | ka_n0 | ka_n4 | adaptive | adam_ka | ohem | focal
"""

from __future__ import annotations
import argparse, os, sys, time, json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# kabena-ml
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kabena.core.filter import calibrate_K
from kabena.integrations.torch_utils import kabena_filter_torch


# ═══════════════════════════════════════════════════════════════════════════
# CIBLES CH.14 (valeurs de référence du livre)
# ═══════════════════════════════════════════════════════════════════════════
TARGETS_CH14 = {
    "cifar10": {
        "standard":  {"top1": 93.2, "gain": 0.0,  "time": 12.4},
        "ka_n0":     {"top1": 94.2, "gain": 18.5, "time": 10.3},
        "ka_n4":     {"top1": 94.6, "gain": 14.2, "time": 10.8},
        "adaptive":  {"top1": 94.9, "gain": 19.3, "time": 10.1},
        "adam_ka":   {"top1": 95.1, "gain": 17.8, "time": 11.4},
        "ohem":      {"top1": 93.9, "gain": 0.0,  "time": 14.2},
        "focal":     {"top1": 93.8, "gain": 0.0,  "time": 12.6},
    },
    "cifar100": {
        "standard": {"top1": 74.1, "top5": 92.3, "gain": 0.0},
        "adaptive": {"top1": 76.4, "top5": 94.1, "gain": 17.9},
    },
    "imagenet": {
        "standard":  {"top1": 76.1, "top5": 92.8, "gain": 0.0},
        "adaptive":  {"top1": 77.2, "top5": 93.7, "gain": 16.8},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

def get_loaders(dataset: str, batch_size: int, imagenet_path: str = "") -> tuple:
    """Retourne (train_loader, test_loader)."""

    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_tfm = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(),
                                T.ToTensor(), T.Normalize(mean, std)])
        test_tfm  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds  = torchvision.datasets.CIFAR10("./data", True,  True, transform=train_tfm)
        test_ds   = torchvision.datasets.CIFAR10("./data", False, True, transform=test_tfm)

    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_tfm = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(),
                                T.ToTensor(), T.Normalize(mean, std)])
        test_tfm  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds  = torchvision.datasets.CIFAR100("./data", True,  True, transform=train_tfm)
        test_ds   = torchvision.datasets.CIFAR100("./data", False, True, transform=test_tfm)

    elif dataset == "imagenet":
        assert imagenet_path, "Spécifier --imagenet-path /path/to/imagenet"
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_tfm = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                                T.ToTensor(), normalize])
        test_tfm  = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
        train_ds  = torchvision.datasets.ImageFolder(f"{imagenet_path}/train", transform=train_tfm)
        test_ds   = torchvision.datasets.ImageFolder(f"{imagenet_path}/val",   transform=test_tfm)
    else:
        raise ValueError(f"Dataset inconnu: {dataset}")

    n_workers = min(os.cpu_count() or 1, 8)
    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=n_workers, pin_memory=True),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=n_workers, pin_memory=True))


# ═══════════════════════════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════════════════════════

def get_model(name: str, n_classes: int, device: torch.device) -> nn.Module:
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc      = nn.Linear(512, n_classes)
    elif name == "resnet50":
        m = torchvision.models.resnet50(weights=None)
        if n_classes != 1000:
            m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
        m.fc = nn.Linear(2048, n_classes)
    elif name == "vit_s16":
        m = torchvision.models.vit_s_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
    else:
        raise ValueError(f"Modèle inconnu: {name}")
    return m.to(device)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINES ALTERNATIVES (OHEM et Focal Loss)
# ═══════════════════════════════════════════════════════════════════════════

def ohem_loss(losses: torch.Tensor, ratio_keep: float = 0.7) -> torch.Tensor:
    """OHEM : garder les ratio_keep exemples avec la PLUS GRANDE perte."""
    k = max(1, int(len(losses) * ratio_keep))
    top_losses, _ = losses.topk(k)
    return top_losses.mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Focal Loss (Lin et al. 2017) : (1-p)^gamma pondère la CE."""
    ce     = F.cross_entropy(logits, targets, reduction="none")
    p_t    = torch.exp(-ce)
    fl     = alpha * (1 - p_t) ** gamma * ce
    return fl.mean()


# ═══════════════════════════════════════════════════════════════════════════
# BOUCLE D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════

class KabenaScheduler:
    """Warm-up progressif de K (Ch.12 — stratégie recommandée)."""
    def __init__(self, q_init=5, q_target=20, T_warmup=20):
        self.q_init, self.q_target, self.T_warmup = q_init, q_target, T_warmup

    def step(self, losses_np: np.ndarray, epoch: int) -> float:
        q = self.q_init + (self.q_target - self.q_init) * min(epoch / self.T_warmup, 1.0)
        return float(np.percentile(losses_np, q))


def train_epoch(model, loader, optimizer, lr_sched, variant, K, N, sched_K, epoch, device):
    model.train()
    total_m, total_n, ep_losses = 0, 0, []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Forward — identique pour TOUTES les variantes
        logits = model(X)
        losses = F.cross_entropy(logits, y, reduction="none")  # ← toujours 'none'

        # ── Sélection de la perte selon la variante ─────────────────────────
        if variant == "standard":
            L = losses.mean()
            m = len(y)

        elif variant in ("ka_n0", "ka_n4", "adam_ka"):
            mask = kabena_filter_torch(losses, K=K, N=N)       # ← +1 ligne K-ABENA
            m    = mask.sum().item()
            L    = losses[mask].mean() if m > 0 else losses.mean()

        elif variant == "adaptive":
            K_t  = sched_K.step(losses.detach().cpu().numpy(), epoch)
            mask = kabena_filter_torch(losses, K=K_t, N=N)     # ← +1 ligne K-ABENA
            m    = mask.sum().item()
            L    = losses[mask].mean() if m > 0 else losses.mean()
            K    = K_t  # pour le log

        elif variant == "ohem":
            L = ohem_loss(losses, ratio_keep=0.75)
            m = int(len(y) * 0.75)

        elif variant == "focal":
            L = focal_loss(logits, y)
            m = len(y)
        # ────────────────────────────────────────────────────────────────────

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        total_m += m
        total_n += len(y)
        ep_losses.append(L.item())

    lr_sched.step()
    gain = round((1 - total_m / total_n) * 100) if total_n > 0 else 0
    return float(np.mean(ep_losses)), gain, K


@torch.no_grad()
def evaluate(model, loader, device, topk=(1, 5)):
    model.eval()
    correct = {k: 0 for k in topk}
    total   = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        for k in topk:
            k_eff = min(k, out.size(1))
            _, pred = out.topk(k_eff, 1, True, True)
            correct[k] += pred.eq(y.view(-1, 1).expand_as(pred)).any(1).sum().item()
        total += len(y)
    return {k: correct[k] / total * 100 for k in topk}


# ═══════════════════════════════════════════════════════════════════════════
# EXPÉRIENCE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    dataset:      str,
    model_name:   str,
    variant:      str,
    epochs:       int,
    batch_size:   int,
    seed:         int,
    device:       torch.device,
    imagenet_path: str = "",
    verbose:      bool = True,
) -> dict:
    """Lance une expérience complète et retourne les résultats."""

    # Paramètres K-ABENA selon la variante
    VARIANT_PARAMS = {
        "standard": {"K": None,  "N": 0.0, "opt": "sgd"},
        "ka_n0":    {"K": None,  "N": 0.0, "opt": "sgd"},
        "ka_n4":    {"K": None,  "N": 0.4, "opt": "sgd"},
        "adaptive": {"K": None,  "N": 0.3, "opt": "sgd"},
        "adam_ka":  {"K": None,  "N": 0.3, "opt": "adam"},
        "ohem":     {"K": None,  "N": 0.0, "opt": "sgd"},
        "focal":    {"K": None,  "N": 0.0, "opt": "sgd"},
    }
    params = VARIANT_PARAMS[variant]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Données
    n_classes = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}[dataset]
    train_loader, test_loader = get_loaders(dataset, batch_size, imagenet_path)

    # Modèle
    model = get_model(model_name, n_classes, device)

    # Optimiseur
    if params["opt"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sched_K  = KabenaScheduler() if variant == "adaptive" else None
    K        = params["K"]
    N        = params["N"]

    history = []

    for epoch in range(epochs):
        t0 = time.time()

        # Calibrage auto à l'époque 0
        if epoch == 0 and K is None and variant not in ("standard", "ohem", "focal"):
            model.eval()
            sample_losses = []
            with torch.no_grad():
                for X_b, y_b in train_loader:
                    l = F.cross_entropy(model(X_b.to(device)), y_b.to(device), reduction="none")
                    sample_losses.extend(l.cpu().numpy())
                    if len(sample_losses) > 5000:
                        break
            K = calibrate_K(np.array(sample_losses), target_pct=0.10)
            if verbose:
                print(f"  [Époque 0] K calibré = {K:.4f}")

        loss, gain, K = train_epoch(model, train_loader, optimizer, lr_sched,
                                     variant, K, N, sched_K, epoch, device)
        accs = evaluate(model, test_loader, device)
        dt   = time.time() - t0

        record = {"epoch": epoch, "loss": loss, "top1": accs[1],
                  "top5": accs.get(5, 0.0), "gain": gain, "time": dt, "K": K}
        history.append(record)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  [{variant}] Ép {epoch+1:3d}/{epochs} | "
                  f"Top-1={accs[1]:.2f}% | Top-5={accs.get(5,0):.2f}% | "
                  f"gain={gain}% | {dt:.1f}s/ép")

    final = history[-1]
    mean_gain = float(np.mean([r["gain"] for r in history if r["gain"] > 0] or [0]))
    mean_time = float(np.mean([r["time"] for r in history]))

    return {
        "dataset":   dataset,
        "model":     model_name,
        "variant":   variant,
        "seed":      seed,
        "top1":      final["top1"],
        "top5":      final["top5"],
        "gain":      mean_gain,
        "time_ep":   mean_time,
        "K_final":   K,
        "N":         N,
        "history":   history,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ═══════════════════════════════════════════════════════════════════════════

def print_report(results: list[dict], dataset: str):
    targets = TARGETS_CH14.get(dataset, {})
    print("\n" + "=" * 75)
    print(f"  RÉSULTATS CH.14 — {dataset.upper()} (PyTorch) vs Cibles du livre")
    print("=" * 75)
    fmt = "{:<18} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("Variante", "Top-1 obt.", "Top-1 cib.", "Δ", "Gain", "Temps/ép"))
    print("-" * 75)
    for r in results:
        v    = r["variant"]
        tgt  = targets.get(v, {})
        t1_t = tgt.get("top1", "-")
        delta = f"{r['top1'] - t1_t:+.2f}%" if isinstance(t1_t, float) else "N/A"
        print(fmt.format(
            v,
            f"{r['top1']:.2f}%",
            f"{t1_t:.1f}%" if isinstance(t1_t, float) else "-",
            delta,
            f"{r['gain']:.1f}%",
            f"{r['time_ep']:.1f}s"
        ))
    print("=" * 75)
    print("Note : Δ = différence avec la cible du livre K-ABENA v8.")


def save_results(results: list[dict], output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "results_ch14_pytorch.json"
    with open(out_path, "w") as f:
        # Exclure les historiques pour compacité (sauf si souhaité)
        compact = [{k: v for k, v in r.items() if k != "history"} for r in results]
        json.dump(compact, f, indent=2)
    print(f"\nRésultats sauvegardés : {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reproduction Ch.14 K-ABENA — PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--dataset",       default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--model",         default="resnet18",
                        choices=["resnet18", "resnet50", "vit_s16"])
    parser.add_argument("--variant",       default="all",
                        choices=["all", "standard", "ka_n0", "ka_n4",
                                 "adaptive", "adam_ka", "ohem", "focal"])
    parser.add_argument("--epochs",        type=int, default=None,
                        help="Nombre d'époques (défaut: 200 CIFAR, 90 ImageNet)")
    parser.add_argument("--batch-size",    type=int, default=128)
    parser.add_argument("--seeds",         type=int, nargs="+", default=[42],
                        help="Seeds pour la reproductibilité (ex: --seeds 42 7 13)")
    parser.add_argument("--imagenet-path", default="",
                        help="Chemin vers ImageNet-1k (requis pour --dataset imagenet)")
    parser.add_argument("--output-dir",    default="./results",
                        help="Dossier de sauvegarde des résultats JSON")
    parser.add_argument("--all",           action="store_true",
                        help="Lancer toutes les variantes du ch.14")
    parser.add_argument("--no-top5",       action="store_true",
                        help="Ne pas calculer Top-5 (plus rapide)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device} | Dataset: {args.dataset} | Modèle: {args.model}")

    # Époques par défaut
    default_epochs = 200 if args.dataset in ("cifar10", "cifar100") else 90
    epochs = args.epochs or default_epochs
    print(f"Époques: {epochs} | Seeds: {args.seeds} | Batch: {args.batch_size}")

    # Variantes à lancer
    if args.all or args.variant == "all":
        variants = list(TARGETS_CH14.get(args.dataset, {}).keys())
    else:
        variants = [args.variant]

    # Modèle adapté au dataset
    model = args.model
    if args.dataset == "imagenet" and model == "resnet18":
        model = "resnet50"  # ResNet-18 non utilisé sur ImageNet dans ch.14
        print(f"Note : modèle ajusté à resnet50 pour ImageNet (ch.14).")

    # Lancer les expériences
    all_results = []
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  Variante : {variant.upper()}")
        print(f"{'='*60}")
        for seed in args.seeds:
            print(f"\n  → Seed {seed}")
            res = run_experiment(
                dataset=args.dataset, model_name=model, variant=variant,
                epochs=epochs, batch_size=args.batch_size, seed=seed,
                device=device, imagenet_path=args.imagenet_path, verbose=True
            )
            all_results.append(res)

    # Rapport (moyenne sur les seeds)
    from itertools import groupby
    by_variant = {}
    for r in all_results:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = []
    for v, runs in by_variant.items():
        summary.append({
            "variant": v,
            "top1":    float(np.mean([r["top1"] for r in runs])),
            "top5":    float(np.mean([r["top5"] for r in runs])),
            "gain":    float(np.mean([r["gain"] for r in runs])),
            "time_ep": float(np.mean([r["time_ep"] for r in runs])),
            "top1_std": float(np.std([r["top1"] for r in runs])),
        })

    print_report(summary, args.dataset)
    save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
