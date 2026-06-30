"""
tutorials/niveau2_scripts/pytorch_project.py
=============================================
Projet complet PyTorch avec K-ABENA.
Couvre : MLP, CNN vision, gradient boosting (résidu).

Usage :
    python pytorch_project.py --model mlp
    python pytorch_project.py --model cnn
    python pytorch_project.py --model xgboost
    python pytorch_project.py --model mlp --K 0.25 --N 0.3 --epochs 100 --plot
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from kabena_ml import calibrate_K, KabenaConfig
from kabena_ml.integrations.torch_utils import (
    kabena_filter_torch, kabena_safe_torch, KabenaTrainer
)
from kabena_ml.utils.logger import KabenaLogger, plot_stats


# ─── Architectures ────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), n_classes=2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """CNN pour images 32×32 couleur (ex : CIFAR-10)."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── MLP ─────────────────────────────────────────────────────────────────────
def run_mlp(args):
    print("\n" + "="*60)
    print("K-ABENA — MLP multicouche (PyTorch)")
    print("="*60)

    # Données
    X_np, y_np = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_np = StandardScaler().fit_transform(X_np)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Calibrage auto
    model0 = MLP(20)
    with torch.no_grad():
        losses0 = F.cross_entropy(model0(X_tr), y_tr, reduction="none")
    K_auto = calibrate_K(losses0.numpy())
    print(f"K calibré : {K_auto:.4f}")

    K_ = args.K if args.K else K_auto

    # ── Option A : boucle manuelle K-ABENA (transparence totale) ────────────
    model_A  = MLP(20)
    opt_A    = torch.optim.SGD(model_A.parameters(), lr=0.05, momentum=0.9)
    logger_A = KabenaLogger("./logs/")
    history_A = []

    for epoch in range(args.epochs):
        losses = F.cross_entropy(model_A(X_tr), y_tr, reduction="none")  # ← "none"
        mask   = kabena_filter_torch(losses, K=K_, N=args.N)             # ← +1 ligne
        if not mask.any():
            continue
        L_KA = losses[mask].mean()
        opt_A.zero_grad(); L_KA.backward(); opt_A.step()

        m    = mask.sum().item()
        gain = round((1 - m/len(y_tr))*100)
        history_A.append({"epoch": epoch, "loss": L_KA.item(),
                           "m": m, "n": len(y_tr), "gain_pct": gain})
        logger_A.log(epoch, L_KA.item(), m, len(y_tr), K_, args.N)

        if args.verbose and epoch % 10 == 0:
            print(f"Ep {epoch:3d} | loss={L_KA.item():.4f} | m={m}/{len(y_tr)} | gain={gain}%")

    with torch.no_grad():
        acc_A = (model_A(X_te).argmax(1) == y_te).float().mean().item()
    logger_A.save()

    # ── Option B : KabenaTrainer (clé en main) ───────────────────────────────
    model_B = MLP(20)
    cfg     = KabenaConfig(K=K_, N=args.N, task="classification", verbose=args.verbose)
    trainer = KabenaTrainer(model_B, cfg, lr=0.05, epochs=args.epochs)
    trainer.fit(X_tr, y_tr)
    acc_B = trainer.evaluate(X_te, y_te)

    print(f"\nRésultats :")
    print(f"  MLP K-ABENA boucle manuelle  : acc={acc_A:.4f}")
    print(f"  MLP K-ABENA KabenaTrainer    : acc={acc_B:.4f}")
    print(f"  Gain moyen (option A)        : {np.mean([r['gain_pct'] for r in history_A]):.1f}%")

    if args.plot:
        plot_stats(history_A, title="MLP K-ABENA — Boucle manuelle")
        trainer.plot_stats()


# ─── CNN ─────────────────────────────────────────────────────────────────────
def run_cnn(args):
    """CNN vision avec DataLoader — K-ABENA sur les pertes par image."""
    print("\n" + "="*60)
    print("K-ABENA — CNN Vision (PyTorch)")
    print("="*60)

    try:
        import torchvision, torchvision.transforms as T
    except ImportError:
        print("torchvision requis : pip install torchvision")
        return

    tfm  = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    ds   = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    model = SmallCNN(n_classes=10)
    opt   = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    K_    = args.K or 0.30

    history = []
    for epoch in range(min(args.epochs, 10)):  # max 10 pour démo
        total_m, total_n, losses_ep = 0, 0, []
        for X_b, y_b in loader:
            losses = F.cross_entropy(model(X_b), y_b, reduction="none")  # ← "none"
            mask   = kabena_filter_torch(losses, K=K_, N=args.N)         # ← +1 ligne
            if not mask.any():
                continue
            L_KA = losses[mask].mean()
            opt.zero_grad(); L_KA.backward(); opt.step()
            total_m += mask.sum().item(); total_n += len(y_b)
            losses_ep.append(L_KA.item())

        gain = round((1 - total_m/total_n)*100)
        history.append({"epoch": epoch, "loss": np.mean(losses_ep),
                         "m": total_m, "n": total_n, "gain_pct": gain})
        print(f"Epoch {epoch+1}/{min(args.epochs,10)} | "
              f"loss={np.mean(losses_ep):.4f} | gain={gain}%")

    if args.plot:
        plot_stats(history, title="CNN K-ABENA — CIFAR-10")


# ─── XGBoost ────────────────────────────────────────────────────────────────
def run_xgboost(args):
    """XGBoost avec objectif K-ABENA sur les gradients (hessiens)."""
    print("\n" + "="*60)
    print("K-ABENA — XGBoost (objectif personnalisé)")
    print("="*60)

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost requis : pip install xgboost")
        return

    from kabena_ml import kabena_filter

    X_np, y_np = make_classification(n_samples=3000, n_features=20, random_state=42)
    X_np = StandardScaler().fit_transform(X_np)
    split = int(0.8 * len(X_np))
    X_tr, X_te = X_np[:split], X_np[split:]
    y_tr, y_te = y_np[:split], y_np[split:]

    K_ = args.K or 0.10

    def kabena_objective(y_true, y_pred):
        """Objectif XGBoost avec K-ABENA sur les gradients."""
        # Logistic gradient
        p    = 1 / (1 + np.exp(-y_pred))
        grad = p - y_true
        hess = p * (1 - p)

        # Filtre K-ABENA sur |gradient|
        active = kabena_filter(np.abs(grad), K=K_, N=args.N)
        grad[~active] = 0.0
        hess[~active] = 1e-6  # éviter division par zéro
        return grad, hess

    # Standard
    clf_std = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                  eval_metric="logloss", verbosity=0)
    clf_std.fit(X_tr, y_tr)
    acc_std = accuracy_score(y_te, clf_std.predict(X_te))

    # K-ABENA
    clf_ka = xgb.XGBClassifier(n_estimators=100, objective=kabena_objective,
                                 eval_metric="logloss", verbosity=0)
    clf_ka.fit(X_tr, y_tr)
    acc_ka = accuracy_score(y_te, clf_ka.predict(X_te))

    print(f"\nAccuracy XGBoost standard : {acc_std:.4f}")
    print(f"Accuracy XGBoost K-ABENA  : {acc_ka:.4f}")
    print(f"(K={K_}, N={args.N} — gradients |g_i| <= K exclus)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="K-ABENA PyTorch project")
    parser.add_argument("--model",   default="mlp",
                        choices=["mlp", "cnn", "xgboost"])
    parser.add_argument("--K",       type=float, default=None,
                        help="Seuil K (None = auto-calibré)")
    parser.add_argument("--N",       type=float, default=0.0)
    parser.add_argument("--epochs",  type=int,   default=80)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    dispatch = {"mlp": run_mlp, "cnn": run_cnn, "xgboost": run_xgboost}
    dispatch[args.model](args)


if __name__ == "__main__":
    main()
