"""
kabena_ml.integrations.torch_utils
====================================
Intégration PyTorch — coût syntaxique : 2 lignes.

AVANT (code PyTorch standard) :
    loss = F.cross_entropy(logits, y)
    loss.backward()

APRÈS K-ABENA (+2 lignes, rien d'autre ne change) :
    losses = F.cross_entropy(logits, y, reduction="none")  # ← "none"
    mask   = kabena_filter_torch(losses, K=0.25)           # ← +1 ligne
    losses[mask].mean().backward()                         # ← inchangé
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np

from kabena_ml.core.config import KabenaConfig
from kabena_ml.core.filter import kabena_filter, kabena_safe


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch requis : pip install kabena-ml[torch]")


# ─────────────────────────────────────────────────────────────────────────────
def kabena_filter_torch(
    losses: "torch.Tensor",
    K: float,
    N: float = 0.0,
) -> "torch.Tensor":
    """
    Filtre K-ABENA pour tenseurs PyTorch.
    Retourne un masque BoolTensor directement utilisable.

    Paramètres
    ----------
    losses : Tensor (n,) — pertes individuelles (reduction="none")
    K      : float — seuil absolu
    N      : float — proportion de mineures conservées

    Exemples
    --------
    >>> losses = F.cross_entropy(logits, y, reduction="none")
    >>> mask   = kabena_filter_torch(losses, K=0.25)
    >>> losses[mask].mean().backward()
    """
    _require_torch()
    active_np = kabena_filter(losses.detach().cpu().numpy(), K=K, N=N)
    return torch.tensor(active_np, dtype=torch.bool, device=losses.device)


def kabena_safe_torch(
    losses: "torch.Tensor",
    K: float,
    N: float = 0.0,
    min_active: int = 1,
) -> tuple["torch.Tensor", int]:
    """Version sécurisée — garantit m >= min_active."""
    _require_torch()
    active_np, m = kabena_safe(losses.detach().cpu().numpy(), K=K, N=N, min_active=min_active)
    return torch.tensor(active_np, dtype=torch.bool, device=losses.device), m


# ─────────────────────────────────────────────────────────────────────────────
class KabenaTrainer:
    """
    Trainer PyTorch clé en main — intègre K-ABENA dans la boucle d'entraînement.

    Paramètres
    ----------
    model     : nn.Module
    config    : KabenaConfig
    optimizer : classe optimizer (ex: torch.optim.SGD)
    lr        : float
    epochs    : int

    Exemples
    --------
    >>> trainer = KabenaTrainer(model, KabenaConfig(K=0.25), lr=1e-3, epochs=50)
    >>> history = trainer.fit(X_train, y_train)
    >>> trainer.plot_stats()
    """

    def __init__(
        self,
        model,
        config: KabenaConfig,
        optimizer=None,
        lr: float = 1e-3,
        epochs: int = 100,
    ):
        _require_torch()
        self.model = model
        self.cfg = config
        self.epochs = epochs
        self.history = []

        _opt = optimizer or torch.optim.SGD
        self.optimizer = _opt(model.parameters(), lr=lr)

    def fit(
        self,
        X: "torch.Tensor",
        y: "torch.Tensor",
        val_data: Optional[tuple] = None,
    ) -> list[dict]:
        """Boucle d'entraînement K-ABENA complète."""
        self.model.train()

        for epoch in range(self.epochs):
            logits = self.model(X)
            losses = self._compute_losses(logits, y)

            # ── K-ABENA : 2 lignes ──────────────────────────────────────
            mask, m = kabena_safe_torch(losses, self.cfg.K, self.cfg.N, self.cfg.min_active)
            if m == 0:
                continue
            L_KA = losses[mask].mean()
            # ────────────────────────────────────────────────────────────

            self.optimizer.zero_grad()
            L_KA.backward()
            self.optimizer.step()

            record = {
                "epoch": epoch,
                "loss": L_KA.item(),
                "m": m,
                "n": len(y),
                "gain_pct": round((1 - m / len(y)) * 100),
            }
            if val_data:
                record["val_acc"] = self._accuracy(*val_data)
            self.history.append(record)

            if self.cfg.verbose and epoch % 10 == 0:
                print(
                    f"Ep {epoch:4d} | loss={record['loss']:.4f} | "
                    f"actifs={m}/{len(y)} | gain={record['gain_pct']}%"
                    + (f" | val_acc={record['val_acc']:.4f}" if val_data else "")
                )

        return self.history

    def fit_loader(self, loader, val_loader=None) -> list[dict]:
        """Boucle K-ABENA avec DataLoader."""
        self.model.train()
        for epoch in range(self.epochs):
            epoch_losses, epoch_m, epoch_n = [], 0, 0
            for X_b, y_b in loader:
                logits = self.model(X_b)
                losses = self._compute_losses(logits, y_b)

                mask, m = kabena_safe_torch(losses, self.cfg.K, self.cfg.N, self.cfg.min_active)
                if m == 0:
                    continue
                L_KA = losses[mask].mean()
                self.optimizer.zero_grad()
                L_KA.backward()
                self.optimizer.step()

                epoch_losses.append(L_KA.item())
                epoch_m += m
                epoch_n += len(y_b)

            if not epoch_losses:
                continue
            record = {
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "m": epoch_m,
                "n": epoch_n,
                "gain_pct": round((1 - epoch_m / epoch_n) * 100),
            }
            self.history.append(record)
            if self.cfg.verbose and epoch % 10 == 0:
                print(f"Ep {epoch:4d} | loss={record['loss']:.4f} | " f"gain={record['gain_pct']}%")
        return self.history

    def evaluate(self, X, y) -> float:
        self.model.eval()
        with torch.no_grad():
            return self._accuracy(X, y)

    def plot_stats(self, save_to: Optional[str] = None):
        """Affiche les courbes de monitoring K-ABENA."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib requis pour plot_stats()")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = [r["epoch"] for r in self.history]

        axes[0].plot(epochs, [r["loss"] for r in self.history], color="#1DD0FF", lw=2)
        axes[0].set(xlabel="Époque", ylabel="Loss K-ABENA", title="Loss")
        axes[0].grid(alpha=0.2)

        pct_active = [r["m"] / r["n"] * 100 for r in self.history]
        axes[1].plot(epochs, pct_active, color="#1D9E75", lw=2)
        axes[1].axhline(100, color="#B4B2A9", lw=1, ls="--", alpha=0.5)
        axes[1].fill_between(
            epochs, pct_active, 100, alpha=0.15, color="#EF9F27", label="Gain computationnel"
        )
        axes[1].set(
            xlabel="Époque",
            ylabel="Observations actives (%)",
            title="Observations actives m/n",
            ylim=(0, 105),
        )
        axes[1].legend()
        axes[1].grid(alpha=0.2)

        fig.suptitle("Monitoring K-ABENA", fontweight="bold")
        fig.tight_layout()
        if save_to:
            fig.savefig(save_to, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    # ── Privé ──────────────────────────────────────────────────────────────

    def _compute_losses(self, logits, y):
        if self.cfg.task == "regression":
            return F.mse_loss(logits.squeeze(), y.float(), reduction="none")
        else:
            return F.cross_entropy(logits, y.long(), reduction="none")

    def _accuracy(self, X, y) -> float:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).argmax(dim=1)
        return (preds == y).float().mean().item()
