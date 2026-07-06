"""Intégration PyTorch — 2 lignes autour d'une loss `reduction='none'`.

    kb = KabenaTorch()                                          # ligne 1
    for xb, yb in loader:
        losses = criterion(model(xb), yb)          # reduction='none'
        loss = kb.reduce(losses, y=yb)                          # ligne 2
        loss.backward(); opt.step(); opt.zero_grad()

`reduce` détache les pertes pour la sélection, applique masque + poids HT,
et retourne un scalaire différentiable : moyenne pondérée sur S*.
NB: la sélection annule la contribution au gradient des exclues ; le forward
complet du batch reste payé — pour économiser aussi le forward, sélectionner
AVANT le forward avec les pertes de l'époque précédente (motif 'pertes
retardées', voir tutorials/mlp/niveau2_script_pytorch.py).
"""
from __future__ import annotations
import numpy as np
from ..core import Kabena

__all__ = ["KabenaTorch"]


class KabenaTorch:
    def __init__(self, N: float = 0.3, strategy: str = "auto", seed: int | None = None, **adv):
        self._kb = Kabena(N=N, strategy=strategy, seed=seed, **adv)
        self.last_gain_ = None

    def select_numpy(self, losses_np: np.ndarray, y=None):
        a, w = self._kb.select(losses_np, y=y)
        self.last_gain_ = self._kb.last_gain_
        return a, w

    def reduce(self, losses, y=None):
        """losses : tensor 1-D (reduction='none'). Retourne un scalaire différentiable."""
        try:
            import torch
        except ImportError as e:            # pragma: no cover
            raise ImportError("PyTorch requis pour KabenaTorch.reduce()") from e
        with torch.no_grad():
            ln = losses.detach().cpu().numpy()
            yn = y.detach().cpu().numpy() if (y is not None and hasattr(y, "detach")) else y
        a, w = self.select_numpy(ln, y=yn)
        mask = torch.as_tensor(a, device=losses.device)
        wt = torch.as_tensor(w, dtype=losses.dtype, device=losses.device)
        sel_w = wt[mask]
        return (losses[mask] * sel_w).sum() / sel_w.sum()
