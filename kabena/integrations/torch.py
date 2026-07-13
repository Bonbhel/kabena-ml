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
import warnings

__all__ = ["KabenaTorch"]


def _warn_if_numpy_torch_abi_mismatch():
    """Detect the torch<2.3 + numpy>=2 pairing and explain the fix.

    torch wheels < 2.3 are compiled against the NumPy 1.x ABI; importing
    them under NumPy 2.x triggers undefined behavior ("may crash").
    We warn loudly with the exact fix commands instead of letting the
    user hit a cryptic crash later.
    """
    try:
        import numpy
        import torch
    except ImportError:
        return  # the existing import guard handles missing deps

    np_major = int(numpy.__version__.split(".")[0])
    torch_mm = tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    )
    if np_major >= 2 and torch_mm < (2, 3):
        warnings.warn(
            "kabena[torch]: incompatible pairing detected -- "
            f"torch {torch.__version__} was built against NumPy 1.x, "
            f"but NumPy {numpy.__version__} is installed. "
            "This can crash at runtime. Fix (choose ONE):\n"
            "  pip install 'numpy<2'          # keep this torch "
            "(required on Intel macOS, where torch stops at 2.2.2)\n"
            "  pip install --upgrade torch    # torch>=2.3 supports NumPy 2 "
            "(not available on Intel macOS)\n"
            "Details: https://github.com/Bonbhel/kabena-ml#installation",
            UserWarning,
            stacklevel=2,
        )


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
