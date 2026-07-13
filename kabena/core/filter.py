"""API centrale — la promesse « 2 lignes » :

    from kabena import Kabena
    kb = Kabena()                                   # ligne 1
    ...
    active, w = kb.select(losses, y=y_train)        # ligne 2 (dans la boucle)
    model.fit(X[active], y[active], sample_weight=w[active])

Compatibilité 1.x : kabena_filter(...) et kabena_safe(...) sont conservées.
"""
from __future__ import annotations
import numpy as np
from .config import KabenaConfig
from .gate import resolve_strategy
from . import sampling
import difflib

__all__ = ["Kabena", "kabena_filter", "kabena_safe"]


_VALID_PARAMS = {"N", "strategy", "seed", "k_percentile", "alpha", "min_active"}


# Erreurs prévisibles → explication ciblée (la terminologie du papier
# diffère volontairement de l'API : K est un seuil CALCULÉ, pas un réglage)
_PARAM_HINTS = {
    "K": (
        "there is no 'K' parameter: the threshold K from the paper is "
        "computed internally at every select() call. What you can set is "
        "'k_percentile' (a percentile in [0, 100], default 40.0) that "
        "determines where K lands on the current loss distribution"
    ),
    "k": "did you mean 'k_percentile'? (the threshold K itself is computed internally)",
    "percentile": "did you mean 'k_percentile'?",
    "threshold": "the threshold is computed internally -- tune 'k_percentile' instead",
    "n": "did you mean 'N' (uppercase), the sampling ratio in (0, 1)?",
    "ratio": "did you mean 'N', the sampling ratio in (0, 1)?",
    "floor": "did you mean 'alpha', the defensive floor in (0, 1]?",
}


def _validate_params(advanced: dict) -> None:
    """Raise a helpful TypeError for any unknown keyword argument."""
    for name in advanced:
        if name in _VALID_PARAMS:
            continue
        hint = _PARAM_HINTS.get(name)
        if hint is None:
            close = difflib.get_close_matches(name, _VALID_PARAMS, n=1)
            hint = (
                f"did you mean '{close[0]}'?"
                if close
                else f"valid parameters: {', '.join(sorted(_VALID_PARAMS))}"
            )
        raise TypeError(
            f"Kabena() got an unexpected parameter '{name}' -- {hint}. "
            "Full parameter reference: "
            "https://github.com/Bonbhel/kabena-ml#parameters"
        )



class Kabena:
    """Sélecteur K-ABENA. Trois paramètres, tout le reste en défauts du preprint."""

    def __init__(self, N: float = 0.3, strategy: str = "auto",
                 seed: int | None = None, **advanced):
        _validate_params(advanced)
        self.cfg = KabenaConfig(N=N, strategy=strategy, seed=seed, **advanced)
        self.cfg.validate()
        self._rng = np.random.default_rng(seed)
        self._forced = False
        self.last_gain_ = None      # fraction de backward passes économisée au dernier appel

    def force(self) -> "Kabena":
        """Assume explicitement la stratégie demandée (désactive le garde-fou)."""
        self._forced = True
        return self

    def select(self, losses, y=None):
        """Retourne (active: bool[n], weights: float[n]).

        losses : pertes individuelles courantes (array-like, n).
        y      : cibles (optionnel) — permet au garde-fou de détecter le
                 déséquilibre extrême quand strategy='v2'.
        """
        losses = np.asarray(losses, dtype=float)
        strat = self.cfg.strategy if self._forced else resolve_strategy(self.cfg.strategy, y)
        if strat == "auto":
            strat = "v3"
        K = float(np.percentile(losses, self.cfg.k_percentile))
        fn = {"v1": sampling.select_v1, "v2": sampling.select_v2,
              "v3": sampling.select_v3}[strat]
        kwargs = {"alpha": self.cfg.alpha} if strat == "v3" else {}
        active, w = fn(losses, K=K, N=self.cfg.N, rng=self._rng, **kwargs)
        # plancher de sécurité : ré-inclure les plus petites pertes exclues
        deficit = self.cfg.min_active - int(active.sum())
        if deficit > 0:
            excl = np.where(~active)[0]
            active[excl[np.argsort(losses[excl])[:deficit]]] = True
        self.last_gain_ = 1.0 - active.mean()
        return active, w


# ---------- API fonctionnelle rétro-compatible (1.x) ----------
def kabena_filter(abs_errors, K: float, N: float = 0.0, strategy: str = "v1",
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Signature historique 1.x (défaut strategy='v1' pour ne rien casser).
    Retourne le masque booléen seul ; utiliser Kabena().select() pour les poids v3."""
    abs_errors = np.asarray(abs_errors, dtype=float)
    fn = {"v1": sampling.select_v1, "v2": sampling.select_v2,
          "v3": sampling.select_v3}[strategy]
    active, _ = fn(abs_errors, K=K, N=N, rng=rng or np.random.default_rng())
    return active


def kabena_safe(abs_errors, K: float, N: float = 0.0, min_active: int = 1,
                strategy: str = "v1", rng=None):
    """Variante 1.x garantissant |S*| >= min_active. Retourne (active, m)."""
    abs_errors = np.asarray(abs_errors, dtype=float)
    active = kabena_filter(abs_errors, K=K, N=N, strategy=strategy, rng=rng)
    deficit = min_active - int(active.sum())
    if deficit > 0:
        excl = np.where(~active)[0]
        active[excl[np.argsort(abs_errors[excl])[:deficit]]] = True
    return active, int(active.sum())
