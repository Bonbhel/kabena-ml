"""Plans de tirage K-ABENA — v1 / v2 / v3 (canonique, preprint Def. 1).

Chaque fonction retourne (active: bool[n], weights: float[n]) :
- active : masque de l'ensemble actif S* (majeures + mineures retenues)
- weights : poids d'échantillon (1 partout sauf mineures v3, repondérées 1/pi)

Le contrat (active, weights) est identique pour les trois stratégies :
c'est ce qui rend le passage v1/v2/v3 transparent pour l'appelant.
"""
from __future__ import annotations
import numpy as np

__all__ = ["select_v1", "select_v2", "select_v3"]


def _empty_minors(n: int):
    return np.ones(n, dtype=bool), np.ones(n, dtype=float)


def select_v1(losses: np.ndarray, K: float, N: float,
              rng: np.random.Generator | None = None):
    """Historique (<2.0) : exclusion déterministe des mineures proches de K."""
    n = len(losses)
    minors = np.where(losses <= K)[0]
    if len(minors) == 0:
        return _empty_minors(n)
    order = np.argsort(losses[minors])[::-1]          # proches de K d'abord
    n_excl = int((1.0 - N) * len(minors))
    active = np.ones(n, dtype=bool)
    active[minors[order[:n_excl]]] = False
    return active, np.ones(n, dtype=float)


def select_v2(losses: np.ndarray, K: float, N: float,
              rng: np.random.Generator | None = None):
    """Mode régularisé (preprint §6) : demi-domaine bas, p_i ∝ eps_i, sans repondération.

    Biaisé PAR CONSTRUCTION (Lemma 5) — bonus multiclasse possible, mais
    contre-indiqué hors zone de validité (voir core.gate). N <= 0.5 requis.
    """
    n = len(losses)
    if N > 0.5:
        raise ValueError("v2 impose N <= 0.5 ; utiliser v3 au-delà.")
    rng = rng or np.random.default_rng()
    minors = np.where(losses <= K)[0]
    if len(minors) == 0:
        return _empty_minors(n)
    eps = losses[minors]
    pool = np.where(eps <= np.median(eps))[0]         # moitié basse uniquement
    m = min(max(1, int(round(N * len(minors)))), len(pool))
    w = np.maximum(eps[pool], 1e-12); w = w / w.sum()
    keep = minors[rng.choice(pool, size=m, replace=False, p=w)]
    active = np.ones(n, dtype=bool)
    active[minors] = False
    active[keep] = True
    return active, np.ones(n, dtype=float)


def select_v3(losses: np.ndarray, K: float, N: float,
              rng: np.random.Generator | None = None, alpha: float = 0.3):
    """Canonique (preprint Def. 1) : mélange défensif sur TOUT M_K + poids Horvitz-Thompson.

    p_i = alpha/k + (1-alpha) * eps_i / sum(eps) ; pi_i = min(1, m*p_i) ;
    poids w_i = 1/pi_i sur les mineures retenues. Estimateur (quasi) sans
    biais (Lemmas 2-3) ; à alpha=1, pi = m/k est EXACT (SRSWOR).
    """
    n = len(losses)
    rng = rng or np.random.default_rng()
    minors = np.where(losses <= K)[0]
    k = len(minors)
    weights = np.ones(n, dtype=float)
    if k == 0:
        return np.ones(n, dtype=bool), weights
    m = min(max(1, int(round(N * k))), k)
    eps = np.maximum(losses[minors], 1e-12)
    p = alpha / k + (1.0 - alpha) * eps / eps.sum()
    p = p / p.sum()
    keep = rng.choice(minors, size=m, replace=False, p=p)
    pi = np.minimum(1.0, m * p)                       # Assumption 1(b) du preprint
    pos = {mi: j for j, mi in enumerate(minors)}
    weights[keep] = 1.0 / np.array([pi[pos[i]] for i in keep])
    active = np.ones(n, dtype=bool)
    active[minors] = False
    active[keep] = True
    return active, weights
