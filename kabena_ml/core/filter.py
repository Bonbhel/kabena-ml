"""
kabena_ml.core.filter
=====================
Primitives K-ABENA pures NumPy — compatibles numpy, scipy, JAX (jnp).

Coût syntaxique minimal :
    active = kabena_filter(losses, K=0.25)   # 1 ligne
"""

from __future__ import annotations
import numpy as np
from typing import Union

Array = Union[np.ndarray, list]


# ─────────────────────────────────────────────────────────────────────────────
def kabena_filter(
    abs_errors: Array,
    K: float,
    N: float = 0.0,
) -> np.ndarray:
    """
    Filtre K-ABENA — retourne un masque booléen des observations actives.

    Paramètres
    ----------
    abs_errors : array-like (n,)
        Erreurs absolues |εᵢ| ou pertes individuelles ℓᵢ (BCE, MSE…).
    K : float
        Seuil absolu. Toute erreur <= K est considérée "mineure".
    N : float in [0, 1]
        Proportion de mineures à CONSERVER dans le gradient.
        N=0 → toutes les mineures exclues (mode agressif).
        N=1 → aucune exclue (= gradient standard).

    Retourne
    --------
    active : np.ndarray bool (n,)
        True  = observation dans S* (contribue au gradient).
        False = observation exclue (mineure éliminée).

    Exemples
    --------
    >>> import numpy as np
    >>> errors = np.array([0.05, 0.82, 0.12, 0.41, 0.03, 0.67])
    >>> kabena_filter(errors, K=0.15)
    array([False,  True, False,  True, False,  True])

    >>> kabena_filter(errors, K=0.15, N=0.5)  # garde 50% des mineures
    array([ True,  True, False,  True, False,  True])
    """
    abs_errors = np.asarray(abs_errors, dtype=float)
    minor_idx  = np.where(abs_errors <= K)[0]

    if len(minor_idx) == 0:
        return np.ones(len(abs_errors), dtype=bool)  # rien à exclure

    # Tri décroissant : proches de K exclus en premier
    order        = np.argsort(abs_errors[minor_idx])[::-1]
    minor_sorted = minor_idx[order]
    n_excl       = int((1.0 - N) * len(minor_sorted))

    active = np.ones(len(abs_errors), dtype=bool)
    active[minor_sorted[:n_excl]] = False
    return active


# ─────────────────────────────────────────────────────────────────────────────
def kabena_safe(
    abs_errors: Array,
    K: float,
    N: float = 0.0,
    min_active: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Version sécurisée de kabena_filter — garantit m >= min_active.

    Retourne
    --------
    (active, m) : (bool array, int)

    Exemples
    --------
    >>> active, m = kabena_safe(errors, K=0.99, N=0.0, min_active=2)
    >>> assert m >= 2
    """
    abs_errors = np.asarray(abs_errors, dtype=float)
    n          = len(abs_errors)
    minor_idx  = np.where(abs_errors <= K)[0]

    if len(minor_idx) == 0:
        return np.ones(n, dtype=bool), n

    order        = np.argsort(abs_errors[minor_idx])[::-1]
    minor_sorted = minor_idx[order]
    n_excl       = int((1.0 - N) * len(minor_sorted))
    n_excl       = min(n_excl, n - min_active)   # clip de sécurité

    active = np.ones(n, dtype=bool)
    active[minor_sorted[:n_excl]] = False
    return active, int(active.sum())


# ─────────────────────────────────────────────────────────────────────────────
def calibrate_K(
    losses_epoch1: Array,
    target_pct: float = 0.10,
    strategy: str = "percentile",
) -> float:
    """
    Calibrage automatique de K depuis les pertes de la première époque.

    Stratégies
    ----------
    "percentile" : K = percentile(target_pct * 100) — recommandé.
    "std"        : K = mean(losses) - std(losses).
    "iqr"        : K = Q1 - 1.5 * IQR (outlier-aware).

    Exemples
    --------
    >>> K = calibrate_K(losses_ep1, target_pct=0.10)
    >>> print(f"K calibré : {K:.4f}")
    """
    losses = np.asarray(losses_epoch1, dtype=float)

    if strategy == "percentile":
        return float(np.percentile(losses, target_pct * 100))
    elif strategy == "std":
        return float(max(0.0, np.mean(losses) - np.std(losses)))
    elif strategy == "iqr":
        q1, q3 = np.percentile(losses, [25, 75])
        return float(max(0.0, q1 - 1.5 * (q3 - q1)))
    else:
        raise ValueError(f"strategy inconnu: {strategy!r}. Choisir parmi: percentile, std, iqr")


# ─────────────────────────────────────────────────────────────────────────────
def calibrate_K_noisy(
    losses_epoch1: Array,
    estimated_noise_pct: float = 0.0,
    noise_factor: float = 2.0,
    q_cap: float = 25.0,
) -> float:
    """
    Calibrage de K adapté au bruit (label noise / feature noise).
    / Noise-adaptive K calibration.

    Sous bruit, les exemples corrompus bien mémorisés produisent des pertes
    artificiellement faibles et seraient exclus à tort par un calibrage
    standard (target_pct=10%). Cette fonction augmente le percentile cible
    proportionnellement au niveau de bruit estimé.
    / Under noise, well-memorized corrupted examples produce artificially
    low losses and would be incorrectly excluded under standard calibration.
    This function raises the target percentile proportionally to the
    estimated noise level.

    Formule / Formula
    ------------------
        q = min(10 + noise_factor * estimated_noise_pct * 100, q_cap)
        K = percentile(losses, q)

    ⚠️ AVERTISSEMENT SUR LA PORTÉE EMPIRIQUE / SCOPE WARNING
    -----------------------------------------------------------------------
    Les valeurs par défaut (noise_factor=2.0, q_cap=25.0) ont été observées
    comme un compromis raisonnable sur DEUX configurations seulement
    (CIFAR-10 vision, SST-2 NLP) — un échantillon insuffisant pour conclure
    à une valeur universelle. Contrairement à N=0.3 (validé par ablation sur
    8 datasets, voir le livre K-ABENA ch.16), ces constantes N'ONT PAS été
    soumises à une étude d'ablation systématique. Documenté comme Limite L10
    du livre K-ABENA (ch.18.10.2).

    Default values (noise_factor=2.0, q_cap=25.0) were observed as a
    reasonable tradeoff on TWO configurations only (CIFAR-10 vision,
    SST-2 NLP) — an insufficient sample to conclude universality. Unlike
    N=0.3 (validated by ablation across 8 datasets, see K-ABENA book ch.16),
    these constants have NOT undergone systematic ablation. Documented as
    Limitation L10 of the K-ABENA book (ch.18.10.2).

    RECOMMANDATION / RECOMMENDATION : sur tout nouveau dataset, valider
    noise_factor via une grille [1.0, 1.5, 2.0, 2.5, 3.0] sur un ensemble
    de validation avec bruit connu plutôt que d'utiliser 2.0 par défaut.
    / On any new dataset, validate noise_factor via a grid search
    [1.0, 1.5, 2.0, 2.5, 3.0] on a held-out set with known noise rather
    than using 2.0 as a default.

    CONSTAT CONCRET (vérifié empiriquement) / CONCRETE FINDING (verified):
    avec les valeurs par défaut (factor=2, q_cap=25), le plafond q_cap est
    atteint dès estimated_noise_pct=0.075 (7.5%). Au-delà de ce seuil,
    TOUTE variation de bruit produit exactement le même K — la formule
    devient insensible au bruit sur la majorité de sa plage d'usage prévue.
    / with default values (factor=2, q_cap=25), the cap is reached at
    estimated_noise_pct=0.075 (7.5%) already. Beyond this threshold, ANY
    noise variation produces exactly the same K — the formula becomes
    insensitive to noise across most of its intended operating range.

    Paramètres / Parameters
    ------------------------
    losses_epoch1 : array-like (n,)
        Pertes individuelles à l'époque 1. / Individual losses at epoch 1.
    estimated_noise_pct : float in [0, 1]
        Fraction de bruit estimée (labels ou features corrompus).
        / Estimated noise fraction (corrupted labels or features).
        0.0 = données propres, calibrage standard. / 0.0 = clean data.
    noise_factor : float, default 2.0
        Facteur multiplicatif — HEURISTIQUE NON VALIDÉE (voir avertissement).
        / Multiplicative factor — UNVALIDATED HEURISTIC (see warning above).
    q_cap : float, default 25.0
        Plafond du percentile cible — HEURISTIQUE NON VALIDÉE.
        / Target percentile cap — UNVALIDATED HEURISTIC.

    Retourne / Returns
    -------------------
    float : seuil K calibré / calibrated K threshold

    Exemples / Examples
    ---------------------
    >>> K = calibrate_K_noisy(losses_ep1, estimated_noise_pct=0.0)   # q=10% (standard)
    >>> K = calibrate_K_noisy(losses_ep1, estimated_noise_pct=0.30)  # q=16%
    >>> # Valider le facteur sur votre dataset avant usage en production :
    >>> # Validate the factor on your dataset before production use:
    >>> for f in [1.0, 1.5, 2.0, 2.5, 3.0]:
    ...     K_f = calibrate_K_noisy(losses_ep1, estimated_noise_pct=0.30, noise_factor=f)
    ...     # ... évaluer l'accuracy de validation pour chaque K_f
    """
    losses = np.asarray(losses_epoch1, dtype=float)
    q = min(10.0 + noise_factor * estimated_noise_pct * 100, q_cap)
    return float(np.percentile(losses, q))

