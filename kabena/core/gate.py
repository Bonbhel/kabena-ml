"""Garde-fou v2 -> v3 (preprint §6, 'quantified contraindications').

Le mode régularisé v2 n'est sûr QUE dans sa zone de validité mesurée :
  - signal minoritaire >= ~5 %  (échec mesuré : AUC 0,53 à 0,17 %)
  - N <= 0.5 (contrainte de faisabilité)
Le bruit d'étiquettes (>~25 % : effondrement 0,386 mesuré à 40 %) n'est
pas observable depuis les pertes seules — il est documenté, pas détecté.
"""
from __future__ import annotations
import warnings
import numpy as np

MINORITY_SHARE_FLOOR = 0.05   # preprint : à 5 % toutes les variantes sont saines

def v2_is_safe(y=None) -> tuple[bool, str]:
    """Vérifie les contre-indications observables du mode v2."""
    if y is not None:
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) >= 2:
            share = counts.min() / counts.sum()
            if share < MINORITY_SHARE_FLOOR:
                return False, (f"classe minoritaire à {share:.2%} < {MINORITY_SHARE_FLOOR:.0%} : "
                               "régime de la Proposition 2 du preprint (échec v2 mesuré, AUC 0,53).")
    return True, ""

def resolve_strategy(strategy: str, y=None) -> str:
    """'auto' -> 'v3'. 'v2' explicite -> vérifié ; bascule v3 avec warning si hors zone."""
    if strategy == "auto":
        return "v3"
    if strategy == "v2":
        ok, why = v2_is_safe(y)
        if not ok:
            warnings.warn("kabena: mode v2 contre-indiqué ici (" + why +
                          ") — bascule automatique sur v3. Passer strategy='v2' avec "
                          "un y équilibré, ou assumer explicitement via Kabena(strategy='v2').force().",
                          stacklevel=3)
            return "v3"
    return strategy
