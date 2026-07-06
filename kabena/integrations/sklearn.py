"""Intégration scikit-learn — la promesse 2 lignes, vérifiée par validation/.

    kb = Kabena()                                               # ligne 1
    for epoch in range(E):
        losses = per_sample_loss(model, X, y)
        a, w = kb.select(losses, y=y)                           # ligne 2
        model.partial_fit(X[a], y[a], sample_weight=w[a], classes=classes)
"""
from __future__ import annotations
import numpy as np
from ..core import Kabena

__all__ = ["logloss_per_sample", "squared_loss_per_sample", "fit_sgdclassifier_kabena"]


def logloss_per_sample(proba_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba_pos, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def squared_loss_per_sample(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(y_pred, float) - np.asarray(y, float)) ** 2


def fit_sgdclassifier_kabena(model, X, y, epochs: int = 20, kb: Kabena | None = None):
    """Boucle prête à l'emploi pour sklearn.linear_model.SGDClassifier (log_loss)."""
    kb = kb or Kabena()
    X = np.asarray(X); y = np.asarray(y)
    classes = np.unique(y)
    model.partial_fit(X, y, classes=classes)          # init sur 1 passe complète
    for _ in range(epochs):
        p = model.predict_proba(X)[:, 1]
        losses = logloss_per_sample(p, y)
        a, w = kb.select(losses, y=y)
        model.partial_fit(X[a], y[a], sample_weight=w[a])
    return model
