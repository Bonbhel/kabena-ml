"""Intégration TensorFlow/Keras — 2 lignes via sample_weight recalculé par époque.

    kb = KabenaKeras(model, X, y, loss_fn)                      # ligne 1
    model.fit(X, y, epochs=E, callbacks=[kb], sample_weight=kb.weights)   # ligne 2

Le callback met à jour IN PLACE le tableau `kb.weights` (0 = exclue,
1/pi = mineure v3 retenue, 1 = majeure) au début de chaque époque —
Keras relit le buffer, aucune reconstruction du dataset n'est requise.
"""
from __future__ import annotations
import numpy as np
from ..core import Kabena

__all__ = ["KabenaKeras"]

try:                                          # import paresseux et optionnel
    from tensorflow import keras as _keras
    _Base = _keras.callbacks.Callback
except Exception:                             # pragma: no cover - env sans TF
    _Base = object


class KabenaKeras(_Base):
    def __init__(self, model, X, y, per_sample_loss_fn,
                 N: float = 0.3, strategy: str = "auto", seed: int | None = None, **adv):
        super().__init__()
        self._m, self._X, self._y = model, np.asarray(X), np.asarray(y)
        self._loss_fn = per_sample_loss_fn         # (model, X, y) -> losses[n]
        self._kb = Kabena(N=N, strategy=strategy, seed=seed, **adv)
        self.weights = np.ones(len(self._y), dtype="float32")
        self.last_gain_ = None

    def on_epoch_begin(self, epoch, logs=None):   # signature Keras
        losses = np.asarray(self._loss_fn(self._m, self._X, self._y), float)
        a, w = self._kb.select(losses, y=self._y)
        self.weights[:] = np.where(a, w, 0.0).astype("float32")
        self.last_gain_ = self._kb.last_gain_
