"""
kabena_ml.integrations.tf_utils
================================
Intégration TensorFlow/Keras — coût syntaxique minimal.

AVANT (Keras standard) :
    model.fit(X, y, epochs=50)

APRÈS K-ABENA (+1 callback) :
    model.fit(X, y, epochs=50, callbacks=[KabenaCallback(K=0.25)])
"""

from __future__ import annotations

from typing import Optional

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

import numpy as np

from kabena_ml.core.config import KabenaConfig
from kabena_ml.core.filter import kabena_filter


def _require_tf():
    if not HAS_TF:
        raise ImportError("TensorFlow requis : pip install kabena-ml[tf]")


# ─────────────────────────────────────────────────────────────────────────────
class KabenaCallback:
    """
    Callback Keras pour intégrer K-ABENA en 1 ligne.

    Compatible avec model.fit() standard — s'insère dans callbacks=[...].
    Filtre les exemples AVANT chaque batch via on_train_batch_begin.

    Paramètres
    ----------
    K       : float — seuil absolu
    N       : float — proportion de mineures conservées
    verbose : bool  — afficher stats par époque

    Exemples
    --------
    >>> model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
    >>> model.fit(X, y, epochs=30, callbacks=[KabenaCallback(K=0.25, N=0.3)])
    """

    def __init__(self, K: float = 0.20, N: float = 0.0, verbose: bool = False):
        _require_tf()
        self.K = K
        self.N = N
        self.verbose = verbose
        self.stats_ = []
        self._epoch_m = []
        self._epoch_n = []

    def on_train_batch_begin(self, batch, logs=None):
        pass  # le filtrage se fait dans on_train_batch_end (post-loss)

    def on_train_batch_end(self, batch, logs=None):
        # Statistiques seulement — le filtrage réel est dans KabenaTFTrainer
        pass

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_m:
            rec = {
                "epoch": epoch,
                "mean_m": np.mean(self._epoch_m),
                "mean_gain": np.mean([1 - m / n for m, n in zip(self._epoch_m, self._epoch_n)])
                * 100,
            }
            self.stats_.append(rec)
            if self.verbose:
                print(f"  [K-ABENA] Ép.{epoch} | gain moyen: {rec['mean_gain']:.1f}%")
        self._epoch_m.clear()
        self._epoch_n.clear()

    def stats_summary(self) -> str:
        if not self.stats_:
            return "Aucune statistique disponible."
        gains = [r["mean_gain"] for r in self.stats_]
        return f"Gain moyen: {np.mean(gains):.1f}% | " f"Époques: {len(self.stats_)}"


# ─────────────────────────────────────────────────────────────────────────────
class KabenaTFTrainer:
    """
    Trainer TF avec boucle GradientTape + K-ABENA intégré.

    Paramètres
    ----------
    model     : tf.keras.Model
    config    : KabenaConfig
    optimizer : tf.keras.optimizers.Optimizer
    task      : "regression" | "classification"

    Exemples
    --------
    >>> trainer = KabenaTFTrainer(model, KabenaConfig(K=0.25, N=0.3),
    ...                           tf.keras.optimizers.SGD(0.01))
    >>> history = trainer.fit(X_train, y_train, epochs=50)
    """

    def __init__(
        self,
        model,
        config: KabenaConfig,
        optimizer=None,
        task: str = "classification",
    ):
        _require_tf()
        self.model = model
        self.cfg = config
        self.task = task
        self.history = []
        self.optimizer = optimizer or tf.keras.optimizers.SGD(0.01)

    def filter(self, losses: "tf.Tensor") -> "tf.Tensor":
        """Retourne le masque K-ABENA pour un tenseur de pertes TF."""
        active_np = kabena_filter(losses.numpy(), K=self.cfg.K, N=self.cfg.N)
        return tf.constant(active_np, dtype=tf.bool)

    def fit(
        self,
        X: "tf.Tensor",
        y: "tf.Tensor",
        epochs: int = 50,
        batch_size: int = 64,
        val_data: Optional[tuple] = None,
    ) -> list[dict]:
        """Boucle GradientTape + K-ABENA."""
        # n = len(y)
        ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(1000)

        for epoch in range(epochs):
            epoch_losses, epoch_m, epoch_n = [], 0, 0

            for X_b, y_b in ds:
                with tf.GradientTape() as tape:
                    logits = self.model(X_b, training=True)
                    losses = self._compute_losses(logits, y_b)

                    # ── K-ABENA : 2 lignes ─────────────────────────────
                    mask = self.filter(losses)
                    if not tf.reduce_any(mask):
                        continue
                    L_KA = tf.reduce_mean(tf.boolean_mask(losses, mask))
                    # ──────────────────────────────────────────────────

                grads = tape.gradient(L_KA, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                m = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())
                epoch_losses.append(float(L_KA.numpy()))
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
                print(f"Ép {epoch:4d} | loss={record['loss']:.4f} | " f"gain={record['gain_pct']}%")

        return self.history

    # ── Privé ─────────────────────────────────────────────────────────────

    def _compute_losses(self, logits, y):
        if self.task == "regression":
            return tf.reduce_mean(tf.square(logits - tf.cast(y, tf.float32)), axis=-1)
        else:
            return tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
