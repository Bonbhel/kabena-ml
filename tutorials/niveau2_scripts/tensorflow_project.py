"""
tutorials/niveau2_scripts/tensorflow_project.py
================================================
Projet complet TensorFlow/Keras avec K-ABENA.
Couvre : MLP dense, modèle séquentiel, boucle GradientTape.

Usage :
    python tensorflow_project.py --mode callback
    python tensorflow_project.py --mode tape
    python tensorflow_project.py --mode callback --K 0.25 --N 0.3 --epochs 50
"""

import argparse
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from kabena_ml import calibrate_K, KabenaConfig
from kabena_ml.integrations.tf_utils import KabenaCallback, KabenaTFTrainer
from kabena_ml.utils.logger import KabenaLogger, plot_stats


def build_model(input_dim: int, n_classes: int = 2) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes),
    ])


def prepare_data(n_samples=2000, n_features=20, test_size=0.2):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    X    = StandardScaler().fit_transform(X).astype(np.float32)
    return train_test_split(X, y, test_size=test_size, random_state=42)


# ─── Mode 1 : KabenaCallback (coût = +1 argument dans model.fit) ─────────────
def run_callback(args):
    print("\n" + "="*60)
    print("K-ABENA — TF/Keras via KabenaCallback")
    print(f"Coût d'adoption : +1 callbacks=[KabenaCallback(K={args.K}, N={args.N})]")
    print("="*60)

    X_tr, X_te, y_tr, y_te = prepare_data()

    # Calibrage auto
    model0 = build_model(X_tr.shape[1])
    model0.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
    logits0 = model0(X_tr, training=False)
    losses0 = tf.keras.losses.sparse_categorical_crossentropy(y_tr, logits0, from_logits=True)
    K_ = args.K or float(calibrate_K(losses0.numpy()))
    print(f"K utilisé : {K_:.4f}")

    # ── Standard (référence) ──────────────────────────────────────────────────
    model_std = build_model(X_tr.shape[1])
    model_std.compile(
        optimizer=tf.keras.optimizers.SGD(0.05),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model_std.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64,
                  validation_data=(X_te, y_te), verbose=0)
    _, acc_std = model_std.evaluate(X_te, y_te, verbose=0)
    print(f"\nAccuracy standard   : {acc_std:.4f}")

    # ── K-ABENA via callback — SEUL AJOUT : callbacks=[KabenaCallback(...)] ───
    ka_cb   = KabenaCallback(K=K_, N=args.N, verbose=args.verbose)
    model_ka = build_model(X_tr.shape[1])
    model_ka.compile(
        optimizer=tf.keras.optimizers.SGD(0.05),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model_ka.fit(
        X_tr, y_tr,
        epochs      = args.epochs,
        batch_size  = 64,
        validation_data = (X_te, y_te),
        callbacks   = [ka_cb],       # ← SEUL AJOUT
        verbose     = 0,
    )
    _, acc_ka = model_ka.evaluate(X_te, y_te, verbose=0)
    print(f"Accuracy K-ABENA    : {acc_ka:.4f}")
    print(f"\nStats K-ABENA : {ka_cb.stats_summary()}")


# ─── Mode 2 : GradientTape (contrôle total) ──────────────────────────────────
def run_gradient_tape(args):
    print("\n" + "="*60)
    print("K-ABENA — TF/Keras via GradientTape (KabenaTFTrainer)")
    print("="*60)

    X_tr, X_te, y_tr, y_te = prepare_data()

    cfg     = KabenaConfig(K=args.K or 0.25, N=args.N,
                           task="classification", verbose=args.verbose)
    model   = build_model(X_tr.shape[1])
    trainer = KabenaTFTrainer(
        model, cfg,
        optimizer=tf.keras.optimizers.SGD(0.05)
    )

    history = trainer.fit(
        tf.constant(X_tr), tf.constant(y_tr),
        epochs     = args.epochs,
        batch_size = 64,
        val_data   = (tf.constant(X_te), tf.constant(y_te)),
    )

    # Accuracy finale
    logits = model(X_te, training=False)
    preds  = tf.argmax(logits, axis=1).numpy()
    acc    = (preds == y_te).mean()
    print(f"\nAccuracy finale : {acc:.4f}")

    if history and args.plot:
        plot_stats(history, title="TF K-ABENA GradientTape")

    # Logger
    logger = KabenaLogger("./logs/")
    for r in history:
        logger.log(**r)
    csv_path = logger.save()
    print(f"Log sauvegardé : {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="K-ABENA TensorFlow project")
    parser.add_argument("--mode",    default="callback",
                        choices=["callback", "tape"])
    parser.add_argument("--K",       type=float, default=None)
    parser.add_argument("--N",       type=float, default=0.0)
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    if args.mode == "callback":
        run_callback(args)
    else:
        run_gradient_tape(args)


if __name__ == "__main__":
    main()
