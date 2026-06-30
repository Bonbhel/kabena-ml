"""
experiments/ch14/niveau2_scripts/reproduce_ch14_tensorflow.py
==============================================================
Script complet de reproduction des résultats du Chapitre 14 (TensorFlow/Keras).
Miroir TF de reproduce_ch14_pytorch.py — mêmes datasets, mêmes hyperparamètres.

Usage :
    # CIFAR-10 — toutes les variantes
    python reproduce_ch14_tensorflow.py --dataset cifar10

    # CIFAR-10 — variante spécifique
    python reproduce_ch14_tensorflow.py --dataset cifar10 --variant adaptive

    # CIFAR-100
    python reproduce_ch14_tensorflow.py --dataset cifar100

    # Comparer les résultats avec le script PyTorch :
    python reproduce_ch14_pytorch.py    --dataset cifar10 --variant adaptive
    python reproduce_ch14_tensorflow.py --dataset cifar10 --variant adaptive
    → Les deux doivent converger vers les mêmes cibles du ch.14.

Table de correspondance K-ABENA PyTorch ↔ TensorFlow :
    PyTorch :    losses = F.cross_entropy(..., reduction='none')
                 mask   = kabena_filter_torch(losses, K, N)  ← +1 ligne
    TensorFlow : callbacks=[KabenaCallback(K, N)]            ← +1 argument
"""

from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kabena_ml.core.filter import calibrate_K
from kabena_ml.integrations.tf_utils import KabenaCallback, KabenaTFTrainer
from kabena_ml import KabenaConfig


# ═══════════════════════════════════════════════════════════════════════════
# CIBLES CH.14
# ═══════════════════════════════════════════════════════════════════════════
TARGETS_CH14 = {
    "cifar10": {
        "standard": {"top1": 93.2, "gain": 0.0},
        "ka_n4":    {"top1": 94.6, "gain": 14.2},
        "adaptive": {"top1": 94.9, "gain": 19.3},
        "adam_ka":  {"top1": 95.1, "gain": 17.8},
        "focal":    {"top1": 93.8, "gain": 0.0},
    },
    "cifar100": {
        "standard": {"top1": 74.1, "top5": 92.3, "gain": 0.0},
        "adaptive": {"top1": 76.4, "top5": 94.1, "gain": 17.9},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

def get_tf_datasets(dataset: str, batch_size: int):
    """Charge et prépare les datasets TF avec augmentation identique au protocole ch.14."""

    def augment_fn(x, y, img_size=32):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, img_size + 8, img_size + 8)
        x = tf.image.random_crop(x, [img_size, img_size, 3])
        return x, y

    if dataset == "cifar10":
        (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.cifar10.load_data()
        MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    elif dataset == "cifar100":
        (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.cifar100.load_data()
        MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
        STD  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
    else:
        raise ValueError(f"Dataset non supporté ici : {dataset}. Utiliser imagenet_pytorch.py.")

    X_tr = (X_tr.astype("float32") / 255.0 - MEAN) / STD
    X_te = (X_te.astype("float32") / 255.0 - MEAN) / STD
    y_tr, y_te = y_tr.squeeze(), y_te.squeeze()

    n = len(X_tr)
    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(n, seed=42)
                .batch(batch_size)
                .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    test_ds  = (tf.data.Dataset.from_tensor_slices((X_te, y_te))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    n_classes = 10 if dataset == "cifar10" else 100
    return train_ds, test_ds, n_classes


# ═══════════════════════════════════════════════════════════════════════════
# MODÈLE — ResNet adapté CIFAR
# ═══════════════════════════════════════════════════════════════════════════

def residual_block(x, filters, stride=1, name=""):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, stride, padding="same",
                                use_bias=False, name=f"{name}_c1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, 1, padding="same",
                                use_bias=False, name=f"{name}_c2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, stride,
                                          use_bias=False, name=f"{name}_skip")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)
    return tf.keras.layers.ReLU()(x + shortcut)

def build_resnet_cifar(n_classes: int, depth: int = 18):
    """ResNet adapté CIFAR : conv 3×3 stride=1, pas de MaxPool."""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Pas de MaxPool (identique à l'adaptation PyTorch)

    configs_18 = [(64,2,1), (128,2,2), (256,2,2), (512,2,2)]
    configs_50 = [(64,3,1), (128,4,2), (256,6,2), (512,3,2)]
    configs = configs_18 if depth == 18 else configs_50

    for i, (filters, n_blocks, stride) in enumerate(configs):
        x = residual_block(x, filters, stride, name=f"l{i}b0")
        for j in range(1, n_blocks):
            x = residual_block(x, filters, name=f"l{i}b{j}")

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(n_classes)(x)
    return tf.keras.Model(inputs, outputs)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════

class FocalLossWrapper(tf.keras.losses.Loss):
    """Focal Loss (Lin et al. 2017) pour TF/Keras."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # y_pred = logits
        ce   = tf.keras.losses.sparse_categorical_crossentropy(
                   y_true, y_pred, from_logits=True)
        p_t  = tf.exp(-ce)
        fl   = self.alpha * (1 - p_t) ** self.gamma * ce
        return tf.reduce_mean(fl)


# ═══════════════════════════════════════════════════════════════════════════
# BOUCLE D'EXPÉRIENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment_tf(
    dataset:    str,
    model_name: str,
    variant:    str,
    epochs:     int,
    batch_size: int,
    seed:       int,
    verbose:    bool = True,
) -> dict:
    """Lance une expérience complète TF et retourne les résultats."""

    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_ds, test_ds, n_classes = get_tf_datasets(dataset, batch_size)
    depth = 18 if model_name == "resnet18" else 50
    model = build_resnet_cifar(n_classes, depth=depth)

    # Learning rate cosine (identique PyTorch)
    steps_per_epoch = 50000 // batch_size
    total_steps     = epochs * steps_per_epoch
    lr_schedule     = tf.keras.optimizers.schedules.CosineDecay(
                          initial_learning_rate=0.1, decay_steps=total_steps)

    # Optimiseur
    if variant == "adam_ka":
        optimizer = tf.keras.optimizers.Adam(1e-3, weight_decay=1e-4)
    else:
        optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=0.9, weight_decay=1e-4)

    # Perte selon la variante
    if variant == "focal":
        loss_fn = FocalLossWrapper(gamma=2.0, alpha=0.25)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Métriques
    metrics = ["accuracy"]
    if dataset == "cifar100":
        metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"))

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # ── Callbacks K-ABENA ───────────────────────────────────────────────────
    callbacks = []
    ka_cb = None

    if variant in ("ka_n0", "ka_n4", "adaptive", "adam_ka"):
        # Calibrage auto de K sur quelques batchs
        sample_losses = []
        dummy_opt = tf.keras.optimizers.SGD(0.01)
        model_temp = build_resnet_cifar(n_classes, depth=depth)
        for X_b, y_b in train_ds.take(5):
            logits = model_temp(X_b, training=False)
            l = tf.keras.losses.sparse_categorical_crossentropy(
                    y_b, logits, from_logits=True)
            sample_losses.extend(l.numpy())
        K_auto = calibrate_K(np.array(sample_losses), target_pct=0.10)
        del model_temp

        N_val = 0.0 if variant == "ka_n0" else (0.4 if variant == "ka_n4" else 0.3)

        if variant == "adaptive":
            # Warm-up progressif via KabenaScheduler dans le callback
            # Note : implémentation simplifiée (K fixe par époque)
            # Pour la version complète avec warm-up, utiliser KabenaTFTrainer
            ka_cb = KabenaCallback(K=K_auto, N=N_val, verbose=verbose)
        else:
            ka_cb = KabenaCallback(K=K_auto, N=N_val, verbose=verbose)

        callbacks.append(ka_cb)
        if verbose:
            print(f"  [K-ABENA] K={K_auto:.4f}, N={N_val} | Coût adoption : +1 callback")

    # ── Entraînement ────────────────────────────────────────────────────────
    t0   = time.time()
    hist = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )
    total_time = time.time() - t0

    # Résultats
    top1   = hist.history["val_accuracy"][-1] * 100
    top5   = hist.history.get("val_top5", [0.0])[-1] * 100
    mean_gain = 0.0
    if ka_cb and ka_cb.stats_:
        mean_gain = float(np.mean([s["mean_gain"] for s in ka_cb.stats_]))

    return {
        "dataset":  dataset,
        "model":    model_name,
        "variant":  variant,
        "seed":     seed,
        "top1":     top1,
        "top5":     top5,
        "gain":     mean_gain,
        "time_ep":  total_time / epochs,
        "K_auto":   K_auto if ka_cb else None,
        "history":  {k: [float(v) for v in vals]
                     for k, vals in hist.history.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# RAPPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_report_tf(results: list[dict], dataset: str):
    targets = TARGETS_CH14.get(dataset, {})
    print("\n" + "=" * 75)
    print(f"  RÉSULTATS CH.14 — {dataset.upper()} (TensorFlow) vs Cibles du livre")
    print("=" * 75)
    fmt = "{:<18} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("Variante", "Top-1 obt.", "Top-1 cib.", "Δ", "Gain", "Temps/ép"))
    print("-" * 75)
    for r in results:
        v     = r["variant"]
        tgt   = targets.get(v, {})
        t1_t  = tgt.get("top1", "-")
        delta = f"{r['top1'] - t1_t:+.2f}%" if isinstance(t1_t, float) else "N/A"
        print(fmt.format(
            f"{v} (TF)",
            f"{r['top1']:.2f}%",
            f"{t1_t:.1f}%" if isinstance(t1_t, float) else "-",
            delta,
            f"{r['gain']:.1f}%",
            f"{r['time_ep']:.1f}s",
        ))
    print("=" * 75)
    print("\nGuide migration PyTorch → TensorFlow :")
    print("  PyTorch  : mask = kabena_filter_torch(losses, K, N)  → losses[mask].mean().backward()")
    print("  TF Keras : callbacks=[KabenaCallback(K, N)]           → model.fit(..., callbacks=[...])")
    print("  K et N identiques dans les deux frameworks.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Reproduction Ch.14 K-ABENA — TensorFlow")
    parser.add_argument("--dataset",    default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--model",      default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--variant",    default="all",
                        choices=["all", "standard", "ka_n0", "ka_n4",
                                 "adaptive", "adam_ka", "focal"])
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    print(f"\nTensorFlow version : {tf.__version__}")
    print(f"GPU disponibles   : {tf.config.list_physical_devices('GPU')}")
    print(f"Dataset: {args.dataset} | Modèle: {args.model}")

    epochs   = args.epochs or (200 if args.dataset in ("cifar10","cifar100") else 90)
    variants = (list(TARGETS_CH14.get(args.dataset, {}).keys())
                if args.variant == "all" else [args.variant])

    all_results = []
    for variant in variants:
        print(f"\n{'='*60}\n  Variante : {variant.upper()}\n{'='*60}")
        for seed in args.seeds:
            print(f"\n  → Seed {seed}")
            res = run_experiment_tf(
                dataset=args.dataset, model_name=args.model, variant=variant,
                epochs=epochs, batch_size=args.batch_size, seed=seed, verbose=True
            )
            all_results.append(res)

    print_report_tf(all_results, args.dataset)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir) / "results_ch14_tensorflow.json"
    with open(out, "w") as f:
        compact = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
        json.dump(compact, f, indent=2)
    print(f"\nRésultats sauvegardés : {out}")


if __name__ == "__main__":
    main()
