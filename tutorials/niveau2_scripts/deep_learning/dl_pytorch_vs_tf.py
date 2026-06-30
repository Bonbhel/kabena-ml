"""
tutorials/niveau2_scripts/deep_learning/dl_pytorch_vs_tf.py
===========================================================
Projet complet Deep Learning : PyTorch ↔ TensorFlow avec K-ABENA.
Structure miroir — même algorithme dans les deux frameworks.

Usage :
    # Lancer les deux versions pour comparer
    python dl_pytorch_vs_tf.py --arch mlp
    python dl_pytorch_vs_tf.py --arch cnn
    python dl_pytorch_vs_tf.py --arch transformer

    # Comparer directement sur un même run
    python dl_pytorch_vs_tf.py --arch mlp --compare

    # Spécifier K et N
    python dl_pytorch_vs_tf.py --arch mlp --K 0.25 --N 0.3 --epochs 80

Philosophie : chaque section PyTorch est immédiatement suivie de son
              équivalent TensorFlow dans le même ordre.
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# DONNÉES COMMUNES
# ═════════════════════════════════════════════════════════════════════════════

def get_classification_data(n=2000, d=20, seed=42):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=n, n_features=d, n_informative=12,
                                random_state=seed)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return train_test_split(X, y.astype(np.int64), test_size=0.2, random_state=seed)


def accuracy(preds, labels):
    return (np.asarray(preds) == np.asarray(labels)).mean()


# ═════════════════════════════════════════════════════════════════════════════
# MLP — PYTORCH
# ═════════════════════════════════════════════════════════════════════════════

def mlp_pytorch(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kabena_ml.integrations.torch_utils import kabena_filter_torch

    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64),            nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 2)
    )
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    X_t, y_t = torch.tensor(X_tr), torch.tensor(y_tr)
    history = []

    for epoch in range(epochs):
        # ── K-ABENA : 2 lignes de différence avec SGD standard ─────────────
        losses = F.cross_entropy(model(X_t), y_t, reduction="none")
        mask   = kabena_filter_torch(losses, K=K, N=N)
        # ────────────────────────────────────────────────────────────────────
        if not mask.any(): continue
        L_KA = losses[mask].mean()
        opt.zero_grad(); L_KA.backward(); opt.step()

        m = mask.sum().item()
        history.append({"loss": L_KA.item(), "m": m, "gain_pct": round((1-m/len(y_t))*100)})
        if verbose and epoch % (epochs//5) == 0:
            print(f"  [PT-MLP] Ep {epoch:4d} | loss={L_KA.item():.4f} | gain={history[-1]['gain_pct']}%")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te)).argmax(1).numpy()
    return accuracy(preds, y_te), history


# ═════════════════════════════════════════════════════════════════════════════
# MLP — TENSORFLOW
# ═════════════════════════════════════════════════════════════════════════════

def mlp_tensorflow(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose):
    import tensorflow as tf
    from kabena_ml.integrations.tf_utils import KabenaTFTrainer
    from kabena_ml import KabenaConfig

    # Même architecture que PyTorch
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_tr.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2),
    ])
    cfg     = KabenaConfig(K=K, N=N, task="classification", verbose=verbose)
    trainer = KabenaTFTrainer(model, cfg, tf.keras.optimizers.SGD(lr, momentum=0.9))

    # ── K-ABENA intégré dans la boucle GradientTape ─────────────────────────
    history = trainer.fit(
        tf.constant(X_tr), tf.constant(y_tr),
        epochs=epochs, batch_size=len(X_tr)
    )

    preds = tf.argmax(model(X_te, training=False), axis=1).numpy()
    return accuracy(preds, y_te), history


# ═════════════════════════════════════════════════════════════════════════════
# CNN — PYTORCH
# ═════════════════════════════════════════════════════════════════════════════

def cnn_pytorch(K, N, epochs, lr, verbose):
    """CNN CIFAR-10 avec K-ABENA — PyTorch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kabena_ml.integrations.torch_utils import kabena_filter_torch

    try:
        import torchvision, torchvision.transforms as T
    except ImportError:
        print("torchvision requis : pip install torchvision"); return None, []

    tfm    = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tfm),
        batch_size=256, shuffle=True
    )

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.h = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256),
                                    nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10))
        def forward(self, x): return self.h(self.f(x))

    model = CNN()
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    history = []

    for epoch in range(epochs):
        ep_m, ep_n, ep_loss = 0, 0, []
        for Xb, yb in loader:
            # ── K-ABENA : 2 lignes ──────────────────────────────────────────
            losses = F.cross_entropy(model(Xb), yb, reduction="none")
            mask   = kabena_filter_torch(losses, K=K, N=N)
            # ────────────────────────────────────────────────────────────────
            if not mask.any(): continue
            L_KA = losses[mask].mean()
            opt.zero_grad(); L_KA.backward(); opt.step()
            ep_m += mask.sum().item(); ep_n += len(yb); ep_loss.append(L_KA.item())

        if not ep_loss: continue
        gain = round((1-ep_m/ep_n)*100)
        history.append({"epoch": epoch, "loss": np.mean(ep_loss),
                         "m": ep_m, "n": ep_n, "gain_pct": gain})
        if verbose:
            print(f"  [PT-CNN] Ep {epoch+1}/{epochs} | loss={np.mean(ep_loss):.4f} | gain={gain}%")

    return None, history  # accuracy nécessite test loader


# ═════════════════════════════════════════════════════════════════════════════
# CNN — TENSORFLOW
# ═════════════════════════════════════════════════════════════════════════════

def cnn_tensorflow(K, N, epochs, lr, verbose):
    """CNN CIFAR-10 avec K-ABENA — TensorFlow."""
    import tensorflow as tf
    from kabena_ml.integrations.tf_utils import KabenaCallback

    (X_tr, y_tr), _ = tf.keras.datasets.cifar10.load_data()
    X_tr = (X_tr.astype("float32") / 127.5 - 1.0)
    y_tr = y_tr.squeeze()

    # Même architecture que CNN PyTorch (format NHWC pour TF)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding="same", input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(64, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    ka_cb = KabenaCallback(K=K, N=N, verbose=verbose)
    # AVANT : model.fit(X_tr, y_tr, epochs=epochs, batch_size=256)
    # APRÈS ← +1 callback :
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=256,
              callbacks=[ka_cb], verbose=0)

    return None, []  # stats dans ka_cb.stats_


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMER — PYTORCH
# ═════════════════════════════════════════════════════════════════════════════

def transformer_pytorch(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose):
    """Transformer Encoder pour séquences d'embeddings — PyTorch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kabena_ml.integrations.torch_utils import kabena_filter_torch

    # Reshape X en séquences : (n, seq=5, d=4) pour démo légère
    seq_len = 5
    d_model = (X_tr.shape[1] // seq_len) * seq_len
    X_tr_s = X_tr[:, :d_model].reshape(-1, seq_len, d_model // seq_len).astype(np.float32)
    X_te_s = X_te[:, :d_model].reshape(-1, seq_len, d_model // seq_len).astype(np.float32)
    d = d_model // seq_len

    class TransformerCLS(nn.Module):
        def __init__(self):
            super().__init__()
            enc = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=32,
                                              dropout=0.1, batch_first=True)
            self.enc  = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Linear(d, 2)
        def forward(self, x):
            return self.head(self.enc(x).mean(1))

    model = TransformerCLS()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    X_t   = torch.tensor(X_tr_s); y_t = torch.tensor(y_tr)
    history = []

    for epoch in range(epochs):
        losses = F.cross_entropy(model(X_t), y_t, reduction="none")
        mask   = kabena_filter_torch(losses, K=K, N=N)
        if not mask.any(): continue
        L_KA = losses[mask].mean()
        opt.zero_grad(); L_KA.backward(); opt.step()

        m = mask.sum().item()
        history.append({"loss": L_KA.item(), "gain_pct": round((1-m/len(y_t))*100)})
        if verbose and epoch % (epochs//5) == 0:
            print(f"  [PT-TRF] Ep {epoch:4d} | loss={L_KA.item():.4f} | gain={history[-1]['gain_pct']}%")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te_s)).argmax(1).numpy()
    return accuracy(preds, y_te), history


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMER — TENSORFLOW
# ═════════════════════════════════════════════════════════════════════════════

def transformer_tensorflow(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose):
    """Transformer Encoder pour séquences d'embeddings — TensorFlow."""
    import tensorflow as tf
    from kabena_ml.integrations.tf_utils import KabenaTFTrainer
    from kabena_ml import KabenaConfig

    seq_len = 5
    d_model = (X_tr.shape[1] // seq_len) * seq_len
    X_tr_s = X_tr[:, :d_model].reshape(-1, seq_len, d_model // seq_len).astype(np.float32)
    X_te_s = X_te[:, :d_model].reshape(-1, seq_len, d_model // seq_len).astype(np.float32)
    d = d_model // seq_len

    # Même architecture que PyTorch (TF syntax)
    inputs  = tf.keras.Input(shape=(seq_len, d))
    x       = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=d//2)(inputs, inputs)
    x       = tf.keras.layers.LayerNormalization()(inputs + x)
    x       = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model   = tf.keras.Model(inputs, outputs)

    cfg     = KabenaConfig(K=K, N=N, task="classification", verbose=verbose)
    trainer = KabenaTFTrainer(model, cfg, tf.keras.optimizers.Adam(lr))
    history = trainer.fit(tf.constant(X_tr_s), tf.constant(y_tr),
                          epochs=epochs, batch_size=len(X_tr_s))

    preds = tf.argmax(model(X_te_s, training=False), axis=1).numpy()
    return accuracy(preds, y_te), history


# ═════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ ET GUIDE DE MIGRATION
# ═════════════════════════════════════════════════════════════════════════════

MIGRATION_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║             GUIDE DE MIGRATION K-ABENA : PyTorch ↔ TensorFlow           ║
╠══════════════╦══════════════════════════════╦═══════════════════════════╣
║ Concept      ║ PyTorch                      ║ TensorFlow                ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ Import       ║ from kabena_ml.integrations  ║ from kabena_ml.integrations║
║              ║   .torch_utils import        ║   .tf_utils import        ║
║              ║   kabena_filter_torch        ║   KabenaCallback          ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ Coût         ║ +2 lignes dans la boucle     ║ +1 callback dans fit()    ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ Loss indiv.  ║ F.cross_entropy(             ║ automatique dans callback ║
║              ║   ..., reduction='none')     ║                           ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ Filtre       ║ mask = kabena_filter_torch(  ║ callbacks=[               ║
║              ║   losses, K=K, N=N)          ║   KabenaCallback(K, N)]   ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ Backward     ║ losses[mask].mean()          ║ automatique               ║
║              ║   .backward()                ║                           ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ GradTape TF  ║ N/A                          ║ KabenaTFTrainer(          ║
║              ║                              ║   model, cfg, optimizer)  ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ K-ABENA safe ║ kabena_safe_torch(           ║ trainer.filter(losses)    ║
║              ║   losses, K, N, min_m=1)     ║   (dans KabenaTFTrainer)  ║
╠══════════════╬══════════════════════════════╬═══════════════════════════╣
║ Format img   ║ (N, C, H, W)                 ║ (N, H, W, C)              ║
║              ║ K-ABENA agnostique au format — filtre sur la perte       ║
╚══════════════╩══════════════════════════════╩═══════════════════════════╝
Règle d'or : K et N sont identiques dans les deux frameworks.
Seule la syntaxe de manipulation des tenseurs change.
"""


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="K-ABENA Deep Learning — PyTorch ↔ TensorFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--arch",    default="mlp",
                        choices=["mlp", "cnn", "transformer"])
    parser.add_argument("--K",       type=float, default=0.25)
    parser.add_argument("--N",       type=float, default=0.3)
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--lr",      type=float, default=0.05)
    parser.add_argument("--compare", action="store_true",
                        help="Lancer PyTorch ET TensorFlow et comparer")
    parser.add_argument("--framework", default="both",
                        choices=["both", "pytorch", "tensorflow"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"K-ABENA Deep Learning — {args.arch.upper()}")
    print(f"K={args.K}, N={args.N}, epochs={args.epochs}")
    print(f"{'='*60}\n")

    use_pt = args.framework in ("both", "pytorch")
    use_tf = args.framework in ("both", "tensorflow")

    results = {}

    if args.arch == "mlp":
        X_tr, X_te, y_tr, y_te = get_classification_data()
        if use_pt:
            print("── PyTorch MLP ──")
            acc_pt, hist_pt = mlp_pytorch(X_tr, X_te, y_tr, y_te,
                                           args.K, args.N, args.epochs, args.lr, args.verbose)
            results["PyTorch"] = {"acc": acc_pt,
                                  "gain": np.mean([r["gain_pct"] for r in hist_pt])}
            print(f"[PyTorch]  Accuracy={acc_pt:.4f} | Gain={results['PyTorch']['gain']:.1f}%\n")

        if use_tf:
            print("── TensorFlow MLP ──")
            acc_tf, hist_tf = mlp_tensorflow(X_tr, X_te, y_tr, y_te,
                                              args.K, args.N, args.epochs, args.lr, args.verbose)
            results["TensorFlow"] = {"acc": acc_tf,
                                     "gain": np.mean([r["gain_pct"] for r in hist_tf]) if hist_tf else 0}
            print(f"[TensorFlow] Accuracy={acc_tf:.4f} | Gain={results['TensorFlow']['gain']:.1f}%\n")

    elif args.arch == "cnn":
        if use_pt:
            print("── PyTorch CNN CIFAR-10 ──")
            _, hist_pt = cnn_pytorch(args.K, args.N, min(args.epochs, 5), args.lr, args.verbose)
            if hist_pt:
                print(f"[PyTorch] Gain moyen={np.mean([r['gain_pct'] for r in hist_pt]):.1f}%\n")
        if use_tf:
            print("── TensorFlow CNN CIFAR-10 ──")
            cnn_tensorflow(args.K, args.N, min(args.epochs, 5), args.lr, args.verbose)

    elif args.arch == "transformer":
        X_tr, X_te, y_tr, y_te = get_classification_data()
        if use_pt:
            print("── PyTorch Transformer ──")
            acc_pt, hist_pt = transformer_pytorch(X_tr, X_te, y_tr, y_te,
                                                   args.K, args.N, args.epochs, args.lr, args.verbose)
            print(f"[PyTorch]  Accuracy={acc_pt:.4f}\n")
        if use_tf:
            print("── TensorFlow Transformer ──")
            acc_tf, hist_tf = transformer_tensorflow(X_tr, X_te, y_tr, y_te,
                                                      args.K, args.N, args.epochs, args.lr, args.verbose)
            print(f"[TensorFlow] Accuracy={acc_tf:.4f}\n")

    # Résumé comparatif
    if results and args.compare:
        print(f"\n{'─'*50}")
        print("COMPARAISON RÉSULTATS :")
        for fw, r in results.items():
            print(f"  {fw:12s} | Accuracy={r['acc']:.4f} | Gain={r['gain']:.1f}%")

    # Guide de migration
    if args.compare:
        print(MIGRATION_GUIDE)


if __name__ == "__main__":
    main()
