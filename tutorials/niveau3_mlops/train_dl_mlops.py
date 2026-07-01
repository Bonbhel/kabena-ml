"""
tutorials/niveau3_mlops/train_dl_mlops.py
==========================================
MLOps pour Deep Learning K-ABENA — switch PyTorch / TensorFlow via config.

Usage :
    # PyTorch MLP
    python train_dl_mlops.py model=mlp model.framework=torch

    # TensorFlow MLP — même résultats, plateforme différente
    python train_dl_mlops.py model=mlp model.framework=tf

    # CNN PyTorch
    python train_dl_mlops.py model=cnn model.framework=torch kabena.K=0.30

    # Transformer TF
    python train_dl_mlops.py model=transformer model.framework=tf kabena.N=0.4

    # Comparer les deux frameworks sur le même run
    python train_dl_mlops.py --multirun model.framework=torch,tf model=mlp

    # MLflow UI
    mlflow ui --backend-store-uri ./mlruns
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


# ─── Chargement données ───────────────────────────────────────────────────────
def load_data(cfg):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=cfg.dataset.n_samples,
        n_features=cfg.dataset.n_features,
        random_state=cfg.training.seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=cfg.training.seed)


# ─── Dispatch framework ───────────────────────────────────────────────────────
def train(cfg, X_tr, X_te, y_tr, y_te):
    fw   = cfg.model.framework    # "torch" ou "tf"
    arch = cfg.model.name         # "mlp", "cnn", "transformer"
    K    = cfg.kabena.K
    N    = cfg.kabena.N
    ep   = cfg.training.epochs
    lr   = cfg.training.lr
    v    = cfg.kabena.verbose

    log.info(f"Framework: {fw.upper()} | Arch: {arch} | K={K}, N={N}")

    # ── MLP ──────────────────────────────────────────────────────────────────
    if arch == "mlp" and fw == "torch":
        return _mlp_torch(X_tr, X_te, y_tr, y_te, K, N, ep, lr, v, cfg.model)

    elif arch == "mlp" and fw == "tf":
        return _mlp_tf(X_tr, X_te, y_tr, y_te, K, N, ep, lr, v, cfg.model)

    # ── Transformer ──────────────────────────────────────────────────────────
    elif arch == "transformer" and fw == "torch":
        return _transformer_torch(X_tr, X_te, y_tr, y_te, K, N, ep, lr, v, cfg.model)

    elif arch == "transformer" and fw == "tf":
        return _transformer_tf(X_tr, X_te, y_tr, y_te, K, N, ep, lr, v, cfg.model)

    else:
        raise ValueError(f"Combinaison arch={arch} / fw={fw} non supportée ici. "
                         f"Voir dl_pytorch_vs_tf.py pour CNN.")


# ─── MLP PyTorch ─────────────────────────────────────────────────────────────
def _mlp_torch(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose, model_cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kabena.integrations.torch_utils import kabena_filter_torch

    hidden = getattr(model_cfg, "hidden_sizes", [128, 64])
    layers = []; prev = X_tr.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(getattr(model_cfg, "dropout", 0.2))]
        prev = h
    layers.append(nn.Linear(prev, getattr(model_cfg, "n_classes", 2)))
    model = nn.Sequential(*layers)
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    X_t, y_t = torch.tensor(X_tr), torch.tensor(y_tr)
    gains = []

    for epoch in range(epochs):
        losses = F.cross_entropy(model(X_t), y_t, reduction="none")
        mask   = kabena_filter_torch(losses, K=K, N=N)
        if not mask.any(): continue
        L_KA = losses[mask].mean()
        opt.zero_grad(); L_KA.backward(); opt.step()

        m = mask.sum().item()
        gains.append(round((1 - m / len(y_t)) * 100))
        if verbose and epoch % max(1, epochs // 5) == 0:
            log.info(f"  [torch-mlp] Ep {epoch:4d} | loss={L_KA.item():.4f} | gain={gains[-1]}%")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te)).argmax(1).numpy()
    acc = (preds == y_te).mean()
    return acc, float(np.mean(gains)), "torch-mlp"


# ─── MLP TensorFlow ──────────────────────────────────────────────────────────
def _mlp_tf(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose, model_cfg):
    import tensorflow as tf
    from kabena.integrations.tf_utils import KabenaTFTrainer
    from kabena import KabenaConfig

    hidden = getattr(model_cfg, "hidden_sizes", [128, 64])
    dropout = getattr(model_cfg, "dropout", 0.2)
    n_classes = getattr(model_cfg, "n_classes", 2)

    layers = [tf.keras.layers.Input(shape=(X_tr.shape[1],))]
    for h in hidden:
        layers += [tf.keras.layers.Dense(h, activation="relu"),
                   tf.keras.layers.Dropout(dropout)]
    layers.append(tf.keras.layers.Dense(n_classes))

    model = tf.keras.Sequential(layers[1:])
    model.build((None, X_tr.shape[1]))

    cfg     = KabenaConfig(K=K, N=N, task="classification", verbose=verbose)
    trainer = KabenaTFTrainer(model, cfg, tf.keras.optimizers.SGD(lr, momentum=0.9))
    history = trainer.fit(tf.constant(X_tr), tf.constant(y_tr),
                          epochs=epochs, batch_size=len(X_tr))

    preds = tf.argmax(model(X_te, training=False), axis=1).numpy()
    acc   = (preds == y_te).mean()
    gains = [r["gain_pct"] for r in history] if history else [0]
    return acc, float(np.mean(gains)), "tf-mlp"


# ─── Transformer PyTorch ─────────────────────────────────────────────────────
def _transformer_torch(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose, model_cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kabena.integrations.torch_utils import kabena_filter_torch

    seq_len  = getattr(model_cfg, "seq_len", 5)
    d_model  = getattr(model_cfg, "d_model", 64)
    n_heads  = getattr(model_cfg, "num_heads", 4)
    n_layers = getattr(model_cfg, "num_layers", 2)
    n_cls    = getattr(model_cfg, "n_classes", 2)
    d_feat   = X_tr.shape[1] // seq_len

    X_tr_s = X_tr[:, :d_feat*seq_len].reshape(-1, seq_len, d_feat).astype(np.float32)
    X_te_s = X_te[:, :d_feat*seq_len].reshape(-1, seq_len, d_feat).astype(np.float32)

    proj = nn.Linear(d_feat, d_model)
    enc  = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, n_heads, 128, 0.1, batch_first=True),
        num_layers=n_layers
    )
    head = nn.Linear(d_model, n_cls)

    def forward(x):
        return head(enc(proj(x)).mean(1))

    params = list(proj.parameters()) + list(enc.parameters()) + list(head.parameters())
    opt    = torch.optim.Adam(params, lr=lr)
    X_t    = torch.tensor(X_tr_s); y_t = torch.tensor(y_tr)
    gains  = []

    for epoch in range(epochs):
        losses = F.cross_entropy(forward(X_t), y_t, reduction="none")
        mask   = kabena_filter_torch(losses, K=K, N=N)
        if not mask.any(): continue
        L_KA = losses[mask].mean()
        opt.zero_grad(); L_KA.backward(); opt.step()
        m = mask.sum().item()
        gains.append(round((1 - m / len(y_t)) * 100))
        if verbose and epoch % max(1, epochs // 5) == 0:
            log.info(f"  [torch-trf] Ep {epoch:4d} | loss={L_KA.item():.4f} | gain={gains[-1]}%")

    with torch.no_grad():
        preds = forward(torch.tensor(X_te_s)).argmax(1).numpy()
    return (preds == y_te).mean(), float(np.mean(gains)), "torch-transformer"


# ─── Transformer TensorFlow ──────────────────────────────────────────────────
def _transformer_tf(X_tr, X_te, y_tr, y_te, K, N, epochs, lr, verbose, model_cfg):
    import tensorflow as tf
    from kabena.integrations.tf_utils import KabenaTFTrainer
    from kabena import KabenaConfig

    seq_len  = getattr(model_cfg, "seq_len", 5)
    d_model  = getattr(model_cfg, "d_model", 64)
    n_heads  = getattr(model_cfg, "num_heads", 4)
    n_cls    = getattr(model_cfg, "n_classes", 2)
    d_feat   = X_tr.shape[1] // seq_len

    X_tr_s = X_tr[:, :d_feat*seq_len].reshape(-1, seq_len, d_feat).astype(np.float32)
    X_te_s = X_te[:, :d_feat*seq_len].reshape(-1, seq_len, d_feat).astype(np.float32)

    inputs  = tf.keras.Input(shape=(seq_len, d_feat))
    x       = tf.keras.layers.Dense(d_model)(inputs)
    x       = tf.keras.layers.MultiHeadAttention(n_heads, d_model // n_heads)(x, x)
    x       = tf.keras.layers.LayerNormalization()(x)
    x       = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(n_cls)(x)
    model   = tf.keras.Model(inputs, outputs)

    cfg     = KabenaConfig(K=K, N=N, task="classification", verbose=verbose)
    trainer = KabenaTFTrainer(model, cfg, tf.keras.optimizers.Adam(lr))
    history = trainer.fit(tf.constant(X_tr_s), tf.constant(y_tr),
                          epochs=epochs, batch_size=len(X_tr_s))

    preds = tf.argmax(model(X_te_s, training=False), axis=1).numpy()
    acc   = (preds == y_te).mean()
    gains = [r["gain_pct"] for r in history] if history else [0]
    return acc, float(np.mean(gains)), "tf-transformer"


# ─── Point d'entrée ───────────────────────────────────────────────────────────
try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        log.info(f"K-ABENA DL MLOps | fw={cfg.model.framework} | arch={cfg.model.name}")

        X_tr, X_te, y_tr, y_te = load_data(cfg)
        acc, gain, tag = train(cfg, X_tr, X_te, y_tr, y_te)
        log.info(f"[{tag}] Accuracy={acc:.4f} | Gain moyen={gain:.1f}%")

        try:
            import mlflow
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)
            with mlflow.start_run(run_name=tag):
                mlflow.log_params({
                    "K": cfg.kabena.K, "N": cfg.kabena.N,
                    "framework": cfg.model.framework, "arch": cfg.model.name,
                    "epochs": cfg.training.epochs,
                })
                mlflow.log_metrics({"accuracy": acc, "gain_pct_mean": gain})
        except ImportError:
            pass

except ImportError:
    def main():
        import argparse
        parser = argparse.ArgumentParser(description="K-ABENA DL MLOps (sans Hydra)")
        parser.add_argument("--arch",      default="mlp", choices=["mlp","transformer"])
        parser.add_argument("--framework", default="torch", choices=["torch","tf"])
        parser.add_argument("--K",         type=float, default=0.25)
        parser.add_argument("--N",         type=float, default=0.3)
        parser.add_argument("--epochs",    type=int, default=50)
        parser.add_argument("--lr",        type=float, default=0.05)
        parser.add_argument("--verbose",   action="store_true")
        args = parser.parse_args()

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X    = StandardScaler().fit_transform(X).astype(np.float32)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y.astype(np.int64), test_size=0.2)

        class MCfg:
            framework = args.framework; name = args.arch
            hidden_sizes = [128, 64]; dropout = 0.2; n_classes = 2
            d_model = 64; num_heads = 4; num_layers = 2; seq_len = 5

        acc, gain, tag = train(
            type("C", (), {"model": MCfg(), "kabena": type("K", (), {"K": args.K, "N": args.N, "verbose": args.verbose})(),
                           "training": type("T", (), {"epochs": args.epochs, "lr": args.lr})()})(),
            X_tr, X_te, y_tr, y_te
        )
        print(f"\n[{tag}] Accuracy={acc:.4f} | Gain moyen={gain:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main()
