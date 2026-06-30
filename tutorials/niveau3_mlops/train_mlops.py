"""
tutorials/niveau3_mlops/train_mlops.py
=======================================
Projet MLOps complet : Hydra (config) + MLflow (tracking) + K-ABENA.

Usage :
    python train_mlops.py                          # config par défaut
    python train_mlops.py kabena.K=0.25 kabena.N=0.3
    python train_mlops.py model=logistic dataset=housing training.epochs=200
    python train_mlops.py --multirun kabena.K=0.10,0.20,0.30 kabena.N=0.0,0.3

Le même script gère sklearn ET PyTorch selon la config model.framework.
"""

import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def get_data(cfg):
    """Charge le dataset selon la configuration Hydra."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds_cfg = cfg.dataset

    if ds_cfg.name == "make_classification":
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples  = ds_cfg.n_samples,
            n_features = ds_cfg.n_features,
            random_state = ds_cfg.random_state,
        )
    elif ds_cfg.name == "california_housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X, y = data.data, data.target
        cfg.kabena.task = "regression"
    else:
        raise ValueError(f"Dataset inconnu : {ds_cfg.name}")

    X = StandardScaler().fit_transform(X).astype(np.float32)
    return train_test_split(X, y, test_size=ds_cfg.test_size,
                            random_state=ds_cfg.random_state)


def train_sklearn(cfg, X_tr, X_te, y_tr, y_te):
    """Entraînement scikit-learn + K-ABENA."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from kabena_ml.integrations.sklearn_wrapper import KabenaWrapper
    from kabena_ml import calibrate_K

    if cfg.model.name == "logistic_regression":
        base = LogisticRegression(C=cfg.model.C, max_iter=cfg.model.max_iter)
        task = "classification"
    else:
        base = Ridge(alpha=1.0)
        task = "regression"

    # Calibrage auto si K non spécifié
    base_ref = LogisticRegression(max_iter=500).fit(X_tr, y_tr) \
               if task == "classification" else Ridge().fit(X_tr, y_tr)
    if task == "classification":
        proba = base_ref.predict_proba(X_tr)
        errors = -np.log(np.clip(proba[np.arange(len(y_tr)), y_tr.astype(int)], 1e-9, 1))
    else:
        errors = np.abs(y_tr - base_ref.predict(X_tr))
    K_ = cfg.kabena.K or float(calibrate_K(errors))

    model = KabenaWrapper(
        base, K=K_, N=cfg.kabena.N,
        epochs=cfg.training.epochs, task=task,
        verbose=cfg.kabena.verbose,
    )
    model.fit(X_tr, y_tr)
    return model, K_


def train_torch(cfg, X_tr, X_te, y_tr, y_te):
    """Entraînement PyTorch + K-ABENA."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kabena_ml import KabenaConfig
    from kabena_ml.integrations.torch_utils import KabenaTrainer

    # Modèle MLP
    hidden = cfg.model.hidden_sizes
    layers = []
    prev   = X_tr.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.model.dropout)]
        prev = h
    n_classes = len(np.unique(y_tr))
    layers.append(nn.Linear(prev, n_classes))
    model = nn.Sequential(*layers)

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)

    ka_cfg = KabenaConfig(K=cfg.kabena.K, N=cfg.kabena.N,
                          task="classification", verbose=cfg.kabena.verbose)
    trainer = KabenaTrainer(model, ka_cfg, lr=cfg.training.lr,
                             epochs=cfg.training.epochs)
    trainer.fit(X_t, y_t)
    return trainer, cfg.kabena.K


def evaluate(model, X_te, y_te, task, framework):
    """Calcule le score du modèle."""
    from sklearn.metrics import accuracy_score, mean_squared_error

    if framework == "sklearn":
        preds = model.predict(X_te)
    else:  # torch trainer
        import torch
        X_t = torch.tensor(X_te, dtype=torch.float32)
        y_t = torch.tensor(y_te, dtype=torch.long)
        return model.evaluate(X_t, y_t)

    if task == "classification":
        return accuracy_score(y_te, preds)
    else:
        return -mean_squared_error(y_te, preds)  # négatif pour compatibilité


# ─── Point d'entrée Hydra ─────────────────────────────────────────────────────
def train_with_mlflow(cfg):
    """Lance l'entraînement et logue dans MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        use_mlflow = True
    except ImportError:
        log.warning("MLflow non installé — pip install mlflow. Logs désactivés.")
        use_mlflow = False

    # Seed
    np.random.seed(cfg.training.seed)

    # Données
    X_tr, X_te, y_tr, y_te = get_data(cfg)
    log.info(f"Dataset : {X_tr.shape[0]} train, {X_te.shape[0]} test")

    framework = cfg.model.framework
    task      = cfg.kabena.task

    run_ctx = mlflow.start_run() if use_mlflow else _NullContext()

    with run_ctx:
        # Entraînement
        if framework == "sklearn":
            model, K_used = train_sklearn(cfg, X_tr, X_te, y_tr, y_te)
            gain = model.stats_["mean_gain_pct"]
        else:
            model, K_used = train_torch(cfg, X_tr, X_te, y_tr, y_te)
            gain = np.mean([r["gain_pct"] for r in model.history]) if model.history else 0

        # Évaluation
        score = evaluate(model, X_te, y_te, task, framework)
        metric_name = "accuracy" if task == "classification" else "neg_mse"

        log.info(f"{metric_name} = {score:.4f} | gain = {gain:.1f}% | K = {K_used:.4f}")

        # MLflow logging
        if use_mlflow:
            mlflow.log_params({
                "K":        K_used,
                "N":        cfg.kabena.N,
                "epochs":   cfg.training.epochs,
                "lr":       cfg.training.lr,
                "model":    cfg.model.name,
                "dataset":  cfg.dataset.name,
                "framework":framework,
            })
            mlflow.log_metrics({
                metric_name:     score,
                "gain_pct_mean": gain,
            })

            if cfg.mlflow.log_model and framework == "sklearn":
                mlflow.sklearn.log_model(model.estimator_, "model")

        # Sauvegarde locale
        if cfg.output.save_model and framework == "sklearn":
            Path(cfg.output.model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.output.model_path, "wb") as f:
                pickle.dump(model, f)
            log.info(f"Modèle sauvegardé : {cfg.output.model_path}")

        # Visualisation
        if cfg.output.plot and framework == "torch":
            model.plot_stats(save_to="kabena_stats.png")


class _NullContext:
    """Context manager vide quand MLflow n'est pas disponible."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ─── Hydra main ───────────────────────────────────────────────────────────────
try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        log.info(f"K-ABENA MLOps — K={cfg.kabena.K}, N={cfg.kabena.N}, "
                 f"model={cfg.model.name}, dataset={cfg.dataset.name}")
        train_with_mlflow(cfg)

except ImportError:
    # Fallback sans Hydra — utilise argparse
    def main():
        import argparse

        parser = argparse.ArgumentParser(description="K-ABENA MLOps (sans Hydra)")
        parser.add_argument("--K",        type=float, default=0.20)
        parser.add_argument("--N",        type=float, default=0.0)
        parser.add_argument("--epochs",   type=int,   default=100)
        parser.add_argument("--lr",       type=float, default=0.05)
        parser.add_argument("--model",    default="logistic",
                            choices=["logistic", "mlp"])
        parser.add_argument("--dataset",  default="classification",
                            choices=["classification", "housing"])
        parser.add_argument("--verbose",  action="store_true")
        args = parser.parse_args()

        # Config minimale compatible avec train_with_mlflow
        class Cfg:
            pass

        cfg = Cfg()
        cfg.kabena = Cfg()
        cfg.kabena.K = args.K; cfg.kabena.N = args.N
        cfg.kabena.task = "regression" if args.dataset == "housing" else "classification"
        cfg.kabena.verbose = args.verbose; cfg.kabena.stratified = False
        cfg.training = Cfg()
        cfg.training.epochs = args.epochs; cfg.training.lr = args.lr
        cfg.training.batch_size = 64; cfg.training.seed = 42
        cfg.model = Cfg()
        cfg.model.name = "logistic_regression" if args.model == "logistic" else "mlp"
        cfg.model.framework = "sklearn" if args.model == "logistic" else "torch"
        cfg.model.C = 1.0; cfg.model.max_iter = 1
        cfg.model.hidden_sizes = [128, 64]; cfg.model.dropout = 0.2
        cfg.dataset = Cfg()
        cfg.dataset.name = "california_housing" if args.dataset == "housing" else "make_classification"
        cfg.dataset.n_samples = 3000; cfg.dataset.n_features = 20
        cfg.dataset.test_size = 0.2; cfg.dataset.random_state = 42
        cfg.mlflow = Cfg()
        cfg.mlflow.experiment_name = "kabena_experiments"
        cfg.mlflow.tracking_uri = "./mlruns"; cfg.mlflow.log_model = False
        cfg.output = Cfg()
        cfg.output.log_dir = "./logs/"; cfg.output.plot = False
        cfg.output.save_model = False; cfg.output.model_path = "./models/model.pkl"

        logging.basicConfig(level=logging.INFO)
        train_with_mlflow(cfg)


if __name__ == "__main__":
    main()
