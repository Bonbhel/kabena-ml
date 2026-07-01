"""
tutorials/niveau2_scripts/sklearn_project.py
=============================================
Projet complet scikit-learn avec K-ABENA.
Couvre : régression, classification, données déséquilibrées.

Usage :
    python sklearn_project.py --task regression
    python sklearn_project.py --task classification
    python sklearn_project.py --task imbalanced
"""

import argparse
import numpy as np
from sklearn.datasets import (fetch_california_housing, make_classification)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_squared_error, classification_report,
                              f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler

from kabena import calibrate_K, KabenaConfig
from kabena.integrations.sklearn_wrapper import KabenaWrapper
from kabena.utils.logger import KabenaLogger, plot_stats, benchmark_KN


# ─────────────────────────────────────────────────────────────────────────────
def run_regression(args):
    """Régression California Housing — Ridge vs Ridge+K-ABENA."""
    print("\n" + "="*60)
    print("K-ABENA — Régression linéaire (California Housing)")
    print("="*60)

    # Données
    data  = fetch_california_housing()
    X     = StandardScaler().fit_transform(data.data)
    y     = data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calibrage auto de K
    ridge0   = Ridge(alpha=1.0).fit(X_tr, y_tr)
    K_auto   = calibrate_K(np.abs(y_tr - ridge0.predict(X_tr)))
    print(f"K calibré automatiquement : {K_auto:.4f}")

    # Modèle standard
    mse_std = mean_squared_error(y_te, ridge0.predict(X_te))

    # Modèle K-ABENA — seul changement : wrapper
    logger   = KabenaLogger("./logs/")
    ka_ridge = KabenaWrapper(
        Ridge(alpha=1.0),
        K=K_auto, N=args.N, epochs=args.epochs,
        task="regression", verbose=args.verbose
    )
    ka_ridge.fit(X_tr, y_tr)

    mse_ka = mean_squared_error(y_te, ka_ridge.predict(X_te))

    print(f"\nRésultats :")
    print(f"  MSE Ridge standard : {mse_std:.4f}")
    print(f"  MSE Ridge K-ABENA  : {mse_ka:.4f}")
    print(f"  Gain computationnel : {ka_ridge.stats_['mean_gain_pct']:.1f}%")

    # Rapport
    logger.save()
    if args.plot:
        plot_stats(ka_ridge.history_, title="Régression K-ABENA")


# ─────────────────────────────────────────────────────────────────────────────
def run_classification(args):
    """Classification binaire — LogReg vs LogReg+K-ABENA + GridSearch K×N."""
    print("\n" + "="*60)
    print("K-ABENA — Classification logistique")
    print("="*60)

    X, y = make_classification(n_samples=3000, n_features=20, random_state=42)
    X    = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standard
    lr_std = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    acc_std = accuracy_score(y_te, lr_std.predict(X_te))

    # K-ABENA
    ka_lr = KabenaWrapper(
        LogisticRegression(max_iter=1),
        K=args.K, N=args.N, epochs=args.epochs,
        task="classification", verbose=args.verbose
    ).fit(X_tr, y_tr)

    acc_ka = accuracy_score(y_te, ka_lr.predict(X_te))
    print(f"\nAccuracy standard : {acc_std:.4f}")
    print(f"Accuracy K-ABENA  : {acc_ka:.4f}")
    print(f"Gain computationnel : {ka_lr.stats_['mean_gain_pct']:.1f}%")

    # Grid search K × N
    if args.grid:
        print("\nGrid search K × N...")
        results = benchmark_KN(
            X_tr, y_tr, X_te, y_te,
            estimator_fn=lambda: LogisticRegression(max_iter=1),
            K_range=[0.10, 0.15, 0.20, 0.30],
            N_range=[0.0, 0.3, 0.6],
            epochs=100, scoring="f1",
        )
        best = results.best_params()
        print(f"Meilleur : K={best['K']}, N={best['N']} → F1={best['score']:.4f}")
        if args.plot:
            results.plot_heatmap(save_to="benchmark_KN.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_imbalanced(args):
    """Données déséquilibrées — K-ABENA Stratifié."""
    print("\n" + "="*60)
    print("K-ABENA — Données déséquilibrées (ratio 85/15)")
    print("="*60)

    X, y = make_classification(n_samples=3000, weights=[0.85, 0.15], random_state=42)
    X    = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Standard
    lr_std = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    f1_std = f1_score(y_te, lr_std.predict(X_te), average="weighted")

    # K-ABENA Standard
    ka_std = KabenaWrapper(
        LogisticRegression(max_iter=1),
        K=args.K, N=args.N, epochs=args.epochs, task="classification"
    ).fit(X_tr, y_tr)
    f1_ka = f1_score(y_te, ka_std.predict(X_te), average="weighted")

    # K-ABENA Stratifié — filtrage indépendant par classe
    ka_strat = KabenaWrapper(
        LogisticRegression(max_iter=1),
        K=args.K, N=args.N, epochs=args.epochs,
        task="classification", stratified=True
    ).fit(X_tr, y_tr)
    f1_strat = f1_score(y_te, ka_strat.predict(X_te), average="weighted")

    print(f"\nF1 Weighted — Standard    : {f1_std:.4f}")
    print(f"F1 Weighted — K-ABENA     : {f1_ka:.4f}")
    print(f"F1 Weighted — K-ABENA Str.: {f1_strat:.4f}")
    print(f"\nGain K-ABENA standard  : {ka_std.stats_['mean_gain_pct']:.1f}%")
    print(f"Gain K-ABENA stratifié : {ka_strat.stats_['mean_gain_pct']:.1f}%")

    print("\n--- Rapport K-ABENA Stratifié ---")
    print(classification_report(y_te, ka_strat.predict(X_te)))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="K-ABENA scikit-learn")
    parser.add_argument("--task",    default="classification",
                        choices=["regression", "classification", "imbalanced"])
    parser.add_argument("--K",       type=float, default=0.20)
    parser.add_argument("--N",       type=float, default=0.0)
    parser.add_argument("--epochs",  type=int,   default=200)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot",    action="store_true")
    parser.add_argument("--grid",    action="store_true")
    args = parser.parse_args()

    if args.task == "regression":
        run_regression(args)
    elif args.task == "classification":
        run_classification(args)
    elif args.task == "imbalanced":
        run_imbalanced(args)


if __name__ == "__main__":
    main()
