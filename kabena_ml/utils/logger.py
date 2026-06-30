"""
kabena_ml.utils.logger
=======================
Monitoring K-ABENA — KabenaLogger, plot_stats, kabena_report.
"""

from __future__ import annotations
import csv
import json
import datetime
from pathlib import Path
from typing import Optional

import numpy as np


class KabenaLogger:
    """
    Enregistre les statistiques K-ABENA par itération.
    Sauvegarde automatique en CSV + JSON.

    Exemples
    --------
    >>> logger = KabenaLogger("./logs/")
    >>> logger.log(epoch=0, loss=0.42, m=85, n=100, K_used=0.20, N_used=0.3)
    >>> logger.save()
    """

    def __init__(self, log_dir: str = "./kabena_logs/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(
        self,
        epoch: int,
        loss: float,
        m: int,
        n: int,
        K_used: float = None,
        N_used: float = None,
        **extra,
    ):
        record = {
            "epoch":    epoch,
            "loss":     round(float(loss), 6),
            "m":        int(m),
            "n":        int(n),
            "gain_pct": round((1 - m / n) * 100, 2),
            "K_used":   K_used,
            "N_used":   N_used,
            **extra,
        }
        self.records.append(record)

    def save(self) -> Path:
        base = self.log_dir / f"run_{self._run_id}"
        csv_path  = base.with_suffix(".csv")
        json_path = base.with_suffix(".json")

        if self.records:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
                writer.writeheader()
                writer.writerows(self.records)

            with open(json_path, "w") as f:
                json.dump(self.records, f, indent=2)

        return csv_path

    def summary(self) -> dict:
        if not self.records:
            return {}
        return {
            "epochs":        len(self.records),
            "final_loss":    self.records[-1]["loss"],
            "mean_gain_pct": round(np.mean([r["gain_pct"] for r in self.records]), 2),
            "mean_m":        round(np.mean([r["m"] for r in self.records]), 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
def plot_stats(
    history: list[dict],
    save_to: Optional[str] = None,
    title: str = "Monitoring K-ABENA",
):
    """
    Visualisation standard du monitoring K-ABENA.
    3 graphiques : loss, observations actives, distribution des pertes.

    Exemples
    --------
    >>> plot_stats(trainer.history, save_to="stats.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib requis : pip install matplotlib")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs    = [r["epoch"] for r in history]

    axes[0].plot(epochs, [r["loss"] for r in history], color="#1DD0FF", lw=2)
    axes[0].set(xlabel="Époque", ylabel="Loss K-ABENA", title="Convergence")
    axes[0].grid(alpha=0.2)

    pct = [r["m"] / r["n"] * 100 for r in history]
    axes[1].plot(epochs, pct, color="#1D9E75", lw=2, label="Actifs (%)")
    axes[1].axhline(100, color="#B4B2A9", lw=1, ls="--", alpha=0.5)
    axes[1].fill_between(epochs, pct, 100, alpha=0.12, color="#EF9F27",
                         label="Gain comp.")
    axes[1].set(xlabel="Époque", ylabel="Observations actives (%)",
                title="Observations actives m/n", ylim=(0, 108))
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)

    fig.suptitle(title, fontweight="bold", color="#1A3A44")
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"Stats sauvegardées : {save_to}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def benchmark_KN(
    X_train, y_train, X_test, y_test,
    estimator_fn,
    K_range: list[float] = None,
    N_range: list[float] = None,
    epochs: int = 100,
    scoring: str = "accuracy",
    verbose: bool = True,
) -> "BenchmarkResult":
    """
    Exploration automatique de la grille K × N.
    Retourne un BenchmarkResult avec plot_heatmap() et best_params().

    Exemples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> res = benchmark_KN(X_tr, y_tr, X_te, y_te,
    ...     estimator_fn=lambda: LogisticRegression(max_iter=1),
    ...     K_range=[0.05, 0.10, 0.20], N_range=[0.0, 0.3, 0.6])
    >>> res.plot_heatmap()
    >>> print(res.best_params())
    """
    from kabena_ml.integrations.sklearn_wrapper import KabenaWrapper
    from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

    K_range = K_range or [0.05, 0.10, 0.15, 0.20, 0.30]
    N_range = N_range or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = []
    task = "regression" if scoring in ("mse", "r2") else "classification"

    for K in K_range:
        for N in N_range:
            est = KabenaWrapper(estimator_fn(), K=K, N=N,
                                epochs=epochs, task=task)
            est.fit(X_train, y_train)
            preds = est.predict(X_test)

            if scoring == "accuracy":
                score = accuracy_score(y_test, preds)
            elif scoring == "f1":
                score = f1_score(y_test, preds, average="weighted")
            elif scoring == "mse":
                score = -mean_squared_error(y_test, preds)
            else:
                score = accuracy_score(y_test, preds)

            gain = est.stats_["mean_gain_pct"]
            results.append({"K": K, "N": N, "score": score, "gain_pct": gain})

            if verbose:
                print(f"K={K:.2f}, N={N:.1f} → {scoring}={score:.4f}, gain={gain:.1f}%")

    return BenchmarkResult(results, K_range, N_range, scoring)


class BenchmarkResult:
    def __init__(self, records, K_range, N_range, metric):
        self.records = records
        self.K_range = K_range
        self.N_range = N_range
        self.metric  = metric

    def best_params(self) -> dict:
        return max(self.records, key=lambda r: r["score"])

    def plot_heatmap(self, save_to: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib requis")

        import numpy as np
        scores = np.array([[r["score"] for r in self.records
                            if r["K"] == K and r["N"] == N][0]
                           for N in self.N_range for K in self.K_range])
        scores = scores.reshape(len(self.N_range), len(self.K_range))

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(scores, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(self.K_range)))
        ax.set_xticklabels([f"{k:.2f}" for k in self.K_range])
        ax.set_yticks(range(len(self.N_range)))
        ax.set_yticklabels([f"{n:.1f}" for n in self.N_range])
        ax.set_xlabel("Seuil K"); ax.set_ylabel("N (conservé)")
        ax.set_title(f"Grille K×N — {self.metric}", fontweight="bold")
        plt.colorbar(im, ax=ax, label=self.metric)

        best = self.best_params()
        print(f"Meilleur : K={best['K']}, N={best['N']} → {self.metric}={best['score']:.4f}")

        if save_to:
            fig.savefig(save_to, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
