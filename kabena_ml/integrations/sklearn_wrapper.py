"""
kabena_ml.integrations.sklearn_wrapper
=======================================
Intégration scikit-learn — coût syntaxique minimal.

Avant K-ABENA :
    model = LogisticRegression()
    model.fit(X, y)

Après K-ABENA (+1 ligne, même API) :
    model = KabenaWrapper(LogisticRegression(), K=0.20)
    model.fit(X, y)       # ← API identique
    model.predict(X)      # ← identique
    model.score(X, y)     # ← identique
"""

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from kabena_ml.core.filter import kabena_filter, kabena_safe, calibrate_K
from kabena_ml.core.config import KabenaConfig


class KabenaWrapper(BaseEstimator):
    """
    Wrapper K-ABENA pour tout estimateur scikit-learn gradient-based.

    Paramètres
    ----------
    estimator : BaseEstimator — estimateur sklearn (max_iter=1 recommandé)
    K         : float | "auto" — seuil absolu ou calibrage automatique
    N         : float in [0, 1] — proportion de mineures conservées
    epochs    : int — nombre d'itérations K-ABENA
    task      : "regression" | "classification"
    stratified: bool — filtrage indépendant par classe

    Exemples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = KabenaWrapper(LogisticRegression(max_iter=1), K=0.20)
    >>> model.fit(X_train, y_train)
    >>> model.score(X_test, y_test)
    """

    def __init__(
        self,
        estimator,
        K: float | str = "auto",
        N: float = 0.0,
        epochs: int = 200,
        task: str = "classification",
        stratified: bool = False,
        verbose: bool = False,
    ):
        self.estimator  = estimator
        self.K          = K
        self.N          = N
        self.epochs     = epochs
        self.task       = task
        self.stratified = stratified
        self.verbose    = verbose

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.estimator_ = clone(self.estimator)
        self.history_   = []
        self.n_features_in_ = X.shape[1]

        # Première époque pour calibrer K si "auto"
        self.estimator_.fit(X, y)
        K_ = self._get_K(X, y)

        for epoch in range(self.epochs):
            errors = self._individual_errors(X, y)

            if self.stratified and self.task == "classification":
                active = self._stratified_filter(errors, y, K_)
            else:
                active, m = kabena_safe(errors, K_, self.N)

            X_s, y_s = X[active], y[active]
            m = len(X_s)
            if m == 0:
                continue

            self.estimator_.fit(X_s, y_s)
            loss_val = errors[active].mean()
            gain_pct = round((1 - m / len(X)) * 100)
            self.history_.append({"epoch": epoch, "loss": loss_val,
                                  "m": m, "n": len(X), "gain_pct": gain_pct})

            if self.verbose and epoch % 20 == 0:
                print(f"Ep {epoch:4d} | loss={loss_val:.4f} | "
                      f"actifs={m}/{len(X)} | gain={gain_pct}%")

        self.K_  = K_
        self.stats_ = {
            "mean_m":        np.mean([r["m"] for r in self.history_]),
            "mean_gain_pct": np.mean([r["gain_pct"] for r in self.history_]),
            "epochs":        self.epochs,
            "K_used":        K_,
        }
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        check_is_fitted(self, "estimator_")
        return self.estimator_.score(X, y)

    # ── Privé ─────────────────────────────────────────────────────────────

    def _get_K(self, X, y) -> float:
        if isinstance(self.K, (int, float)):
            return float(self.K)
        return calibrate_K(self._individual_errors(X, y), target_pct=0.10)

    def _individual_errors(self, X, y) -> np.ndarray:
        if self.task == "regression":
            pred = self.estimator_.predict(X)
            return np.abs(y - pred)
        else:
            proba = self.estimator_.predict_proba(X)
            y_int = y.astype(int)
            p_true = proba[np.arange(len(y)), y_int]
            return -np.log(np.clip(p_true, 1e-9, 1.0))

    def _stratified_filter(self, errors, y, K_) -> np.ndarray:
        active = np.zeros(len(y), dtype=bool)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            act_c, _ = kabena_safe(errors[idx], K_, self.N)
            active[idx[act_c]] = True
        return active


# ── Alias court ──────────────────────────────────────────────────────────────
KabenaSKLearn   = KabenaWrapper
KabenaStratified = lambda est, **kw: KabenaWrapper(est, stratified=True, **kw)
