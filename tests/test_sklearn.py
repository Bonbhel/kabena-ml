"""
tests/test_sklearn.py
=====================
Tests d'intégration scikit-learn — KabenaWrapper.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

pytest.importorskip("sklearn")
from kabena_ml.integrations.sklearn_wrapper import KabenaWrapper


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=300, n_features=10, noise=0.5, random_state=42)
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)
    y = (y - y.mean()) / y.std()
    return train_test_split(X, y, test_size=0.2, random_state=42)


class TestKabenaWrapper:
    def test_fit_predict_classification(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1), K=0.20, N=0.0, epochs=50, task="classification"
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)
        assert set(preds).issubset({0, 1})

    def test_fit_predict_regression(self, reg_data):
        X_tr, X_te, y_tr, y_te = reg_data
        model = KabenaWrapper(Ridge(alpha=1.0), K=0.30, N=0.0, epochs=50, task="regression")
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)
        assert np.isfinite(preds).all()

    def test_score_classification(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1), K=0.20, N=0.0, epochs=50, task="classification"
        )
        model.fit(X_tr, y_tr)
        score = model.score(X_te, y_te)
        assert 0.0 <= score <= 1.0

    def test_stats_populated(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1), K=0.20, N=0.0, epochs=30, task="classification"
        )
        model.fit(X_tr, y_tr)
        assert "mean_gain_pct" in model.stats_
        assert "epochs" in model.stats_
        assert 0 <= model.stats_["mean_gain_pct"] <= 100

    def test_N1_equivalent_to_standard(self, clf_data):
        """N=1 → K-ABENA désactivé → comportement similaire au standard."""
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1), K=0.50, N=1.0, epochs=50, task="classification"
        )
        model.fit(X_tr, y_tr)
        # Gain doit être ~0 (toutes les mineures conservées)
        assert model.stats_["mean_gain_pct"] < 2.0, "N=1 → aucune exclusion → gain ≈ 0%"

    def test_K_auto(self, clf_data):
        """K='auto' doit fonctionner sans erreur."""
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1), K="auto", N=0.0, epochs=30, task="classification"
        )
        model.fit(X_tr, y_tr)
        assert model.K_ > 0

    def test_stratified_fit(self, clf_data):
        """Mode stratifié ne lève pas d'erreur."""
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1),
            K=0.20,
            N=0.0,
            epochs=30,
            task="classification",
            stratified=True,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)

    def test_predict_proba(self, clf_data):
        X_tr, X_te, y_tr, y_te = clf_data
        model = KabenaWrapper(
            LogisticRegression(max_iter=1), K=0.20, N=0.0, epochs=30, task="classification"
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.shape == (len(y_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
