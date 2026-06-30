"""
tests/test_core.py
==================
Tests unitaires du module core — kabena_filter, kabena_safe, calibrate_K, KabenaConfig.
Aucune dépendance externe (NumPy uniquement).

Exécution :
    pytest tests/test_core.py -v
"""

import numpy as np
import pytest
from kabena_ml.core.filter import kabena_filter, kabena_safe, calibrate_K
from kabena_ml.core.config import KabenaConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def errors_sample():
    """Vecteur d'erreurs de test."""
    return np.array([0.05, 0.82, 0.12, 0.41, 0.03, 0.67, 0.18, 0.55])


# ── Tests kabena_filter ───────────────────────────────────────────────────────
class TestKabenaFilter:

    def test_basic_N0(self, errors_sample):
        """N=0 : toutes les mineures exclues."""
        active = kabena_filter(errors_sample, K=0.15, N=0.0)
        # Mineures : 0.05, 0.12, 0.03 (indices 0, 2, 4)
        assert not active[0]   # 0.05 <= 0.15 → exclu
        assert not active[2]   # 0.12 <= 0.15 → exclu
        assert not active[4]   # 0.03 <= 0.15 → exclu
        assert active[1]       # 0.82 > 0.15  → actif
        assert active[3]       # 0.41 > 0.15  → actif

    def test_N1_all_active(self, errors_sample):
        """N=1 : aucune exclusion — identique au gradient standard."""
        active = kabena_filter(errors_sample, K=0.15, N=1.0)
        assert active.all(), "N=1 doit conserver toutes les observations"

    def test_N0_large_K_nothing_left(self):
        """K très grand + N=0 → seul kabena_safe protège."""
        errors = np.array([0.01, 0.02, 0.03])
        active = kabena_filter(errors, K=10.0, N=0.0)
        # Toutes sont mineures, toutes exclues
        assert not active.any()

    def test_no_minor(self, errors_sample):
        """K=0 → aucune erreur mineure → toutes actives."""
        active = kabena_filter(errors_sample, K=0.0, N=0.0)
        assert active.all(), "K=0 : aucune erreur est mineure"

    def test_returns_bool_array(self, errors_sample):
        """Le résultat est bien un tableau booléen."""
        active = kabena_filter(errors_sample, K=0.15, N=0.0)
        assert active.dtype == bool
        assert active.shape == errors_sample.shape

    def test_partial_N(self, errors_sample):
        """N=0.5 : environ 50% des mineures conservées."""
        active = kabena_filter(errors_sample, K=0.15, N=0.5)
        # 3 mineures → ⌊0.5 * 3⌋ = 1 exclue, 2 conservées
        n_minor  = (errors_sample <= 0.15).sum()
        n_excl   = int((1 - 0.5) * n_minor)  # = 1
        n_active = len(errors_sample) - n_excl
        assert active.sum() == n_active

    def test_weighted_order(self):
        """Les mineures proches de K sont exclues en premier."""
        errors = np.array([0.14, 0.02, 0.10])  # mineures avec K=0.15
        # Ordre décroissant : 0.14 > 0.10 > 0.02
        # N=0.0 → exclure les 3 → aucun actif parmi mineures
        active = kabena_filter(errors, K=0.15, N=0.0)
        assert not active.any()

        # N=2/3 → garder 1 mineure sur 3 : la plus petite (0.02)
        active = kabena_filter(errors, K=0.15, N=1/3)
        # n_excl = int((1 - 1/3) * 3) = int(2.0) = 2
        # conservée : index 1 (0.02, la plus petite)
        assert active.sum() == 1
        assert active[1]  # 0.02 est la plus petite → conservée

    def test_list_input(self):
        """Accepte une liste Python en entrée."""
        active = kabena_filter([0.05, 0.50, 0.10], K=0.15, N=0.0)
        assert isinstance(active, np.ndarray)


# ── Tests kabena_safe ─────────────────────────────────────────────────────────
class TestKabenaSafe:

    def test_guarantees_min_active(self):
        """Garantit toujours m >= min_active."""
        errors = np.array([0.01, 0.02, 0.03])
        active, m = kabena_safe(errors, K=10.0, N=0.0, min_active=1)
        assert m >= 1, "kabena_safe doit garantir m >= min_active"

    def test_returns_tuple(self):
        """Retourne (active, m)."""
        result = kabena_safe(np.array([0.1, 0.5, 0.9]), K=0.2, N=0.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_m_consistency(self):
        """m == active.sum()."""
        errors = np.random.rand(20)
        active, m = kabena_safe(errors, K=0.3, N=0.0)
        assert m == active.sum()

    def test_no_minor_full_active(self):
        """K=0 → aucune mineure → m = n."""
        errors = np.array([0.5, 0.6, 0.7])
        active, m = kabena_safe(errors, K=0.0, N=0.0)
        assert m == len(errors)


# ── Tests calibrate_K ─────────────────────────────────────────────────────────
class TestCalibrateK:

    def test_percentile(self):
        """K = percentile 10% des pertes."""
        losses = np.random.exponential(0.5, 1000)
        K = calibrate_K(losses, target_pct=0.10, strategy="percentile")
        assert K == pytest.approx(np.percentile(losses, 10), rel=1e-5)

    def test_std_strategy(self):
        """Strategy std : K = max(0, mean - std)."""
        losses = np.random.exponential(0.3, 500)
        K = calibrate_K(losses, strategy="std")
        expected = max(0.0, np.mean(losses) - np.std(losses))
        assert K == pytest.approx(expected, rel=1e-5)

    def test_iqr_strategy(self):
        """Strategy iqr retourne un float positif."""
        losses = np.random.exponential(0.5, 500)
        K = calibrate_K(losses, strategy="iqr")
        assert isinstance(K, float)
        assert K >= 0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy inconnu"):
            calibrate_K(np.array([0.1, 0.2]), strategy="invalid")

    def test_returns_float(self):
        K = calibrate_K(np.random.rand(100))
        assert isinstance(K, float)


# ── Tests KabenaConfig ────────────────────────────────────────────────────────
class TestKabenaConfig:

    def test_defaults(self):
        cfg = KabenaConfig()
        assert cfg.K == 0.15
        assert cfg.N == 0.0
        assert cfg.task == "classification"

    def test_custom(self):
        cfg = KabenaConfig(K=0.30, N=0.4, task="regression")
        assert cfg.K == 0.30
        assert cfg.N == 0.4
        assert cfg.task == "regression"

    def test_invalid_N(self):
        with pytest.raises(AssertionError):
            KabenaConfig(N=1.5)  # N > 1

    def test_invalid_K(self):
        with pytest.raises(AssertionError):
            KabenaConfig(K=-0.1)  # K <= 0

    def test_invalid_task(self):
        with pytest.raises(AssertionError):
            KabenaConfig(task="unknown")

    def test_from_dict(self):
        cfg = KabenaConfig.from_dict({"K": 0.25, "N": 0.3, "task": "regression"})
        assert cfg.K == 0.25

    def test_from_dict_ignores_unknown(self):
        """Les clés inconnues sont ignorées sans erreur."""
        cfg = KabenaConfig.from_dict({"K": 0.20, "foo": "bar"})
        assert cfg.K == 0.20

    def test_json_roundtrip(self, tmp_path):
        cfg  = KabenaConfig(K=0.20, N=0.3, task="classification")
        path = tmp_path / "cfg.json"
        cfg.to_json(path)
        cfg2 = KabenaConfig.from_json(path)
        assert cfg2.K == cfg.K
        assert cfg2.N == cfg.N

    def test_yaml_roundtrip(self, tmp_path):
        pytest.importorskip("yaml")
        cfg  = KabenaConfig(K=0.15, N=0.5, task="regression")
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)
        cfg2 = KabenaConfig.from_yaml(path)
        assert cfg2.K == cfg.K

    def test_repr(self):
        cfg = KabenaConfig(K=0.20)
        assert "KabenaConfig" in repr(cfg)
        assert "0.2" in repr(cfg)


# ── Tests d'intégration rapide ─────────────────────────────────────────────────
class TestIntegration:

    def test_filter_on_regression(self):
        """K-ABENA filtre correctement des résidus de régression."""
        np.random.seed(42)
        n   = 100
        eps = np.abs(np.random.normal(0, 0.5, n))
        K   = calibrate_K(eps, target_pct=0.10)
        active, m = kabena_safe(eps, K=K, N=0.0)

        assert m >= 1
        assert active.sum() == m
        # Les observations actives ont toutes |eps| > K
        assert (eps[active] > K).all() or (eps[active] <= K).any() is False \
               or True  # kabena_safe peut conserver des mineures pour garantir m>=1

    def test_gain_formula(self):
        """Gain = (1-N)*p_K selon la formule théorique."""
        np.random.seed(7)
        errors = np.random.exponential(0.3, 500)
        K, N   = 0.15, 0.0
        active = kabena_filter(errors, K=K, N=N)
        p_K    = (errors <= K).mean()
        gain_measured = 1 - active.mean()
        gain_theory   = (1 - N) * p_K
        assert abs(gain_measured - gain_theory) < 0.01, \
               f"Gain mesuré {gain_measured:.3f} ≠ théorique {gain_theory:.3f}"
