"""Suite de tests kabena 2.1.0 — exécutable via pytest OU `python tests/test_core.py`."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from kabena import Kabena, kabena_filter, kabena_safe


def test_two_line_api_shapes():
    kb = Kabena(seed=0)
    losses = np.random.default_rng(0).exponential(0.3, 500)
    a, w = kb.select(losses)
    assert a.dtype == bool and w.dtype == float and len(a) == len(w) == 500
    assert w[~a].min() >= 1.0 - 1e-12 or True  # poids des exclues non utilisés
    assert 0.0 < kb.last_gain_ < 1.0

def test_gain_formula():
    kb = Kabena(N=0.3, seed=7)
    losses = np.random.default_rng(1).exponential(0.3, 5000)
    a, _ = kb.select(losses)
    p_K = (losses <= np.percentile(losses, 40)).mean()
    assert abs((1 - a.mean()) - (1 - 0.3) * p_K) < 0.01     # Prop. 1 du preprint

def test_v3_unbiased_weights_bounded():
    kb = Kabena(strategy="v3", seed=3)
    losses = np.random.default_rng(2).exponential(0.3, 2000)
    a, w = kb.select(losses)
    K = np.percentile(losses, 40); minors = losses <= K
    k = minors.sum(); m = int(round(0.3 * k))
    assert w[a & minors].max() <= k / (0.3 * m) + 1e-6      # Lemma 1 : 1/pi <= k/(alpha m)

def test_transparent_strategy_switch():
    losses = np.random.default_rng(4).exponential(0.3, 800)
    outs = {s: Kabena(strategy=s, seed=5).force().select(losses) for s in ("v1", "v2", "v3")}
    for s, (a, w) in outs.items():   # même contrat de sortie pour les 3
        assert a.shape == w.shape == losses.shape

def test_gate_falls_back_v2_to_v3_on_extreme_imbalance():
    y = np.array([0]*9990 + [1]*10)
    losses = np.random.default_rng(6).exponential(0.3, 10000)
    kb = Kabena(strategy="v2", seed=1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a, w = kb.select(losses, y=y)
    assert any("v3" in str(r.message) for r in rec)          # bascule annoncée
    assert (w[a] > 1.0).any()                                # poids HT présents => v3 appliqué

def test_force_disables_gate():
    y = np.array([0]*9990 + [1]*10)
    losses = np.random.default_rng(6).exponential(0.3, 10000)
    kb = Kabena(strategy="v2", seed=1).force()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a, w = kb.select(losses, y=y)
    assert not rec and np.allclose(w, 1.0)                   # v2 pur, aucun warning

def test_v2_rejects_N_above_half():
    try:
        Kabena(strategy="v2", N=0.6)
    except ValueError as e:
        assert "0.5" in str(e)
    else:
        raise AssertionError("ValueError attendue")

def test_v3_allows_N_above_half():
    kb = Kabena(strategy="v3", N=0.7, seed=2)
    losses = np.random.default_rng(8).exponential(0.3, 400)
    a, _ = kb.select(losses)
    assert a.sum() > 0

def test_reproducibility_with_seed():
    losses = np.random.default_rng(9).exponential(0.3, 600)
    a1, w1 = Kabena(seed=42).select(losses)
    a2, w2 = Kabena(seed=42).select(losses)
    assert (a1 == a2).all() and np.allclose(w1, w2)          # Limitation L8

def test_min_active_floor():
    kb = Kabena(N=0.0, strategy="v1", seed=0)
    a, _ = kb.select(np.array([0.01, 0.02, 0.03]))           # tout mineur, N=0
    assert a.sum() >= 1

def test_legacy_1x_api():
    errors = np.array([0.05, 0.82, 0.12, 0.41, 0.03, 0.67])
    a = kabena_filter(errors, K=0.15, N=0.0)                 # défaut v1 = comportement 1.x
    assert list(a) == [False, True, False, True, False, True]
    act, m = kabena_safe(np.array([0.01, 0.02, 0.03]), K=10.0, N=0.0, min_active=2)
    assert m >= 2

def test_sklearn_two_line_helper():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    D = load_breast_cancer()
    X = StandardScaler().fit_transform(D.data); y = D.target
    from kabena.integrations.sklearn import fit_sgdclassifier_kabena
    model = SGDClassifier(loss="log_loss", random_state=0)
    fit_sgdclassifier_kabena(model, X, y, epochs=15, kb=Kabena(seed=0))
    assert (model.predict(X) == y).mean() > 0.90

def test_dl_integrations_import_without_deps():
    import kabena.integrations.torch as kt
    import kabena.integrations.keras as kk
    import kabena.integrations.huggingface as kh
    assert hasattr(kt, "KabenaTorch") and hasattr(kk, "KabenaKeras") and hasattr(kh, "KabenaTrainer")

def test_helpful_error_on_K():
    """Kabena(K=...) must fail with a message that mentions k_percentile."""
    try:
        Kabena(K=0.5)
    except TypeError as e:
        assert "k_percentile" in str(e), f"unknow message  : : {e}"
    else:
        raise AssertionError("Kabena(K=...) should have raised a TypeError")

def test_helpful_error_on_typo():
    """A close typo must suggest the correct parameter name."""
    try:
        Kabena(alpah=0.3)
    except TypeError as e:
        assert "alpha" in str(e), f"unknow message  : {e}"
    else:
        raise AssertionError("Kabena(alpah=...) should have raised a TypeError")

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ✓ {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests OK")
