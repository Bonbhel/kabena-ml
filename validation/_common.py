"""Utilitaires partagés des scripts de validation (seedés, QUICK=1 pour smoke test)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

QUICK = os.environ.get("QUICK") == "1"
SEEDS = range(2) if QUICK else range(5)
EPOCHS = (30 if QUICK else 100)

def sig(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))

def logloss(w, X, y):
    p = np.clip(sig(X @ w), 1e-9, 1-1e-9)
    return -(y*np.log(p) + (1-y)*np.log(1-p))

def full_grad(w, X, y):
    return X.T @ (sig(X @ w) - y) / len(y)

def grad_weighted(w, X, y, sw):
    p = sig(X @ w)
    return X.T @ ((p - y) * sw) / sw.sum()

def make_fraud(n=20000, share=0.0017, d=15, seed=0, sep=1.2):
    rng = np.random.default_rng(seed)
    npos = max(2, int(n*share))
    X = np.vstack([rng.normal(0,1,(n-npos,d)), rng.normal(sep,1,(npos,d))])
    y = np.array([0]*(n-npos) + [1]*npos)
    idx = rng.permutation(n)
    return X[idx], y[idx]

def split_std(X, y, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    strat = y if len(np.unique(y)) < 20 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=strat)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte

def train_logreg(Xtr, ytr, mode, epochs, lr, seed, N=0.3):
    """mode in {'base','v1','v2','v3'} — reproduit exactement le protocole du preprint."""
    from kabena import Kabena
    n, d = Xtr.shape
    w = np.zeros(d)
    kb = None if mode == "base" else Kabena(N=N, strategy=mode, seed=seed+50).force()
    gains = []
    for _ in range(epochs):
        if kb is None:
            a, sw = np.ones(n, bool), np.ones(n)
        else:
            a, sw = kb.select(logloss(w, Xtr, ytr))
            gains.append(kb.last_gain_)
        w = w - lr * grad_weighted(w, Xtr[a], ytr[a], sw[a])
    return w, (float(np.mean(gains)) if gains else 0.0)


# ---- Statistiques appariées (révision peer-review : Table 4 du preprint) ----
def perm_test(diffs, n_perm=20000, seed=0):
    """Test de permutation apparié (sign-flip), bilatéral.
    diffs : différences par graine (méthode - baseline), design apparié."""
    diffs = np.asarray(diffs, dtype=float)
    obs = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (n_perm + 1))

def ci95(x):
    """Moyenne et IC 95% (approx. normale sur graines)."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, m - h, m + h
