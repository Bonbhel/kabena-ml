"""Table 4 du preprint — rivaux à budget de calcul égal, tests de permutation appariés.
Méthodes : baseline, K-ABENA v3, Focal Loss (gamma=2, plein batch), OHEM-style
(top-pertes au même budget), IS global loss-proportionnel + correction auto-normalisée.
Caveat honnête (L4) : re-implémentations CPU à budget égal, pas les codebases officielles."""
from _common import *
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from kabena import Kabena

METHODS = ("base", "v3", "focal", "ohem", "is")

def train(Xtr, ytr, method, epochs, lr, seed, Kpct=40.0, N=0.3):
    n, d = Xtr.shape
    w = np.zeros(d)
    rng = np.random.default_rng(seed + 50)
    kb = Kabena(N=N, seed=seed + 50, k_percentile=Kpct) if method == "v3" else None
    used = 0
    for _ in range(epochs):
        L = logloss(w, Xtr, ytr)
        sw = np.ones(n)
        if method == "base":
            a = np.ones(n, bool)
        elif method == "v3":
            a, sw = kb.select(L)
        elif method == "focal":                       # plein batch : aucune économie
            p = sig(Xtr @ w); pt = np.where(ytr == 1, p, 1 - p)
            a = np.ones(n, bool); sw = (1 - pt) ** 2 + 1e-8
        else:
            K = np.percentile(L, Kpct)
            budget = int(round(n * (1 - (1 - N) * (L <= K).mean())))   # même budget que v3
            if method == "ohem":                      # top-pertes, moyenne simple (biaisé)
                idx = np.argsort(L)[::-1][:budget]
                a = np.zeros(n, bool); a[idx] = True
            else:                                     # "is" : proposal global p ∝ L + Hájek
                p = np.maximum(L, 1e-12); p = p / p.sum()
                idx = rng.choice(n, size=budget, replace=False, p=p)
                a = np.zeros(n, bool); a[idx] = True
                pi = np.minimum(1.0, budget * p)
                sw[idx] = 1.0 / pi[idx]
        used += a.sum()
        w = w - lr * grad_weighted(w, Xtr[a], ytr[a], sw[a])
    return w, 1 - used / (n * epochs)

def bench(name, data_fn, metric_fn, seeds, epochs, lr):
    print(f"== Table 4 — {name} (design apparié, {len(list(seeds))} graines) ==")
    res = {m: [] for m in METHODS}; gains = {m: [] for m in METHODS}
    for s in seeds:
        Xtr, Xte, ytr, yte = data_fn(s)
        for m in METHODS:
            w, g = train(Xtr, ytr, m, epochs, lr, s)
            res[m].append(metric_fn(w, Xte, yte)); gains[m].append(g)
    for m in METHODS:
        mu, lo, hi = ci95(res[m])
        p = perm_test(np.array(res[m]) - np.array(res["base"])) if m != "base" else float("nan")
        print(f"  {m:6s} {mu:.4f} [{lo:.4f},{hi:.4f}]  gain={np.mean(gains[m])*100:5.1f}%  p_perm={p:.3f}")

def main():
    seeds_bc = range(3) if QUICK else range(10)
    seeds_fr = range(2) if QUICK else range(5)
    def bc(s):
        D = load_breast_cancer(); return split_std(D.data, D.target, s)
    def fr(s):
        X, y = make_fraud(seed=s); return split_std(X, y, s)
    bench("Breast Cancer (accuracy)", bc,
          lambda w, X, y: (((sig(X @ w) > .5).astype(int)) == y).mean(),
          seeds_bc, max(EPOCHS, 60), 0.5)
    bench("Fraude 0,17% (AUC)", fr,
          lambda w, X, y: roc_auc_score(y, sig(X @ w)),
          seeds_fr, EPOCHS, 0.1)
    print("  (preprint Table 4 : v3 p=1.000/0.504 ; IS global p=0.002 en standard ; OHEM 0.45 en fraude)")

if __name__ == "__main__":
    main()
