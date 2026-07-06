"""Table 3 du preprint — moments du gradient (biais/variance) et bruit d'étiquettes."""
from _common import *
from sklearn.datasets import load_breast_cancer
from kabena import Kabena

def moments():
    print("== Table 3a (moments du gradient, fraude, itéré pré-entraîné) ==")
    for m in ("v1", "v2", "v3"):
        bs, vs = [], []
        for s in SEEDS:
            X, y = make_fraud(seed=s)
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
            w = np.zeros(X.shape[1])
            for _ in range(8):
                w = w - 0.1*full_grad(w, X, y)
            gf = full_grad(w, X, y)
            kb = Kabena(strategy=m, seed=s+99).force()
            gs = []
            for _ in range(8 if QUICK else 12):
                a, sw = kb.select(logloss(w, X, y))
                gs.append(grad_weighted(w, X[a], y[a], sw[a]))
            gs = np.array(gs); gm = gs.mean(0)
            bs.append(np.linalg.norm(gm-gf)); vs.append(np.mean(np.linalg.norm(gs-gm, axis=1)**2))
        print(f"  {m}: biais={np.mean(bs):.4f}  variance={np.mean(vs):.2e}   "
              f"(preprint : v1 0.128 | v2 0.151 | v3 0.004)")

def noise():
    print("== Table 3b (bruit d'étiquettes 40%, Breast Cancer) ==")
    res = {m: [] for m in ("base", "v2", "v3")}
    for s in SEEDS:
        D = load_breast_cancer()
        Xtr, Xte, ytr, yte = split_std(D.data, D.target, s)
        rng = np.random.default_rng(s+500)
        yn = ytr.copy(); flip = rng.random(len(ytr)) < 0.40; yn[flip] = 1 - yn[flip]
        for m in res:
            w, _ = train_logreg(Xtr, yn, m, epochs=max(EPOCHS, 60), lr=0.5, seed=s)
            res[m].append(((sig(Xte @ w) > .5).astype(int) == yte).mean())
    for m in res:
        print(f"  {m:4s} acc(test propre) = {np.mean(res[m]):.4f}   "
              f"(preprint : base 0.832 | v2 0.386 EFFONDREMENT | v3 0.808)")

if __name__ == "__main__":
    moments(); noise()
