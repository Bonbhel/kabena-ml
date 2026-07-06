"""Bac à sable K-ABENA — explorer gains et limites du preprint en une commande.

Exemples :
    python playground.py --regime standard --strategy v3
    python playground.py --regime fraude --strategy v2      # déclenche le garde-fou
    python playground.py --regime bruit --noise 0.4 --strategy v2   # Limite L5
    python playground.py --regime standard --sweep alpha    # sensibilité alpha
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _common import *          # noqa
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

def build(regime, noise, seed):
    if regime == "fraude":
        X, y = make_fraud(seed=seed); metric = "AUC"
    else:
        D = load_breast_cancer(); X, y = D.data, D.target; metric = "accuracy"
    Xtr, Xte, ytr, yte = split_std(X, y, seed)
    if regime == "bruit" and noise > 0:
        rng = np.random.default_rng(seed+500)
        flip = rng.random(len(ytr)) < noise
        ytr = ytr.copy(); ytr[flip] = 1 - ytr[flip]
    return Xtr, Xte, ytr, yte, metric

def evaluate(w, Xte, yte, metric):
    s = sig(Xte @ w)
    return roc_auc_score(yte, s) if metric == "AUC" else ((s > .5).astype(int) == yte).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["standard", "fraude", "bruit"], default="standard")
    ap.add_argument("--strategy", choices=["base", "v1", "v2", "v3", "auto"], default="v3")
    ap.add_argument("--N", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sweep", choices=["alpha", "N", "none"], default="none")
    a = ap.parse_args()

    grid = {"alpha": [0.0 + 1e-9, 0.15, 0.3, 0.5, 1.0], "N": [0.1, 0.2, 0.3, 0.4],
            "none": [None]}[a.sweep]
    for val in grid:
        scores, gains = [], []
        for s in range(a.seeds):
            Xtr, Xte, ytr, yte, metric = build(a.regime, a.noise, s)
            if a.strategy == "base":
                w, g = train_logreg(Xtr, ytr, "base", EPOCHS, 0.1 if a.regime=="fraude" else 0.5, s)
            else:
                from kabena import Kabena
                kw = {"alpha": val} if a.sweep == "alpha" else {}
                N = val if a.sweep == "N" else a.N
                kb = Kabena(N=N, strategy=a.strategy, seed=s+50, **kw)
                n, d = Xtr.shape; w = np.zeros(d)
                for _ in range(EPOCHS):
                    act, sw = kb.select(logloss(w, Xtr, ytr), y=ytr)
                    w = w - (0.1 if a.regime=="fraude" else 0.5)*grad_weighted(w, Xtr[act], ytr[act], sw[act])
                gains.append(kb.last_gain_)
            scores.append(evaluate(w, Xte, yte, metric))
        tag = f" [{a.sweep}={val}]" if a.sweep != "none" else ""
        gtxt = f"  gain={np.mean(gains)*100:.1f}%" if gains else ""
        print(f"{a.regime}/{a.strategy}{tag}: {metric}={np.mean(scores):.4f} ± {np.std(scores):.4f}{gtxt}")

if __name__ == "__main__":
    main()
