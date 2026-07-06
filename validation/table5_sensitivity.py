"""§Sensitivity du preprint — grille K-percentile × N : la performance est plate,
le gain suit la Proposition 1. (QUICK=1 pour une sous-grille rapide.)"""
from _common import *
import itertools
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from kabena import Kabena

def run(Xtr, ytr, Kpct, N, epochs, lr, seed):
    n, d = Xtr.shape; w = np.zeros(d)
    kb = Kabena(N=N, seed=seed + 50, k_percentile=Kpct)
    gains = []
    for _ in range(epochs):
        a, sw = kb.select(logloss(w, Xtr, ytr))
        gains.append(kb.last_gain_)
        w = w - lr * grad_weighted(w, Xtr[a], ytr[a], sw[a])
    return w, float(np.mean(gains))

def main():
    Ks = (40, 70) if QUICK else (20, 40, 60, 70)
    Ns = (0.1, 0.3) if QUICK else (0.1, 0.3, 0.5)
    seeds = range(2) if QUICK else range(5)

    print("== §Sensitivity — Breast Cancer (accuracy / gain) ==")
    for Kp, N in itertools.product(Ks, Ns):
        accs, gains = [], []
        for s in seeds:
            D = load_breast_cancer()
            Xtr, Xte, ytr, yte = split_std(D.data, D.target, s)
            w, g = run(Xtr, ytr, Kp, N, max(EPOCHS, 60), 0.5, s)
            accs.append((((sig(Xte @ w) > .5).astype(int)) == yte).mean()); gains.append(g)
        print(f"  K={Kp:2d}pct N={N:.1f} : acc={np.mean(accs):.4f}  gain={np.mean(gains)*100:4.1f}%")
    print("  (preprint : acc plate 0.9664-0.9678 ; gain 10.8% -> 63.4%)")

    print("== §Sensitivity — Fraude (AUC) ==")
    for Kp, N in itertools.product(Ks, Ns):
        aucs = []
        for s in seeds:
            X, y = make_fraud(seed=s)
            Xtr, Xte, ytr, yte = split_std(X, y, s)
            w, _ = run(Xtr, ytr, Kp, N, EPOCHS, 0.1, s)
            aucs.append(roc_auc_score(yte, sig(Xte @ w)))
        print(f"  K={Kp}pct N={N:.1f} : AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("  (preprint : AUC >= 0.9961 sur toute la grille)")

if __name__ == "__main__":
    main()
