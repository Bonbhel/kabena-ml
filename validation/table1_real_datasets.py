"""Table 1 du preprint — parité à coût réduit sur jeux réels sklearn."""
from _common import *
from sklearn.datasets import load_breast_cancer

def main():
    print("== Table 1 (extrait exécutable : LogReg / Breast Cancer) ==")
    accs = {m: [] for m in ("base", "v2", "v3")}; gains = []
    for s in SEEDS:
        D = load_breast_cancer()
        Xtr, Xte, ytr, yte = split_std(D.data, D.target, s)
        for m in accs:
            w, g = train_logreg(Xtr, ytr, m, epochs=max(EPOCHS, 60), lr=0.5, seed=s)
            accs[m].append(((sig(Xte @ w) > .5).astype(int) == yte).mean())
            if m == "v3": gains.append(g)
    for m in accs:
        print(f"  {m:4s} acc = {np.mean(accs[m]):.4f} ± {np.std(accs[m]):.4f}")
    print(f"  gain calcul v3 = {np.mean(gains)*100:.1f}%   (preprint : parité, 28.5%)")

if __name__ == "__main__":
    main()
