"""Table 2 du preprint — échec des variantes non compensées et résolution v3 (0,17%)."""
from _common import *
from sklearn.metrics import roc_auc_score

def main():
    print("== Table 2 (fraude 0,17%) ==")
    aucs = {m: [] for m in ("base", "v1", "v2", "v3")}
    for s in SEEDS:
        X, y = make_fraud(seed=s)
        Xtr, Xte, ytr, yte = split_std(X, y, s)
        for m in aucs:
            w, _ = train_logreg(Xtr, ytr, m, epochs=EPOCHS, lr=0.1, seed=s)
            aucs[m].append(roc_auc_score(yte, sig(Xte @ w)))
    for m in aucs:
        print(f"  {m:4s} AUC = {np.mean(aucs[m]):.4f} ± {np.std(aucs[m]):.4f}")
    print("  (preprint : base 0.9998 | v1 0.56 | v2 0.53 | v3 0.9991 — Prop. 2)")

if __name__ == "__main__":
    main()
