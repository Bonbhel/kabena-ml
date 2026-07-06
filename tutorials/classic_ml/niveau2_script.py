"""Niveau 2 — ML classique : les 3 familles (logistique, SVM, softmax) en un script.
Exécutable tel quel : python niveau2_script.py   (CPU, ~30 s)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kabena import Kabena

def sig(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))
def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); e = np.exp(Z); return e/e.sum(1, keepdims=True)

def run_logistic(seed=0):
    D = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=.25, random_state=seed, stratify=D.target)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    kb = Kabena(seed=seed)                                            # ligne 1
    w = np.zeros(Xtr.shape[1])
    for _ in range(60):
        p = np.clip(sig(Xtr @ w), 1e-9, 1-1e-9)
        losses = -(ytr*np.log(p) + (1-ytr)*np.log(1-p))
        a, sw = kb.select(losses)                                     # ligne 2
        w = w - 0.5 * Xtr[a].T @ ((sig(Xtr[a] @ w) - ytr[a]) * sw[a]) / sw[a].sum()
    return ((sig(Xte @ w) > .5).astype(int) == yte).mean(), kb.last_gain_

def run_svm(seed=0):
    D = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=.25, random_state=seed, stratify=D.target)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    y2 = 2*ytr - 1
    kb = Kabena(seed=seed)
    w = np.zeros(Xtr.shape[1])
    for _ in range(60):
        margins = 1 - y2 * (Xtr @ w)
        losses = np.maximum(0, margins)**2                            # hinge au carré
        a, sw = kb.select(losses)
        act = a & (margins > 0)
        if act.sum():
            g = -2 * (Xtr[act] * (y2[act]*margins[act]*sw[act])[:, None]).sum(0) / sw[act].sum()
            w = w - 0.05 * g
    return (((Xte @ w) > 0).astype(int) == yte).mean(), kb.last_gain_

def run_softmax(seed=0):
    D = load_digits(); C = 10
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=.25, random_state=seed, stratify=D.target)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    kb = Kabena(seed=seed)
    W = np.zeros((Xtr.shape[1], C))
    for _ in range(60):
        P = softmax(Xtr @ W)
        losses = -np.log(np.clip(P[np.arange(len(ytr)), ytr], 1e-9, 1))
        a, sw = kb.select(losses)
        Y = np.zeros_like(P); Y[np.arange(len(ytr)), ytr] = 1
        W = W - 0.5 * Xtr[a].T @ ((P[a] - Y[a]) * sw[a][:, None]) / sw[a].sum()
    return (softmax(Xte @ W).argmax(1) == yte).mean(), kb.last_gain_

if __name__ == "__main__":
    for name, fn in [("logistique", run_logistic), ("SVM", run_svm), ("softmax", run_softmax)]:
        acc, gain = fn()
        print(f"{name:11s}: accuracy={acc:.4f}  gain calcul={gain*100:.1f}%")
