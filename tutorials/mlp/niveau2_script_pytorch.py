"""Niveau 2 — MLP PyTorch, avec le motif 'pertes retardées' (vrai gain forward+backward).
Le masque de l'époque t est décidé avec les pertes de t-1 : les exclues n'entrent
même pas dans le forward. (Nécessite torch — non exécuté dans l'env. de validation, L4.)"""
import numpy as np
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kabena import Kabena

def main(epochs=40, seed=0):
    D = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=.25,
                                          random_state=seed, stratify=D.target)
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32); yt = torch.tensor(ytr)
    model = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    crit = nn.CrossEntropyLoss(reduction="none")
    kb = Kabena(seed=seed)                                            # ligne 1
    prev_losses = np.zeros(len(ytr))                                  # époque 0 : tout passe
    for ep in range(epochs):
        a, sw = kb.select(prev_losses)                                # ligne 2 (pertes t-1)
        xb, yb = Xt[a], yt[a]
        wts = torch.tensor(sw[a], dtype=torch.float32)
        losses = crit(model(xb), yb)                                  # forward des retenues SEULEMENT
        loss = (losses * wts).sum() / wts.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                         # rafraîchit les pertes (léger, sans grad)
            prev_losses = crit(model(Xt), yt).numpy()
    with torch.no_grad():
        acc = (model(torch.tensor(sc.transform(Xte), dtype=torch.float32)).argmax(1).numpy() == yte).mean()
    print(f"accuracy={acc:.4f}  gain(fwd+bwd)={kb.last_gain_*100:.1f}%")

if __name__ == "__main__":
    main()
