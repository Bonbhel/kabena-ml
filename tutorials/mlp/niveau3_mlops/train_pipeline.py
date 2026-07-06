"""Niveau 3 — Pipeline MLP PyTorch + kabena (pertes retardées), artefacts TorchScript.
(Nécessite torch — non exécuté dans l'env. de validation du package, cf. L4.)

    python train_pipeline.py --out artefacts --epochs 40
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kabena import Kabena, __version__ as kabena_version

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artefacts"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--N", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    D = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=0.25,
                                          random_state=a.seed, stratify=D.target)
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32); yt = torch.tensor(ytr)
    model = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.5)                 # SGD : Limitation L1
    crit = nn.CrossEntropyLoss(reduction="none")
    kb = Kabena(N=a.N, seed=a.seed)
    prev = np.zeros(len(ytr)); t0 = time.time()
    for _ in range(a.epochs):
        act, sw = kb.select(prev)
        wts = torch.tensor(sw[act], dtype=torch.float32)
        losses = crit(model(Xt[act]), yt[act])
        ((losses * wts).sum() / wts.sum()).backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            prev = crit(model(Xt), yt).numpy()
    with torch.no_grad():
        acc = float((model(torch.tensor(sc.transform(Xte), dtype=torch.float32)).argmax(1).numpy() == yte).mean())
    torch.jit.script(model).save(os.path.join(a.out, "model.pt"))
    np.savez(os.path.join(a.out, "scaler.npz"), mean=sc.mean_, scale=sc.scale_)
    json.dump({"kabena_version": kabena_version, "test_accuracy": round(acc, 4),
               "compute_saving": round(float(kb.last_gain_ or 0), 4), "N": a.N, "seed": a.seed,
               "train_seconds": round(time.time()-t0, 2)},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print("OK — voir", a.out)

if __name__ == "__main__":
    main()
