"""Niveau 3 — Pipeline d'entraînement MLOps (ML classique).
Produit des artefacts versionnés : modèle joblib + métriques JSON + manifeste kabena.

    python train_pipeline.py --out artefacts/ --epochs 15 --N 0.3
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kabena import Kabena, __version__ as kabena_version
from kabena.integrations.sklearn import fit_sgdclassifier_kabena

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artefacts")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--N", type=float, default=0.3)
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    D = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=0.25,
                                          random_state=a.seed, stratify=D.target)
    scaler = StandardScaler().fit(Xtr)
    kb = Kabena(N=a.N, strategy=a.strategy, seed=a.seed)
    model = SGDClassifier(loss="log_loss", random_state=a.seed)
    t0 = time.time()
    fit_sgdclassifier_kabena(model, scaler.transform(Xtr), ytr, epochs=a.epochs, kb=kb)
    acc = float((model.predict(scaler.transform(Xte)) == yte).mean())

    joblib.dump({"model": model, "scaler": scaler}, os.path.join(a.out, "model.joblib"))
    manifest = {"kabena_version": kabena_version, "strategy": a.strategy, "N": a.N,
                "seed": a.seed, "epochs": a.epochs, "test_accuracy": round(acc, 4),
                "compute_saving": round(float(kb.last_gain_ or 0.0), 4),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "train_seconds": round(time.time() - t0, 2)}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
