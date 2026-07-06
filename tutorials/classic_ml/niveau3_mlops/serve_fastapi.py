"""Niveau 3 — Service d'inférence FastAPI (l'entraînement kabena est en amont ;
au serving, le modèle est un modèle sklearn standard : aucune dépendance kabena).

    uvicorn serve_fastapi:app --port 8000
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
         -d '{"features": [[...30 valeurs...]]}'
"""
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

bundle = joblib.load("artefacts/model.joblib")
app = FastAPI(title="kabena-served model")

class Payload(BaseModel):
    features: list[list[float]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(p: Payload):
    X = bundle["scaler"].transform(np.asarray(p.features, dtype=float))
    proba = bundle["model"].predict_proba(X)[:, 1]
    return {"proba": proba.tolist(), "pred": (proba > 0.5).astype(int).tolist()}
