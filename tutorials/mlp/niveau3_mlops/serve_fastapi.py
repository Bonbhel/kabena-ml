"""Serving TorchScript — kabena absent au runtime (entraînement seulement)."""
import numpy as np, torch
from fastapi import FastAPI
from pydantic import BaseModel

model = torch.jit.load("artefacts/model.pt"); model.eval()
_s = np.load("artefacts/scaler.npz")

app = FastAPI(title="kabena MLP")

class Payload(BaseModel):
    features: list[list[float]]

@app.post("/predict")
def predict(p: Payload):
    X = (np.asarray(p.features, float) - _s["mean"]) / _s["scale"]
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32)).argmax(1).tolist()
    return {"pred": out}
