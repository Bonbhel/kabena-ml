"""Serving du modèle fine-tuné (pipeline HF standard — kabena absent au runtime)."""
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

clf = pipeline("text-classification", model="artefacts/model")
app = FastAPI(title="kabena DistilBERT")

class Payload(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(p: Payload):
    return {"results": clf(p.texts)}
