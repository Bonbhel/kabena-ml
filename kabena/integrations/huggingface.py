"""Intégration Hugging Face Transformers — 2 lignes via un Trainer surchargé.

    trainer = KabenaTrainer(model=model, args=args, train_dataset=ds, ...)  # ligne 1
    trainer.train()                                                          # ligne 2

`compute_loss` calcule la perte par échantillon (reduction='none') et la
réduit via KabenaTorch.reduce — masque + poids HT à chaque batch.
"""
from __future__ import annotations
from .torch import KabenaTorch

__all__ = ["KabenaTrainer"]

try:
    from transformers import Trainer as _Trainer
except Exception:                             # pragma: no cover - env sans HF
    _Trainer = object


class KabenaTrainer(_Trainer):
    def __init__(self, *args, kabena_N: float = 0.3, kabena_strategy: str = "auto",
                 kabena_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._kb = KabenaTorch(N=kabena_N, strategy=kabena_strategy, seed=kabena_seed)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        import torch
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        loss = self._kb.reduce(losses, y=labels.view(-1))
        return (loss, outputs) if return_outputs else loss
