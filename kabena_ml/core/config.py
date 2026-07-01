"""
kabena_ml.core.config
=====================
Configuration centralisée K-ABENA — lecture depuis dict, YAML ou JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass
class KabenaConfig:
    """
    Configuration K-ABENA.

    Paramètres
    ----------
    K          : float  — seuil absolu (erreurs <= K = mineures)
    N          : float  — proportion de mineures conservées [0,1]
    task       : str    — "regression" | "classification"
    min_active : int    — protection m >= min_active
    stratified : bool   — K-ABENA Stratifié (données déséquilibrées)
    verbose    : bool   — afficher stats par époque

    Exemples
    --------
    >>> cfg = KabenaConfig(K=0.15, N=0.3, task="classification")
    >>> cfg = KabenaConfig.from_yaml("kabena.yaml")
    >>> cfg = KabenaConfig.from_dict({"K": 0.20, "N": 0.0})
    """

    K: float = 0.15
    N: float = 0.0
    task: Literal["regression", "classification"] = "classification"
    min_active: int = 1
    stratified: bool = False
    verbose: bool = False

    def __post_init__(self):
        assert 0.0 <= self.N <= 1.0, f"N doit être dans [0, 1], reçu : {self.N}"
        assert self.K > 0, f"K doit être > 0, reçu : {self.K}"
        assert self.min_active >= 1, "min_active doit être >= 1"
        assert self.task in (
            "regression",
            "classification",
        ), f"task doit être 'regression' ou 'classification', reçu : {self.task!r}"

    # ── Constructeurs alternatifs ─────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KabenaConfig":
        """Charge la configuration depuis un fichier YAML."""
        import yaml  # optionnel

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "KabenaConfig":
        """Charge la configuration depuis un fichier JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> "KabenaConfig":
        """Crée une config depuis un dictionnaire (clés inconnues ignorées)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    # ── Export ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return (
            f"KabenaConfig(K={self.K}, N={self.N}, task={self.task!r}, "
            f"stratified={self.stratified}, verbose={self.verbose})"
        )
