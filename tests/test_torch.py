"""
tests/test_torch.py
===================
Tests d'intégration PyTorch — kabena_filter_torch, KabenaTrainer.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from kabena_ml import KabenaConfig
from kabena_ml.integrations.torch_utils import (
    kabena_filter_torch, kabena_safe_torch, KabenaTrainer
)


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = torch.randn(200, 10)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X, y


@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))


class TestKabenaFilterTorch:

    def test_returns_bool_tensor(self, simple_data):
        X, y = simple_data
        losses = F.cross_entropy(nn.Linear(10, 2)(X), y, reduction="none")
        mask   = kabena_filter_torch(losses, K=0.5, N=0.0)
        assert mask.dtype == torch.bool
        assert mask.shape == losses.shape

    def test_N1_all_true(self, simple_data):
        X, y = simple_data
        losses = torch.rand(len(y))
        mask   = kabena_filter_torch(losses, K=0.5, N=1.0)
        assert mask.all(), "N=1 → toutes les observations conservées"

    def test_device_preserved(self):
        losses = torch.rand(50)
        mask   = kabena_filter_torch(losses, K=0.3, N=0.0)
        assert mask.device == losses.device

    def test_gradient_flows(self, simple_model, simple_data):
        X, y = simple_data
        losses = F.cross_entropy(simple_model(X), y, reduction="none")
        mask   = kabena_filter_torch(losses, K=0.5, N=0.0)
        L_KA   = losses[mask].mean()
        L_KA.backward()
        for p in simple_model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestKabenaSafeTorch:

    def test_guarantees_min_active(self):
        losses = torch.full((10,), 0.01)  # toutes < K
        mask, m = kabena_safe_torch(losses, K=1.0, N=0.0, min_active=3)
        assert m >= 3

    def test_returns_int_m(self):
        losses = torch.rand(20)
        mask, m = kabena_safe_torch(losses, K=0.3, N=0.0)
        assert isinstance(m, int)
        assert m == mask.sum().item()


class TestKabenaTrainer:

    def test_fit_returns_history(self, simple_model, simple_data):
        X, y = simple_data
        cfg  = KabenaConfig(K=0.5, N=0.0, task="classification")
        t    = KabenaTrainer(simple_model, cfg, lr=0.05, epochs=10)
        h    = t.fit(X, y)
        assert isinstance(h, list)
        assert len(h) > 0
        assert "loss" in h[0] and "gain_pct" in h[0]

    def test_evaluate_returns_float(self, simple_model, simple_data):
        X, y = simple_data
        cfg  = KabenaConfig(K=0.5, N=0.0, task="classification")
        t    = KabenaTrainer(simple_model, cfg, lr=0.05, epochs=5)
        t.fit(X, y)
        acc = t.evaluate(X, y)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_N1_similar_to_sgd(self, simple_data):
        """N=1 → même comportement que SGD — pas d'erreur."""
        X, y = simple_data
        model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
        cfg   = KabenaConfig(K=0.5, N=1.0, task="classification")
        t     = KabenaTrainer(model, cfg, lr=0.05, epochs=10)
        h     = t.fit(X, y)
        gains = [r["gain_pct"] for r in h]
        assert max(gains) < 5, "N=1 → gain ≈ 0%"

    def test_gain_pct_in_range(self, simple_model, simple_data):
        X, y   = simple_data
        cfg    = KabenaConfig(K=0.3, N=0.0, task="classification")
        t      = KabenaTrainer(simple_model, cfg, lr=0.05, epochs=20)
        h      = t.fit(X, y)
        for r in h:
            assert 0 <= r["gain_pct"] <= 100

    def test_fit_loader(self, simple_data):
        from torch.utils.data import TensorDataset, DataLoader
        X, y   = simple_data
        loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
        model  = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
        cfg    = KabenaConfig(K=0.4, N=0.0, task="classification")
        t      = KabenaTrainer(model, cfg, lr=0.05, epochs=5)
        h      = t.fit_loader(loader)
        assert isinstance(h, list)
