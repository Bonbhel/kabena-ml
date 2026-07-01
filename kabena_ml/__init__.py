"""
kabena-ml — K-ABENA: K-Adaptive Backpropagation with Error-based N-exclusion Algorithm
YekoElite University × NeuroSoft IA — contact@neurosoft-ia.com

Philosophie : coût d'adoption minimal — 2 lignes suffisent.

    from kabena_ml import kabena_filter
    mask = kabena_filter(losses, K=0.25)
    losses[mask].mean().backward()
"""

from kabena_ml.core.config import KabenaConfig as KabenaConfig
from kabena_ml.core.filter import (
    calibrate_K as calibrate_K,
)
from kabena_ml.core.filter import (
    calibrate_K_noisy as calibrate_K_noisy,
)
from kabena_ml.core.filter import (
    kabena_filter as kabena_filter,
)
from kabena_ml.core.filter import (
    kabena_safe as kabena_safe,
)

# from kabena_ml.core.config import KabenaConfig
# from kabena_ml.core.filter import calibrate_K, calibrate_K_noisy, kabena_filter, kabena_safe

__version__ = "1.1.0"
__author__ = "M. Bonbhel — YekoElite University / NeuroSoft IA"
__email__ = "contact@neurosoft-ia.com"

__all__ = ["kabena_filter", "kabena_safe", "calibrate_K", "calibrate_K_noisy", "KabenaConfig"]
