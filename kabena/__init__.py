"""kabena — K-ABENA : exclusion compensée d'échantillons par perte (preprint 2026).

Intégration 2 lignes :
    from kabena import Kabena
    kb = Kabena()
    active, w = kb.select(losses)      # dans la boucle d'entraînement
"""
from .core import Kabena, kabena_filter, kabena_safe, KabenaConfig  # noqa: F401
__version__ = "2.1.0"
