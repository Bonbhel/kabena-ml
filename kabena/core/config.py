"""Configuration K-ABENA — un seul objet, trois paramètres utilisateur.

Philosophie : le ML est déjà sur-hyperparamétré. kabena n'expose que
N (budget de rétention), strategy et seed. K est calibré automatiquement
(percentile des pertes courantes) ; alpha est un défaut sûr documenté.
Les avancés restent accessibles mais jamais requis.
"""
from dataclasses import dataclass

@dataclass
class KabenaConfig:
    # --- Les 3 paramètres utilisateur ---
    N: float = 0.3            # proportion de mineures conservées (preprint : optimum multi-domaines)
    strategy: str = "auto"    # "auto" (= v3 + garde-fou) | "v3" | "v2" | "v1"
    seed: int | None = None   # reproductibilité du tirage (Limitation L8 du preprint)
    # --- Avancés (défauts issus du preprint, à ne toucher qu'en connaissance de cause) ---
    k_percentile: float = 40.0   # calibration de K à chaque appel (Section 7 du preprint)
    alpha: float = 0.3           # mélange défensif v3 (Lemma 1 : 1/pi <= k/(alpha*m))
    min_active: int = 1          # plancher de sécurité sur |S*|

    def validate(self) -> None:
        if not (0.0 <= self.N < 1.0):
            raise ValueError(f"N doit être dans [0,1), reçu {self.N}")
        if self.strategy not in ("auto", "v3", "v2", "v1"):
            raise ValueError(f"strategy inconnue: {self.strategy!r} (choix: auto/v3/v2/v1)")
        if self.strategy in ("auto", "v2") and self.strategy == "v2" and self.N > 0.5:
            raise ValueError("strategy='v2' impose N <= 0.5 (tirage restreint à la moitié basse). "
                             "Utiliser strategy='v3' (ou 'auto') pour N > 0.5.")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha doit être dans (0,1], reçu {self.alpha}")
        if not (0.0 < self.k_percentile < 100.0):
            raise ValueError(f"k_percentile doit être dans (0,100), reçu {self.k_percentile}")
