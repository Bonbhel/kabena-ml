# Étendre kabena à une nouvelle famille d'algorithmes

Le cœur actuel couvre la famille `gradient` (tout apprentissage par descente
de gradient). Les quais `neighbors` (KNN), `trees`, `ensembles` sont réservés.

## Contrat d'une famille
Un module `kabena/families/<nom>/` expose :
```python
def select(scores, config: KabenaConfig, rng) -> tuple[np.ndarray, np.ndarray]:
    """scores : mesure d'utilité par observation propre à la famille
    (pertes pour gradient ; à définir pour KNN : ex. marge de voisinage).
    Retourne (active, weights) — le contrat de sortie NE CHANGE PAS."""
```

## Règles de la maison
1. **Compensation d'abord** : toute sélection corrélée au score doit être
   repondérée (leçon centrale du preprint, Proposition 2) ou justifier
   formellement son biais avec zone de validité mesurée.
2. **Validation avant fusion** : reproduire l'équivalent des Tables 1-2 du
   preprint sur la famille visée. Rappel : le boosting mesuré est NEUTRE —
   l'extension arbres/ensembles devra démontrer un mécanisme de gain réel.
3. **Trois paramètres maximum** exposés à l'utilisateur final.
