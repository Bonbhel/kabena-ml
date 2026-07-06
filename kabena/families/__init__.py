"""Familles d'algorithmes couvertes par kabena.

Aujourd'hui : `gradient` (toute méthode entraînée par descente de gradient :
linéaire, logistique, SVM, softmax, MLP/ANN, CNN, Transformer).

Extension prévue (quais réservés, non implémentés — voir docs/EXTENDING.md) :
`neighbors` (KNN), `trees` (arbres de décision), `ensembles` (forêts, boosting).
Le contrat d'une famille : exposer select(scores, config) -> (active, weights),
où `scores` est la mesure d'utilité propre à la famille (pertes pour gradient).
Note de périmètre (preprint, Table 1) : le boosting mesuré est NEUTRE sous
K-ABENA — toute extension devra apporter sa propre validation empirique.
"""
