# US : interaction entre nombre de features et gate marché

Statut au 3 octobre 2026 : protocole prêt, pas de résultat trouvé dans `artifacts/comparisons/us-relational-market/04-feature-gate-interaction`. Je ne classe donc aucun candidat comme gagnant de ce test.

Je compare caps 32/64 pour GRU, GNN résiduel et GNN identité sans relations, chacun avec et sans gate Transformer. La grille contient six candidats par cap, dont `identity` et `identity_market` : douze variantes, 3 folds et 3 seeds, soit 108 entraînements au maximum. L'ancien budget de 90 entraînements correspondait à la grille historique sans `identity_market`.

Le pool commun est `technical,market,sector`. Le run historique utilisait effectivement `technical,sector` : les baselines doivent être réentraînées. Hidden size reste 32. Le test mesure le nombre de features d'entrée, pas une hausse simultanée de capacité cachée.

Je lis l'interaction et les effets simples, en brut puis à exposition comparable. Une interaction positive ne prouve pas que gate64 bat plain64. Je compare `identity_market` à `identity` au même cap, fold et seed pour contrôler l'effet du gate et de sa capacité sans relations. Je compare ensuite le GNN résiduel à ces contrôles appariés, avec et sans gate. Leur présence ne démontre pas un alpha relationnel : les résultats restent en attente et je n'ai lancé aucun entraînement pour cet ajout.

Le [plan et les commandes](../plan/us-feature-gate-interaction.md) conservent tous les paramètres, audits, limites de provenance et modalités de reprise. Les résultats précédents sont dans [la comparaison d'exposition](us-exposure-comparison.md).
