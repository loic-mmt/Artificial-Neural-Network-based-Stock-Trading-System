# MT5 US : fréquence de réentraînement

J'ai comparé toutes les 5, 10, 20, 40 et 60 séances. Même GRU `open_gap`, loss combinée 0,25, seed 42, décodage confiance réglé sur validation passée, coût 5 bps. 196 actions, du 13 mai au 25 septembre 2026, 93 rendements. Les cinq runs sont terminés, sans échec enregistré.

| Toutes les N séances | Réentraînements | PnL net | Sharpe régularisé | Drawdown | Exposition moyenne | FLAT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 19 | +254,49 | 2,5167 | -0,82 % | 20,56 % | 79,66 % |
| 10 | 10 | +299,44 | 2,8682 | -0,79 % | 19,50 % | 80,71 % |
| 20 | 5 | +326,41 | 2,6561 | -2,13 % | 18,33 % | 81,86 % |
| 40 | 3 | +478,60 | 2,9851 | -2,06 % | 34,84 % | 65,53 % |
| 60 | 2 | +380,09 | 2,3300 | -2,06 % | 15,66 % | 84,51 % |

Buy-and-hold : +1 126,33, soit +11,26 % pour 10 000 de capital, à 100 % d'exposition. Aucun modèle de la grille ne le dépasse en PnL brut.

40 a le meilleur PnL et Sharpe régularisé observés, mais engage presque deux fois l'exposition de 10 et 20. 10 a le plus faible drawdown. 60 ne confirme pas un avantage systématique des fréquences longues : son Sharpe baisse malgré un PnL supérieur à 10 et 20.

Je ne divise pas simplement les PnL par l'exposition pour déclarer un gagnant. L'exposition varie dans le temps, les shorts diffèrent et la composition des rendements compte. Les rapports ne sauvegardent pas ici une comparaison complète à risque ou gross identique. Celle de l'étude GNN US est un autre protocole.

Je garde 10 comme contrôle de faible drawdown et 40 comme challenger brut. Avant de fixer une fréquence de déploiement, il faut plusieurs périodes et seeds, des règles de décodage figées et un contrôle d'exposition recalculant les frais. Changer la fréquence modifie aussi les sélections/scalers et les seuils ajustés à chaque réentraînement.

Sources : [rapport complet](../../artifacts/mt5/benchmarks/retraining-frequency-confidence/report.json), [tableau agrégé](../../artifacts/mt5/benchmarks/retraining-frequency-confidence/summary.csv), [script](../../scripts/benchmark_walkforward_retraining.py).
