# 11. Décision après l'ouverture

J'ai comparé `lagged`, features jusqu'à J-1, à `open_gap`, mêmes features plus le gap open(J)/close(J-1). Dix actions, 3 folds, seeds 1/7/19, objectif Sharpe et coûts 5 bps : 18 résultats sauvegardés.

| Variante | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen | Exposition moyenne | Rendement sous stress défavorable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sans gap | 0,7526 | +8,35 % | -4,16 % | 13,47 % | +5,83 % |
| Avec gap | 0,7879 | +7,67 % | -3,82 % | 12,39 % | +5,09 % |

Le gap gagne en Sharpe et en rendement dans 5 paires sur 9, mais son rendement moyen reste inférieur. Le Sharpe moyen du stress défavorable passe de 0,4405 à 0,5027. Je garde `open_gap` comme candidat pour les essais suivants, sans en faire une preuve de profit exécutable.

L'open ajusté J sert de proxy vers J+1. Observer l'open avant de décider ne garantit pas un fill à ce même prix. Les extrêmes high/low et les 250 tirages uniformes par paire sont des sensibilités mathématiques, pas des distributions de fills réels ni des intervalles de confiance. Le holdout CAC40 à partir du 10 mai 2022 reste fermé.

Sources : [résultats](../../../artifacts/comparisons/gru-optim/11-post-open-pilot/results.json), [métadonnées](../../../artifacts/comparisons/gru-optim/11-post-open-pilot/metadata.json), [fonctionnement et commande](../../src/post-open-pilot.md).
