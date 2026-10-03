# MT5 US : prédictions live et sélection des tickers

Le modèle enregistré utilise `open_gap`, contexte 60, seed 42 et loss combinée 0,25. Une prédiction pour J utilise les données terminées jusqu'à J-1 et la première ouverture régulière de J. Elle ne donne pas vingt jours de prévisions à l'avance : les entrées des prochains jours ne sont pas encore connues.

## Deux filtres différents dans les fichiers du 28 septembre 2026

| Fichier | Filtre historique | Tickers | LONG | SHORT |
| --- | --- | ---: | ---: | ---: |
| `predictions-2026-09-28.csv` | Sharpe positif du backtest signe ±1 | 106 | 20 | 86 |
| `predictions-2026-09-28-positive-backtest-sharpe.csv` | Intersection de ces 106 avec Sharpe positif du backtest continu | 85 | 13 | 72 |

La métadonnée du modèle pointe vers `sign_kpis_by_ticker.csv` pour la première liste. Le second fichier correspond au filtre de `kpis_by_ticker.csv`. Les 21 lignes retirées ne prouvent donc pas que leurs opens étaient manquants. Sur l'univers complet, 99 tickers ont un Sharpe continu positif et 106 un Sharpe signe positif ; les ensembles ne coïncident pas.

Les positions de signe des fichiers cités sont LONG ou SHORT, sans FLAT. Cela ne veut pas dire que l'argmax des probabilités n'aurait jamais choisi Hold. Le décodage détermine l'action finale.

## Limites

Filtrer un univers après lecture de son backtest améliore potentiellement les résultats par sélection a posteriori. Ces prédictions sont un essai de publication, pas un benchmark de performance live. Je n'ai pas de PnL MT5 réalisé audité dans ces fichiers. Il faut figer la liste, les règles et l'heure de décision avant la période forward, puis enregistrer les fills et les frais du broker.

Je ne complète pas un open absent par un prix pré-market ou une clôture. Le fichier d'une journée ne garantit pas la couverture des journées suivantes. Le modèle sauvegardé n'est pas réentraîné par un simple `--predict-only`.

Sources : [métadonnées](../../artifacts/mt5/stocks-us-gru-live-combined-025-seed42/metadata.json), [106 prédictions](../../artifacts/mt5/stocks-us-gru-live-combined-025-seed42/predictions-2026-09-28.csv), [85 après second filtre](../../artifacts/mt5/stocks-us-gru-live-combined-025-seed42/predictions-2026-09-28-positive-backtest-sharpe.csv), [explication du backtest US](mt5-us.md).
