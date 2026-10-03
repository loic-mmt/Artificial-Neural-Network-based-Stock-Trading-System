# Learning improvements : protocole et avancement

Dernière mise à jour : 2026-09-18.

Je conserve ici le protocole de la série et ses décisions historiques au
18 septembre 2026. La [synthèse générale](../../benchmark-summary.md) tient
compte des expériences suivantes. Les anciens plans et journaux restent
dans `docs/papers/old_docs/`, sans modification.

## Conventions et limites

- Sauf indication contraire, les valeurs sont des moyennes sur folds et seeds.
- Le holdout final commençant le 2022-05-10 est toujours fermé.
- Les folds ont des historiques d'entraînement qui se chevauchent : ils ne sont
  pas des réplications indépendantes.
- L'univers utilise des constituants actuels sur tout l'historique : les résultats
  restent exposés au biais de survivance.
- Le benchmark 1 utilise l'ancien parquet non nettoyé. À partir du benchmark 2,
  la référence est `data/processed/cac40_daily_clean.parquet`, SHA-256
  `dce05c66d89d76a16649389b2482139aee82dd791504163efb83f5f10700ae2c`.
- Jusqu'au benchmark 9, les 5 bps configurés appartiennent aux labels ; les
  backtests historiques affichés ne débitent pas de coûts de transaction.
- À partir du benchmark 11, les métriques continues débitent explicitement les
  coûts proportionnels déclarés. Elles ne sont donc pas directement comparables
  aux anciens backtests de classification.
- Le benchmark 1 compare cinq modèles. Les benchmarks 2 à 6 comparent GRU, RNN
  et Transformer. Les benchmarks 7 à 20 et 22 ont ensuite été restreints au GRU
  pour réduire le coût de calcul. Le benchmark 21 ayant été sauté, la
  configuration financière finale n'a pas été revalidée sur les autres
  architectures.

## Avancement

Les anciens dossiers `artifacts/comparisons/02-estimator-*`, hors
`ohlc-clean/`, ne sont pas les sources de cette série nettoyée. ATR y a
27 folds en erreur, alors que les deux autres estimateurs ont 27 folds
réussis chacun. Je conserve cette trace, mais ne mélange pas ces résultats
aux runs nettoyés. Le [benchmark 02](02-volatility-estimator.md) pointe vers
les trois rapports complets utilisés ensuite.

| Benchmark | Statut | Décision |
| --- | --- | --- |
| 1 | Terminé, historique | GRU meilleur classifieur de départ |
| 2 | Terminé | `rolling_std` |
| 3 | Terminé | barrières `0.75 / 0.75` |
| 4 | Terminé | filtre CUSUM `0.5` |
| 5 | Terminé | politique `hold` |
| 6 | Terminé | horizon `10` |
| 7 | Terminé | FracDiff désactivé |
| 8 | Terminé | sample weighting désactivé |
| 9 | Terminé | `technical,market,sector` selon score primaire |
| 10 | Sauté | données externes point-in-time absentes |
| 11 | Terminé | loss Sharpe à 5 bps |
| 12 | Terminé | Sharpe à 0–5 bps ; PnL à 10–20 bps |
| 13 | Terminé | contrôle du surapprentissage activé |
| 14 | Terminé | 32 features, corrélation maximale 0,95 |
| 15 | Terminé | règles de stabilité validées par tests |
| 16 | Terminé, GRU seul | 1 couche, weight decay `1e-5` selon score primaire |
| 17 | Terminé | 3 folds principaux ; 5/10 diagnostics |
| 18 | Terminé | gap `5` conservé |
| 19 | Terminé | embargo `0` |
| 20 | Terminé | initial `0.50`, validation interne `0.20` |
| 21 | Sauté | comparaison finale des architectures non relancée |
| 22 | Terminé | features techniques + marché + secteur |
| 23 | En attente | holdout final encore scellé |


## Fiches détaillées

- [1 - Reference baseline](01-baseline.md)
- [2 - Triple Barrier volatility estimator](02-volatility-estimator.md)
- [3 - Triple Barrier profit/stop multiples](03-profit-stop-barriers.md)
- [4 - Triple Barrier event filter](04-event-filter.md)
- [5 - Triple Barrier between-event policy](05-between-events.md)
- [6 - Triple Barrier horizon](06-horizon.md)
- [7 - FracDiff activation](07-fracdiff.md)
- [8 - Sample weighting](08-sample-weighting.md)
- [9 - Feature-family ablation](09-feature-families.md)
- [10 - Historical fundamentals and sentiment](10-external-features.md)
- [11 - Loss objective](11-loss-objectives.md)
- [12 - Transaction-cost stress](12-transaction-costs.md)
- [13 - Overfitting-control profile](13-overfitting-control.md)
- [14 - Feature-selection sensitivity](14-feature-selection.md)
- [15 - Stability-gate sensitivity](15-stability-gates.md)
- [16 - GRU regularization](16-gru-regularization.md)
- [17 - Purged-CV fold count](17-cv-folds.md)
- [18 - Purged-CV gap](18-cv-gap.md)
- [19 - Purged-CV embargo](19-cv-embargo.md)
- [20 - Purged-CV history fractions](20-cv-history.md)
- [21 - Architecture comparison](21-architectures.md)
- [22 - Market and sector context](22-market-sector.md)
- [23 - Final holdout](23-final-holdout.md)
