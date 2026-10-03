# 2 - Triple Barrier volatility estimator

**But.** Comparer `rolling_std`, ATR et largeur de Bollinger.

Chaque cellule contient `macro-F1 / rendement historique`.

| Estimateur | GRU | RNN | Transformer |
| --- | ---: | ---: | ---: |
| **rolling_std** | **0,530 / −6,39 %** | 0,502 / −6,00 % | 0,491 / −10,08 % |
| ATR | **0,473 / +3,92 %** | 0,446 / −8,98 % | 0,381 / −26,90 % |
| Bollinger | **0,437 / −6,40 %** | 0,414 / −11,41 % | 0,405 / −14,36 % |

**Décision.** `rolling_std` retenu par macro-F1 et par la sélection agrégée du
lanceur. ATR reste un ancien challenger financier, sans avantage robuste.

**Sources.** `artifacts/comparisons/ohlc-clean/02-estimator-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [02-estimator-atr](../../../artifacts/comparisons/ohlc-clean/02-estimator-atr/report.json)
- [02-estimator-bollinger](../../../artifacts/comparisons/ohlc-clean/02-estimator-bollinger/report.json)
- [02-estimator-rolling_std](../../../artifacts/comparisons/ohlc-clean/02-estimator-rolling_std/report.json)
