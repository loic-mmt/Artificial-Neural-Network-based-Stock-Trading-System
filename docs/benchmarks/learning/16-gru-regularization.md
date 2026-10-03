# 16 - GRU regularization

**But.** Comparer profondeur, dropout, weight decay et patience du GRU.

| GRU | Rendement | Sharpe | σ Sharpe | Pire Sharpe | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| **1 couche, WD `1e-5`** | **+9,13 %** | **0,685** | 0,139 | 0,390 | 24,03 |
| 2 couches, dropout 0,3, WD `1e-4` | +7,36 % | 0,674 | **0,123** | 0,477 | **11,39** |
| 2 couches, dropout 0,5, WD `1e-3` | +8,42 % | 0,682 | 0,123 | **0,483** | 12,84 |

**Décision.** Le score primaire sélectionne 1 couche et weight decay `1e-5`.
La variante fortement régularisée reste challenger robuste : Sharpe presque
identique et turnover divisé par environ 1,9.

**Source.** [Rapport](../../../artifacts/comparisons/ohlc-clean/16-gru-regularization/report.json).

Protocole commun : [conventions et limites](README.md).
