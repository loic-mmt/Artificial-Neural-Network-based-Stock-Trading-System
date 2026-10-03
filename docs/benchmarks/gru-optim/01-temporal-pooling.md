# 01. Pooling temporel et longueur de contexte

Douze sous-benchmarks : contextes `10, 20, 40, 60` croisés avec
`rolling_std`, `bollinger`, `atr`. Chacun compare `last`, `mean`, `flatten`,
`attention` et `last_attention`, sous losses PnL et Sharpe. Protocole : 3 seeds,
3 folds, 90 entraînements par sous-benchmark, **1 080/1 080 réussis**.

Meilleur candidat de chaque sous-benchmark selon le Sharpe régularisé moyen :

| Contexte | Rolling std | Bollinger | ATR |
| ---: | --- | --- | --- |
| 10 | Sharpe + `last` : 0,6694 | Sharpe + `last` : 0,6834 | Sharpe + `last` : 0,6826 |
| 20 | Sharpe + `attention` : 0,6813 | Sharpe + `last` : 0,6881 | Sharpe + `attention` : 0,6890 |
| 40 | Sharpe + `attention` : 0,7172 | Sharpe + `last` : 0,6965 | Sharpe + `attention` : 0,7188 |
| 60 | Sharpe + `attention` : 0,7441 | Sharpe + `attention` : 0,7408 | **Sharpe + `attention` : 0,7528** |

Dans le sous-benchmark 60/ATR, comparaison des agrégations sous la **même loss
Sharpe** :

| Pooling | Moyenne | Écart-type | Minimum |
| --- | ---: | ---: | ---: |
| `last` | 0,6620 | 0,1742 | 0,3214 |
| `mean` | 0,7185 | 0,0681 | 0,5844 |
| `flatten` | 0,6469 | 0,1166 | 0,4804 |
| **`attention`** | **0,7528** | 0,0806 | **0,6620** |
| `last_attention` | 0,7010 | 0,0941 | 0,5681 |

Interprétation : l'attention seule domine ici la fusion `last_attention` et le
dernier état, surtout au contexte 60. À contexte 10, le dernier état gagne les
trois estimateurs : l'effet de l'agrégation dépend donc du contexte. Ce run
était une exploration de configurations, pas une preuve que l'ATR est
universellement supérieur aux deux autres estimateurs.

Sources : [60/ATR](../../../artifacts/comparisons/gru-optim/01-temporal-pooling/context-60-atr/report.json)
et les douze `report.json` dans
[`01-temporal-pooling`](../../../artifacts/comparisons/gru-optim/01-temporal-pooling/).

Protocole commun : [règles de lecture](README.md).
