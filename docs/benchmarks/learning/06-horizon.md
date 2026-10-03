# 6 - Triple Barrier horizon

**But.** Comparer horizons 5, 10 et 20 observations.

| Horizon | Modèle | Macro-F1 | Rendement historique | Runs positifs |
| ---: | --- | ---: | ---: | ---: |
| 5 | **GRU** | **0,498** | −10,33 % | 3/9 |
| 5 | RNN | 0,472 | −20,51 % | 2/9 |
| 5 | Transformer | 0,446 | −13,22 % | 2/9 |
| **10** | **GRU** | **0,529** | **+6,48 %** | **6/9** |
| 10 | RNN | 0,500 | −8,13 % | 3/9 |
| 10 | Transformer | 0,501 | −16,23 % | 1/9 |
| 20 | **GRU** | **0,524** | −14,80 % | 3/9 |
| 20 | RNN | 0,503 | −16,18 % | 2/9 |
| 20 | Transformer | 0,493 | **+0,29 %** | **4/9** |

**Décision.** Horizon `10`.

**Sources.** `artifacts/comparisons/ohlc-clean/06-horizon-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [06-horizon-10](../../../artifacts/comparisons/ohlc-clean/06-horizon-10/report.json)
- [06-horizon-20](../../../artifacts/comparisons/ohlc-clean/06-horizon-20/report.json)
- [06-horizon-5](../../../artifacts/comparisons/ohlc-clean/06-horizon-5/report.json)
