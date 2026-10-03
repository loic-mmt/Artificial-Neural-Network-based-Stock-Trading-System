# 3 - Triple Barrier profit/stop multiples

**But.** Tester cinq couples de barrières avec `rolling_std` et CUSUM.

| Profit / stop | Macro-F1 moyen, tous modèles | P&L moyen |
| --- | ---: | ---: |
| **0.75 / 0.75** | **0,510** | **−595,94** |
| 0.75 / 1.00 | 0,504 | −1 139,08 |
| 1.00 / 0.75 | 0,508 | −759,98 |
| 1.00 / 1.00 | 0,508 | −749,05 |
| 1.50 / 1.50 | 0,456 | −760,97 |

**Décision.** `0.75 / 0.75`. Les branches nommées `macro-f1` et `pnl` sont des
runs cross-entropy identiques ; seul leur critère de sélection différait.

GRU domine les deux autres architectures pour chacun des cinq couples :

| Profit / stop | GRU F1 | RNN F1 | Transformer F1 |
| --- | ---: | ---: | ---: |
| **0.75 / 0.75** | **0,529** | 0,500 | 0,501 |
| 0.75 / 1.00 | **0,523** | 0,504 | 0,485 |
| 1.00 / 0.75 | **0,529** | 0,508 | 0,488 |
| 1.00 / 1.00 | **0,530** | 0,502 | 0,491 |
| 1.50 / 1.50 | **0,491** | 0,453 | 0,423 |

**Sources.** `artifacts/comparisons/ohlc-clean/03-*-barriers-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [03-macro-f1-barriers-0.75-0.75](../../../artifacts/comparisons/ohlc-clean/03-macro-f1-barriers-0.75-0.75/report.json)
- [03-macro-f1-barriers-0.75-1.0](../../../artifacts/comparisons/ohlc-clean/03-macro-f1-barriers-0.75-1.0/report.json)
- [03-macro-f1-barriers-1.0-0.75](../../../artifacts/comparisons/ohlc-clean/03-macro-f1-barriers-1.0-0.75/report.json)
- [03-macro-f1-barriers-1.0-1.0](../../../artifacts/comparisons/ohlc-clean/03-macro-f1-barriers-1.0-1.0/report.json)
- [03-macro-f1-barriers-1.5-1.5](../../../artifacts/comparisons/ohlc-clean/03-macro-f1-barriers-1.5-1.5/report.json)
- [03-pnl-barriers-0.75-0.75](../../../artifacts/comparisons/ohlc-clean/03-pnl-barriers-0.75-0.75/report.json)
- [03-pnl-barriers-0.75-1.0](../../../artifacts/comparisons/ohlc-clean/03-pnl-barriers-0.75-1.0/report.json)
- [03-pnl-barriers-1.0-0.75](../../../artifacts/comparisons/ohlc-clean/03-pnl-barriers-1.0-0.75/report.json)
- [03-pnl-barriers-1.0-1.0](../../../artifacts/comparisons/ohlc-clean/03-pnl-barriers-1.0-1.0/report.json)
- [03-pnl-barriers-1.5-1.5](../../../artifacts/comparisons/ohlc-clean/03-pnl-barriers-1.5-1.5/report.json)
