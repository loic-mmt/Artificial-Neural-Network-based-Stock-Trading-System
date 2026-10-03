# 4 - Triple Barrier event filter

**But.** Comparer événements sur toutes les lignes et événements CUSUM.

| Filtre | Modèle | Macro-F1 | Rendement historique |
| --- | --- | ---: | ---: |
| `all` | GRU | 0,188 | −28,98 % |
| `all` | RNN | 0,187 | −22,82 % |
| `all` | Transformer | **0,197** | **−19,39 %** |
| **CUSUM** | **GRU** | **0,529** | **+6,48 %** |
| **CUSUM** | RNN | 0,500 | −8,13 % |
| **CUSUM** | Transformer | 0,501 | −16,23 % |

**Décision.** CUSUM `0.5`. Le run CUSUM reproduit la configuration gagnante du
benchmark 3 ; il ne constitue pas une réplication indépendante.

**Sources.** `artifacts/comparisons/ohlc-clean/04-*-event-filter-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [04-macro-f1-event-filter-all](../../../artifacts/comparisons/ohlc-clean/04-macro-f1-event-filter-all/report.json)
- [04-macro-f1-event-filter-cusum](../../../artifacts/comparisons/ohlc-clean/04-macro-f1-event-filter-cusum/report.json)
- [04-pnl-event-filter-all](../../../artifacts/comparisons/ohlc-clean/04-pnl-event-filter-all/report.json)
- [04-pnl-event-filter-cusum](../../../artifacts/comparisons/ohlc-clean/04-pnl-event-filter-cusum/report.json)
