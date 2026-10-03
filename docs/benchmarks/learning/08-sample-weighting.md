# 8 - Sample weighting

**But.** Tester pondérations par rendement net, volatilité et unicité.

**Portée.** GRU uniquement.

| Pondération, GRU | Macro-F1 | Rendement historique |
| --- | ---: | ---: |
| **Aucune** | 0,529 | **+6,48 %** |
| rendement net | 0,531 | −4,31 % |
| volatilité | **0,534** | −5,11 % |
| unicité | 0,512 | +1,32 % |

**Décision.** Aucune pondération. Les gains marginaux de F1 ne produisent pas de
gain financier robuste.

**Sources.** `artifacts/comparisons/ohlc-clean/08-weight-*`; contrôle non pondéré
dans `06-horizon-10`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [08-weight-net_return](../../../artifacts/comparisons/ohlc-clean/08-weight-net_return/report.json)
- [08-weight-uniqueness](../../../artifacts/comparisons/ohlc-clean/08-weight-uniqueness/report.json)
- [08-weight-volatility](../../../artifacts/comparisons/ohlc-clean/08-weight-volatility/report.json)
- [06-horizon-10](../../../artifacts/comparisons/ohlc-clean/06-horizon-10/report.json)
