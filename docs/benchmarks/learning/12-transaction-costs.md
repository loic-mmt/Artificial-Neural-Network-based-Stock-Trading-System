# 12 - Transaction-cost stress

**But.** Tester le classement des losses sous 0, 5, 10 et 20 bps.

| Coût | Loss gagnante | Rendement | Sharpe régularisé | Turnover |
| ---: | --- | ---: | ---: | ---: |
| 0 bps | Sharpe | +7,98 % | **0,710** | 34,1 |
| 5 bps | Sharpe | +8,55 % | **0,634** | 33,6 |
| 10 bps | PnL | **+47,78 %** | 0,634 | 22,1 |
| 20 bps | PnL | **+50,23 %** | 0,647 | 11,6 |

**Décision.** Conserver Sharpe à l'hypothèse centrale de 5 bps. Les modèles sont
réentraînés pour chaque coût : la hausse du rendement PnL avec les coûts provient
d'une position plus persistante, pas d'un bénéfice mécanique des frais.

**Sources.** `artifacts/comparisons/ohlc-clean/12-cost-*bps`; point 5 bps dans
`11-loss-objectives-v2`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [12-cost-0bps](../../../artifacts/comparisons/ohlc-clean/12-cost-0bps/report.json)
- [12-cost-10bps](../../../artifacts/comparisons/ohlc-clean/12-cost-10bps/report.json)
- [12-cost-20bps](../../../artifacts/comparisons/ohlc-clean/12-cost-20bps/report.json)
- [11-loss-objectives-v2](../../../artifacts/comparisons/ohlc-clean/11-loss-objectives-v2/report.json)
