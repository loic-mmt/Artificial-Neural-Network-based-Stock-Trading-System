# 18 - Purged-CV gap

**But.** Tester gaps 0, 5 et 10 avec purging des intervalles d'information.

| Gap | Rendement | Sharpe | σ Sharpe | Pire Sharpe |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +8,56 % | **0,689** | 0,139 | 0,439 |
| **5** | **+9,13 %** | 0,685 | 0,139 | 0,390 |
| 10 | +9,01 % | 0,688 | **0,130** | **0,480** |

**Décision.** Gap 5 conservé comme compromis pré-déclaré. Le modèle est peu
sensible au gap parce que le purging Triple Barrier retire déjà les événements
qui chevauchent les frontières.

**Sources.** `artifacts/comparisons/ohlc-clean/18-cv-gap-*`; point gap 5 dans
`17-cv-folds-3`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [18-cv-gap-0](../../../artifacts/comparisons/ohlc-clean/18-cv-gap-0/report.json)
- [18-cv-gap-10](../../../artifacts/comparisons/ohlc-clean/18-cv-gap-10/report.json)
- [17-cv-folds-3](../../../artifacts/comparisons/ohlc-clean/17-cv-folds-3/report.json)
