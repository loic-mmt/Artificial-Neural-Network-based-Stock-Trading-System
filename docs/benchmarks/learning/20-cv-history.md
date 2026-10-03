# 20 - Purged-CV history fractions

**But.** Tester historique initial 0,40/0,50/0,60 et validation interne
0,15/0,20/0,25.

| Initial | Inner val | Sharpe | σ Sharpe | Pire Sharpe | Runs + |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,40 | 0,15 | 0,510 | 0,546 | −0,984 | 8/9 |
| 0,40 | 0,20 | 0,518 | 0,481 | −0,762 | 8/9 |
| 0,40 | 0,25 | 0,522 | 0,447 | −0,665 | 8/9 |
| 0,50 | 0,15 | 0,408 | 0,601 | −1,164 | 8/9 |
| **0,50** | **0,20** | **0,685** | **0,139** | **0,390** | **9/9** |
| 0,50 | 0,25 | 0,632 | 0,139 | 0,377 | 9/9 |
| 0,60 | 0,15 | 0,634 | 0,273 | 0,229 | 9/9 |
| 0,60 | 0,20 | 0,658 | 0,268 | 0,190 | 9/9 |
| 0,60 | 0,25 | 0,663 | 0,292 | 0,305 | 9/9 |

**Décision.** Initial 0,50 et validation interne 0,20. Meilleur Sharpe, faible
dispersion et historique test plus large que 0,60. Les scores entre fractions
initiales ne couvrent pas exactement les mêmes périodes ; 0,40 inclut notamment
une période plus difficile dès 2008.

**Sources.** `artifacts/comparisons/ohlc-clean/20-cv-history-*`; point 0,50/0,20
dans `17-cv-folds-3`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [20-cv-history-0.4-0.15](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.4-0.15/report.json)
- [20-cv-history-0.4-0.20](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.4-0.20/report.json)
- [20-cv-history-0.4-0.25](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.4-0.25/report.json)
- [20-cv-history-0.5-0.15](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.5-0.15/report.json)
- [20-cv-history-0.5-0.25](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.5-0.25/report.json)
- [20-cv-history-0.6-0.15](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.6-0.15/report.json)
- [20-cv-history-0.6-0.20](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.6-0.20/report.json)
- [20-cv-history-0.6-0.25](../../../artifacts/comparisons/ohlc-clean/20-cv-history-0.6-0.25/report.json)
- [17-cv-folds-3](../../../artifacts/comparisons/ohlc-clean/17-cv-folds-3/report.json)
