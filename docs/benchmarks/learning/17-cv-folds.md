# 17 - Purged-CV fold count

**But.** Comparer 3, 5 puis 10 folds, ce dernier ajouté comme diagnostic.

| Folds | Sharpe moyen | σ Sharpe | Pire Sharpe | Runs + | Rendement quotidien moyen |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **3** | 0,685 | **0,139** | **0,390** | 9/9 | 0,00934 % |
| 5 | 0,712 | 0,285 | 0,143 | 15/15 | 0,00891 % |
| 10 | **0,830** | 0,786 | −0,430 | 25/30 | 0,00995 % |

**Décision.** 3 folds pour la sélection principale : fenêtres longues et score
stable. 5 et 10 folds restent des diagnostics de régimes. Le score moyen élevé à
10 folds est gonflé par certaines courtes périodes très favorables.

**Sources.** `artifacts/comparisons/ohlc-clean/17-cv-folds-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [17-cv-folds-10](../../../artifacts/comparisons/ohlc-clean/17-cv-folds-10/report.json)
- [17-cv-folds-3](../../../artifacts/comparisons/ohlc-clean/17-cv-folds-3/report.json)
- [17-cv-folds-5](../../../artifacts/comparisons/ohlc-clean/17-cv-folds-5/report.json)
