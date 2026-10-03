# 14 - Feature-selection sensitivity

**But.** Tester limites de 16/32/64 features et corrélations 0,90/0,95/0,98.

| Configuration | Rendement | Sharpe | σ Sharpe | Pire Sharpe | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 / 0,98 | +8,82 % | **0,694** | 0,247 | 0,172 | 22,0 |
| **32 / 0,95** | +9,13 % | 0,685 | **0,139** | 0,390 | 24,0 |
| 64 / 0,90 | **+10,48 %** | 0,671 | 0,171 | **0,401** | 34,7 |

**Décision.** 32 features, corrélation maximale 0,95. Presque le meilleur Sharpe,
mais meilleure dispersion globale et complexité modérée. Chaque limite était
atteinte ; la sélection est bien contraignante.

**Sources.** `artifacts/comparisons/ohlc-clean/14-selector-*`; point 32/0,98 dans
`13-overfitting-on`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [14-selector-16-0.90](../../../artifacts/comparisons/ohlc-clean/14-selector-16-0.90/report.json)
- [14-selector-16-0.95](../../../artifacts/comparisons/ohlc-clean/14-selector-16-0.95/report.json)
- [14-selector-16-0.98](../../../artifacts/comparisons/ohlc-clean/14-selector-16-0.98/report.json)
- [14-selector-32-0.90](../../../artifacts/comparisons/ohlc-clean/14-selector-32-0.90/report.json)
- [14-selector-32-0.95](../../../artifacts/comparisons/ohlc-clean/14-selector-32-0.95/report.json)
- [14-selector-64-0.90](../../../artifacts/comparisons/ohlc-clean/14-selector-64-0.90/report.json)
- [14-selector-64-0.95](../../../artifacts/comparisons/ohlc-clean/14-selector-64-0.95/report.json)
- [14-selector-64-0.98](../../../artifacts/comparisons/ohlc-clean/14-selector-64-0.98/report.json)
- [13-overfitting-on](../../../artifacts/comparisons/ohlc-clean/13-overfitting-on/report.json)
