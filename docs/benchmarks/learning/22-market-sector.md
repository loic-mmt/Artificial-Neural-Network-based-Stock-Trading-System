# 22 - Market and sector context

**But.** Revalider l'apport des familles techniques, marché et secteur après
passage à la loss Sharpe et au contrôle du surapprentissage.

| Features | Rendement | Sharpe | σ Sharpe | Pire Sharpe | Runs + |
| --- | ---: | ---: | ---: | ---: | ---: |
| techniques | +3,72 % | 0,514 | 0,437 | −0,262 | 7/9 |
| techniques + marché | +8,71 % | 0,624 | 0,255 | 0,006 | 8/9 |
| techniques + secteur | +2,56 % | 0,407 | 0,474 | −0,370 | 6/9 |
| **techniques + marché + secteur** | **+9,13 %** | **0,685** | **0,139** | **0,390** | **9/9** |

**Décision.** Combinaison complète. Le marché apporte le gain principal. Le
secteur seul dégrade les résultats, mais améliore le couple techniques + marché,
probablement via interaction et remplacement de features sous le plafond de 32.
Ce benchmark teste du contexte, pas un modèle de régime HMM/VIX explicite.

**Sources.** `artifacts/comparisons/ohlc-clean/22-regime-*`; combinaison complète
dans `17-cv-folds-3`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [22-regime-technical](../../../artifacts/comparisons/ohlc-clean/22-regime-technical/report.json)
- [22-regime-technical-market](../../../artifacts/comparisons/ohlc-clean/22-regime-technical-market/report.json)
- [22-regime-technical-sector](../../../artifacts/comparisons/ohlc-clean/22-regime-technical-sector/report.json)
- [17-cv-folds-3](../../../artifacts/comparisons/ohlc-clean/17-cv-folds-3/report.json)
