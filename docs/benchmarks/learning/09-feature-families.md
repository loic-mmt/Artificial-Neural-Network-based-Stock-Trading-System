# 9 - Feature-family ablation

**But.** Mesurer valeur marginale des features techniques, marché et secteur.

**Portée.** GRU uniquement.

| Features, GRU | Macro-F1 | Rendement historique | Runs positifs |
| --- | ---: | ---: | ---: |
| techniques | 0,527 | −8,42 % | 1/9 |
| techniques + marché | 0,549 | **+13,74 %** | **5/9** |
| techniques + secteur | 0,518 | −9,63 % | 1/9 |
| **techniques + marché + secteur** | **0,556** | +13,25 % | 4/9 |

**Décision.** Combinaison complète selon le score primaire macro-F1. La variante
marché seule était légèrement meilleure financièrement et restait challenger.

**Sources.** `artifacts/comparisons/ohlc-clean/09-features-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [09-features-technical](../../../artifacts/comparisons/ohlc-clean/09-features-technical/report.json)
- [09-features-technical-market](../../../artifacts/comparisons/ohlc-clean/09-features-technical-market/report.json)
- [09-features-technical-market-sector](../../../artifacts/comparisons/ohlc-clean/09-features-technical-market-sector/report.json)
- [09-features-technical-sector](../../../artifacts/comparisons/ohlc-clean/09-features-technical-sector/report.json)
