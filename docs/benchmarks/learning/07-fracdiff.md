# 7 - FracDiff activation

**But.** Comparer données inchangées, ordre automatique et ordres fixes.

**Portée.** GRU uniquement, après sa sélection répétée aux benchmarks 1 à 6.

| Variante, GRU | Macro-F1 | Rendement historique |
| --- | ---: | ---: |
| **Sans FracDiff** | 0,529 | **+6,48 %** |
| ordre 0,3 | 0,526 | −15,72 % |
| ordre 0,5 | **0,535** | −1,78 % |
| ordre 0,7 | 0,532 | +1,68 % |
| automatique | 0,533 | −12,06 % |

**Décision.** FracDiff désactivé. Petit gain de F1 insuffisant face à la baisse
financière et à la complexité ajoutée.

**Sources.** `artifacts/comparisons/ohlc-clean/07-fracdiff-*`; contrôle sans
FracDiff dans `06-horizon-10`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [07-fracdiff-0.3](../../../artifacts/comparisons/ohlc-clean/07-fracdiff-0.3/report.json)
- [07-fracdiff-0.5](../../../artifacts/comparisons/ohlc-clean/07-fracdiff-0.5/report.json)
- [07-fracdiff-0.7](../../../artifacts/comparisons/ohlc-clean/07-fracdiff-0.7/report.json)
- [07-fracdiff-auto](../../../artifacts/comparisons/ohlc-clean/07-fracdiff-auto/report.json)
- [06-horizon-10](../../../artifacts/comparisons/ohlc-clean/06-horizon-10/report.json)
