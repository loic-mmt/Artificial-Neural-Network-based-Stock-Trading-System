# 11 - Loss objective

**But.** Comparer cross-entropy, PnL net direct et Sharpe régularisé avec coûts
effectifs de 5 bps.

| Loss | Rendement net | Sharpe régularisé | Drawdown | Turnover | Runs + |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-entropy | −2,46 % | −0,251 | −9,36 % | 156,1 | 3/9 |
| PnL | **+45,39 %** | 0,617 | −27,40 % | 36,2 | 9/9 |
| **Sharpe** | +8,55 % | **0,634** | **−5,87 %** | **33,6** | 9/9 |

**Décision.** Loss Sharpe à 5 bps : meilleur compromis risque/rendement. PnL
reste challenger à forte exposition.

**Source.** [Rapport corrigé](../../../artifacts/comparisons/ohlc-clean/11-loss-objectives-v2/report.json).
Le premier dossier `11-loss-objectives` contient le run échoué avant alignement
des calendriers et ne doit pas être utilisé.

Protocole commun : [conventions et limites](README.md).
