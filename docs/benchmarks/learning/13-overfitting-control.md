# 13 - Overfitting-control profile

**But.** Comparer profil standard et contrôle opt-in sur GRU + loss Sharpe.

| Profil | Rendement | Sharpe | Drawdown | Turnover | σ Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | **+8,55 %** | 0,634 | −5,87 % | 33,57 | 0,246 |
| **On** | +7,63 % | **0,659** | **−4,32 %** | **24,65** | **0,190** |

**Décision.** Activer `--overfitting-control`. Rendement légèrement inférieur,
mais meilleur Sharpe, drawdown, turnover et pire cas. Le profil réduit 95 features
à 32 et applique GRU 32, weight decay `1e-4`, patience 15.

**Sources.** `artifacts/comparisons/ohlc-clean/13-overfitting-{off,on}`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [13-overfitting-off](../../../artifacts/comparisons/ohlc-clean/13-overfitting-off/report.json)
- [13-overfitting-on](../../../artifacts/comparisons/ohlc-clean/13-overfitting-on/report.json)
