# 03. Gate sur l'amplitude du signal Sharpe

Le GRU et ses positions brutes sont inchangés. Chaque fold choisit sur sa
validation interne un seuil parmi `0` et les quantiles `0,2 / 0,4 / 0,6 / 0,8`
de `|q|`, avec couverture minimale de 20 %, puis l'applique au fold externe.
**9/9 entraînements réussis**. Il s'agit de sélectivité financière, pas de
calibration des probabilités de classes.

| Politique | Sharpe moyen | Écart-type | Couverture des signaux |
| --- | ---: | ---: | ---: |
| Position brute | 0,7509 | 0,0705 | 100 % |
| Gate choisi dans le fold | 0,7567 | 0,0733 | 91,94 % |

L'écart moyen de **+0,0059** est trompeur : le gate gagne **1/9**, reste
identique **4/9** car le seuil choisi vaut zéro, et perd **4/9**. Sur les cinq
cas de filtrage effectif, le rendement net baisse et le turnover augmente.
L'unique gain marqué est seed 7, fold 0 : **0,7238 → 0,8096** en Sharpe
régularisé, malgré un rendement net inférieur. Décision : garder la position
brute ; ne pas ouvrir le holdout pour départager ce gate.

Source : [rapport 03](../../../artifacts/comparisons/gru-optim/03-sharpe-signal-gate-60-atr/report.json).

Protocole commun : [règles de lecture](README.md).
