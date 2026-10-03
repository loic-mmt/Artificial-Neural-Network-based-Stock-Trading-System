# 04. Normalisation causale par fenêtre

Six candidats, avec le même GRU `attention + Linear` et la même loss Sharpe :
**54/54 entraînements réussis**. Chaque candidat garde d'abord le
`SequenceStandardizer` appris sur le train. Les variantes additionnelles
agissent sur la fenêtre d'entrée observée de 60 pas ; `revin_side` redonne au
classifieur la moyenne et l'échelle de cette fenêtre.

| Variante additionnelle | Sharpe moyen | Écart-type | Minimum | Victoires appariées face au témoin |
| --- | ---: | ---: | ---: | ---: |
| **Aucune, N0** | **0,7509** | **0,0705** | **0,6554** | référence |
| GAS inspiré, taux fixe 0,1 | 0,6978 | 0,1205 | 0,4714 | 3/9 |
| RevIN + statistiques | 0,5966 | 0,1078 | 0,4255 | 0/9 |
| RevIN | 0,5143 | 0,2520 | −0,0503 | 2/9 |
| Moyenne/variance glissantes, 20 pas | 0,4563 | 0,2366 | 0,0063 | 1/9 |
| Moyenne/variance cumulatives | 0,4131 | 0,3000 | −0,0938 | 1/9 |

GAS est le plus proche, mais perd 6/9 comparaisons. Son exposition absolue
moyenne tombe de **0,1403** à **0,0358** et son rendement net moyen de **7,96 %**
à **2,27 %**. `revin_side` atteint **20,15 %** de rendement net moyen, mais son
drawdown moyen passe de **−4,16 %** à **−12,29 %** et son Sharpe baisse. Ces
rendements, issus de folds à périodes et historiques différents, servent au
diagnostic et non à estimer une performance future unique.

Limite décisive : les variantes ont normalisé **toutes** les features retenues,
y compris variables calendaires et indicateurs de secteur. Leurs transformations
peuvent en effacer la signification. Les filtres cumulatif, glissant et GAS
redémarrent aussi à chaque fenêtre ; GAS n'est pas ici une reproduction apprise
avec état persistant par actif. Le résultat rejette cette **implémentation
globale par fenêtre**, pas toutes les normalisations adaptatives. Un nouveau
test exigerait une sélection de features par **nom après sélection dans chaque
fold**, puis des ablations `niveaux`, `continues`, `toutes`. Les indices fixes
de colonnes seraient fragiles puisque les features sélectionnées peuvent varier
entre folds.

Source : [rapport 04](../../../artifacts/comparisons/gru-optim/04-causal-normalization-60-atr/report.json),
[configuration des six candidats](../../../configs/benchmark/gru_normalization_ladder.json).

Protocole commun : [règles de lecture](README.md).
