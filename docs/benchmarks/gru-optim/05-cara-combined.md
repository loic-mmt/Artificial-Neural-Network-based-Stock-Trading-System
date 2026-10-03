# 05. CARA et loss combinée PnL/Sharpe

Les **36/36 entraînements** ont réussi : 30 dans le run initial, puis six
terminés séparément après interruption. Chaque candidat a trois seeds et trois
folds. Les données, dates et coûts correspondent au témoin Sharpe du run 04.

| Loss | Sharpe régularisé moyen ± écart-type | Écart apparié au témoin | Victoires Sharpe | Rendement net moyen | Drawdown moyen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sharpe, témoin | **0,7509 ± 0,0705** | référence | référence | 7,96 % | −4,16 % |
| Combinée, poids PnL 0,25 | 0,6918 ± 0,0988 | −0,0591 | 3/9 | **37,13 %** | −20,95 % |
| Combinée, poids PnL 0,50 | 0,6513 ± 0,0746 | −0,0996 | 1/9 | 52,04 % | −29,00 % |
| CARA, γ=50 | 0,1685 ± 0,6310 | −0,5824 | 2/9 | 2,19 % | −3,53 % |
| CARA, γ=250 | 0,0616 ± 0,6217 | −0,6893 | 2/9 | 0,66 % | −3,02 % |

La combinée 0,25 gagne en rendement **9/9** comparaisons, mais son drawdown
empire **9/9**. Par rapport au témoin, rendement moyen ×4,67, ampleur du
drawdown moyen ×5,04 ; pire drawdown observé **−33,77 %**, contre **−8,71 %**.
La meilleure décision dépend donc de la contrainte de risque choisie à l'avance.
Pour les prochains contrôles : Sharpe reste la référence prudente et la combinée
0,25 reste un candidat offensif. Ne pas choisir le vainqueur sur le holdout final
en changeant le critère après coup.

Sources : [30 premiers folds](../../../artifacts/comparisons/gru-optim/05-cara-combined-60-atr/folds.json),
[six folds restants](../../../artifacts/comparisons/gru-optim/05-cara-combined-remaining-60-atr/report.json).

Protocole commun : [règles de lecture](README.md).
