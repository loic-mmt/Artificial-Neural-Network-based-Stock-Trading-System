# 12. Relations entre actions après le GRU

J'ai testé cinq variantes sur les mêmes 3 folds et 3 seeds : 45 résultats. Le GRU encode chaque ticker séparément, puis un bloc optionnel mélange les représentations du même jour. Ce n'est pas le GNN indépendant.

| Variante | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen | Exposition moyenne | Rendement sous stress défavorable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aucun mélange | 0,7901 | +7,68 % | -3,82 % | 12,40 % | +5,10 % |
| Moyenne des autres | 0,7574 | +9,35 % | -4,48 % | 15,13 % | +7,08 % |
| StockMixer | 0,8162 | +7,61 % | -3,62 % | 13,63 % | +4,91 % |
| Attention sur soi | 0,7485 | +11,12 % | -5,88 % | 17,74 % | +7,76 % |
| Attention entre actions | 0,6939 | +9,97 % | -5,29 % | 16,16 % | +7,13 % |

StockMixer gagne en Sharpe dans 5/9 paires, mais en rendement dans 4/9. Son Sharpe sous stress défavorable est 0,4723 contre 0,5041 sans mélange. Le petit gain du proxy ne suffit pas à le promouvoir.

L'attention entre actions ne bat son contrôle sur soi qu'en 4/9 paires, en Sharpe comme en rendement. Les gains bruts de certaines variantes s'accompagnent de plus d'exposition. Je garde le GRU seul comme référence, StockMixer comme piste à recontrôler et pas comme amélioration validée.

Univers fixe de dix tickers, holdout fermé. StockMixer apprend par place de ticker : ce test ne prouve pas une généralisation à un autre univers. Les résultats du témoin sont proches, mais pas identiques à ceux du pilote 11 ; je compare les variantes à leur propre témoin.

Sources : [45 résultats](../../../artifacts/comparisons/gru-optim/12-stock-mixer-post-open/results.json), [protocole et lancement](../../plan/stock-mixer-post-open.md).
