# 06. GNN et contrôles de graphe sur dix actions

J'ai comparé cinq branches isolées, sur 3 folds et les seeds 1, 7, 19 : 45 résultats réussis. Le GNN reçoit ses propres features de nœuds, jamais les embeddings GRU. Objectif Sharpe, coût 5 bps, contexte 60, largeur 32. Le contrôle `identity` n'échange aucune information entre actions.

| Candidat | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen |
| --- | ---: | ---: | ---: |
| GRU | 0,7535 | +7,19 % | -3,42 % |
| Identité | 0,6665 | +4,77 % | -3,29 % |
| Secteur | 0,6705 | +4,87 % | -3,30 % |
| Pearson train | 0,6665 | +4,77 % | -3,29 % |
| Pearson rolling | 0,6672 | +4,93 % | -3,28 % |

Le GRU reste devant en moyenne. Le graphe Pearson train ne fournit pas d'arêtes dans ce test et reproduit exactement identité. Le rolling ne gagne en Sharpe face au GRU que dans 3 paires sur 9 ; identité et secteur dans 4 sur 9. Ce résultat ne prouve pas que les relations sont inutiles : dix actions et un seuil de corrélation élevé constituent un test limité.

Je conserve le GRU et le contrôle identité. La suite élargit l'univers US et teste top-k et corrélations résiduelles, sans présenter ce changement de protocole comme un gain déjà acquis.

Le dataset porte l'empreinte `caa4c426…` et le holdout commence le 24 mai 2022, pas le 10 mai du premier journal. Il reste fermé. Comparaison appariée à l'intérieur de ce run uniquement.

Source : [rapport, métriques et diagnostics des graphes](../../../artifacts/comparisons/gru-optim/06-gnn-graph-controls-60-atr/report.json).
