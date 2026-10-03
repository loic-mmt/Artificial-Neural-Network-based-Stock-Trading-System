# 08 à 10. Transformer d'état de marché et gating

J'ai comparé six candidats, 3 folds et 3 seeds : 54 résultats réussis. Le Transformer reçoit une séquence globale séparée des actions, ici rendement du marché, niveau du VIX et rendement du VIX. Il peut guider les features, sans être obligatoirement un vote de trading.

| Candidat | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen | Exposition brute moyenne |
| --- | ---: | ---: | ---: | ---: |
| GRU témoin | 0,5031 | +2,36 % | -1,84 % | 6,61 % |
| Concaténation | 0,5490 | +2,82 % | -2,16 % | 7,89 % |
| Gate statique | 0,5038 | +2,37 % | -1,84 % | 6,61 % |
| Gate marché simple | 0,5092 | +2,41 % | -1,84 % | 6,76 % |
| Gate Transformer | 0,5089 | +2,08 % | -1,71 % | 6,76 % |
| Transformer seul | 0,2694 | +3,75 % | -10,79 % | 17,84 % |

La concaténation gagne en moyenne mais seulement dans 4 paires sur 9 en Sharpe. Le gate Transformer gagne aussi dans 4/9, pour un gain moyen de seulement 0,0058. Son turnover moyen est 3,35 contre 1,29 pour le GRU. Le Transformer seul prend plus de risque et atteint un Sharpe minimal négatif de -0,5503.

Je garde le GRU sans gate comme contrôle. Je ne généralise pas ce test de dix actions et trois features marché à un contexte US ETF plus riche. L'US teste justement cette hypothèse séparément.

Les features actions sont `technical,sector`. Dataset `caa4c426…`, holdout à partir du 24 mai 2022, toujours fermé. Le GRU 0,5031 doit être comparé à ses candidats ici, pas au GRU 0,7535 du benchmark 06 dont les entrées diffèrent.

Source : [rapport marché](../../../artifacts/comparisons/gru-optim/08-10-market-context-60-atr/report.json).
