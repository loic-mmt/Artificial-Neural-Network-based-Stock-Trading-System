# Premier replay du module de trading

J'ai rejoué les positions figées `open_gap` du fold 0, seed 1, sur les dix actions CAC40. Même moteur OHLC avec et sans règles, capital 10 000, frais 5 bps et slippage 2 bps. Décision après open et exécution next-open, sans réentraîner le modèle ni ouvrir le holdout.

| Variante | PnL net | Rendement net | Sharpe | Drawdown | Exposition brute moyenne |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sans règles | +440,69 | +4,407 % | 0,854 | -1,195 % | 6,081 % |
| Ensemble de règles | -5,63 | -0,056 % | -0,445 | -0,085 % | 0,100 % |

Le très faible drawdown avec règles vient surtout de la quasi-absence d'exposition. Il ne prouve pas une meilleure stratégie. Les sorties et le blocage de réentrée réduisent presque entièrement les positions.

Cela motive l'[ablation des sorties](trading-exit-ablation.md) : je sépare TP, durée, SL, trailing et politique de réentrée pour identifier le mécanisme. Les frais inférieurs du bras avec règles ne compensent pas son manque d'exposition.

Ce premier replay utilise son ancienne configuration sauvegardée. Je ne lui attribue pas rétroactivement le nouveau mode `equal_active`. Les nouveaux paramètres d'allocation doivent être testés dans un nouveau rapport. Une seule paire fold/seed ne permet pas de choisir une règle universelle.

Source : [rapport vérifié](../../artifacts/trading/fold-0-seed-1-verified/report.json). Les dossiers `example` et `verified` sont des replays techniques du même cas, pas des périodes de validation indépendantes.
