# MT5 US : comparaison des décodages

J'ai rejoué les probabilités du walk-forward US every 20, sans réentraîner cinq modèles différents. Même seed 42, 196 actions, 93 rendements, capital 10 000 et coûts 5 bps. Les seuils de confiance et de zone neutre sont choisis pour chaque bloc sur sa validation passée, pas sur la période externe.

| Décodage | Règle | PnL net | Sharpe régularisé | Drawdown | Exposition | FLAT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Continu | `P(Buy)-P(Sell)` | -28,31 | -0,6487 | -1,17 % | 8,80 % | 0 % |
| Signe | ±1 selon le signe | -155,14 | -0,3952 | -8,75 % | 100 % | 0 % |
| Argmax | Classe la plus probable | +40,83 | 0,1767 | -7,36 % | 84,40 % | 15,53 % |
| Zone neutre | FLAT sous seuil d'amplitude | +172,00 | 1,4807 | -1,59 % | 22,18 % | 78,06 % |
| Confiance | FLAT si les deux meilleures probabilités sont trop proches | +326,41 | 2,6561 | -2,13 % | 18,33 % | 81,86 % |

Buy-and-hold : +1 126,33, Sharpe régularisé 2,9363. Le taux FLAT compte les décisions ticker/date, tandis que l'exposition moyenne reflète les positions exécutées et la convention de liquidation. Ils ne sont donc pas exactement complémentaires.

Je garde confiance comme candidat pour le benchmark de fréquence. C'est le meilleur compromis observé ici, pas une preuve que les probabilités sont calibrées. Argmax confond encore trop souvent les directions et forcer ±1 ne répare pas ce problème.

Les diagnostics de seuils fixes ne doivent pas servir à sélectionner le meilleur seuil sur ce test externe. Après décodage, turnover et coûts sont recalculés. Les univers, dates et objectif restent identiques entre variantes.

Sources : [rapport](../../artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/decoder-benchmark/report.json), [script de replay](../../scripts/benchmark_walkforward_decoders.py).
