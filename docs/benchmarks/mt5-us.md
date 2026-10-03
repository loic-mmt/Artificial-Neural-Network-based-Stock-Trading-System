# MT5 US : walk-forward et trades

J'ai appliqué le GRU `open_gap`, loss combinée PnL/Sharpe à 0,25, aux 196 actions US téléchargées et nettoyées. Seed 42, coût 5 bps, réentraînement toutes les 20 séances : cinq checkpoints. Du 13 mai au 25 septembre 2026, 94 dates et 93 rendements open-to-open. Capital 10 000.

| Décodage | PnL net | Rendement | Sharpe régularisé | Drawdown | Exposition moyenne |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continu | -28,31 | -0,28 % | -0,6487 | -1,17 % | 8,80 % |
| Signe ±1 | -155,14 | -1,55 % | -0,3952 | -8,75 % | 100 % |
| Buy-and-hold | +1 126,33 | +11,26 % | 2,9363 | -2,55 % | 100 % |

La sous-performance n'est pas uniquement une question de quantité achetée : le signe à 100 % reste négatif. Les directions sont majoritairement short dans une période haussière. Le dernier jour du signe compte 34 longs et 162 shorts, sans FLAT, pour une exposition nette de -65,31 %.

Le continu n'a pas de FLAT exact dans le diagnostic de décodage, mais de petites amplitudes. En signe, le taux short sur les lignes ticker/date est 67,83 %. Ce taux ne représente pas le nombre de transactions. Les coûts sont petits par rapport aux erreurs de direction ; le [diagnostic par ticker](us-ticker-diagnostics.md) détaille aussi les erreurs de timing et les prix ajustés à contrôler.

99 tickers ont un Sharpe net continu positif ; 106 ont un Sharpe positif en signe. Ces listes sont différentes. Les sélectionner après observation du backtest est une sélection exploratoire, à confirmer sur une période ultérieure figée.

Je teste donc les [décodages](mt5-decoders.md) et la [fréquence de réentraînement](mt5-retraining.md), sans utiliser ce seul backtest pour choisir une blacklist ou inverser automatiquement les positions perdantes.

Sources : [métriques continues](../../artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/portfolio_metrics.json), [signe et allocation finale](../../artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/sign_portfolio_metrics.json), [KPI continus](../../artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/kpis_by_ticker.csv), [KPI signe](../../artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/sign_kpis_by_ticker.csv), [réentraînements](../../artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/retrain_logs.json).
