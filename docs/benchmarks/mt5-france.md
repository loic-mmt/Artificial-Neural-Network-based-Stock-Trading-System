# MT5 France : statique, walk-forward et exposition

J'ai repris le GRU `open_gap` du pilote 11 sur 39 actions françaises, seed 42. Période du 20 mars au 24 septembre 2026, 131 rendements open-to-open. Capital 10 000, coûts proportionnels 5 bps, features J-1 plus gap de l'open J.

| Test | PnL net | Rendement | Sharpe régularisé | Drawdown | Exposition brute moyenne |
| --- | ---: | ---: | ---: | ---: | ---: |
| Statique, Sharpe | +127,77 | +1,28 % | 2,2193 | -0,27 % | 13,50 % |
| Walk-forward, Sharpe | +94,39 | +0,94 % | 1,7384 | -0,41 % | 12,22 % |
| Walk-forward, combinée 0,25 | +119,73 | +1,20 % | 1,7032 | -0,48 % | 18,06 % |
| Même combinée, signe ±1 | +798,23 | +7,98 % | 2,4609 | -1,10 % | 100 % |
| Buy-and-hold | +636,13 | +6,36 % | 0,7906 | -7,75 % | 100 % |

Le modèle statique n'était pas réentraîné pendant la période. Les deux walk-forward réentraînent toutes les 20 séances, soit sept checkpoints, pas chaque jour. Entre deux checkpoints, les entrées changent quotidiennement et la prévision reste celle du prochain intervalle open-to-open. Le contexte 60 décrit le passé lu, pas une prévision à 60 jours.

En continu, le rendement plus faible que buy-and-hold vient en partie de la faible exposition. Le passage au signe utilise les mêmes directions à pleine exposition et bat buy-and-hold sur ce test France. Il n'ajoute pas une information prédictive nouvelle et ce gain ne se généralise pas au [test US](mt5-us.md).

La position d'un ticker peut rester du même sens pendant un bloc entier. Cela ne signifie pas qu'il n'existe qu'un long et un short pour tout le portefeuille. Les positions par ticker/date et les redimensionnements doivent être distingués des changements de direction.

Je garde le walk-forward pour vérifier une utilisation séquentielle, et compare les décodages séparément. Une seule seed et une période déjà consultée ne suffisent pas à valider un déploiement MT5. L'open reste un proxy, sans frais de financement short ou contraintes de lots du broker validés ici.

Sources : [statique](../../artifacts/mt5/france-gru-benchmark11-seed42/portfolio_metrics.json), [walk-forward Sharpe](../../artifacts/mt5/france-gru-benchmark11-walkforward-seed42/portfolio_metrics.json), [walk-forward combinée](../../artifacts/mt5/france-gru-benchmark11-walkforward-combined-025-seed42/portfolio_metrics.json), [signe et allocation](../../artifacts/mt5/france-gru-benchmark11-walkforward-combined-025-seed42/sign_portfolio_metrics.json), [journal des réentraînements](../../artifacts/mt5/france-gru-benchmark11-walkforward-combined-025-seed42/retrain_logs.json).
