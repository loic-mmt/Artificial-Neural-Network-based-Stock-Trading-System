# Comparaison initiale des labels et des modèles

J'ai comparé breakout, forward return, volatility position et triple barrier sur les dix actions CAC40, avant le nettoyage OHLC. Chaque méthode utilise cinq modèles et les seeds 1, 7, 19, 42, 1337 : 25 runs par méthode, sans échec enregistré. C'est une comparaison statique, pas les folds purgés de la série learning.

## Résultats des modèles

PnL moyen par architecture pour un capital initial de 10 000. Buy-and-hold commun : +6 102,10. Je ne compare pas les macro-F1 comme si les quatre méthodes définissaient les mêmes classes ou la même tâche.

| Méthode | GRU | LSTM | Manual ANN | RNN | Transformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breakout | +960,71 | +480,13 | +71,04 | +903,45 | +470,10 |
| Forward return | -3 858,96 | -3 861,26 | -3 216,94 | -3 961,21 | -3 575,39 |
| Volatility position | -262,55 | -654,32 | -864,70 | -994,56 | -709,38 |
| Triple barrier | -438,05 | -19,45 | +22,48 | -1 160,08 | +144,76 |

Aucune moyenne de modèle ne dépasse buy-and-hold. Triple Barrier n'était donc pas le meilleur PnL de modèle dans ce premier tableau. La décision de poursuivre cette méthode venait aussi du diagnostic de ses labels parfaits, pas seulement des performances apprises.

## Labels parfaits et capacité à apprendre

Le notebook compare l'exécution exacte des labels au modèle et à buy-and-hold. J'avais observé un potentiel élevé du Triple Barrier, mais un modèle très loin de le reproduire. C'est ce décalage qui a motivé les features supplémentaires, les losses financières et le contrôle du surapprentissage.

Un label calculé avec les prix futurs n'est pas une prévision disponible à la décision. Même un excellent backtest de labels ne prouve pas que ces labels sont apprenables. Inversement, des classes très déséquilibrées peuvent produire une forte accuracy en prédisant presque toujours Hold.

Les quatre `report.json` ne sauvegardent pas les courbes du diagnostic parfait calculé dans le notebook. Je ne transforme pas le chiffre arrondi discuté à l'époque en KPI vérifié. Il reste à persister ce diagnostic avec données, paramètres, coûts et timestamps pour conserver une comparaison exacte.

## Décision

Je garde Triple Barrier comme cadre de labeling pour les expériences suivantes, mais pas comme méthode universellement rentable. La série learning fait ensuite varier estimateur, barrières, horizon et filtre sur données nettoyées. Les losses Sharpe/PnL optimisent une trajectoire financière : leur score n'est pas une mesure de reproduction fidèle des labels.

Sources : [breakout](../../artifacts/comparisons/20260902T083027Z/report.json), [forward return](../../artifacts/comparisons/20260903T100057Z-m1-forward-return/report.json), [volatility position](../../artifacts/comparisons/20260903T100057Z-m2-volatility-position/report.json), [triple barrier](../../artifacts/comparisons/20260903T100057Z-m3-triple-barrier/report.json), [notebook de comparaison](../../notebooks/compare_model_benchmarks.py).
