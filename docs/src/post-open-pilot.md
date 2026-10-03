# Pilote GRU après ouverture

Le protocole ci-dessous décrit le code du pilote 11. Le run de 18
entraînements est terminé : [résultats et limites](../benchmarks/gru-optim/11-post-open.md).

Ce pilote compare deux GRU Sharpe appariés, sur les mêmes trois folds externes et les seeds `1,7,19` du témoin 04 :

- `lagged` : features journalières achevées jusqu'à J−1 ;
- `open_gap` : mêmes features, plus `night_pct = open(J) / close(J−1) − 1`.

Le modèle est supposé décider juste après l'ouverture de J. Il ne voit jamais `high(J)`, `low(J)`, `close(J)` ou `adj_close(J)` dans ses features. Les jours de split connus dans les données ont un gap manquant plutôt qu'une variation artificielle. La sélection des colonnes, leur remplissage et leur normalisation sont ajustés uniquement sur le train de chaque fold. Les 31 colonnes historiques sélectionnées sont communes aux deux variantes ; `open_gap` reçoit `night_pct` en 32e colonne. Les cinq jours de gap restent disponibles comme contexte passé, mais sont exclus des cibles d'entraînement et de validation intérieure.

La loss Sharpe est réentraînée sur des rendements **open ajusté J vers open ajusté J+1**, avec 5 bps de coût par changement de position. L'open J est un **proxy d'exécution**, pas un prix que l'on prétend obtenir après avoir observé cet open. Le facteur d'ajustement dérivé de `adj_close(J)` ne sert qu'à calculer les rendements historiques cibles, jamais à produire `night_pct` ou une autre feature du matin. Aucun relabeling triple-barrier n'est nécessaire pour ce pilote : les labels de classe ne sont pas utilisés comme cibles de l'entraînement Sharpe.

L'analyse d'exécution sur le fold externe conserve les positions figées et recalcule les résultats avec trois prix déterministes : high défavorable / low défavorable selon le sens de chaque transaction, milieu `(high+low)/2`, et l'inverse favorable. Un tirage uniforme entre low et high est répété 250 fois par défaut avec des nombres aléatoires identiques pour les deux variantes. Il s'agit **uniquement d'une sensibilité mathématique**, pas de fills observés : une extrémité de la bougie a pu être touchée avant la décision, et la loi uniforme n'est pas une distribution empirique de prix d'exécution.

Lancer le benchmark complet, 18 entraînements :

```bash
.venv/bin/python scripts/run_post_open_pilot.py \
  --data data/processed/cac40_daily_clean.parquet \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --reference-cv artifacts/comparisons/gru-optim/04-causal-normalization-60-atr \
  --seeds 1,7,19 \
  --folds 0,1,2 \
  --draws 250 \
  --device auto \
  --output-dir artifacts/comparisons/gru-optim/11-post-open-pilot
```

En cas d'interruption, relancer exactement la même commande avec `--resume`. Chaque résultat sauvegardé contient les positions, les métriques du proxy et la sensibilité OHLC. `--max-epochs 1` existe seulement pour les tests techniques ; ne pas l'utiliser pour conclure sur la performance. Le holdout final à partir du 10 mai 2022 reste fermé.

Le témoin historique 04, à Sharpe régularisé moyen de 0,7509, reste une référence secondaire : il utilise d'autres features et une autre convention d'exécution. **Le test causal de l'information apportée par l'open est la comparaison appariée `open_gap` contre `lagged` dans ce nouveau pilote.** Si le signal résiste à cette sensibilité, l'étape suivante est d'obtenir des prix intrajournaliers horodatés à une heure de décision fixe pour valider une exécution réelle.
