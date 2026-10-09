# Benchmark des états de trade et de portefeuille

## Question

Il faut vérifier si les informations sur les positions déjà prises améliorent les décisions du GRU. Le benchmark précédent ne permet pas de conclure que leur absence explique, à elle seule, les shorts prolongés sur AAPL.

## Grille S0 à S3

| Variante | Inputs d'état actifs |
| --- | --- |
| S0 | Aucun, dix inputs remplis de zéros pour conserver la même architecture. |
| S1 | Position précédente, position ouverte ou non, âge de l'épisode en séances. |
| S2 | S1 plus rendement signé depuis l'entrée, recul depuis le meilleur rendement de l'épisode, séances consécutives sous le prix d'entrée. |
| S3 | S2 plus expositions brute et nette du portefeuille, fraction de capital non exposée, drawdown. |

Deux familles sont comparées : CE sur `volatility_position`, horizon 10, et loss financière PnL/Sharpe combinée, poids PnL 0,25. Il n'y a pas de famille hybride, de protocole intraday ni de nouvelle règle de stop.

Le contexte marché reste GRU32 attention, contexte 60, cap de 32 features, groupes technical/market/sector. Les états courants sont concaténés après le pooling, jamais répétés dans les 60 barres historiques. Chaque variante dispose des mêmes dix emplacements d'état et du même nombre de paramètres. S0 est un témoin de capacité, pas une reproduction bit à bit de l'ancien GRU.

## Protocole commun

Les données achevées à J-1 et le gap open J sont observés. L'état de la position précédente est marqué à l'open J avant la décision. L'exécution utilise l'open J comme proxy, puis le rendement open J vers open J+1. Ce proxy ne garantit pas le prix réel après observation de l'ouverture.

L'allocation conserve les slots fixes `q/N`, les coûts de 5 bps par sens et la liquidation finale. Un signal manquant signifie cash, sans supprimer de séance. Il n'y a ni réallocation aux seuls actifs ouverts, ni levier, ni changement des règles de sortie. Horizon 10 concerne la cible CE, pas une durée de détention imposée.

Les états proviennent uniquement des décisions du modèle courant. Aucun état n'est calculé à partir des labels parfaits ou d'un autre backtest. Chaque partition TRAIN/inner/outer commence en cash. Le resizing dans le même sens conserve l'âge et le prix d'entrée ; un passage Flat ou une inversion réinitialise l'épisode.

Les deux familles s'entraînent avec des positions continues. Le continu est l'évaluation principale, le signe ±1 l'évaluation secondaire. Chaque décodage est simulé depuis le début avec ses propres états et ses nouvelles probabilités. Il ne faut pas redécoder un unique fichier de probabilités.

## Gradients et sélection

Tous les états sont détachés. Les paramètres du GRU et de la tête restent entraînables. La loss financière reçoit la trajectoire complète et les coûts de turnover, avec une seule update AdamW par époque. Les conséquences d'une décision sur les états futurs sont exclues du gradient : c'est un semi-gradient, pas une différentiation complète du portefeuille.

Il faut tester le détachement contre une autre méthode, par exemple TBPTT, dans un benchmark ultérieur. Cette comparaison n'est pas implémentée ici.

Les médianes, le scaler, la sélection des features et les poids CE utilisent TRAIN uniquement. Les checkpoints sont choisis sur la loss inner. Les folds outer servent au reporting et le holdout à partir du 22 juin 2023 reste fermé.

Trois folds et seeds 1, 7, 19 donnent **72 entraînements et 144 trajectoires outer**, avec 300 époques maximum, patience 20. Il faut lire les seeds/folds ensemble, l'exposition, le turnover, les shorts prolongés et les pertes d'apprentissage avant de sélectionner une variante.

## Commande

```shell
.venv/bin/python -u scripts/run_trade_state_benchmark.py \
  --data data/processed/mt5_stocks_us_daily_clean.parquet \
  --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json \
  --state-variants S0,S1,S2,S3 \
  --device auto \
  --output-dir artifacts/comparisons/us-trade-state-overnight
```

Sur le PC GPU, avec l'environnement activé, utiliser cette commande PowerShell :

```powershell
python -u scripts/run_trade_state_benchmark.py --data data/processed/mt5_stocks_us_daily_clean.parquet --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json --state-variants S0,S1,S2,S3 --device cuda --output-dir artifacts/comparisons/us-trade-state-overnight
```

Il faut reprendre la même commande avec `--resume`. Les fits terminés sont vérifiés et conservés ; un fit interrompu repart depuis son initialisation. Les données, les sources Python, la recette et les versions numériques doivent rester identiques. `--dry-run` affiche la grille sans lire les prix ni entraîner.

Les formats et limites des états sont détaillés dans [le guide du package](../src/trade-state-benchmark.md).
