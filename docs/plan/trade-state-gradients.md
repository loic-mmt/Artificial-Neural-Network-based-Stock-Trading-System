# Benchmark des gradients d'état en overnight

Statut au 12 octobre 2026 : benchmark terminé sur le PC GPU, 45 fits et
90 trajectoires, holdout fermé. TBPTT 40 reste très proche du S3 détaché,
sans gain robuste entre périodes. TBPTT 10 augmente surtout l'exposition et
dégrade le Sharpe moyen. S3 détaché reste la référence pour le
[benchmark des décisions short](short-decision-benchmark.md), désormais prêt.
Les résultats sont dans `artifacts/comparisons/us-trade-state-gradients`.

## Question

Il faut vérifier si donner du crédit aux décisions passées via les états
futurs améliore la stratégie, notamment la fermeture des shorts perdants.
Le [benchmark S0-S3](../benchmarks/us-trade-state-overnight.md) n'a testé que
des états détachés : davantage de rendement financier, mais surtout davantage
d'exposition longue, et des positions perdantes qui persistent.

## Grille retenue

| Candidat | Famille | États | Gradient d'état |
| --- | --- | --- | --- |
| `financial-S0-detached` | PnL/Sharpe 0,25 | Aucun état actif | Détaché, contrôle de capacité |
| `financial-S3-detached` | PnL/Sharpe 0,25 | Trade et portefeuille | Détaché, contrôle principal |
| `financial-S3-tbptt-10` | PnL/Sharpe 0,25 | Trade et portefeuille | Propagation tronquée sur 10 séances globales |
| `financial-S3-tbptt-40` | PnL/Sharpe 0,25 | Trade et portefeuille | Propagation tronquée sur 40 séances globales |
| `cross_entropy-S0-detached` | CE volatilité | Aucun état actif | Référence supervisée |

Cinq configurations, trois folds et seeds 1, 7, 19 donnent **45 entraînements
et 90 trajectoires outer**. `--tbptt-windows` peut modifier séparément les
horizons du gradient. `--families financial` retire uniquement le témoin CE,
soit 36 fits et 72 trajectoires avec les deux fenêtres par défaut.

Les contrôles détachés sont réentraînés dans le nouveau run. Il ne faut pas
importer implicitement des résultats ou des poids du run S0-S3 précédent.

## Ce qui ne change pas

- GRU32, attention, tête linéaire, contexte 60 et cap de 32 features ; même
  nombre de paramètres et mêmes poids initiaux à seed égale.
- Même univers de 143 actions, features J-1 plus gap open J, état observé
  avant décision, exécution proxy open J vers open J+1.
- Slots fixes `q/N`, coûts 5 bps par sens, absence de signal en cash et
  liquidation finale. Ni stop, ni durée maximale ajoutée.
- Loss PnL/Sharpe globale, poids PnL 0,25. Le Sharpe n'est pas calculé
  séparément par fenêtre TBPTT.
- Une seule update AdamW par époque, même plafond de 300 époques et patience
  20. Les gradients de toutes les fenêtres s'accumulent avant clipping/update.
- Préparation TRAIN uniquement et sélection du checkpoint sur inner. Le
  holdout à partir du 22 juin 2023 reste fermé.
- Entraînement continu, évaluation continue principale et signe ±1 secondaire,
  avec une simulation indépendante pour chaque décodage.

Le modèle ne ferme pas ses positions aux frontières TBPTT. Le portefeuille
et ses compteurs continuent ; seule la propagation du gradient est tronquée.
Une fenêtre de 10 séances n'impose donc pas une détention de 10 séances.

## Portée réelle du gradient

La propagation traverse la position précédente, les expositions brute/nette,
la fraction cash et le drawdown issu de l'equity nette. Les coûts et les
rendements passés peuvent ainsi influencer les décisions suivantes via ces
états, à l'intérieur de la fenêtre.

Le signe, les changements d'épisode, le prix d'entrée, l'âge et le compteur
underwater restent discrets et sans gradient. Il ne faut pas présenter cette
méthode comme une différentiation complète d'un moteur d'ordres, ni comme un
surrogate pour sign/argmax. La CE avec états n'est pas propagée en TBPTT dans
cette première grille ; son instabilité au signe reste un chantier distinct.

Les embeddings sont mis en cache sans graphe. Le petit graphe temporel de la
tête et du portefeuille vit sur CPU, puis transmet une VJP aux embeddings.
Le GRU est rejoué par blocs avec le même flux dropout. La mémoire GPU ne croît
donc pas avec toute la durée du portefeuille. La mémoire CPU du graphe croît
avec la fenêtre choisie, en plus des caches par ligne.

## Commande complète sur Mac

```shell
.venv/bin/python -u scripts/run_trade_state_gradient_benchmark.py \
  --data data/processed/mt5_stocks_us_daily_clean.parquet \
  --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json \
  --tbptt-windows 10,40 \
  --device auto \
  --output-dir artifacts/comparisons/us-trade-state-gradients
```

## Commande complète sur PC GPU

Avec l'environnement Python du projet activé, PowerShell :

```powershell
python -u scripts/run_trade_state_gradient_benchmark.py --data data/processed/mt5_stocks_us_daily_clean.parquet --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json --tbptt-windows 10,40 --device cuda --output-dir artifacts/comparisons/us-trade-state-gradients
```

Pour reprendre, ajouter `--resume` à la même commande. Les fits complets sont
vérifiés par taille et SHA-256 ; un fit interrompu repart de son initialisation.
La reprise exige données, recette, sources et versions numériques identiques.
Il faut utiliser un nouveau dossier, pas celui du benchmark S0-S3 terminé.

Pour voir la grille sans lire les données ni entraîner :

```shell
.venv/bin/python scripts/run_trade_state_gradient_benchmark.py --dry-run
```

## Petit contrôle CUDA avant le run complet

```powershell
python -u scripts/run_trade_state_gradient_benchmark.py --data data/processed/mt5_stocks_us_daily_clean.parquet --tickers AAPL MSFT --start 2019-01-02 --end 2021-12-31 --folds 0 --seeds 1 --epochs 2 --epoch-multiplier 1 --tbptt-windows 2,4 --device cuda --no-plots --output-dir artifacts/comparisons/trade-state-gradients-cuda-pilot
```

Ce pilote vérifie les cinq candidats et dix trajectoires, pas la rentabilité.
Les tests logiciels locaux portent sur CPU. Le run complet terminé dans
`us-trade-state-gradients` utilise CUDA ; un nouvel environnement doit encore
faire ce pilote avant sa propre grille complète.

## Lecture des résultats

Les exports S0-S3 sont conservés : probabilités, états, positions, portefeuille,
trades, KPI par ticker, classification et courbes. Les clés date/ticker restent
alignées, les quotes manquantes ne sont pas remplacées et les labels inconnus
ne deviennent pas de vrais Flat.

`gradient-paired.csv` compare chaque TBPTT au S3 détaché du même fold, seed et
décodage. Les deltas incluent rendement, Sharpe, drawdown, exposition long/short,
turnover, coûts, épisodes, contribution short et durée d'entraînement.
`common-exposure.csv` reste le contrôle d'exposition moyen a posteriori.

Il faut examiner en priorité :

1. Gain apparié face à S3 détaché, stabilité entre folds/seeds, coût de calcul.
2. Gain à exposition comparable, sans confondre gross identique et beta égal.
3. Contribution nette des shorts, fermeture des shorts durablement perdants
   et exemple AAPL, plutôt que seulement davantage de Long.
4. Durées, inversions et coûts. Le signe reste un diagnostic secondaire,
   pas une preuve que la politique continue a appris ses décisions binaires.
5. Cohérence des pertes et normes de gradients, erreur de replay et fenêtre
   réellement conservée. Les fenêtres modifient le crédit temporel, pas la
   cadence des updates.

Les résultats outer restent descriptifs. Il ne faut pas choisir une seed ou
un ticker gagnant après inspection. Après ce test, la priorité reste le
[benchmark des décisions long/short](short-decision-benchmark.md).

Les détails techniques sont dans [le guide du package](../src/trade-state-gradients.md).
