# Benchmark labels, CE et loss financière

Le [plan](../plan/plan-benchmark-labels.md) fixe les questions. Ce guide décrit le runner dédié, ses commandes et ses limites. Aucun comportement des anciens benchmarks n'est remplacé.

## Vérifier la grille sans entraînement

```shell
.venv/bin/python scripts/run_label_loss_benchmark.py --dry-run
```

Valeurs par défaut : 143 tickers US du fichier `configs/benchmark/stocks_us_gnn_complete_2005.json`, GRU32 attention, contexte 60, 32 features au maximum dont `night_pct`, trois folds, seeds 1/7/19, coûts 5 bps par sens. La grille comporte **99 fits** et **378 trajectoires d'évaluation**. Le holdout à partir du 22 juin 2023 reste fermé.

Les paramètres peuvent être limités pour un pilote avec `--tickers`, `--folds`, `--seeds`, `--families`, `--label-methods`, `--protocols` et `--decoders`. Ces listes, sauf `--tickers`, utilisent des valeurs séparées par des virgules. Le nombre de fits est recalculé par `--dry-run`.

## Lancement complet

```shell
.venv/bin/python -u scripts/run_label_loss_benchmark.py \
  --data data/processed/mt5_stocks_us_daily_clean.parquet \
  --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json \
  --device auto \
  --output-dir artifacts/comparisons/us-label-loss-post-open
```

Sur le PC, utiliser le Python de l'environnement et `--device cuda`. Le script n'effectue aucun téléchargement et n'utilise ni news ni fondamentaux. Les groupes d'inputs autorisés sont `technical`, `market`, `sector`.

## Reprise

Relancer **la même commande** avec `--resume`.

```shell
.venv/bin/python -u scripts/run_label_loss_benchmark.py \
  --data data/processed/mt5_stocks_us_daily_clean.parquet \
  --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json \
  --device auto \
  --output-dir artifacts/comparisons/us-label-loss-post-open \
  --resume
```

Les fits terminés sont réutilisés après vérification des fichiers, tailles et SHA-256. Les données de développement, la recette, les folds, les sources Python et les versions numériques doivent correspondre. Un checkpoint seul n'est pas une tâche terminée. Un fit interrompu est réentraîné ; la reprise intra-époque de l'optimizer n'est pas implémentée. Les exports sont atomiques.

## Pilote réduit

Ce pilote vérifie les trois familles, les labels et les deux protocoles sur un fold et une seed. Il ne permet pas de conclure sur les performances de l'univers US.

```shell
.venv/bin/python -u scripts/run_label_loss_benchmark.py \
  --tickers AAPL MSFT \
  --start 2019-01-02 --end 2021-12-31 \
  --folds 0 --seeds 1 \
  --epochs 20 --epoch-multiplier 1 \
  --device cpu --no-plots \
  --output-dir artifacts/comparisons/label-loss-pilot
```

Le plafond est `--epochs × --epoch-multiplier`, donc 300 par défaut. Le multiplier est commun aux trois familles. `--patience 20` et `--min-delta 0.0001` contrôlent l'arrêt anticipé. Il faut lire `budget_insufficient`, les pertes de validation et les gradients avant d'interpréter un modèle arrêté au plafond. Un petit pilote ne valide pas la convergence du benchmark complet.

## Contrats temporels

Les inputs de séance J comprennent les features achevées jusqu'à J-1 et le gap calculé avec l'open J et la clôture brute J-1. Les prix cibles ne sont jamais des inputs. L'open est un proxy, pas une promesse de fill une fois observé.

Les labels `intraday_return`, `forward_return`, `volatility_position` utilisent tous les classes Short/Flat/Long. Le neutre forward signifie Flat uniquement dans ce nouveau contrat. H=10 correspond à open J vers close J+9. Les rendements à horizon supérieur à un jour utilisent des prix ajustés cohérents ; l'intraday utilise open/close bruts d'une même séance.

La volatilité utilise seulement les clôtures jusqu'à J-1. Son état persistant repart de Flat à chaque partition. Les labels dont le futur requis dépasse la partition ne sont pas supervisés. Le gap est compté en dates globales, y compris pour l'intraday. Les prix des périodes de gap peuvent fournir le contexte passé aux fenêtres suivantes, jamais une cible TRAIN hors partition.

La sélection non supervisée des features, les médianes de remplissage et le scaler sont ajustés une fois sur TRAIN et réutilisés pour toutes les familles. Les poids de classes sont calculés séparément sur les seuls labels TRAIN connus. Les masques des labels n'éliminent aucune séance du calcul financier.

## Exécution et allocation

Intraday : position à open J, rendement open J vers close J, coût d'entrée et de sortie chaque jour actif. Overnight : position à open J, rendement vers l'open suivant, coûts de changement, entrée initiale et liquidation finale. La dernière séance overnight contient seulement la liquidation, aucun nouveau trade ni rendement fictif.

Chaque ticker occupe un slot fixe `q/N`. Les absences initiales/finales et les signaux indisponibles restent cash. Aucun levier ni réallocation aux seuls actifs n'est ajouté. Un trou interne de cotation est rejeté, sans prix remplis et sans rendement sur plusieurs séances présenté comme quotidien.

Les références sont always-long intraday, buy-and-hold overnight et cash. Une référence long est aussi réduite à l'exposition moyenne du modèle. Ce contrôle égalise une moyenne, pas le timing de l'exposition ; il ne prouve pas une alpha.

`common-exposure.csv` compare aussi les modèles entre eux à l'exposition moyenne la plus faible de leur groupe fold/seed/protocole/décodage. Les positions, rendements et coûts sont réduits ensemble, sans levier. Si un candidat reste entièrement cash, cette comparaison devient dégénérée ; les résultats bruts restent indispensables.

## Entraînement

Chaque époque fige les poids pendant un parcours complet. Les forwards et gradients sont calculés par blocs, puis une seule update AdamW est effectuée. La loss financière voit la trajectoire chronologique globale, pas une moyenne de Sharpes de minibatches. Le dropout est rejoué à l'identique entre le calcul des positions et leur backward.

Les objectifs sont CE pondérée, combinée financière avec poids PnL 0,25 et échelle 0,0001, ou hybride avec poids CE 0,5. Les deux coefficients sont indépendants. La financière ne consomme aucun label. La CE produit un seul checkpoint par méthode, réutilisé dans les deux protocoles ; les hybrides sont entraînés par méthode et protocole.

Le meilleur checkpoint utilise exclusivement la validation intérieure, selon la loss totale de sa famille. Les folds OUTER servent au reporting. Les décodages continu, signe et argmax sont appliqués aux mêmes probabilités sauvegardées, sans réentraînement.

Les blocs limitent les activations et gradients conservés sur le GPU. Les fenêtres 3D restent matérialisées en mémoire CPU pour le fold courant ; réduire `--batch-size` ne réduit donc pas toute la mémoire CPU. Le runner n'est pas un dataset entièrement lazy.

## Fichiers produits

- `metadata.json` : contrat, configuration, provenance et grille.
- `results.csv`, `report.json` : KPIs et complétude globale. Aucune sélection sur OUTER.
- `summary.csv` : agrégats par modèle, protocole et décodage, sans sélectionner le meilleur modèle sur OUTER.
- `classification.csv` à la racine : scores TRAIN, validation intérieure et OUTER pour chaque labellisation.
- `common-exposure.csv` : comparaison des modèles à exposition moyenne commune.
- `fold-N/preprocessing.json` : features, remplissage, scaler et limites temporelles.
- `fold-N/execution_contracts.json` : calendrier, prix, allocation et masques financiers de chaque partition/protocole.
- `fold-N/*-labels.parquet`, `label_oracles.json` : cibles OUTER et exécution parfaite rétrospective. Les labels à horizon 10 ne garantissent pas un plafond financier sous un autre protocole.
- `fold-N/seed-S/candidat/checkpoint.pt` : meilleur checkpoint intérieur.
- `probabilities.parquet`, `classification.csv` : probabilités alignées date/ticker, scores contre chacune des trois méthodes et référence majoritaire TRAIN.
- `learning_diagnostics.json` : composantes CE/financière, gradients, pertes TRAIN/validation, epochs, updates et budget insuffisant.
- `intraday-*-*.parquet`, `overnight-*-*.parquet` : portefeuille, positions, KPIs par ticker et épisodes de trades pour chaque décodage applicable.
- `*-metrics.json` : KPIs et contrôles d'exposition.
- `complete.json` : fichiers attendus et hashes de la tâche terminée.
- `oppositions.csv`, `oppositions/` : fréquences Long contre Short, épisodes, exposition engagée et contributions.

Les graphiques présentent les courbes de portefeuille, les positions par couleurs et les courbes de pertes/gradients. Ils peuvent être désactivés par `--no-plots`. La progression tient sur une barre ; `--no-progress` la désactive.

Les scores de classification des modèles purement financiers restent diagnostiques. Deux distributions de classes peuvent produire la même position `P(Long)-P(Short)` et donc la même loss financière.

Les exports distinguent les moyennes sur les trois classes fixes (`macro_f1_fixed3`, `balanced_accuracy_fixed3`) et sur les seules classes présentes dans les cibles (`macro_f1_present_classes`, `balanced_accuracy_present_classes`). Pour l'intraday sans égalité open/close, Flat peut être absente : une classification parfaite plafonne alors à 2/3 sur les moyennes fixes. Il faut lire les supports et les scores par classe, puis les moyennes sur classes présentes, pour juger l'apprenabilité. Les scores historiques `macro_f1` et `balanced_accuracy` restent conservés.

## Validation technique

- 86 nouveaux tests et deux tests d'architecture passent, soit 88 tests ciblés. Ils couvrent causalité, partitions, alignement, coûts, gradients, décodages, allocation et reprise.
- Le pilote AAPL/MSFT, 2019 à 2021, un fold et une seed, a terminé ses 11 fits et 42 évaluations avec un plafond de 300 époques. Tous les fits se sont arrêtés par early stopping entre 21 et 133 époques ; aucun n'a atteint le plafond. Les résultats sont dans `artifacts/comparisons/label-loss-pilot-300-20261009`.
- Un smoke test final et sa reprise avec `--resume` vérifient le runner courant et la réutilisation des fits terminés.
- La suite complète exécutée pendant l'implémentation conserve 25 échecs FNSPID/news, dans des modules non modifiés par ce benchmark. Un échec a aussi été reproduit isolément. Rapport : `artifacts/diagnostics/label-loss-full-suite-20261009.xml`.

La validation locale utilise le CPU. CUDA et MPS n'ont pas pu être testés sur cette machine. Le pilote ne valide ni la convergence sur 143 tickers ni les performances de la grille complète, qui reste à lancer.
