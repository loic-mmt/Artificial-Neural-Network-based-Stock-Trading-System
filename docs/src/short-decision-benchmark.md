# Benchmark des décisions short

Le [plan](../plan/short-decision-benchmark.md) fixe la recette et les commandes.
Le runner est isolé : aucun modèle ni comportement historique n'est remplacé.

## Modules

- `models/neural/short_decision_gru.py` : détecteur binaire, encodeur partagé
  avec deux têtes, ou deux encodeurs GRU indépendants avec leurs propres têtes.
- `training/short_decision.py` : décodages, projections, cibles binaires,
  poids TRAIN, rollouts causaux et trainer à une update globale par époque.
- `analysis/short_decisions.py` : qualité des activations, erreurs journalières,
  épisodes short et attribution selon une tendance passée de chaque actif.
- `experiments/short_decision_benchmark.py` : préparation commune, témoins,
  checkpoints inner, exports, comparaisons appariées et reprise vérifiée.
- `pipelines/short_decision_benchmark.py` et
  `scripts/run_short_decision_benchmark.py` : commande du benchmark.

## Architecture et budget

Les états S3 courants sont concaténés après le pooling, pas répétés dans les
séquences. Les gradients des états restent détachés. Les encodeurs sont
identiques à la recette GRU existante. Les deux têtes partagées et les deux
GRU séparés utilisent le même arbitrage.

La famille financière optimise le portefeuille Long/Short combiné, coûts
inclus. Elle n'optimise pas séparément un Sharpe par tête. Les deux encodeurs
séparés ont des paramètres indépendants mais un checkpoint commun. La CE
moyenne les losses binaires pondérées et normalisées par tête.

La tête Short distingue Short de not-Short. not-Short comprend les vrais
labels Long et Flat connus, pas les labels inconnus. Pour le détecteur,
not-Short signifie Long dans la politique binaire Long par défaut. Il ne
s'agit pas d'un label Flat transformé silencieusement en Long.

Chaque époque réalise un rollout sans graphe, puis rejoue les mêmes blocs
avec états figés et paramètres inchangés. Les gradients s'accumulent avant
clipping et update AdamW. Les caches ont une taille `[rows, branches, hidden]`
et `[rows, 10]`, sans tenseur récurrent couvrant tout le portefeuille.

## Arguments modulaires

| Argument | Défaut | Effet |
| --- | --- | --- |
| `--architectures` | `short_detector,dual_head,separate_grus` | Variantes entraînées, sélection indépendante. |
| `--families` | `cross_entropy,financial` | CE binaire ou PnL/Sharpe 0,25. |
| `--policies` | `long_short,long_flat,short_flat` | Projections évaluées sans réentraînement. |
| `--decoders` | `continuous,sign` | Continu ou activation binaire des têtes ; témoins avec leur signe historique. |
| `--long-activation-threshold` | `0.5` | Seuil binaire Long, distinct du seuil des labels volatilité. |
| `--short-activation-threshold` | `0.5` | Seuil binaire Short, distinct du seuil des labels volatilité. |
| `--conflict-policy` | `flat` | Deux têtes actives : Flat ou plus grand dépassement normalisé avec `margin`. |
| `--regime-window` | `200` | Diagnostic de tendance passée ; `0` le retire, aucun filtre d'exécution. |
| `--no-controls` | Désactivé | Retire les témoins réentraînés. |
| `--resume` | Désactivé | Vérifie et réutilise seulement les fits complets compatibles. |
| `--dry-run` | Désactivé | Affiche grille et budgets sans lire les données de prix. |

Les paramètres communs restent disponibles : périodes, folds, seeds, features,
context length, coûts, poids PnL, device et budget. Les anciens runners ne
reçoivent pas ces nouvelles options. Les seuils sont fixés avant le run,
pas optimisés sur OUTER ; aucune calibration n'est revendiquée.

## Exports

Un fit est stocké dans `fold-N/seed-N/<candidate>/`. Il contient checkpoint,
préparation référencée, diagnostic d'apprentissage et `complete.json` avec
taille et SHA-256 de tous ses fichiers. Chaque politique possède son sous-dossier.
Pour chaque décodage : scores, signaux, états, positions exécutées, portefeuille,
trades, KPI par ticker, contrôles always-long et courbes sont conservés.

Les exports globaux sont :

- `results.csv`, `summary.csv` : performances par fit, décodage et politique.
- `binary-heads.csv` : TRAIN, inner et OUTER identifiés séparément, masques,
  supports, précision/rappel, balanced accuracy et AUROC par tête. Une classe
  absente rend l'AUROC indisponible, pas égal à zéro. Les scores financiers
  ne sont pas des probabilités de réussite calibrées.
- `short-alerts.csv` : succès des shorts par séance et par épisode, durées,
  shorts perdants de 100/252 séances, compteur underwater aux fermetures.
  Les fermetures décidées par le modèle sont distinguées des liquidations
  forcées. Les compteurs aux fermetures ne mesurent pas le délai de fermeture
  des épisodes non fermés. Ces compteurs concernent les lignes de signal
  observées ; les exports de trades conservent aussi les sorties sans signal.
- `regimes.csv` : contribution PnL additive Long/Short/Flat par tendance
  passée de l'actif. Ce n'est pas un backtest de portefeuilles indépendants
  selon les régimes. La somme se réconcilie avec le PnL global. Ici le côté
  décrit la position courante ; Flat inclut les coûts de liquidation lorsque
  la position cible est nulle. L'attribution globale Long/Short du moteur
  conserve séparément les coûts d'entrée et de sortie de chaque côté.
- `short-paired.csv` : deltas des architectures face au témoin de leur famille,
  même fold, seed, décodage et politique, avec S3 dans les deux familles.
  CE-S0 reste un témoin historique séparé. La CE binaire et la CE trois classes
  n'ont pas les mêmes cibles ni la même normalisation ; les deux sont décrites.
- `common-exposure.csv` : déleveraging a posteriori, séparément par politique.
  Une politique entièrement Flat rend ce contrôle dégénéré, ce qui est signalé.
- `frozen-alert-counterfactuals.csv` : mêmes signaux du Long/Short projetés en
  Long/Flat ou Short/Flat, coûts recalculés, sans feedback des nouveaux états.
- `oppositions.csv` et graphiques : désaccords par ticker et courbes comparées,
  sans mélanger les politiques.

## Limites

Le portefeuille se remet en cash à chaque partition et simulation. L'open J
est un proxy après observation du gap, pas une garantie de prix exécutable.
L'horizon des labels ne force ni durée de détention ni sortie du modèle.
Les frais de prêt, financement et disponibilité broker des shorts ne sont
pas modélisés. Aucun filtre de tickers gagnants, stop automatique, sentiment,
fusion ou nouveau filtre de régime n'est introduit dans ce benchmark.

Le holdout reste fermé. Il faut comparer les neuf couples sans choisir une
seed gagnante et tenir compte des trois périodes communes. Les différences
de capacité, d'exposition et de temps de calcul doivent accompagner les scores.
