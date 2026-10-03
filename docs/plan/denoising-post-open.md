# Benchmark 13 : débruitage avant le GRU post-open

Statut : terminé, 27 résultats GRU. Je conserve ici le protocole.
Les [résultats et la décision](../benchmarks/gru-optim/13-denoising.md)
sont séparés du plan.

Trois variantes appariées, sans StockMixer ni GNN :

| Candidat | Prétraitement |
| --- | --- |
| `raw` (D0) | Aucun, témoin GRU post-open avec gap |
| `dae` (D1) | Petit autoencodeur débruiteur, puis GRU inchangé |
| `attention_dae` (D2) | Même autoencodeur avec pondération softmax du latent, puis GRU inchangé |

Il s'agit d'une adaptation légère du papier Attention-Based Autoencoder, pas
d'une reproduction d'AGRUA. L'encodeur pointwise est un MLP 64 unités, puis un
latent de 8 unités au maximum (au plus la moitié des features continues).
Le décodeur reconstruit les mêmes colonnes via 64 unités et une sortie linéaire.
La sortie n'est pas sigmoid : nos features standardisées sont signées et non
bornées à [0, 1]. Le GRU garde ses dimensions, son architecture et sa loss Sharpe.

## Causalité et apprentissage

- Les mêmes sélections de features, médianes de remplissage et normalisations
  train-only que le pilote 11 sont utilisées pour les trois variantes.
- Le débruiteur apprend uniquement sur les dernières observations des fenêtres
  du train purgé. Une observation ticker/date n'est ainsi pas répétée jusqu'à
  60 fois à cause du recouvrement des fenêtres.
- Un bruit gaussien d'écart-type 0,1 est ajouté aux entrées continues
  standardisées pendant cet apprentissage uniquement. La cible est l'observation
  originale sans corruption artificielle, pas un signal financier « propre » connu.
- La validation intérieure, sans bruit ajouté, sélectionne le checkpoint par MSE
  de reconstruction. Elle ne participe pas aux mises à jour des poids.
- Le débruiteur est ensuite gelé. Il transforme chaque pas indépendamment :
  aucune lecture des pas futurs, aucune interaction entre tickers, aucun
  ajustement sur le fold externe. Le GRU est réinitialisé avec la même seed
  entre variantes et entraîné sur les fenêtres transformées.
- Calendrier, secteurs one-hot, régimes et indicateurs de disponibilité restent
  inchangés. Les rendements sectoriels restent des features continues.
  `night_pct` est inclus dans le débruitage, car c'est une feature continue.
- Seules les features sont reconstruites. Les prix cibles, coûts, positions de
  référence, splits et rendements évalués restent ceux des données originales.

Les features disponibles sont celles de J−1 plus le gap d'ouverture J. L'open
ajusté est toujours un proxy d'exécution, pas un fill observé après décision.
Les 250 tirages OHLC restent une sensibilité mathématique de prix. Le holdout
final à partir du 10 mai 2022 reste fermé.

## Lancement

27 entraînements GRU (3 variantes × 3 folds × 3 seeds), plus 18 préentraînements
de petits autoencodeurs. Le plafond du débruiteur est de 30 epochs avec patience
5 ; le budget GRU est celui du témoin historique.

```bash
.venv/bin/python scripts/run_denoising_ablation.py \
  --data data/processed/cac40_daily_clean.parquet \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --reference-cv artifacts/comparisons/gru-optim/04-causal-normalization-60-atr \
  --seeds 1,7,19 \
  --folds 0,1,2 \
  --denoiser-epochs 30 \
  --noise-std 0.1 \
  --draws 250 \
  --device auto \
  --output-dir artifacts/comparisons/gru-optim/13-denoising-post-open
```

Relancer à l'identique avec `--resume` après une interruption. La reprise se fait
par candidat terminé, pas au milieu d'un entraînement. Les options doivent
rester identiques. Pour un essai technique, utiliser un autre dossier avec
`--folds 0 --seeds 1 --max-epochs 1 --denoiser-epochs 1 --draws 2`.

Les sorties comprennent les positions, checkpoints GRU/débruiteur, colonnes
transformées ou préservées, MSE de validation, nombre d'observations de fit,
temps d'apprentissage, métriques financières et stress OHLC.

Le critère de décision est le gain apparié de rendement net et de risque face à
D0, sa stabilité entre folds/seeds et sa résistance aux prix défavorables. Une
MSE faible ne suffit pas : la reconstruction peut effacer un signal utile.
Examiner aussi exposition et turnover pour ne pas confondre débruitage et
simple changement d'amplitude des positions.
