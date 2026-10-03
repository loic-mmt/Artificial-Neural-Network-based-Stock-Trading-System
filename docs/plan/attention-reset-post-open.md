# Benchmark 14 : cellule GRU avec reset par attention

Statut : terminé, 27 résultats. Je conserve ici le protocole.
Les [résultats et la décision](../benchmarks/gru-optim/14-attention-reset.md)
sont séparés du plan.

Ce benchmark isole la cellule récurrente. Il ne reproduit pas le modèle MCI-GRU
complet : aucun GAT, aucun état latent de marché, aucune fusion, aucun débruiteur.
Les features restent celles de J−1 plus `night_pct(J)`, et le GRU garde le même
pooling temporel attention et la même tête linéaire que le témoin post-open.

| Candidat | Rôle |
| --- | --- |
| `native` (M0) | `nn.GRU` inchangé |
| `manual` (M1) | Mêmes équations PyTorch et mêmes poids initiaux, déroulés explicitement |
| `attention_reset` (M2) | Même déroulement explicite, reset remplacé par attention sur les canaux |

M1 est indispensable : M2 doit gagner face au contrôle manuel, pas uniquement
face à M0, sinon les effets d'implémentation et d'optimisation restent confondus.
M0 et M1 ont exactement le même nombre de paramètres. Les paramètres de reset
inutilisés sont supprimés dans M2 ; ses projections Q/K/V ajoutent une capacité
différente, dont le nombre de paramètres est enregistré. Un gain M2 éventuel
ne permet donc pas d'attribuer à lui seul le bénéfice à l'attention plutôt qu'à
sa capacité supplémentaire.

Avec les 32 features et le hidden size 32 du témoin, M0/M1 comptent 7 523
paramètres et M2 en compte 8 579 (environ +14 %).

## Adaptation explicite des équations du papier

Le texte de MCI-GRU définit une query issue de `h(t−1)` et une seule key/value
issue de `x(t)`. Un softmax temporel sur une seule clé donne toujours 1 et ne
peut pas sélectionner l'information. Les dimensions du texte sont aussi ambiguës.
Nous ne prétendons pas résoudre cela en reproduisant fidèlement le papier.

L'adaptation implémentée utilise des scores élément par élément, normalisés
sur les **H canaux cachés**, jamais sur le batch ni sur des dates futures :

```text
x_t : [B,F] ; h_prev : [B,H]
q = Linear(H,H)(h_prev)       : [B,H]
k = Linear(F,H)(x_t)          : [B,H]
v = Linear(F,H)(x_t)          : [B,H]
alpha = softmax(q * k / sqrt(H), dim=hidden)
r = sigmoid(H * alpha * v)    : [B,H]
```

Le facteur H compense la moyenne 1/H des poids softmax ; le sigmoid maintient
un reset borné entre 0 et 1. Ces deux choix sont des adaptations locales, pas
des opérations démontrées par le papier. C'est une attention sur les canaux,
pas une sélection explicite d'anciens pas de temps ni une attention multi-head.

La mise à jour suit les conventions PyTorch pour garder M0/M1 équivalents :

```text
z = sigmoid(x_z + h_z)
n = tanh(x_n + r * h_n)
h = (1 - z) * n + z * h_prev
```

Les projections `x_z/x_n` et `h_z/h_n`, leurs biais et les poids pooling/tête
commencent aux mêmes valeurs dans les trois modèles appariés. La cellule
personnalisée ne supporte ici qu'une couche unidirectionnelle sans dropout
inter-couches, ce qui correspond au témoin historique.

## Protocole et lancement

Train/validation/folds, scalers train-only, loss Sharpe de trajectoire complète,
coûts de 5 bps et sensibilité OHLC restent ceux du pilote 11. Les prix cibles
ne changent pas. Le holdout final à partir du 10 mai 2022 reste fermé. L'open
ajusté J vers J+1 reste un proxy d'exécution, pas un fill observé après décision.

Le déroulement Python sur 60 pas perd les kernels GRU optimisés : le temps
d'apprentissage peut augmenter sensiblement, notamment sur MPS. Le device
`auto` garde le choix habituel ; CPU et CUDA sont aussi disponibles. Le code
ne désactive pas le déterminisme.

27 entraînements, soit 3 variantes × 3 folds × 3 seeds :

```bash
.venv/bin/python scripts/run_attention_reset_ablation.py \
  --data data/processed/cac40_daily_clean.parquet \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --reference-cv artifacts/comparisons/gru-optim/04-causal-normalization-60-atr \
  --seeds 1,7,19 \
  --folds 0,1,2 \
  --draws 250 \
  --device auto \
  --output-dir artifacts/comparisons/gru-optim/14-attention-reset-post-open
```

Après interruption, même commande avec `--resume`. La reprise se fait par
candidat terminé. Ne pas utiliser `--max-epochs 1` pour conclure : cette option
sert aux essais techniques dans un autre dossier. Les sorties comprennent
positions, checkpoints, trajectoires de loss train/validation intérieure,
meilleure epoch, temps d'entraînement et nombre de paramètres.

Lecture : vérifier d'abord M1 face à M0, puis M2 face à M1 et M0. De petits
écarts numériques de kernels peuvent diverger au cours d'un long entraînement,
même avec des équations et paramètres identiques. Comparer les résultats
appariés et leur stabilité, pas seulement la meilleure seed. Un gain doit
survivre en rendement net, risque et prix défavorables et rester pertinent au
regard du coût de calcul.
