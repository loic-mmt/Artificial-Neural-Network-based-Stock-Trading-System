# GRU avec état de trade détaché

Le [plan S0-S3](../plan/trade-state-overnight.md) fixe la grille. Le script `scripts/run_trade_state_benchmark.py` lance un runner isolé, sans remplacer les workflows historiques.

## Modules

- `training/trade_state.py` : livre overnight causal, observations, activation S0-S3, transformations fixes.
- `models/neural/trade_state_gru.py` : encodeur GRU existant et tête linéaire sur embedding + dix états courants.
- `training/trade_state_trainer.py` : simulations fermées, caches détachés, gradients et sélection inner.
- `experiments/trade_state_benchmark.py` : grille, partitions communes, exports, comparaisons et reprise.

## Sens des états

Le rendement d'entrée est `signe(position) × (open_ajusté_courant / open_ajusté_entrée - 1)`. Il décrit le prix depuis le début de l'épisode, pas un PnL pondéré par les quantités successives. Le giveback est l'écart avec son meilleur rendement, au minimum zéro. Le compteur underwater mesure les séances consécutives avec ce rendement négatif.

Les expositions brute et nette décrivent les précédents targets `q/N`. La fraction cash vaut `1 - exposition_brute` ; ce n'est pas la marge disponible d'un broker pour ses shorts. Le drawdown utilise la courbe nette, coûts compris, marquée jusqu'à l'open courant. Ces conventions correspondent au moteur existant à slots fixes.

Les âges et le compteur underwater utilisent `log1p(x)/log1p(252)`. Rendement et giveback utilisent `tanh(x/0.1)`. Les échelles sont fixes avant les tests, modifiables via `--state-age-scale` et `--state-return-scale`. Les groupes inactifs sont mis à zéro après transformation, sans changement de dimension.

Un état brut est calculé pour chaque asset avant toute décision de la séance. Les décisions sont exécutées ensemble, puis le livre avance vers l'open suivant. Ainsi ni le prochain rendement ni l'ordre d'itération des tickers ne peuvent modifier l'état utilisé pour la séance courante.

## Entraînement et mémoire

Chaque parcours repart en cash. Les embeddings marché sont calculés en blocs, sans gradients, et seuls `[lignes, hidden_size]` sont conservés. La petite tête linéaire déterministe est évaluée sur CPU pendant la simulation chronologique, même si l'encodeur est sur GPU. Cela évite un lancement de GRU et un transfert GPU par séance. De petits écarts d'arrondi CPU/GPU restent possibles ; aucune équivalence bit à bit intermatériel n'est revendiquée.

Après calcul de la loss complète, les mêmes blocs d'encodeur sont rejoués avec les états cachés et détachés. Le flux dropout est réinitialisé à l'identique. Les gradients s'accumulent, sont clippés, puis une seule update AdamW est effectuée. Les positions et états sont reconstruits avec les nouveaux poids pour mesurer TRAIN et inner. Il n'y a pas de graphe de gradients sur toute la durée du portefeuille.

La CE utilise les poids de classes TRAIN et ignore les labels inconnus. La financière ne consomme pas les labels. Les masques de labels n'enlèvent aucune séance financière. Les décisions terminales restent visibles comme probabilités diagnostiques, mais le moteur interdit une nouvelle entrée le dernier jour et liquide les positions.

Il faut utiliser le trainer et les rollouts dédiés : les appels génériques `fit` et `predict_proba` sont refusés, car ils n'ont pas le livre de positions nécessaire.

## Exports

À la racine : `metadata.json`, `report.json`, `results.csv`, `summary.csv`, `classification.csv`, `common-exposure.csv`, `oppositions.csv` et détails d'oppositions. Les contrôles d'exposition égalisent une moyenne, pas le timing ni le risque.

Chaque fit sauvegarde son `checkpoint.pt`, `fit.json`, les diagnostics et les courbes d'apprentissage. Chaque décodage outer sauvegarde ses probabilités, ses inputs d'état bruts et transformés, les positions exécutées, le portefeuille, les KPI par ticker et les épisodes de trade. Les tables partagent des clés exactes date/ticker. Les inputs d'état sont observés avant la décision et peuvent donc différer de la position exécutée dans la table des positions.

Les fichiers `overnight-continuous-*` et `overnight-sign-*` proviennent de simulations indépendantes. Les courbes peuvent être désactivées avec `--no-plots`. `complete.json` protège les fits terminés par taille/SHA-256 ; `--resume` réutilise ces fits, pas les poids d'une époque interrompue.

## Pilote

```shell
.venv/bin/python -u scripts/run_trade_state_benchmark.py \
  --tickers AAPL MSFT \
  --start 2019-01-02 --end 2021-12-31 \
  --folds 0 --seeds 1 --epochs 2 --epoch-multiplier 1 \
  --device cpu --no-plots \
  --output-dir artifacts/comparisons/trade-state-pilot
```

Ce pilote vérifie les huit configurations et seize trajectoires, pas leur convergence ni leur rentabilité. Aucune donnée supplémentaire ni nouvelle dépendance n'est requise.

## Vérification locale du 9 octobre 2026

Les tests ciblés des états, du trainer, du benchmark précédent, du moteur post-open et de l'architecture passent : **88 tests**. Ils couvrent notamment le resizing, l'inversion, les quotes finales manquantes, le cutoff causal, les simulations indépendantes par décodage, le détachement, le replay dropout et la reprise avec contrôle d'intégrité.

Le pilote AAPL/MSFT 2019-2021 a terminé huit fits de deux époques et seize trajectoires, avec le même nombre de paramètres entre variantes. La reprise n'a pas réentraîné les fits terminés. Un deuxième pilote S3 financier a vérifié les exports graphiques. Ces runs de contrôle ne permettent aucune conclusion de performance.

La suite complète lancée pendant cette vérification a donné 1 620 succès, 11 skips et 25 échecs dans les tests FNSPID existants. Elle n'est donc pas entièrement verte. Les trois derniers tests ajoutés ensuite sont inclus dans les 88 succès ciblés. Aucun code FNSPID n'a été modifié ; le benchmark S0-S3 ne consomme pas de news. CUDA reste à vérifier sur le PC GPU.
