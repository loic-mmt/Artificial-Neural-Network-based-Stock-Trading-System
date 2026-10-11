# Gradients temporels des états de trade

Le [plan du benchmark](../plan/trade-state-gradients.md) fixe la comparaison.
`scripts/run_trade_state_gradient_benchmark.py` lance un nouveau run, sans
modifier le protocole des backtests historiques ni ouvrir leur holdout.

## Modules et configuration

- `training/trade_state_gradients.py` : livre à états continus différentiables
  et VJP temporelle tronquée de la tête.
- `training/trade_state_trainer.py` : mode `detached` historique ou `tbptt`,
  puis replay du GRU et une seule update par époque.
- `experiments/trade_state_gradient_benchmark.py` : grille, contrôles,
  comparaison appariée des gradients et runner commun.
- `pipelines/trade_state_gradient_benchmark.py` : CLI et dry-run.

`TradeStateConfig.gradient_mode` accepte `detached` ou `tbptt`. Le fit exige
`tbptt_steps` positif en mode TBPTT ; le mode détaché refuse une fenêtre.
TBPTT est limité à la famille financière dans cette première étude.
Le runner S0-S3 original reste explicitement limité aux états détachés.

## Deux niveaux de backward

1. Le premier parcours conserve les embeddings sans graphe et la trajectoire
   continue avec le livre causal existant. Le panel calcule la loss financière
   globale et sa dérivée par rapport à chaque position, coûts compris.
2. La tête est rejouée chronologiquement sur CPU, avec ses poids gelés et les
   embeddings comme feuilles différentiables. Position précédente, exposition
   et equity/drawdown relient les séances dans une fenêtre TBPTT.
3. À chaque frontière, le livre continue avec ses valeurs courantes, mais
   ses tenseurs continus sont détachés. Les VJP des embeddings et gradients
   de tête s'accumulent, sans update intermédiaire.
4. L'encodeur GRU est rejoué dans les mêmes blocs et avec le même flux dropout
   que le premier parcours. Les VJP sont propagées dans ses paramètres. Une
   seule update AdamW suit l'accumulation et le clipping global.

Il n'y a ni GRU récurrent entre séances de portefeuille, ni graphe de GRU
retenu sur tout l'historique. Les frontières sont comptées en dates globales,
pas en nombre de lignes ticker ou de minibatches.

## Valeurs forward et limites de différentiation

Le livre NumPy de référence conserve les règles exactes d'entrée, resizing,
inversion, cash et liquidation. Les états transformés conservent ses valeurs
forward, tout en utilisant les dérivées des canaux continus.

Les différences numériques de replay sont contrôlées avant d'aligner les
valeurs de position sur celles du premier parcours. Cet alignement évite
qu'un arrondi change un épisode ; ce n'est pas un gradient artificiel du signe.
Une divergence au-delà de la tolérance arrête le fit.

Les signes, indicateurs ouverts, compteurs d'âge/underwater et ancres d'entrée
restent sans gradient. Le maximum d'equity utilise une dérivée par morceaux.
Ce contrat est une propagation temporelle partielle, pas un simulateur
entièrement différentiable de fills, stops ou lots de broker.

La validation et les rollouts outer utilisent le même livre forward sans
graphe, avec remise à zéro entre partitions et entre décodages. Les états
du signe ne sont jamais repris de la trajectoire continue.

## Traçabilité et reprise

Chaque candidat inclut `state_gradient_mode` et `tbptt_steps`. La métadonnée
du run déclare les modes présents et porte le schéma `trade_state_gradients_v1`.
Les diagnostics enregistrent le contrat du gradient, les fenêtres, le maximum
de séances dans un graphe, l'erreur de replay et les normes de gradients.

Les exports habituels et les protections de reprise du
[runner S0-S3](trade-state-benchmark.md) restent utilisés. La synthèse inclut
les contrôles détachés même si leur fenêtre TBPTT est absente. Un nouveau run
est obligatoire après modification des sources ou du protocole.

Les tests vérifient la parité forward, les différences finies sur une petite
trajectoire à événements fixes, le passage du crédit temporel, la troncature,
le dropout, les signaux manquants et la liquidation. S0 TBPTT et S3 avec fenêtre
d'une séance retrouvent l'update détachée à tolérance numérique près.
Une petite grille complète vérifie exports, comparaisons appariées et reprise.

## Vérification locale du 11 octobre 2026

Les 25 nouveaux tests passent ; avec les contrôles S0-S3, labels/losses et
panel post-open, les 85 tests ciblés passent. Le dry-run du véritable univers
confirme 143 tickers, 45 fits et 90 trajectoires.

La suite complète donne 1 648 succès, 11 skips et les 25 échecs FNSPID déjà
signalés dans la vérification du 9 octobre. Aucun module FNSPID n'a été
modifié. La suite n'est donc pas entièrement verte. Les tests du nouveau
trainer sont exécutés sur CPU ; le pilote CUDA et le benchmark financier
complet restent à lancer sur le PC.
