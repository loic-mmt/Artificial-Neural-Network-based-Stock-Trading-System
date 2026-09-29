# Benchmark US GNN et état de marché

## Objectif

Le benchmark vérifie séparément trois hypothèses sur un univers US assez large :

1. les relations entre actions ajoutent une information absente du GRU ;
2. retirer les facteurs SPY et secteur avant de calculer les corrélations produit un graphe plus informatif ;
3. un Transformer compact de marché améliore le choix des features du GRU ou du GNN sans devenir un vote autonome.

Le sentiment news est entièrement exclu de ce protocole.

## Données

`scripts/download_us_market_context.py` télécharge 14 ETF de contexte. SPY, QQQ et IWM décrivent le marché large. Les ETF sectoriels décrivent les facteurs sectoriels. VOX et VNQ remplacent XLC et XLRE afin de disposer d'un historique qui commence avant 2005.

Ces ETF ne sont jamais tradés et ne contribuent jamais directement au PnL. Ils servent uniquement :

- au calcul des rendements résiduels utilisés par le graphe ;
- à la séquence globale reçue par le Transformer de marché.

La sélection `configs/benchmark/stocks_us_gnn_complete_2005.json` contient les 143 actions ayant un calendrier complet depuis le 3 janvier 2005 dans le Parquet actuel. Cette restriction évite d'inventer des prix pour les introductions récentes. Elle ne corrige pas le biais de survivance de l'univers courant.

## Candidats

- `gru` : référence temporelle actuelle.
- `gru_market` : même GRU, features modulées par l'état du Transformer.
- `identity` : MLP de nœud sans message passing, contrôle de capacité du GNN.
- `sector` : GNN avec arêtes entre actions du même secteur.
- `rolling_topk` : GNN avec les cinq corrélations absolues les plus fortes par action.
- `rolling_residual_topk` : même top-k après régression de chaque rendement sur SPY et l'ETF sectoriel correspondant.
- `rolling_topk_market` : graphe brut plus gate guidé par le Transformer.
- `rolling_residual_topk_market` : graphe résiduel plus gate guidé par le Transformer.

Les graphes utilisent uniquement les 252 séances connues jusqu'à J. Ils sont recalculés toutes les 20 séances et réutilisés entre deux recalculs avec leur vraie date de provenance. Le Transformer lit uniquement les ETF, le VIX, la breadth, le rendement équipondéré, la dispersion, le rendement absolu moyen et la volatilité réalisée. Il ne reçoit aucune séquence propre à une action.

## Protocole

Tous les candidats partagent les mêmes actions, dates, labels triple barrier, coûts, folds, seeds, features sélectionnées sur train et objectif combiné PnL/Sharpe de poids 0,25. Le holdout final reste scellé. Le run complet contient 8 candidats multipliés par 3 seeds et 3 folds, soit 72 entraînements.

Télécharger et nettoyer les 143 actions depuis un clone Git, sans dépendre du
dossier local `MT5/` :

```bash
.venv/bin/python scripts/download_us_benchmark_data.py
```

Préparer ou actualiser les ETF :

```bash
.venv/bin/python scripts/download_us_market_context.py --overwrite
```

Lancer le benchmark :

```bash
bash scripts/run_us_gnn_market_benchmark.sh
```

Reprendre un run interrompu en ajoutant `--resume` à la commande Python du script, sans modifier ses autres paramètres.

## Lecture des résultats

Comparer d'abord chaque GNN à `identity`, puis les variantes résiduelles aux variantes brutes. Comparer ensuite chaque candidat suffixé `_market` à sa version sans suffixe. Un gain n'est retenu que s'il améliore le Sharpe régularisé moyen, ne dépend pas d'une seule seed et ne vient pas seulement d'une exposition ou d'un turnover supérieur.
