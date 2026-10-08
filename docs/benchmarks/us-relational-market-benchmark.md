# Benchmark US GNN et état de marché

Statut au 3 octobre 2026 : les 72 entraînements sont terminés. Je garde le
protocole ci-dessous et les conclusions en fin de fiche. Le contrôle
complémentaire est dans [l'analyse à exposition comparable](us-exposure-comparison.md).

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

## Résultats observés

Trois folds externes et seeds 1/7/19, neuf résultats par candidat. Période
externe du 28 mars 2014 au 21 juin 2023 ; holdout à partir du 22 juin 2023,
toujours fermé. Fractions CV initiale 0,50 et validation intérieure 0,20,
gap 5, embargo 0. Les rendements sont des moyennes par run, non annualisées
et non concaténées.

| Candidat | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen | Exposition brute moyenne |
| --- | ---: | ---: | ---: | ---: |
| GRU | 0,9538 | +19,47 % | -4,98 % | 35,17 % |
| GRU + marché | 0,7555 | +5,70 % | -5,22 % | 22,15 % |
| Identité | 0,8951 | +38,41 % | -15,70 % | 69,13 % |
| Secteur | 0,8816 | +32,21 % | -14,14 % | 60,87 % |
| Top-k | 0,8897 | +35,33 % | -15,96 % | 67,38 % |
| Top-k résiduel | 0,8899 | +36,58 % | -16,73 % | 69,04 % |
| Top-k + marché | 0,7941 | +22,88 % | -10,74 % | 43,28 % |
| Top-k résiduel + marché | 0,7314 | +25,81 % | -14,43 % | 55,46 % |
| Buy-and-hold du simulateur | 0,9359 | +55,89 % | -22,36 % | 99,87 % |

Le GRU gagne en Sharpe moyen, pas en PnL. Les branches de nœuds engagent
beaucoup plus de capital et sont plus directionnelles. Identité fait presque
aussi bien que les graphes : le rendement supplémentaire ne prouve pas un
apport des relations entre actions. Les trois gates marché dégradent le Sharpe
moyen de leur branche.

Les 32 features actions réellement retenues sont quatre techniques et 28
sectorielles. Le pool effectif historique est `technical,sector`. Les 21
features de contexte du Transformer sont une autre entrée. Je ne présente
donc pas ce test comme un essai déjà réalisé avec 64 features actions ou avec
une sélection des meilleures 32 parmi elles par le Transformer.

## Décision

La décision historique conserve GRU comme contrôle principal de risque, identité
comme contrôle obligatoire, top-k résiduel comme challenger et top-k brut comme
sensibilité. Le gate marché reste désactivé par défaut. La comparaison 32/64
sur pool commun avec nouvelles baselines est maintenant terminée :
[résultats d'interaction](us-feature-gate-interaction.md), 108/108 tâches vérifiées.
Elle conserve cap 32, sans gain stable des relations ou du gate. Identité32 avec
marché reste un challenger fragile ; le classement du GRU historique ne doit
pas être transféré à cette nouvelle grille. Le [plan d'interaction](../plan/us-feature-gate-interaction.md)
garde les commandes de reproduction.

L'[analyse normalisée](us-exposure-comparison.md) vérifie que le gain GNN ne
vient pas seulement du gross, sans établir un alpha relationnel. Les anciens
checkpoints restent exploratoires et non réutilisables selon les exigences
de provenance P0, malgré leur replay contrôlé. Les prix/contexte proviennent
des fichiers PC figés dans `data/data_pc/` pour cette vérification.

Sources : [rapport original](../../artifacts/comparisons/us-relational-market/01-combined-025/report.json),
[contrôle d'exposition](../../artifacts/comparisons/us-relational-market/03-exposure-comparison-pc/report.json),
[contrat de provenance](../src/us-multimodal-p0.md).
