# Journal des benchmarks GRU optim

Mise à jour : 3 octobre 2026. Je conserve les résultats observés et les
décisions, avec une fiche par benchmark. Les hypothèses et étapes à venir restent dans
[GRU_optim.md](../../plan/GRU_optim.md). Les `report.json` liés ci-dessous font foi pour les
valeurs non reprises ici.

## Règles de lecture

- La métrique de sélection est le **Sharpe régularisé** calculé sur les folds
  externes, après coûts proportionnels configurés à 5 bps. `moyenne ± écart-type`
  résume les 3 seeds `1, 7, 19` et les 3 folds, soit 9 scores par candidat.
- Les folds expansifs se recouvrent dans leur historique d'entraînement. Les 9
  scores ne sont donc pas 9 observations indépendantes et aucun écart ci-dessous
  n'est présenté comme statistiquement significatif.
- Les holdouts sont restés scellés : 10 mai 2022 dans les premières études
  et le pilote post-open, 24 mai 2022 dans les rapports 06 et 08 à 10.
  L'univers de constituants actuels introduit un
  biais de survivance : ces résultats restent provisoires pour une utilisation
  réelle.
- Le run 01 a été exécuté sur CUDA et porte l'empreinte de données CV
  `bcd8a0f3…`. Les runs 02 à 04 portent tous `dce05c66…` et `device=auto`.
  Chaque groupe se compare en interne, mais les écarts absolus entre 01 et les
  suivants ne constituent **pas** une comparaison strictement appariée.
- Les décisions ci-dessous suivent la métrique primaire, pas le rendement seul.
  Une baisse d'exposition peut simultanément réduire rendement et drawdown.

## État des décisions

| Run | Question | Décision |
| --- | --- | --- |
| 01 | Agrégation temporelle et longueur de contexte | Retenir l'attention additive `A3`, loss Sharpe, contexte 60, estimateur ATR comme configuration à recontrôler. |
| 02 | Tête non linéaire ou normalisée après attention | Garder `attention + Linear`. Aucune tête testée ne l'améliore en moyenne. |
| 03 | Abstention par seuil sur l'amplitude de la position | Ne pas activer le gate : gain moyen porté par un seul couple seed-fold. |
| 04 | Normalisation causale par fenêtre | Garder la normalisation globale apprise sur train, sans normaliseur additionnel. |
| 05 | CARA et loss PnL/Sharpe combinée | Garder Sharpe comme référence de risque ; conserver la combinée PnL 0,25 comme candidat offensif à comparer sous limite de drawdown prédéfinie. |
| 06 | Branches et graphes GNN | GRU devant ; Pearson train sans arêtes et identique au contrôle identité. |
| 07 | Fusion fixe de positions | Aucun gain stable, même face au contrôle d'exposition. |
| 08 à 10 | Contexte marché et gates | Petits gains non robustes ; Transformer seul trop risqué. |
| 11 | Gap de l'open | Léger gain Sharpe moyen, sans gain de rendement moyen ni fill réel validé. |
| 12 | StockMixer et attention inter-actions | StockMixer exploratoire, gain non confirmé au stress ; attention sans avantage robuste. |
| 13 | Débruitage | Garder les features brutes. |
| 14 | Reset par attention | Garder la cellule native. |

La configuration courante de référence pour les prochains contrôles est donc
`GRU(attention, head_type="linear", input_normalization="none")`, entraîné avec
la loss Sharpe et un contexte de 60 jours. Son score sur les runs 02 et 04 est
**0,7509 ± 0,0705**, minimum **0,6554**. La répétition exacte de ses 9 scores
dans ces deux runs vérifie la cohérence du témoin.

## Fiches détaillées

- [01. Pooling temporel et contexte](01-temporal-pooling.md).
- [02. Tête après attention](02-attention-head.md).
- [03. Gate de signal](03-signal-gate.md).
- [04. Normalisation](04-normalization.md).
- [05. CARA et combinaison](05-cara-combined.md).
- [06. Contrôles de graphe](06-graph-controls.md).
- [07. Fusion fixe](07-fixed-fusion.md).
- [08 à 10. Contexte marché](08-10-market-context.md).
- [11. Après l'open](11-post-open.md).
- [12. StockMixer](12-stock-mixer.md).
- [13. Débruitage](13-denoising.md).
- [14. Reset attention](14-attention-reset.md).

Les dossiers `11-post-open-smoke*` sont des essais techniques, pas des
benchmarks de performance à classer. Le [résumé général](../../benchmark-summary.md)
relie cette série aux tests MT5, trading et US.

## Pour ajouter le prochain benchmark

Je note la question, le statut et le nombre de runs réellement terminés,
puis les données et leur empreinte, dates, seeds, coûts et convention
d'exécution. J'ajoute les métriques du candidat et de son témoin, les gains
appariés, exposition et turnover, les limites, la décision et les liens
vers les rapports. Une hypothèse ou un essai technique ne devient pas un
résultat de performance.
