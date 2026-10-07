# Résumé des benchmarks

Mise à jour : 7 octobre 2026. Je garde ici les grandes lignes et les décisions. Chaque fiche liée contient les résultats détaillés, le protocole, les limites et les sources.

## Ce que je retiens

Une bonne accuracy ne garantit pas un bon PnL. Le backtest des labels utilise une information future absente des entrées du modèle : il mesure un potentiel, pas une performance atteignable.

Ma référence GRU reste compacte : une couche, hidden size 32, attention temporelle, tête linéaire, contexte 60, sélection train-only de 32 features et weight decay `1e-5`. Je garde Sharpe comme objectif de risque et la combinaison PnL/Sharpe à 0,25 comme variante plus offensive. Les 32 features d'entrée ne sont pas les 32 unités cachées.

Sur l'US, le GRU reste le contrôle de risque, `identity` le contrôle obligatoire et le GNN résiduel une piste exploratoire. Le gain relationnel n'est pas démontré. Le gate Transformer actuel ne devient pas le réglage par défaut. Je teste ensuite son interaction avec 32/64 features sur un pool commun réentraîné, avec `identity_market` aux deux caps pour contrôler gate et capacité sans relations. Les résultats restent en attente.

Sur FNSPID, l'inventaire des 143 actions retient 53 titres selon la disponibilité du premier TRAIN, sans classement de PnL. Les 81 entraînements et cinq permutations ne démontrent pas d'avantage robuste de la polarité FinBERT. Le neutralisé améliore la moyenne, mais surtout grâce à deux runs seed 7 et à des changements d'exposition. Je ne le retiens pas comme vainqueur validé ; le holdout reste fermé et la disponibilité historique des news n'est pas prouvée.

Pour MT5, décodage et exposition changent fortement les résultats. Le seuil de confiance est le meilleur décodage du test US disponible. Réentraîner toutes les 40 séances gagne la grille brute, avec plus d'exposition que 10 ou 20. Ce n'est pas une fréquence universelle. Je garde l'allocation égale entre actifs non FLAT, sans sizing selon les performances passées.

## Limites communes

Les anciennes métriques de classification et les rendements continus nets ne sont pas interchangeables. Les premiers backtests ne facturaient pas les 5 bps du labeling. Certaines séries ont aussi des datasets, dates réservées et conventions d'exécution différents.

Les seeds partagent les dates des folds : neuf résultats ne font pas neuf périodes indépendantes. Les univers sont survivants et les secteurs ne sont pas historiquement point-in-time. Les holdouts CAC40 réservés en 2022 et US réservé en 2023 restent fermés dans les études CV citées. La période MT5 2026 a déjà été examinée et n'est plus un holdout intact.

Exposition brute égale ne signifie pas même exposition nette, beta ou risque. Les opens et stress OHLC sont des simulations, pas des fills MT5 observés. La disponibilité historique des news et l'exécution réelle restent à valider.

## Labels et apprentissage

Protocole : [learning improvements](benchmarks/learning/README.md).

| Benchmark | Conclusion | Détail |
| --- | --- | --- |
| Comparaison initiale M0 à M3 | Triple Barrier devient la piste principale. Le décalage labels parfaits/modèle motive le travail sur l'apprentissage. | [Labels et modèles](benchmarks/label-methods.md) |
| 01. Baseline | GRU gagne en macro-F1. Pas de preuve de rentabilité robuste. | [01](benchmarks/learning/01-baseline.md) |
| 02. Estimateur | `rolling_std` en classification initiale ; ATR challenger financier. | [02](benchmarks/learning/02-volatility-estimator.md) |
| 03. Barrières | `0.75/0.75`. Les branches macro-F1/PnL sont identiques, pas deux réplications. | [03](benchmarks/learning/03-profit-stop-barriers.md) |
| 04. Événements | CUSUM 0,5 plutôt que tous les jours. | [04](benchmarks/learning/04-event-filter.md) |
| 05. Entre événements | `hold` conservé. `carry` apprend mal dans ce protocole. | [05](benchmarks/learning/05-between-events.md) |
| 06. Horizon | 10 observations plutôt que 5 ou 20. | [06](benchmarks/learning/06-horizon.md) |
| 07. FracDiff | Pas de gain financier régulier. Désactivé. | [07](benchmarks/learning/07-fracdiff.md) |
| 08. Sample weighting | Pas de gain stable. `none`. | [08](benchmarks/learning/08-sample-weighting.md) |
| 09. Features | Technique + marché + secteur selon macro-F1 ; marché seul challenger financier. | [09](benchmarks/learning/09-feature-families.md) |
| 10. Données externes | Sauté faute de corpus historique PIT. | [10](benchmarks/learning/10-external-features.md) |
| 11. Loss | Sharpe pour le risque, PnL plus offensif. Utiliser le run corrigé `v2`. | [11](benchmarks/learning/11-loss-objectives.md) |
| 12. Coûts | Modèles réentraînés à chaque coût. Les frais ne créent pas mécaniquement du rendement. | [12](benchmarks/learning/12-transaction-costs.md) |
| 13. Overfitting control | Activé pour stabiliser et réduire le risque. | [13](benchmarks/learning/13-overfitting-control.md) |
| 14. Sélection | Cap 32, corrélation maximale 0,95 : compromis stabilité/coût. | [14](benchmarks/learning/14-feature-selection.md) |
| 15. Gates de stabilité | Tests logiciels validés, pas benchmark financier. | [15](benchmarks/learning/15-stability-gates.md) |
| 16. Régularisation | Une couche, weight decay `1e-5`. Deux couches restent une sensibilité. | [16](benchmarks/learning/16-gru-regularization.md) |
| 17. Folds | 3 en référence, 5/10 en diagnostic. Fenêtres courtes plus instables. | [17](benchmarks/learning/17-cv-folds.md) |
| 18. Gap | 5 conservé, avec purge des intervalles d'événements. | [18](benchmarks/learning/18-cv-gap.md) |
| 19. Embargo | 0. Passer à 5 ne retire rien de plus dans ce protocole expanding. | [19](benchmarks/learning/19-cv-embargo.md) |
| 20. Historique CV | Fraction initiale 0,50, validation intérieure 0,20. | [20](benchmarks/learning/20-cv-history.md) |
| 21. Architectures finales | Sauté. Les losses financières n'ont pas été revalidées sur toutes les architectures. | [21](benchmarks/learning/21-architectures.md) |
| 22. Marché et secteur | Les trois familles restent la référence Sharpe de cette série. | [22](benchmarks/learning/22-market-sector.md) |
| 23. Holdout final | Non lancé, à garder fermé jusqu'au gel du protocole. | [23](benchmarks/learning/23-final-holdout.md) |

## Optimisation GRU

Protocole : [GRU optim](benchmarks/gru-optim/README.md). Numérotation indépendante de la série précédente.

| Benchmark | Conclusion | Détail |
| --- | --- | --- |
| 01. Pooling et contexte | Attention additive, contexte 60, ATR retenus dans la grille de 1 080 entraînements. | [01](benchmarks/gru-optim/01-temporal-pooling.md) |
| 02. Tête | La tête linéaire reste devant MLP et LayerNorm en moyenne. | [02](benchmarks/gru-optim/02-attention-head.md) |
| 03. Gate de signal | Gain porté par une seule paire. Désactivé. Ce n'est pas une calibration des classes. | [03](benchmarks/gru-optim/03-signal-gate.md) |
| 04. Normalisation | Garder le scaler train-only, sans normalisation additionnelle par fenêtre. | [04](benchmarks/gru-optim/04-normalization.md) |
| 05. CARA et combinaison | CARA ne convainc pas. La combinaison 0,25 augmente rendement et drawdown. | [05](benchmarks/gru-optim/05-cara-combined.md) |
| 06. GNN sur dix actions | GRU reste devant. Pearson train est sans arêtes et équivalent à identité. | [06](benchmarks/gru-optim/06-graph-controls.md) |
| 07. Fusion fixe | Pas de gain stable face au GRU ou à son contrôle d'exposition. Pas de fusion apprise ici. | [07](benchmarks/gru-optim/07-fixed-fusion.md) |
| 08 à 10. Marché | Concaténation à surveiller. Gate Transformer peu utile, Transformer seul moins robuste. | [08 à 10](benchmarks/gru-optim/08-10-market-context.md) |
| 11. Après l'open | Le gap améliore légèrement le Sharpe moyen, pas le rendement moyen. Intraday à valider. | [11](benchmarks/gru-optim/11-post-open.md) |
| 12. StockMixer | Petit gain Sharpe, non confirmé sous stress défavorable. Attention inter-actions sans avantage moyen. | [12](benchmarks/gru-optim/12-stock-mixer.md) |
| 13. Débruitage | Les deux autoencodeurs restent derrière le GRU brut. Désactivés. | [13](benchmarks/gru-optim/13-denoising.md) |
| 14. Reset attention | Pas de gain face au natif ou au contrôle manuel. Cellule native conservée. | [14](benchmarks/gru-optim/14-attention-reset.md) |

## US, GNN et Transformer marché

| Benchmark | Conclusion | Détail |
| --- | --- | --- |
| 72 entraînements, 143 actions | Plus de rendement brut GNN, mais plus d'exposition. GRU meilleur Sharpe moyen. | [US](benchmarks/us-relational-market-benchmark.md) |
| Exposition égale | Le gain de rendement GNN persiste, mais identité fait presque aussi bien. Pas d'alpha relationnel démontré. | [Comparaison](benchmarks/us-exposure-comparison.md) |
| Interaction features/gate | Prêt, sans résultats locaux. 12 variantes, 108 entraînements maximum ; `identity` et `identity_market` aux caps 32/64, sur le même pool réentraîné. | [Statut](benchmarks/us-feature-gate-interaction.md) |

## Sentiment news

| Benchmark | Conclusion | Détail |
| --- | --- | --- |
| Pilote RSS et FinBERT | Collecte, scoring et exports vérifiés sur 409 articles ; validation technique, sans conclusion de PnL. | [RSS](benchmarks/news-sentiment-rss-pilot.md) |
| FNSPID, 53 actions et cinq permutations | 81 entraînements terminés. Les cinq mélanges dépassent l'original en Sharpe moyen ; avantage du neutralisé fragile, positif seulement sur 4/9 couples. Pas d'apport robuste de polarité démontré. | [FNSPID](benchmarks/news-sentiment-fnspid.md) |

## MT5 et portefeuille

| Test | Conclusion | Détail |
| --- | --- | --- |
| France, statique puis walk-forward | Réentraînement par blocs de 20 séances, pas quotidien. Le signe améliore ce test France. | [France](benchmarks/mt5-france.md) |
| US, walk-forward | Le signe à 100 % reste perdant : l'écart n'est pas seulement dû aux quantités. | [US](benchmarks/mt5-us.md) |
| Décodages | Seuil de confiance sélectionné sur validation meilleur dans ce test. | [Décodages](benchmarks/mt5-decoders.md) |
| Fréquences 5/10/20/40/60 | 40 meilleur score brut, 10 drawdown plus faible, 60 sans confirmation. Une seule période/seed. | [Fréquences](benchmarks/mt5-retraining.md) |
| Régime et shorts | Gain surtout dû au retrait des shorts en période haussière. Régime baissier non validé. | [Régime](benchmarks/market-regime-short-filter.md) |
| Tickers | Erreurs de direction/timing, pas seulement coûts ou volatilité. Pas de blacklist a posteriori. | [Diagnostic](benchmarks/us-ticker-diagnostics.md) |
| Allocation | Égale entre actifs non FLAT. Sizing par performances et inversion après pertes non validés. | [Allocation](benchmarks/mt5-allocation.md) |
| Live | Publication journalière disponible. Filtrer les 106 Sharpes positifs reste exploratoire. | [Live](benchmarks/mt5-live.md) |

## Trading

| Test | Conclusion | Détail |
| --- | --- | --- |
| Premier replay OHLC | Le paquet de règles réduit presque toute l'exposition. Son petit drawdown n'est pas un gain de qualité du signal. | [Replay](benchmarks/trading-initial-replay.md) |
| TP, durée, SL et trailing | Certaines règles bloquent presque toute réentrée. Les modifier change aussi exposition et frais. | [Sorties](benchmarks/trading-exit-ablation.md) |
| Ornstein-Uhlenbeck | Pas de gain robuste, peu de fits admissibles, fragilité aux coûts. Désactivé. | [OU](benchmarks/trading-ou-validation.md) |

Le trading reste une couche optionnelle, séparée du modèle. Après tout changement de décodage ou de sizing, je recalcule coûts et turnover.

Les prochaines étapes sont dans [plan](plan/README.md), le fonctionnement du code dans [src](src/README.md). Les sources scientifiques sont dans [papers](papers/).
