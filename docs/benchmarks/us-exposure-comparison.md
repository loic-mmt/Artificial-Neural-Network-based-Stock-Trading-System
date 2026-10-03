# Benchmark US : résultats à exposition comparable

## Synthèse

Les GNN conservent davantage de rendement que le GRU à exposition brute égale. Le GRU conserve cependant le meilleur Sharpe net moyen et le plus faible drawdown moyen. L'écart de rendement n'était donc pas uniquement dû à l'exposition brute.

72 checkpoints rejoués, sans réentraînement, sur les données PC fournies. Trois folds, trois seeds, 143 actions. Les folds externes couvrent du 28 mars 2014 au 21 juin 2023 ; le holdout final reste fermé. Chaque métrique financière sauvegardée a été contrôlée avant la normalisation.

### Exposition moyenne commune

Les positions sont réduites par un facteur constant propre à chaque modèle, fold et seed. La cible est la plus faible exposition moyenne du groupe. Elle est identique entre modèles dans chaque fold/seed, mais varie entre ces neuf cas de 5,16 % à 74,24 %, pour une moyenne de 14,74 %. Ce calcul utilise les prédictions de tout le fold : diagnostic ex post, pas règle de trading déployable.

| Modèle | Rendement net moyen par fold | Sharpe net moyen | Drawdown moyen |
| --- | ---: | ---: | ---: |
| GRU | +6,21 % | 0,972 | -1,92 % |
| GRU + marché | +2,61 % | 0,780 | -3,64 % |
| GNN identité, sans relations | +8,69 % | 0,897 | -3,29 % |
| GNN sectoriel | +8,65 % | 0,885 | -3,37 % |
| GNN top-k | +8,49 % | 0,892 | -3,38 % |
| GNN top-k résiduel | +8,83 % | 0,892 | -3,37 % |
| GNN top-k + marché | +7,05 % | 0,796 | -4,22 % |
| GNN top-k résiduel + marché | +6,40 % | 0,733 | -4,26 % |
| Référence buy-and-hold du simulateur | +9,26 % | 0,936 | -3,27 % |

Les rendements ne sont pas annualisés ni issus d'une courbe concaténée. Chaque drawdown du tableau est la moyenne des drawdowns maximaux des neuf runs. Le pire cas GRU est -7,88 %, contre -15,19 % pour le GNN résiduel.

Le GNN résiduel reste presque entièrement long : exposition nette moyenne 14,61 %, contre 9,95 % pour le GRU. Sa volatilité nette annualisée moyenne est 2,60 %, contre 1,63 %. Exposition brute égale ne signifie donc ni même risque, ni même exposition directionnelle, ni alpha démontré.

### Contrôle séance par séance

Une seconde normalisation réduit chaque modèle à la plus faible exposition brute prédite pour la séance. Elle ne dépend pas des rendements futurs, mais change le sizing ; turnover et coûts sont recalculés. Exposition moyenne commune : 14,06 %.

- GRU : +5,43 %, Sharpe 0,932, drawdown moyen -1,89 %.
- GNN identité : +7,51 %, Sharpe 0,875, drawdown moyen -3,12 %.
- GNN résiduel : +7,82 %, Sharpe 0,880, drawdown moyen -3,18 %.
- Buy-and-hold redimensionné : +8,00 %, Sharpe 0,891, drawdown moyen -3,24 %.

Le constat reste similaire. La sensibilité par paires contre le GRU confirme aussi le gain de rendement du résiduel : +3,82 points en moyenne, positif dans les trois moyennes de folds. Les cibles diffèrent entre ces paires : leurs rendements ne servent pas à classer tous les modèles entre eux.

## Ce que je garde pour les prochains benchmarks

1. Conserver le GRU comme contrôle principal de Sharpe et de risque, sans le présenter comme gagnant du PnL.
2. Conserver `identity` comme contrôle obligatoire du GNN. Le résiduel ne gagne que +0,14 point de rendement contre lui en exposition moyenne commune, avec un Sharpe inférieur. En exposition quotidienne commune : +0,31 point et +0,005 de Sharpe ; le gain de Sharpe n'est positif que dans un fold sur trois. Pas de gain relationnel stable démontré.
3. Retenir `rolling_residual_topk` comme candidat exploratoire pour la suite, pas comme vainqueur confirmé. Garder `rolling_topk` comme sensibilité, sans grosse grille de profondeurs à ce stade.
4. Donner priorité au test 32 contre 64 features, avec `hidden_size=32` inchangé et contrôle identité. Les 32 features actuelles comprennent seulement quatre features techniques et 28 sectorielles : vérifier ce que le cap 64 ajoute réellement.
5. Ne pas retenir le gate Transformer actuel comme configuration par défaut. Les variantes marché dégradent le Sharpe moyen de leurs branches dans les trois contrôles quotidiens : GRU, top-k et résiduel. Une interaction avec davantage de features reste une hypothèse à tester séparément, pas un gain acquis.
6. Pour la fusion future, comparer GRU + GNN résiduel à GRU + identité et à un mélange de positions GRU + buy-and-hold. Cela distingue un apport relationnel d'un simple ajout d'exposition longue. Ces mélanges ne sont pas implémentés ou évalués par ce diagnostic.

Ces choix restent exploratoires. Les neuf couples ne constituent pas neuf périodes indépendantes : trois seeds partagent les dates de chaque fold. L'univers est survivant et les secteurs ne sont pas point-in-time. Aucun modèle ne dépasse le buy-and-hold redimensionné en rendement moyen dans les deux normalisations.

## Vérification et limites de provenance

Les empreintes des prix et du contexte ETF/VIX PC correspondent exactement aux données sources. Le contexte de marché dérivé n'a pas la même empreinte ; cette différence est explicitement auditée. Les scalers marché des trois folds sont identiques aux checkpoints. Un ordre ex aequo de deux colonnes sectorielles au fold 1 a été restauré depuis les checkpoints, sans changer l'ensemble des features.

Les 144 ensembles de métriques financières et les 72 classifications externes respectent les tolérances originales (`atol=rtol=5e-4`). Écarts maximaux financiers, validation interne et externe confondues : rendement 0,00785 point de pourcentage, PnL 0,78521 pour un capital de 10 000, Sharpe régularisé 0,0000884. Le replay n'est pas bit-identique et ne certifie pas une provenance historique absente ; il reste non réutilisable au sens strict de P0. Les données et rapports sources n'ont pas été modifiés. Suite logicielle : 918 tests passent, 12 ignorés.

## Détail technique des trois méthodes

No training. Final holdout remains sealed.

Mean matching is ex post. Daily matching changes sizing and recomputes costs.

| Method | Candidate | Mean exposure | Net return | Net Sharpe | Regularized Sharpe | Mean drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | gru | 35.17% | +19.47% | 0.9719 | 0.9538 | -4.98% |
| raw | gru_market | 22.15% | +5.70% | 0.7802 | 0.7555 | -5.22% |
| raw | identity | 69.13% | +38.41% | 0.8967 | 0.8951 | -15.70% |
| raw | sector | 60.87% | +32.21% | 0.8849 | 0.8816 | -14.14% |
| raw | rolling_topk | 67.38% | +35.33% | 0.8923 | 0.8897 | -15.96% |
| raw | rolling_residual_topk | 69.04% | +36.58% | 0.8924 | 0.8899 | -16.73% |
| raw | rolling_topk_market | 43.28% | +22.88% | 0.7965 | 0.7941 | -10.74% |
| raw | rolling_residual_topk_market | 55.46% | +25.81% | 0.7334 | 0.7314 | -14.43% |
| raw | buy_hold | 99.87% | +55.89% | 0.9359 | 0.9359 | -22.36% |
| mean_min | gru | 14.74% | +6.21% | 0.9719 | 0.9429 | -1.92% |
| mean_min | gru_market | 14.74% | +2.61% | 0.7802 | 0.7511 | -3.64% |
| mean_min | identity | 14.74% | +8.69% | 0.8967 | 0.8882 | -3.29% |
| mean_min | sector | 14.74% | +8.65% | 0.8849 | 0.8760 | -3.37% |
| mean_min | rolling_topk | 14.74% | +8.49% | 0.8923 | 0.8837 | -3.38% |
| mean_min | rolling_residual_topk | 14.74% | +8.83% | 0.8924 | 0.8841 | -3.37% |
| mean_min | rolling_topk_market | 14.74% | +7.05% | 0.7965 | 0.7896 | -4.22% |
| mean_min | rolling_residual_topk_market | 14.74% | +6.40% | 0.7334 | 0.7276 | -4.26% |
| mean_min | buy_hold | 14.74% | +9.26% | 0.9359 | 0.9270 | -3.27% |
| daily_min | gru | 14.06% | +5.43% | 0.9317 | 0.9034 | -1.89% |
| daily_min | gru_market | 14.06% | +2.61% | 0.7740 | 0.7451 | -3.39% |
| daily_min | identity | 14.06% | +7.51% | 0.8745 | 0.8658 | -3.12% |
| daily_min | sector | 14.06% | +7.48% | 0.8604 | 0.8517 | -3.23% |
| daily_min | rolling_topk | 14.06% | +7.48% | 0.8755 | 0.8669 | -3.24% |
| daily_min | rolling_residual_topk | 14.06% | +7.82% | 0.8795 | 0.8711 | -3.18% |
| daily_min | rolling_topk_market | 14.06% | +6.33% | 0.7799 | 0.7724 | -3.82% |
| daily_min | rolling_residual_topk_market | 14.06% | +5.67% | 0.7225 | 0.7154 | -3.85% |
| daily_min | buy_hold | 14.06% | +8.00% | 0.8913 | 0.8827 | -3.24% |

- Gross exposure is not net exposure, beta or equal volatility.
- Constant scaling preserves ordinary net Sharpe; fixed-epsilon regularized Sharpe is not invariant.
- Targets use only executable predictions; no leverage or position clipping.
- Whole-fold scales are ex post, not a deployable strategy or model-selection validation.
- Daily matching is a different shared sizing overlay; it may raise turnover.
- Buy-and-hold is the repository ReturnPanel equal-weight always-long reference.
- Mean fold returns are not a concatenated backtest; seeds share market dates.
- Raw PC prices/context hashes and fold preprocessing match; the derived market hash differs. Every raw financial metric is rechecked, but legacy results remain exploratory and non-reusable.
- Legacy tied feature rankings were permuted to the recorded checkpoint order; the feature sets and named scalers were still checked, with no model retraining.

Sources : [rapport JSON](../../artifacts/comparisons/us-relational-market/03-exposure-comparison-pc/report.json), [rapport de calcul original](../../artifacts/comparisons/us-relational-market/03-exposure-comparison-pc/report.md). Le nombre de tests mentionné décrit la vérification historique, pas une nouvelle exécution lors du classement des docs.
