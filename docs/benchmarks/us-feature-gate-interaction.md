# US : interaction entre nombre de features et gate marché

Statut au 8 octobre 2026 : benchmark terminé, **108/108 entraînements vérifiés**.
Les passages « résultats en attente » étaient périmés. Aucun entraînement
supplémentaire n'est nécessaire pour terminer cette grille. Le holdout reste fermé
et aucun candidat n'est promu automatiquement.

## Objectif et protocole

Comparer 32/64 features pour GRU, GNN résiduel `rolling_residual_topk` et contrôle
identité sans relations, chacun avec ou sans gate Transformer : six candidats
par cap, trois folds et seeds 1/7/19. `identity_market` est inclus aux deux caps.

143 actions, pool commun `technical,market,sector`, contexte 60, objectif
PnL/Sharpe combiné à 0,25, coûts 5 bps et exécution différée d'une séance.
Labels triple barrier ATR20, horizon 10, barrières 0,75/0,75 et CUSUM 0,5.
GRU hidden size 32, GNN largeur 32/profondeur 1, Transformer largeur 32/profondeur 1.
Graphe causal lookback 252, top-k 5 et rééquilibrage 20. Ni news ni nouvelles données.

Les folds externes couvrent le 28 mars 2014 au 21 juin 2023. Holdout à partir du
22 juin 2023. Les seeds partagent les dates : neuf couples ne constituent pas
neuf périodes de marché indépendantes.

Les contrôles à 32 ont été réentraînés avec le même pool que les variantes à 64.
Le run historique utilisait `technical,sector` : ses scores ne remplacent pas
ces références. Le cap change les entrées, pas la largeur cachée.

## Complétude et comparabilité

- 54/54 tâches à chaque cap ; aucun échec, doublon ou export requis manquant.
- 108 résultats immuables concordants avec le registre et les scores du rapport.
- 672 fichiers référencés vérifiés par taille et SHA256 : 108 checkpoints,
  108 résultats, 216 prédictions, 216 chemins journaliers et 24 graphes.
- Trois audits confirment exactement 32/64 features, petit cap préfixe ordonné
  du grand, mêmes calendriers, fills/scalers communs et sélection train-only.
- Comparaisons appariées et contrôles d'exposition complets ; `unavailable=[]`,
  `selected=null`, aucun test final exécuté.

## Résultats bruts

Moyennes des neuf couples fold/seed. Sharpe régularisé du simulateur ; rendements
non annualisés, non concaténés. Drawdown = moyenne des drawdowns maximaux par run.

| Candidat | Features | Sharpe régularisé | Rendement net | Exposition brute | Drawdown moyen |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 32 | 0,8044 | +23,62 % | 46,42 % | -10,44 % |
| GRU + marché | 32 | 0,7707 | +9,59 % | 30,16 % | -8,30 % |
| GNN résiduel | 32 | 0,8668 | +26,50 % | 48,70 % | -11,71 % |
| GNN résiduel + marché | 32 | 0,8656 | +19,85 % | 40,94 % | -10,44 % |
| Identité | 32 | 0,8774 | +30,33 % | 58,17 % | -13,86 % |
| Identité + marché | 32 | 0,8889 | +20,79 % | 42,17 % | -11,67 % |
| GRU | 64 | 0,6222 | +17,72 % | 43,01 % | -11,30 % |
| GRU + marché | 64 | 0,4120 | +12,48 % | 50,15 % | -15,04 % |
| GNN résiduel | 64 | 0,8050 | +12,86 % | 20,28 % | -4,19 % |
| GNN résiduel + marché | 64 | 0,7282 | +12,64 % | 30,38 % | -7,83 % |
| Identité | 64 | 0,8794 | +11,04 % | 19,21 % | -3,73 % |
| Identité + marché | 64 | 0,7444 | +14,36 % | 33,76 % | -8,70 % |
| Buy-and-hold du simulateur | Sans cap | 0,9359 | +55,89 % | 99,87 % | -22,36 % |

### Effets features et gate

Différences de Sharpe régularisé. Interaction = gain 64 avec gate moins gain 64
sans gate. Les trois interactions sont négatives dans chacune des trois méthodes
d'exposition. Le gate ne rend pas le passage à 64 avantageux ici.

| Branche | Gain64 sans gate | Gain64 avec gate | Gain gate32 | Gain gate64 | Interaction brute |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | -0,1822 | -0,3587 | -0,0337 | -0,2102 | -0,1765 |
| GNN résiduel | -0,0618 | -0,1373 | -0,0013 | -0,0768 | -0,0755 |
| Identité | +0,0020 | -0,1445 | +0,0115 | -0,1349 | -0,1464 |

Le recul GRU64 sans gate n'est pas universel : cinq écarts sur neuf sont positifs,
mais les effets moyens par seed valent -0,8095/+0,0749/+0,1880. Une mauvaise seed
domine la baisse moyenne. En revanche, gate64 dégrade le score moyen sur les
trois seeds de chacune des trois architectures. Aucun gain stable ne justifie 64.

`identity_market32` a le meilleur Sharpe moyen des modèles, mais son gain brut
contre identité32 reste fragile : +0,0115, six victoires sur neuf, gains par fold
+0,0443/-0,1277/+0,1178. La dispersion des scores passe de 0,2007 à 0,3175 et le
minimum de 0,5625 à 0,2473. Turnover moyen de 50,28 à 63,25, rendement de 30,33 %
à 20,79 %, exposition de 58,17 % à 42,17 %. Ce n'est pas un réglage par défaut validé.

Le GNN résiduel perd contre identité dans les quatre cellules brutes : différences
moyennes -0,0106 sans gate32, -0,0743 sans gate64, -0,0233 avec gate32 et -0,0162
avec gate64. Les signes changent par fold/seed. Pas d'avantage relationnel stable.

## Exposition comparable

Les douze variantes et le buy-and-hold partagent une cible par fold/seed.
Exposition brute moyenne de 7,55 % avec facteur constant ex post ; 7,19 % avec
redimensionnement quotidien. Turnover et coûts sont recalculés.

| Candidat | Rendement, exposition moyenne commune | Sharpe net | Rendement, exposition quotidienne commune | Sharpe net |
| --- | ---: | ---: | ---: | ---: |
| GRU32 | +3,00 % | 0,8071 | +2,81 % | 0,7777 |
| GRU64 | +2,74 % | 0,6209 | +2,40 % | 0,5581 |
| Identité32 | +3,51 % | 0,8802 | +3,27 % | 0,8458 |
| Identité64 | +3,73 % | 0,8857 | +3,16 % | 0,8376 |
| Identité32 + marché | +3,69 % | 0,8924 | +3,63 % | 0,9010 |
| GNN résiduel32 | +3,47 % | 0,8708 | +3,27 % | 0,8537 |
| GNN résiduel64 | +3,47 % | 0,8117 | +3,17 % | 0,8343 |
| Buy-and-hold | +3,80 % | 0,9359 | +3,50 % | 0,9050 |

Le petit gain d'identité64 en exposition moyenne commune disparaît en contrôle
quotidien. Identité32 + marché conserve un gain quotidien moyen contre identité32,
mais seulement quatre écarts sur neuf sont positifs, médiane -0,0118 en Sharpe
régularisé. Son rendement dépasse légèrement le buy-and-hold redimensionné,
pas son Sharpe net. Cela ne démontre ni robustesse ni alpha.

Égaliser le gross ne neutralise ni exposition nette, beta, ni volatilité. Le
facteur moyen utilise tout le fold : diagnostic ex post, pas sizing déployable.
Le contrôle quotidien change l'allocation. Sharpe net et Sharpe régularisé ne
sont pas interchangeables ; l'epsilon fixe du second le rend sensible au scaling.

## Décision et suite

1. Garder **32 features par défaut** pour le prochain protocole commun.
   Aucun avantage stable ne justifie 64 ; cela ne condamne pas chaque feature ajoutée.
2. Garder **gate marché désactivé par défaut**. Identité32 + marché reste
   un challenger exploratoire, pas un gagnant promu.
3. Garder identité comme contrôle obligatoire et GNN résiduel32 comme challenger.
   Le GRU reste la référence temporelle, mais n'est pas le meilleur de cette grille.
4. Aucun entraînement manquant. Poids temporels du gate, attribution individuelle
   des features, profondeurs et fusion sont des travaux distincts, pas des folds
   à terminer.

Le sélecteur de ce runner filtre la couverture, classe par variance TRAIN avant
scaling et écarte les colonnes trop corrélées. Son rang ne mesure ni contribution
au Sharpe ni pourcentage de PnL expliqué ; les poids du gate ne le mesurent pas
non plus. Une étude
d'importance devra mesurer les variations de score après perturbation ou retrait,
en tenant compte des corrélations et interactions entre features.

Univers survivant et secteurs non PIT restent des limites. Aucun test final,
entraînement ou changement des artifacts sources pendant cette consolidation.

Sources : [rapport complet](../../artifacts/comparisons/us-relational-market/04-feature-gate-interaction/reports/features-interaction-20261006T122100403976Z/report.md),
[rapport JSON](../../artifacts/comparisons/us-relational-market/04-feature-gate-interaction/reports/features-interaction-20261006T122100403976Z/report.json),
[effets appariés](../../artifacts/comparisons/us-relational-market/04-feature-gate-interaction/reports/features-interaction-20261006T122100403976Z/feature_interactions.csv),
[audit des features](../../artifacts/comparisons/us-relational-market/04-feature-gate-interaction/reports/features-interaction-20261006T122100403976Z/feature_audit.csv),
[registre](../../artifacts/comparisons/us-relational-market/04-feature-gate-interaction/registry.json)
et [plan et commandes](../plan/us-feature-gate-interaction.md).
