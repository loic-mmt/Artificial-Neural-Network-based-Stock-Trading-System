# États de trade et de portefeuille US en overnight

Documentation vérifiée le 11 octobre 2026. Sources :
`artifacts/comparisons/us-trade-state-overnight`.

## Conclusions

- Les 72 entraînements et 144 trajectoires outer sont terminés. Le holdout
  final reste fermé.
- Les états améliorent le rendement de la financière PnL/Sharpe 0,25, mais
  beaucoup moins son Sharpe. Une grande partie du gain continu vient de
  l'augmentation de l'exposition.
- S3 financière au signe donne +51,15 %, Sharpe 0,961, contre +34,04 % et
  0,855 pour S0. Elle devient davantage long, moins short et moins active.
  Cela ne démontre pas une meilleure gestion des sorties ou des drawdowns.
- Buy-and-hold reste devant en rendement dans les 144 trajectoires.
- La CE ne bénéficie pas clairement des états. Son décodage au signe peut
  provoquer des inversions répétées et des coûts très élevés.
- AAPL reste short pendant 773 séances dans S3 financière, fold 2, seed 1,
  alors que l'âge et les pertes de la position sont bien fournis au modèle.
- La suite retenue compare d'abord les gradients des états, puis les
  architectures et politiques spécialisées dans les shorts. Aucun de ces
  prochains tests n'est un résultat acquis.

## Protocole

Univers de 143 actions US, défini dans
`configs/benchmark/stocks_us_gnn_complete_2005.json`, avec les données nettoyées
`data/processed/mt5_stocks_us_daily_clean.parquet`. Le développement commence
le 3 janvier 2005. Le holdout à partir du 22 juin 2023 est exclu.

GRU : une couche, hidden size 32, attention temporelle, tête linéaire, contexte
60, cap de 32 features technique/marché/secteur, weight decay `1e-5`.
Les états sont concaténés après le pooling, pas répétés dans la fenêtre
historique. Toutes les variantes ont dix emplacements d'état et 7 553
paramètres. S0 est un témoin de même capacité, pas une reproduction bit à bit
du GRU du [benchmark labels/losses précédent](us-label-loss-post-open.md).

| Variante | Informations d'état utilisées |
| --- | --- |
| S0 | Aucun état actif ; dix entrées mises à zéro. |
| S1 | Position précédente, position ouverte ou non, âge de l'épisode. |
| S2 | S1 plus rendement signé depuis l'entrée, giveback, séances consécutives underwater. |
| S3 | S2 plus exposition brute/nette, fraction cash et drawdown du portefeuille. |

Deux familles : CE pondérée sur `volatility_position`, horizon 10, et
financière PnL/Sharpe combinée, poids PnL 0,25. Les labels volatilité utilisent
une fenêtre 20, seuil Long 1, seuil Short 1,5, sortie 0,25 et maintien minimal
des labels 5. Ces paramètres ne forcent aucune durée ni sortie du modèle.
Il n'y a ni hybride, ni intraday, ni argmax, ni sentiment dans cette grille.

Les données achevées à J-1 et le gap open J entrent dans le modèle. Les états
sont observés à l'open J avant la décision. Le moteur utilise l'open J comme
proxy d'exécution et mesure le rendement jusqu'à l'open J+1. Il n'entre pas
à la clôture J+1. Le proxy open ne garantit pas un fill réel après observation
de l'ouverture.

Allocation par slots fixes `q/N`, coûts de 5 bps par sens, sans levier ni
redistribution aux seuls actifs non FLAT, avec liquidation finale. Les états
proviennent des décisions du modèle courant, jamais des labels parfaits.
Chaque partition et chaque décodage repartent en cash. Un resizing dans le
même sens conserve l'épisode ; Flat ou inversion le réinitialisent.

| Fold | Période outer |
| --- | --- |
| 0 | 28 mars 2014 au 25 avril 2017 |
| 1 | 26 avril 2017 au 22 mai 2020 |
| 2 | 26 mai 2020 au 21 juin 2023 |

Seeds 1, 7 et 19, gap de cinq séances globales, préparation ajustée sur TRAIN,
checkpoint choisi sur la loss inner. Tous les états sont détachés : les
paramètres du GRU et de la tête apprennent, mais le gradient ne traverse pas
les conséquences d'une décision sur les futurs états.

Les deux familles s'entraînent en continu `P(Long)-P(Short)`. Le continu est
l'évaluation principale, le signe ±1 l'évaluation secondaire. Chaque décodage
recalcule sa propre trajectoire et ses états : ce ne sont pas deux conversions
d'un même fichier de probabilités.

Huit configurations, trois folds et trois seeds donnent 72 fits, puis
144 trajectoires. Une update AdamW globale par époque, plafond 300, patience
20. Tous les fits arrêtent par early stopping, entre 25 et 202 époques ; aucun
budget insuffisant n'est signalé. Cela ne prouve pas un optimum global.

## Résultats financiers

Les tableaux donnent les moyennes arithmétiques des neuf couples fold/seed,
pas des rendements annualisés ni une courbe concaténée. Les drawdowns sont
les moyennes des drawdowns maximaux par trajectoire. Les seeds partagent les
mêmes dates : neuf couples ne sont pas neuf périodes indépendantes.

### Financière en continu

| État | Rendement net | Sharpe net | Drawdown maximal | Exposition brute |
| --- | ---: | ---: | ---: | ---: |
| S0 | +18,59 % | 0,894 | -10,46 % | 40,79 % |
| S1 | +34,56 % | 0,903 | -17,57 % | 69,55 % |
| S2 | +42,48 % | 0,895 | -19,44 % | 79,64 % |
| S3 | +45,99 % | 0,921 | -20,06 % | 82,56 % |

S3 gagne 27,40 points de rendement face à S0, mais seulement 0,027 de Sharpe.
L'exposition double presque et le drawdown s'aggrave dans les neuf couples.
Le rendement progresse dans 8/9 couples, le Sharpe dans 6/9.

### Financière au signe ±1

| État | Rendement net | Sharpe net | Drawdown maximal | Exposition long | Exposition short |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | +34,04 % | 0,855 | -17,44 % | 81,61 % | 18,26 % |
| S1 | +45,26 % | 0,937 | -20,78 % | 90,29 % | 9,58 % |
| S2 | +46,89 % | 0,939 | -20,64 % | 90,13 % | 9,74 % |
| S3 | +51,15 % | 0,961 | -21,41 % | 93,97 % | 5,90 % |
| Buy-and-hold | +58,62 % | 0,963 | -22,22 % | 99,87 % | 0,00 % |

L'exposition brute est d'environ 99,87 % pour toutes ces trajectoires, la
liquidation finale expliquant l'écart à 100 %. S3 améliore le rendement dans
8/9 couples, mais le Sharpe seulement dans 5/9. Le gain moyen de Sharpe vaut
0,106, contre un gain médian de 0,016 : la hausse n'est pas uniforme.

Le turnover moyen passe de 12,21 en S0 à 4,54 en S3, soit environ 63 % de
réduction. Le nombre moyen d'épisodes passe de 873 à 325. Le gain n'est donc
pas produit par davantage de prises de position : S3 est surtout plus long et
plus persistant. La corrélation moyenne des rendements journaliers S3 avec
buy-and-hold atteint 0,986, contre 0,893 pour S0.

Les 144 trajectoires ont un rendement inférieur à leur référence always-long
du même moteur, avec les mêmes dates et coûts. Exposition brute égale ne
signifie pas exposition nette, beta ou risque égaux.

### CE : pas de gain clair

| État | Rendement continu | Sharpe continu | Rendement au signe | Sharpe au signe |
| --- | ---: | ---: | ---: | ---: |
| S0 | +3,99 % | 0,873 | +41,53 % | 0,877 |
| S1 | +2,87 % | 0,534 | +22,43 % | 0,259 |
| S2 | +3,91 % | 0,761 | +35,20 % | 0,738 |
| S3 | +3,26 % | 0,788 | +28,76 % | 0,542 |

Les expositions continues restent faibles : de 7,77 % à 9,78 % en moyenne.
Les trois ajouts d'état ont un Sharpe moyen inférieur à S0 dans les deux
décodages. Certaines paires progressent, mais cela ne suffit pas à valider
une variante CE avec états.

## Contrôle d'exposition

Le contrôle `common-exposure.csv` ramène, dans chaque couple fold/seed et
décodage, les trajectoires à la plus faible exposition moyenne parmi les huit
candidats. En continu, cette exposition vaut 6,91 % en moyenne.

| Financière | Rendement net à exposition commune | Sharpe net |
| --- | ---: | ---: |
| S0 | +2,61 % | 0,894 |
| S1 | +2,81 % | 0,903 |
| S2 | +2,55 % | 0,895 |
| S3 | +2,83 % | 0,921 |

L'écart S3/S0 devient 0,22 point de rendement, contre 27,40 avant
normalisation. CE-S0 atteint +3,10 % dans ce contrôle, sans que cela suffise
à établir une supériorité générale.

C'est un déleveraging a posteriori des trajectoires déjà produites, pas une
nouvelle simulation en boucle fermée avec états recalculés. La faible
exposition commune est imposée par les candidats CE. Ce contrôle neutralise
une moyenne, pas le calendrier des positions, la direction ou le beta.
Au signe, l'exposition brute est déjà comparable ; le biais long/short reste
différent.

## Apprentissage des labels

Scores CE sur OUTER uniquement, contre les labels volatilité connus :

| État | Balanced accuracy continu | Macro-F1 continu | Balanced accuracy signe | Macro-F1 signe |
| --- | ---: | ---: | ---: | ---: |
| S0 | 0,350 | 0,298 | 0,350 | 0,298 |
| S1 | 0,341 | 0,290 | 0,341 | 0,282 |
| S2 | 0,345 | 0,295 | 0,346 | 0,280 |
| S3 | 0,342 | 0,298 | 0,337 | 0,264 |

Le macro-F1 continu S3 est presque identique à S0 et sa balanced accuracy
recule. Les états n'ont donc pas rendu les labels clairement plus apprenables
dans cette recette. TRAIN et inner ne sont pas mélangés à ces moyennes.

La financière reste autour de 0,333 de balanced accuracy contre ces labels.
C'est un diagnostic, pas son objectif : elle optimise la trajectoire nette,
sans composante CE. Son meilleur rendement ne signifie pas meilleure
imitation des classes.

## Instabilité de la CE au signe

La moyenne cache des trajectoires très instables. Exemple CE-S3, fold 1,
seed 19 :

- 93 756 épisodes de position et turnover 1 311,27.
- 84,58 % des décisions inversent une position déjà ouverte.
- Rendement brut des positions exécutées +17,87 %, rendement net -38,83 %.
- Le même checkpoint en continu inverse seulement 3,23 % des positions
  ouvertes et a une amplitude moyenne de position précédente de 3,25 %.

En CE-S1, fold 1, seed 19, le brut vaut +0,81 % et le net -29,51 %.
Les coûts de rotation jouent donc un rôle important dans ces échecs, sans
expliquer tous les mauvais résultats CE.

L'entraînement utilise le continu, alors que le signe réinjecte des états
de position proches de ±1. Ce changement modifie les nouvelles probabilités
et les décisions suivantes. Le décalage de politique est une explication
plausible de l'instabilité, pas une causalité démontrée. Les états de portefeuille
et les coûts changent aussi entre les deux simulations.

Il faut examiner cette cohérence entraînement/exécution avant de promouvoir
la CE avec états au signe. Il ne faut pas appeler une simple suppression de
`detach` sur le décodage discontinu une différentiation complète.

## AAPL : les états sont présents, le short persiste

Pour financière S3, fold 2, seed 1, au signe :

- Un seul épisode Short, du 26 mai 2020 à la liquidation du 21 juin 2023.
- 773 séances de détention, rendement net de l'épisode -69,07 %.
- À la liquidation, l'état renseigné avant décision contient un âge de 773
  séances et 765 séances consécutives underwater.
- La position reste également short dans la simulation continue, avec une
  amplitude variable.

Le rendement d'épisode est la métrique composée de l'export de trades, pas
une perte de 69,07 % du portefeuille entier. L'état de rendement d'entrée
décrit le mouvement de prix signé, pas le PnL pondéré par toutes les quantités
successives.

L'information n'est plus absente, mais son simple ajout n'apprend pas au
modèle à sortir. L'âge n'est pas un stop. Ce cas ne prouve pas que le détachement
est la cause : les gradients temporels restent à comparer explicitement.

## Décisions pour la suite

1. Conserver S0 et S3 financière pour le benchmark des gradients d'état,
   avec CE-S0 comme référence supervisée. Le continu reste la lecture
   principale ; l'instabilité CE au signe constitue un diagnostic séparé.
2. Comparer détachement et une méthode temporelle, par exemple TBPTT, sans
   confondre effet du gradient, budget d'updates et changement de décodage.
   Vérifier sorties de positions perdantes, timing, exposition et coûts,
   pas seulement le rendement global.
3. Après les gradients, prioriser la qualité des shorts : détecteur Short
   avec Long par défaut, deux têtes Long/Flat et Short/Flat, puis deux modèles
   spécialisés indépendants. Ces hypothèses sont fixées dans
   [le plan des décisions long/short](../plan/short-decision-benchmark.md).
4. Garder buy-and-hold et Long/Flat comme contrôles. Éviter un drawdown en
   restant cash peut aussi créer de la valeur ; shorter les baisses n'est
   pas le seul moyen possible de dépasser une référence always-long.

Le choix de S3 est un choix de travail exploratoire, pas une validation
indépendante ni une preuve d'alpha. Les périodes outer ont déjà été examinées ;
aucune seed ne doit être choisie a posteriori comme gagnante universelle.

## Sources et limites

- [Complétude](../../artifacts/comparisons/us-trade-state-overnight/report.json)
  et [configuration, folds et provenance](../../artifacts/comparisons/us-trade-state-overnight/metadata.json).
- [Résultats par trajectoire](../../artifacts/comparisons/us-trade-state-overnight/results.csv)
  et [agrégats](../../artifacts/comparisons/us-trade-state-overnight/summary.csv).
- [Classification](../../artifacts/comparisons/us-trade-state-overnight/classification.csv),
  [exposition commune](../../artifacts/comparisons/us-trade-state-overnight/common-exposure.csv)
  et [oppositions](../../artifacts/comparisons/us-trade-state-overnight/oppositions.csv).
- AAPL S3 financière : [trades](../../artifacts/comparisons/us-trade-state-overnight/fold-2/seed-1/financial-S3/overnight-sign-trades.parquet),
  [états](../../artifacts/comparisons/us-trade-state-overnight/fold-2/seed-1/financial-S3/overnight-sign-state-inputs.parquet)
  et [positions](../../artifacts/comparisons/us-trade-state-overnight/fold-2/seed-1/financial-S3/overnight-sign-positions.parquet).
- CE-S3 fold 1, seed 19 : [portefeuille](../../artifacts/comparisons/us-trade-state-overnight/fold-1/seed-19/cross_entropy-S3/overnight-sign-portfolio.parquet),
  [états au signe](../../artifacts/comparisons/us-trade-state-overnight/fold-1/seed-19/cross_entropy-S3/overnight-sign-state-inputs.parquet)
  et [états continus](../../artifacts/comparisons/us-trade-state-overnight/fold-1/seed-19/cross_entropy-S3/overnight-continuous-state-inputs.parquet).
- [Plan S0-S3](../plan/trade-state-overnight.md)
  et [fonctionnement du package](../src/trade-state-benchmark.md).

Les portefeuilles journaliers servent aux recalculs brut/net et corrélations ;
les positions et états alignés date/ticker servent aux taux d'inversion.
Les 72 manifests de complétion et les tailles des 1 872 fichiers référencés ont
été vérifiés, sans fichier manquant. Cette lecture n'a lancé aucun entraînement.

L'univers est survivant et les secteurs ne sont pas historiquement PIT.
Les frais simulés ne valident pas le coût réel des shorts, leur disponibilité,
le financement ou les fills. Les liens d'artifacts nécessitent les résultats
locaux, non suivis par Git. Les gradients non détachés et les modèles short
spécialisés ne sont pas évalués dans ce benchmark. Le
[runner des gradients](../plan/trade-state-gradients.md) est désormais prêt à
lancer ; les modèles short spécialisés restent à implémenter.
