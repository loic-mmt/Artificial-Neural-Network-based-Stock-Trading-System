# Labels, losses et décodages US après l'open

Documentation vérifiée le 10 octobre 2026. Sources :
`artifacts/comparisons/us-label-loss-post-open`.

## Conclusions

- Les 99 entraînements et 378 trajectoires sont terminés. Le holdout reste fermé.
- L'intraday est surtout pénalisé par les frais : 170 trajectoires sur 189 sont
  positives en brut, une seule en net. Inverser les positions ne corrige pas cela.
- En overnight, CE sur `volatility_position` avec décodage signe a le meilleur
  rendement moyen de la grille. La financière combinée reste un témoin utile.
- Les 63 trajectoires overnight au signe ont moins de rendement que buy-and-hold,
  malgré une exposition brute comparable. L'écart n'est donc pas seulement une
  question de taille de position.
- Les hybrides suivent presque les mêmes positions que la financière. Un poids
  CE de 0,5 ne signifie pas un équilibre des gradients.
- L'apprentissage des classes reste faible et des shorts peuvent persister
  plusieurs années. Il faut tester les informations d'état de trade et de
  portefeuille, sans attribuer automatiquement le problème à leur absence.

## Protocole

Univers de 143 actions US dans
`configs/benchmark/stocks_us_gnn_complete_2005.json`, données OHLC nettoyées,
développement du 3 janvier 2005 au 21 juin 2023. Le holdout à partir du
22 juin 2023 est exclu avant construction des features et des labels.

GRU : une couche, hidden size 32, attention temporelle, tête linéaire, contexte
60, cap de 32 features, groupes technique/marché/secteur, weight decay `1e-5`.
Sélection des features, remplissage, scaler et poids de classes CE ajustés sur
TRAIN uniquement. Aucune feature externe de sentiment ou de fondamentaux
n'est ajoutée.

Les données achevées à J-1 et le gap open J entrent dans le modèle. L'open J est
un proxy d'exécution, pas un fill garanti après observation de l'ouverture.
Ce benchmark n'entre donc pas à la clôture J+1.

| Fold | Période outer |
| --- | --- |
| 0 | 28 mars 2014 au 25 avril 2017 |
| 1 | 26 avril 2017 au 22 mai 2020 |
| 2 | 26 mai 2020 au 21 juin 2023 |

Les seeds sont 1, 7 et 19. Les partitions sont séparées par cinq séances globales.
Chaque fit utilise son meilleur checkpoint inner ; aucun checkpoint n'est
choisi sur outer. Chaque époque applique une seule update AdamW sur la loss
globale, avec 300 époques maximum et patience 20.

Deux protocoles restent distincts :

- Intraday : entrée à open J, liquidation à close J, chaque séance.
- Overnight : target à open J, rendement jusqu'à open J+1, liquidation finale.

Les coûts sont de 5 bps par sens. L'allocation utilise des slots fixes `q/N`,
sans levier ni redistribution aux seuls actifs non FLAT. Les quotes manquantes
ne sont pas remplacées. La dernière séance overnight liquide les positions.

## Labels et losses

Trois cibles Short/Flat/Long sont testées :

- `intraday_return` : signe du rendement open J vers close J ; égalité = Flat.
- `forward_return` : rendement open J vers close J+9, seuils stricts ±0,2 % ;
  entre les seuils = Flat.
- `volatility_position` : même horizon 10, volatilité historique 20 arrêtée
  à J-1, seuil Long 1 / Short 1,5, sortie 0,25 et maintien minimal des labels 5.

Un futur incomplet est inconnu, pas un vrai label Flat. Les labels
`volatility_position` repartent de Flat à chaque partition. Ces cibles
post-open ne sont pas identiques aux labels historiques de l'ancien runner.

| Famille | Objectif | Configurations |
| --- | --- | ---: |
| CE | Cross-entropy pondérée, une fois par méthode de labels | 3 |
| Financière | PnL/Sharpe combinée, poids PnL 0,25, une fois par protocole, sans supervision des classes | 2 |
| Hybride | 0,5 CE + 0,5 financière, une fois par méthode et protocole | 6 |

L'échelle PnL et l'epsilon Sharpe valent `1e-4`, l'annualisation 252. La loss
financière voit la trajectoire nette complète, pas une moyenne de Sharpes de
minibatches. Elle ne dépend pas des labels. Les poids CE et PnL sont distincts.

Onze configurations, trois folds et trois seeds donnent 99 entraînements.
La CE est réutilisée dans les deux protocoles ; les autres familles sont
évaluées dans leur protocole d'entraînement. Continu `P(Long)-P(Short)`, signe
±1 et argmax Short/Flat/Long sont appliqués aux mêmes probabilités, sans refit.
Cela donne 189 trajectoires par protocole, soit 378 au total.

Le maintien minimal et l'horizon 10 des labels n'imposent pas de sortie au
modèle. Il n'y a ni état de position en input, ni fermeture automatique après
dix séances, ni sortie triple-barrier dans ce run.

## Intraday : frais et test d'inversion

Le gross et l'inverse ci-dessous ont été recalculés à partir des courbes
journalières exportées. Pour l'inversion, le rendement brut change de signe
mais les frais restent identiques. Aucun modèle n'a été réentraîné.

| Décodage | Rendement brut moyen | Rendement net moyen | Rendement net moyen après inversion |
| --- | ---: | ---: | ---: |
| Continu | +1,50 % | -2,79 % | -5,61 % |
| Signe | +14,49 % | -47,25 % | -60,66 % |
| Argmax | +11,13 % | -29,80 % | -43,76 % |

Sur les 189 trajectoires, 170 sont positives en brut, une en net et zéro après
inversion. L'exception nette est financière-intraday, argmax, fold 2, seed 1 :
+0,68 %, Sharpe 0,073, exposition moyenne 17,40 %.

Fermer et rouvrir quotidiennement une position au signe coûte 10 bps par séance
active, même si son sens ne change pas. Le gain brut moyen ne suffit pas.
Les frais expliquent donc une grande partie du passage en négatif dans cette
grille ; cela ne valide ni la qualité du signal ni la viabilité à frais plus bas.

## Overnight : comparaison des stratégies

Les tableaux donnent des moyennes arithmétiques des neuf couples fold/seed,
pas un rendement annualisé ni une courbe concaténée. Buy-and-hold est la
référence always-long du même moteur, calendrier et coûts ; ses trois folds
sont réutilisés pour les seeds.

### Signe ±1

| Entraînement | Rendement net | Sharpe net | Drawdown maximal | Exposition brute |
| --- | ---: | ---: | ---: | ---: |
| Buy-and-hold | +58,62 % | 0,963 | -22,22 % | 99,87 % |
| CE intraday | +25,30 % | 0,722 | -16,58 % | 99,87 % |
| CE forward | +30,48 % | 0,756 | -17,81 % | 99,87 % |
| CE volatilité | +42,81 % | 0,848 | -18,04 % | 99,87 % |
| Financière | +35,48 % | 0,822 | -18,30 % | 99,87 % |
| Hybride intraday | +36,14 % | 0,842 | -17,95 % | 99,87 % |
| Hybride forward | +36,03 % | 0,835 | -18,12 % | 99,87 % |
| Hybride volatilité | +36,00 % | 0,832 | -18,01 % | 99,87 % |

« Hybride intraday » désigne ici les labels intraday, mais la composante
financière et l'exécution sont overnight. Ce n'est pas le modèle entraîné
financièrement pour une liquidation chaque soir.

CE volatilité dépasse la financière sur cinq des neuf couples en rendement
et sur cinq en Sharpe. Le gain moyen est de 7,33 points de rendement, mais
ce n'est pas une domination systématique.

Les 63 trajectoires de ce tableau ont un rendement inférieur à leur référence
buy-and-hold. Leur exposition brute est quasiment totale ; les choix Long/Short
et leur timing restent déterminants. Exposition brute égale ne signifie pas
exposition nette, beta ou risque égaux.

### Continu

| Entraînement | Rendement net | Sharpe net | Drawdown maximal | Exposition brute |
| --- | ---: | ---: | ---: | ---: |
| CE intraday | +0,75 % | 0,655 | -0,57 % | 2,51 % |
| CE forward | +2,18 % | 0,756 | -1,36 % | 5,56 % |
| CE volatilité | +3,98 % | 0,824 | -1,99 % | 8,57 % |
| Financière | +26,87 % | 0,854 | -10,71 % | 48,38 % |
| Hybride intraday | +12,24 % | 0,890 | -5,81 % | 23,82 % |
| Hybride forward | +11,92 % | 0,892 | -5,53 % | 22,87 % |
| Hybride volatilité | +11,26 % | 0,891 | -5,35 % | 22,19 % |

La faible amplitude des probabilités CE donne de petites positions continues.
Le signe augmente leur exposition, pas leur exactitude de classification.
Les hybrides continus ont un meilleur Sharpe moyen que la financière, mais
moins de rendement et d'exposition. Il ne faut pas les classer seulement au PnL.

Le contrôle `common-exposure.csv` réduit tous les continus à la plus faible
exposition moyenne de leur couple fold/seed, soit 2,51 % en moyenne. À cette
exposition, CE volatilité donne +1,19 %, la financière +1,00 % et les trois
hybrides environ +1,12 %. Ce contrôle réduit l'écart brut sans égaliser le
timing de l'exposition ni démontrer une alpha.

### Argmax

| Entraînement | Rendement net | Sharpe net | Drawdown maximal | Exposition brute |
| --- | ---: | ---: | ---: | ---: |
| CE intraday | +21,66 % | 0,714 | -15,26 % | 79,28 % |
| CE forward | +17,23 % | 0,705 | -11,91 % | 46,12 % |
| CE volatilité | +28,89 % | 0,828 | -14,04 % | 78,87 % |
| Financière | +35,30 % | 0,841 | -17,85 % | 92,43 % |
| Hybride intraday | +35,82 % | 0,853 | -17,70 % | 91,51 % |
| Hybride forward | +35,73 % | 0,852 | -17,62 % | 91,48 % |
| Hybride volatilité | +35,31 % | 0,836 | -17,74 % | 91,13 % |

Argmax permet Flat, contrairement au signe sauf égalité exacte des probabilités
directionnelles. Pour CE volatilité, il réduit le rendement et l'exposition
par rapport au signe ; pour la financière, le rendement est proche et le Sharpe
un peu supérieur. Il n'y a pas un décodage gagnant sur tous les critères.

## Apprentissage des labels et oracle

Scores outer de chaque CE contre sa propre cible, labels connus uniquement :

| Cible | Accuracy | Accuracy majoritaire TRAIN | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Intraday | 40,31 % | 50,99 % | 0,357 | 0,270 |
| Forward | 26,43 % | 53,91 % | 0,366 | 0,223 |
| Volatilité | 36,71 % | 52,24 % | 0,350 | 0,289 |

La balanced accuracy majoritaire vaut 0,333. La CE améliore légèrement le rappel
équilibré, mais reste loin de reproduire ses labels. Les poids de classes rendent
la comparaison d'accuracy seule insuffisante. Le meilleur rendement CE-volatilité
ne signifie pas la meilleure balanced accuracy : stratégie et classification
mesurent deux choses différentes.

L'exécution parfaite rétrospective des labels overnight reste très positive :

| Fold | Oracle forward | Oracle volatilité |
| --- | ---: | ---: |
| 0 | +1 286,78 % | +629,52 % |
| 1 | +2 313,50 % | +1 097,86 % |
| 2 | +4 046,09 % | +1 528,86 % |

Ces oracles rééquilibrent les targets au même rythme que le moteur, avec les
labels inconnus en cash. Ils utilisent le futur et ne sont pas réalisables.
L'oracle intraday atteint aussi des rendements extrêmes, par construction.
Cela ne prouve ni apprenabilité ni plafond financier universel sous un autre
protocole. Il ne faut pas confondre ces cibles post-open avec le
[diagnostic antérieur de qualité des labels](label-quality-us.md).

Les 99 fits ont tous arrêté par early stopping, entre 25 et 167 époques ;
`budget_insufficient_fits=0`. Il n'y a pas ici de fit tronqué à 300 époques.
Cela ne prouve pas qu'un optimum global est atteint, mais ne justifie pas
automatiquement d'augmenter le budget d'époques pour corriger les trades.

## Similarité des décisions et équilibre de l'hybride

En overnight au signe, moyennes des comparaisons appariées par fold/seed :

- Financière contre hybride intraday : corrélation des rendements journaliers
  nets 0,991 ; environ 96,56 % de directions identiques lorsque les deux
  positions sont non FLAT.
- Hybride forward contre hybride volatilité : corrélation 0,999 ; environ
  99,63 % de directions identiques lorsque les deux positions sont non FLAT.
- CE volatilité contre financière : corrélation moyenne 0,605, allant de
  -0,577 à 0,988 selon le couple. Cette branche apporte des décisions plus
  distinctes, pas un gain de diversification déjà validé.

L'impression de courbes similaires est donc confirmée pour financière/hybrides.
Ce ne sont pas plusieurs stratégies indépendantes.

Les médianes du ratio de normes de gradients pondérés financière/CE des hybrides
overnight vont de 70,17 à 88,58 à la première époque et de 6,76 à 7,97 au meilleur
checkpoint. La composante financière domine donc les normes malgré les poids
0,5/0,5. Cela aide à expliquer la proximité des décisions, sans prouver que
les gradients ont toujours la même direction. Il faut contrôler leurs échelles
avant de présenter l'hybride comme un compromis équilibré.

## Seeds, shorts et exemple AAPL

Sur les sept variantes overnight au signe, moyennes par seed et sur les
trois folds :

| Seed | Rendement net | Sharpe net |
| --- | ---: | ---: |
| 1 | +34,86 % | 0,830 |
| 7 | +36,28 % | 0,833 |
| 19 | +32,68 % | 0,761 |

La seed 19 n'est donc pas la meilleure globalement dans ce run. La choisir
après inspection d'un graphique ou d'un fold ne fournit pas un résultat robuste.
Les seeds partagent les mêmes périodes : neuf couples ne font pas neuf périodes
de marché indépendantes.

La contribution nette moyenne des shorts est négative pour les sept variantes
overnight au signe. Pour la financière, elle est de -969,32 sur un capital
initial de 10 000, contre +4 517,36 pour les longs. Cela reste un constat sur
ces périodes, pas une justification d'interdire universellement les shorts.

Pour AAPL, financière overnight au signe : 88,20 % des séances sont short en
moyenne sur les neuf couples. Le fold 2, seed 1, conserve 773 séances short
consécutives, puis liquide le dernier jour. Les folds 0 et 1, seed 1, présentent
également 774 séances short consécutives.

Une position très longue peut donc être réellement présente dans les exports,
pas seulement un problème de graphique. Le modèle n'observe ici ni son âge
de position, ni son rendement d'entrée, ni le drawdown du portefeuille.
L'absence de ces inputs est une hypothèse à tester, pas une cause démontrée.

## Décision pour la suite

Il faut conserver deux familles pour le prochain test : CE volatilité et
financière PnL/Sharpe 0,25, en overnight, avec continu en principal et signe en
secondaire. Le [plan S0-S3](../plan/trade-state-overnight.md) isole l'effet des
états de trade et de portefeuille ; chaque décodage y reconstruit ses propres
états et probabilités. Les états y sont détachés. Le benchmark du détachement
contre une autre méthode reste ultérieur.

Suivi : le [benchmark S0-S3](us-trade-state-overnight.md) est désormais terminé.
Les états améliorent surtout l'exposition longue de la financière et ne
corrigent pas tous les shorts prolongés. La suite retenue compare les gradients,
puis les [décisions short spécialisées](../plan/short-decision-benchmark.md).

Il ne faut pas retenir l'inversion intraday ni augmenter les époques sans
diagnostic supplémentaire. L'hybride reste une piste après contrôle de ses
gradients, pas une stratégie de diversification démontrée. Aucune nouvelle
configuration n'est validée sur le holdout final.

Les anciens tests [Sharpe seul et combinaison](gru-optim/05-cara-combined.md)
restent pertinents : faible rendement avec Sharpe seul, davantage de rendement
et de drawdown avec la combinée. Sharpe seul n'est pas une famille testée dans
la présente grille ; ces séries aux protocoles différents ne doivent pas être
classées directement ensemble.

## Sources et conventions

- [Complétude](../../artifacts/comparisons/us-label-loss-post-open/report.json)
  et [configuration, folds, provenance](../../artifacts/comparisons/us-label-loss-post-open/metadata.json).
- [Résultats par trajectoire](../../artifacts/comparisons/us-label-loss-post-open/results.csv)
  et [moyennes par variante/protocole/décodage](../../artifacts/comparisons/us-label-loss-post-open/summary.csv).
- [Classification](../../artifacts/comparisons/us-label-loss-post-open/classification.csv),
  [oppositions](../../artifacts/comparisons/us-label-loss-post-open/oppositions.csv)
  et [exposition commune](../../artifacts/comparisons/us-label-loss-post-open/common-exposure.csv).
- [Oracle du fold 0](../../artifacts/comparisons/us-label-loss-post-open/fold-0/label_oracles.json),
  [fold 1](../../artifacts/comparisons/us-label-loss-post-open/fold-1/label_oracles.json)
  et [fold 2](../../artifacts/comparisons/us-label-loss-post-open/fold-2/label_oracles.json).
- [Exemple AAPL fold 2, seed 1](../../artifacts/comparisons/us-label-loss-post-open/fold-2/seed-1/financial-overnight/overnight-sign-positions.parquet).
- [Guide du runner et des exports](../src/label-loss-benchmark.md).

Les fichiers `*-portfolio.parquet` servent au recalcul brut/net/inversé et aux
corrélations, `*-positions.parquet` aux durées/directions, et
`learning_diagnostics.json` aux ratios de gradients. Les sondes d'inversion
et de corrélation sont des recalculs de reporting, pas des candidats ajoutés
au benchmark ni des entraînements supplémentaires.

Les liens d'artifacts nécessitent les résultats locaux, non suivis par Git.
L'univers et les secteurs ne sont pas historiquement point-in-time. Les outer
ont été examinés pour préparer la suite : les choix restent exploratoires et
la validation indépendante doit rester séparée.
