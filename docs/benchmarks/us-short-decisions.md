# Spécialisation des décisions short US en overnight

Documentation vérifiée le 14 octobre 2026. Sources :
`artifacts/comparisons/us-short-decisions`.

## Conclusions

- Les 81 fits, 99 encodeurs GRU, 486 trajectoires outer et 324
  contre-factuels sont terminés. Le holdout final reste fermé.
- Aucun spécialiste ne démontre un meilleur signal short. Les contributions
  Short moyennes sont négatives dans les deux décodages, y compris lorsque
  l'actif est sous sa moyenne mobile passée de 200 séances.
- La financière à deux GRU atteint +48,39 % en continu et +52,39 % au
  décodage binaire. Mais son gain continu face à S3 dépend d'un seul couple
  fold/seed ; son Sharpe binaire moyen recule et ses shorts restent perdants.
- Remplacer les shorts par Flat sur les mêmes signaux améliore le rendement
  dans 159/162 comparaisons. La suppression du côté Short est mieux étayée
  ici que sa spécialisation, sans valider pour autant un modèle Long/Flat.
- Le meilleur rendement moyen des politiques fermées est +55,47 % pour le
  détecteur financier Long/Flat binaire, contre +58,62 % pour always-long.
  Une seule des 486 trajectoires dépasse cette référence en rendement net
  sans ajustement d'exposition, de seulement 0,28 point.
- La CE binaire apprend peu les labels Short hors TRAIN. Son exécution
  binaire reste instable ; les conflits entre têtes et les coûts de rotation
  sont importants. Deux GRU ne règlent pas ce problème.
- AAPL reste short 773 séances avec les deux têtes financières. Les deux GRU
  raccourcissent cet épisode au binaire, mais gardent un short de 652 séances ;
  leur continu reste short sur toute la période. Aucun stop n'a été appris
  de manière fiable.

Je conserve S3 détaché comme témoin, pas comme stratégie validée. Le détecteur
financier et Long/Flat restent des contrôles utiles pour la suite. Je ne
remplace pas la référence par deux GRU au seul vu du meilleur rendement moyen.

## Protocole et comparabilité

Même univers de 143 titres US, mêmes prix et mêmes partitions que les
[états S0-S3](us-trade-state-overnight.md) :
`configs/benchmark/stocks_us_gnn_complete_2005.json` et
`data/processed/mt5_stocks_us_daily_clean.parquet`. Développement à partir du
3 janvier 2005 ; toutes les observations à partir du 22 juin 2023 sont exclues
avant les features. Les trois témoins Long/Short reproduisent exactement les
18 rendements continu/binaire de leur variante dans le benchmark S0-S3.

| Fold | Période outer | Séances du portefeuille, liquidation incluse |
| --- | --- | ---: |
| 0 | 28 mars 2014 au 25 avril 2017 | 775 |
| 1 | 26 avril 2017 au 22 mai 2020 | 775 |
| 2 | 26 mai 2020 au 21 juin 2023 | 774 |

Seeds 1, 7 et 19, gap de cinq séances, préparation et sélection des 32 features
ajustées sur TRAIN, checkpoint choisi sur la loss inner. GRU32, une couche,
attention temporelle, tête linéaire, contexte 60, weight decay `1e-5`.
Les dix états S3 sont concaténés après le pooling et détachés ; CE-S0 conserve
dix emplacements mis à zéro. Aucun TBPTT, filtre de régime, stop ou durée
maximale n'est ajouté.

Les entrées utilisent les informations achevées à J-1 et le gap open J.
Le rendement exécuté va de l'open J à l'open J+1. L'open reste un proxy après
observation de l'ouverture, pas un fill garanti. Allocation fixe par slots
`q/N`, sans levier, coût 5 bps par sens, liquidation finale. Chaque partition,
politique et décodage repartent en cash et recalculent leurs propres états.

Deux losses : CE binaire pondérée avec poids TRAIN, et financière PnL/Sharpe
combinée, poids PnL 0,25. La financière n'imite pas les labels. Pour la CE,
`volatility_position` conserve horizon 10, fenêtre 20, seuils Long 1 et Short
1,5, sortie 0,25 et maintien minimal des labels 5 séances. Ces paramètres
n'imposent pas de sortie au modèle. Les labels inconnus sont masqués.

### Architectures et décisions

| Architecture | Position continue Long/Short | Décision binaire, seuils 0,5 | Paramètres |
| --- | --- | --- | ---: |
| Détecteur Short | `1 - 2 * score_short` | Long sans alerte, Short avec alerte | 7 467 |
| Un GRU, deux têtes | `score_long - score_short` | Une seule tête active : son côté ; aucune ou deux : Flat | 7 510 |
| Deux GRU séparés | Même différence, encodeurs indépendants | Même arbitrage Flat | 14 934 |
| Témoins trois classes S3/S0 | `P(Long) - P(Short)` | Signe de cette différence | 7 553 |

Le décodage nommé `sign` n'est donc pas simplement le signe du continu pour
les architectures à deux têtes. Les seuils et l'arbitrage sont fixés avant
OUTER, sans optimisation ni calibration revendiquée.

La CE Short oppose Short à not-Short, qui comprend Long et Flat connus ;
la tête Long oppose Long à not-Long. Pour le détecteur, not-Short est exécuté
Long au binaire : ce n'est pas une stratégie Short/Flat. Les scores
financiers sont des activations, pas des probabilités calibrées.

Les deux encodeurs séparés ont leurs propres paramètres mais une loss et un
checkpoint communs pour le portefeuille exécuté. En CE, les deux losses
binaires pondérées et normalisées sont moyennées. Ce ne sont pas deux modèles
optimisés isolément puis assemblés après OUTER.

Les six spécialistes et les témoins CE-S3, CE-S0 et financière-S3 donnent
neuf configurations, chacune répétée sur neuf couples fold/seed. Seul le
continu Long/Short est entraîné. Les politiques `long_flat = max(q, 0)` et
`short_flat = min(q, 0)` sont ensuite évaluées sans réentraînement, avec leur
propre boucle d'état. Les contre-factuels gèlent au contraire les signaux
Long/Short et les projettent sans feedback : ils ne comptent pas comme des
stratégies supplémentaires entraînées.

## Lecture des métriques

Les tableaux donnent les moyennes arithmétiques des neuf couples fold/seed.
Les rendements ne sont ni annualisés ni concaténés. Le Sharpe net utilise
252 séances et l'écart-type population ; les drawdowns sont les moyennes des
drawdowns maximaux par trajectoire. Les gains appariés comparent le même
fold, seed, décodage et politique. Neuf couples ne sont que trois périodes
de marché répétées avec différentes initialisations.

La référence appelée buy-and-hold dans les études précédentes est
l'`always_long` du même moteur : slots égaux, mêmes dates, coûts et liquidation.
Ce n'est pas un ETF observé ni un portefeuille de quantités fixes acheté une
fois. Le déleveraging est descriptif, pas une sélection de sizing déployable.

## Résultats Long/Short

### Financière : meilleur rendement moyen, pas meilleur short démontré

| Architecture | Rendement continu | Sharpe continu | Drawdown continu | Exposition brute | Exposition short |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3 trois classes | +45,99 % | 0,921 | -20,06 % | 82,56 % | 1,60 % |
| Détecteur Short | +44,61 % | 0,922 | -19,37 % | 80,09 % | 1,28 % |
| Deux têtes | +47,69 % | 0,936 | -20,26 % | 86,39 % | 0,82 % |
| Deux GRU | +48,39 % | 0,938 | -20,76 % | 85,75 % | 0,46 % |

| Architecture | Rendement binaire | Sharpe binaire | Drawdown binaire | Exposition brute | Exposition short |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3 trois classes | +51,15 % | 0,961 | -21,41 % | 99,87 % | 5,90 % |
| Détecteur Short | +51,93 % | 0,958 | -21,28 % | 99,87 % | 5,16 % |
| Deux têtes | +51,12 % | 0,927 | -21,55 % | 97,40 % | 3,09 % |
| Deux GRU | +52,39 % | 0,943 | -21,92 % | 95,41 % | 1,63 % |
| Always-long | +58,62 % | 0,963 | -22,22 % | 99,87 % | 0,00 % |

Au binaire, les deux GRU gagnent 1,24 point de rendement face à S3, mais
perdent 0,018 de Sharpe moyen. Leur contribution Long passe de +54,34 à
+53,65 points de rendement ; la contribution Short passe de -3,20 à -1,26.
Le gain vient donc de shorts moins destructeurs, pas de shorts devenus
rentables. Les deux contributions sont des attributions additives du PnL
du portefeuille, pas des rendements indépendants de deux sous-portefeuilles.

La corrélation moyenne des rendements journaliers des deux GRU avec
always-long atteint 0,992 en continu et 0,993 au binaire. La financière
S3 est à 0,975 et 0,986. Davantage de proximité avec always-long n'établit
pas une meilleure détection des baisses.

### CE : spécialiser les classes ne stabilise pas l'exécution

| Architecture | Rendement continu | Sharpe continu | Rendement binaire | Sharpe binaire |
| --- | ---: | ---: | ---: | ---: |
| S0 trois classes | +3,99 % | 0,873 | +41,53 % | 0,877 |
| S3 trois classes | +3,26 % | 0,788 | +28,76 % | 0,542 |
| Détecteur Short | +5,16 % | 0,731 | +30,80 % | 0,627 |
| Deux têtes | +2,47 % | 0,650 | +4,95 % | -0,216 |
| Deux GRU | +2,36 % | 0,594 | +21,27 % | 0,527 |

Le détecteur CE gagne du rendement continu mais prend davantage d'exposition :
13,68 %, contre 7,77 % en S3. Son drawdown passe de -1,48 % à -3,49 % ; il
s'aggrave dans les neuf couples et le Sharpe ne gagne que dans un couple.
Les deux GRU dégradent le rendement et le Sharpe continus dans 8/9 couples.
Le témoin CE-S0 reste devant les trois spécialistes en Sharpe moyen, dans
les deux décodages.

## Gains appariés et dépendance aux seeds

### Face au témoin S3 de la même famille, en continu Long/Short

| Spécialiste | Gain de rendement moyen, points | Gain médian, points | Gains de rendement | Gain de Sharpe moyen | Gains de Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE détecteur | +1,91 | +2,34 | 6/9 | -0,057 | 1/9 |
| CE deux têtes | -0,78 | -0,37 | 4/9 | -0,137 | 5/9 |
| CE deux GRU | -0,90 | -1,01 | 1/9 | -0,194 | 1/9 |
| Financière détecteur | -1,38 | -0,87 | 3/9 | +0,001 | 3/9 |
| Financière deux têtes | +1,70 | -2,08 | 3/9 | +0,015 | 6/9 |
| Financière deux GRU | +2,40 | -0,13 | 3/9 | +0,017 | 6/9 |

Les gains financiers de rendement continu sont concentrés :

- Deux têtes, fold 0/seed 1 : rendement +11,27 % en S3 contre +48,33 %,
  exposition 30,83 % contre 94,91 %, drawdown -4,74 % contre -14,14 %.
  Sans ce couple, le gain moyen sur les huit autres devient -2,72 points.
- Deux GRU, fold 0/seed 7 : +5,71 % contre +37,54 %, exposition 26,51 %
  contre 76,03 %, drawdown -2,28 % contre -13,35 %. Sans ce couple, le gain
  moyen sur les huit autres devient -1,28 point.

Retirer un couple est une sensibilité descriptive, pas une nouvelle sélection.
Les trajectoires d'optimisation changent aussi : les checkpoints de ces deux
spécialistes sont aux époques 108 et 83, contre 54 et 57 pour S3. Je ne
présente pas le gain comme l'effet isolé de meilleures décisions Short.
Pour les deux architectures, le drawdown s'améliore dans 7/9 couples, mais
sa moyenne se dégrade à cause des couples à forte hausse d'exposition.

| Fold | S3 financière, continu | Deux GRU financiers, continu | Always-long |
| --- | ---: | ---: | ---: |
| 0 | +21,23 % | +31,41 % | +54,91 % |
| 1 | +32,71 % | +30,26 % | +34,34 % |
| 2 | +84,04 % | +83,51 % | +86,60 % |

Le gain moyen des deux GRU n'est pas répété sur les trois folds. Le fold 1
inclut la chute de 2020 : au binaire, leur drawdown vaut -33,12 %, contre
-33,40 % pour always-long. Leur exposition short moyenne de ce fold est
seulement 0,46 %. Ce résultat ne valide pas une protection active des crises.

Au binaire Long/Short, les deux GRU financiers gagnent en rendement dans
5/9 couples et en Sharpe dans 5/9 ; le détecteur gagne dans 4/9 et 3/9.
Les deux têtes n'améliorent ni le rendement moyen ni le Sharpe moyen.

## Éviter une baisse n'est pas la shorter

### Même signal gelé : remplacer Short par Flat

Les contre-factuels conservent exactement les signaux de la simulation
Long/Short. Ils recalculent positions et coûts, sans nouvelle inférence,
sans entraîner une autre politique et sans réinjecter ses états dans le modèle.

| Spécialiste | Gain moyen Long/Flat gelé, continu, points | Gains | Gain moyen Long/Flat gelé, binaire, points | Gains |
| --- | ---: | ---: | ---: | ---: |
| CE détecteur | +1,29 | 9/9 | +13,09 | 9/9 |
| CE deux têtes | +0,50 | 9/9 | +8,30 | 8/9 |
| CE deux GRU | +0,50 | 9/9 | +4,83 | 9/9 |
| Financière détecteur | +0,93 | 8/9 | +3,30 | 9/9 |
| Financière deux têtes | +0,91 | 9/9 | +2,76 | 9/9 |
| Financière deux GRU | +0,50 | 8/9 | +1,44 | 9/9 |

Témoins compris, Long/Flat gelé gagne en rendement dans 159/162 comparaisons.
Cela isole mieux le coût économique du Short que deux simulations avec des
alertes différentes. Cela ne prouve pas une utilité du détecteur : une
stratégie presque toujours Long reste un contrôle indispensable.

### Politiques fermées : les alertes peuvent changer

Avec feedback propre à chaque politique, Long/Flat gagne face à Long/Short
dans 155/162 comparaisons. Les moyennes financières deviennent :

| Architecture | Long/Short continu | Long/Flat continu | Long/Short binaire | Long/Flat binaire |
| --- | ---: | ---: | ---: | ---: |
| S3 trois classes | +45,99 % | +46,97 % | +51,15 % | +55,13 % |
| Détecteur Short | +44,61 % | +45,52 % | +51,93 % | +55,47 % |
| Deux têtes | +47,69 % | +48,60 % | +51,12 % | +53,95 % |
| Deux GRU | +48,39 % | +48,88 % | +52,39 % | +53,64 % |

Les versions gelée et fermée ne sont pas interchangeables. Pour le détecteur
CE binaire, Long/Flat gelé donne +43,89 %, la boucle fermée +48,73 % :
4,84 points viennent de trajectoires de feedback différentes. Pour CE-S0,
sans états actifs, les deux versions coïncident exactement.

Le Short/Flat fermé perd dans 157/162 trajectoires ; quatre restent cash et
une seule gagne, de +0,0057 % avec 0,15 % d'exposition moyenne. Les dix-huit
moyennes candidat/décodage sont négatives. Les projections Short/Flat gelées
n'ont elles aussi qu'une trajectoire positive. Le côté Short n'est donc pas
un moteur autonome de rendement démontré ici. Cash à rendement zéro est un
contrôle plus pertinent que le seul rendement brut always-long pour ce côté.

### Always-long et exposition comparable

Le détecteur financier Long/Flat binaire atteint +55,47 %, Sharpe 0,951,
drawdown -22,08 %, exposition 96,35 %. Toujours sous les +58,62 % et
0,963 d'always-long. L'unique exception sans ajustement d'exposition est
fold 2/seed 7 : +86,89 %
contre +86,60 %, soit +0,28 point. Dans cette même paire, le contre-factuel
gelé vaut +86,58 % et reste sous la référence : l'exception n'est pas une
preuve du gain des seules alertes initiales.

À exposition moyenne propre au candidat, la référence always-long est
réduite d'un facteur constant pour chaque trajectoire, coûts recalculés.
Les deux GRU financiers Long/Short donnent +48,39 % contre +50,31 % en
continu, et +52,39 % contre +55,80 % au binaire. Ils ne la dépassent que
dans 1/9 couples dans chaque décodage. Les deux têtes ne la dépassent dans
aucun couple Long/Short. Le détecteur Long/Flat binaire reste à -0,97 point
de sa référence ainsi redimensionnée en moyenne, avec un seul gain sur neuf.

Le contrôle `common-exposure.csv` réduit les neuf candidats à la plus faible
exposition moyenne du même fold/seed, décodage et politique. En Long/Short
continu, elle est de 5,79 % en moyenne, imposée par les candidats CE.

| Financière Long/Short | Rendement continu à exposition commune | Gain face à S3, points | Gains appariés |
| --- | ---: | ---: | ---: |
| S3 trois classes | +2,43 % | — | — |
| Détecteur Short | +2,51 % | +0,08 | 4/9 |
| Deux têtes | +2,62 % | +0,19 | 4/9 |
| Deux GRU | +2,76 % | +0,33 | 6/9 |

Une petite différence persiste, mais la hausse brute de 2,40 points des deux
GRU est fortement réduite. Ce contrôle ne réexécute pas les états à cette
nouvelle taille. Il égalise une moyenne, pas le timing, la direction, le beta
ou la volatilité. En Short/Flat, deux couples sur neuf ont une exposition
commune nulle dans chaque décodage : les neuf candidats y sont artificiellement
ramenés à cash. Ces comparaisons dégénérées ne valident aucun short.

## Ce qu'apprennent les têtes

Lecture des scores Short sur OUTER, en continu Long/Short, contre les labels
volatilité connus. Moyennes par couple, pas une accuracy regroupant TRAIN,
inner et OUTER.

| CE binaire | Balanced accuracy | AUROC | Précision Short | Rappel Short |
| --- | ---: | ---: | ---: | ---: |
| Détecteur | 0,507 | 0,541 | 15,34 % | 24,83 % |
| Deux têtes | 0,510 | 0,534 | 16,10 % | 38,69 % |
| Deux GRU | 0,511 | 0,528 | 15,58 % | 31,04 % |

La prévalence Short moyenne est 14,75 %. Les précisions restent proches de
cette fréquence. Le détecteur passe d'une AUROC 0,602 sur TRAIN à 0,572
sur inner puis 0,541 sur OUTER ; sa balanced accuracy passe de 0,569 à
0,530 puis 0,507. Le classement conserve un petit signal descriptif, mais
le seuil fixe ne donne pas un bon classifieur hors entraînement.

Les têtes Long CE ont aussi une balanced accuracy proche de 0,5 : 0,508
avec encodeur partagé, 0,512 avec deux GRU. Leur rappel élevé ne suffit pas
à démontrer une bonne discrimination.

Les activations Short financières ont des AUROC de 0,500 pour le détecteur,
0,481 pour les deux têtes et 0,488 pour les deux GRU. Elles ne sont pas
entraînées pour ces labels : ce constat ne mesure ni une probabilité de
baisse calibrée ni directement leur valeur économique. Les bonnes métriques
financières globales ne doivent pas être présentées comme une bonne imitation
des classes Short.

Pour les témoins trois classes, le diagnostic binaire teste `P(Short)>0,5`,
alors que leur décision au signe compare `P(Short)` et `P(Long)`. Un rappel
binaire presque nul de ces témoins n'implique donc pas absence de shorts
exécutés. Les critères de décision ne sont pas identiques.

## Instabilité binaire, conflits et coûts

Au binaire, les deux têtes CE sont simultanément actives dans 14,96 % des
signaux et toutes deux inactives dans 29,33 % : 44,28 % donnent Flat.
Avec deux GRU CE, ces fractions sont 23,09 % et 15,95 %, soit 39,04 % Flat.
Leur exposition moindre ne signifie pas une abstention bien calibrée ;
une partie est un conflit entre signaux Long et Short.

Les deux GRU financiers ont 2,10 % de conflits et 2,37 % de doubles
inactivations. Leur moindre fréquence de Short vient aussi de cet arbitrage :
une tête Short active ne suffit pas lorsque Long est également actif.
Ces fractions concernent les signaux du modèle avant liquidation forcée.

| CE binaire Long/Short | Fold/seed | Turnover cumulé | Rendement brut des positions | Rendement net |
| --- | --- | ---: | ---: | ---: |
| S3 trois classes | 2/19 | 1 119,03 | +60,82 % | -8,10 % |
| Détecteur | 2/19 | 727,29 | +46,22 % | +1,63 % |
| Deux têtes | 1/7 | 604,01 | +0,86 % | -25,44 % |

Le brut est recomposé sur les positions déjà exécutées en retirant seulement
les coûts journaliers : il ne simule pas un modèle réentraîné sans coûts
ni une nouvelle boucle d'état sans coûts. Les charges monétaires de ces
trajectoires sont respectivement 6 645,54, 4 205,54 et 2 692,16 pour un
capital initial de 10 000. La différence de rendements composés n'est pas
simplement cette somme de charges divisée par le capital initial.

L'entraînement utilise le continu, mais les positions binaires réinjectent
des états proches de ±1 ou Flat. Cela peut changer les futurs scores et
alimenter une oscillation. C'est une explication plausible, pas une cause
isolée expérimentalement. La CE binaire ne résout pas ce décalage par le
seul changement du nombre de sorties.

## Qualité économique des shorts et régimes

En Long/Short, la contribution Short est positive dans seulement 2/162
trajectoires, contre 156/162 pour la contribution Long. Les moyennes Short
restent négatives pour les neuf candidats dans chaque décodage.

Le régime est défini par actif : clôture ajustée J-1 au-dessus ou au-dessous
de sa moyenne passée de 200 cotations. Il décrit une tendance passée, pas
une baisse future certaine, et ne modifie aucun signal. Aucun régime
`unknown` n'apparaît dans les exports outer de cette grille.

| Financière Long/Short | PnL Short en tendance baissière, continu | PnL Short en tendance haussière, continu | PnL Short baissier, binaire | PnL Short haussier, binaire |
| --- | ---: | ---: | ---: | ---: |
| S3 trois classes | -46,30 | -48,84 | -145,37 | -173,51 |
| Détecteur | -48,77 | -39,33 | -138,68 | -151,91 |
| Deux têtes | -56,78 | -29,43 | -139,34 | -107,35 |
| Deux GRU | -29,06 | -17,01 | -61,91 | -60,55 |

PnL moyen en unités monétaires pour un capital initial de 10 000, attribué
selon la position courante. Le coût d'une liquidation peut être attribué à
Flat dans ce tableau ; l'attribution Long/Short globale rattache séparément
les coûts d'entrée et de sortie à chaque côté. Les deux conventions ne
doivent donc pas être confondues.

Même sous la moyenne passée, les shorts ne deviennent pas rentables en
moyenne. Moins perdre dans les deux régimes n'est pas découvrir un alpha
baissier. Les contributions de tous les côtés et régimes se réconcilient
avec le PnL global ; ce ne sont pas des portefeuilles autonomes investis
uniquement dans un régime.

## Persistance des pertes et AAPL

Diagnostics Long/Short binaire ; les fréquences sont des moyennes par run,
les nombres de shorts perdants sont des sommes sur les neuf couples.

| Financière | Séances short suivies d'une baisse | Épisodes short gagnants | Épisodes perdants d'au moins 252 séances | Durée short maximale |
| --- | ---: | ---: | ---: | ---: |
| S3 trois classes | 47,54 % | 35,82 % | 48 | 774 |
| Détecteur | 46,24 % | 32,12 % | 47 | 774 |
| Deux têtes | 45,80 % | 27,97 % | 21 | 774 |
| Deux GRU | 46,14 % | 33,17 % | 7 | 652 |

Le succès par séance ignore amplitude et coûts ; il n'est pas le taux de
gain des épisodes. Réduire le nombre de pertes longues en prenant moins de
shorts ne prouve pas que leur timing devient meilleur. Ces comptes répètent
les mêmes périodes selon les seeds, pas des événements indépendants.

Les compteurs underwater aux fermetures décidées par le modèle sont
censurés : ils excluent les shorts non fermés par le modèle et distinguent
les liquidations forcées. Leur baisse ne mesure pas à elle seule un meilleur
délai de sortie après perte. Les exports de trades restent nécessaires pour
les épisodes et les sorties sans ligne de signal.

### AAPL, fold 2/seed 1

| Financière Long/Short binaire | Nombre d'épisodes Short | Épisode le plus long | Rendement net de cet épisode |
| --- | ---: | ---: | ---: |
| S3 trois classes | 1 | 773 séances | -69,07 % |
| Détecteur | 4 | 175 séances | -18,17 % |
| Deux têtes | 1 | 773 séances | -69,07 % |
| Deux GRU | 4 | 652 séances | -54,88 % |

Les deux têtes restent short du 26 mai 2020 à la liquidation du 21 juin 2023.
À la dernière décision, âge 773 et 765 séances underwater sont présents.
Leur encodage logarithmique vaut environ 1,202 et 1,200 ; le rendement signé
depuis l'entrée de -132,86 % est comprimé par `tanh` à -1, et le giveback à +1.
Une perte sévère est donc signalée, mais son amplitude devient presque
indistinguable d'autres pertes déjà saturées. Ce n'est pas une preuve que
cette saturation cause la persistance : une ablation serait nécessaire.

Les deux GRU au binaire ferment leur premier short après 36 séances, puis
restent short du 24 juillet 2020 au 27 février 2023 pendant 652 séances.
Leur continu conserve, lui, un épisode de 773 séances, rendement net
-42,30 %. Le détecteur modifie davantage le comportement de cet actif,
mais trois de ses quatre épisodes binaires perdent et d'autres titres
restent short jusqu'à 774 séances.

Le rendement d'un épisode est composé sur son slot, pas la perte du
portefeuille entier. Le rendement signé de l'état mesure le mouvement de
prix depuis l'entrée ; il n'est pas le rendement composé du short ni son
PnL pondéré par les tailles successives.

## Budget de calcul et apprentissage

Une update AdamW globale par époque, plafond 300, patience 20. Les 81 fits
arrêtent par early stopping, entre 22 et 217 époques, checkpoints entre 2
et 197. Aucun `budget_insufficient` n'est signalé ; cela ne démontre pas
un optimum global ou une recherche de paramètres suffisante.

| Architecture | Paramètres | Temps d'entraînement CE moyen | Temps financier moyen |
| --- | ---: | ---: | ---: |
| Détecteur | 7 467 | 11,89 min | 18,64 min |
| Deux têtes | 7 510 | 8,85 min | 22,71 min |
| Deux GRU | 14 934 | 10,58 min | 31,73 min |
| S3 trois classes | 7 553 | 8,45 min | 20,58 min |

Temps enregistrés par fit, pas durée totale du script avec inférence,
exports, graphiques et reprises. Les fits cumulés représentent 21,59 heures.
Les deux GRU ont presque deux fois les paramètres des deux têtes et coûtent
1,40 fois leur temps financier moyen. Les durées dépendent aussi du nombre
d'époques retenu ; ce n'est pas une comparaison à budget d'updates égal.

## Décisions pour la suite

1. Ne pas promouvoir les deux GRU comme nouveau défaut. Garder le témoin
   S3 détaché et always-long, avec le détecteur financier comme challenger
   de faible capacité supplémentaire, sans choisir une seed gagnante.
2. Donner priorité au contrôle Long/Flat : vérifier si l'abstention améliore
   le timing et le risque à exposition comparable. Tester une politique
   entraînée de façon cohérente avec son exécution, pas seulement une
   projection d'un modèle entraîné Long/Short continu.
3. Avant une nouvelle grande grille CE binaire, diagnostiquer la rotation
   et le feedback. Si des seuils ou une calibration sont testés, les choisir
   uniquement sur TRAIN/inner, avec une recette fixée avant la nouvelle
   lecture OUTER. Un seuil 0,5 sur scores pondérés n'est pas une calibration.
4. Pour les shorts, mesurer une valeur marginale face à Flat sur les mêmes
   alertes, les erreurs pendant les hausses, l'amplitude des baisses captées,
   les sorties des pertes persistantes et le coût réel du prêt. Ne pas
   déduire un gain d'un simple nombre d'épisodes plus faible.
5. Garder les cas AAPL et les couples à exposition extrême comme diagnostics,
   pas comme blacklist ou critères de sélection après coup. Une autre
   représentation d'état ou une règle de risque serait un nouveau protocole,
   à ablater explicitement, pas un correctif à injecter silencieusement.
6. Laisser le holdout final fermé jusqu'au gel du protocole. Les OUTER
   2014-2023 ont déjà guidé plusieurs benchmarks : la répétition de nouvelles
   recettes sur ces mêmes dates n'est pas une validation indépendante.

## Vérifications, sources et limites

Les 81 manifests correspondent à la même identité. Existence, taille et
SHA-256 des 5 589 fichiers référencés, soit 4,69 Go, sont vérifiés sans erreur.
Le hash des sources Python actuelles correspond au hash enregistré. Les
préparations des folds correspondent aux signatures des fits.

Les 486 portefeuilles journaliers reproduisent les rendements, Sharpes,
drawdowns, coûts monétaires et attributions du CSV. Les 486 exports de scores
ont les mêmes clés date/ticker par fold, des scores finis dans `[0,1]` et
aucune date à partir du 22 juin 2023. Les contributions par régime se
réconcilient avec le PnL à moins de `1e-9` unité monétaire. Aucun entraînement,
nouvelle inférence ou modification d'artifact n'a été lancé pour cette lecture.

- [Complétude](../../artifacts/comparisons/us-short-decisions/report.json)
  et [recette, folds et provenance](../../artifacts/comparisons/us-short-decisions/metadata.json).
- [Trajectoires](../../artifacts/comparisons/us-short-decisions/results.csv),
  [agrégats](../../artifacts/comparisons/us-short-decisions/summary.csv)
  et [gains appariés](../../artifacts/comparisons/us-short-decisions/short-paired.csv).
- [Têtes binaires](../../artifacts/comparisons/us-short-decisions/binary-heads.csv),
  [alertes et épisodes](../../artifacts/comparisons/us-short-decisions/short-alerts.csv)
  et [régimes](../../artifacts/comparisons/us-short-decisions/regimes.csv).
- [Contre-factuels gelés](../../artifacts/comparisons/us-short-decisions/frozen-alert-counterfactuals.csv),
  [exposition commune](../../artifacts/comparisons/us-short-decisions/common-exposure.csv)
  et [oppositions](../../artifacts/comparisons/us-short-decisions/oppositions.csv).
- AAPL deux têtes financières : [trades binaires](../../artifacts/comparisons/us-short-decisions/fold-2/seed-1/financial-dual_head/long_short/overnight-sign-trades.parquet)
  et [états](../../artifacts/comparisons/us-short-decisions/fold-2/seed-1/financial-dual_head/long_short/overnight-sign-state-inputs.parquet).
- AAPL deux GRU financiers : [trades binaires](../../artifacts/comparisons/us-short-decisions/fold-2/seed-1/financial-separate_grus/long_short/overnight-sign-trades.parquet)
  et [trades continus](../../artifacts/comparisons/us-short-decisions/fold-2/seed-1/financial-separate_grus/long_short/overnight-continuous-trades.parquet).
- CE-S3 fold 2/seed 19 : [portefeuille](../../artifacts/comparisons/us-short-decisions/fold-2/seed-19/cross_entropy-S3-reference/long_short/overnight-sign-portfolio.parquet).
- [Plan fixé](../plan/short-decision-benchmark.md),
  [modules et exports](../src/short-decision-benchmark.md),
  [états S0-S3](us-trade-state-overnight.md)
  et [benchmark labels/losses](us-label-loss-post-open.md).

L'univers à historique complet est rétrospectif et survivant ; les secteurs
ne sont pas historiquement PIT. Les 143 tickers ne sont pas 143 paris
indépendants, et certaines classes d'actions partagent le même émetteur.
Les frais de prêt des shorts, financement, rémunération du cash et
disponibilité broker ne sont pas modélisés. Les coûts sur changements de
position et le proxy open ne valident pas une exécution réelle. Aucune
calibration, robustesse à d'autres coûts ou utilité de filtre de régime
n'a été testée dans cette grille. Les liens d'artifacts nécessitent les
résultats locaux, non suivis par Git.
