# Benchmark news FNSPID : contrôle de la polarité

Statut au 7 octobre 2026 : les 81 entraînements sont terminés. Les résultats et
les 486 artefacts enregistrés ont été vérifiés lors de l'analyse ; le holdout
reste fermé. Je ne retiens pas la polarité FinBERT comme amélioration validée.
Le neutralisé gagne en moyenne, mais ce gain reste fragile.

## Objectif

Je vérifie si les scores des titres FNSPID ajoutent une information utile au
GRU, au-delà de l'activité et de la disponibilité des news. Deux contrôles :
mélanger les paquets de scores et neutraliser les canaux de polarité.

## Données

L'inventaire couvre les 143 tickers du Parquet US, tous avec prix complets
du 9 mars 2020 au 29 décembre 2023. La sélection figée exige au moins 60
séances avec news et 10 % de séances observées dans le premier TRAIN, du
1er juillet 2020 au 15 juin 2021 inclus. Aucun classement par performance
n'intervient : 53 actions passent ces règles, 90 sont exclues pour les news.

L'export contient 50 933 décisions ticker/séance, dont 30 793 avec news observées.
Checkpoint FinBERT figé : 14 931 textes uniques réutilisent le cache vérifié,
94 719 sont nouvellement scorés. Pas de rescoring à chaque entraînement.

Décision à minuit UTC, disponibilité supposée à publication + 24 heures,
fenêtre d'agrégation de 24 heures et cutoff strict. Cette disponibilité est
exploratoire, pas historiquement prouvée. Un intervalle sans article ne
prouve pas qu'aucune news n'a existé ; la couverture reste inconnue.

## Candidats

- `gru` : contrôle prix et features de marché/secteur sélectionnées sur TRAIN.
- `gru_activity` : même contrôle avec l'activité et la disponibilité des news.
- `gru_features` : GRU enrichi des features news originales.
- `gru_features_shuffled` : mêmes articles, dates, tickers, volumes et masques,
  mais paquets cohérents de scores permutés ; cinq permutations distinctes.
- `gru_features_neutralized` : même architecture enrichie, réentraînée avec
  les neuf canaux de polarité mis à zéro après normalisation.

Le neutralisé désactive `sentiment_mean`, `sentiment_std`, `p_positive_mean`,
`p_neutral_mean`, `p_negative_mean`, `positive_share`, `negative_share`,
`sentiment_ewm` et `sentiment_momentum`. Il garde les prix, le nombre de news,
la récence, `confidence_mean` et les masques de présence/disponibilité.
Zéro standardisé ne signifie pas « article classé neutre ». La confiance
FinBERT reste présente : ce n'est pas un contrôle entièrement sans NLP.
Original, mélangé et neutralisé gardent 56 entrées et 9 827 paramètres.

## Protocole

GRU de taille 32, une couche, pooling attention, contexte 60, weight decay
0,00001 et patience 15. Objectif combiné PnL/Sharpe de poids PnL 0,25 ;
positions long/short, coûts 5 bps et délai d'exécution d'une séance. Labels
triple barrier ATR, barrières 0,75, horizon 10, CUSUM 0,5 et maintien entre
événements. Pool `technical,market,sector`, cap 32, corrélation maximale 0,95.

Trois folds externes, seeds d'entraînement 1/7/19, fraction CV initiale
0,50, validation intérieure 0,20, gap 5 et embargo 0. Les quatre références
sont entraînées une fois : 36 runs. Les cinq permutations ajoutent 45 runs,
soit 81 au total. Le holdout commence le 5 juin 2023 et reste scellé.

Le mélange est séparé par ticker, fold et partition TRAIN/validation
intérieure/OUTER : aucun échange train/test. Les probabilités positive,
neutre et négative, le score et la confiance voyagent ensemble. Selon la
partition, 99,0627 % à 99,8953 % des paquets changent réellement. Les
contrôles sont donc effectifs, pas des permutations presque identiques.

Les permutations sont rétrospectives au sein de leur partition : elles
servent de contrôles nuls, pas de stratégies déployables. La normalisation
est ajustée uniquement sur TRAIN ; le checkpoint est choisi sur validation
intérieure, sans sélection par performance OUTER.

## Résultats observés

Rendements et expositions : moyennes par fold/seed, non annualisées ni
concaténées. Sharpe régularisé du simulateur. La ligne mélangée moyenne
cinq permutations, chacune sur les mêmes neuf couples fold/seed.

| Candidat | Sharpe régularisé moyen | Rendement net moyen | Exposition brute moyenne |
| --- | ---: | ---: | ---: |
| GRU prix | -0,2369 | -2,36 % | 48,03 % |
| GRU + activité news | -0,4808 | -1,28 % | 28,74 % |
| GRU + features originales | -0,0838 | -1,84 % | 46,50 % |
| GRU + features mélangées, moyenne de cinq permutations | -0,0596 | -1,79 % | 45,56 % |
| GRU + features neutralisées | +0,0869 | -0,81 % | 32,42 % |

### Cinq permutations contre les features originales

Les cinq Sharpe moyens dépassent celui de l'original. Ce constat ne signifie
pas que chaque run mélangé gagne : seuls 14 des 45 écarts appariés sont
positifs. Quatre permutations sur cinq ont un rendement moyen inférieur
à l'original ; quelques écarts importants influencent les moyennes.

| Seed de permutation | Sharpe moyen | Écart de Sharpe | Rendement net moyen | Écart de rendement, points | Gains de Sharpe appariés |
| --- | ---: | ---: | ---: | ---: | ---: |
| 314159 | -0,0564 | +0,0274 | -1,425 % | +0,4152 | 3/9 |
| 271828 | -0,0688 | +0,0150 | -1,890 % | -0,0498 | 2/9 |
| 161803 | -0,0656 | +0,0182 | -1,851 % | -0,0110 | 2/9 |
| 57721 | -0,0515 | +0,0322 | -1,892 % | -0,0511 | 4/9 |
| 141421 | -0,0560 | +0,0278 | -1,888 % | -0,0475 | 3/9 |

Sharpe moyen mélangé de -0,0688 à -0,0515, contre -0,0838 pour l'original.
Je ne sélectionne pas la meilleure permutation : contrôle, pas grille de trading.

### Neutralisé : gain moyen et instabilité

| Fold externe | Sharpe original | Sharpe neutralisé | Rendement original | Rendement neutralisé |
| --- | ---: | ---: | ---: | ---: |
| 19 octobre 2021 – 3 mai 2022 | -0,3750 | +0,0135 | -3,91 % | -2,15 % |
| 4 mai – 15 novembre 2022 | +0,8121 | +0,9006 | +0,69 % | +0,70 % |
| 16 novembre 2022 – 2 juin 2023 | -0,6884 | -0,6534 | -2,30 % | -0,99 % |

Le neutralisé gagne en Sharpe sur seulement quatre des neuf couples.
Deux runs seed 7, folds 0 et 2, expliquent 90,56 % du gain total de Sharpe.
Leurs checkpoints neutralisés sont retenus à l'époque 1, contre les
époques 57 et 55 pour l'original. L'exposition brute tombe respectivement
de 92,38 % à 17,56 % et de 84,36 % à 31,91 %. Un passage presque entièrement
long vers short contribue aussi au premier cas : le gain ne démontre pas
une supériorité stable de l'information neutralisée.

### Exposition moyenne commune

Une réduction constante des positions par fold/seed ramène les quatre
références, les cinq permutations et le buy-and-hold à la plus faible
exposition moyenne du groupe : 3,30 % à 12,03 %, moyenne
6,95 %. Turnover et coûts sont recalculés. Ce contrôle utilise tout le fold :
diagnostic ex post, pas règle de sizing déployable.

| Candidat | Sharpe régularisé moyen | Rendement net moyen |
| --- | ---: | ---: |
| Buy-and-hold du simulateur | +0,1856 | +0,0762 % |
| Features originales | -0,0804 | -0,0929 % |
| Features neutralisées | +0,0895 | +0,0233 % |

Le neutralisé reste derrière le buy-and-hold redimensionné en rendement moyen.
Exposition brute égale ne signifie ni même risque, ni même beta, ni même
direction des positions ; ce tableau n'établit pas un alpha.

## Décision et limites

Je ne retiens pas la polarité FinBERT comme amélioration robuste démontrée
sur ces 53 actions. Je conserve le neutralisé comme contrôle exploratoire,
pas comme vainqueur validé. Avant d'élargir encore, je donne priorité au
diagnostic des positions, de l'exposition et de l'early stopping.

Les neuf couples et les 45 runs mélangés ne sont pas indépendants : trois
périodes de marché seulement. Cinq permutations ne suffisent pas à une
conclusion universelle sur le sentiment. L'univers est survivant et filtré
rétrospectivement sur les prix complets ; il diffère du pilote sur cinq actions.
FNSPID ne prouve ni couverture historique ni intégrité des versions à
publication ; le cutoff d'entraînement de l'encodeur n'est pas vérifié.
Le holdout fermé ne valide pas encore une stratégie finale.

Sources : [rapport élargi](../../artifacts/comparisons/fnspid-news-expanded-permutations/report.json),
[références et détail des runs](../../artifacts/comparisons/fnspid-news-expanded-permutations/reference/folds.json),
[manifest du protocole](../../artifacts/comparisons/fnspid-news-expanded-permutations/study.manifest.json),
[inventaire et règles figées](../../data/derived/fnspid/inventory-v1/inventory-report.json),
[import FNSPID](../../data/derived/fnspid/inventory-v1/import/import-report.json),
[scoring figé](../../data/derived/fnspid/expanded-v1/scoring/scoring.manifest.json),
[export quotidien](../../data/derived/fnspid/expanded-v1/company_daily.manifest.json),
[manuel PC](../plan/fnspid-pc-manual.md) et [plan du pilote](../plan/news-sentiment-pilot.md).
