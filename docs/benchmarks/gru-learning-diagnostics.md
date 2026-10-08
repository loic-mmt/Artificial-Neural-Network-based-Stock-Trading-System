# GRU32 US : diagnostic de l'apprentissage et des positions

Statut au 8 octobre 2026 : neuf entraînements terminés et vérifiés, avec traces
par époque. Aucun holdout final ouvert, aucune modification de l'entraînement
pendant l'analyse. Ce diagnostic complète les résultats
[32/64 features et gates](us-feature-gate-interaction.md), sans recommencer leur grille.

## Protocole et contrôles

143 actions US du calendrier PC, GRU seul, cap 32, contexte 60, attention temporelle,
hidden size 32, une couche, tête linéaire, weight decay `1e-5`. Loss combinée
PnL/Sharpe, poids PnL 0,25, `pnl_scale=1e-4`, epsilon Sharpe `1e-4`, coûts 5 bps,
signal après clôture et délai d'exécution d'une séance. Trois folds purgés,
seeds 1/7/19, mêmes calendriers de développement que le contrôle GRU32 précédent.

Les neuf manifestes et les 18 partitions INNER/OUTER sont cohérents : identités,
clés date/ticker, logits, probabilités, positions, prix, coûts et trajectoires
financières vérifiés. Couverture des signaux 100 %. Aucun checkpoint désérialisé.
Le holdout commence le 22 juin 2023 et reste fermé.

Le nouveau Sharpe régularisé OUTER moyen vaut 0,803992, contre 0,804413 pour
l'ancien GRU32. Les calendriers, prix, labels et listes de features exportés
concordent. Les environnements Mac/PC, dépendances et signatures diffèrent :
ce rapprochement n'est pas une promesse de reproductibilité bit à bit.

## Résultats OUTER

Moyennes descriptives sur les neuf couples fold/seed, pas rendement d'un
portefeuille concaténé. Les trois seeds partagent les dates de chaque fold.

| Mesure | GRU32 | Buy-and-hold du même univers |
| --- | ---: | ---: |
| Sharpe régularisé | 0,8040 | 0,9359 |
| Sharpe net ordinaire | 0,8067 | 0,9359 |
| Rendement net | +23,70 % | +55,89 % |
| Exposition brute exécutée | 46,51 % | 99,87 % |
| Drawdown maximal moyen | -10,48 % | -22,36 % |

| Fold | Seed | Époque retenue / exécutées | Rendement net | Sharpe net | Exposition brute | Signaux SHORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 21 / 36 | +5,56 % | 1,014 | 14,22 % | 13,72 % |
| 0 | 7 | 8 / 23 | +3,69 % | 1,055 | 8,79 % | 10,01 % |
| 0 | 19 | 74 / 89 | +19,53 % | 0,998 | 63,46 % | 27,62 % |
| 1 | 1 | 33 / 48 | +5,61 % | 0,453 | 22,56 % | 13,40 % |
| 1 | 7 | 15 / 30 | +3,14 % | 0,358 | 13,62 % | 7,80 % |
| 1 | 19 | 100 / 100 | +25,71 % | 0,451 | 97,05 % | 1,40 % |
| 2 | 1 | 100 / 100 | +73,34 % | 1,167 | 96,13 % | 2,54 % |
| 2 | 7 | 85 / 100 | +75,04 % | 1,165 | 96,12 % | 1,58 % |
| 2 | 19 | 6 / 21 | +1,69 % | 0,599 | 6,62 % | 42,18 % |

Le B&H des folds 0/1/2 rapporte respectivement 53,61 % / 33,89 % / 80,18 %, pour
Sharpes nets 1,1125 / 0,5365 / 1,1588. Sept des neuf runs ont un Sharpe inférieur
au B&H, même avant coûts. Les deux dépassements, fold 2 seeds 1/7, restent faibles.

Huit runs sur neuf ont des rendements quotidiens corrélés au B&H entre 0,901
et 0,997. Fold 2 seeds 1/7 : environ 92 % des signaux saturent à `abs(q)>=0.95`
et 97 à 98 % sont LONG ; corrélation entre leurs trajectoires quotidiennes 0,9992.
Cette ressemblance ne démontre pas un alpha robuste.

Les coûts OUTER cumulés représentent 7,77 à 102,27 bps de rendement sur 773/774
intervalles. Ils ne suffisent pas à expliquer les écarts de performance.
Fold 2 seed 19 est aussi plus concentré : contribution GME 57,23 sur PnL total 168,84.
Il s'agit de contributions au portefeuille, pas de backtests par titre autonomes.

## Traces et objectif réellement appris

Les neuf traces totalisent 547 époques. Loss TRAIN strictement décroissante sur
les 538 transitions ; normes globales de gradients finies, de 0,7089 à 15,4169.
Clipping au seuil 1 dans 373/547 époques, soit 68,2 %. Pas d'échec numérique global
manifeste ; cette mesure ne certifie pas l'apprentissage de chaque sous-partie.

Sept arrêts par patience, deux au plafond avec checkpoint 100. Trois runs
exécutent 100 époques, car fold 2 seed 7 termine aussi sa patience à cette époque.

Fold 2 seed 19, époque 6 vers 21 : TRAIN passe de -0,2667 à -0,9082, alors que
validation passe de -0,7472 à -0,5406. L'amélioration TRAIN ne généralise donc
pas à cette validation ; early stopping conserve correctement époque 6.
TRAIN est observé avant update, validation après update ; leur différence
absolue n'est pas un gap classique mesuré dans les mêmes conditions.
Dropout et head dropout sont nuls dans ce GRU.

Le score opposé à la loss est :

```text
0.75 * regularized_sharpe + 0.25 * mean_daily_net_return / 0.0001
```

Au checkpoint INNER, le terme PnL représente 63,8 % du score fold 1 seed 19 et
74,1 % / 74,3 % fold 2 seeds 1/7, contre 10,0 % fold 2 seed 19. Poids 0,25 ne signifie
donc pas 25 % du score observé. Le terme PnL favorise l'amplitude lorsque le
rendement moyen est positif ; le Sharpe régularisé avec epsilon fixe n'est
pas parfaitement invariant à l'échelle non plus. La causalité exclusive du
terme PnL dans la saturation n'est pas démontrée.

Les traces n'enregistrent ni les probabilités ni les composantes séparées de
loss par époque. La décomposition ci-dessus est reconstruite au checkpoint
depuis les métriques INNER, pas inventée pour chaque époque.

## Shorts, décisions et exemple AAPL

Des shorts existent : 133 204 / 996 996 signaux OUTER, soit 13,36 %. Il ne s'agit pas
d'un nombre de transactions. Le côté short contribue négativement aux
rendements quotidiens bruts des neuf runs, avant frais. Aucun verrou long-only :
`q=P(Buy)-P(Sell)` peut être négatif. Les périodes haussières rendent une
solution dominée par les longs plausible, sans expliquer à elles seules son origine.

Sur AAPL, fold 2 seed 1, 772/773 intervalles sont short ; position moyenne environ
-0,5012, cours ajusté en hausse de 136,63 % sur la fenêtre du 26 mai 2020 au
21 juin 2023. Entrée short effective le 27 mai 2020, liquidation le 21 juin 2023 :
environ trois ans, pas quatre.
La taille est néanmoins ajustée pendant cet épisode : un seul changement de
sens n'est pas un seul ordre ni une position de quantité constante.

Les labels connus comportent 26,7 à 28,1 % de Sell selon les folds. Leur absence
n'explique donc pas le peu de shorts de certains checkpoints. Mais ce run
optimise une loss financière, sans cross-entropy : les labels servent aux
métriques et à l'éligibilité/purge, pas directement au gradient.
L'horizon triple-barrier 10 borne la construction du label, pas la durée réelle
du portefeuille ; aucune sortie TP/SL n'est imposée par ce `ReturnPanel`.

## Graphiques

L'explorateur couvre les 143 tickers et neuf checkpoints. Signal de clôture J,
entrée à clôture J+1, rendement jusqu'à clôture J+2. Triangles = entrées/inversions
de sens, avec tolérance descriptive `1e-6`, pas tous les ajustements de taille.
La liquidation à la dernière clôture est imposée par le protocole.
Dans ce simulateur, poids d'un ticker = `q/143`, pas allocation `equal_active`.

- [Explorateur interactif, autonome et sans réseau](../../artifacts/diagnostics/learning-trades-gru32-combined-025-20261008/explore-trades.html).
- [Expositions effectives des neuf runs](../../artifacts/diagnostics/learning-trades-gru32-combined-025-20261008/portfolio-exposure.png).
- [Cours et entrées AAPL/NVDA/JPM du fold 2](../../artifacts/diagnostics/learning-trades-gru32-combined-025-20261008/ticker-entries-fold2.png).

## Historique des labels et de l'exécution

Dans l'historique Git disponible, aucune sortie triple-barrier n'a été retirée.
Le premier commit contenant le labeler, `6ef0147` du 7 septembre 2026, borne
déjà seulement le calcul du label futur et de `label_end_date`. La seule
modification ultérieure du fichier, `4f6e2a9` du 3 octobre, concerne les dates
UTC des statistiques de durée. Le premier commit visible regroupe du travail
antérieur : sa date ne prouve pas le jour exact de conception de la méthode.

La voie financière a été ajoutée comme expérience distincte, sans retirer les
trainers cross-entropy. Le [plan historique M7](../papers/old_docs/labeling-implementation-plan.md)
prévoyait explicitement une position continue optimisant Sharpe/PnL plutôt
qu'une cible de classification. Le [document financier initial](../papers/old_docs/financial-loss.md)
conserve cette séparation. Les règles d'exécution SL/TP et `max_holding_bars`
sont dans le moteur OHLC optionnel ajouté par `b9bcc24` le 29 septembre ;
elles ne sont pas appliquées par le `ReturnPanel` de ce benchmark.

Avant toute supervision hybride, il faut aussi contrôler l'alignement temporel :
le label actuel est ancré à la clôture J et regarde dès J+1, alors que le run
entre à la clôture J+1 et gagne le rendement suivant. Une barrière touchée dès
J+1 peut donc représenter un mouvement non capturable avec cette exécution.
Il faut également distinguer Hold comme conservation d'une position d'un
état cible Flat : le décodeur continu n'interprète pas ces deux notions pareil.
Ces contrats sont à définir ; aucun label ni runner n'est modifié par ce constat.

## Conclusions et suite

Sharpe seul a déjà été comparé au combiné :
[benchmark CARA/combiné](gru-optim/05-cara-combined.md), Sharpe 0,7509 et rendement 7,96 %,
contre 0,6918 et 37,13 % pour combiné 0,25, avec drawdown aggravé 9/9. Il ne faut
pas présenter ce compromis comme une nouvelle découverte ni relancer une grille
générique pour le constater. Ces résultats historiques ne sont pas un témoin
strictement apparié aux 143 actions du nouveau run.

Les [contrôles d'exposition du benchmark 32/64](us-feature-gate-interaction.md#exposition-comparable)
existent déjà. Ils ne neutralisent pas entièrement risque, beta ou direction.
Le [walk-forward MT5 US](mt5-us.md) avait 67,83 % de signaux SHORT : la domination
LONG du run actuel n'est pas une impossibilité structurelle de prendre des shorts.

Priorité : distinguer supervision des labels, optimisation financière et
politique d'entrée/sortie ; examiner timing et état de position avant d'augmenter
la complexité du modèle. Un objectif supervisé/financier hybride, des labels
plus sélectifs et un modèle conditionné par l'état du portefeuille sont des
pistes à définir puis tester, pas des correctifs déjà validés.
Les [ablations des sorties trading](trading-exit-ablation.md) montrent déjà
qu'une durée maximale ou un stop peut dégrader résultat et bloquer les réentrées.
Une sortie forcée doit être évaluée avec son contrat de réentrée et ses coûts.

Sources : [run tracé](../../artifacts/comparisons/learning-diagnostics/gru32-combined-025-v1/report.json),
[métadonnées](../../artifacts/comparisons/learning-diagnostics/gru32-combined-025-v1/metadata.json),
[résultats et traces](../../artifacts/comparisons/learning-diagnostics/gru32-combined-025-v1/folds.json),
[loss et positions](../src/financial-loss.md), [outil de diagnostic](../src/learning-diagnostics.md).
