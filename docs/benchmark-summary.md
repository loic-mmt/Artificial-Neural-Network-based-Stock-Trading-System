# Benchmark summary

Dernière mise à jour : 2026-09-18.

Ce document consolide les décisions, le protocole gelé et les principales
limites identifiées après chaque benchmark. Les anciens plans et journaux
devenus obsolètes sont archivés localement dans `docs/papers/old_docs/` et ne
font plus partie du dépôt suivi.

## Conventions et limites

- Sauf indication contraire, les valeurs sont des moyennes sur folds et seeds.
- Le holdout final commençant le 2022-05-10 est toujours fermé.
- Les folds ont des historiques d'entraînement qui se chevauchent : ils ne sont
  pas des réplications indépendantes.
- L'univers utilise des constituants actuels sur tout l'historique : les résultats
  restent exposés au biais de survivance.
- Le benchmark 1 utilise l'ancien parquet non nettoyé. À partir du benchmark 2,
  la référence est `data/processed/cac40_daily_clean.parquet`, SHA-256
  `dce05c66d89d76a16649389b2482139aee82dd791504163efb83f5f10700ae2c`.
- Jusqu'au benchmark 9, les 5 bps configurés appartiennent aux labels ; les
  backtests historiques affichés ne débitent pas de coûts de transaction.
- À partir du benchmark 11, les métriques continues débitent explicitement les
  coûts proportionnels déclarés. Elles ne sont donc pas directement comparables
  aux anciens backtests de classification.
- Le benchmark 1 compare cinq modèles. Les benchmarks 2 à 6 comparent GRU, RNN
  et Transformer. Les benchmarks 7 à 20 et 22 ont ensuite été restreints au GRU
  pour réduire le coût de calcul. Le benchmark 21 ayant été sauté, la
  configuration financière finale n'a pas été revalidée sur les autres
  architectures.

## Avancement

| Benchmark | Statut | Décision |
| --- | --- | --- |
| 1 | Terminé, historique | GRU meilleur classifieur de départ |
| 2 | Terminé | `rolling_std` |
| 3 | Terminé | barrières `0.75 / 0.75` |
| 4 | Terminé | filtre CUSUM `0.5` |
| 5 | Terminé | politique `hold` |
| 6 | Terminé | horizon `10` |
| 7 | Terminé | FracDiff désactivé |
| 8 | Terminé | sample weighting désactivé |
| 9 | Terminé | `technical,market,sector` selon score primaire |
| 10 | Sauté | données externes point-in-time absentes |
| 11 | Terminé | loss Sharpe à 5 bps |
| 12 | Terminé | Sharpe à 0–5 bps ; PnL à 10–20 bps |
| 13 | Terminé | contrôle du surapprentissage activé |
| 14 | Terminé | 32 features, corrélation maximale 0,95 |
| 15 | Terminé | règles de stabilité validées par tests |
| 16 | Terminé, GRU seul | 1 couche, weight decay `1e-5` selon score primaire |
| 17 | Terminé | 3 folds principaux ; 5/10 diagnostics |
| 18 | Terminé | gap `5` conservé |
| 19 | Terminé | embargo `0` |
| 20 | Terminé | initial `0.50`, validation interne `0.20` |
| 21 | Sauté | comparaison finale des architectures non relancée |
| 22 | Terminé | features techniques + marché + secteur |
| 23 | En attente | holdout final encore scellé |

## 1 - Reference baseline

**But.** Comparer les cinq architectures avec labels Triple Barrier, features
techniques, cross-entropy et données non nettoyées.

| Modèle | Macro-F1 | Rendement | Sharpe | Runs positifs |
| --- | ---: | ---: | ---: | ---: |
| Manual ANN | 0,467 | −4,36 % | −0,016 | 3/9 |
| RNN | 0,505 | −9,17 % | −0,040 | 2/9 |
| LSTM | 0,504 | −11,79 % | −0,068 | 4/9 |
| **GRU** | **0,524** | −4,79 % | −0,023 | 4/9 |
| Transformer | 0,486 | −7,53 % | −0,076 | 5/9 |

**Décision.** GRU retenu comme architecture principale. Résultat historique non
comparable aux runs nettoyés suivants.

**Source.** [Rapport](../artifacts/comparisons/01-baseline/report.json).

## 2 - Triple Barrier volatility estimator

**But.** Comparer `rolling_std`, ATR et largeur de Bollinger.

Chaque cellule contient `macro-F1 / rendement historique`.

| Estimateur | GRU | RNN | Transformer |
| --- | ---: | ---: | ---: |
| **rolling_std** | **0,530 / −6,39 %** | 0,502 / −6,00 % | 0,491 / −10,08 % |
| ATR | **0,473 / +3,92 %** | 0,446 / −8,98 % | 0,381 / −26,90 % |
| Bollinger | **0,437 / −6,40 %** | 0,414 / −11,41 % | 0,405 / −14,36 % |

**Décision.** `rolling_std` retenu par macro-F1 et par la sélection agrégée du
lanceur. ATR reste un ancien challenger financier, sans avantage robuste.

**Sources.** `artifacts/comparisons/ohlc-clean/02-estimator-*`.

## 3 - Triple Barrier profit/stop multiples

**But.** Tester cinq couples de barrières avec `rolling_std` et CUSUM.

| Profit / stop | Macro-F1 moyen, tous modèles | P&L moyen |
| --- | ---: | ---: |
| **0.75 / 0.75** | **0,510** | **−595,94** |
| 0.75 / 1.00 | 0,504 | −1 139,08 |
| 1.00 / 0.75 | 0,508 | −759,98 |
| 1.00 / 1.00 | 0,508 | −749,05 |
| 1.50 / 1.50 | 0,456 | −760,97 |

**Décision.** `0.75 / 0.75`. Les branches nommées `macro-f1` et `pnl` sont des
runs cross-entropy identiques ; seul leur critère de sélection différait.

GRU domine les deux autres architectures pour chacun des cinq couples :

| Profit / stop | GRU F1 | RNN F1 | Transformer F1 |
| --- | ---: | ---: | ---: |
| **0.75 / 0.75** | **0,529** | 0,500 | 0,501 |
| 0.75 / 1.00 | **0,523** | 0,504 | 0,485 |
| 1.00 / 0.75 | **0,529** | 0,508 | 0,488 |
| 1.00 / 1.00 | **0,530** | 0,502 | 0,491 |
| 1.50 / 1.50 | **0,491** | 0,453 | 0,423 |

**Sources.** `artifacts/comparisons/ohlc-clean/03-*-barriers-*`.

## 4 - Triple Barrier event filter

**But.** Comparer événements sur toutes les lignes et événements CUSUM.

| Filtre | Modèle | Macro-F1 | Rendement historique |
| --- | --- | ---: | ---: |
| `all` | GRU | 0,188 | −28,98 % |
| `all` | RNN | 0,187 | −22,82 % |
| `all` | Transformer | **0,197** | **−19,39 %** |
| **CUSUM** | **GRU** | **0,529** | **+6,48 %** |
| **CUSUM** | RNN | 0,500 | −8,13 % |
| **CUSUM** | Transformer | 0,501 | −16,23 % |

**Décision.** CUSUM `0.5`. Le run CUSUM reproduit la configuration gagnante du
benchmark 3 ; il ne constitue pas une réplication indépendante.

**Sources.** `artifacts/comparisons/ohlc-clean/04-*-event-filter-*`.

## 5 - Triple Barrier between-event policy

**But.** Comparer labels d'action `hold` et cibles de position `flat`/`carry`.

| Politique | Modèle | Macro-F1 | Rendement historique | Runs positifs |
| --- | --- | ---: | ---: | ---: |
| **hold** | **GRU** | **0,529** | **+6,48 %** | **6/9** |
| hold | RNN | 0,500 | −8,13 % | 3/9 |
| hold | Transformer | 0,501 | −16,23 % | 1/9 |
| flat | **GRU** | **0,529** | −0,24 % | **6/9** |
| flat | RNN | 0,500 | −9,32 % | 1/9 |
| flat | Transformer | 0,501 | −11,40 % | 2/9 |
| carry | GRU | 0,173 | **+6,77 %** | **4/9** |
| carry | RNN | 0,174 | −1,95 % | 3/9 |
| carry | Transformer | **0,177** | −15,53 % | 3/9 |

**Décision.** `hold`. `carry` change fortement la sémantique et devient peu
apprenable ; `flat` ne gagne rien face à `hold`.

**Sources.** `artifacts/comparisons/ohlc-clean/05-policy-*`.

## 6 - Triple Barrier horizon

**But.** Comparer horizons 5, 10 et 20 observations.

| Horizon | Modèle | Macro-F1 | Rendement historique | Runs positifs |
| ---: | --- | ---: | ---: | ---: |
| 5 | **GRU** | **0,498** | −10,33 % | 3/9 |
| 5 | RNN | 0,472 | −20,51 % | 2/9 |
| 5 | Transformer | 0,446 | −13,22 % | 2/9 |
| **10** | **GRU** | **0,529** | **+6,48 %** | **6/9** |
| 10 | RNN | 0,500 | −8,13 % | 3/9 |
| 10 | Transformer | 0,501 | −16,23 % | 1/9 |
| 20 | **GRU** | **0,524** | −14,80 % | 3/9 |
| 20 | RNN | 0,503 | −16,18 % | 2/9 |
| 20 | Transformer | 0,493 | **+0,29 %** | **4/9** |

**Décision.** Horizon `10`.

**Sources.** `artifacts/comparisons/ohlc-clean/06-horizon-*`.

## 7 - FracDiff activation

**But.** Comparer données inchangées, ordre automatique et ordres fixes.

**Portée.** GRU uniquement, après sa sélection répétée aux benchmarks 1 à 6.

| Variante, GRU | Macro-F1 | Rendement historique |
| --- | ---: | ---: |
| **Sans FracDiff** | 0,529 | **+6,48 %** |
| ordre 0,3 | 0,526 | −15,72 % |
| ordre 0,5 | **0,535** | −1,78 % |
| ordre 0,7 | 0,532 | +1,68 % |
| automatique | 0,533 | −12,06 % |

**Décision.** FracDiff désactivé. Petit gain de F1 insuffisant face à la baisse
financière et à la complexité ajoutée.

**Sources.** `artifacts/comparisons/ohlc-clean/07-fracdiff-*`; contrôle sans
FracDiff dans `06-horizon-10`.

## 8 - Sample weighting

**But.** Tester pondérations par rendement net, volatilité et unicité.

**Portée.** GRU uniquement.

| Pondération, GRU | Macro-F1 | Rendement historique |
| --- | ---: | ---: |
| **Aucune** | 0,529 | **+6,48 %** |
| rendement net | 0,531 | −4,31 % |
| volatilité | **0,534** | −5,11 % |
| unicité | 0,512 | +1,32 % |

**Décision.** Aucune pondération. Les gains marginaux de F1 ne produisent pas de
gain financier robuste.

**Sources.** `artifacts/comparisons/ohlc-clean/08-weight-*`; contrôle non pondéré
dans `06-horizon-10`.

## 9 - Feature-family ablation

**But.** Mesurer valeur marginale des features techniques, marché et secteur.

**Portée.** GRU uniquement.

| Features, GRU | Macro-F1 | Rendement historique | Runs positifs |
| --- | ---: | ---: | ---: |
| techniques | 0,527 | −8,42 % | 1/9 |
| techniques + marché | 0,549 | **+13,74 %** | **5/9** |
| techniques + secteur | 0,518 | −9,63 % | 1/9 |
| **techniques + marché + secteur** | **0,556** | +13,25 % | 4/9 |

**Décision.** Combinaison complète selon le score primaire macro-F1. La variante
marché seule était légèrement meilleure financièrement et restait challenger.

**Sources.** `artifacts/comparisons/ohlc-clean/09-features-*`.

## 10 - Historical fundamentals and sentiment

**Statut.** Sauté. Les historiques point-in-time avec `available_at` ne sont pas
encore disponibles. Utiliser des snapshots actuels créerait une fuite future.

**Décision.** Pipeline préparé, benchmark reporté jusqu'à disponibilité des deux
fichiers historiques fiables.

## 11 - Loss objective

**But.** Comparer cross-entropy, PnL net direct et Sharpe régularisé avec coûts
effectifs de 5 bps.

| Loss | Rendement net | Sharpe régularisé | Drawdown | Turnover | Runs + |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-entropy | −2,46 % | −0,251 | −9,36 % | 156,1 | 3/9 |
| PnL | **+45,39 %** | 0,617 | −27,40 % | 36,2 | 9/9 |
| **Sharpe** | +8,55 % | **0,634** | **−5,87 %** | **33,6** | 9/9 |

**Décision.** Loss Sharpe à 5 bps : meilleur compromis risque/rendement. PnL
reste challenger à forte exposition.

**Source.** [Rapport corrigé](../artifacts/comparisons/ohlc-clean/11-loss-objectives-v2/report.json).
Le premier dossier `11-loss-objectives` contient le run échoué avant alignement
des calendriers et ne doit pas être utilisé.

## 12 - Transaction-cost stress

**But.** Tester le classement des losses sous 0, 5, 10 et 20 bps.

| Coût | Loss gagnante | Rendement | Sharpe régularisé | Turnover |
| ---: | --- | ---: | ---: | ---: |
| 0 bps | Sharpe | +7,98 % | **0,710** | 34,1 |
| 5 bps | Sharpe | +8,55 % | **0,634** | 33,6 |
| 10 bps | PnL | **+47,78 %** | 0,634 | 22,1 |
| 20 bps | PnL | **+50,23 %** | 0,647 | 11,6 |

**Décision.** Conserver Sharpe à l'hypothèse centrale de 5 bps. Les modèles sont
réentraînés pour chaque coût : la hausse du rendement PnL avec les coûts provient
d'une position plus persistante, pas d'un bénéfice mécanique des frais.

**Sources.** `artifacts/comparisons/ohlc-clean/12-cost-*bps`; point 5 bps dans
`11-loss-objectives-v2`.

## 13 - Overfitting-control profile

**But.** Comparer profil standard et contrôle opt-in sur GRU + loss Sharpe.

| Profil | Rendement | Sharpe | Drawdown | Turnover | σ Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | **+8,55 %** | 0,634 | −5,87 % | 33,57 | 0,246 |
| **On** | +7,63 % | **0,659** | **−4,32 %** | **24,65** | **0,190** |

**Décision.** Activer `--overfitting-control`. Rendement légèrement inférieur,
mais meilleur Sharpe, drawdown, turnover et pire cas. Le profil réduit 95 features
à 32 et applique GRU 32, weight decay `1e-4`, patience 15.

**Sources.** `artifacts/comparisons/ohlc-clean/13-overfitting-{off,on}`.

## 14 - Feature-selection sensitivity

**But.** Tester limites de 16/32/64 features et corrélations 0,90/0,95/0,98.

| Configuration | Rendement | Sharpe | σ Sharpe | Pire Sharpe | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 / 0,98 | +8,82 % | **0,694** | 0,247 | 0,172 | 22,0 |
| **32 / 0,95** | +9,13 % | 0,685 | **0,139** | 0,390 | 24,0 |
| 64 / 0,90 | **+10,48 %** | 0,671 | 0,171 | **0,401** | 34,7 |

**Décision.** 32 features, corrélation maximale 0,95. Presque le meilleur Sharpe,
mais meilleure dispersion globale et complexité modérée. Chaque limite était
atteinte ; la sélection est bien contraignante.

**Sources.** `artifacts/comparisons/ohlc-clean/14-selector-*`; point 32/0,98 dans
`13-overfitting-on`.

## 15 - Stability-gate sensitivity

**But.** Vérifier mécaniquement les règles de gouvernance, sans réutiliser les
données de marché ni ouvrir le holdout.

**Résultat.** 2 tests passés, 7 non sélectionnés. Validation de l'écart-type seed
maximal 0,05, du gap train/validation maximal 0,15, de la couverture complète des
seeds et du maintien du holdout fermé pour un candidat instable.

**Décision.** Seuils gelés ; ne pas les optimiser sur le holdout.

## 16 - GRU regularization

**But.** Comparer profondeur, dropout, weight decay et patience du GRU.

| GRU | Rendement | Sharpe | σ Sharpe | Pire Sharpe | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| **1 couche, WD `1e-5`** | **+9,13 %** | **0,685** | 0,139 | 0,390 | 24,03 |
| 2 couches, dropout 0,3, WD `1e-4` | +7,36 % | 0,674 | **0,123** | 0,477 | **11,39** |
| 2 couches, dropout 0,5, WD `1e-3` | +8,42 % | 0,682 | 0,123 | **0,483** | 12,84 |

**Décision.** Le score primaire sélectionne 1 couche et weight decay `1e-5`.
La variante fortement régularisée reste challenger robuste : Sharpe presque
identique et turnover divisé par environ 1,9.

**Source.** [Rapport](../artifacts/comparisons/ohlc-clean/16-gru-regularization/report.json).

## 17 - Purged-CV fold count

**But.** Comparer 3, 5 puis 10 folds, ce dernier ajouté comme diagnostic.

| Folds | Sharpe moyen | σ Sharpe | Pire Sharpe | Runs + | Rendement quotidien moyen |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **3** | 0,685 | **0,139** | **0,390** | 9/9 | 0,00934 % |
| 5 | 0,712 | 0,285 | 0,143 | 15/15 | 0,00891 % |
| 10 | **0,830** | 0,786 | −0,430 | 25/30 | 0,00995 % |

**Décision.** 3 folds pour la sélection principale : fenêtres longues et score
stable. 5 et 10 folds restent des diagnostics de régimes. Le score moyen élevé à
10 folds est gonflé par certaines courtes périodes très favorables.

**Sources.** `artifacts/comparisons/ohlc-clean/17-cv-folds-*`.

## 18 - Purged-CV gap

**But.** Tester gaps 0, 5 et 10 avec purging des intervalles d'information.

| Gap | Rendement | Sharpe | σ Sharpe | Pire Sharpe |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +8,56 % | **0,689** | 0,139 | 0,439 |
| **5** | **+9,13 %** | 0,685 | 0,139 | 0,390 |
| 10 | +9,01 % | 0,688 | **0,130** | **0,480** |

**Décision.** Gap 5 conservé comme compromis pré-déclaré. Le modèle est peu
sensible au gap parce que le purging Triple Barrier retire déjà les événements
qui chevauchent les frontières.

**Sources.** `artifacts/comparisons/ohlc-clean/18-cv-gap-*`; point gap 5 dans
`17-cv-folds-3`.

## 19 - Purged-CV embargo

**But.** Confirmer la sémantique de l'embargo dans une CV expanding past-only.

**Résultat.** Embargo 5 et embargo 0 donnent exactement les mêmes métriques,
checkpoints et états de purging. Nombre d'exclusions supplémentaires : zéro.

**Décision.** Embargo 0. Aucun échantillon d'entraînement n'existe après la
validation dans ce protocole, donc l'embargo post-validation est sans effet.

**Sources.** [Embargo 5](../artifacts/comparisons/ohlc-clean/19-cv-embargo-5/report.json)
et contrôle embargo 0 dans `17-cv-folds-3`.

## 20 - Purged-CV history fractions

**But.** Tester historique initial 0,40/0,50/0,60 et validation interne
0,15/0,20/0,25.

| Initial | Inner val | Sharpe | σ Sharpe | Pire Sharpe | Runs + |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,40 | 0,15 | 0,510 | 0,546 | −0,984 | 8/9 |
| 0,40 | 0,20 | 0,518 | 0,481 | −0,762 | 8/9 |
| 0,40 | 0,25 | 0,522 | 0,447 | −0,665 | 8/9 |
| 0,50 | 0,15 | 0,408 | 0,601 | −1,164 | 8/9 |
| **0,50** | **0,20** | **0,685** | **0,139** | **0,390** | **9/9** |
| 0,50 | 0,25 | 0,632 | 0,139 | 0,377 | 9/9 |
| 0,60 | 0,15 | 0,634 | 0,273 | 0,229 | 9/9 |
| 0,60 | 0,20 | 0,658 | 0,268 | 0,190 | 9/9 |
| 0,60 | 0,25 | 0,663 | 0,292 | 0,305 | 9/9 |

**Décision.** Initial 0,50 et validation interne 0,20. Meilleur Sharpe, faible
dispersion et historique test plus large que 0,60. Les scores entre fractions
initiales ne couvrent pas exactement les mêmes périodes ; 0,40 inclut notamment
une période plus difficile dès 2008.

**Sources.** `artifacts/comparisons/ohlc-clean/20-cv-history-*`; point 0,50/0,20
dans `17-cv-folds-3`.

## 21 - Architecture comparison

**Statut.** Sauté. La comparaison finale à cinq seeds et cinq architectures n'a
pas été relancée après sélection de la loss Sharpe et du profil de features.

**Conséquence.** Les décisions finales 11–22 sont démontrées pour GRU, pas pour
l'ensemble des architectures.

## 22 - Market and sector context

**But.** Revalider l'apport des familles techniques, marché et secteur après
passage à la loss Sharpe et au contrôle du surapprentissage.

| Features | Rendement | Sharpe | σ Sharpe | Pire Sharpe | Runs + |
| --- | ---: | ---: | ---: | ---: | ---: |
| techniques | +3,72 % | 0,514 | 0,437 | −0,262 | 7/9 |
| techniques + marché | +8,71 % | 0,624 | 0,255 | 0,006 | 8/9 |
| techniques + secteur | +2,56 % | 0,407 | 0,474 | −0,370 | 6/9 |
| **techniques + marché + secteur** | **+9,13 %** | **0,685** | **0,139** | **0,390** | **9/9** |

**Décision.** Combinaison complète. Le marché apporte le gain principal. Le
secteur seul dégrade les résultats, mais améliore le couple techniques + marché,
probablement via interaction et remplacement de features sous le plafond de 32.
Ce benchmark teste du contexte, pas un modèle de régime HMM/VIX explicite.

**Sources.** `artifacts/comparisons/ohlc-clean/22-regime-*`; combinaison complète
dans `17-cv-folds-3`.

## 23 - Final holdout

**Statut.** Non lancé.
