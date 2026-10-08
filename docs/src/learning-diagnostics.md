# Diagnostic de l'apprentissage

Le diagnostic sert à distinguer un apprentissage instable d'un changement de direction, de taille ou de coût des positions. Il prépare l'analyse des benchmarks GRU/GNN et news sans changer leur protocole, ni ouvrir le holdout final.

Deux outils sont séparés : analyse en lecture seule des exports existants et journalisation optionnelle des prochains entraînements. Aucun entraînement supplémentaire n'est nécessaire pour analyser les prédictions déjà sauvegardées.

## Analyse des runs existants

`scripts/diagnose_learning.py` accepte un run multimodal ou un dossier contenant plusieurs runs. Les exports doivent suivre le contrat signé `schema_version=2` des comparaisons graphe/news : `metadata.json`, `folds.json`, résultats immuables, prédictions et trajectoires journalières INNER/OUTER, avec manifeste de complétion.

Avant l'analyse, le diagnostic vérifie la cohérence des signatures enregistrées et les empreintes des artefacts, l'identité candidat/fold/seed, l'alignement date/ticker, l'ordre Sell/Hold/Buy et la frontière du holdout. Les checkpoints sont vérifiés comme fichiers signés, mais jamais désérialisés. La signature complète de la spécification d'entraînement n'est pas redérivée : il s'agit d'un contrôle d'intégrité des exports, pas d'une nouvelle certification du modèle ou du preprocessing. Les rendements, coûts et expositions sauvegardés sont contrôlés contre une réexécution des positions avec le délai et la loss financière du run d'origine.

Les sources ne sont pas modifiées. Le rapport doit être créé dans un nouveau dossier extérieur aux sources. Un dossier de sortie déjà présent est refusé, sans écrasement.

Pour commencer sur le benchmark 32/64 features et gates :

```bash
.venv/bin/python scripts/diagnose_learning.py \
  --run-dir artifacts/comparisons/us-relational-market/04-feature-gate-interaction \
  --partitions inner \
  --output-dir artifacts/diagnostics/us-feature-gate-learning-v1
```

Un contrôle court peut se limiter à deux candidats, un fold et une seed :

```bash
.venv/bin/python scripts/diagnose_learning.py \
  --run-dir artifacts/comparisons/us-relational-market/04-feature-gate-interaction \
  --partitions inner \
  --candidates gru,gru_market \
  --folds 0 --seeds 1 \
  --output-dir artifacts/diagnostics/us-feature-gate-learning-smoke-v1
```

INNER est le défaut. `--partitions inner,outer` ajoute une lecture descriptive d'OUTER, sans sélection automatique de feature, seuil ou checkpoint. Aucune option ne permet d'analyser le test final. Les candidats se comparent à la référence `gru` du même run, fold, seed et partition ; les variantes de caps différents ne sont pas fusionnées implicitement.

Pour les news, il faut d'abord rapatrier les artefacts complets du PC, avec leurs manifestes, résultats immuables et fichiers Parquet. La même commande utilise alors le chemin du run rapatrié. Les résultats FNSPID décrits dans les documents ne suffisent pas à reconstruire les exports absents du Mac.

## Ce que contient le rapport

| Fichier | Lecture attendue |
| --- | --- |
| `report.md`, `report.json` | État de complétion, sources vérifiées, tâches retenues ou rejetées, limites et résultats détaillés. Aucun gagnant automatique. |
| `tasks.csv` | Par candidat/fold/seed/partition : époque retenue, époques exécutées, signaux, probabilités, classification et métriques financières. |
| `summary.csv` | Nombre, moyenne, écart-type, minimum et maximum des métriques et époques par run/candidat/partition. Statistiques descriptives, sans test d'indépendance. |
| `tickers.csv` | Disponibilité, positions et contribution de chaque ticker au PnL du portefeuille. |
| `monthly.csv` | Rendements mensuels modèle/B&H, exposition exécutée brute/nette, turnover et coûts. |
| `epochs.csv` | Pertes, distributions des positions et gradients par époque, uniquement lorsqu'une trace a été enregistrée. Données prêtes à tracer, sans graphique généré à cette étape. |
| `paired_signals.csv` | Deltas financiers avec `gru` sur les mêmes clés/prix/labels, et écarts de positions sur les signaux disponibles pour les deux candidats. |

Les tâches manquantes, incomplètes ou incompatibles sont signalées. Un rapport partiel ne devient pas un résultat complet : la commande retourne un code non nul si des problèmes subsistent. Si aucune tâche sélectionnée n'est valide, aucun rapport de résultats n'est publié.

### Lire positions et disponibilité

- Un label inconnu reste `-1`, distinct de Hold. Les métriques de classification utilisent uniquement les labels connus et les signaux disponibles.
- Un signal indisponible reste masqué. Le FLAT de repli explicite du runner peut entrer dans le portefeuille, mais ne devient pas une prédiction observée dans les distributions.
- La position continue reste `P(Buy)-P(Sell)` en long/short, ou `P(Buy)` en long-only. L'argmax décrit une autre lecture des probabilités, sans modifier le décodage du backtest.
- `--flat-tolerance` et `--saturation-threshold` sont des seuils descriptifs. Ils ne changent aucune position, transaction ou performance.
- La position émise à J n'est pas l'exposition exécutée à J. Le calcul financier conserve `execution_delay` et les coûts d'origine.
- L'exposition brute mesure la taille des positions ; l'exposition nette mesure leur direction agrégée. Ni l'une ni l'autre ne neutralise le beta, le risque ou la concentration sectorielle.

Les contributions ticker utilisent le capital du portefeuille avant chaque rendement et son allocation d'origine. Leur somme réconcilie le PnL du portefeuille. Ce ne sont pas des backtests indépendants où chaque action recevrait tout le capital initial.

### Lire probabilités et courbes

Le maximum des probabilités et la marge entre les deux premières classes mesurent la concentration des sorties. Ils ne constituent pas une confiance calibrée. Une sortie très concentrée peut être incorrecte ; aucune calibration n'est ajoutée par ce diagnostic.

Les anciens runs fournissent généralement `best_epoch` et `epochs_run`, mais pas les pertes et gradients de toutes les époques. `trace_available=False` et un fichier d'époques vide signalent cette absence. Ces courbes ne peuvent pas être reconstruites depuis le checkpoint retenu. Une époque 1 retenue ne signifie pas qu'une seule époque a été exécutée, ni qu'un bug est démontré.

## Journaliser les prochains entraînements

`--learning-diagnostics` active une trace optionnelle pour les comparaisons graphe/GRU (`run_gnn_graph_comparison.py`), news (`run_news_sentiment_comparison.py`) et leur orchestration US (`run_us_multimodal_study.py`). Sans cette option, la trace reste désactivée. Les anciens workflows 3D de `run_loss_comparison.py` ne sont pas instrumentés par cette préparation.

L'option s'ajoute à une commande d'entraînement existante, avec un nouveau `--output-dir`. Elle enregistre les observations des passages déjà nécessaires à l'optimisation ; elle n'ajoute pas de prédiction, de loss ou de recherche de paramètres.

Chaque époque sauvegarde :

- La loss financière TRAIN, puis la loss de validation interne.
- Les distributions des positions brutes : moyenne, amplitude moyenne, dispersion, minimum/maximum et proportions long/short/flat.
- La norme L2 globale des gradients avant clipping.
- Le checkpoint amélioré ou non, la patience consommée et l'époque retenue.
- Le motif d'arrêt : `early_stopping` ou `max_epochs`.

La trace est stockée dans `fit.learning_trace` du résultat immutable de la tâche, puis extraite par le diagnostic offline. Les décisions du runner restent celles d'origine : même loss, optimizer, clipping, RNG, early stopping et restauration du meilleur checkpoint. Les observations peuvent ajouter du temps de synchronisation, notamment pour lire les gradients GPU, mais ne sont pas utilisées par l'optimizer.

Attention aux conditions de mesure : TRAIN est observé avant l'update en mode entraînement, avec dropout si configuré ; la validation est observée après l'update en mode évaluation. Les positions résument ici tous les emplacements fournis au calcul financier, y compris les FLAT de repli indisponibles. La différence entre les deux losses n'est donc pas une mesure isolée du surapprentissage. Un probe TRAIN en mode évaluation, dans les mêmes conditions que la validation, reste une extension distincte.

L'activation de la trace entre dans la provenance du run. Le changement de code modifie également les empreintes runtime. Il ne faut pas reprendre un ancien benchmark terminé avec `--resume` pour lui ajouter des courbes : utiliser un nouveau run et conserver les anciens artefacts. La reprise reste réservée à une commande et une provenance strictement compatibles.

## Ordre de diagnostic

1. Vérifier les sources, la complétion et les disponibilités avant d'interpréter les scores.
2. Comparer les checkpoints, distributions et métriques INNER entre seeds, puis identifier les positions saturées ou presque constantes.
3. Localiser les contributions par ticker et période, turnover et coûts, sans créer de blacklist rétrospective.
4. Comparer les changements de signal à ceux d'exposition. Les contrôles d'exposition du benchmark restent nécessaires pour isoler un simple effet de sizing.
5. Utiliser OUTER seulement pour décrire la généralisation du protocole déjà fixé. Toute modification du modèle ou de l'early stopping se règle sur validation interne.

Les seeds partagent les mêmes dates de marché ; folds et seeds ne constituent pas des observations indépendantes. Un gain concentré sur quelques runs exige une analyse appariée, pas seulement une moyenne positive.

## Ce qui reste hors de cette préparation

- Diagnostics causaux : probes TRAIN/validation comparables, analyses de gradients par branche, trajectoires détaillées de fusion ou gates.
- Nouveaux benchmarks d'early stopping, régularisation, losses ou calibration.
- Classification causale des régimes et attribution systématique des features.
- Permutations temporelles par blocs et retraits de features avec réentraînement apparié.
- Intégration et audit PIT des nouvelles données LSE.

L'étude complète d'importance des features reste prévue après l'intégration LSE et ses contrôles de disponibilité. Cette préparation permet de détecter plus tôt un apprentissage dégénéré, mais n'attribue pas un pourcentage de Sharpe à chaque feature.

Voir [le périmètre et la suite](../plan/next_steps.md), [le benchmark features/gate](../benchmarks/us-feature-gate-interaction.md) et [les losses financières](financial-loss.md).

## Validation locale du 8 octobre 2026

Les 162 tests ciblés du diagnostic, de la trace et des workflows multimodaux passent. La parité numérique est testée sur CPU : même poids, mêmes prédictions et même checkpoint avec ou sans trace, y compris avec dropout, clipping et early stopping.

Suite complète exécutée : 1 427 tests passent, 11 sont ignorés et 25 échouent dans les workflows FNSPID décrits ci-dessous. La suite globale n'est donc pas annoncée comme entièrement verte.

Le contrôle réel de la grille 32/64 couvre `gru` et `gru_market`, fold 0, seed 1, sur INNER et OUTER : huit lignes tâche/partition validées, aucune trace historique inventée. Le rapport séparé se trouve dans `artifacts/diagnostics/learning-smoke-20261008-final`.

La suite FNSPID présente 25 échecs préexistants, reproduits en chargeant en mémoire les versions Git d'avant cette implémentation des modules modifiés. Les contrôles concernés portent notamment sur les fenêtres temporelles et l'éligibilité des news. Ils restent à corriger séparément avant un nouvel entraînement FNSPID local ; ce diagnostic ne modifie pas l'agrégation des news pour les masquer.
