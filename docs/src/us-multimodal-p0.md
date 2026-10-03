# P0 : provenance, replay et orchestration

P0 prépare les benchmarks suivants sans changer l'architecture des branches ni lancer de benchmark complet. La fusion et la calibration de P3 ne sont pas encore implémentées par cette étape. Aucune donnée n'est téléchargée.

## Nouveaux artefacts

Les nouveaux runs de `scripts/run_gnn_graph_comparison.py` utilisent le schéma 2. Ils enregistrent :

- la spécification CV complète et les frontières de chaque fold ;
- le calendrier global et l'ordre des tickers ;
- les empreintes des données, du contexte, des sources Python et du runtime numérique ;
- les features réellement retenues, fills, scalers, états des sélecteurs et FracDiff éventuel ;
- les poids du modèle et sa construction effective ;
- les logits Sell/Hold/Buy, probabilités et positions sur validation interne et externe ;
- les chemins journaliers nets/bruts, coûts, turnover et expositions ;
- un résultat immuable et un manifeste de fichiers avec tailles/empreintes.

Les prédictions portent une clé `date|ticker`, la partition, le fold et la seed. Les labels inconnus sont exportés avec `label=-1` et `label_known=false`, jamais comme Hold. Les fichiers Parquet sont écrits par lots.

Le contrat d'export précise les unités et la convention temporelle : signal à la clôture J, exécution à la clôture J+délai, puis rendement close-to-close suivant. Le `source_end` du graphe est une borne synthétique de fin de séance UTC, pas un horodatage de publication fournisseur. Les rendements/coûts journaliers sont des fractions, pas des pourcentages ni des PnL monétaires.

`buy_hold_return` dans les chemins journaliers utilise la même exécution différée et les mêmes coûts que `ReturnPanel`. `buy_hold_gross_return` fournit aussi la référence brute sans ces coûts/délais. Ces conventions sont distinctes et doivent rester visibles dans les comparaisons.

## Resume et reuse

`--resume` reprend exactement le même protocole. Les fichiers terminés sont vérifiés, pas seulement leur statut. Un artefact manquant/corrompu invalide la tâche correspondante, qui est recommencée ; les autres tâches restent conservées. La reprise n'est pas une reprise à l'époque près.

`--reference-run` propose des contrôles à réutiliser. Une signature effective inclut données, dates, seed, prétraitements, modèle, source et environnement de calcul. Changer le chemin de sortie ou les autres candidats ne change pas cette signature. Changer la profondeur GNN ne force pas à réentraîner un contrôle GRU identique.

Les contrôles importés restent dans leurs dossiers sources. Les références sont relatives et les résultats sont lus depuis les fichiers immuables vérifiés. Déplacer une étude nécessite donc de déplacer aussi ses sources en conservant leurs chemins relatifs. Sous Windows, placer l'étude et ses runs sources sur le même volume pour permettre ces références relatives.

La provenance de code est conservatrice : toute modification des sources Python du package peut rendre un contrôle non compatible. Une machine/runtime différent n'est pas supposé numériquement identique.

### Limite des anciens runs

Les anciens artefacts n'enregistrent pas toute la provenance. Ils ne sont donc **pas automatiquement réutilisables** dans les nouveaux entraînements. Un replay peut vérifier leurs métriques et fournir les logits pour une analyse exploratoire, mais ne reconstruit pas une preuve de provenance absente.

Les anciens runs nécessitent des fractions CV explicitement confirmées pour le replay. Ne pas choisir des valeurs par défaut simplement pour faire passer le contrôle. Si la compatibilité ne peut être prouvée, le contrôle doit être réentraîné. Le `--dry-run` affiche ce surcoût.

Un ancien run ne peut pas reprendre sous le nouveau schéma en ajoutant seulement `--resume`. Conserver sa version de code pour le terminer, ou lancer un nouveau dossier après migration ; ne pas modifier ses métadonnées à la main.

## Préparer une étude sans entraîner

Le fichier `configs/benchmark/us_multimodal_optimization.json` reprend le protocole US figé. Le choix du graphe est explicite : l'orchestrateur ne choisit pas un gagnant dans un run encore incomplet.

Pour P1, je prévois maintenant six candidats à chacun des caps 32 et 64 : GRU, GNN résiduel et GNN identité sans relations, chacun avec et sans gate marché. Les contrôles `identity` et `identity_market` sont inclus aux deux caps, soit 12 variantes et 108 entraînements maximum avant réutilisation vérifiée. Je compare leurs effets par fold/seed, sans considérer leur présence comme une preuve d'alpha relationnel. Les résultats restent en attente ; le [guide d'interaction](../plan/us-feature-gate-interaction.md) conserve la commande actuelle sans référence historique incompatible. Je n'ai lancé aucun entraînement pour ajouter ce contrôle.

Exemple avec `rolling_residual_topk`, donné uniquement pour illustrer la commande, pas comme sélection automatique du meilleur graphe :

```bash
.venv/bin/python scripts/run_us_multimodal_study.py \
  --stage features \
  --graph-choice rolling_residual_topk \
  --reference-run artifacts/comparisons/us-relational-market/01-combined-025 \
  --output-dir artifacts/comparisons/us-relational-market/optimization \
  --dry-run
```

Le dry-run prépare les prétraitements train-only nécessaires aux signatures, mais n'entraîne aucun modèle et ne crée pas de dossier de résultats. Il rapporte tâches nouvelles, réutilisées, partagées et incompatibilités. Il peut donc prendre du temps sur le grand univers.

En PowerShell, utiliser le même script avec `.venv\Scripts\python.exe` et une seule ligne :

```powershell
.venv\Scripts\python.exe scripts/run_us_multimodal_study.py --stage features --graph-choice rolling_residual_topk --reference-run artifacts/comparisons/us-relational-market/01-combined-025 --output-dir artifacts/comparisons/us-relational-market/optimization --dry-run
```

Les chemins des prix, du contexte et de la sélection peuvent être remplacés avec `--data`, `--market-context-data`, `--ticker-selection`. `--device cpu|cuda|mps|auto` remplace le choix du fichier d'étude. Aucune dépendance au code ignoré de `MT5/`.

## Exécuter et reprendre

Après validation du graphe et du budget, enlever `--dry-run` pour exécuter l'étape. Ajouter `--resume` pour reprendre la même étude.

Une barre `tqdm` unique affiche progression/ETA et pics RAM/VRAM disponibles. La RAM utilise `resource` sur Unix et l'API Windows ; la VRAM est l'allocation maximale PyTorch lorsque CUDA est déjà initialisé. Une mesure indisponible est indiquée, sans interrompre l'entraînement. Ctrl-C marque l'étape interrompue ; `--resume` conserve les tâches terminées et recommence la tâche incomplète.

Les étapes suivantes utilisent le même dossier d'étude :

- `--stage gnn-depth --graph-choice <graphe> --feature-choice 32|64` ;
- `--stage market-depth --graph-choice <graphe> --feature-choice 32|64`.

Elles ne déterminent jamais seules le cap ou le graphe. Les contrôles compatibles des étapes précédentes sont proposés à la réutilisation. `--stage fusion` explique le blocage : l'implémentation P3 reste nécessaire.

Pour ajouter une étape à un dossier d'étude déjà créé, fournir également `--resume`. Cette option autorise l'ajout de l'étape et vérifie strictement le protocole lorsqu'elle existe déjà.

## Replay et comparaison

`scripts/diagnose_gnn_information.py` accepte maintenant les variantes marché, top-k et résiduelles présentes dans le run. Le replay produit des prédictions internes/externes et des chemins journaliers dans un **nouveau dossier**, en vérifiant les métriques du checkpoint et en gardant le holdout fermé. Pour un ancien run, fournir les fractions CV confirmées avec `--cv-initial-train-fraction` et `--cv-inner-val-fraction`. `--replay-only` exporte sans lancer les tests d'information ; `--market-context-data` permet de remplacer le chemin local du même fichier de contexte figé.

Pour comparer les étapes déjà exécutées :

```bash
.venv/bin/python scripts/compare_us_multimodal_study.py \
  --study-dir artifacts/comparisons/us-relational-market/optimization \
  --output-dir artifacts/comparisons/us-relational-market/optimization/report-p0
```

Le rapport contient JSON, Markdown, résumé CSV et différences appariées par `(fold, seed)`. Les résultats incomplets restent marqués incomplets. Aucun gagnant n'est promu automatiquement et aucun accès au holdout final n'est effectué.

## Vérifications P0

Lors de la validation historique de P0, la suite complète locale comptait 815 tests réussis et 12 ignorés. Les tests P0 couvraient replay inner/outer de huit contrôles, import sans entraînement, signatures sensibles aux données/dates/CV, récupération d'artefacts corrompus, métriques immuables, Ctrl-C/reprise et chemins d'études déplacées. Une petite étude CPU synthétique vérifiait aussi le CLI complet, ses 20 tâches historiques, ses rapports et une reprise sans nouvel entraînement. Ces comptes ne décrivent pas la grille P1 actuelle.

Les APIs mémoire Windows/CUDA sont testées avec des simulations ; aucun entraînement réel Windows/GPU ou benchmark US n'a été lancé à cette étape. Les branches et la loss existantes sont conservées. La soustraction de dates des métadonnées triple-barrier est normalisée en UTC pour accepter des séances déjà timezone-aware, sans modifier la génération des labels.
