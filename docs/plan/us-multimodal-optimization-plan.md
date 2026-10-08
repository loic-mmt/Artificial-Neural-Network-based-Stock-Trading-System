# Plan d'implémentation : optimisation multimodale US

Statut au 8 octobre 2026 : P0 implémenté et testé ; benchmark P1 terminé, **108/108 entraînements vérifiés**, avec `identity_market` aux deux caps. Les audits de features et comparaisons d'exposition sont complets. La configuration de travail reste cap 32 et gate désactivé ; identité32 + marché reste un challenger fragile, sans promotion automatique. Les diagnostics détaillés des poids du gate et les étapes suivantes sont séparés et restent à faire. Guides : [provenance, replay et orchestration](../src/us-multimodal-p0.md), [benchmark d'interaction](us-feature-gate-interaction.md), [résultats P1](../benchmarks/us-feature-gate-interaction.md).

## 1. Objectif et périmètre

Déterminer, dans cet ordre, si nous gagnons à :

1. passer de 32 à 64 features boursières, notamment avec un gate guidé par le marché ;
2. augmenter la profondeur du GNN ou du Transformer de marché ;
3. fusionner les prédictions indépendantes du GRU et du GNN ;
4. pondérer cette fusion avec une confiance calibrée.

Il faut séparer ces effets, conserver des contrôles comparables et réutiliser les entraînements valides. Pas de produit cartésien géant.

Invariants :

- Le GRU traite les séquences propres aux actions.
- Le GNN traite ses propres features de nœuds et un graphe causal. Il ne reçoit pas les embeddings du GRU.
- Le Transformer décrit l'état global du marché. Dans cette étude, il guide les features mais ne vote pas directement sur les positions.
- Les ETF de contexte ne sont ni des actifs tradés ni, actuellement, des nœuds du GNN.
- Sentiment, FinBERT, nouvelles sources macro et allocation restent hors périmètre. Aucun téléchargement de news.
- Les méthodes de décodage, de sizing et de réentraînement ne changent pas pendant ces comparaisons.

Références locales : [MASTER](../papers/MASTER/MASTER.md) pour le gate conditionné par le marché, [FusionLSTM-CNF](../papers/FusionLSTM-CNF/FusionLSTM-CNF.md) pour la séparation des modalités et la fusion guidée par la confiance, [contrat multimodal](next_steps.md) et [benchmark US actuel](../benchmarks/us-relational-market-benchmark.md). Nous testons des adaptations de ces idées, pas une reproduction complète des architectures ou des performances des articles.

## 2. Ce qui existe déjà

| Élément | État actuel | Travail nécessaire |
| --- | --- | --- |
| GRU et GNN indépendants | Opérationnels dans le runner d'ablation | Conserver cette séparation |
| Transformer de marché | Séquence ETF/VIX et agrégats transversaux | Tester sa profondeur, sans ajouter de vote |
| Gate de features | Pondération `F × softmax`, initialisation identité ; comparaison 32/64 terminée | Diagnostics temporels des poids, séparés du benchmark terminé |
| Sélection des features | Train uniquement, filtrage couverture/corrélation et classement par variance ; prétraitements sauvegardés et audits 32/64 validés | Attribution financière des features, pas encore réalisée |
| Profondeurs | `--gnn-layers` et `--market-transformer-layers` existent | Orchestration et contrôles appariés |
| Checkpoints | Schéma 2, prétraitement complet, signatures et exports internes/externes | Replay vérifié disponible ; anciens runs restent exploratoires si provenance absente |
| Reprise | `--resume` strict et réutilisation inter-études par signature/fichiers vérifiés | Conserver le code d'origine pour terminer les anciens runs de schéma 1 |
| Fusion générique | `single`, `mean`, `static`, `confidence` existent côté modèle | Raccordement aux checkpoints et évaluation CV |
| Diagnostic de fusion | Mélange de positions, anciens candidats codés en dur | Nouvelle comparaison de logits, sans confondre les deux méthodes |

### Clarification : 32/64 signifie quoi ?

`--overfitting-max-features 32` limite les **features en entrée**. `hidden_size=32` désigne les **neurones cachés du GRU**. Ces deux paramètres sont indépendants.

Le gate actuel ne choisit pas exactement 32 features parmi 64. Il pondère les 64 features conservées, sans réduire la dimension du tenseur ni garantir un gain de calcul. Le même vecteur de poids est partagé entre actions et entre pas temporels d'une fenêtre, pour une date de prédiction donnée.

Une sélection dure, dynamique, de 32 features parmi 64 serait une expérience différente. Elle n'est pas nécessaire pour le premier benchmark et reste une extension conditionnelle.

## 3. Protocole commun à figer

| Paramètre | Référence |
| --- | --- |
| Actions | `configs/benchmark/stocks_us_gnn_complete_2005.json`, 143 tickers |
| Prix | `data/processed/mt5_stocks_us_daily_clean.parquet` |
| Contexte | `data/processed/us_market_context_daily.parquet`, mêmes colonnes que le script actuel |
| Features | `expanded`, groupes `technical,market,sector`, sources externes désactivées |
| Labels | Triple barrier, ATR 20, horizon maximal 10, barrières 0,75/0,75, CUSUM 0,5, entre événements Hold, coût 5 bps |
| GRU | Configuration de `configs/benchmark/gru_market_context.json`, contexte 60 |
| Objectif | `combined`, poids 0,25 selon la convention existante, coût 5 bps |
| Position | `long_short`, décodage continu existant `P(Buy)-P(Sell)`, délai d'exécution 1 |
| Graphe | Lookback 252, top-k 5, poids absolus, recalcul toutes les 20 séances |
| Capacité initiale | GNN largeur 32/profondeur 1 ; Transformer largeur 32/4 têtes/profondeur 1 |
| Gate | Température 1,0 |
| Évaluation | 3 folds purgés, gap 5, seeds 1/7/19, Sharpe régularisé primaire |
| Budget mémoire | Lots de 16 dates, sauf ajustement déclaré et validé |
| Test final | Holdout fermé pendant toutes les étapes exploratoires |

Les dates exactes doivent provenir du run de référence, pas du nom `complete_2005`. Le calendrier effectivement utilisé peut commencer avant 2005 si le fichier contient cet historique. Enregistrer les bornes et les séances retenues, puis les réutiliser exactement.

Ne pas retélécharger les données entre étapes. Les prix ajustés historiques peuvent être révisés. Copier les mêmes fichiers sur le PC, contrôler leurs empreintes et figer l'univers. Les classifications sectorielles actuelles et l'univers survivant rendent les résultats provisoires, même avec des graphes calculés causalement.

Ce protocole compare des modèles sur folds temporels. Il ne constitue pas encore un benchmark de réentraînement quotidien ni un test live MT5.

## 4. P0 : provenance, replay et orchestration

Cette étape doit précéder les nouveaux benchmarks. Elle permet d'utiliser les résultats actuels sans les mélanger à des runs incompatibles.

P0 terminé : métadonnées et artefacts atomiques, replay des contrôles actuels, réutilisation vérifiée, dry-run et orchestrateur portable. Les anciens checkpoints peuvent être rejoués sans modifier leurs dossiers, mais une provenance manquante ne peut pas être certifiée rétroactivement. La fusion reste explicitement bloquée jusqu'à P3. Les entraînements de validation de P0 utilisent uniquement des données synthétiques et ne remplacent aucun benchmark US.

### 4.1 Compléter les métadonnées

Étendre `src/trading_system/experiments/graph_ablation.py` et factoriser les éléments réutilisables dans un module d'artefacts dédié.

Sauvegarder :

- version du schéma, version du code et empreinte des sources pertinentes, y compris si le worktree est modifié ;
- empreintes des prix, du contexte, de l'univers et ordre explicite des tickers ;
- spécification CV complète : fractions initiales/interne/finale, gap, embargo, frontières et dates effectives de chaque fold ;
- configuration effective du modèle, du graphe, du gate, de la loss et de l'exécution ;
- colonnes retenues dans leur ordre, compte réel, états des sélecteurs, valeurs de remplissage et scalers ;
- état FracDiff si activé, même si ce benchmark ne l'utilise pas ;
- versions des dépendances, matériel, précision numérique et paramètres influençant le calcul ;
- empreintes des checkpoints et des fichiers de prédictions.

Les anciens runs n'enregistrent pas explicitement toute la spécification des folds, notamment `initial_train_fraction`. Le schéma 2 sauvegarde ces paramètres et une modification invalide la reprise.

Séparer deux opérations :

1. **Resume** : reprendre exactement le même run, sans changer ses paramètres.
2. **Reuse** : importer un entraînement terminé dont la signature effective correspond au contrôle demandé.

La signature de réutilisation inclut données, partitions, prétraitements, modèle, seed et paramètres de calcul. Le chemin de sortie, le nom de l'étude et les autres candidats ne doivent pas invalider un entraînement identique. Un changement de matériel doit rester visible ; aucune identité numérique CPU/GPU n'est supposée.

Écrire les artefacts de façon atomique. Marquer une tâche terminée seulement après validation de tous ses fichiers. Un fold interrompu peut être relancé ; une reprise à l'époque près reste hors périmètre.

### 4.2 Exporter les prédictions nécessaires à la fusion

Ajouter un export batché des sorties sur validation interne et fold externe, sans conserver tous les tenseurs en mémoire.

Schéma minimal d'un fichier Parquet :

```text
variant_id, fold, seed, partition, date, ticker, backtest_key
available, label_known, label
logit_sell, logit_hold, logit_buy
p_sell, p_hold, p_buy, position
```

Le manifeste fixe l'ordre des classes, les unités, les dates de disponibilité et la provenance du modèle. Ajouter un fichier de chemins journaliers : rendement brut/net, coût, exposition et turnover.

La clé de backtest doit être stable, même si l'ordre d'entrée change. Les lignes absentes ou indisponibles ne deviennent jamais de vrais signaux zéro ou Hold. Les labels inconnus restent exclus des métriques de classification et de la calibration.

### 4.3 Rejouer les checkpoints existants

Étendre le replay de `src/trading_system/pipelines/diagnose_gnn_information.py` pour les variantes top-k, résiduelles et marché. Il doit reconstruire le **contrôle réellement entraîné**, ses scalers, ses colonnes et ses sources de contexte.

Pour les anciens checkpoints, reconstruire uniquement ce qui peut être vérifié à partir du code et des données figés. Exporter les prédictions dans un nouveau dossier de diagnostic, sans réécrire le run source. Comparer les métriques rejouées aux métriques sauvegardées avec une tolérance numérique déclarée.

Si les dates, les paramètres ou le prétraitement ne sont pas vérifiables, déclarer le contrôle non réutilisable et le réentraîner. Ne jamais compléter silencieusement des métadonnées manquantes par des valeurs supposées.

### 4.4 Orchestration portable

Créer les fichiers proposés suivants :

- `configs/benchmark/us_multimodal_optimization.json` : protocole figé et grille par étape ;
- `src/trading_system/experiments/multimodal_study.py` : expansion des tâches, signatures, dépendances et import des contrôles ;
- `scripts/run_us_multimodal_study.py` : lancement portable Mac/Windows ;
- `scripts/compare_us_multimodal_study.py` : rapport commun et comparaisons appariées.

Interface implémentée dans `scripts/run_us_multimodal_study.py` :

```text
--study-config <json>
--stage features|gnn-depth|market-depth|fusion
--reference-run <directory>
--output-dir <directory>
--resume
--dry-run
--graph-choice sector|rolling_topk|rolling_residual_topk
--feature-choice 32|64
```

`--dry-run` affiche entraînements nouveaux/réutilisés, calibrations, évaluations et incompatibilités. Réutiliser le runner existant pour l'entraînement, après extraction des fonctions nécessaires, plutôt que dupliquer sa préparation et sa loss. L'orchestrateur doit pouvoir évaluer une variante seule avec un contrôle importé ; le runner actuel exige un candidat `gru` dans chaque run.

Utiliser `sys.executable` pour les sous-processus. Prévoir une commande PowerShell utilisant `.venv\Scripts\python.exe`, sans dépendance au dossier ignoré `MT5/`. Une barre `tqdm` unique rapporte progression, ETA et mémoire ; une erreur laisse une tâche reprenable et un diagnostic explicite.

L'implémentation appelle directement le runner commun ; les commandes de lancement proposées utilisent `sys.executable`. `features`, `gnn-depth` et `market-depth` sont disponibles ; `fusion` signale le blocage P3. Le graphe doit être choisi explicitement. Voir le [guide P0](../src/us-multimodal-p0.md) avant le premier lancement, notamment pour le coût des contrôles anciens non réutilisables.

**Critère de fin P0 :** un checkpoint rejoué retrouve ses métriques, un contrôle compatible est importé sans entraînement, et un changement de dates ou de données bloque cet import.

## 5. P1 : 32 contre 64 features, avec et sans marché

### 5.1 Choisir le graphe de référence

Je conserve le benchmark historique de référence : 8 candidats × 3 folds × 3 seeds = **72 entraînements**. Ce compte décrit le run précédent, pas la nouvelle grille P1.

Je compare les graphes au contrôle `identity`. Pour la grille P1 actuelle, je conserve `rolling_residual_topk` comme challenger exploratoire `G*`, pas comme vainqueur confirmé. J'utilise **le même graphe** pour `G*` et `G*_market` : choisir deux graphes différents ne permettrait pas d'isoler l'effet du gate. La règle de décision reste celle de la section 9.

Les références historiques à 32 avec et sans marché existent pour `rolling_topk` et `rolling_residual_topk`, mais leur pool effectif était `technical,sector`. Je réentraîne donc les contrôles 32 et les variantes 64 sur le pool commun `technical,market,sector`, sauf import de schéma 2 strictement compatible. Si une étude distincte retient `sector`, je dois ajouter explicitement `sector_market` : il n'est pas présent dans la référence historique.

Si les graphes sont faibles seuls mais complémentaires au GRU, garder un candidat pour le diagnostic de fusion. Ne pas affirmer qu'il améliore déjà le modèle.

### 5.2 Matrice principale

| Branche | 32 features | 64 features |
| --- | --- | --- |
| GRU sans gate | Réentraînement commun | Réentraînement commun |
| GRU avec gate de marché | Réentraînement commun | Réentraînement commun |
| GNN `G*` sans gate | Réentraînement commun | Réentraînement commun |
| GNN `G*` avec gate de marché | Réentraînement commun | Réentraînement commun |
| GNN identité sans gate (`identity`) | Contrôle sans relations | Contrôle sans relations |
| GNN identité avec gate de marché (`identity_market`) | Contrôle sans relations | Contrôle sans relations |

Conserver `hidden_size=32`, la profondeur 1 et toutes les autres options. Le changement testé porte uniquement sur la limite de features.

Le cap reste un maximum : après filtrage, 64 demandé peut produire moins de 64 features. Rapporter le compte réel par fold. Vérifier que les 32 features du contrôle sont incluses dans les 64 et dans le même ordre relatif. Si ce n'est pas le cas, documenter le changement ou définir prospectivement un classement train figé dont on prend les préfixes 32/64 ; cela peut imposer de refaire le contrôle 32.

J'inclus `identity` et `identity_market` dès P1, aux deux caps. Je compare ces deux contrôles au même cap, fold et seed pour mesurer l'effet du gate et de sa capacité supplémentaire sans relations entre actions. Je compare aussi `G*` à `identity` et `G*_market` à `identity_market` dans les mêmes conditions. Ces contrôles appariés aident à distinguer capacité, gate et relations ; ils ne prouvent pas à eux seuls un alpha relationnel.

### 5.3 Diagnostics et décision

Exporter par date les poids du gate, leur entropie, leur stabilité et le nombre effectif de features pondérées. Vérifier que le gate n'est ni uniformément inactif ni concentré sur une seule colonne.

Mesurer, pour GRU et GNN séparément :

```text
gain_64_sans_gate = score_64_sans_gate - score_32_sans_gate
gain_64_avec_gate = score_64_avec_gate - score_32_avec_gate
interaction = gain_64_avec_gate - gain_64_sans_gate
```

Je mesure aussi ces effets pour la branche identité, afin de comparer son interaction à celle de `G*`. Une interaction positive suggère que le contexte aide à exploiter les features supplémentaires, mais ne prouve ni que gate64 bat plain64, ni qu'une sélection dure serait utile, ni un alpha relationnel.

Retenir 64 uniquement si le gain est stable et justifie le coût. Pour la première étude, conserver un cap commun aux branches afin de garder les comparaisons simples. Des caps propres au GRU/GNN restent configurables pour une étude ultérieure.

Si le gate gagne, prévoir une ablation secondaire gate statique contre encodeur de marché simple avant d'attribuer ce gain au Transformer. Ces contrôles existent côté `MarketGRUControl`, mais leur intégration dans cette étude et leurs entraînements ne sont pas encore réalisés. Ils évitent de confondre pondération des features, information de marché et capacité du Transformer.

**Budget P1 actuel :** 6 candidats × 2 caps × 3 folds × 3 seeds = **108 entraînements maximum**, avant réutilisation vérifiée de contrôles de schéma 2. Je remplace ainsi l'ancienne grille de dix variantes et 90 entraînements maximum. L'estimation initiale de **45 nouveaux entraînements** supposait les contrôles 32 compatibles et excluait `identity_market` ; elle reste historique, pas le budget du run actuel. La comparaison gate statique/encodeur simple reste une extension conditionnelle comptée séparément.

### 5.4 Résultat P1 et suites conditionnelles

Les 108 tâches et leurs exports sont complets. Les trois folds retiennent
exactement 32/64 features avec préfixe ordonné et prétraitements communs
compatibles. Les [résultats détaillés](../benchmarks/us-feature-gate-interaction.md)
remplacent les anciens statuts « en attente ».

- Conserver cap 32 pour le protocole commun, sans annoncer une inutilité
  individuelle des 32 features supplémentaires.
- Conserver le gate désactivé par défaut. Ses trois interactions moyennes avec
  le cap sont négatives en brut et dans les deux contrôles d'exposition.
- Garder identité comme contrôle et GNN résiduel32 comme challenger. Le meilleur
  score moyen appris appartient à identité32 + marché, mais son gain est instable.
- Ne pas lancer automatiquement la grille de profondeur du Transformer sur la
  base de ce résultat. Son intérêt doit rester une hypothèse distincte.
- Les poids temporels du gate et l'importance financière des features restent
  non documentés. Ce manque ne nécessite pas de recommencer les 108 entraînements.

Le classement actuel par variance avant scaling est sensible aux unités des
features ; il n'est pas une mesure de contribution au Sharpe. Après intégration
des données LSE PIT, prévoir une étude d'attribution par familles puis features,
sur validation chronologique : permutations par blocs, retraits avec réentraînement
des principaux candidats, stabilité par fold/seed et régimes définis causalement.
Les contributions ne s'additionnent pas à 100 % en présence de corrélations et
d'interactions. Les folds externes servent à confirmer les choix, pas à sélectionner
les features ; le holdout reste fermé.

## 6. P2 : profondeur du GNN et du Transformer

Fixer le cap retenu en P1, le graphe et les autres paramètres. Les arguments de profondeur existent déjà ; le travail principal est la grille, la réutilisation et les diagnostics.

### 6.1 Profondeur GNN

Comparer profondeur 1 contre 2, séparément :

- `G*` sans gate ;
- `G*_market` avec Transformer de profondeur 1.

Conserver largeur 32, dropout et graphe identiques. Pour une nouvelle profondeur P2, je prévois le contrôle `identity` de même profondeur et, si nécessaire, un nouveau `identity_market`. Les contrôles de profondeur 1 sont déjà inclus aux deux caps en P1 ; leur présence ne remplace pas un contrôle de profondeur 2. Rapporter nombre de paramètres et coût ; un gain partagé par le contrôle sans arêtes n'est pas une preuve d'amélioration relationnelle.

Observer la similarité des représentations entre actions, leur dispersion et les diagnostics du graphe pour détecter l'oversmoothing. Le top-k symétrisé n'impose pas exactement cinq voisins finaux par nœud : conserver les degrés réellement observés dans le rapport.

Ne tester profondeur 3 que si profondeur 2 gagne et reste abordable. Ne pas ajouter simultanément GAT, nouveaux résiduels ou nouveaux graphes : cela empêcherait d'attribuer le gain à la profondeur.

### 6.2 Profondeur Transformer de marché

Comparer profondeur 1 contre 2 pour :

- `gru_market` ;
- `G*_market`, GNN maintenu à profondeur 1 pour isoler ce premier effet.

Conserver largeur 32, quatre têtes, contexte 60 et température du gate 1,0. Le Transformer reste un encodeur de contexte, sans logits ajoutés à la fusion.

Si GNN 2 et Transformer 2 gagnent séparément, tester ensuite leur combinaison sur `G*_market`. Les effets ne sont pas nécessairement additifs.

La température n'est pas une dimension de cette première grille. Avant de la tester, raccorder explicitement sa configuration dans `MarketGRUControl` : actuellement, le chemin GRU utilise le défaut du gate alors que le chemin GNN reçoit le paramètre. Ajouter un test de propagation pour éviter des ablations asymétriques.

**Budget principal :** 18 entraînements pour GNN 2 + 9 pour son contrôle `identity` de profondeur 2 + 18 pour Transformer 2 = **45**. Ajouter **9** pour l'interaction GNN 2/Transformer 2 si les deux gagnent ; les contrôles gated et profondeur 3 sont des extensions budgétées séparément.

## 7. P3 : fusion tardive des branches gelées

### 7.1 Ne pas changer les modèles pendant la comparaison

Sélectionner une paire GRU/GNN à partir de P1/P2, puis figer ses checkpoints et prétraitements pour chaque fold/seed. Mesurer d'abord leur complémentarité : désaccord de positions, corrélation des rendements nets journaliers, erreurs de classification et périodes où chaque branche apporte quelque chose.

Les versions `gru_market` et `G*_market` possèdent chacune leur Transformer et leur gate entraînés indépendamment. Les fusionner via un unique Transformer partagé dans `MultimodalSystem` ne reproduirait pas les checkpoints gagnants.

Créer un adaptateur de fusion de branches gelées conservant chaque contrôle complet. Il pourra lire les logits exportés pour l'évaluation offline, puis les produire à partir des checkpoints pour l'inférence. La version avec Transformer partagé et fine-tuning conjoint sera une autre expérience, pas le contrôle initial.

### 7.2 Méthodes à comparer

| Variante | Calcul | Ajustement |
| --- | --- | --- |
| GRU seul / GNN seul | Contrôles des branches | Aucun |
| Moyenne brute | `0,5 × logits_GRU + 0,5 × logits_GNN` | Aucun |
| Poids fixe GRU 0,75 | `0,75 × logits_GRU + 0,25 × logits_GNN` | Aucun |
| Moyenne calibrée | Même moyenne après température propre à chaque branche | Températures sur validation interne |
| Poids statique appris | `alpha × logits_cal_GRU + (1-alpha) × logits_cal_GNN` | Un scalaire borné sur validation interne |
| Confidence-aware | Poids proportionnels à la confiance calibrée, renormalisés | Mêmes températures, pas de nouvelle recherche externe |

Ajouter les branches seules **calibrées** comme contrôles. Sinon, un gain dû à la calibration et au changement d'exposition pourrait être attribué à tort à la fusion.

Après fusion des logits, recalculer `softmax`, puis `P(Buy)-P(Sell)` et les coûts. Ne pas mélanger directement les positions en prétendant faire la même expérience : ces opérations ne sont pas équivalentes.

### 7.3 Calibration et confiance

Créer ou réutiliser un calibrateur de température par branche, ajusté sur la NLL des labels connus de validation interne, après fixation des checkpoints. Sauvegarder température, dates utilisées, pertes avant/après et métriques de calibration.

Ajuster `alpha` sur la même métrique financière primaire que l'étude, avec une recherche bornée prédéfinie et une préférence pour la moyenne simple en cas d'égalité. Les branches restent gelées. Ne pas ajouter de poids par ticker ou par régime dans ce premier test.

Le premier proxy de confiance est `max(softmax(logits_calibrés))`. Il ne mesure pas à lui seul l'incertitude épistémique. MC dropout, variance apprise et fusion CNF complète restent des extensions.

`MaskedLogitFusion` utilise actuellement les logits calibrés pour calculer les poids de confiance, mais mélange ensuite les logits **bruts**. Ajouter une politique explicite `raw`/`temperature_scaled` et la tester. Dans les variantes calibrées de cette étude, utiliser les logits calibrés pour les poids **et** pour les valeurs fusionnées ; conserver la moyenne brute comme contrôle. Ne pas changer silencieusement la sémantique des anciens appels.

Une loss financière ne garantit pas des probabilités de classes bien calibrées. Une température ajustée sur la NLL peut améliorer la calibration sans améliorer le PnL ; elle peut aussi modifier l'exposition. Les contrôles calibrés seuls sont donc indispensables, et la fusion par confiance reste une hypothèse à tester.

### 7.4 Partitions et masques

- Aucun paramètre de calibration/fusion n'est ajusté sur le fold externe ou le holdout final.
- Pour réutiliser les checkpoints actuels, la validation interne peut servir à l'early stopping puis à ces petits ajustements. Elle constitue un espace de réglage partagé, pas une validation indépendante. L'évaluation reste sur le fold externe.
- Une séparation chronologique supplémentaire entre early stopping, calibration et poids de fusion peut être ajoutée en confirmation. Elle doit être purgée et peut nécessiter de réentraîner les branches ; ne pas la présenter comme compatible automatiquement avec les anciens checkpoints.
- Vérifier l'identité des clés date/ticker, des seeds, des folds et du calendrier entre branches.
- Renormaliser les poids sur les seules branches disponibles. Une seule disponible doit reproduire cette branche ; aucune disponible signifie prédiction indisponible, traitée par une politique d'exécution explicite.
- Garder labels inconnus, actif absent et absence de branche distincts. Un faux zéro ne devient jamais de la confiance.

**Budget :** aucun nouvel entraînement des encodeurs si leurs exports sont compatibles. Chaque méthode est évaluée sur les neuf couples fold/seed. Les calibrations et ajustements de scalaires sont des petits fits distincts, pas neuf entraînements GRU/GNN supplémentaires.

## 8. Modules à créer ou compléter

Les noms nouveaux ci-dessous sont proposés ; réutiliser un module équivalent s'il existe déjà au moment de l'implémentation.

| Fichier | Responsabilité |
| --- | --- |
| `experiments/graph_ablation.py` | Métadonnées complètes, exports, construction des variantes, extraction d'une tâche entraînable |
| `artifacts/multimodal_study.py` | Schémas, signatures, écritures atomiques, contrôles d'intégrité et réutilisation |
| `experiments/multimodal_study.py` | Grilles par étapes et dépendances, sans dupliquer le pipeline |
| `pipelines/diagnose_gnn_information.py` | Replay de tous les contrôles actuels, y compris marché/résidus |
| `models/multimodal_system.py` | Politique explicite de calibration et fusion masquée compatible |
| `models/frozen_branch_fusion.py` | Branches gagnantes indépendantes et gelées, y compris leurs gates |
| `experiments/multimodal_fusion.py` | Ajustements internes, évaluations externes et diagnostics de complémentarité |
| `models/market_gru_ablation.py` | Propagation explicite de la température du gate |
| `scripts/run_us_multimodal_study.py` | Entrée CLI portable et reprise |
| `scripts/compare_us_multimodal_study.py` | Tables, courbes et rapport Markdown |

Les chemins du tableau sont relatifs à `src/trading_system/` sauf `scripts/`. Aucune nouvelle dépendance lourde n'est nécessaire a priori ; utiliser PyTorch, NumPy, Pandas/Parquet et `tqdm` déjà présents. Ne pas ajouter FinBERT ou une API payante.

## 9. Comparaison et règles de décision

Définir les critères avant chaque lancement, pas après lecture des scores.

Métriques principales : Sharpe régularisé, rendement/PnL net, drawdown. Diagnostics : pire fold, dispersion entre seeds, exposition brute/nette, turnover, coûts, macro-F1, NLL, ECE, couverture, durée, nombre de paramètres et pic RAM/VRAM.

Comparer par différences appariées pour chaque `(fold, seed)`. Montrer ensuite l'agrégation par fold et la dispersion entre seeds. Les neuf observations ne sont pas neuf périodes indépendantes. Pour l'incertitude des chemins journaliers, utiliser un bootstrap par blocs en conservant l'appariement des variantes ; ne pas compter chaque ticker/jour comme une observation indépendante.

Règle de promotion exploratoire :

1. Tous les folds/seeds sont complets et les contrôles de causalité passent.
2. Le delta moyen de Sharpe régularisé est positif, avec un gain agrégé positif sur au moins deux des trois folds.
3. Le gain ne repose pas uniquement sur une seed ; afficher tous les résultats, y compris les pertes.
4. Les dégradations de drawdown, les coûts et le budget de calcul sont acceptables selon des limites fixées dans le manifeste avant lancement.
5. À scores proches ou incertains, préférer la variante plus simple. Ne pas annoncer de supériorité statistique à partir du seul nombre de victoires.

L'augmentation d'exposition n'invalide pas automatiquement un gain, mais il faut la rendre visible. Présenter les résultats exécutables d'abord ; un diagnostic à exposition égalisée est descriptif et ne remplace pas une règle d'allocation causale.

Les choix successifs sur les mêmes folds restent exploratoires et exposés au surapprentissage de recherche. Garder un journal de toutes les variantes testées. Après choix définitif, figer le modèle et ouvrir le holdout **une seule fois**, avec une règle de réentraînement définie en amont. Un retour sur les choix après lecture du holdout ferait perdre son statut de test final.

## 10. Artefacts attendus

```text
artifacts/comparisons/us-relational-market/
  01-combined-025/                 # Référence actuelle, conservée
  optimization/
    study.json                    # Protocole et liste de toutes les tâches
    registry.json                 # Signatures et provenance des imports
    02-features/                  # 32/64, sans/avec gate
    03-gnn-depth/
    04-market-depth/
    05-fusion/
    summary.csv
    paired_deltas.csv
    report.md
```

Chaque tâche possède un dossier immuable avec metadata, checkpoint/prétraitement, prédictions internes/externes, chemins journaliers et métriques. Les imports référencent le run source et ses empreintes, sans recopier ni modifier les résultats originaux. Les figures comparent les mêmes périodes et incluent une référence buy-and-hold calculée avec les mêmes conventions.

## 11. Budget d'entraînements

Je budgète maintenant P1 à **108 entraînements maximum** : 12 variantes, trois folds et trois seeds. Les contrôles `identity` et `identity_market` aux deux caps sont inclus, pas conditionnels. Le [guide P1](us-feature-gate-interaction.md) décrit la grille actuelle et ses commandes.

Je conserve ci-dessous l'estimation historique du plan initial, avant l'ajout de `identity_market` à P1 et avant le constat du pool historique différent. Elle supposait trois folds, trois seeds, des références 32 compatibles et aucune relance due à un changement de protocole. Son total de 90 entraînements supplémentaires ne décrit pas le budget actuel de P1.

| Étape du plan initial | Nouveaux entraînements historiques estimés |
| --- | ---: |
| Référence US historique | 72, déjà lancés ou terminés dans l'estimation initiale |
| P0 provenance/replay | 0, hors contrôle non reproductible |
| P1 quatre variantes à 64 features | 36 |
| P1 contrôle `identity` à 64 features | 9 |
| P2 GNN profondeur 2, sans/avec gate | 18 |
| P2 contrôle `identity` à profondeur 2 | 9 |
| P2 Transformer profondeur 2, GRU/GNN | 18 |
| P3 fusion de checkpoints gelés | 0, plus petits fits de calibration/poids |
| **Total supplémentaire historique, contrôles inclus** | **90** |

Dans cette estimation historique, `sector_market` à 32 et `identity_market` étaient des extensions non incluses. Je compte désormais `identity_market` aux deux caps dans les 108 de P1. Les extensions ultérieures restent séparées : nouvelle configuration `identity_market` en P2 (+9 par configuration non disponible), gate statique et encodeur marché simple (+18 pour une branche au cap retenu), interaction GNN 2/Transformer 2 (+9), profondeur GNN 3 (+18 pour les deux variantes), ou renouvellement d'un contrôle incompatible (+9 par variante).

Je vérifie le budget réel avec `--dry-run`. Les références 32 ne sont pas réutilisables via un simple `--resume` après modification des paramètres globaux ; seule la réutilisation vérifiée de P0 peut réduire le nombre d'entraînements nouveaux.

## 12. Tests et validation

### Données et causalité

- Modifier les données futures ne change pas la sélection, les fills, les scalers ou les graphes antérieurs.
- Chaque graphe respecte sa date de provenance ; aucune arête n'utilise une séance future.
- Les contextes marché restent séparés des séquences par action.
- Les dates cibles sont disjointes entre partitions, même si les fenêtres réutilisent un historique antérieur.
- Les timestamps de publication suivent le contrat existant ; clôture J utilisable uniquement pour exécution ultérieure.

### Modèles, gate et fusion

- Gate initial uniforme = identité ; température réellement routée au GRU et au GNN.
- GRU sans gate et GNN sans gate conservent leur comportement actuel.
- Profondeur configurée = profondeur construite, sans modifier largeur, graphe ou contexte.
- Fusion de branches identiques = prédiction de cette branche ; poids extrêmes = branche correspondante.
- Calibration/fusion n'accèdent jamais aux données externes ou finales pendant leurs fits.
- Les températures sont positives/finies ; les probabilités sont normalisées.
- Masques disponibles/absents corrects, poids renormalisés, aucun label inconnu assimilé à Hold.
- Adapter deux contrôles gated ne remplace pas leurs Transformers indépendants par un Transformer partagé.
- Replay et table de logits correspondent exactement aux clés du backtest ; mélange de logits et mélange de positions restent distingués.

### Artefacts et exécution

- Round-trip checkpoint + prétraitement reproduit les prédictions dans la tolérance déclarée.
- Modification de frontière CV, ticker order, prix, contexte ou code pertinent refuse la réutilisation.
- Fichier incomplet/corrompu n'est pas marqué terminé ; reprise après interruption ne duplique pas un fold.
- `--dry-run` produit les comptes attendus et n'entraîne rien.
- Tests CLI sous Mac et Windows, smoke test CPU puis GPU si disponible.
- Exports batchés et graphes persistés sparse, sans matérialisation globale `[dates,tickers,temps,features]`. Conserver la possibilité d'une adjacency dense par date dans le chemin MPS déterministe existant ; ne pas construire de tenseur dense global de tous les graphes.

Après les tests ciblés, lancer la suite complète. Faire ensuite un petit smoke run séparé, jamais utilisé pour choisir les gagnants du benchmark principal.

## 13. Ordre d'implémentation et critères de fin

- [x] **P0a** : figer le protocole, les dates et les sources ; auditer la compatibilité du run actuel.
- [x] **P0b** : compléter provenance/checkpoints, exporter les logits et valider le replay.
- [x] **P0c** : orchestrateur portable, signatures de réutilisation, reprise atomique et `--dry-run`.
- [x] **P1 benchmark** : 108/108 tâches, `identity_market` aux deux caps, audits et comparaisons financières documentés ; cap 32 et gate désactivé comme configuration de travail.
- [ ] **P1 diagnostic complémentaire** : exporter les poids du gate par date, entropie, stabilité et nombre effectif de features ; distinct de la complétude du benchmark.
- [ ] **Attribution des features** : étude financière après intégration LSE PIT ; ne pas confondre classement par variance et importance prédictive.
- [ ] **P2a** : tester GNN 1/2 avec contrôles de capacité.
- [ ] **P2b** : tester Transformer 1/2 et, seulement si justifié, leur interaction.
- [ ] **P3a** : adaptateur de branches gelées et fusions fixes reproductibles.
- [ ] **P3b** : calibration, poids statique et confiance, avec contrôles calibrés seuls.
- [ ] **P4** : rapport complet, décision figée et définition du test final.

Une étape est terminée quand son code est testé, ses artefacts permettent une reproduction, sa comparaison est complète et sa décision est documentée. Un score terminal isolé ou un modèle gagnant sur une seule seed ne suffit pas.

Extensions après décision, dans des études distinctes : température du gate, sélection dure dynamique, top-k/lookback du graphe, Transformer partagé, fine-tuning conjoint, incertitude MC dropout, puis sentiment point-in-time lorsque les données sont disponibles.
