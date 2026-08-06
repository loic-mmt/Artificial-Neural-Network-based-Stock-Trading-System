# Plan d'implémentation des modèles neuronaux séquentiels

## Statut

Plan à implémenter.

Modèles visés :

- ANN dense actuelle, conservée comme référence ;
- RNN simple ;
- LSTM ;
- GRU ;
- Transformer Encoder temporel.

Le but n'est pas de présupposer le résultat, mais de tester proprement l'hypothèse de
recherche : à données, labels et protocole identiques, quelles architectures neuronales
produisent les meilleurs signaux de trading hors échantillon ?

## 1. Problème actuel

La restructuration existante sépare déjà correctement les responsabilités : données,
labels, modèles, évaluation et backtest. Elle n'est toutefois pas encore totalement
adaptée aux réseaux séquentiels :

- `data/windows.py` aplatit les fenêtres en tableaux 2D ;
- `data/scaling.py` accepte uniquement des tableaux 2D ;
- l'ANN dense est directement présente dans `ExperimentConfig` ;
- le runner accepte plusieurs classifieurs, mais ne décrit pas leur forme d'entrée ;
- aucun socle commun n'existe encore pour l'entraînement PyTorch, les checkpoints,
  le device et la reproductibilité.

Un RNN, LSTM, GRU ou Transformer attend normalement une entrée :

```text
(nombre_d_exemples, longueur_de_sequence, nombre_de_features)
```

Il serait possible de reconstruire cette forme dans chaque modèle à partir des fenêtres
aplaties. Cette solution est interdite dans la cible : elle dupliquerait la logique et
rendrait les comparaisons plus difficiles à contrôler.

## 2. Principes non négociables

### 2.1 Comparaison scientifique équitable

Tous les modèles d'une même expérience doivent partager :

- les mêmes lignes de données ;
- les mêmes features et le même ordre de colonnes ;
- les mêmes labels ;
- les mêmes splits chronologiques train/validation/test ;
- la même longueur de contexte ;
- la même normalisation calculée uniquement sur le train ;
- la même politique de décision à partir des probabilités ;
- le même délai d'exécution, les mêmes frais et le même backtest ;
- le même budget de recherche d'hyperparamètres ;
- la même liste de seeds.

Le test final ne doit jamais servir à choisir un modèle, un seuil ou un
hyperparamètre. Toute sélection utilise seulement train et validation.

### 2.2 Absence de fuite temporelle

- Une fenêtre se terminant à la date `t` ne contient aucune donnée après `t`.
- Une fenêtre multi-ticker ne traverse jamais deux tickers.
- Le scaler est ajusté uniquement sur les fenêtres du train.
- La calibration des seuils utilise uniquement la validation.
- Une prédiction produite à `t` est exécutée au plus tôt à `t+1`.
- En walk-forward, chaque réentraînement utilise uniquement l'historique disponible.

### 2.3 Une responsabilité par couche

```text
data         -> séquences, splits, normalisation
features     -> calcul des variables explicatives
labels       -> construction des cibles
models       -> architecture, entraînement, probabilités
evaluation   -> métriques et décisions
backtest     -> positions, frais, PnL et benchmarks
experiments  -> orchestration et comparaison
pipelines    -> CLI et configuration uniquement
```

Les modèles ne doivent jamais lire un parquet, calculer des labels, choisir un split,
lancer un backtest ou produire un graphique.

## 3. Architecture cible

```text
src/trading_system/
├── data/
│   ├── windows.py
│   └── scaling.py
├── models/
│   ├── base.py
│   ├── factory.py
│   ├── specs.py
│   ├── manual_ann/
│   │   ├── manual_nn.py
│   │   └── sequence_adapter.py
│   └── neural/
│       ├── __init__.py
│       ├── base.py
│       ├── config.py
│       ├── trainer.py
│       ├── rnn.py
│       ├── lstm.py
│       ├── gru.py
│       └── transformer.py
├── experiments/
│   ├── config.py
│   ├── runner.py
│   ├── walkforward.py
│   ├── search.py
│   └── comparison.py
└── artifacts/
    └── serialization.py
```

Le nom exact de certains fichiers peut évoluer pendant l'implémentation. Les frontières
de responsabilité ci-dessus doivent rester stables.

## 4. Contrat de données canonique

### 4.1 Séquences 3D

`data/windows.py` doit produire des séquences non aplaties :

```python
X.shape == (n_samples, context_len, n_features)
y.shape == (n_samples,)
```

API cible :

```python
build_sequence_dataset(...)
build_sequence_features(...)
build_sequence_dataset_with_history(...)
```

Chaque fonction doit pouvoir retourner les lignes ou indices alignés avec les cibles.
La gestion de `group_col` doit empêcher toute fenêtre entre deux actifs.

Les anciennes fonctions `build_context_*` restent temporairement disponibles comme
wrappers compatibles avec `layout="flat"`. Elles seront dépréciées après migration de
tous les appels internes.

### 4.2 Normalisation séquentielle

Créer un `SequenceStandardizer` qui :

- accepte uniquement `(N, T, F)` ;
- calcule une moyenne et un écart-type par feature sur les axes `(N, T)` ;
- conserve un état de forme `(1, 1, F)` ;
- est ajusté sur le train seulement ;
- transforme validation, test et chunks walk-forward sans nouvel ajustement ;
- vérifie le nombre et l'ordre des features.

La normalisation doit être faite avant l'adaptation propre au modèle. L'ANN dense reçoit
donc les mêmes valeurs normalisées que les modèles séquentiels, puis aplatit `(T, F)`.

### 4.3 Adaptateur de l'ANN dense

Créer `ManualANNSequenceAdapter` :

```text
(N, T, F) -> normalisation commune -> (N, T*F) -> ManualANNClassifier
```

La fonction d'aplatissement ne doit exister qu'à cet endroit. Le code mathématique de
`ManualANNClassifier` reste indépendant des données temporelles.

La migration peut modifier légèrement les résultats historiques de l'ANN, car la
normalisation deviendra identique entre architectures. Sauvegarder les métriques de
référence avant migration et documenter cette rupture méthodologique.

## 5. Contrat commun des modèles

Conserver l'idée de `ProbabilisticClassifier`, mais documenter que l'entrée canonique du
runner est une séquence 3D.

```python
class ProbabilisticSequenceClassifier(Protocol):
    classes_: np.ndarray
    model_name: str

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> FitResult: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
```

Contraintes de sortie :

- `predict_proba()` renvoie `(N, 3)` ;
- ordre fixe des colonnes : `[Sell, Hold, Buy]` ;
- chaque valeur est finie et comprise entre 0 et 1 ;
- chaque ligne somme à 1 à la tolérance numérique près ;
- `predict_proba()` ne modifie pas l'état d'entraînement ;
- `fit()` retourne un `FitResult` commun.

Étendre `FitResult` et `TrainingHistory` sans les rendre dépendants de PyTorch :

- loss train et validation ;
- balanced accuracy et macro-F1 optionnelles ;
- meilleur epoch ;
- raison d'arrêt ;
- durée d'entraînement ;
- nombre de paramètres ;
- seed et device utilisés.

## 6. Configuration et factory

### 6.1 Séparer expérience et modèle

`ExperimentConfig` doit contenir uniquement les choix communs : données, features,
labels, splits, contexte, décision et backtest. Retirer son champ `manual_ann`.

Créer des configurations typées :

```text
CommonTrainingConfig
├── epochs
├── batch_size
├── learning_rate
├── weight_decay
├── patience
├── min_delta
├── seed
├── device
└── gradient_clip_norm

ManualANNConfig
RNNConfig
LSTMConfig
GRUConfig
TransformerConfig
```

Les champs spécifiques restent dans leur configuration :

- RNN/LSTM/GRU : `hidden_size`, `num_layers`, `bidirectional`, `dropout` ;
- Transformer : `d_model`, `n_heads`, `num_layers`, `dim_feedforward`, `dropout`,
  `pooling` et type d'encodage positionnel.

`bidirectional` doit être `False` par défaut. Il n'est pas une fuite temporelle lorsque
toute la fenêtre appartient au passé, mais il change fortement la capacité du modèle et
doit être testé comme hyperparamètre explicite.

### 6.2 Factory construite après les features

Le runner connaît `context_len`, `n_features` et `n_classes` seulement après préparation
des données. Utiliser une factory recevant un contexte de construction :

```python
ModelBuildContext(
    input_size=n_features,
    context_len=context_len,
    num_classes=3,
    seed=seed,
    device=device,
)
```

Le registre doit mapper un nom stable vers une factory :

```text
manual_ann
rnn
lstm
gru
transformer
```

Une architecture inconnue doit provoquer une erreur claire. Aucun `if/elif` propre à une
architecture ne doit être ajouté dans le runner.

L'adaptateur sklearn peut rester pour compatibilité, mais il n'appartient pas au chemin
principal des expériences neuronales et ne doit pas influencer leur conception.

## 7. Socle PyTorch commun

Utiliser PyTorch pour RNN, LSTM, GRU et Transformer. L'ANN NumPy actuelle reste une
implémentation indépendante.

Ajouter une dépendance optionnelle afin que les composants non neuronaux restent
utilisables sans PyTorch :

```toml
[project.optional-dependencies]
neural = ["torch"]
dev = ["pytest>=8"]
```

Les imports PyTorch doivent être locaux ou confinés à `models/neural/`. En son absence,
une erreur explique d'installer `.[neural]`.

### 7.1 `TorchSequenceClassifier`

Créer un socle commun chargé de :

- conversion NumPy vers tenseurs ;
- sélection `cpu`, `cuda` ou `mps`, avec `auto` explicite ;
- `DataLoader` et minibatches ;
- calcul des poids de classes sur le train ;
- `CrossEntropyLoss` à partir des logits ;
- optimizer Adam ou AdamW configuré ;
- clipping du gradient ;
- early stopping sur la loss de validation ;
- copie et restauration du meilleur `state_dict` ;
- passage correct entre `train()` et `eval()` ;
- inférence sous `torch.no_grad()` ;
- softmax uniquement dans `predict_proba()` ;
- historique et diagnostics communs ;
- sauvegarde/rechargement du meilleur état.

Les classes d'architecture ne doivent définir que leur réseau et leur `forward()`.

### 7.2 Reproductibilité

Pour chaque run :

- initialiser NumPy, Python et PyTorch avec la seed du run ;
- initialiser les workers du `DataLoader` ;
- rendre le shuffle reproductible ;
- documenter les limites de déterminisme GPU ;
- enregistrer versions, device et seed dans l'artefact ;
- ne jamais utiliser une seed globale au moment de l'import.

## 8. Implémentation des architectures

### 8.1 RNN simple

- Utiliser `nn.RNN` avec `batch_first=True`.
- Partir du dernier état caché valide.
- Ajouter une tête linéaire vers trois logits.
- Supporter plusieurs couches et dropout entre couches.
- Commencer par une version unidirectionnelle.

Critère d'acceptation : entraînement CPU, probabilités valides, restauration du meilleur
epoch et capacité à sur-apprendre un minuscule dataset synthétique.

### 8.2 LSTM

- Utiliser `nn.LSTM` avec `batch_first=True`.
- Utiliser le dernier état caché `h_n`, pas l'état cellule `c_n`, pour la classification.
- Gérer correctement `num_layers` et le mode bidirectionnel éventuel.
- Réutiliser intégralement le trainer commun.

Critère d'acceptation : aucun code de boucle d'entraînement dupliqué depuis RNN.

### 8.3 GRU

- Utiliser `nn.GRU` avec `batch_first=True`.
- Utiliser le dernier état caché pour la tête de classification.
- Partager les mêmes options que RNN/LSTM lorsque leur sens est identique.

Critère d'acceptation : seule la cellule et sa configuration spécifique diffèrent.

### 8.4 Transformer temporel

Architecture minimale :

```text
features F
-> projection linéaire vers d_model
-> encodage positionnel
-> TransformerEncoder
-> pooling temporel
-> LayerNorm
-> tête linéaire vers 3 logits
```

Règles :

- vérifier que `d_model` est divisible par `n_heads` ;
- fournir un véritable encodage positionnel ;
- choisir explicitement `pooling="last"`, `"mean"` ou token CLS ;
- ne jamais introduire des lignes après la date cible ;
- documenter le choix du masque d'attention.

Pour une classification de fenêtre ne contenant que le passé jusqu'à `t`, l'attention
complète dans la fenêtre n'est pas une fuite. Un masque causal peut rester une option de
recherche, mais ne doit pas changer silencieusement entre expériences.

Critère d'acceptation : un test prouve que l'encodage positionnel rend le modèle sensible
à l'ordre d'une séquence.

## 9. Intégration aux expériences

### 9.1 Runner statique

Modifier `run_experiment` pour suivre strictement :

```text
filtrer univers
-> construire labels
-> construire features
-> split chronologique
-> construire séquences 3D
-> ajuster scaler sur train
-> transformer train/val/test
-> construire le modèle via factory
-> fit(train, val)
-> predict_proba(train/val/test)
-> calibrer décision sur val
-> métriques test
-> backtest test
-> artefact structuré
```

Le runner ne doit connaître aucun détail RNN, LSTM, GRU ou Transformer.

### 9.2 Walk-forward

- Réutiliser la même factory et le même contrat 3D.
- Recréer un modèle neuf pour chaque fenêtre de réentraînement.
- Ajuster un nouveau scaler uniquement sur l'historique du chunk.
- Dériver chaque seed de façon stable depuis `run_seed` et `chunk_id`.
- Interdire la réutilisation accidentelle des poids du chunk suivant, sauf option de
  warm-start explicitement étudiée plus tard.
- Conserver les métriques et durées de chaque réentraînement.

### 9.3 Recherche d'hyperparamètres

- Remplacer les grilles ANN-spécifiques par des espaces propres à chaque modèle.
- Imposer le même nombre maximum de trials par architecture.
- Sélectionner sur une métrique validation définie à l'avance.
- Conserver les trials échoués avec leur erreur.
- Ne jamais trier ou filtrer selon une métrique test.
- Enregistrer le nombre de paramètres et la durée pour comparer le coût des modèles.

## 10. Protocole expérimental final

### 10.1 Étapes

1. Définir une configuration de données gelée.
2. Définir les espaces d'hyperparamètres avant de regarder le test.
3. Rechercher chaque architecture avec le même budget.
4. Choisir sa meilleure configuration sur validation.
5. Réentraîner avec plusieurs seeds, sans utiliser le test pour décider.
6. Évaluer une seule fois les modèles retenus sur le test.
7. Rapporter moyenne, écart-type et résultats par seed.

### 10.2 Mesures obligatoires

Classification :

- macro-F1 ;
- balanced accuracy ;
- précision et rappel par classe ;
- matrice de confusion ;
- taux d'action Buy/Sell/Hold.

Trading :

- capital final et rendement ;
- surperformance contre buy-and-hold ;
- Sharpe et Sortino ;
- maximum drawdown ;
- turnover ;
- nombre de transactions ;
- frais totaux.

Coût du modèle :

- nombre de paramètres ;
- meilleur epoch ;
- durée d'entraînement ;
- durée d'inférence ;
- mémoire maximale si disponible.

### 10.3 Rapport comparatif

Créer une table longue avec une ligne par combinaison :

```text
dataset / ticker / feature_set / label_mode / model / config_hash / seed
```

Puis produire une synthèse par modèle avec moyenne et dispersion. Ne pas conclure qu'une
architecture est meilleure à partir d'un seul ticker, d'une seule période ou d'une seule
seed.

## 11. Artefacts et sérialisation

Chaque run doit sauvegarder :

- configuration complète et hash stable ;
- nom et hyperparamètres du modèle ;
- `state_dict` PyTorch ou poids NumPy, jamais seulement l'objet Python picklé ;
- état du scaler ;
- liste ordonnée des features ;
- longueur de contexte ;
- schéma `[Sell, Hold, Buy]` ;
- seuils de décision calibrés ;
- historique d'entraînement ;
- métriques classification et trading ;
- seed, device et versions des dépendances.

Le chargement doit vérifier la compatibilité de l'architecture, des dimensions, des
features et du schéma de labels avant toute prédiction.

## 12. Tests à ajouter

### 12.1 Données

- [ ] Les séquences ont exactement la forme `(N, T, F)`.
- [ ] L'ordre temporel est conservé.
- [ ] Les cibles sont alignées avec la dernière ligne de leur séquence.
- [ ] Aucune séquence ne traverse deux tickers.
- [ ] L'historique permet de construire le début de validation/test sans fuite.
- [ ] Le scaler produit une statistique par feature.
- [ ] Modifier validation/test ne change pas le scaler ajusté.
- [ ] L'adaptateur ANN aplatit en `(N, T*F)` sans changer l'ordre.

### 12.2 Contrat modèle

- [ ] Chaque modèle accepte les séquences 3D attendues.
- [ ] Chaque modèle renvoie `(N, 3)`.
- [ ] Les probabilités sont finies, positives et normalisées.
- [ ] `classes_` vaut toujours `[0, 1, 2]`.
- [ ] Deux entraînements CPU avec la même seed donnent le même résultat à tolérance fixée.
- [ ] Deux seeds différentes peuvent produire des poids différents.
- [ ] Le meilleur état est restauré après early stopping.
- [ ] `predict_proba()` utilise le mode évaluation et désactive les gradients.
- [ ] Chaque modèle peut sur-apprendre un petit dataset synthétique.

### 12.3 Architecture

- [ ] RNN, LSTM et GRU gèrent correctement `batch_first`.
- [ ] Les dimensions cachées multi-couches sont correctes.
- [ ] Le Transformer refuse `d_model % n_heads != 0`.
- [ ] Le Transformer contient un encodage positionnel testé.
- [ ] Aucun modèle ne duplique la boucle du trainer commun.
- [ ] Aucun modèle n'importe `pipelines`, `backtest` ou `data/io.py`.

### 12.4 Intégration

- [ ] Un smoke test statique CPU passe pour les cinq architectures.
- [ ] Un smoke test walk-forward CPU passe pour les cinq architectures.
- [ ] Le même split et les mêmes cibles sont utilisés par tous les modèles.
- [ ] Les décisions sont toujours calibrées sur validation uniquement.
- [ ] Le délai d'exécution reste `t+1`.
- [ ] Sauvegarder puis recharger un modèle conserve ses probabilités.
- [ ] Le package reste importable sans PyTorch lorsque les modules neuronaux ne sont pas utilisés.

## 13. Phases d'implémentation

### Phase 0 — Geler la référence actuelle

- [ ] Exécuter et enregistrer les 17 tests existants.
- [ ] Créer un petit dataset synthétique de référence versionné dans les tests.
- [ ] Enregistrer splits, labels, métriques et backtest actuels de l'ANN.
- [ ] Documenter les changements attendus dus à la normalisation séquentielle.

Sortie : référence permettant de distinguer une amélioration voulue d'une régression.

### Phase 1 — Données séquentielles

- [ ] Ajouter les builders 3D.
- [ ] Ajouter `SequenceStandardizer`.
- [ ] Ajouter l'adaptateur de l'ANN dense.
- [ ] Migrer le runner statique.
- [ ] Migrer le walk-forward.
- [ ] Garder les wrappers 2D compatibles.
- [ ] Ajouter tous les tests de données et de fuite.

Sortie : l'ANN actuelle fonctionne de bout en bout à partir des séquences 3D.

### Phase 2 — Contrat, configurations et factory

- [ ] Séparer `ExperimentConfig` des configurations modèle.
- [ ] Créer `ModelBuildContext`.
- [ ] Créer le registre de modèles.
- [ ] Étendre `FitResult` et l'artefact entraîné.
- [ ] Supprimer les branches ANN-spécifiques du runner et de la recherche.

Sortie : ajouter une architecture ne nécessite aucune modification du runner.

### Phase 3 — Socle PyTorch

- [ ] Ajouter l'extra `neural`.
- [ ] Implémenter le trainer commun.
- [ ] Implémenter seeds, device, early stopping et checkpoints.
- [ ] Ajouter les tests contractuels paramétrés.

Sortie : une fausse petite architecture PyTorch peut être entraînée et rechargée.

### Phase 4 — RNN, LSTM et GRU

- [ ] Implémenter RNN.
- [ ] Valider tous les tests avant de continuer.
- [ ] Implémenter LSTM sans recopier le trainer.
- [ ] Implémenter GRU sans recopier le trainer.
- [ ] Ajouter les espaces de recherche propres à chaque modèle.

Sortie : trois modèles récurrents utilisables en statique et walk-forward.

### Phase 5 — Transformer

- [ ] Implémenter projection, positions, encoder, pooling et tête.
- [ ] Ajouter masque causal optionnel documenté.
- [ ] Ajouter validations de dimensions.
- [ ] Ajouter tests d'ordre et d'encodage positionnel.
- [ ] Ajouter son espace de recherche.

Sortie : Transformer utilisable avec exactement le même runner et les mêmes données.

### Phase 6 — CLI, artefacts et comparaison

- [ ] Ajouter `--model manual_ann|rnn|lstm|gru|transformer`.
- [ ] Charger une configuration par fichier ou arguments typés.
- [ ] Sauvegarder configurations, poids, métriques et historiques.
- [ ] Ajouter une commande de comparaison multi-modèle et multi-seed.
- [ ] Produire CSV/JSON exploitables pour les tableaux du mémoire ou papier.

Sortie : une commande reproductible lance et compare toutes les architectures.

### Phase 7 — Validation finale et documentation

- [ ] Exécuter tous les tests CPU.
- [ ] Exécuter les tests GPU disponibles sans les rendre obligatoires en CI.
- [ ] Vérifier lint, formatage et imports optionnels.
- [ ] Mettre à jour le README et `docs/repo-structure.md`.
- [ ] Documenter le protocole expérimental avant les résultats finaux.
- [ ] Marquer les anciennes API 2D comme dépréciées.

Sortie : implémentation reproductible, documentée et prête pour les expériences finales.

## 14. Ordre recommandé des premiers commits

1. `test: characterize current ANN experiment outputs`
2. `feat(data): add canonical 3D sequence windows`
3. `feat(data): add train-only sequence standardizer`
4. `refactor(models): adapt manual ANN to canonical sequences`
5. `refactor(experiments): introduce model factory and neutral configs`
6. `feat(models): add shared PyTorch sequence trainer`
7. `feat(models): add simple RNN classifier`
8. `feat(models): add LSTM classifier`
9. `feat(models): add GRU classifier`
10. `feat(models): add temporal Transformer classifier`
11. `feat(experiments): add fair multi-model comparison runner`
12. `docs: document neural model protocol and experiment methodology`

Chaque commit doit garder les tests existants au vert et ajouter les tests de sa phase.

## 15. Définition de terminé

L'implémentation est terminée lorsque :

- les cinq modèles utilisent les mêmes séquences 3D normalisées ;
- aucune architecture ne duplique préparation, entraînement commun, évaluation ou backtest ;
- chaque modèle fonctionne en statique et walk-forward ;
- les résultats sont reproductibles par configuration et seed ;
- la sélection des modèles ne consulte jamais le test ;
- les checkpoints peuvent être rechargés pour reproduire les probabilités ;
- les tests couvrent formes, fuites, probabilités, early stopping et sérialisation ;
- une commande compare toutes les architectures avec un budget équitable ;
- les résultats contiennent qualité prédictive, performance de trading et coût de calcul ;
- le README explique comment ajouter une nouvelle architecture neuronale sans modifier le runner.

