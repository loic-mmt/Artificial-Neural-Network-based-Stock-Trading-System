# Plan d'implémentation du labeling

Statut : proposition de travail. Ce document décrit l'architecture, les méthodes,
la CLI, les expériences et les tests. Il ne constitue pas encore une validation
de performance.

## 1. Point de départ

Le benchmark `20260902T083027Z` montre, sur le test courant :

- buy & hold : PnL `+6 102,10` ;
- reproduction parfaite des labels `breakout` : PnL `+734,38` ;
- sous-performance parfaite des labels : `-5 367,72` ;
- distribution test : `9 499 Hold`, `326 Sell`, `325 Buy`.

Le problème comporte donc deux dimensions distinctes :

1. **alignement économique** : une reproduction parfaite des labels doit produire
   une stratégie cohérente avec l'objectif déclaré ;
2. **apprenabilité** : les états doivent avoir une structure temporelle, une
   couverture suffisante et une fréquence de transitions raisonnable.

Le test 2022–2026 ayant déjà été consulté, il devient une période de développement.
Il ne peut plus servir de test final non biaisé après modification des labels. La
validation finale devra utiliser de nouvelles dates, des tickers jamais consultés,
ou un protocole walk-forward imbriqué préenregistré.

## 2. Objectif et non-objectifs

### Objectif principal

Construire un système de labeling interchangeable qui permette de :

- sélectionner chaque méthode depuis `argparse` ;
- reproduire exactement une configuration depuis un fichier JSON et un hash ;
- mesurer la valeur économique des labels avant d'entraîner un réseau ;
- mesurer leur apprenabilité avec des baselines simples ;
- utiliser les mêmes labels en statique, walk-forward, grid search et comparaison
  de modèles ;
- empêcher tout accès au test pendant la sélection.

### Objectifs économiques explicites

La configuration doit déclarer l'un des objectifs suivants :

- `absolute_return` : produire un PnL net positif face au cash ;
- `excess_buy_hold` : améliorer buy & hold, généralement avec une base long/flat ;
- `cross_sectional_alpha` : produire un rendement relatif entre actifs ; son
  benchmark doit être un portefeuille neutre cohérent, pas automatiquement buy & hold.

Les résultats ne doivent jamais mélanger ces objectifs ou leurs benchmarks.

### Non-objectifs

- Garantir qu'un label profitable restera prédictible hors échantillon.
- Utiliser l'Oracle comme performance de production.
- Optimiser les paramètres de labels sur le test final.
- Ajouter immédiatement une dépendance externe de labeling ; les premières
  versions peuvent être écrites avec NumPy et pandas.

## 3. Contrat technique commun

### 3.1 Configuration

Créer `src/trading_system/labels/config.py` :

```python
@dataclass(frozen=True)
class LabelConfig:
    method: str
    objective: str = "absolute_return"
    semantics: str = "action"
    horizon: int = 5
    volatility_window: int = 20
    long_threshold: float = 1.0
    short_threshold: float = 1.25
    exit_threshold: float = 0.25
    min_holding_period: int = 5
    cooldown: int = 0
    cost_bps: float = 5.0
    parameters: dict[str, object] = field(default_factory=dict)
```

Les paramètres propres à une famille restent dans `parameters`. Les paramètres
communs, nécessaires aux manifests et aux garde-fous, restent typés.

`ExperimentConfig` reçoit `label: LabelConfig`. Les anciens champs
`label_mode`, `label_window`, `forward_horizon`, etc. restent temporairement des
adaptateurs de compatibilité, puis sont supprimés après migration des scripts.

### 3.2 Résultat standardisé

Créer un résultat indépendant du labeler :

```python
@dataclass
class LabelResult:
    frame: pd.DataFrame
    known_mask: np.ndarray
    class_names: tuple[str, ...]
    semantics: Literal["action", "target_position", "meta"]
    metadata: dict[str, object]
```

Le `frame` contient au minimum :

- `Label_id` ;
- `Label` ;
- `label_score` continu avant discrétisation, lorsque disponible ;
- `target_position` pour les labels d'état ;
- `label_event_id` et `label_end_date` pour les méthodes événementielles.

`known_mask` interdit l'apprentissage sur une cible dont l'horizon dépasse la
frontière du split.

### 3.3 Sémantique explicite

Deux contrats différents doivent être supportés :

| Sémantique | Classes | Interprétation de la classe centrale |
|---|---|---|
| `action` | Sell / Hold / Buy | Conserver la position précédente |
| `target_position` | Short / Flat / Long | Position cible quotidienne ; Flat ferme la position |

Le backtest doit recevoir la sémantique explicitement. Il ne doit jamais déduire
le comportement à partir du nombre `0`, `1` ou `2`. Cette séparation empêche de
traiter accidentellement `Flat` comme `Hold`.

La meta-labelisation est binaire (`Skip`, `Take`). Le registry de modèles et les
métriques doivent donc accepter `num_classes=2` ou `3` au lieu de supposer trois
classes partout.

### 3.4 Registry

Créer `src/trading_system/labels/registry.py` avec une interface unique :

```python
Labeler = Callable[[pd.DataFrame, LabelConfig, SplitContext], LabelResult]

registry.register("breakout", build_breakout_labels)
registry.register("forward_return", build_forward_return_labels)
registry.register("volatility_position", build_volatility_position_labels)
registry.register("triple_barrier", build_triple_barrier_labels)
registry.register("meta_label", build_meta_labels)
registry.register("cross_sectional_rank", build_cross_sectional_rank_labels)
registry.register("regularized_oracle", build_regularized_oracle_labels)
```

Le runner appelle uniquement le registry. Aucun `if label_mode == ...` ne doit
rester dans `experiments/runner.py` ou `experiments/walkforward.py`.

## 4. Méthodes à implémenter

### M0 — `breakout`

Conserver la méthode actuelle comme baseline contrôlée.

- sémantique : `action` ;
- paramètres : fenêtre, buffers achat/vente, alternance optionnelle ;
- objectif : référence historique, pas candidat principal ;
- migration : supprimer les backtests dupliqués de `breakout_gridsearch.py` et
  utiliser le moteur canonique.

### M1 — `forward_return`

Conserver le rendement futur à horizon fixe comme baseline simple.

```text
r(t,h) = P(t+h) / P(t) - 1
```

- horizons initiaux : `1`, `3`, `5`, `10` ;
- seuils achat/vente asymétriques ;
- zone neutre couvrant bruit et coûts ;
- targets qui franchissent une frontière de split : inconnues et exclues du fit ;
- sémantique actuelle : `action` pour compatibilité.

### M2 — `volatility_position` — candidat principal

Construire des états persistants Short/Flat/Long :

```text
score(t,h) = [r(t,h) - coûts estimés] / [vol(t) × sqrt(h)]
```

Politique initiale :

- entrer Long si `score > long_threshold` ;
- entrer Short si `score < -short_threshold` ;
- fermer si `abs(score) < exit_threshold` ;
- conserver l'état pendant au moins `min_holding_period` jours ;
- interdire une nouvelle transition pendant `cooldown` jours ;
- sémantique : `target_position`.

Grille pilote volontairement petite :

- horizon : `5`, `10`, `20` ;
- volatilité : fenêtre `20` ou `60` jours ;
- seuil Long : `0.5`, `1.0` ;
- seuil Short : `1.0`, `1.5` ;
- seuil de sortie : `0.0`, `0.25` ;
- durée minimale : `5`, `10` jours ;
- coûts : `5`, `10` bps par unité de turnover.

Ne pas exécuter le produit cartésien complet au premier essai. Préenregistrer un
maximum de 12 configurations motivées pour limiter le multiple testing.

Créer d'abord une variante `long_flat`. Ajouter Short uniquement si cette baseline
est stable, car `long_flat` correspond mieux à l'objectif `excess_buy_hold`.

### M3 — `triple_barrier`

Pour chaque événement :

- barrière de profit ajustée à la volatilité ;
- barrière de perte ajustée à la volatilité ;
- barrière temporelle maximale ;
- résultat défini par la première barrière touchée ;
- filtrage événementiel optionnel `all` ou `cusum`.

Paramètres pilotes : horizon maximal `5/10/20`, profit `1.0/1.5 vol`, stop
`0.75/1.0 vol`, seuil CUSUM `0.5/1.0 vol`.

La méthode doit fournir une politique explicite entre événements. Elle ne peut pas
laisser des trous implicitement interprétés comme Hold ou Flat.

Référence : [triple-barrier et seuils dynamiques](https://mlfinpy.readthedocs.io/en/stable/Labelling.html).

### M4 — `meta_label`

Décomposer le problème :

1. une règle primaire propose le sens (`side`) ;
2. le modèle apprend seulement `Take` ou `Skip` selon le résultat net du trade ;
3. la probabilité `Take` contrôle l'entrée ou la taille.

Règles primaires candidates : momentum, breakout directionnel, ou
`volatility_position` gelé. La règle primaire ne doit pas être ajustée en utilisant
les résultats du meta-modèle.

Cette méthode cible particulièrement les faux positifs et la rotation. Référence :
[Meta-Labeling: Theory and Framework](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018).

### M5 — `regularized_oracle`

Conserver l'Oracle comme professeur et diagnostic uniquement. Étendre la DP avec :

- frais réels ;
- pénalité supplémentaire de turnover ;
- durée minimale de position ;
- cooldown ;
- nombre maximal de transitions par fenêtre de 20/60/252 jours ;
- état Flat autorisé après une position ;
- pénalité de risque optionnelle.

Le résultat doit être converti en **états persistants**, puis éventuellement en
actions. Rapporter séparément PnL brut, PnL net, turnover et durée des régimes.

Interdictions : pas de sélection de modèle avec l'Oracle, pas d'Oracle sur le test
final, pas de performance Oracle présentée comme résultat apprenable.

### M6 — `cross_sectional_rank`

À chaque date, classer le rendement futur ajusté de la volatilité :

- quantile supérieur : Long ;
- quantile inférieur : Short ;
- centre : Flat ;
- neutralité sectorielle optionnelle.

Commencer avec l'univers complet, car dix tickers seulement donnent des quantiles
grossiers. L'objectif et le benchmark sont `cross_sectional_alpha`; cette méthode
ne doit pas être classée directement contre le même buy & hold directionnel sans
explication.

### M7 — objectif financier direct, hors labeling

Prévoir ensuite une expérience distincte où le réseau produit une position continue
et optimise Sharpe/PnL avec pénalité de turnover. Cette méthode ne doit pas être
exposée comme `--label-method`, puisqu'elle supprime la cible de classification.

Référence : [Deep Momentum Networks](https://arxiv.org/abs/1904.04912).

## 5. Interface argparse commune

Créer `src/trading_system/pipelines/label_arguments.py` :

```python
def add_label_arguments(parser: argparse.ArgumentParser) -> None: ...
def label_config_from_args(args: argparse.Namespace) -> LabelConfig: ...
```

Arguments communs :

```text
--label-method
--label-config PATH
--label-objective
--label-horizon
--label-cost-bps
--label-semantics
```

Arguments spécialisés :

```text
--label-window
--label-buy-buffer
--label-sell-buffer
--label-vol-window
--label-long-threshold
--label-short-threshold
--label-exit-threshold
--label-min-hold
--label-cooldown
--label-profit-barrier
--label-stop-barrier
--label-max-holding
--label-event-filter {all,cusum}
--label-cusum-threshold
--label-primary-side
--label-top-quantile
--label-bottom-quantile
--label-turnover-penalty
--label-max-transitions
```

Règles :

- valeurs par défaut de la méthode < fichier JSON < arguments CLI explicites ;
- utiliser `argparse.SUPPRESS` pour distinguer une valeur explicitement passée
  d'une valeur par défaut ;
- rejeter les paramètres incompatibles avec la méthode choisie ;
- normaliser les noms CLI avec tirets vers des noms internes avec underscores ;
- conserver `--label-mode` comme alias déprécié pendant la migration ;
- afficher la configuration résolue et son hash avant tout entraînement ;
- persister la configuration résolue dans chaque manifest.

Tous ces points d'entrée doivent utiliser le même helper :

- `scripts/run_model_comparison.py` ;
- `scripts/run_walkforward.py` ;
- `scripts/run_gridsearch_walkforward.py` ;
- les scripts static single/multi ;
- le nouveau `scripts/run_label_benchmark.py`.

Exemples attendus :

```bash
python3 scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --preset multi_ticker_long_short \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --label-method volatility-position \
  --label-horizon 10 \
  --label-vol-window 20 \
  --label-long-threshold 1.0 \
  --label-short-threshold 1.5 \
  --label-exit-threshold 0.25 \
  --label-min-hold 5 \
  --label-cost-bps 5
```

```bash
python3 scripts/run_label_benchmark.py \
  --data data/processed/cac40_daily.parquet \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --label-method triple-barrier \
  --label-profit-barrier 1.5 \
  --label-stop-barrier 1.0 \
  --label-max-holding 10 \
  --label-event-filter cusum
```

Les configurations retenues doivent aussi être sauvegardées sous
`configs/labels/*.json`, afin d'éviter des commandes longues et de figer les
expériences :

```bash
python3 scripts/run_model_comparison.py \
  --label-config configs/labels/volatility_position_v1.json
```

## 6. Benchmark de labels avant modèles

Créer `scripts/run_label_benchmark.py`. Il ne doit entraîner aucun réseau.

Pour chaque configuration et chaque fold de développement :

1. construire les labels sans franchir les frontières temporelles ;
2. exécuter le backtest perfect-label canonique ;
3. comparer cash, buy & hold et benchmark adapté à l'objectif ;
4. calculer les diagnostics d'apprenabilité ;
5. agréger par ticker, secteur, période et configuration ;
6. conserver tous les candidats et échecs, pas seulement le meilleur.

Sortie :

```text
artifacts/labeling/<timestamp>/
├── candidates.csv
├── per_ticker.csv
├── per_fold.csv
├── transitions.csv
├── failures.csv
├── report.json
└── selected_config.json
```

Métriques économiques :

- PnL et rendement nets ;
- outperformance ;
- Sharpe, Sortino, max drawdown ;
- turnover, transactions, durée moyenne/médiane ;
- exposition Long/Flat/Short ;
- performance médiane et pire ticker/période.

Métriques d'apprenabilité :

- distribution des classes ;
- taux de transition ;
- longueur moyenne/médiane des régimes ;
- matrice de transition ;
- entropie des classes et des transitions ;
- stabilité des labels sous perturbation légère des paramètres ;
- macro-F1/balanced accuracy d'une régression logistique et d'un arbre peu profond ;
- écart à une baseline majorité et à des labels mélangés dans chaque split.

Score de présélection calculé uniquement sur les folds de validation :

```text
robust_score = median(outperformance)
             - 0.50 × IQR(outperformance)
             - penalty(turnover)
             - penalty(instability)
```

Le score exact et ses coefficients doivent être préenregistrés dans la
configuration du benchmark.

## 7. Critères de passage

Une politique ne passe à l'entraînement neuronal que si elle respecte :

### Gate économique

- PnL net perfect-label positif sur la médiane des folds ;
- outperformance médiane positive si l'objectif est `excess_buy_hold` ;
- aucun résultat dépendant d'un seul ticker ;
- drawdown et turnover sous limites préenregistrées ;
- robustesse à des coûts doublés.

### Gate d'apprenabilité

- aucune classe nécessaire complètement absente ;
- aucune classe dominante au-delà d'une limite préenregistrée ;
- durée médiane des régimes compatible avec l'horizon ;
- taux de transition compatible avec le budget de turnover ;
- labels stables sous une variation de ±10 % des seuils ;
- au moins une baseline simple dépasse la majorité et le contrôle mélangé sur
  plusieurs folds.

Les seuils numériques définitifs seront fixés après le rapport descriptif du
premier pilote, puis gelés avant la comparaison des candidats.

## 8. Plan de tests

### 8.1 Tests unitaires par labeler

- série constante : Flat/Hold et zéro turnover ;
- série strictement croissante : états Long attendus ;
- série strictement décroissante : états Short ou Flat selon le mode ;
- oscillation autour d'un seuil : hystérésis sans flip excessif ;
- durée minimale et cooldown respectés exactement ;
- seuils asymétriques appliqués du bon côté ;
- coûts intégrés au score ;
- barrière profit/perte : première barrière gagnante ;
- barrière verticale en absence de touch ;
- Oracle régularisé : contraintes de transition toujours satisfaites ;
- quantiles cross-sectionnels : nombres Long/Flat/Short attendus.

### 8.2 Propriétés communes

- déterminisme complet ;
- IDs, noms et positions alignés ;
- absence de NaN/infini parmi les cibles connues ;
- invariance des labels d'un ticker lorsqu'un autre ticker est ajouté ;
- aucun mélange de ticker lors des horizons ou rolling windows ;
- changement de prix futur modifie la cible, jamais les features passées ;
- aucune cible connue si son horizon dépasse le split ;
- `target_position` et `action` produisent le même backtest après conversion
  explicite lorsque les chemins sont équivalents.

### 8.3 Tests argparse

- chaque `--label-method` est parsé et normalisé ;
- paramètres invalides rejetés avec message utile ;
- paramètre d'une autre méthode rejeté ;
- fichier JSON chargé ;
- override CLI prioritaire ;
- alias `--label-mode` compatible et déprécié ;
- configuration/hashes identiques entre CLI équivalente et JSON ;
- arguments identiques en static, walk-forward et model comparison.

### 8.4 Tests anti-leakage

- test final jamais lu pendant le benchmark de labels ;
- derniers `horizon` targets de chaque split marqués inconnus ;
- aucun calcul de volatilité utilisant une ligne future ;
- sélection identique après corruption volontaire du test final ;
- walk-forward : chaque cible d'entraînement entièrement réalisée avant le début
  du chunk prédit ;
- Oracle limité au train et impossible comme méthode de comparaison/promote.

### 8.5 Tests d'intégration

- chaque méthode fonctionne sur un ticker et plusieurs tickers ;
- chaque méthode compatible fonctionne en long-only et long-short ;
- les cinq modèles reçoivent exactement les mêmes lignes et labels ;
- perfect-label notebook et CLI donnent les mêmes métriques ;
- artefacts rechargeables avec la même configuration et le même hash ;
- erreur d'une configuration conservée dans `failures.csv` sans masquer les autres ;
- smoke test CPU petit et rapide pour toutes les méthodes.

### 8.6 Tests de non-régression

- `breakout` et `forward_return` reproduisent les résultats historiques ;
- moteur de backtest canonique inchangé pour la sémantique `action` ;
- anciens scripts fonctionnent pendant la période de dépréciation ;
- comparaison de modèles reste équitable lorsqu'un nouveau label est sélectionné.

## 9. Ordre d'implémentation

1. Figer le benchmark actuel et ajouter les tests de non-régression.
2. Introduire `LabelConfig`, `LabelResult`, sémantiques et registry.
3. Migrer `breakout` et `forward_return` sans changement de résultat.
4. Centraliser les arguments et connecter `--label-method` / `--label-config`.
5. Créer `run_label_benchmark.py` et les rapports perfect-label/apprenabilité.
6. Implémenter `volatility_position`, d'abord `long_flat`, puis `long_short`.
7. Exécuter le pilote préenregistré sur plusieurs folds et tickers.
8. Geler un premier candidat, puis seulement lancer les baselines simples.
9. Implémenter l'Oracle régularisé comme diagnostic.
10. Implémenter triple-barrier, puis meta-labeling.
11. Implémenter le ranking cross-sectionnel avec benchmark adapté.
12. Comparer les architectures neuronales uniquement pour les labels ayant passé
    les deux gates.
13. Étudier séparément l'objectif financier direct.

## 10. Definition of Done

- Une seule API de labeling utilisée partout.
- Toutes les méthodes sélectionnables depuis `argparse` et JSON.
- Sémantique action/position impossible à confondre.
- Label benchmark sauvegardé et reproductible.
- Tests unitaires, intégration, CLI et anti-leakage verts.
- Sélection effectuée uniquement sur validation multi-fold/multi-ticker.
- Configuration de labels gelée avant tout nouveau test final.
- Aucun résultat Oracle présenté comme performance apprenable.
