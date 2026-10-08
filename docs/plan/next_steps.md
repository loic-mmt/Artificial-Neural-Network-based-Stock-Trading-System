# Prochaines étapes : fusion multimodale modulaire

Invariant : GRU, Transformer de marché, GNN et sentiment sont des branches indépendantes. Le GRU lit les séquences par ticker ; le Transformer lit uniquement une séquence globale macro/micro/volatilité ; le GNN reçoit ses propres features de nœuds et un graphe causal, jamais les embeddings du GRU. La fusion peut fonctionner avec une seule branche prédictive active.

1. Fixer contrat multimodal : échantillons alignés par date/ticker ; séquences par ticker du GRU, séquence globale du Transformer, features de nœuds GNN, graphe causal, sentiment et masques de disponibilité. Chaque branche produit logits Sell/Hold/Buy + représentation. GNN ne consomme jamais sortie GRU.
2. Construire branches isolées : GRU actuel ; GNN seul sur features boursières/relationnelles ; Transformer dédié à l'état de marché. Sentiment : scores datés importés de la librairie FinBERT externe, sans chargement de FinBERT pendant l'entraînement ; l'extra optionnel sert uniquement à la préparation séparée des données. Absence de news ≠ sentiment neutre.
3. Ajouter Market-Guided Gating, activable indépendamment : aucun gate / gate statique / gate guidé par contexte marché disponible au moment de prédiction.
4. Ajouter fusion modulaire : branche seule → moyenne fixe → poids appris → confidence-aware fusion. Calibration des branches sur validation uniquement ; abstention vers Hold testée séparément de fusion. Pondérations renormalisées sur les seules branches disponibles ; disponibilité de news distincte d'un score neutre.
5. Benchmarks par ablations, pas produit cartésien géant : mêmes dates/tickers, labels, coûts, folds, seeds, budget de recherche. Comparer aussi GRU avec mêmes features marché pour isoler apport du graphe ; tester chaque branche seule et chaque retrait de branche. Rapporter PnL net, Sharpe, F1, calibration, couverture, temps/VRAM.
CLI à créer : sélections indépendantes pour branches, graph, market-gate, fusion, calibration, sentiment et politique de trading. Le benchmark d'agrégation temporelle du GRU est terminé ; il reste distinct du chantier multimodal.

## Contrat du point 1

`build_multimodal_dataset` reçoit des frames déjà préparées, un ordre explicite de tickers et, facultativement, historique antérieur, état de marché, sentiment daté et graphes préconstruits. Il ne calcule ni features, ni arêtes, ni statistiques de normalisation. Un lot à la demande expose `temporal[B,N,T,F]` pour le GRU, `market_sequence[B,Tm,Fm]` pour le Transformer, `node[B,N,F]`, l'ancien contexte `market[B,F]`, `sentiment[B,N,F]`, labels et positions des lignes d'origine. Les axes `B` et `N` sont respectivement les dates de séance et les places fixes de tickers ; les places absentes valent zéro **avec masque faux**.

Masques séparés : actif présent, fenêtre GRU complète, features de nœud valides, ancien contexte marché disponible, séquence de marché complète, sentiment disponible, label connu et graphe utilisable. Label inconnu = `-1`, jamais Hold. Sentiment neutre avec source couvrante = score zéro et masque vrai ; source absente = masque faux, même si la valeur de remplissage est zéro. La source de sentiment fournit `source_available` et `available_at` ; aucune librairie FinBERT n'est importée ici.

Les fenêtres et features de graphe fondées sur prix peuvent inclure la clôture de J, pour exécution ultérieure. Publications externes (`available_at`, `fund_available_at`) doivent précéder strictement J minuit UTC. Chaque graphe sparse fourni porte sa date et `source_end` ; aucune observation de J+1 n'est admise. Train/validation/test doivent être disjoints par date globale, même si leurs tickers diffèrent. Le runner 3D et sa CLI restent inchangés ; raccordement prévu avec les premières branches multimodales.

Pour le Transformer, `market_frame` fournit une ligne par séance et un ordre temporel indépendant de l'ordre des tickers. Les colonnes macro/micro publiées vont dans `market_publication_columns` et nécessitent `available_at < J minuit UTC`. Les séries issues de la clôture de J, par exemple VIX, vont dans `market_close_columns` et nécessitent `source_end < J+1 minuit UTC` : leur usage suppose une exécution après clôture. Le choix des colonnes rend macro, micro et VIX activables séparément ; aucune colonne de `temporal_columns` ne peut être routée au Transformer. Les horodatages sont aujourd'hui communs à chaque groupe de colonnes par ligne : si les sources ont des heures différentes, les préparer en amont de façon conservatrice. Fenêtre de marché incomplète = masque faux, jamais données imputées comme observées.

La librairie FinBERT externe est maintenant raccordée par un export Parquet et
un manifeste vérifiés, sans corpus historique sur le Mac. Voir
`docs/src/news-sentiment-integration.md`. `source_available` vaut vrai seulement
pour une fenêtre explicitement `covered`. Si cette fenêtre contient zéro news,
`available_at` peut être manquant : le journal de couverture validé par la
librairie, pas un faux horodatage d'article, justifie alors le masque vrai.
Les statistiques manquantes ont des indicateurs `_present` et ne sont pas
confondues avec un score neutre observé. L'ancien runner 3D et sa CLI ne
consomment pas encore cet export.

## Point 2 : branches isolées

Les quatre branches vivent dans `src/trading_system/models/multimodal_branches.py`
et retournent le même `BranchOutput` `[B,N,3]`, `[B,N,d]`, `[B,N]` :

- `GRUBranch` réutilise l'encodeur, le pooling et les têtes configurables du GRU
  actuel, en ne traitant que les fenêtres temporelles complètes.
- `MarketTransformerBranch` lit exclusivement `market_sequence[B,Tm,Fm]`,
  sans séquence boursière du GRU. Sa représentation est globale à la date et
  répliquée par ticker ; ses logits seuls sont donc identiques pour tous les
  actifs présents. `TransformerBranch` reste un alias d'import vers cette
  nouvelle branche. Le `TransformerClassifier` du runner 3D historique reste
  inchangé et conserve son ancien rôle de comparateur sur données boursières.
- `GNNBranch` lit seulement `node[B,N,F]` et les `GraphSnapshot` datés. Son mode
  `identity` est le contrôle node-MLP ; `provided` utilise une convolution GCN
  sparse avec boucles propres et normalisation du degré. Poids négatifs interdits
  dans cette variante ; les graphes signés demanderont des canaux séparés.
  Une date sans graphe en mode `provided` reste indisponible, sans repli caché.
- `SentimentBranch` lit les agrégats numériques FinBERT déjà exportés, jamais le
  texte ni FinBERT. Son masque suit la couverture auditée : score neutre observé,
  zéro news couvert et source inconnue restent trois états distincts.

`masked_branch_cross_entropy` permet un pas d'entraînement supervisé par branche
sur les seuls labels connus et places disponibles. Aucun GNN ne reçoit une sortie
GRU. `BranchSelection` et `BranchRouter` instancient uniquement les branches
choisies, sans fusion implicite.

## Contrôles modulaires disponibles

`MultimodalOptions` sépare les encodeurs actifs (`selection.enabled`) des
branches qui votent réellement (`prediction_branches`). Le Transformer peut
ainsi produire uniquement un état de marché destiné au gate, sans imposer ses
logits au portefeuille. `gate_gru` et `gate_gnn` sont indépendants : `none`,
`static` ou `market`. Le gate multiplie les features par `F × softmax` ; son
initialisation uniforme est une identité, et une date sans état de marché garde
ses features inchangées. Avec une seule feature, le gate est nécessairement une
identité : tester au moins deux features pour cette ablation.

`fusion` vaut `single`, `mean`, `static` (poids appris) ou `confidence`. Les
poids sont renormalisés seulement sur les branches disponibles. Le mode
`confidence` utilise des températures positives fournies explicitement,
ajustées **sur validation uniquement** avant le test ; aucun ajustement ni
politique d'abstention ne sont encore automatisés. Chaque branche, gate et
fusion peut être désactivé séparément. Cette implémentation reprend les idées
de séparation de [FusionLSTM-CNF](../papers/FusionLSTM-CNF/FusionLSTM-CNF.md)
et de gate conditionné par le marché de [MASTER](../papers/MASTER/MASTER.md),
sans reprendre leurs revendications de performance.

Exemple d'ablation côté API, une fois `sample` construit avec les groupes de
colonnes souhaités :

```python
from trading_system.models.multimodal_router import BranchSelection
from trading_system.models.multimodal_system import MultimodalOptions, MultimodalSystem

options = MultimodalOptions(
    selection=BranchSelection(
        enabled=("gru", "market_transformer", "gnn"),
        gnn_graph_mode="provided",
    ),
    prediction_branches=("gru", "gnn"),  # Transformer = contexte, sans vote
    gate_gru="market",                    # gate_gnn="none" par défaut
    fusion="mean",
)
model = MultimodalSystem(sample, options)
prediction = model(sample)
```

Omettre `market_transformer` et mettre `gate_gru="none"` permet le contrôle GRU
seul ; `gnn_graph_mode="identity"` donne le contrôle sans arêtes. La sélection
des features du Transformer se fait séparément dans `build_multimodal_dataset`
via `market_publication_columns` et `market_close_columns`.

Les arêtes ne sont **pas** estimées par ces modèles. Le point 7 ajoute des
constructeurs causaux de `GraphSnapshot` et un runner CV dédié à la comparaison
GRU seul / GNN seul. Le runner 3D historique reste inchangé. Le raccordement
général de la fusion et l'ajustement des températures restent à faire. Un
runner distinct teste désormais les contrôles de marché à base de séries de
clôture, comme décrit plus bas.

## Point 7 : contrôles de graphe GNN seul

`data/causal_graphs.py` construit quatre contrôles : `identity` sans arêtes,
secteur à la date de prédiction, Pearson statique, Pearson glissant causal.
Le Pearson statique est estimé sur un préfixe de `graph_lookback` rendements du
fold d'entraînement, puis figé. Toutes les variantes, y compris GRU seul,
commencent après ce préfixe pour garder les mêmes dates éligibles. Le graphe
glissant utilise les derniers rendements connus jusqu'à la clôture de J.
Le `source_end` enregistré vaut la borne supérieure de la séance utilisée,
avant J+1 minuit UTC. Les graphes positifs et les valeurs absolues sont
possibles ; les corrélations signées ne sont jamais passées au GCN standard.

`scripts/run_gnn_graph_comparison.py` compare cinq candidats (`gru`, `identity`,
`sector`, `train_pearson`, `rolling_pearson`) sur les mêmes dates, tickers,
features sélectionnées sur train, seeds, folds purgés et coûts. GRU lit une
fenêtre par ticker, GNN seulement les features de nœud du jour. La loss Sharpe
est le défaut ; la loss combinée peut être choisie explicitement. La sélection
primaire reste le Sharpe régularisé, avec rendement net et drawdown rapportés.
Chaque fold sauvegarde les intervalles et arêtes des graphes, l'état des modèles,
les métriques internes/externes et les diagnostics de densité. `--resume` vérifie
les métadonnées avant de passer les entraînements déjà terminés. Le holdout
final reste scellé.

Limites : `sector` dépend de la qualité historique des classifications source ;
le fichier CAC40 actuel reste biaisé par survivance. Le GRU de ce runner utilise
un scaler par ligne appris sur train, commun avec le GNN ; ses scores ne sont
pas strictement interchangeables avec ceux des anciens benchmarks GRU, qui
utilisent un scaler pondéré par fenêtre. Les comparaisons du point 7 sont
appariées **à l'intérieur du nouveau run**. Le contrôle G5 à deux canaux signés,
la permutation aléatoire et la fusion restent des expériences distinctes.

## Points 8 et 10 : contexte de marché, sans fusion

`scripts/run_market_gru_comparison.py` raccorde un benchmark CV apparié de six
candidats : GRU seul, GRU avec contexte de marché concaténé, gate statique,
gate conditionné par un encodeur simple (moyenne temporelle + projection),
gate conditionné par un Transformer de marché indépendant, puis Transformer
de marché prédictif seul. Chaque candidat a **une seule branche votante**.
Ce test ne relance pas la fusion GNN/GRU du point 9. Le Transformer prédictif
produit les mêmes logits pour tous les actifs d'une date : c'est un contrôle
de direction globale, pas un modèle de sélection des titres.

Le premier run n'utilise que `market_close` et `vix_close` du Parquet nettoyé.
Il dérive leurs variations journalières et le niveau VIX. Les valeurs doivent
être identiques entre tickers sur chaque date, et toutes les fenêtres de marché
doivent être complètes. Le `source_end` à la fin de séance est une **borne
synthétique conservatrice**, pas une heure de publication du fournisseur.
Le protocole suppose une décision après la dernière clôture utilisée le jour J,
notamment celle du VIX américain, puis `execution_delay >= 1`. Une décision
à la clôture parisienne J ne peut pas utiliser le VIX de clôture du même jour.
Les taux, données macro et autres publications ne sont **pas** admis sans
historique de disponibilité point-in-time auditable. La standardisation des
features de marché est ajustée uniquement sur le train de chaque fold. Même
dates, tickers, labels, coûts et seeds pour les six contrôles ; holdout final
scellé et `--resume` vérifié par empreinte.

Commande pour le premier benchmark :

```bash
PYTHONPATH=src python scripts/run_market_gru_comparison.py \
  --data data/processed/cac40_daily_clean.parquet \
  --model-parameter-sets configs/benchmark/gru_market_context.json \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --seeds 1,7,19 --losses sharpe --context-len 60 \
  --feature-set expanded --no-external-features \
  --label-method triple-barrier --label-volatility-estimator atr \
  --label-profit-barrier 0.75 --label-stop-barrier 0.75 \
  --label-event-filter cusum --label-cusum-threshold 0.5 \
  --label-between-events hold --position-mode long_short \
  --cv-folds 3 --cv-gap-bars 5 --cv-score regularized_sharpe \
  --overfitting-control --overfitting-max-features 32 \
  --overfitting-max-feature-correlation 0.95 \
  --device auto \
  --output-dir artifacts/comparisons/gru-optim/08-10-market-context-60-atr \
  --fail-fast
```

Le benchmark comporte 6 candidats × 3 seeds × 3 folds, soit 54 entraînements.
Ajouter `--resume` avec la même commande et le même dossier pour reprendre.
Comparer principalement `market_gate_simple` et `market_gate_transformer` au
GRU avec concaténation et au gate statique. Une hausse de rendement avec un
drawdown excessif ou un simple effet d'exposition ne suffit pas.

## Extension US relationnelle et état de marché

Le benchmark US a été exécuté. Son protocole et ses résultats sont dans
[`us-relational-market-benchmark.md`](../benchmarks/us-relational-market-benchmark.md). Il
utilise 143 actions au calendrier complet depuis 2005 et 14 ETF de contexte non
tradables. Les nouveaux graphes `rolling_topk` et `rolling_residual_topk`
évitent les nœuds isolés du seuil Pearson fixe. La variante résiduelle retire
causalement SPY et le facteur sectoriel avant de sélectionner cinq voisins.

Le suffixe de candidat `_market` active un Transformer compact qui lit seulement
la séquence globale ETF, VIX et statistiques cross-sectionnelles. Son état
module les features du GRU ou du GNN avec le gate de marché. Ses logits ne sont
pas fusionnés et il ne vote pas. Chaque axe reste désactivable indépendamment
via `--graph-candidates`, `--market-close-columns`, `--graph-neighbors` et
`--graph-rebalance-bars`. Le sentiment reste hors protocole.

Les données structurées London Strategic Edge et les corpus de sentiment à
récupérer sont fixés dans [l'inventaire de collecte](data-acquisition-lse-sentiment.md).

## Pilote news sentiment, avant les prochains ajouts de données

Le [plan sentiment](news-sentiment-pilot.md) prépare une collecte Alpha Vantage
bornée à 25 tentatives par jour et sur 24 heures glissantes, avec compteur partagé.
Le scoring FinBERT reste séparé de l'entraînement. Les exports entreprise et
macro suivent deux routes distinctes ; aucun contexte global n'est transformé
en ticker tradable.

Le nouveau runner de comparaison propose GRU seul, activité news sans polarité,
features news dans le GRU, sentiment seul, puis moyenne masquée GRU/sentiment.
Le train apprend ses scalers sur les seules observations couvertes ; l'absence
de sentiment ne rend pas le GRU indisponible. Les runners existants ne changent pas.

Le pilote RSS a validé collecte/scoring/export ; il n'apporte pas de news aux
dates anciennes. Le benchmark [FNSPID](../benchmarks/news-sentiment-fnspid.md)
est désormais terminé : 81 entraînements sur 53 actions, cinq permutations et
contrôle neutralisé. Aucun apport robuste de polarité FinBERT n'est démontré.
FNSPID reste exploratoire : disponibilité historique supposée, couverture et
versions à publication non prouvées. Le sentiment reste désactivé par défaut,
sans conclure que toute source future de news sera inutile.

Le raccordement du panel macro au Transformer, la calibration et la fusion
apprise restent des étapes suivantes, à benchmarker séparément.
Ce document distingue historique et point-in-time, les entrées de chaque branche,
les sources complémentaires et les contrôles nécessaires avant ingestion.
Aucune nouvelle source n'est activée automatiquement par cet inventaire.

Les décisions suivantes viennent du [contrôle d'exposition](../benchmarks/us-exposure-comparison.md).
Le [benchmark d'interaction features/gate](../benchmarks/us-feature-gate-interaction.md)
est terminé au 8 octobre 2026 : 108/108 tâches vérifiées, avec `identity_market`
aux deux caps. Les résultats fixent une configuration de travail cap 32, gate
désactivé, identité obligatoire et GNN résiduel comme challenger. Identité32
avec marché a le meilleur score moyen appris, mais son gain est fragile et
ne justifie pas une promotion. Aucun entraînement de cette grille ne manque.

## Diagnostic de l'apprentissage : périmètre

Il faut distinguer une information réellement apprise d'un changement de
direction ou de taille des positions. Cela ne signifie ni rechercher un
nouveau label ni calculer l'importance de chaque feature.

1. Examiner pertes TRAIN/validation, métrique financière de validation,
   époque retenue et nombre d'époques exécutées ; comparer stabilité entre seeds.
2. Examiner distributions des probabilités et positions, saturation long/short,
   exposition brute/nette, turnover et coûts ; décomposer les périodes qui gagnent
   ou perdent, sans créer de blacklist rétrospective.
3. Vérifier si les écarts viennent d'un signal différent, de l'early stopping,
   ou d'un simple changement d'exposition. Les contrôles de gross déjà produits
   ne neutralisent pas beta, risque ou direction.
4. Sur FNSPID, comprendre les deux runs seed 7 expliquant 90,56 % du gain du
   neutralisé : checkpoints époque 1 contre 57/55 pour l'original. Une époque 1
   retenue ne prouve pas un bug ni un entraînement absent.

Les exports et checkpoints permettent une première analyse sans nouvel
entraînement. Les courbes et normes de gradients non sauvegardées ne peuvent
pas être reconstruites à partir du seul checkpoint final : compléter la
journalisation pour les prochains runs si nécessaire. Tout nouveau réglage
d'early stopping se choisit sur validation interne, jamais sur OUTER ou holdout.

La préparation du diagnostic est désormais disponible, sans nouvel entraînement
ni modification du protocole. [`scripts/diagnose_learning.py`](../../scripts/diagnose_learning.py)
vérifie les exports multimodaux signés et produit un rapport séparé sur les
checkpoints retenus, probabilités, positions, exposition exécutée, coûts et
contributions ticker/période. INNER est le défaut ; OUTER reste une lecture
descriptive explicite et le holdout final est interdit. Les labels inconnus et
signaux indisponibles ne sont pas remplacés par des observations Hold ou zéro.

Pour les prochains runs graphe/news et leur orchestration US,
`--learning-diagnostics` ajoute une trace optionnelle des pertes, positions et
normes de gradients par époque, sans changer l'optimizer ni la sélection du
checkpoint. Les courbes absentes des anciens runs restent absentes. La mesure
TRAIN pré-update avec dropout n'est pas directement comparable à la validation
post-update en mode évaluation ; un probe dans les mêmes conditions reste à
ajouter si ce diagnostic devient nécessaire. Il faut utiliser un nouveau dossier
de run pour activer cette trace, sans tenter d'enrichir un ancien run signé via
`--resume`.

Les commandes et limites sont décrites dans
[la documentation du diagnostic](../src/learning-diagnostics.md). Les artefacts
FNSPID complets doivent être rapatriés du PC avant leur analyse offline sur le
Mac. Cette étape ne lance ni benchmark de nouveaux réglages, ni attribution
feature par feature, ni collecte LSE.

### Résultats du diagnostic instrumenté

Le [diagnostic GRU32 instrumenté](../benchmarks/gru-learning-diagnostics.md)
est terminé : neuf entraînements, 547 époques, exports INNER/OUTER vérifiés,
holdout fermé. L'optimisation fonctionne, mais les solutions de direction et
d'amplitude restent très dépendantes des seeds. Les shorts ne sont pas interdits ;
des épisodes durables mal orientés existent, notamment AAPL fold 2 seed 1.
Un épisode conservant son signe peut comporter de nombreux resizings.

Avant un nouveau chantier d'architecture, il faut préciser le contrat entre
labels, loss et politique d'exécution. Le run utilise uniquement une loss
financière : triple-barrier 10 ne force ni imitation Sell/Hold/Buy, ni TP/SL,
ni liquidation après dix séances. Objectif hybride supervisé/financier,
sélectivité des événements et état de position causal sont des pistes distinctes,
pas des implémentations acquises. Les sorties/réentrées doivent être testées
ensemble, en tenant compte des [ablations déjà faites](../benchmarks/trading-exit-ablation.md).
Les comparaisons Sharpe/combiné et d'exposition existent déjà ; aucune grille
générique supplémentaire n'est nécessaire pour redécouvrir leurs compromis.

## Importance des features et données LSE

Les ablations de familles et le test cap32/64 existent ; une attribution
financière systématique feature par feature, notamment par régime, reste à faire.
Le runner multimodal classe par variance TRAIN et filtre les corrélations.
Ce classement et les poids d'un gate ne sont pas des pourcentages de score expliqué.
Le module `backtest/lib.py` calcule déjà des importances par permutation sur
l'AUC d'un arbre auxiliaire de scoring des trades. Ce n'est pas une attribution
du PnL/Sharpe aux entrées du GRU ou du GNN, ni une validation PIT de ces entrées.

Il est pertinent de réserver l'étude complète à l'intégration LSE, après audit
de `available_at`, couverture et révisions. Préparer une interface modulaire
maintenant ne nécessite ni collecte immédiate ni grosse grille d'attribution.

- Commencer par les familles pour mesurer leur apport net et leurs interactions.
- Mesurer ensuite la baisse de PnL/Sharpe après permutation temporelle par blocs
  sur un modèle figé, avec groupes de features corrélées ; un mélange ligne par
  ligne casserait l'autocorrélation et les relations entre actions.
- Confirmer les retraits importants par réentraînement apparié : dépendance du
  modèle à une feature et utilité après réapprentissage ne sont pas équivalentes.
- Rapporter stabilité par fold, seed, secteur et régime défini avec données connues
  au moment de décision. Choisir les features sur validation interne uniquement.

Il n'existe pas de décomposition naturelle additive « featureX explique 12 % du
Sharpe ». Utiliser des deltas de score et des incertitudes, pas un classement
universel. Ajouter des données ou changer le modèle peut modifier les interactions :
l'importance devra alors être recalculée, même si l'outil d'analyse est réutilisable.
