# Visualisation des benchmarks et backtests

`trading_system.visualization` explore les résultats déjà exportés. Aucun
entraînement, replay, téléchargement de marché ou envoi d'ordre ne démarre depuis
le dashboard. Streamlit reste une dépendance optionnelle ; le catalogue et les
graphiques Plotly fonctionnent sans UI.

## Lancement

Depuis la racine du dépôt :

```bash
python -m pip install -e ".[visualization]"
python scripts/run_visualization.py
```

Le serveur écoute sur `http://127.0.0.1:8501`. Pour un dossier précis ou un autre
port :

```bash
python scripts/run_visualization.py \
  --artifacts-root artifacts/mt5 \
  --port 8502 --headless
```

Le dossier par défaut est `artifacts/`. Le champ latéral permet de changer de
dossier ; **Actualiser** invalide les caches. Le CLI propose aussi un inventaire
sans Streamlit :

```bash
python scripts/run_visualization.py --list --limit 20
```

## Sources reconnues

| Famille | Entrées | Détail disponible |
| --- | --- | --- |
| Backtest avancé | `manifest.json`, `config.json`, `metrics/core_metrics.json` | `backtest/equity_curve.parquet`, positions, trades et marché persisté ; timestamps indexés acceptés. |
| Benchmark | `report.json` avec `runs`/`rows`, ou `runs.csv` | Métriques individuelles ; historique si l'artifact référencé existe. |
| CV / post-open | `report.json` avec `folds`, `folds.json`, `results.json` | Candidat/fold/seed isolés, loss, positions quand exportées. |
| Graphes / exposition | `daily_paths.parquet`, ou `daily_path_artifacts` par partition | Courbes, rendement brut/net, coûts et exposition ; prédictions projetées quand disponibles. |
| MT5 | `equity-curves.parquet`, `equity_curve.parquet`, rapports et métadonnées | Courbes stratégie/buy-and-hold, seeds/fréquence/décodeur, positions. |
| Trading | `report.json` avec `comparisons`, ou résultat direct `equity.parquet` | Bras avec/sans règles, equity, positions, ordres exécutés et trades. |
| Gridsearch | CSV/JSON `gridsearch_walkforward_*` ou `gridsearch_*` | Essais, sélection et test final distingués ; anciens exports sans partition explicite signalés `unknown`. |

Les résultats incrémentaux restent signalés comme partiels. Des fichiers absents,
corrompus ou des identités mélangées produisent une observation explicite ; les
autres runs restent consultables. Les checkpoints et modèles ne sont pas parcourus
ni chargés. Les séries sont filtrées par leur identité complète avant affichage.

Un benchmark contenant seulement des métriques ne peut pas fournir sa courbe de
capital ou ses transactions. Le dashboard ne les invente pas : un export de série
datée doit être ajouté au producteur concerné pour activer ces panneaux.

## Vues et filtres

Deux onglets organisent la lecture :

- **Portefeuille** : capital stratégie et Buy & Hold, puis drawdown sur le même
  axe de dates ; KPI de la fenêtre affichée. Le sélecteur du run observé permet
  d'inspecter ses actifs : un actif sélectionné par défaut, un graphique prix et
  achats/ventes pour chaque actif choisi. Les options regroupent base 100 et
  exports CSV/HTML autonome utilisable hors ligne.
- **Comparer les runs** : huit runs maximum, barres horizontales des KPI
  enregistrés sur leur intervalle original, noms de métriques exacts et provenance.
  Les dates choisies dans Portefeuille ne modifient pas ces scores.

La barre latérale regroupe le dossier source et les filtres avancés : état,
modèle, candidat, partition, seed, fold, décodeur et univers. Le filtre univers
sélectionne les runs contenant un ticker ; il ne recalcule pas un portefeuille
réduit. Le choix des actifs limite les graphiques d'actifs, pas le capital du
portefeuille.

Le capital se charge sans les détails lourds. Les prix, positions, ordres et
trades se chargent ensuite pour les tickers choisis. Le chargement Parquet
projette seulement les colonnes utiles et filtre
candidat/fold/seed/partition/méthode/décodeur/ticker côté lecture. La réduction du
nombre de points Plotly conserve les extrémités et extrema par intervalle ; elle
ne change pas les statistiques ni les exports CSV. Les marqueurs d'événements
conservent leurs dates exactes.

Le thème reprend les couleurs de `web/frontend/styles.css` : bleu stratégie,
Buy & Hold gris pointillé, drawdown bleu translucide, ventes rouge sobre. Police
système, surfaces claires et hiérarchie visuelle suivent le style Apple. Les
préférences de réduction des animations et de transparence sont prises en compte.

## Prix et marqueurs de position

Les marqueurs indiquent leur source et leur type au survol :

- `executed_orders` : fills exportés par le replay, avec prix d'exécution et sens
  de la quantité échangée ; un achat peut clôturer un short, une vente en ouvrir un.
- `executed_trades` : entrées/sorties des trades exportés, utilisées lorsque les
  ordres ne sont pas disponibles.
- `observed_positions` : transitions de signe des positions observées ; prix de
  barre indicatif, sans reconstruction d'un fill.
- `position_signals` : transitions de signe des cibles du modèle, présentées
  comme **Signal achat/vente**, distinctes des exécutions enregistrées.

Une table d'ordres exportée pour l'actif reste prioritaire, même si ses lignes
ne fournissent aucun fill valide. Aucun événement n'est inventé à partir de la
première position observée. Les transitions sont déterminées sur le run complet **avant** le
filtre de dates ; une position déjà ouverte au début de la fenêtre ne devient
donc pas une nouvelle entrée. Les prévisions indisponibles interrompent la
continuité des signaux. Un retournement long/short conserve les événements de
clôture et d'ouverture.

Les prix proviennent du marché persisté ou d'une source locale référencée par le
run, avec contrôle de provenance lorsque son empreinte est fournie. La lecture
reste limitée aux sessions et actifs évalués. Les exports de positions/prédictions
peuvent fournir les prix lorsqu'ils les contiennent. La base de prix brute ou
ajustée reste explicite ; les marqueurs de signal utilisent la cotation de leur
barre exacte. En l'absence de prix, le panneau le signale ; seuls les prix
d'exécution déjà exportés peuvent encore apparaître.

## Conventions de calcul

**Métriques enregistrées** : valeurs et noms du moteur d'origine, sur son intervalle
complet. `sharpe`, `net_sharpe`, `sharpe_ratio` et `regularized_sharpe` restent
distincts. Les barres convertissent en pourcentage uniquement une liste explicite
de métriques de rendement/drawdown stockées en fractions ; les ratios et comptes
gardent leurs unités. Les résultats partiels peuvent montrer leurs observations
déjà disponibles, avec leur état visible ; les échecs ne deviennent pas des scores.

**Buy & Hold** : une courbe ou des rendements de référence déjà exportés restent
prioritaires. Pour les anciens backtests déclarant
`legacy_close_buy_hold_no_fees`, une référence peut être calculée avec le capital
initial enregistré et les `close` bruts persistés, alignés exactement sur les dates
de l'equity, **sans frais**. Cette convention reste distincte d'un benchmark net de
coûts. Des prix ou une référence absents ne sont pas inventés.

**Fenêtre affichée** : premier capital réellement observé comme référence. Le
rendement affiché vaut `dernier / premier - 1` ; le drawdown part de ce premier
point ; les rendements mensuels composent les transitions internes à cette
fenêtre. Le premier rendement n'est donc pas reconstruit depuis un capital initial
supposé. Les coûts ou rendements précédant ce point appartiennent aux métriques
originales. L'export des séries conserve les champs `source_net_return` et
`source_drawdown` lorsqu'ils existaient, à côté des champs de la fenêtre.

Pour l'API Plotly directe, `monthly_returns_figure` utilise les `net_return`
exportés lorsqu'ils existent ; les fonctions `slice_run` et `window_statistics`
explicitent la convention de la fenêtre utilisée par l'UI. Les détails de trades
se filtrent par date de sortie ; l'historique d'entraînement reste complet.

**Comparabilité** : sélection multi-runs contrôlée sur famille, dataset, univers,
période, partition, prix, exécution, timing, délai, coûts, annualisation et convention de capital.
Les informations manquantes sont signalées. Les folds, seeds et partitions ne
sont jamais concaténés pour fabriquer une performance globale. Un test final
absent reste absent ; l'UI n'exécute pas son évaluation.

## API

```python
from trading_system.visualization import discover_runs, load_run, comparison_issues
from trading_system.visualization.charts import capital_drawdown_figure, position_figure
from trading_system.visualization.positions import asset_names

catalog = discover_runs("artifacts/mt5")
record = next(record for record in catalog.records if "equity" in record.tables)
data = load_run(record, include_details=False)
figure = capital_drawdown_figure([data], normalize=True)
figure.write_html("capital-drawdown.html", include_plotlyjs=True)

# Chargement ciblé ; événements déterminés avant la fenêtre du graphique.
assets = load_run(record, include_details=True, tickers=["SAF.PA"])
if "SAF.PA" in asset_names(assets):
    positions = position_figure(assets, "SAF.PA", start="2024-01-01", end="2024-12-31")
```

Modules : `schemas` (contrats), `catalog` (découverte/provenance), `adapters`
(lecture ciblée), `analytics` (fenêtre), `positions` (prix/événements sourcés),
`theme` (palette web et styles), `charts` (Plotly), `app` (Streamlit), `cli`
(lancement). Les vues principales sont `capital_drawdown_figure`,
`kpi_bar_figure(frame, metric=...)` et `position_figure(data, ticker, start=..., end=...)`.
Les fonctions de courbes, distributions, exposition et historique restent
disponibles sans Streamlit. Le package ne dépend pas des modèles ni des moteurs
d'entraînement.

## Validation

```bash
python -m pip install -e ".[dev,visualization]"
python -m pytest -q tests/test_visualization_*.py
```

Tests données/identités/partiels, isolation folds, conventions des courbes,
exports HTML, filtres dates/tickers, prix bruts/ajustés, sources des événements,
shorts et chronologie des marqueurs, puis smoke Streamlit. Les tests UI sont
ignorés si l'extra `visualization` n'est pas installé.
