# Diagnostic des labels sans entraînement

`scripts/analyze_labels.py` compare les cibles, sans charger de modèle,
télécharger de données ou lancer d'entraînement. Les résultats sont enregistrés,
pas seulement affichés dans le terminal.

## Lancement

Depuis la racine du dépôt, comparaison des cinq méthodes sur l'univers US local :

```bash
.venv/bin/python scripts/analyze_labels.py \
  --data data/data_pc/processed/mt5_stocks_us_daily_clean.parquet \
  --end 2023-06-21 \
  --output-dir artifacts/diagnostics/labels-us-development
```

Cette borne conserve le holdout final fermé. Les prix après `--end` sont exclus
avant de construire les labels ; un événement ne peut pas chercher sa cible
au-delà de cette borne. `--start` conserve l'historique précédent pour le warmup
et le décodage, puis limite les statistiques à la période demandée.

Pour un ticker et deux méthodes :

```bash
.venv/bin/python scripts/analyze_labels.py \
  --data data/data_pc/processed/mt5_stocks_us_daily_clean.parquet \
  --tickers AAPL --start 2020-01-01 --end 2023-06-21 \
  --methods triple-barrier intraday-return \
  --output-dir artifacts/diagnostics/labels-aapl
```

Les méthodes et leurs paramètres sont enregistrés dans `summary.json`. Les
défauts sont breakout fenêtre 20, forward-return horizon 10 et seuils 0,2 %,
volatility-position horizon 10 en long/short avec maintien minimum 5 séances,
triple-barrier ATR 20 / horizon 10 / barrières 0,75 / CUSUM 0,5 / Hold,
et intraday-return sans seuil. `--help` expose les paramètres modifiables.

Les dossiers de résultats non vides ne sont pas écrasés.

## Objets mesurés

Il faut distinguer trois durées :

- **Couleur / séquence de labels** : nombre de séances consécutives portant la
  même classe. Une couleur Buy d'une séance ne constitue pas un trade d'une séance.
- **Position décodée** : pour les labels d'action, Buy passe Long, Sell passe
  Short et Hold conserve le sens courant. Pour les positions cibles,
  Short/Flat/Long indiquent directement l'état. Les épisodes utilisent les
  clôtures disponibles, sans délai, uniquement pour diagnostiquer les labels.
- **Événement natif** : triple-barrier utilise `label_end_date` et son rendement
  effectif ; forward-return et volatility-position utilisent leur horizon.
  Breakout n'a pas d'événement futur natif. Intraday utilise open et close du
  même jour et ferme toujours à la clôture, même si le lendemain est encore Long.

Les dates inconnues interrompent les séquences et positions. Les épisodes
tronqués par les limites ou une interruption sont signalés comme censurés.
Les durées min/moyenne/médiane/p90/max et rendements min/moyenne/médiane/p90/max
sont disponibles dans les CSV/JSON, y compris par ticker et côté Long/Short.

## Rendements et turnover

Le rendement brut d'un épisode vaut
`side * (prix_sortie / prix_entrée - 1)`, avec `side` égal à +1 ou -1.
Il correspond à un notionnel initial fixe, pas à un rééquilibrage quotidien.

`--cost-bps 5` signifie ici **5 bps par sens** : entrée + sortie coûtent 10 bps.
Le coût du diagnostic est indépendant du coût utilisé dans les barrières du
labeler, enregistré dans sa configuration. Le rendement net retire les coûts
d'entrée et de sortie.

Le turnover est mesuré en unités de notionnel : une entrée vaut 1, une sortie
1 et un retournement 2. La liquidation aux limites est incluse. Intraday entre
et sort chaque séance active, donc turnover 2 par séance active même si le sens
ne change pas. Le turnover moyen est rapporté aux séances/intervalles observés,
en conservant les inconnus comme catégorie distincte.

Les événements futurs se chevauchent souvent. Leurs rendements ne sont jamais
composés en performance de portefeuille. Aucun chiffre n'est une performance
réalisable : ces labels contiennent volontairement de l'information future.

## Bruit, transitions et opportunités

Les seuils par défaut sont 0,2 %, 0,5 %, 1 % et 1,5 %. Les fractions de petits
mouvements mesurent leur amplitude économique, pas leur apprenabilité.

Les sondes à horizons close-to-close 1, 5 et 10 utilisent uniquement les lignes
connues par toutes les méthodes sélectionnées. Elles mesurent le bon/mauvais
sens, les petits mouvements et les labels neutres face à un mouvement dépassant
le seuil. Hold est neutre dans cette sonde directionnelle, **pas dans le décodage
des positions**. Une opportunité neutre n'est donc pas nécessairement un trade
raté par une stratégie réelle.

Les sondes close-to-close ne mesurent pas la tâche open-to-close native de la
méthode intraday. Les deux contrats restent distincts. L'entropie des classes
et les changements fréquents ne suffisent pas à prouver que les labels sont
aléatoires ; il faudrait ensuite étudier leur lien avec les features connues
au moment de la décision.

## Exports

- `report.md`, `summary.json`, `summary.csv` : comparaison globale.
- `by_ticker.csv` : métriques détaillées par actif/méthode.
- `label_runs.parquet` : séquences de couleurs et censure.
- `position_trades.parquet` : épisodes décodés, durées et rendements.
- `native_events.parquet` : événements natifs, rendements et chevauchements.
- `transitions.csv` : matrice des transitions entre labels connus adjacents.
- `horizon_probes.csv`, `horizon_summary.csv` : sondes par ticker et agrégées.

## Nouveau label open-to-close

`LabelConfig.intraday_return()` et `--label-method intraday-return` utilisent
les prix **open et close bruts du même jour**. Close supérieur à open donne
Long, inférieur donne Short, égal donne Flat. Les prix invalides sont inconnus,
pas Flat. `adj_close` n'est jamais comparé à un open brut.

La cible n'est connue qu'à la clôture. Pour un futur entraînement décidant après
l'open, les features de clôture/high/low/volume du jour ne doivent pas entrer
dans les inputs. Le runner et son calendrier d'exécution n'ont pas été modifiés
par cet ajout.

Le visualiseur accepte les cinq méthodes via `--label-method` : `breakout`,
`forward-return`, `volatility-position`, `triple-barrier`, `intraday-return`.
Les noms avec underscores sont aussi acceptés. Les paramètres par défaut sont
ceux de l'analyse ci-dessus ; triple-barrier reste le choix par défaut.

Exemple avec le label intraday :

```bash
.venv/bin/python scripts/plot_ticker_labels.py \
  --data data/data_pc/processed/mt5_stocks_us_daily_clean.parquet \
  --ticker AAPL --start 2022-01-01 --end 2022-12-31 \
  --label-method intraday-return
```

La courbe de prix reste illustrative. Sa couleur représente le label à J,
pas une position overnight ni une exécution réelle.

Pour comparer une autre méthode, remplacer `--label-method intraday-return`,
par exemple par `--label-method volatility-position --label-horizon 10`.
`--help` liste les réglages ; les arguments d'une autre méthode sont refusés
pour éviter qu'un réglage soit ignoré silencieusement. Chaque méthode a son
nom de fichier PNG distinct, hors nom historique triple-barrier conservé.
