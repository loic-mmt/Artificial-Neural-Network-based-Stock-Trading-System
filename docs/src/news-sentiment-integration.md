# Intégration des features de news FinBERT

Les benchmarks **ne collectent pas** de news et **ne lancent pas** FinBERT pendant
l'entraînement. Des scripts séparés permettent maintenant une petite collecte
RSS ou Alpha Vantage et leur scoring hors entraînement. La librairie
`news-sentiment-feature-engineering` produit des news
déjà scorées, un journal de couverture, puis un export de features Parquet avec
manifeste. Ici, `scripts/export_news_sentiment.py` construit les décisions à
partir des séances du marché, appelle cet export et vérifie qu'il est lisible
par le contrat multimodal.

Le protocole de collecte bornée et les cinq variantes préparées sont dans
[le plan du pilote sentiment](../plan/news-sentiment-pilot.md).

## Pilote local : titres et résumés

Après installation de l'extra `sentiment`, récupérer un checkpoint immuable,
sans doublonner les poids TensorFlow :

```shell
.venv/bin/hf download yiyanghkust/finbert-tone \
  config.json vocab.txt pytorch_model.bin \
  --revision 4921590d3c0c3832c0efea24c8381ce0bda7844b \
  --local-dir .cache/finbert-tone-4921590
```

Le checkpoint téléchargé représente environ 419 MiB. Il n'est pas poussé sur
GitHub. Son chargement local conserve les poids originaux et adapte uniquement
les métadonnées BERT legacy nécessaires à Transformers récent. Les poids sont
partagés par lien physique temporaire, avec copie de secours si nécessaire,
sans exiger de privilège pour un lien symbolique sur Windows.

L'essai Alpha Vantage a été bloqué après un appel par l'accès premium.
Un second pilote RSS a effectivement récupéré 409 articles le 6 octobre.
La commande suivante reprend ce corpus et les scores vérifiés, sans nouvelle
collecte ni inférence. Le dossier vide de l'essai Alpha Vantage reste refusé.

```shell
.venv/bin/python scripts/score_news_sentiment_pilot.py \
  --input-dir data/external/news-sentiment-rss-pilot-20261006 \
  --output-dir data/derived/news-sentiment-rss-pilot-20261006 \
  --model-dir .cache/finbert-tone-4921590 \
  --model-repository yiyanghkust/finbert-tone \
  --model-revision 4921590d3c0c3832c0efea24c8381ce0bda7844b \
  --model-weights-sha256 f31c2036e91c9854bcc35141d16669dd07b9726adfe391d1011bff1de7ea4b32 \
  --data data/processed/mt5_stocks_us_daily_clean.parquet \
  --tickers AAPL,JPM,XOM,WMT,JNJ \
  --decision-start 2026-09-01 \
  --decision-end 2026-10-01 \
  --preview-next-midnight \
  --device cpu \
  --resume
```

Le checksum s'obtient avec
`shasum -a 256 .cache/finbert-tone-4921590/pytorch_model.bin` sur le Mac,
ou `Get-FileHash .cache/finbert-tone-4921590/pytorch_model.bin -Algorithm SHA256`
sur Windows. La commande ci-dessus contient le checksum vérifié des poids
de cette révision ; un autre fichier ou checkpoint doit avoir sa propre empreinte.

Le dossier de collecte conserve `articles.parquet`, `associations.parquet`,
les XML RSS ou JSON fournisseur bruts et le manifeste. Le scoring écrit les articles scorés une seule
fois, les associations entreprise et macro séparées, puis les exports quotidiens.
`company_sentiment.parquet` est lu par le bridge existant ;
`macro_sentiment.parquet` est un panel global séparé, pas un ticker à trader.
Les scores fournisseur restent séparés des scores FinBERT.

`--resume` vérifie les empreintes des entrées et du checkpoint avant réutilisation.
Une nouvelle collecte ou un autre modèle demande une nouvelle version de sortie.
Sans journal de couverture authentique, les exports sont `unknown` et les
masques restent faux. Scorer aujourd'hui les news de septembre ne permet pas
de les utiliser rétrospectivement en septembre.

`scripts/download_rss_news_sentiment_pilot.py` utilise `feedparser` et les
règles de nettoyage/alias de la librairie épinglée. Un adaptateur local ajoute
les réponses XML brutes et la provenance exigée par le scorer, avec une seule
récupération bornée par flux. Il ne lit pas `.env` et ne touche pas au compteur
Alpha Vantage. Les valeurs `vendor_*` restent absentes, jamais inventées.

`--preview-next-midnight` écrit dans `technical-preview/` deux points de
décision artificiels : minuit avant la collecte et le minuit suivant. Aucun
prix ni signal de trading n'est créé. L'aperçu vérifie les comptes, l'agrégation
et la borne stricte sur de vrais textes, mais garde la couverture `unknown`
et les masques faux. Les manifests identifient explicitement ce protocole
comme `synthetic_cutoffs_not_a_backtest`. Les exports issus des vraies séances
de septembre restent distincts et ne contiennent aucune news utilisable.

Les [résultats du pilote RSS](../benchmarks/news-sentiment-rss-pilot.md)
documentent les vérifications réelles et les tailles observées.

## Sur le PC qui héberge le corpus

Installer l'extra optionnel, épinglé à une révision de la librairie :

```shell
python -m pip install -e '.[sentiment]'
```

Préparer d'abord un Parquet de news **déjà scorées par FinBERT** avec les
colonnes normalisées de la librairie, notamment `available_at`,
`availability_kind` et `availability_reference`. Une date de publication ou
une collecte actuelle ne remplace pas une preuve de disponibilité historique.
Préparer séparément un journal de couverture. Sans preuve de couverture, une
absence de news est `unknown` et le masque de sentiment reste faux.

```shell
PYTHONPATH=src python scripts/export_news_sentiment.py \
  --data data/processed/cac40_daily_clean.parquet \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --scored-news /chemin/vers/scored_news.parquet \
  --coverage /chemin/vers/coverage.parquet \
  --required-sources feed-a,feed-b \
  --checkpoint 'yiyanghkust/finbert-tone@REVISION_IMMUABLE' \
  --output data/derived/cac40_finbert_daily.parquet
```

La commande lit le corpus uniquement sur le PC, impose
`available_at < J 00:00 UTC`, écrit le Parquet et son
`cac40_finbert_daily.manifest.json`, puis vérifie leur checksum. Elle ne
télécharge ni news ni checkpoint. L'argument `--checkpoint` décrit le modèle
qui a **déjà** produit les scores ; il ne déclenche aucune inférence.

## Chargement dans le contrat multimodal

```python
from trading_system.data import build_multimodal_dataset, load_news_sentiment_export

sentiment = load_news_sentiment_export("data/derived/cac40_finbert_daily.parquet")
dataset = build_multimodal_dataset(
    market_rows,
    tickers=selected_tickers,
    context_len=60,
    temporal_columns=temporal_columns,
    sentiment_frame=sentiment.frame,
    sentiment_columns=sentiment.columns,
)
```

Le chargeur refuse un manifeste incompatible, un checksum différent, une
borne inclusive, des clés répétées et une news future. Les statistiques
indéfinies sont remplacées par zéro **avec un indicateur `_present` distinct**.
Un sentiment neutre observé a donc `sentiment_mean=0` et
`sentiment_mean_present=1` ; une fenêtre couverte sans news a
`news_count=0`, indicateurs nuls et `sentiment_mask=True` ; une couverture
inconnue ou incomplète a `sentiment_mask=False` même si des articles isolés
sont présents. Aucun ajustement statistique n'est appris sur validation/test.

## Runner de comparaison séparé

Le runner `scripts/run_loss_comparison.py` et son ancienne source
`--sentiment` basée sur des événements/VADER restent séparés. Cet export
alimente le contrat multimodal et `SentimentBranch`. Le nouveau
`scripts/run_news_sentiment_comparison.py` accepte `--news-sentiment-export`,
`--sentiment-candidates`, `--dry-run` et `--resume`.

Les variantes sont `gru`, `gru_activity`, `gru_features`, `sentiment`,
`gru_sentiment_mean`. Le lancement US est préparé dans
`scripts/run_us_news_sentiment_benchmark.sh`. Il conserve la sélection des
features de prix, puis ajoute uniquement les canaux news de la variante choisie.
L'absence de sentiment n'interrompt pas le GRU ; une fusion moyenne renormalise
ses poids sur les branches disponibles. Les probabilités FinBERT ne sont pas
des prédictions Sell/Hold/Buy : la branche de trading les apprend séparément.

La macro reste hors de ce runner de comparaison. Le panel global est préparé
pour une expérience ultérieure dans le Transformer marché, sans modifier
silencieusement les benchmarks GNN/GRU existants.
