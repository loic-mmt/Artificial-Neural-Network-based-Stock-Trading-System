# Intégration des features de news FinBERT

Le dépôt de trading **ne collecte pas** de news et **ne lance pas** FinBERT pendant
un benchmark. La librairie `news-sentiment-feature-engineering` produit des news
déjà scorées, un journal de couverture, puis un export de features Parquet avec
manifeste. Ici, `scripts/export_news_sentiment.py` construit les décisions à
partir des séances du marché, appelle cet export et vérifie qu'il est lisible
par le contrat multimodal.

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

Le runner actuel `scripts/run_loss_comparison.py` et son ancienne source
`--sentiment` basée sur des événements/VADER restent séparés. Cet export
alimente le contrat multimodal et peut entrer dans `SentimentBranch` ; le runner
multimodal et ses options de benchmark restent à créer. Le benchmark lancé avec
`--no-external-features` ne change pas.
