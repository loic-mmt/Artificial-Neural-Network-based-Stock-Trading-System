# Manuel PC : FNSPID, FinBERT et benchmark sentiment

Procédure Windows PowerShell, préparée le 6 octobre 2026. Il faut exécuter les commandes depuis la racine du dépôt. Les chemins du PC sont à adapter ; les commandes utilisent directement le Python du `.venv`, sans activation obligatoire.

## Objectif et état actuel

Il faut télécharger les news sur le PC, vérifier leur couverture réelle, scorer les titres avec FinBERT sur GPU, puis comparer les variantes sentiment sur les mêmes actions, dates et coûts.

| Étape | État |
| --- | --- |
| Télécharger FNSPID et vérifier les fichiers | Possible maintenant. |
| Auditer les colonnes, dates et tickers | Possible maintenant. |
| Télécharger le checkpoint FinBERT immuable | Possible maintenant. |
| Importer FNSPID et scorer son historique par lots reprenables | Adaptateur spécifique à implémenter. Le scorer du pilote RSS n'est pas un importateur FNSPID. |
| Exporter les features depuis un corpus scoré | Disponible pour des données respectant le contrat de disponibilité. |
| Benchmark PIT strict | Runner disponible ; exige une couverture et une disponibilité historiques prouvées. |
| Benchmark FNSPID exploratoire sans preuves PIT | Protocole explicite à implémenter, sans détourner les masques du runner strict. |

**Le téléchargement ne rend pas FNSPID automatiquement admissible au benchmark strict.** Il ne faut ni antidater `collected_at`, ni inventer `available_at`, ni fabriquer un journal `covered` pour faire passer le dry-run. Les étapes de téléchargement peuvent être effectuées avant cet adaptateur ; les étapes de benchmark restent conditionnelles.

Le [projet FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) annonce un dataset global sur 1999 à 2023. Ce n'est pas une garantie de news sur chaque ticker et chaque année. Il ne faut pas supposer une couverture 2024 à 2026, ni remplacer nos prix par ceux de ce dataset. Les conditions d'usage du corpus et des éditeurs doivent être relues avant redistribution ou usage commercial ; le README contient des mentions de licence contradictoires.

## 1. Préparer le dépôt et le GPU

Sur le Mac, il faut d'abord pousser la version contenant les scripts et les dépendances du chantier sentiment. Sur le PC, il faut récupérer la même révision. Vérifier les modifications locales avant le pull ; s'il est refusé, ne pas les écraser.

```powershell
git status --short
git pull --ff-only
git rev-parse HEAD

$Python = ".\.venv\Scripts\python.exe"
$Hf = ".\.venv\Scripts\hf.exe"
```

Si le `.venv` fonctionne déjà avec CUDA, il faut le conserver. Pour un environnement neuf seulement, avec Python 3.12 et Git installés :

```powershell
py -3.12 -m venv .venv
& $Python -m pip install --upgrade pip
```

Il faut installer une version CUDA de PyTorch avant les dépendances sentiment, pour ne pas laisser un environnement CPU par défaut. Pour un environnement neuf, voici un exemple versionné CUDA 11.8 proposé dans la [documentation PyTorch](https://pytorch.org/get-started/previous-versions/#v271). Ce n'est pas une consigne de rétrograder un environnement GPU déjà fonctionnel. Une autre version compatible peut être conservée et doit être enregistrée dans le protocole.

```powershell
& $Python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
```

Installer ensuite le package et la librairie de sentiment épinglée par `pyproject.toml`. La CLI Hugging Face utilisée ci-dessous est disponible dans la série 1.32 :

```powershell
& $Python -m pip install -e ".[sentiment,dev]"
& $Python -m pip install "huggingface_hub>=1.32,<2"
& $Python -m pip check
& $Hf version

& $Python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); assert torch.cuda.is_available(), "CUDA indisponible"; print(torch.cuda.get_device_name(0)); x=torch.ones(4, device="cuda"); print((x*x).sum().item())'
```

Le test doit afficher le GPU et calculer `4.0` sur CUDA. Sinon, il faut corriger le driver ou le wheel PyTorch avant le scoring. Le simple fait d'avoir une RTX 2060 ne suffit pas à confirmer le fonctionnement du runtime.

Les prix ne sont pas suivis par Git. Il faut copier le fichier figé `data/processed/mt5_stocks_us_daily_clean.parquet` utilisé par les benchmarks, ou produire une nouvelle version clairement identifiée. Pour reproduire le même socle, la copie est préférable à un nouveau téléchargement Yahoo. Aucun fichier dans `MT5/` n'est nécessaire pour le lancement de ce benchmark.

```powershell
Test-Path "data/processed/mt5_stocks_us_daily_clean.parquet"
Get-FileHash "data/processed/mt5_stocks_us_daily_clean.parquet" -Algorithm SHA256
```

Il faut comparer ce hash avec celui du fichier source. Une source actualisée constitue un nouveau dataset et ne doit pas servir à reprendre un run ancien.

## 2. Choisir les fichiers et figer leur révision

Les [fichiers news officiels](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_news) affichent :

| Fichier | Taille distante indicative |
| --- | ---: |
| `Stock_news/All_external.csv` | 5,73 Go |
| `Stock_news/nasdaq_exteral_data.csv` | 23,2 Go |

Il faut commencer par `All_external.csv`, puis auditer sa couverture. Le second fichier ne doit être ajouté que si son contenu apporte les périodes/tickers manquants ou un texte utile. Le nom `All_external` ne garantit ni l'exhaustivité ni l'absence de recouvrement avec l'autre fichier.

Ces CSV ne sont pas téléchargeables par ticker avec la commande Hub ci-dessous : le fichier choisi est récupéré en entier, puis filtré localement. Prévoir une marge pour le cache, les Parquet, les checkpoints et les résultats. Pour les deux CSV et l'ensemble du travail, 60 Go libres constituent une marge de départ, pas une mesure de la taille finale filtrée.

Le corpus est public ; aucune clé Alpha Vantage n'est utilisée. Il ne faut pas lancer le script de téléchargement de FNSPID qui récupère aussi les prix, ni télécharger tout le dépôt Hub sans filtre.

```powershell
$FnspidRepo = "Zihan1004/FNSPID"
$FnspidRoot = "data/external/fnspid"
$FnspidFile = "Stock_news/All_external.csv"
$FnspidAudit = Join-Path $FnspidRoot "dataset-info.json"

New-Item -ItemType Directory -Path $FnspidRoot -Force | Out-Null

if (Test-Path $FnspidAudit) {
    $FnspidInfo = Get-Content $FnspidAudit -Raw | ConvertFrom-Json
} else {
    $FnspidInfoJson = & $Hf datasets info $FnspidRepo --expand "sha,siblings" --format json
    if ($LASTEXITCODE -ne 0) { throw "Lecture des métadonnées FNSPID échouée" }
    $FnspidInfo = ($FnspidInfoJson -join "`n") | ConvertFrom-Json
    $FnspidInfo | ConvertTo-Json -Depth 20 | Set-Content $FnspidAudit -Encoding UTF8
}

$FnspidRevision = $FnspidInfo.sha
if ($FnspidRevision -notmatch '^[0-9a-f]{40}$') { throw "Révision FNSPID invalide" }
Write-Output "FNSPID revision=$FnspidRevision file=$FnspidFile"

& $Hf download $FnspidRepo $FnspidFile --repo-type dataset --revision $FnspidRevision --local-dir $FnspidRoot --dry-run
```

Il faut conserver `dataset-info.json`. Sur une reprise, ce fichier conserve la révision initiale ; il ne faut pas remplacer silencieusement la révision par le nouveau `main`.

## 3. Télécharger et vérifier

Après contrôle du dry-run :

```powershell
& $Hf download $FnspidRepo $FnspidFile --repo-type dataset --revision $FnspidRevision --local-dir $FnspidRoot
if ($LASTEXITCODE -ne 0) { throw "Téléchargement FNSPID incomplet" }

& $Hf cache verify $FnspidRepo --repo-type dataset --revision $FnspidRevision --local-dir $FnspidRoot
if ($LASTEXITCODE -ne 0) { throw "Vérification FNSPID échouée" }

$FnspidCsv = Join-Path $FnspidRoot $FnspidFile
Get-FileHash $FnspidCsv -Algorithm SHA256
```

Il faut vérifier que le rapport concerne bien le CSV téléchargé. Il peut signaler les autres fichiers du dépôt comme absents : c'est normal pour un téléchargement sélectif. Ne pas ajouter `--fail-on-missing-files`, qui exigerait aussi les fichiers non sélectionnés. Les [commandes Hub](https://huggingface.co/docs/huggingface_hub/en/guides/cli) réutilisent les fichiers valides ; après interruption, relancer la même commande avec la même révision, sans `--force-download`. Les contrôles restent nécessaires après la reprise.

Si le second CSV est nécessaire, modifier seulement le fichier demandé, en gardant la révision :

```powershell
$FnspidFile = "Stock_news/nasdaq_exteral_data.csv"
& $Hf download $FnspidRepo $FnspidFile --repo-type dataset --revision $FnspidRevision --local-dir $FnspidRoot --dry-run
```

Après contrôle de la place, relancer les commandes de téléchargement et de vérification de cette section. Il ne faut pas supprimer le premier CSV ni fusionner les deux sans déduplication.

## 4. Auditer les dates et les tickers avant de choisir la période

Afficher seulement un petit échantillon, sans charger plusieurs Go en mémoire :

```powershell
$FnspidCsv = Join-Path $FnspidRoot $FnspidFile
& $Python -c 'import sys,pandas as pd; f=pd.read_csv(sys.argv[1], nrows=5, dtype=str); print(f.columns.tolist()); print(f[["Date","Article_title","Stock_symbol"]].to_string(index=False))' $FnspidCsv
```

Il faut ensuite parcourir le CSV par chunks, en commençant par les cinq tickers du pilote. Le code suivant est un audit de dates déclarées, pas une preuve de timezone ou de disponibilité historique :

```powershell
$FnspidAuditCode = @'
import sys
import pandas as pd

tickers = ["AAPL", "JPM", "XOM", "WMT", "JNJ"]
columns = ["Date", "Article_title", "Stock_symbol"]
stats = {ticker: {"rows": 0, "invalid_dates": 0, "empty_titles": 0, "first": None, "last": None} for ticker in tickers}
for chunk in pd.read_csv(sys.argv[1], usecols=columns, dtype=str, keep_default_na=False, chunksize=100_000):
    symbols = chunk["Stock_symbol"].str.strip().str.upper()
    for ticker in tickers:
        part = chunk.loc[symbols.eq(ticker)]
        if part.empty:
            continue
        dates = pd.to_datetime(part["Date"], format="mixed", utc=True, errors="coerce")
        row = stats[ticker]
        row["rows"] += len(part)
        row["invalid_dates"] += int(dates.isna().sum())
        row["empty_titles"] += int(part["Article_title"].str.strip().eq("").sum())
        if dates.notna().any():
            first, last = dates.min(), dates.max()
            row["first"] = min(row["first"], first) if row["first"] is not None else first
            row["last"] = max(row["last"], last) if row["last"] is not None else last
print(pd.DataFrame.from_dict(stats, orient="index").to_string())
'@
$FnspidAuditCode | & $Python - $FnspidCsv
```

Une date sans timezone sera interprétée comme UTC uniquement pour ce profilage. Cette hypothèse ne doit pas devenir une preuve dans l'import. En cas de colonnes manquantes ou de lignes CSV invalides, il faut auditer le schéma ; ne pas utiliser `on_bad_lines="skip"` pour masquer des pertes.

Avant le benchmark, le rapport d'import devra également mesurer les comptes par mois, les trous, les doublons, les associations ambiguës et la couverture de chaque ticker. Deux bornes min/max ne prouvent pas une présence continue. Il faut commencer sur quelques années communes réellement présentes, puis étendre l'univers ; il ne faut pas fixer 2005 à 2026 depuis la seule période annoncée du dataset.

## 5. Télécharger le même checkpoint FinBERT

Le pilote a utilisé cette révision immuable, sans poids TensorFlow supplémentaires :

```powershell
& $Hf download yiyanghkust/finbert-tone config.json vocab.txt pytorch_model.bin --revision 4921590d3c0c3832c0efea24c8381ce0bda7844b --local-dir .cache/finbert-tone-4921590
if ($LASTEXITCODE -ne 0) { throw "Téléchargement FinBERT incomplet" }

$FinbertHash = (Get-FileHash ".cache/finbert-tone-4921590/pytorch_model.bin" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($FinbertHash -ne "f31c2036e91c9854bcc35141d16669dd07b9726adfe391d1011bff1de7ea4b32") {
    throw "Checksum FinBERT différent de la révision vérifiée"
}
```

Le modèle occupe environ 419 MiB. Il faut utiliser le chargeur local déjà présent pour les métadonnées BERT legacy, sans réécrire les poids. Sa date de publication et ses données d'entraînement devront aussi être auditées si le benchmark revendique une simulation historiquement déployable sur une période antérieure.

## 6. Point d'arrêt : import et politique historique à implémenter

Après téléchargement et audit, il faut ajouter un importateur FNSPID dans la librairie de sentiment ou un adaptateur mince ici, en conservant le runner strict. Aucune commande `import_fnspid.py` ou option d'exploration n'existe actuellement ; il ne faut pas les supposer disponibles.

Le travail attendu est le suivant :

1. Lecture par chunks, filtre du ticker et de la période choisie, normalisation des noms/changements de ticker, langue et routage explicites.
2. Titres réels comme première expérience. Exclure les résumés `Lsa`, `Luhn`, `Textrank`, `Lexrank` de cette référence ; ils sont des dérivés, pas nécessairement des résumés publiés à l'époque.
3. Identités article et version de texte, déduplication entre CSV et entre tickers, conservation des URL, dates brutes, hashes et révision Hub.
4. Séparation `published_at`, date de collecte actuelle et disponibilité historique. En l'absence de preuve, une hypothèse de publication avec délai reste une hypothèse exploratoire, pas un `available_at` PIT vérifié.
5. Journal de couverture distinct. Un CSV sans news un jour ne prouve pas une fenêtre couverte sans news. Une association `Stock_symbol` ne prouve pas à elle seule la pertinence du texte pour l'action.
6. Scoring hors entraînement avec la librairie épinglée, checkpoint local, CUDA, déduplication des textes et shards reprenables. Un batch de 16 est un point de départ à mesurer sur la RTX 2060 ; réduire le batch si nécessaire, sans modifier le texte/tokenization en cours de reprise.
7. Protocole exploratoire séparé si aucune preuve PIT n'est fournie. Il faut tracer les hypothèses, les pertes et l'inconnu, tester plusieurs délais et ne jamais remplir les masques stricts avec de faux `covered`.
8. Tests du mapping, des dates, des versions, des reprises et de l'absence de lookahead, puis smoke test sur le vrai corpus filtré.

Le script `score_news_sentiment_pilot.py` vérifie des observations `collector_first_seen` et leurs preuves brutes. Un historique FNSPID ne doit pas être déguisé en collecte RSS pour le faire accepter.

Livrables à produire, sous des chemins de sortie versionnés :

| Fichier prévu | Rôle |
| --- | --- |
| `data/external/fnspid/dataset-info.json` | Révision Hub téléchargée. |
| `data/derived/fnspid/import-report.json` | Couverture mesurée, rejets, timezone, mapping et politique historique. |
| `data/derived/fnspid/scored_company.parquet` | Titres scorés, versions, probabilités et provenance. |
| `data/derived/fnspid/company_daily.parquet` et `.manifest.json` | Features quotidiennes et qualification du protocole. |
| `data/derived/fnspid/prices-matched.parquet` | Même période et même univers pour toutes les variantes. |
| `data/derived/fnspid/tickers-matched.json` | Univers déterminé par disponibilité, jamais par PnL du test. |

Ces chemins sont des livrables attendus, pas des fichiers déjà créés. Si un journal PIT authentique existe, son export peut utiliser `scripts/export_news_sentiment.py`, décrit dans [l'intégration FinBERT](../src/news-sentiment-integration.md). Sans preuves, le benchmark exploratoire doit attendre son implémentation explicite.

## 7. Figer la comparaison

Il faut conserver les mêmes prix, calendrier, coûts, labels et partitions pour toutes les variantes. La référence GRU doit être réentraînée sur cette sous-période ; les anciens résultats 2005 à 2026 ne constituent pas son contrôle apparié. Les cinq tickers forment un smoke test, pas une validation suffisante pour tout l'univers US.

Le fichier de sélection ne filtre actuellement que les tickers. Une clé `start` dans son JSON n'est pas un filtre de dates du runner. Il faut donc créer réellement le Parquet `prices-matched.parquet` limité à la période choisie, avec une politique de warmup historique explicite et aucune cible hors partition.

Il faut également vérifier que les features marché/secteur du dataset source n'utilisent pas des actifs extérieurs au nouvel univers sans déclaration. La population servant au contexte doit être fixée et identique pour les cinq variantes. Normalisations et sélection restent ajustées seulement sur le train.

| Variante | Question |
| --- | --- |
| `gru` | Contrôle prix seul. |
| `gru_activity` | Effet de l'activité news, sans polarité. |
| `gru_features` | Ajout des 23 canaux sentiment aux entrées temporelles. |
| `sentiment` | Signal propre de la branche sentiment. |
| `gru_sentiment_mean` | Moyenne masquée des logits des deux branches. |

La macro, le GNN, le gate marché et la fusion de confiance ne sont pas ajoutés à cette première ablation. Le corpus FNSPID ne doit pas être converti indistinctement en news macro pour tous les tickers.

## 8. Commandes du runner strict, seulement après admissibilité de l'export

Les arguments ci-dessous correspondent au runner existant. Ils ne sont **pas encore exécutables sur le CSV FNSPID brut**. Il faut les utiliser uniquement après création des livrables et vérification de leur admissibilité PIT. Si l'étude reste exploratoire, sa future interface devra adapter ce lancement et enregistrer cette qualification ; aucune option exploratoire fictive n'est donnée ici.

Définir les arguments communs PowerShell :

```powershell
$NewsBenchmarkArgs = @(
    "--data", "data/derived/fnspid/prices-matched.parquet",
    "--ticker-selection", "data/derived/fnspid/tickers-matched.json",
    "--preset", "multi_ticker_long_short",
    "--models", "gru",
    "--model-parameter-sets", "configs/benchmark/gru_market_context.json",
    "--losses", "combined",
    "--combined-weights", "0.25",
    "--loss-cost-bps", "5",
    "--selection-metric", "regularized_sharpe",
    "--context-len", "60",
    "--position-mode", "long_short",
    "--execution-delay", "1",
    "--train-ratio", "0.7",
    "--val-ratio", "0.15",
    "--label-method", "triple-barrier",
    "--label-max-holding", "10",
    "--label-vol-window", "20",
    "--label-volatility-estimator", "atr",
    "--label-profit-barrier", "0.75",
    "--label-stop-barrier", "0.75",
    "--label-event-filter", "cusum",
    "--label-cusum-threshold", "0.5",
    "--label-between-events", "hold",
    "--label-cost-bps", "5",
    "--feature-set", "expanded",
    "--feature-groups", "technical,market,sector",
    "--no-external-features",
    "--overfitting-control",
    "--overfitting-max-features", "32",
    "--overfitting-max-feature-correlation", "0.95",
    "--news-sentiment-export", "data/derived/fnspid/company_daily.parquet",
    "--date-batch-size", "16",
    "--cv-gap-bars", "5",
    "--cv-score", "regularized_sharpe",
    "--device", "cuda",
    "--fail-fast"
)
```

`--no-external-features` désactive les anciennes sources fundamentals/VADER ; il ne désactive pas l'export FinBERT passé séparément. La perte conserve la combinaison PnL/Sharpe à 0,25 et le PnL scale par défaut du runner, à enregistrer avec les autres paramètres.

### Smoke test : deux variantes, deux folds, seed 42

Le dry-run vérifie l'export, la présence d'observations couvertes dans le train et les signatures, sans entraîner. Une erreur de couverture doit être résolue par les données ou le protocole, pas par un journal inventé.

```powershell
$NewsSmokeArgs = $NewsBenchmarkArgs + @(
    "--sentiment-candidates", "gru,gru_features",
    "--cv-folds", "2",
    "--seeds", "42",
    "--output-dir", "artifacts/comparisons/fnspid-news-smoke"
)

& $Python scripts/run_news_sentiment_comparison.py @NewsSmokeArgs --dry-run
if ($LASTEXITCODE -ne 0) { throw "Dry-run smoke refusé" }

& $Python scripts/run_news_sentiment_comparison.py @NewsSmokeArgs
```

Cela fait **4 entraînements**, indépendamment du nombre de tickers : deux variantes × deux folds × une seed. Le holdout final reste fermé.

### Comparaison complète : cinq variantes, trois folds, trois seeds

```powershell
$NewsFullArgs = $NewsBenchmarkArgs + @(
    "--sentiment-candidates", "gru,gru_activity,gru_features,sentiment,gru_sentiment_mean",
    "--cv-folds", "3",
    "--seeds", "1,7,19",
    "--output-dir", "artifacts/comparisons/fnspid-news-comparison"
)

& $Python scripts/run_news_sentiment_comparison.py @NewsFullArgs --dry-run
if ($LASTEXITCODE -ne 0) { throw "Dry-run complet refusé" }

& $Python scripts/run_news_sentiment_comparison.py @NewsFullArgs
```

Cela fait **45 entraînements** : cinq variantes × trois folds × trois seeds, sans entraînement final supplémentaire. Il ne faut pas ajouter `--final-test` ou `--cv-final-test` pendant le choix des variantes.

### Reprise

Dans la même session PowerShell, avec les variables et arguments inchangés :

```powershell
& $Python scripts/run_news_sentiment_comparison.py @NewsFullArgs --resume --dry-run
if ($LASTEXITCODE -ne 0) { throw "Reprise incompatible" }

& $Python scripts/run_news_sentiment_comparison.py @NewsFullArgs --resume
```

Après redémarrage, il faut redéfinir `$Python`, `$NewsBenchmarkArgs` et `$NewsFullArgs`. Le runner reprend les folds terminés et enregistrés, pas les epochs d'un fold interrompu. Des fichiers partiels non enregistrés provoquent un refus d'écrasement ; il faut les conserver pour diagnostic et choisir une nouvelle sortie si nécessaire. Ne pas changer code, données, manifeste, CUDA/runtime, seeds ou paramètres au milieu d'une reprise.

## 9. Vérifications et lecture des résultats

Les tests locaux de base peuvent être lancés avant l'import, puis complétés par les nouveaux tests FNSPID :

```powershell
& $Python -m pytest -q tests/test_news_pilot_scoring.py tests/test_news_sentiment_ablation.py tests/test_rss_news_collection.py
```

À la fin du benchmark, il faut retrouver :

- `metadata.json` : configuration, données, export, partitions et provenance.
- `folds.json` : tâches terminées, fit, couverture train/inner/outer et métriques.
- `report.json` : synthèse, différences appariées au GRU, contrôle descriptif d'exposition et `final_test=[]`.
- `fold-*-outer-predictions.parquet` : positions, probabilités, disponibilité et clés du backtest.
- `fold-*-outer-daily.parquet` : chemins financiers.
- `exposure-mean-min-daily.parquet` : comparaison descriptive à exposition brute moyenne comparable.

Il faut analyser le Sharpe régularisé, le rendement net, les coûts, l'exposition, le turnover et la couverture, pas seulement l'accuracy. Le contrôle d'exposition utilise des facteurs calculés ex post sur le fold ; il ne constitue pas un sizing déployable. Une branche sentiment entièrement FLAT peut faire tomber cette exposition commune à zéro. Les folds/seeds ne sont pas des marchés indépendants, et leur moyenne n'est pas un backtest concaténé.

Le petit univers ne permet pas de conclure sur les 143 actions. Il faut ensuite refaire les cinq variantes sur l'univers élargi, à dates communes, avec les mêmes contrôles. Les hypothèses historiques de FNSPID et le biais de survivance de l'univers actuel doivent rester visibles dans toute conclusion.

## Checklist avant le lancement financier

- [ ] Révision du dépôt, environnement et CUDA enregistrés.
- [ ] CSV téléchargé à révision Hub figée, checksum vérifié.
- [ ] Dates, timezone, trous, contenus et tickers audités.
- [ ] Importateur et scoring FNSPID reprenables implémentés et testés.
- [ ] Choix explicite : PIT prouvé ou recherche exploratoire, sans mélange.
- [ ] Même univers, contexte, calendrier et prix pour toutes les variantes.
- [ ] Couverture et disponibilité traitées sans faux journal `covered`.
- [ ] Dry-run accepté dans le protocole approprié, puis smoke test terminé.
- [ ] Sortie complète distincte et holdout final fermé.

Références du dépôt : [plan sentiment](news-sentiment-pilot.md), [intégration](../src/news-sentiment-integration.md), [pilote RSS réel](../benchmarks/news-sentiment-rss-pilot.md).
