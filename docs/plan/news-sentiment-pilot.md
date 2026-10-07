# Pilote news sentiment

Le chantier sentiment est indépendant du GNN et du gate marché. Il faut commencer par vérifier la chaîne de données, puis comparer les variantes sur un historique admissible. Les titres et résumés ne doivent pas être présentés comme un corpus de textes complets.

## Premier essai Alpha Vantage

Le pilote porte sur septembre 2026, avec cinq actions : `AAPL`, `JPM`, `XOM`, `WMT`, `JNJ`. Les thèmes `economy_macro`, `economy_monetary`, `economy_fiscal` sont récupérés séparément, sans ticker. Les filtres multiples Alpha Vantage correspondent à une intersection, pas à une union. [Documentation fournisseur](https://www.alphavantage.co/documentation/).

Le collecteur garde au maximum 2 000 articles uniques. Pour ce premier essai, les réponses sont limitées à 100 articles et le nombre d'appels à 20. Une fenêtre saturée est subdivisée ; si le budget ne permet pas de terminer, le manifeste reste explicitement incomplet. Ce petit échantillon est adapté à une vérification technique, pas à une conclusion financière.

La limite dure reste 25 tentatives par clé, sur la journée Europe/Paris et sur les dernières 24 heures. Le compteur SQLite est persistant, partagé entre processus et compatible avec celui de la librairie de sentiment. Chaque tentative réserve une place avant le réseau, y compris en cas d'erreur. Il faut utiliser le même fichier de compteur pour tous les clients et déclarer les appels effectués ailleurs. Le compteur local ne connaît pas le solde du fournisseur. [Quota annoncé](https://www.alphavantage.co/support/).

La clé est lue depuis `ALPHAVANTAGE_API_KEY` ou `ALPHA_VANTAGE_API_KEY`, dans l'environnement ou le `.env` local. Elle ne doit apparaître ni dans une commande, ni dans un fichier suivi par Git. Aucun appel n'est effectué par `--dry-run`.

Le test authentifié du 6 octobre 2026 a utilisé **un seul appel**. L'API a refusé `NEWS_SENTIMENT` en indiquant que cet endpoint exige un abonnement premium pour cette clé. Aucun article n'a donc été récupéré et aucun autre appel n'a été tenté. Le quota de 25 ne signifie pas que cet endpoint est accessible gratuitement. Il faut résoudre l'accès fournisseur ou choisir une autre source gratuite avant la collecte réelle ; aucun abonnement n'est souscrit par le script.

```shell
.venv/bin/python scripts/download_news_sentiment_pilot.py \
  --start 2026-09-01T00:00:00Z \
  --end 2026-10-01T00:00:00Z \
  --window-days 30 \
  --limit 100 \
  --max-calls 20 \
  --calls-already-used 0 \
  --max-unique-articles 2000 \
  --output data/external/news-sentiment-pilot-202609 \
  --dry-run
```

Retirer `--dry-run` lance la collecte. Pour reprendre, conserver les mêmes dates, limites de réponse et dossier, puis ajouter `--resume`. Les requêtes terminées ne sont pas refaites. Les erreurs ne sont jamais réessayées implicitement ; une reprise explicite reste soumise au compteur de 25 appels.

## Pilote RSS gratuit exécuté

Le 6 octobre 2026, une collecte réelle a récupéré **409 articles** via cinq recherches Google News en anglais et le flux officiel Fed consacré à la politique monétaire. FinBERT a scoré **405 textes distincts**. Les 394 associations entreprise et les 15 associations macro restent séparées. Aucun nouvel appel Alpha Vantage et aucun téléchargement supplémentaire du modèle.

Les [résultats et vérifications du pilote RSS](../benchmarks/news-sentiment-rss-pilot.md) détaillent les volumes, le routage et les limites. La collecte et les exports occupent environ **2,23 MiB**, sans compter le checkpoint déjà présent. Ce test valide la chaîne technique, pas une performance de trading.

```shell
.venv/bin/python scripts/download_rss_news_sentiment_pilot.py \
  --output data/external/news-sentiment-rss-pilot-20261006 \
  --resume
```

Pour une nouvelle collecte, choisir un nouveau dossier et retirer `--resume`. Chaque flux est récupéré une seule fois, sans suivre les liens vers les pages des éditeurs. La limite est de 100 entrées et 2 MiB par réponse. Les recherches Google sont traitées comme des titres seuls ; leurs descriptions HTML répétées ne sont pas présentées comme de vrais résumés. Le flux Fed peut contenir des publications anciennes, observées pour la première fois lors de cette collecte.

Le routage entreprise exige un alias dans le texte retenu : une réponse à une recherche ne suffit pas à établir une association. Les 27 entrées sans correspondance ont été écartées. La Fed est routée uniquement vers `economy_monetary`, sans ticker. Les XML bruts, checksums, versions de texte et observations propres à chaque association sont conservés. La reprise vérifie les fichiers et ne refait pas les requêtes réussies.

## Disponibilité et couverture

`published_at` décrit la publication annoncée. `available_at` décrit le moment où cette version a réellement été observée par le collecteur. Un téléchargement effectué en octobre ne rend pas les articles disponibles lors des décisions de septembre.

Le pilote conserve les réponses brutes, leur checksum, la version exacte du titre/résumé, les associations et le manifeste de collecte. Il ne fabrique aucun journal `covered` historique. Une collecte terminée pour une requête ne prouve ni l'exhaustivité de toute la presse, ni la disponibilité historique du contenu.

Il faut respecter `available_at < J 00:00 UTC`. Sans journal admissible, l'export conserve `coverage_status=unknown` et le masque faux, même si des articles ont été scorés. Les trois états restent distincts : fenêtre couverte sans news, sentiment neutre observé, source inconnue. Le benchmark refuse d'entraîner une branche sentiment sans observations couvertes dans son train.

## Scoring et routage

Le scoring utilise [news-sentiment-feature-engineering](https://github.com/loic-mmt/news-sentiment-feature-engineering), épinglé à `15424a2c9fd086f4af1740a22c4a9cd032981e40`. FinBERT est exécuté hors entraînement. Le checkpoint, sa révision et le checksum de ses poids sont enregistrés. Un texte partagé entre plusieurs actions ou thèmes est scoré une seule fois.

Le chargement local réel de FinBERT a été vérifié sur trois phrases de contrôle : positive, négative et neutre, puis sur les 409 articles RSS réellement récupérés. Les tests unitaires utilisent des fixtures explicitement synthétiques ; la vérification du pilote réel est consignée séparément.

Pour une simulation historiquement déployable, il faut aussi auditer la date
de disponibilité et les données d'entraînement du checkpoint de sentiment.
Des articles PIT ne rendent pas automatiquement causal un encodeur publié ou
entraîné après la période simulée. Le pilote récent et le futur benchmark long
restent deux protocoles distincts.

Le scoring vérifie les checksums des tables et des réponses brutes, puis rapproche `available_at` de l'observation du collecteur. Une collection vide est refusée avant inférence. Une reprise modifiée ou corrompue est refusée sans écraser l'ancienne version ; les poids, métadonnées du modèle et vocabulaire font partie de son empreinte.

- Entreprise : association explicite vers un ticker, export séance/ticker et 23 features du bridge existant.
- Macro : export global par séance, avec groupes par thème et masques distincts. Aucun faux ticker tradable et aucune copie globale dans les features de chaque action.
- Les scores et relevances Alpha Vantage restent identifiés comme fournisseur. Ils ne sont pas confondus avec les probabilités FinBERT.

Chaque association conserve aussi sa propre date d'observation. Un article vu
avant minuit, mais associé à un thème ou ticker seulement après minuit, n'est
pas utilisé dans cette seconde route pour la décision déjà passée.

La commande de scoring, les fichiers et le chargement sont décrits dans [l'intégration FinBERT](../src/news-sentiment-integration.md).

## Comparaison préparée

| Variante | Ce qui est testé |
| --- | --- |
| `gru` | Contrôle prix seul, sans dépendance au sentiment. |
| `gru_activity` | Nombre de news et couverture, sans polarité : contrôle de l'effet d'activité. |
| `gru_features` | Ajout des 23 features sentiment aux entrées du GRU, après la sélection des features de prix. |
| `sentiment` | Branche sentiment seule, pour mesurer son signal propre. Source manquante : indisponible, avec position FLAT explicite. |
| `gru_sentiment_mean` | Branches GRU et sentiment indépendantes, moyenne des logits disponibles. GRU seul si le sentiment manque. |

Les prix, dates, labels, coûts, seeds et partitions sont communs. Normalisation des news sur les seules observations couvertes du train ; indicateurs de présence et couverture non standardisés. L'early stopping utilise une validation interne, puis le fold externe est évalué avec le checkpoint figé. Le holdout final reste fermé. Les résultats enregistrent les couvertures, probabilités, positions, chemins financiers et diagnostics d'exposition comparable.

```shell
bash scripts/run_us_news_sentiment_benchmark.sh \
  --news-sentiment-export data/derived/us_finbert_company.parquet \
  --dry-run
```

Avec cinq variantes, trois seeds et trois folds : 45 entraînements. Le pilote d'un mois n'est pas ce benchmark. Il faut auparavant disposer d'un historique couvert, sélectionner la même sous-période pour toutes les variantes, puis vérifier le plan. La macro, la calibration de confiance, la fusion apprise et l'abstention sont des expériences suivantes, pas des gains déjà établis.

## Suite

Le [manuel FNSPID sur PC Windows](fnspid-pc-manual.md) détaille le téléchargement
sélectif, l'audit, le checkpoint GPU et le lancement conditionnel du benchmark.
L'import FNSPID, le scoring par shards reprenables et le protocole exploratoire
explicite sont maintenant disponibles via `scripts/run_fnspid_news_benchmark.py`.
Ils restent distincts du pilote RSS et du benchmark PIT strict ; voir la section 10
du manuel. Le téléchargement ne transforme pas le corpus en preuve PIT.

L'extension est décrite en [section 12 du manuel](fnspid-pc-manual.md#12-inventaire-des-143-tickers-et-comparaison-avec-plusieurs-permutations) :
inventaire des 143 tickers attendus du socle prix, sélection selon les observations
du premier TRAIN, réutilisation de l'import et des scores FinBERT du pilote,
puis quatre références et cinq permutations indépendantes des seeds modèle
(81 entraînements prévus). La complétude des prix reste un filtre rétrospectif ;
la couverture reste inconnue et le holdout fermé. Cette procédure ne revendique
pas encore un inventaire réel ou un benchmark élargi exécuté.

Le scoring réel, les probabilités, les checksums, les exports et les reprises
ont été vérifiés sur le corpus RSS du 6 octobre. Les **84 tests ciblés** couvrent
la collecte, le scoring et l'ablation, dont 29 spécifiques au RSS. La reprise
réelle n'a fait aucune requête HTTP ni nouvelle inférence. Le compteur Alpha
Vantage reste à un seul appel réservé ; sa réponse premium reste dans
`data/external/news-sentiment-pilot-202609`.

La suite complète passe : **1 132 tests réussis, 11 ignorés**.

1. Collecte, scoring réel, exports, masques et reprise : vérifiés sur le petit pilote RSS.
2. Taille réelle : environ 2,23 MiB pour les données et exports du pilote ; conserver le corpus étendu sur le PC.
3. Fournir un corpus historique et une couverture PIT admissibles, ou constituer une collecte prospective.
4. Lancer les cinq variantes sur dates et expositions comparables, sans ouvrir le holdout pour choisir les paramètres.
5. Tester ensuite le sentiment macro dans le contexte global du Transformer, puis la calibration/fusion confidence-aware séparément.
