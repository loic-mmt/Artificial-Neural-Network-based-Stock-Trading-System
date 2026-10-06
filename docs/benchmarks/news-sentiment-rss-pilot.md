# Pilote RSS et FinBERT

## Résultat du 6 octobre 2026

La chaîne collecte RSS, nettoyage, routage, scoring FinBERT et export a été exécutée sur de vraies news. **409 articles, 405 textes distincts**, sans nouvel appel Alpha Vantage. Aucun entraînement du modèle de trading et aucune conclusion de PnL à ce stade.

| Route | Entrées du flux | Articles retenus |
| --- | ---: | ---: |
| AAPL | 100 | 96 |
| JPM | 100 | 99 |
| XOM | 64 | 60 |
| WMT | 91 | 88 |
| JNJ | 66 | 51 |
| Macro `economy_monetary` | 15 | 15 |
| Total | 436 | 409 |

Les recherches [Google News RSS](https://news.google.com/rss/search?q=Apple+stock+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen) portent sur cinq entreprises, en anglais, sur les sept derniers jours. Les 27 titres sans alias explicite ont été rejetés. Il s'agit d'un routage lexical, pas d'une résolution d'entités validée. Les descriptions répétées du flux Google ne sont pas utilisées comme résumés. Aucun texte complet ni page d'éditeur n'est téléchargé.

Le [flux Fed](https://www.federalreserve.gov/feeds/press_monetary.xml) est indépendant et peut contenir des publications anciennes. Il est routé uniquement vers la macro, jamais copié dans les news de chaque action. Son téléchargement ne constitue pas un historique couvert de toute la presse.

## Taille et fichiers

- XML bruts : 575 624 octets, soit environ 562 KiB.
- Collecte complète, tables et audit : 1 602 294 octets, soit environ 1,53 MiB.
- Scores, exports et aperçus : 734 917 octets, soit environ 0,70 MiB.
- Total des nouvelles données : environ **2,23 MiB**. Le checkpoint FinBERT de 419 MiB était déjà présent ; il n'a pas été retéléchargé.

Collecte : `data/external/news-sentiment-rss-pilot-20261006/`.
Scores et exports : `data/derived/news-sentiment-rss-pilot-20261006/`.
Ces dossiers sont ignorés par Git.

Le checkpoint utilisé est `yiyanghkust/finbert-tone`, révision `4921590d3c0c3832c0efea24c8381ce0bda7844b`. Le checksum des poids et les métadonnées locales sont conservés dans le manifeste de scoring. Les 409 articles reçoivent 256 labels neutres, 95 positifs et 58 négatifs. Ces labels décrivent le texte ; ils ne sont pas des ordres Hold/Buy/Sell et leur distribution ne mesure pas une accuracy.

## Vérifications réelles

1. Six réponses RSS réussies, enregistrées avec leur checksum. Six premières tentatives bloquées par le sandbox sont aussi conservées dans l'audit, soit 12 tentatives au total. Aucun appel Alpha Vantage supplémentaire : le compteur partagé reste à 1.
2. Vérification des tables et de chaque preuve brute avant scoring. Les probabilités FinBERT, leur somme, les classes et les scores dérivés sont validés.
3. Un texte identique est scoré une seule fois ; ses associations conservent leurs dates propres. Les 394 associations entreprise et 15 macro sont exportées séparément.
4. Export de 23 canaux entreprise via le bridge existant. La macro reste un panel date seule, sans ticker tradable.
5. Reprise de la collecte : zéro requête HTTP. Reprise du scoring : zéro nouvelle inférence, cache validé.
6. Les 90 lignes séance/ticker de septembre ont zéro news dans leur fenêtre admissible et des masques faux. Les données de marché se terminent le 25 septembre ; elles ne sont pas prolongées artificiellement.
7. Un aperçu technique séparé vérifie deux bornes synthétiques, sans prix. Au 6 octobre à minuit : zéro article disponible. Au 7 octobre à minuit : 394 associations entreprise et 15 macro agrégées. La couverture reste `unknown` et les masques restent faux.

Les observations de collecte vont du 6 octobre à 19:34:05,692825 UTC à 19:34:08,004088 UTC. Les dates de publication annoncées vont du 8 avril au 6 octobre. Elles ne remplacent jamais l'observation réelle dans `available_at`.

Les 84 tests ciblés passent, dont 29 spécifiques RSS : HTML, Atom, dates, alias, absence de routage macro vers entreprise, limites, erreurs, checksums, reprise, versions révisées et exports. Les fixtures unitaires restent synthétiques ; les contrôles ci-dessus portent sur le corpus réellement téléchargé.

Suite complète : **1 132 tests passent, 11 sont ignorés**, en 105 secondes.
Les 41 avertissements concernent les chemins déjà existants : APIs dépréciées,
fragmentation pandas, configuration Transformer et parsing de dates. Aucun
test en échec ; `git diff --check` passe aussi.

## Reproduire la vérification

La collecte déjà terminée se vérifie sans réseau :

```shell
.venv/bin/python scripts/download_rss_news_sentiment_pilot.py \
  --output data/external/news-sentiment-rss-pilot-20261006 \
  --resume
```

Le [guide d'intégration](../src/news-sentiment-integration.md) fournit la commande complète du scoring et des exports. Une nouvelle collecte doit utiliser un nouveau dossier. La commande `--dry-run` ne fait aucune requête et n'écrit aucun fichier.

```shell
.venv/bin/python -m pytest -q \
  tests/test_rss_news_collection.py \
  tests/test_news_collection.py \
  tests/test_news_pilot_scoring.py \
  tests/test_news_sentiment_ablation.py
```

## Conclusion et suite

Le raccordement technique fonctionne avec le corpus réel et sans API payante. **Ce corpus ne permet pas encore le benchmark financier historique** : il manque une preuve admissible de couverture et de disponibilité sur les dates de train/validation/test. La réussite d'une requête RSS ne justifie pas `covered`, ni l'interprétation « aucune news = zéro ».

Il faut ensuite un corpus historique auditable ou une collecte prospective avec politique de couverture et de fraîcheur explicite, notamment pour écarter les anciennes publications du snapshot initial. Les cinq variantes sentiment peuvent alors être comparées sur des dates communes, à exposition comparable. Aucun ancien benchmark GRU/GNN/Transformer n'est modifié par ce pilote.
