# Archive ALFRED, données macro américaines

La commande suivante télécharge les versions historiques accessibles via l'API FRED/ALFRED :

```bash
.venv/bin/python scripts/download_fred_vintages.py
```

Elle lit `FRED_API_KEY` dans l'environnement ou dans `.env`. Elle refuse d'écraser un fichier existant. Le résultat brut est `data/processed/fred_vintages_raw.parquet`, ignoré par Git. Utiliser `--output` pour une nouvelle extraction et `--start`, `--end` ou `--series` pour changer le périmètre.

Archive téléchargée le 25 septembre 2026, observations à partir de janvier 2000 :

| Série | Information | Versions d'observation |
| --- | --- | ---: |
| `CPIAUCSL` | Indice des prix américain | 1 587 |
| `UNRATE` | Taux de chômage américain | 596 |
| `PAYEMS` | Emploi salarié non agricole | 4 114 |
| `INDPRO` | Production industrielle | 5 473 |

Total : 11 770 lignes. Chaque ligne conserve la date d'observation, la valeur et l'intervalle `realtime_start` / `realtime_end` fourni par ALFRED. La collecte est découpée en fenêtres de quatre ans pour respecter la limite de l'API, puis les fragments contigus artificiels sont réunis. `realtime_end=9999-12-31` signifie que la version est toujours ouverte. Deux observations ont une valeur manquante dans la source. Les quatre séries vont jusqu'aux observations d'août 2026 dans cette extraction.

Ce fichier n'est **pas encore une feature prête pour le benchmark**. ALFRED fournit ici une date, pas l'heure exacte à laquelle le marché a pu lire l'annonce. Un futur adaptateur doit définir explicitement la disponibilité, au plus tôt à la séance suivante si aucune heure de publication vérifiée n'est utilisée, puis faire une jointure as-of par date de décision. Ne jamais joindre directement sur `observation_date` ni utiliser les valeurs actuelles révisées pour le passé.

`DGS2` existe sur FRED mais l'API a répondu qu'il n'existe pas dans ALFRED pour la plage historique demandée. Il a donc été écarté, de même que la pente `T10Y2Y` non vérifiée comme archive. Les taux déjà présents dans les données du projet restent une source distincte à auditer.

Documentation officielle : [observations et types de sortie](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [intervalles en temps réel](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).
