# BCE RTD : archive historique de versions, pas encore des features PIT

Le 25 septembre 2026, les réponses `includeHistory=true` de l'API officielle
de la [Real Time Database de la BCE](https://data.ecb.europa.eu/data/datasets/RTD/data-information)
ont été enregistrées dans `data/processed/ecb_rtd_vintages_raw.parquet`.
L'archive contient 12 458 lignes :

| Série | Indicateur | Lignes | Dates de version |
| --- | --- | ---: | ---: |
| `RTD.M.S0.N.P_C_OV.X` | Indice HICP zone euro | 2 625 | 258 |
| `RTD.Q.S0.S.G_GDPM_TO_C.E` | PIB réel zone euro | 9 833 | 205 |

Les périodes d'observation couvertes vont de `1990-01` à `2026-08` pour
l'HICP et de `2000-Q1` à `2022-Q2` pour le PIB. Pour éviter les expirations de
requête, la seconde série a été récupérée en deux tranches, `2000-Q1` à
`2009-Q4` et `2010-Q1` à `2022-Q2`, puis réunie sans chevauchement.

Chaque ligne garde `database_valid_from_utc`, `database_valid_to_utc`,
`action`, période d'observation, valeur, clé de série, URL de source et empreinte
du contenu traité. Les actions `Delete` et les valeurs manquantes sont
préservées. **Il n'y a volontairement pas de colonne `available_at_utc`.**
Selon la documentation BCE, les bases servant à RTD sont figées un jour ouvré
avant la réunion du Conseil des gouverneurs. Le début de validité d'une
version dans la base n'établit donc pas à lui seul son heure de publication
au public. Le fichier n'est pas admis dans le benchmark tant qu'un calendrier
de publication n'a pas fourni un `available_at_utc` conservateur par version.

Le téléchargement est reproductible avec :

```bash
.venv/bin/python scripts/download_ecb_rtd_vintages.py \
  --output data/processed/ecb_rtd_vintages_raw_new.parquet
```

Le script refuse d'écraser un fichier existant. Les deux séries sont les
valeurs par défaut ; `--series` permet de n'en choisir qu'une. Le paramètre
`--input-dir` importe des CSV déjà récupérés sous les noms
`euro_area_hicp_index.csv` et `euro_area_real_gdp.csv`.

La clé `FRED_API_KEY` n'était pas configurée lors de cette récupération.
[ALFRED](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
reste une seconde source possible, notamment pour des séries américaines,
mais demande une clé API et une validation série par série des dates de
première publication.
