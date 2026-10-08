# Qualité des labels US, sans entraînement

## Question

Il faut déterminer si les labels produisent trop de changements courts ou de
petits mouvements économiques. Le diagnostic compare les durées, rendements,
transitions, turnover et fractions de faibles amplitudes. Aucun GRU n'est entraîné.

## Protocole

Exécution du 8 octobre 2026 sur les 143 actions du fichier
`data/data_pc/processed/mt5_stocks_us_daily_clean.parquet` : 664 664 séances-tickers
du 3 janvier 2005 au 21 juin 2023. Le holdout final reste fermé. Les prix après
cette borne sont exclus avant de générer les cibles.

Les sondes directionnelles utilisent 660 374 lignes communes aux cinq méthodes.
Les rendements natifs sont propres à chaque cible, avec leurs horizons distincts.
Les agrégats sont pondérés par observations, pas par moyenne simple des tickers.

Configuration :

- Breakout : fenêtre 20, signaux alternés.
- Forward return : horizon 10, seuils Buy/Sell 0,2 %.
- Volatility position : horizon 10, volatilité 20, seuils Long 1 / Short 1,5,
  sortie 0,25, maintien minimum 5, long/short.
- Triple barrier : ATR 20, horizon maximum 10, profit/stop 0,75,
  CUSUM 0,5, entre événements Hold.
- Intraday return : Long si close J > open J, Short si inverse, Flat si égalité.

Coûts du diagnostic : 5 bps par sens, soit 10 bps entrée + sortie.
Les labels futurs sont rétrospectifs. Ces chiffres ne constituent pas une
performance réalisable.

## Durées et turnover

| Méthode | Couleur directionnelle min / moy. / max | Position décodée min / moy. / max | Événement natif moy. | Changements de label | Turnover / séance |
| --- | --- | --- | ---: | ---: | ---: |
| Breakout | 1 / 1 / 1 | 1 / 30,70 / 261 | non défini | 6,47 % | 0,065 |
| Forward return | 1 / 6,67 / 85 | 1 / 7,84 / 101 | 10 | 17,77 % | 0,254 |
| Volatility position | 1 / 11,24 / 74 | 1 / 11,24 / 74 | 10 | 8,06 % | 0,085 |
| Triple barrier | 1 / 1,67 / 24 | 1 / 5,20 / 58 | 3,22 | 61,45 % | 0,382 |
| Intraday return | 1 / 1,97 / 19 | 1 / 1 / 1 | 1 | 51,17 % | 1,986 |

Les changements de label incluent les transitions vers et depuis Hold/Flat,
pas seulement les retournements Long/Short. Le turnover mesure entrée/sortie
en unités de notionnel, pas un pourcentage de capital d'un portefeuille.

Buy suivi de Hold ne ferme pas une position pour les méthodes d'action.
La couleur d'une séance sur le graphique n'est donc pas une détention d'une
séance. Intraday ferme toujours à la clôture, même si la couleur suivante est
encore Long.

## Rendements natifs et faible amplitude

| Méthode | Rendement brut natif min / moy. / max | Médiane | < 0,2 % | < 1 % | < 1,5 % |
| --- | --- | ---: | ---: | ---: | ---: |
| Breakout | non défini | non défini | non défini | non défini | non défini |
| Forward return | 0,20 / 4,40 / 1641,90 % | 3,15 % | 0 % | 15,36 % | 24,42 % |
| Volatility position | -44,73 / 5,67 / 1641,90 % | 4,73 % | 0,48 % | 3,48 % | 7,69 % |
| Triple barrier | 0,45 / 3,07 / 134,84 % | 2,45 % | 0 % | 1,69 % | 15,16 % |
| Intraday return | 0,0003 / 1,16 / 105,17 % | 0,79 % | 13,87 % | 59,01 % | 74,77 % |

Il ne faut pas classer les méthodes sur ces moyennes : les horizons, la présence
overnight et le chevauchement des événements diffèrent. La direction correcte
du forward-return sur son propre horizon est garantie par construction, pas par
un modèle capable de la prédire.

Les valeurs extrêmes sont conservées. Le maximum à dix séances provient de GME
du 12 au 27 janvier 2021. Le décodage breakout contient aussi un short GME avec
perte brute de 816,25 % : le diagnostic utilise un notionnel initial fixe, sans
simulation de marge ni liquidation. Une perte short peut dépasser 100 %.
Les médianes doivent accompagner les moyennes.

## Conclusions

- La fragmentation triple-barrier est réelle : 63,61 % des séquences de couleurs,
  toutes classes confondues, ne durent qu'une séance. Cela ne correspond pas à
  63,61 % de trades d'une séance.
- Les événements triple-barrier ne sont pas principalement des mouvements de
  0,2 % : minimum 0,448 %, médiane 2,453 %, aucun événement directionnel sous
  0,2 %. Un filtre à 1,5 % toucherait néanmoins 15,16 % des événements.
- Le label open-to-close demandé produit davantage de petits mouvements et
  d'entrées/sorties quotidiennes. Son turnover est plus de cinq fois celui du
  décodage triple-barrier ; ce n'est pas une preuve qu'il serait moins apprenable.
- Volatility-position fournit des cibles nettement plus persistantes. Cette
  propriété peut servir de référence pour une analyse de stabilité, sans
  conclure qu'un GRU entraîné dessus serait meilleur.
- Une séquence visuellement irrégulière ne prouve ni aléatoire ni inapprenabilité.
  Il reste à étudier le lien avec les features disponibles à la décision et,
  séparément, à aligner les cibles avec le protocole après l'open.
- Les sondes d'opportunités sont directionnelles : Hold y est neutre, alors
  qu'il conserve la position décodée. Elles ne comptent pas des trades réellement
  ratés par le modèle.

## Reproduction et exports

```bash
.venv/bin/python scripts/analyze_labels.py \
  --data data/data_pc/processed/mt5_stocks_us_daily_clean.parquet \
  --end 2023-06-21
```

L'absence de `--output-dir` crée un dossier daté et conserve le run précédent.
Résultats de cette exécution :
`artifacts/diagnostics/labels-us-development-20261008`.

Le [rapport détaillé local](../../artifacts/diagnostics/labels-us-development-20261008/report.md)
contient les tableaux ; les CSV/JSON/Parquet conservent les statistiques globales
et par ticker, les durées, rendements, inconnus, censures et transitions.
Le [manuel du diagnostic](../src/label-diagnostics.md) précise les conventions.
