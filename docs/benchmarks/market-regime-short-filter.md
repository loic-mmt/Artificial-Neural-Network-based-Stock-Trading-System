# Filtrage des positions SHORT par regime de marche

## Objectif

Verifier si les positions SHORT du GRU peuvent etre filtrees par un regime de
marche global, calcule sans lookahead, plutot que d'inverser les predictions
apres des pertes recentes.

Le test reutilise les predictions sauvegardees dans :

```text
artifacts/mt5/benchmarks/retraining-frequency-confidence/
```

Les frequences `every 5`, `every 10` et `every 20`, avec la seed 42 et le
decodeur de confiance, sont comparees sur les 93 seances du holdout US.

## Construction causale du regime

Le regime de la seance J utilise uniquement les informations connues a la
cloture de J-1 :

- ecart entre les EMA 20 et 60 de l'indice de marche ;
- momentum de l'indice sur 20 seances ;
- proportion des actions au-dessus de leur moyenne mobile 60 seances ;
- niveau du VIX par rapport a son quantile glissant a 70 %.

Le score applique est le suivant :

| Condition | Score |
|---|---:|
| EMA20 superieure a EMA60 de plus de 0,2 % | +1 |
| EMA20 inferieure a EMA60 de plus de 0,2 % | -1 |
| Momentum marche 20 seances positif | +1 |
| Momentum marche 20 seances negatif | -1 |
| Breadth superieure a 55 % | +1 |
| Breadth inferieure a 45 % | -1 |
| VIX au-dessus de son quantile 70 % avec momentum negatif | -1 |

Le marche est considere baissier lorsque le score est inferieur ou egal a
`-2`. La politique testee conserve toutes les positions LONG et transforme une
position SHORT en FLAT lorsque le marche n'est pas baissier.

## Resultats de la regle stricte

| Frequence | PnL original | PnL filtre | Gain | Sharpe original | Sharpe filtre | Drawdown original | Drawdown filtre |
|---|---:|---:|---:|---:|---:|---:|---:|
| Every 5 | 254,49 | 369,98 | +115,48 | 2,517 | 4,341 | -0,82 % | -0,56 % |
| Every 10 | 299,44 | 461,11 | +161,67 | 2,868 | 4,900 | -0,79 % | -0,43 % |
| Every 20 | 326,41 | 366,74 | +40,33 | 2,656 | 2,911 | -2,13 % | -1,98 % |

La periode ne contient aucune seance classee baissiere par cette definition.
La regle supprime donc tous les SHORT et produit exactement le resultat
long-only. Elle montre que retirer les SHORT aurait aide, mais ne valide pas la
capacite du classifieur de regime a identifier les bonnes periodes de vente a
decouvert.

## Variante plus reactive

Une seconde variante autorise les SHORT des que le momentum marche sur 20
seances devient negatif. Elle identifie 28 seances sur le holdout.

| Frequence | PnL original | PnL avec filtre momentum | PnL long-only | Contribution des SHORT autorises face au long-only |
|---|---:|---:|---:|---:|
| Every 5 | 254,49 | 324,90 | 369,98 | -45,08 |
| Every 10 | 299,44 | 447,63 | 461,11 | -13,48 |
| Every 20 | 326,41 | 363,46 | 366,74 | -3,28 |

Les SHORT restent donc defavorables meme pendant les seances identifiees par
un momentum 20 jours negatif.

Les variantes suivantes ont donne la meme conclusion :

- momentum negatif sur 5 ou 10 seances ;
- indice sous sa moyenne mobile 20 seances ;
- indice sous sa moyenne mobile avec VIX eleve ;
- momentum negatif avec breadth inferieure a 50 %.

## Conclusion

Le filtrage ameliore le benchmark actuel parce qu'il retire des SHORT
deficitaires. Le holdout est cependant trop haussier et trop court pour valider
une strategie `short_bear_only`.

La comparaison a conserver pour un benchmark ulterieur est :

1. `long_short` sans filtre ;
2. `long_only` ;
3. `short_bear_only` ;
4. les memes variantes sur une periode contenant un vrai marche baissier,
   notamment 2022, 2020 ou une autre correction importante ;
5. confirmation sur plusieurs seeds avant toute decision de production.
