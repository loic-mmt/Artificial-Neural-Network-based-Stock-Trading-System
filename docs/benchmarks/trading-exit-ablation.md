# Ablation des sorties trading : TP, durée, SL et trailing

## Protocole

Diagnostic du pilote `open_gap`, **fold 0, seed 1**, du 2011-02-14 au
2014-11-11 : 957 bougies par titre, 10 titres. Même prédiction figée, capital
initial 10 000, frais 5 bps, slippage 2 bps, OHLC ajustés des splits avec
dividendes séparés. Signal après observation de l'open, exécution à l'open
suivant. Les deux bras du précédent replay sont reproduits exactement.

Treize variantes fixées avant leur évaluation ; aucun entraînement ni recherche
de seuils. Holdout final fermé. SL 5 %, TP 10 %, trailing 4 % activé après gain
de 3 %, durée maximale 20 bougies. Ces seuils sont ceux de l'exemple initial.

Les variantes `full_without_*` conservent toutes les autres règles du pack.
Les variantes isolées suppriment TP/durée/break-even/bande sauf la règle nommée.
La politique de réentrée reste `new_signal`, sauf `sl_trailing_next_bar`.
Les limites d'exposition et conventions économiques restent communes.

## Résultats

Rendements **cumulés nets**, drawdowns négatifs. Exposition brute moyenne
mesurée aux clôtures. Le blocage compte les décisions titre/bougie avec cible
modèle non nulle et motif `reentry_block`, divisées par toutes les décisions.

| Variante | Rendement | Sharpe | Drawdown max. | Exposition brute moyenne | Décisions bloquées par réentrée |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sans règles | +4,407 % | 0,854 | −1,195 % | 6,081 % | 0,0 % |
| Pack initial | −0,056 % | −0,445 | −0,085 % | 0,100 % | 89,0 % |
| Pack sans TP | −0,062 % | −0,481 | −0,085 % | 0,101 % | 88,9 % |
| Pack sans durée maximale | −0,059 % | −0,428 | −0,082 % | 0,109 % | 88,5 % |
| Pack sans TP ni durée maximale | −0,064 % | −0,461 | −0,082 % | 0,110 % | 88,5 % |
| TP seul | +2,619 % | 0,810 | −0,847 % | 3,527 % | 35,5 % |
| Durée maximale seule | −0,292 % | −0,888 | −0,334 % | 0,199 % | 86,6 % |
| Bande sans ordre seule | +4,433 % | 0,868 | −1,167 % | 6,074 % | 0,0 % |
| SL seul | −0,200 % | −0,955 | −0,214 % | 0,269 % | 82,7 % |
| Trailing seul | +0,682 % | 0,595 | −0,321 % | 1,321 % | 67,4 % |
| SL + trailing | −0,041 % | −0,298 | −0,071 % | 0,104 % | 90,5 % |
| SL + trailing avec bande initiale | −0,055 % | −0,379 | −0,085 % | 0,119 % | 88,1 % |
| SL + trailing, réentrée dès bougie suivante | +2,125 % | 0,463 | −1,302 % | 5,534 % | 1,4 % |

## Interprétation

- Retirer TP et durée du pack initial laisse le portefeuille presque flat.
  D'autres protections provoquent encore des sorties et leurs verrous de
  réentrée. Le diagnostic ne soutient donc pas l'hypothèse selon laquelle
  retirer seulement ces deux règles suffirait à restaurer le rendement.
- La durée maximale seule pénalise ce replay. Le TP seul réduit rendement et
  drawdown ; son effet ne se résume pas à une perte immédiate.
- Avec SL + trailing, le verrou `new_signal` bloque 90,5 % des décisions. Les
  cibles continues peuvent conserver leur signe longtemps : une variation
  d'amplitude ne suffit pas à débloquer une entrée selon le contrat du module.
- Changer uniquement cette politique en `next_bar` restaure largement
  l'exposition, mais produit 871 sorties SL/trailing, contre 48 avec le verrou.
  Les frais passent de 1,56 à 63,60, contre 12,01 sans règles. Rendement et
  Sharpe restent inférieurs au replay sans règles ; le drawdown devient plus
  profond. Réentrée rapide ne résout donc pas à elle seule le compromis.

Ces résultats concernent **un seul fold/seed et des seuils fixes**. Aucun profil
n'est promu comme configuration optimale. L'interaction sorties/réentrée doit
être évaluée sur les autres folds et seeds avant sélection. Une comparaison
de cooldown ou de distances différentes relève d'une nouvelle expérience
déclarée, avec protocole de validation et holdout maintenu fermé.

## Références scientifiques

Kaminski et Lo étudient comment une politique de stop peut ajouter ou retirer
de la valeur suivant le processus de prix et la fréquence de décision ; leurs
résultats ne garantissent pas l'amélioration d'un modèle donné.
[When do stop-loss rules stop losses?](https://dspace.mit.edu/entities/publication/bb69ca4b-0cdc-487f-831d-63b2e84fafee)

Leung et Zhang montrent l'optimalité d'une sortie par limite de vente associée
au trailing sous les hypothèses de leur problème de liquidation. Le papier
illustre qu'un TP peut être utile dans certains modèles de prix ; il ne teste
pas notre GRU, nos seuils ou notre politique de réentrée.
[Optimal Trading with a Trailing Stop](https://arxiv.org/abs/1701.03960)

## Reproduction et journaux

Dossier : `artifacts/trading/fold-0-seed-1-exit-ablation/`.

- `run.py` : script autonome de reproduction depuis la racine du dépôt, avec
  `PYTHONPATH=src .venv/bin/python <nouveau_dossier>/run.py` ; copier le script
  dans un nouveau dossier pour préserver les résultats existants.
- `manifest.json` : variantes déclarées avant replay, empreintes des entrées
  et du code, versions Python/NumPy/Pandas, dates et garde du holdout.
- `report.json`, `comparison.csv`, `summary.md` : toutes les variantes, sans
  sélection des seuls résultats favorables.
- Chaque sous-dossier contient `config.json` et journaux complets
  equity/positions/orders/trades/decisions, plus métriques et métadonnées.

Les configurations sauvegardées sont directement compatibles avec
`scripts/run_trading_backtest.py --trading-config`. Pour reproduire seulement
SL + trailing, utiliser le `config.json` du sous-dossier `sl_trailing` ; pour
sa variante de réentrée, celui de `sl_trailing_next_bar`.
