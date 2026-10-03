# Validation du patch OU / trailing

Comparaison de prédictions figées du pilote post-open : **18 groupes**, 3 folds × 3 seeds × 2 familles, **180 replays**. Holdout final **fermé**.

Protocole fixé avant scores : trailing 4 % immédiat, TP fixe comparatif 10 %, fenêtres OU 252 bougies, refit 21, discount annuel 0,05 ; frais/slippage 5/2 bps puis 15/6 bps. Réentrée `next_bar`, bande sans ordre nulle. Paramètres expérimentaux, aucune optimisation.

Équations reproduites depuis [Leung–Zhang, version 2](https://arxiv.org/html/1701.03960v2) : benchmark achat **1,94883073**, vente **2,88446314**. L’optimalité du problème continu à une unité ne garantit pas la performance de cette adaptation portefeuille.

## Résultats aux coûts de base

Médianes de 9 replays séparés par famille. Rendement net total de chaque fold, pas rendement annualisé ou rendement d’un portefeuille assemblé. Seeds et familles partagent les prix : ce ne sont pas 18 histoires indépendantes.

### open_gap

| Variante | Rendement net | Sharpe | Drawdown | Exposition brute moyenne | Frais médians |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sans règles | +6.53 % | 0.762 | -2.72 % | 10.73 % | 14.89 |
| Trailing seul | +1.94 % | 0.244 | -2.90 % | 9.50 % | 130.92 |
| Trailing + TP fixe | +1.60 % | 0.243 | -2.73 % | 9.35 % | 144.47 |
| Trailing + TP OU | +1.91 % | 0.257 | -2.61 % | 9.34 % | 130.88 |
| TP OU + achat/réentrée OU | -0.05 % | -0.130 | -0.26 % | 0.18 % | 3.11 |

### lagged

| Variante | Rendement net | Sharpe | Drawdown | Exposition brute moyenne | Frais médians |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sans règles | +7.38 % | 0.777 | -4.58 % | 15.40 % | 17.38 |
| Trailing seul | +1.26 % | 0.317 | -3.84 % | 14.01 % | 151.09 |
| Trailing + TP fixe | +1.51 % | 0.264 | -4.26 % | 13.76 % | 166.41 |
| Trailing + TP OU | +1.65 % | 0.325 | -3.82 % | 13.34 % | 151.55 |
| TP OU + achat/réentrée OU | -0.08 % | -0.212 | -0.24 % | 0.13 % | 2.46 |

## Coûts multipliés par trois

| Variante | open_gap : net médian | lagged : net médian |
| --- | ---: | ---: |
| Sans règles | +6.08 % | +7.02 % |
| Trailing seul | -1.69 % | -1.18 % |
| Trailing + TP fixe | -1.87 % | -3.76 % |
| Trailing + TP OU | -1.44 % | -1.56 % |
| TP OU + achat/réentrée OU | -0.06 % | -0.12 % |

## Effet apparié de TP OU contre trailing seul

| Famille / coûts | Gain net médian apparié | Replays avec gain |
| --- | ---: | ---: |
| lagged / base_costs | +0.01 points | 6/9 |
| lagged / costs_x3 | -0.02 points | 4/9 |
| open_gap / base_costs | +0.08 points | 6/9 |
| open_gap / costs_x3 | +0.06 points | 5/9 |

## Diagnostic et interprétation

- Fit OU valide sur **8,68 %** des lignes de décision en médiane ; principalement rejet de racine unitaire / incertitude. Les diagnostics dépendent des prix, donc se répètent entre seeds.
- Achat/réentrée OU bloque **99,77 %** des signaux longs pour `open_gap`, **99,74 %** pour `lagged`, en médiane. Exposition restante très faible ; les shorts gardent leur logique existante.
- TP OU produit 344 sorties calculées pour `open_gap`, 370 pour `lagged`, aux coûts de base. Ce sont des épisodes des différents replays, pas des observations indépendantes.
- TP OU + trailing ne montre pas d’avantage net robuste contre trailing seul ; les coûts ×3 rendent négatives les médianes des deux variantes. Sans règles conserve ici le rendement et Sharpe médians les plus élevés.
- Aucune politique promue. Module et politique OU restent désactivés par défaut. Un filtre quasi toujours flat ne valide pas la technique du papier sur ces signaux.

## Causalité, vérification et reproductibilité

- Replay au prochain open exécutable après disponibilité du signal. Le pilote a appris un rendement au proxy open courant → suivant : le décalage d’horizon demeure ; des prix intraday seraient nécessaires pour un fill après observation du même open.
- Warmup limité aux 253 bougies précédant chaque fold. Fits uniquement sur closes disponibles avant l’action ; TP figé à entrée ; trailing resserré après clôture ; achats limités aux prix d’action, sans entrée intrabougie inventée.
- Comptabilité commune : cash + quantités signées, coûts sur montant échangé, dividendes séparés, splits sans double ajustement. Identiques dans tous les bras.
- `manifest.json` : configurations, empreintes code/données/prédictions et statut complet. `code-snapshot/` conserve exactement le code utilisé pour ces 180 replays.
- Après correction ciblée de bande sans ordre pour petites entrées, `post-patch-verification.json` compare 10 replays représentatifs : mêmes métriques et mêmes cinq journaux, bit pour bit. Toutes les ablations utilisent une bande nulle et sont hors du cas corrigé ; les empreintes du code final sont également sauvegardées.
- `comparison.csv`, `paired.csv`, `summary.csv` : tous groupes/scénarios/variantes. Chaque dossier conserve equity, positions, orders, trades et decisions Parquet, métriques et diagnostics JSON.
- Tests synthétiques : benchmark/convergence/coûts, OU stable / trend / bruit, future mutation, splits/dividendes, gates/gaps, absence de réentrée intrabougie, signaux retardés, bande/caps et gardes holdout.

Commande de reproduction depuis racine du projet (répertoire de sortie neuf) :

```bash
.venv/bin/python scripts/validate_trading_ou.py \
  --signals-dir artifacts/comparisons/gru-optim/11-post-open-pilot \
  --data data/processed/cac40_daily_clean.parquet \
  --output-dir artifacts/trading/ou-validation-reproduced --workers 3
```
