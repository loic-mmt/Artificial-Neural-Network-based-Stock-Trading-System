# Trailing et frontières OU : adaptation de Leung–Zhang

Implémentation optionnelle de [*Optimal Trading with a Trailing Stop*,
Leung et Zhang, version 2](https://arxiv.org/html/1701.03960v2). Couche après
prédiction ; entraînement, architecture et sélection du modèle inchangés.

## Ce qui est reproduit

Pour une unité **longue**, le papier maximise une valeur de liquidation
actualisée, avec trailing obligatoire, puis résout un problème d'acquisition.
Le prix suit une diffusion exponentielle OU :

```text
Y = log(X)
dY = lambda * (theta - Y) dt + sigma dW
floor(M) = (1 - alpha) * M
```

`solve_ou_boundaries` utilise les solutions fondamentales OU en fonctions
cylindriques paraboliques, l'équation de frontière de vente (21), puis l'ODE
de valeur sur la diagonale (29–30). La frontière d'achat maximise la valeur
d'acquisition transformée dans la géométrie à seuil inférieur de l'exemple OU.
Les prix sont normalisés par `exp(theta)` pour respecter les changements d'unité.
La solution croissante utilise `D_nu(-z)`, la décroissante `D_nu(z)`.
`acquisition_value` est le maximum transformé `(v-h_buy)/phi_minus`, remis
en unité de prix de référence ; ce n'est ni un rendement ni un PnL portefeuille.

Benchmark indépendant du réseau : `lambda=0.6`, `theta=1`, `sigma=0.2`,
actualisation `q=0.05`, trailing `alpha=0.3`, coûts fixes achat/vente `0.02`.
Résultat : **achat 1.94883073, vente 2.88446314**, correspondant aux valeurs
1.9488 et 2.8845 du papier. Tests : convergence en tolérance/domaine, invariance
par changement d'unité, sensibilité aux coûts et absence d'achat profitable.
Ces valeurs servent uniquement à vérifier les équations.

L'intégration emploie les coûts **proportionnels** du moteur, avec
`h_buy(x)=(1+fee)*(1+slip)*x`, `h_sell(x)=(1-fee)*(1-slip)*x`.
Des bps ne sont jamais convertis en coûts fixes en euros. L'API scientifique
`AffineCosts` permet également les coûts fixes du benchmark.

## Activation

```python
from trading_system.trading import OUConfig, TradingConfig, run_trading_backtest

config = TradingConfig(
    enabled=True,
    execution="next_open",
    signal_timing="after_open",
    fees_bps=5, slippage_bps=2,
    trailing_stop_pct=0.04,
    trailing_activation_pct=0,
    reentry="next_bar",
    ou=OUConfig(mode="exit", window=252, refit_every=21),
)
result = run_trading_backtest(prix_validation, predictions_figees, config,
                              history=prix_avant_validation)
```

Valeurs illustratives fixées avant validation, sans optimisation ni promesse
de rendement. `OUConfig()` et `TradingConfig()` gardent cette politique éteinte.
Le module exige un trailing en pourcentage actif immédiatement. Un TP fixe et
un TP OU sont exclusifs ; le proxy à l'open observé est refusé pour cette politique.

| Mode | Nouvelle position longue | Réentrée longue après protection |
| --- | --- | --- |
| `off` | Logique existante | `reentry` existant |
| `exit` | Signal modèle, TP OU si fit valide ; trailing seul sinon | `reentry` existant |
| `entry_exit` | Signal modèle positif **et** prix exécutable dans région d'achat profitable | Frontière d'achat remplace verrou même sens ; cooldown éventuel conservé |

Une nouvelle entrée au-dessus de la frontière de vente est refusée pour éviter
un achat suivi d'une liquidation artificielle immédiate. En `entry_exit`, fit
invalide ou absence de région profitable bloque la nouvelle position longue.
Aucune réentrée dans la bougie d'une sortie forcée, même sur signal opposé.
Une amplitude différente du signal ne contourne pas cette interdiction.

Entrées par limite : seuil connu à partir des bougies terminées, puis vérifié
au **prochain open/close exécutable**. Gap au-dessus du seuil d'achat : ordre
non exécuté. Gap en dessous : prix open/close utilisé, frais/slippage ensuite.
Les entrées intrabougie au contact du low ne sont pas simulées. Ce choix est une
discrétisation conservatrice de la surveillance continue du papier.

À entrée, fit/version/cutoff et TP sont figés jusqu'à fermeture ; les nouveaux
fits ne déplacent pas ce TP. Redimensionnement conserve ancre et niveaux.
Trailing mis à jour après bougie terminée, applicable à suivante. Le trailing OU
se réfère au prix de marché avant coûts ; l'ancre de PnL conserve le prix du fill.
Splits bruts divisent également les deux frontières figées, sans double ajustement.
Les quantités, frais et dividendes restent ceux du moteur commun.

Signal flat/opposé, stops, durée, événements et drawdown restent prioritaires.
La bande sans ordre ne retient pas une position face à flat/opposé sous politique
OU. Le modèle décide toujours direction et taille. **Shorts : logique existante
SL/trailing/signaux ; aucune frontière OU obtenue par symétrie artificielle.**

## Fit causal et critères de rejet

`estimate_ou` ajuste la transition AR(1) exacte de `log(close)` :

```text
Y_next = a + b * Y + epsilon
lambda = -log(b) * annualization
theta = a / (1-b)
sigma = sqrt(var(epsilon) * 2*lambda / (1-b*b))
```

Horloge : **bougies observées**, `annualization` bougies/an. En intraday,
spécifier le nombre correspondant ; les gaps de nuits/weekends ne deviennent
pas implicitement des pas temporels supplémentaires. Les splits bruts sont
neutralisés avec les seules actions déjà observées. Les dividendes ne sont pas
retirés de la série de prix ; un détachement peut donc peser sur le diagnostic.

| Paramètre `ou` | Défaut | Sens |
| --- | --- | --- |
| `mode` | `off` | Activation indépendante du module global |
| `window` | 252 | Nombre de closes terminées, minimum 40 |
| `refit_every` | 21 | Fréquence en bougies, plus passage de warmup à fenêtre complète |
| `discount_rate` | 0.05 | Actualisation subjective annuelle, distincte du taux sans risque |
| `adf_pvalue` | 0.05 | Seuil ADF avec constante, zéro lag |
| `max_half_life_fraction` | 0.25 | Demi-vie maximale en fraction de fenêtre |
| `stability_sigma` | 2 | Écart maximal entre moyennes OU des deux demi-fenêtres, en écart-type stationnaire |

Rejet si fenêtre insuffisante, prix constants, `b` hors `(0,1)`, variance nulle,
borne supérieure d'IC OLS à 95 % atteignant 1, ADF non significatif, demi-vie
trop longue, fits des demi-fenêtres invalides ou moyenne instable. IC OLS
approximatif : biais de petit échantillon et incertitude du modèle persistent.
Rejet numérique si les fondamentales, la racine ou l'ODE échouent, si le maximum
d'acquisition touche le bord du domaine ou n'a pas une géométrie simple. Aucun
seuil arbitraire n'est inventé pour remplacer une solution non convergée.

Chaque fit utilise uniquement les closes dont disponibilité précède strictement
l'action. Même à `next_close`, la close servant d'exécution n'est pas utilisée
pour définir sa propre limite. Modifier des OHLC futurs ne change pas les
décisions antérieures. `history` doit avoir le même univers et finir strictement
avant le début du replay ; il initialise le fit, sans simuler des trades passés.

Ce sont des critères de diagnostic fixés avant l'évaluation, pas une preuve
que l'actif suit OU. Toute calibration de fenêtres/seuils supplémentaire doit
rester sur train/validation ; holdout scellé.

## CLI, journaux et comparaison reproductible

```bash
.venv/bin/python scripts/run_trading_backtest.py \
  --signals artifacts/comparisons/gru-optim/11-post-open-pilot/fold-0-seed-1-open_gap-positions.parquet \
  --data data/processed/cac40_daily_clean.parquet \
  --ou-history data/processed/cac40_daily_clean.parquet \
  --trading-config configs/trading/ou_exit.json \
  --output-dir artifacts/trading/mon-replay-ou
```

Pour achat/réentrée, remplacer `ou.mode` par `entry_exit`. Chaque groupe reçoit
uniquement l'historique antérieur à ses prédictions. Les rapports habituels
incluent empreinte de cet historique, et `metadata.ou_policy.fit_records` contient
cutoff, disponibilité, version, paramètres, IC, demi-vie, ADF, stabilité, motifs
de rejet et frontières en unité de référence.

`decisions` contient fit/version/cutoff, validité, motif et frontières courantes
en unité du titre ; `orders`/`trades` contiennent les frontières **figées de
l'épisode**. Une sortie TP calculée porte motif `ou_take_profit`. `positions`
montre les protections actives. Une différence entre seuil de décision courant
et TP d'un épisode déjà ouvert reflète leur gel à entrée.

```bash
.venv/bin/python scripts/validate_trading_ou.py \
  --signals-dir artifacts/comparisons/gru-optim/11-post-open-pilot \
  --data data/processed/cac40_daily_clean.parquet \
  --output-dir artifacts/trading/ou-validation-verified \
  --workers 3
```

Protocole figé : sans règles, trailing seul, trailing + TP fixe 10 %, trailing
+ TP OU, trailing + TP OU + achat/réentrée OU. Trailing 4 % immédiat, réentrée
`next_bar`, frais 5 bps et slippage 2 bps ; second scénario multiplie ces coûts
par trois. Même moteur, timing, prix et signal pour les cinq bras. Les deux
familles du pilote (`open_gap`, `lagged`), tous folds/seeds disponibles sont
séparés ; chaque replay recommence à capital identique et termine liquidé.
Seules 253 bougies précédant le début de chaque fold servent de warmup.

`manifest.json` sauvegarde configurations avant scores, empreintes données,
code et signaux, benchmark et statut ; `comparison.csv` détaille chaque replay,
`paired.csv` les écarts appariés, `summary.csv` les médianes et étendues par
famille/scénario. Sharpe médian décrit des replays séparés, pas un portefeuille
assemblant les folds. Journaux complets sauvegardés pour chaque bras. Pas de
recherche de combinaison gagnante ; script refuse toute prédiction à partir
du 2022-05-10 et ne propose pas d'ouverture automatique du holdout.

## Limites de l'adaptation

Le papier suppose une diffusion connue, continue, une seule unité et un objectif
de valeur actualisée. Ici les paramètres sont estimés, changent entre épisodes,
les signaux modifient direction/taille, il existe un portefeuille, des gaps et
des coûts supplémentaires. Répéter les transactions et dimensionner selon le
réseau constitue une **extension empirique**, sans garantie d'optimalité globale.
Dividendes futurs, financement et contraintes broker ne figurent pas dans la
valeur théorique ; seuls les dividendes observés sont comptabilisés séparément.

La surveillance des TP/stops via OHLC conserve les limites du moteur : conflit
TP–stop choisi au stop ; aucune trajectoire supposée entre high et low ; trailing
resserré seulement après clôture. Les résultats sont ceux de cette simulation,
pas ceux d'un trailing surveillé continuellement.

Le pilote post-open a appris un rendement open courant → open suivant sous proxy.
Le replay causal `next_open` retarde ce signal ; un vrai fill après observation
de l'open demande des prix intraday. Cette validation mesure l'effet de la
politique sous timing commun, sans corriger le décalage d'horizon du modèle.

Résultats des 180 replays et stress de coûts :
[rapport de validation](../benchmarks/trading-ou-validation.md). Code exact du replay conservé
dans `artifacts/trading/ou-validation-verified/code-snapshot/` ; correction
ultérieure ciblée de bande vérifiée sur 10 replays identiques, sans incidence
sur les ablations à bande nulle. Tests du module et régressions ciblées : **104 passent**.
