# Module de trading optionnel

Le package `trading_system.trading` transforme des **prédictions figées** en
positions et transactions. Il fonctionne avec tous les modèles produisant des
positions, probabilités ou labels compatibles avec les décodeurs du projet.
L'entraînement, les losses financières et les backtests historiques conservent
leurs interfaces et conventions. Ce module est une API de simulation Python ;
il n'envoie pas d'ordres à un broker.

## API et activation

```python
from trading_system.trading import TradingConfig, run_trading_backtest

# ohlc : une ligne par date/ticker ; signaux : positions continues [-1, 1].
config = TradingConfig(
    enabled=True,
    execution="next_open",
    signal_timing="after_close",
    price_basis="split_adjusted",
    allocation_mode="equal_active",
    target_gross_exposure=0.98,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    trailing_stop_pct=0.04,
    trailing_activation_pct=0.03,
)
result = run_trading_backtest(ohlc, signaux, config)
print(result.metrics)
```

Ces seuils illustrent l'API ; ils ne sont ni optimisés ni recommandés comme une
stratégie rentable. `TradingConfig()` désactive les règles. Avec
`enabled=False`, les coûts, l'exécution, la comptabilité et les actions
d'entreprise restent actifs : seules les règles optionnelles sont ignorées.
Les deux bras d'une comparaison passent donc par **le même moteur OHLC**.
Les anciens scores calculés par `ReturnPanel` restent des références sous une
autre comptabilité ; ils ne doivent pas être assimilés au bras sans règles.

`TradingResult` contient `equity`, `positions`, `orders`, `trades`, `decisions`,
`metrics`, `metadata` et l'état final `TradingState`. Chaque replay commence
flat et termine par une liquidation facturée. `TradingState` est un résultat
d'inspection, pas une interface de reprise ou d'exécution réelle.

`targets_from_probabilities` réutilise le décodeur continu existant.
`targets_from_labels(..., label_semantics="action"|"target_position")` exige
la sémantique : Hold conserve la position pour les actions, mais signifie Flat
pour les positions cibles. Convertir les labels séparément par ticker et par
split ; ne pas transporter une position d'un actif vers un autre.

## Allocation du capital

`allocation_mode="fixed_universe"` conserve le contrat historique : chaque
cible vaut `q / N`, où N est le nombre total de tickers. Une cible FLAT garde
donc implicitement son emplacement en cash.

`allocation_mode="equal_active"` retire les cibles FLAT du dénominateur. Pour
des cibles discrètes `-1`, `0`, `1`, chaque position active reçoit :

```text
abs(weight) = target_gross_exposure / nombre_de_positions_actives
```

Une entrée ou sortie modifie ce dénominateur. Le moteur réduit ou ferme d'abord
les positions, puis ouvre ou augmente les autres afin de financer le nouveau
portefeuille. Aucun emplacement de cash n'est réservé par ticker FLAT. Les
frais et le slippage sont réservés globalement avant conversion du poids en
quantité continue selon le prix.

Les règles de réduction de taille, plafonds par actif ou secteur, limites
d'exposition et bande sans ordre restent applicables. Elles peuvent rendre
l'exposition cible irréalisable ; le reliquat reste alors en cash. Le moteur ne
viole jamais une limite pour atteindre `target_gross_exposure`.

## Données, prix et temps

- OHLC : `date`, `ticker`, `open`, `high`, `low`, `close`. Calendrier complet
  identique pour tous les titres, clés uniques, valeurs positives et finies.
  Tous les prix/notionnels d'un portefeuille sont dans la même devise.
- Signaux : frame `date`, `ticker`, `target_position` (ou `position`) et
  éventuellement `signal_available_at`. Un tableau NumPy suit **l'ordre
  original des lignes OHLC**, avant tri interne.
- Daily : `date` est un identifiant de séance à minuit, pas l'heure de décision.
  Horaires inférés par défaut : 09:00–17:30 `Europe/Paris`, changement d'heure
  inclus. Fournir `open_time` et `close_time` pour séances écourtées ou autres
  horaires. Ces deux colonnes sont obligatoires en mode `intraday`.
  Des bougies contiguës peuvent partager une frontière close/open ; un signal
  observé après cette clôture ne peut pas obtenir cet open simultané.
- `after_open` : signal disponible après l'open ; `after_close` : après la
  clôture. L'heure explicite doit suivre strictement l'observation déclarée.
  Sans timestamp explicite, disponibilité = observation + 1 nanoseconde.
- `next_open` et `next_close` sélectionnent le premier prix de ce type
  strictement postérieur à disponibilité. Un signal après open peut être
  exécuté à close du même jour ; il ne peut pas obtenir rétroactivement l'open.
- `open_proxy`, réservé à `after_open`, rejoue au même open par hypothèse.
  `execution_is_proxy=true` figure dans métadonnées. Ce résultat n'est pas une
  validation d'un fill obtenu après observation de l'open.
- Si plusieurs signaux deviennent exécutables au même prix, le plus récent
  remplace les précédents. Leurs disponibilités doivent être croissantes.
- Pas d'entrée à la dernière clôture, faute de rendement futur observable.

### Actions d'entreprise

`price_basis="raw"` : prix historiques réellement non ajustés des splits.
`stock_splits` vaut zéro sans action, sinon ratio nouveau/ancien (2 pour 2:1).
À l'ouverture du split, quantités sont multipliées et niveaux de protection
divisés. Les indicateurs utilisent une unité continue basée uniquement sur les
splits déjà observés.

`price_basis="split_adjusted"` : OHLC déjà ajustés des splits, exprimés dans
une unité cohérente sur tout l'historique. Le moteur ne réapplique pas les splits.
C'est la convention à déclarer pour les OHLC Yahoo historiques du projet.
**La base est déclarée par le fournisseur/caller, jamais déduite d'un gap.**

Dans les deux modes, `dividends` est un montant par unité de prix/quantité fournie.
Crédit aux positions longues, débit aux shorts, avant ordres d'ouverture à date
de détachement. C'est une convention d'accumulation économique : la date réelle
de paiement et son financement ne sont pas simulés. Colonnes absentes = zéro,
ce qui suppose que le caller a vérifié l'absence d'actions manquantes.

`adj_close` n'est jamais mélangé aux OHLC pour les fills ou le mark-to-market.
Ne pas fournir des OHLC déjà ajustés des dividendes avec des dividendes séparés.
Les frais d'emprunt des shorts, intérêts, FX, impôts et contraintes broker ne
sont pas modélisés ; leur absence est enregistrée dans les métadonnées.

## Règles et ordre d'application

Toutes les distances et tailles sont explicites ; aucun seuil économique ne
s'active par défaut, sauf plafonds d'exposition brute/nette de 1 quand module
est activé. Pour chaque protection choisir pourcentage **ou** multiple ATR.

| Famille | Paramètres |
| --- | --- |
| SL / TP | `stop_loss_pct` / `stop_loss_atr`, `take_profit_pct` / `take_profit_atr` |
| Trailing | `trailing_stop_pct` / `trailing_stop_atr`, `trailing_activation_pct` |
| Break-even | `break_even_activation_pct`, `break_even_offset_pct` |
| Durée | `max_holding_bars` |
| Réentrée | `reentry="new_signal"|"cooldown"|"next_bar"`, `cooldown_bars` |
| Turnover | `no_trade_band`, en poids absolu du portefeuille |
| Allocation | `allocation_mode`, `target_gross_exposure` |
| Exposition | `max_asset_weight`, `max_sector_weight`, `max_gross_exposure`, `max_net_exposure` |
| Volatilité | `volatility_target`, `volatility_window`, `volatility_scale_cap` |
| Drawdown | `max_drawdown` |
| Frontières OU (longs) | `ou.mode="off"|"exit"|"entry_exit"`, voir [adaptation du papier](trading-ou.md) |

1. Appliquer actions d'entreprise et tester les protections déjà actives sur
   le gap d'ouverture. Stop franchi : open défavorable + slippage ; TP franchi :
   niveau TP conservateur, sans attribuer tout le gain du gap.
   Si le slippage d'entrée place un stop déjà au-delà du marché, sortie à open,
   sans inventer un fill au niveau de stop plus favorable.
2. À l'heure d'action, convertir les cibles en poids selon `allocation_mode`.
   Appliquer réentrée, durée, risque, événements, plafonds et bande sans ordre.
   Réduire/fermer d'abord, puis ouvrir/augmenter.
3. Tester niveaux de protection actifs sur high/low. Si TP et stop touchés,
   choisir stop et incrémenter compteur d'ambiguïtés. Une sortie forcée interdit
   toute réentrée dans cette bougie, même si signal s'inverse.
4. Après bougie terminée, mettre à jour trailing et break-even ; nouveaux
   niveaux applicables à bougie suivante. Pour une entrée à close, high/low
   antérieurs de cette bougie ne sont jamais réutilisés.

En mode `next_close`, étape d'action a lieu après contrôle intrabougie des
positions antérieures. Les timestamps intrabougie sont des **bornes supérieures**
de temps : `phase="intrabar_unknown"`, `earliest_time`/`latest_time` encadrent
le fill. Aucune trajectoire high-vers-low n'est reconstruite.

ATR = moyenne du true range, fenêtre 14 par défaut. ATR/volatilité utilisés à
open excluent la bougie en cours ; à close elle est terminée. ATR figé à
entrée. Sans historique suffisant, entrée nécessitant cet indicateur bloquée
avec motif de warmup. Ajouter des bougies de préparation et des cibles flat
pour disposer d'un historique avant la période étudiée.

Redimensionnement conserve ancre d'entrée, trailing et durée ; inversion
ferme épisode puis ouvre autre épisode. Après sortie forcée, `new_signal`
bloque même sens jusqu'à signal flat/opposé ; changer amplitude ne suffit pas.
`cooldown_bars=1` bloque toute prochaine bougie après sortie ; `next_bar`
autorise réentrée dès suivante. Durée maximale est constatée après bougie
détenue, puis sortie demandée au prochain prix exécutable.

Volatilité = écart-type des rendements close-to-close passés, annualisé.
Dimensionnement : `q * min(scale_cap, target / vol)`, borné à [-1,1]. Limites
de secteur utilisent colonne `sector` et somme des poids absolus. Limite nette
porte sur valeur absolue de somme des poids. Réductions de sécurité prennent
priorité sur bande ; réserve de frais/slippage protège plafonds après fills.
Ces plafonds s'appliquent aux rééquilibrages, pas continuellement pendant une
bougie. Le mark-to-market peut dépasser une limite entre deux actions.

Drawdown calculé sur capital observé à close, pic initial compris. Une fois
seuil franchi : liquidation au prochain prix d'action, blocage jusqu'à fin
du replay. Aucune liquidation n'est supposée au point bas intrajournalier.

## Calendrier événements

CSV ou Parquet avec `event_type`, `timestamp`, `known_at`, `scope`, `scope_value` ;
`event_id` facultatif. Horodatages UTC ; `scope` = global/ticker/sector.
`known_at` est moment où ce calendrier était connu, pas heure de publication
des résultats ou timestamp de téléchargement utilisé rétroactivement.

Types libres : BCE, Fed, BoE, CPI, emploi, earnings, événements sectoriels…
`event_types` vide sélectionne tous les types. Activer politique avec calendrier
absent, vide ou sélection sans événement provoque erreur.

Politiques : `block_increases` conserve/réduit exposition mais interdit nouvelle
direction ; `reduce` multiplie cible par `event_reduce_factor` ; `flat` demande
sortie. `event_pre_hours`/`event_post_hours` définissent fenêtres. Un événement
connu dont fenêtre tombe entre deux actions est pris en compte à **dernière
action disponible avant fenêtre**, car position resterait exposée entre-temps.
Les positions déjà ouvertes restent exposées quand on bloque seulement entrées.
Un calendrier connu trop tard ne permet pas d'inventer une sortie avant annonce.

`configs/trading/events_example.csv` contient uniquement événements
**synthétiques** pour illustration ; aucune collecte officielle n'est incluse.
Le choix heure/périmètre doit être validé avec le vrai calendrier historique.

## CLI et rapports

```bash
.venv/bin/python scripts/run_trading_backtest.py \
  --signals artifacts/comparisons/gru-optim/11-post-open-pilot/fold-0-seed-1-open_gap-positions.parquet \
  --data data/processed/cac40_daily_clean.parquet \
  --trading-config configs/trading/example.json \
  --allocation-mode equal_active \
  --target-gross-exposure 0.98 \
  --output-dir artifacts/trading/fold-0-seed-1-example
```

Exemple utilise signaux après open et exécution **open suivant**, contrairement
au proxy d'entraînement du pilote. La comparaison avec/sans règles emploie
cette même convention. `--execution`, `--signal-timing`, `--allocation-mode`
et `--target-gross-exposure` remplacent les valeurs JSON. Revenir à
`--allocation-mode fixed_universe` restaure l'allocation historique.
`--events` fournit calendrier quand filtre est activé.

`--ou-history` fournit un historique de prix pour préparer l'estimation OU.
Seules les lignes antérieures à la première prédiction de chaque groupe sont
transmises au moteur. Cet historique ne crée aucune position ni rendement avant
le début du replay. L'API accepte la même préparation via `history=ohlc_passés`.

Chaque combinaison des colonnes présentes `source_candidate`, `candidate`,
`fold`, `seed` est un replay indépendant. Aucun assemblage de folds
chevauchants ni moyenne de Sharpes par ticker présentée comme Sharpe portefeuille.

Sorties : `report.json`, `comparison.csv`, puis equity/positions/orders/trades/
decisions Parquet et métriques/métadonnées JSON pour chaque bras. Le répertoire
doit être neuf. Empreintes prix, signaux, événements et config sont sauvegardées.
Capital = cash + quantités signées ; frais calculés sur montant réellement
échangé, inversion facturée en deux transactions. Reserve non utilisée reste
cash. PnL par motif de sortie décrit épisodes fermés, **pas contribution causale**
de chaque règle. Pour isoler une règle, effectuer ablation à prédictions égales.
Les expositions moyennes sont mesurées aux clôtures ; elles ne représentent pas
une moyenne pondérée par le temps des positions intrabougie.

CLI refuse signaux à partir du 2022-05-10, holdout scellé du projet. Pour un
protocole différent, fournir son `--final-test-start`. Ouverture explicite
seulement via `--allow-final-test` ; jamais recherchée par ce script. Aucun
entraînement, grid search ou accès réseau n'est déclenché.

## Vérification

Voir aussi [l'ablation TP/durée/SL/trailing](../benchmarks/trading-exit-ablation.md) du fold
0/seed 1 : elle documente notamment l'effet du verrou de réentrée sur des
positions cibles continues. Aucun profil n'est sélectionné comme optimal.

```bash
.venv/bin/python -m pytest tests/test_trading_module.py tests/test_backtest_layers.py tests/test_financial_loss.py
```

Les tests couvrent causalité des prix/indicateurs, ambiguïtés OHLC, comptabilité,
réentrée, calendrier point-in-time, groupes, empreintes et garde du holdout.
Backtest daily quantifie les hypothèses ; trailing et exécution après open
nécessitent des prix intrajournaliers observés pour validation de fills réels.
