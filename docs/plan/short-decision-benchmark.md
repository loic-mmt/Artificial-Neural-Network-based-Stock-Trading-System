# Décisions long/short après le benchmark des gradients

Plan retenu le 11 octobre 2026, recette fixée le 12 octobre 2026.
Statut au 14 octobre 2026 : les 81 fits, 486 trajectoires et 324 contre-factuels
sont terminés. Les [résultats détaillés](../benchmarks/us-short-decisions.md)
ne démontrent pas de gain short robuste ; le holdout reste fermé. La recette
ci-dessous conserve le protocole fixé avant OUTER.

## Objectif et ordre

Le [benchmark S0-S3](../benchmarks/us-trade-state-overnight.md) améliore le
rendement de la financière surtout par davantage d'exposition longue. S3 au
signe ressemble fortement à buy-and-hold, tout en restant sous son rendement.
Des shorts perdants persistent malgré les informations d'état.

Le [benchmark détaché/TBPTT](trade-state-gradients.md) est terminé : 45 fits et
90 trajectoires. TBPTT 40 reste un challenger faible, TBPTT 10 n'apporte pas
de gain robuste à exposition comparable. S3 détaché reste la référence.
Il faut maintenant isoler la qualité des décisions short, sans changer en même temps
features, labels, données, exécution et allocation.

La priorité est de vérifier si les shorts gagnent sur les baisses et réduisent
les drawdowns sans détruire les phases haussières. Ce n'est pas une promesse
de surperformance. Éviter certaines baisses en restant Flat peut aussi
améliorer une stratégie : le short n'est pas le seul levier possible.

## Trois pistes indépendantes

### 1. Détecteur Short, stratégie Long par défaut

Tester un détecteur à deux sorties : absence d'alerte et Short. La politique
d'exécution devient :

| Sortie du détecteur | Position exécutée |
| --- | --- |
| Flat, au sens absence d'alerte Short | Long |
| Short | Short |

Ce test est une stratégie Long/Short avec Long par défaut, pas un portefeuille
Short/Flat. Il faut nommer les deux niveaux séparément pour éviter de confondre
la classe du détecteur et la position réellement détenue.

Ajouter un contrôle sur les mêmes alertes : rester Flat au lieu de Short.
Comparer Long par défaut + Short aux alertes avec Long par défaut + Flat aux
alertes permet de séparer le gain d'évitement d'une baisse du gain du short.
Un vrai portefeuille Short/Flat peut être évalué comme diagnostic distinct,
sans le comparer au buy-and-hold uniquement sur son PnL brut.

Le runner fournit deux comparaisons distinctes : une simulation indépendante
en boucle fermée pour chaque politique, puis une projection des signaux
Long/Short gelés, sans réévaluer le modèle. La première peut changer les
alertes futures via les états. La seconde isole le remplacement du Short
par Flat sur les mêmes signaux, mais ne représente pas une nouvelle politique
avec feedback. Il ne faut pas mélanger ces deux contrôles.

### 2. Un encodeur, deux têtes Long/Flat et Short/Flat

Tester un GRU partagé avec deux têtes indépendantes : une estime l'opportunité
Long, l'autre l'opportunité Short. Chacune peut s'abstenir vers Flat.

Il faut fixer avant OUTER la règle lorsque les deux têtes sont actives ou
inactives. Un conflit ne doit pas être résolu à partir du rendement futur.
Les seuils, la calibration éventuelle et l'arbitrage utilisent inner uniquement.

Ce test vérifie si spécialiser les décisions améliore le timing et le côté,
tout en conservant une représentation de marché commune.

### 3. Deux modèles séparés Long/Flat et Short/Flat

Tester deux GRU indépendants, chacun avec ses propres paramètres : l'un pour
Long/Flat, l'autre pour Short/Flat. Conserver la même politique d'arbitrage
que dans le test à deux têtes pour isoler l'effet de la séparation des encodeurs.

Les deux encodeurs ont des paramètres indépendants, mais leur loss financière
est celle du portefeuille combiné. Ils partagent une update globale et un
checkpoint choisi sur inner. Ce ne sont pas deux entraînements isolés dont
les meilleurs checkpoints seraient assemblés après OUTER. En CE, les losses
binaires normalisées sont moyennées, avec poids TRAIN distincts par tête.

Le coût de calcul et le nombre de paramètres augmentent. Il faut les rapporter
et distinguer gain de spécialisation et gain de capacité, sans annoncer un
budget équivalent par défaut.

## Contrôles et mesures

- Référence always-long du même moteur, modèle retenu après les gradients,
  contrôle Long/Flat et diagnostic Short/Flat.
- Mêmes dates, tickers, seeds, folds, protocole overnight après l'open,
  coûts et allocation. Les simulations recalculent leurs propres états.
- Rendement net, Sharpe, drawdown, expositions brute/nette, turnover et coûts.
- Contribution nette séparée des longs et shorts, nombre d'épisodes,
  durées, pertes persistantes et délai de fermeture d'un short perdant.
- Performance des alertes : baisse évitée ou short rentable, faux shorts
  pendant une hausse et coût des entrées/sorties.
- Analyse par régime avec une définition causale fixée sur TRAIN/inner,
  jamais un découpage optimisé sur OUTER. Rapporter aussi les résultats sans
  filtre de régime.
- Comparaisons appariées par fold/seed et contrôles d'exposition ; les
  périodes haussières ne suffisent pas à valider un détecteur de baisse.

Les paramètres du short doivent être indépendants de ceux du long quand cela
est pertinent, mais leur recherche reste bornée et sélectionnée sur inner.
Il ne faut pas choisir seulement les tickers ou seeds qui gagnent après lecture
du backtest. Le holdout final reste fermé jusqu'au gel du protocole.

## Recette fixée avant OUTER

- Même GRU32, attention, tête linéaire, contexte 60 et cap32 ; mêmes 143
  tickers, folds, seeds 1/7/19, allocation fixe `q/N`, coûts 5 bps par sens,
  features achevées J-1 plus gap open J, exécution proxy open J vers open J+1.
- États S3 détachés pour les architectures spécialisées et la référence
  financière trois classes et le témoin CE-S3 apparié. Un témoin CE-S0 garde
  aussi la référence supervisée historique. Aucun TBPTT, stop, durée maximale ou filtre
  de régime ajouté.
- Deux familles : `cross_entropy` binaire et financière PnL/Sharpe, poids
  PnL 0,25. Pas d'hybride dans cette grille.
- CE : cibles `volatility_position`, horizon 10. La tête Short apprend
  `Short` contre `not-Short` : les labels Long et Flat connus sont négatifs.
  La tête Long apprend `Long` contre `not-Long`. Les inconnus sont masqués,
  jamais convertis en absence d'alerte connue. Les poids binaires viennent
  de TRAIN uniquement. La référence CE conserve sa CE trois classes.
- Financière : aucune supervision par les labels. Les activations bornées
  commandent la position ; elles ne sont pas des probabilités calibrées.
- En continu, le détecteur donne `q = 1 - 2 * score_short`. Les deux têtes
  donnent `q = score_long - score_short`. Un désaccord atténue la position.
  Long par défaut à taille entière est la politique binaire, pas ce continu.
- Au décodage `sign`, seuils Long et Short fixés à 0,5, activation si
  `score > seuil`. Détecteur : absence d'alerte donne +1, alerte donne -1.
  Deux têtes : une seule active donne son côté ; aucune ou les deux actives
  donnent Flat. Ce décodage spécialisé n'est pas simplement `sign(q)` pour
  les deux têtes. Les témoins trois classes gardent leur signe historique.
- `--conflict-policy margin` autorise une autre règle explicitement choisie
  avant le run : le plus grand dépassement normalisé du seuil gagne ; égalité
  donne Flat. Aucun seuil, conflit ou calibrateur sélectionné sur OUTER.
- `long_flat = max(q, 0)` et `short_flat = min(q, 0)`, après le décodage choisi.
  Le modèle n'est pas réentraîné pour chaque projection. Chaque simulation
  fermée repart en cash et recalcule ses propres états.
- Même plafond 300 époques, patience 20, une seule update AdamW par époque.
  Le batch size reste une taille de bloc mémoire. Sélection des poids sur
  inner uniquement ; holdout à partir du 22 juin 2023 exclu avant les features.
- Régime descriptif : clôture ajustée J-1 comparée à sa moyenne de 200
  cotations précédentes, par actif. Définition fixée, historique insuffisant
  nommé `unknown`. Aucune modification des signaux selon ce régime.

### Taille de la grille

Trois architectures fois deux familles, plus financière S3, CE-S3 et CE-S0 témoins :
**9 configurations, 81 fits, 99 encodeurs GRU entraînés**. Deux des neuf
configurations entraînent chacune deux encodeurs, d'où la distinction.

Trois politiques fois deux décodages donnent **486 trajectoires outer**.
Les projections des mêmes alertes donnent en plus **324 contre-factuels**
économiques sans entraînement ni nouvelle inférence. Ces contre-factuels
ne comptent pas comme de nouvelles stratégies en boucle fermée.

Avec `--families financial` : 36 fits, 45 encodeurs, 216 trajectoires et
144 contre-factuels. Les architectures, politiques et décodages se
sélectionnent séparément ; les témoins peuvent être retirés explicitement
avec `--no-controls` pour un pilote, pas pour choisir le vainqueur.

### Commande PC GPU

```powershell
python -u scripts/run_short_decision_benchmark.py --data data/processed/mt5_stocks_us_daily_clean.parquet --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json --device cuda --output-dir artifacts/comparisons/us-short-decisions
```

Ajouter `--resume` à la même commande pour réutiliser les fits complets
vérifiés par taille et SHA-256. Un fit interrompu repart de son initialisation.
Il faut conserver sources, versions numériques, paramètres et données identiques.

Pour inspecter la grille sans entraîner ni lire les prix :

```shell
.venv/bin/python scripts/run_short_decision_benchmark.py --dry-run
```

Pour vérifier CUDA sur un petit pilote, dans un dossier distinct :

```powershell
python -u scripts/run_short_decision_benchmark.py --tickers AAPL MSFT --start 2019-01-02 --end 2021-12-31 --cv-folds 1 --seeds 1 --epochs 1 --epoch-multiplier 1 --device cuda --no-plots --output-dir artifacts/comparisons/short-decisions-cuda-pilot
```

Les [exports et modules](../src/short-decision-benchmark.md) permettent de
comparer les gains appariés, l'exposition, les erreurs d'alertes, les épisodes
short prolongés, les régimes et le coût de calcul. Les alertes journalières
et le taux de réussite des épisodes ne mesurent pas la même chose.

Chaque piste doit être activable séparément. Pas de fusion implicite, de filtre
de régime obligatoire ni de règle de sortie ajoutée silencieusement. Les frais
de prêt, financement et contraintes de disponibilité des shorts restent à
traiter pour une validation d'exécution réelle.

Voir aussi le [plan des états](trade-state-overnight.md), les
[diagnostics ticker](../benchmarks/us-ticker-diagnostics.md) et les
[premiers tests de régime](../benchmarks/market-regime-short-filter.md).
