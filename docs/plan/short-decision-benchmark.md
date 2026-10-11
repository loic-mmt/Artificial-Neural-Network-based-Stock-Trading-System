# Décisions long/short après le benchmark des gradients

Plan retenu le 11 octobre 2026. Statut : hypothèses à tester après la comparaison
des gradients d'état. Aucun nouveau modèle ni benchmark n'est implémenté ici.

## Objectif et ordre

Le [benchmark S0-S3](../benchmarks/us-trade-state-overnight.md) améliore le
rendement de la financière surtout par davantage d'exposition longue. S3 au
signe ressemble fortement à buy-and-hold, tout en restant sous son rendement.
Des shorts perdants persistent malgré les informations d'état.

Il faut d'abord lancer le [benchmark détaché/TBPTT](trade-state-gradients.md),
dont le runner est prêt. Ensuite,
il faut isoler la qualité des décisions short, sans changer en même temps
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

## À fixer avant implémentation

La comparaison des gradients détermine d'abord la recette d'état de référence.
Il faudra ensuite préciser les cibles des têtes binaires, la loss par branche,
les seuils, l'arbitrage des conflits et la grille minimale. Le Flat d'un label
et l'absence d'alerte d'un détecteur ne doivent pas être remappés implicitement.

Chaque piste doit être activable séparément. Pas de fusion implicite, de filtre
de régime obligatoire ni de règle de sortie ajoutée silencieusement. Les frais
de prêt, financement et contraintes de disponibilité des shorts restent à
traiter pour une validation d'exécution réelle.

Voir aussi le [plan des états](trade-state-overnight.md), les
[diagnostics ticker](../benchmarks/us-ticker-diagnostics.md) et les
[premiers tests de régime](../benchmarks/market-regime-short-filter.md).
