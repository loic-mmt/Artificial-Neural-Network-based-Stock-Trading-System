# Allocation égale et sizing : décision conservée

Je garde une allocation égale en capital entre les actifs non FLAT, pas un nombre égal d'actions. Avec trois longs, trois shorts et deux FLAT, chaque position active reçoit un sixième du gross cible. Les deux FLAT ne réservent pas chacun un huitième du capital.

Quand le nombre d'actifs actifs change, le mode `equal_active` recalcule les poids et redimensionne les positions existantes. Les coûts du resizing doivent être débités. La quantité dépend ensuite du prix, de la taille du contrat et des contraintes du broker. Un gross de 100 % ne veut pas dire que des shorts financent librement tous les longs ni qu'il n'y a aucun besoin de marge.

## Ce qui a été testé et ce qui est vérifiable

Les [tests France](mt5-france.md) et [US](mt5-us.md) sauvegardent un replay en signe équipondéré sur l'univers complet. France progresse à pleine exposition ; US reste perdant. Cela montre que l'allocation ne corrige pas une mauvaise direction.

Le passage à `equal_active` est ensuite implémenté dans le module de trading et activable séparément. Les anciens benchmarks MT5 gardent leur dénominateur historique : je ne les présente pas comme des benchmarks déjà réalisés avec exclusion des FLAT. Les replays OHLC ont aussi leurs règles et coûts propres.

J'avais exploré un sizing selon les performances passées après allocation égale. La décision conservée est de ne garder que l'égalité. Je n'ai pas retrouvé de rapport autonome chiffré de cet essai dans les artifacts disponibles : je ne reconstitue pas un gain ou une perte de mémoire. Même limite pour la règle d'inversion après plusieurs positions perdantes : ce n'est pas une stratégie validée ni activée par défaut.

## Suite si je reprends le sizing adaptatif

Je dois fixer la règle et ses bornes avant le test, ne lire que les performances déjà réalisées, garder le même gross et facturer les rééquilibrages. Comparer au sizing égal et séparer la sélection des tickers de la taille de leurs positions. Sans cela, une amélioration peut venir d'un changement de levier ou d'une sélection a posteriori.

Références : [API, contraintes et CLI du trading](../src/trading-module.md), [ablation des sorties](trading-exit-ablation.md), [validation OU](trading-ou-validation.md). Les rapports France/US liés dans ces fiches sont les sources chiffrées disponibles.
