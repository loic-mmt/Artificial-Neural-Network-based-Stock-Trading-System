# Losses financières et positions continues

Je sépare l'apprentissage des classes de l'optimisation financière. Cross-entropy apprend Sell/Hold/Buy. PnL, Sharpe, CARA et combinaison optimisent une trajectoire de rendements nets. Leurs trois sorties softmax paramètrent une position, sans garantie de calibration des classes.

## Position et coûts

En long/short, `q = P(Buy)-P(Sell)`, entre -1 et 1. En long-only, `q = P(Buy)`. `ReturnPanel` évalue des calendriers synchronisés : les lignes absentes ou prix invalides ne deviennent pas des rendements inventés.

Dans le protocole close-to-close, le signal à la clôture J est exécuté après le délai, puis gagne le rendement suivant. Chaque partition commence FLAT, facture entrée, changement de position et liquidation finale. Le coût proportionnel porte sur `abs(e[t]-e[t-1])` ; une inversion est plus coûteuse qu'une simple fermeture de même taille.

Le rendement du portefeuille est la moyenne des rendements d'actifs `exposition * rendement - coût`. C'est le dénominateur fixe de l'univers, pas le mode `equal_active` du moteur de trading. Les workflows post-open fixent un autre panel de prix et une convention explicite ; il ne faut pas leur appliquer le timing close-to-close par analogie.

## Objectifs

| Objectif | Ce qui est minimisé |
| --- | --- |
| PnL | Opposé du rendement net arithmétique moyen |
| Sharpe | Opposé du Sharpe avec dénominateur `sqrt(variance + epsilon²)` |
| CARA | Moyenne de `expm1(-gamma * net) / gamma` ; gamma 0 retrouve PnL |
| Combinée | `(1-w) * loss_sharpe + w * (-mean(net) / pnl_scale)` |

Valeurs courantes : annualisation 252, epsilon `1e-4`, `pnl_scale=1e-4`. Le poids PnL 0,25 n'est donc pas une allocation de 25 % du capital. La métrique `net_pnl` rapportée compose les rendements et multiplie par le capital initial ; elle n'est pas exactement l'objectif arithmétique d'entraînement.

Le Sharpe net ordinaire est aussi rapporté. À epsilon fixe, le Sharpe régularisé n'est pas invariant à un changement d'amplitude des positions. Je garde cette distinction dans les contrôles d'exposition.

## Entraînement et sélection

L'objectif porte sur toute la trajectoire chronologique, pas sur une moyenne de Sharpes de mini-batches mélangés. Le calcul par blocs limite la mémoire d'activation : positions sans graphe, gradient exact de trajectoire, puis recalcul des activations avec le même dropout. Une mise à jour par epoch, early stopping sur validation et restauration du meilleur checkpoint.

Prétraitements ajustés sur train uniquement. Les pondérations d'événements ne sont pas appliquées aux rendements financiers. La loss financière n'essaie pas de reproduire des labels parfaits.

`scripts/run_loss_comparison.py` compare les losses avec les mêmes prix et coûts. Sans demande explicite de test final, le holdout reste fermé. Les options de purged CV sélectionnent sur les folds externes du développement, pas sur le holdout final. Walk-forward et grid search disposent aussi d'objectifs financiers ; leurs conventions et interfaces restent séparées des replays locaux MT5.

L'ancienne vue discrete/legacy de cross-entropy et la vue continue ne partagent pas nécessairement leur comptabilité. Pour comparer les losses, je prends la vue continue commune.

Sources : [définitions et ReturnPanel](../../src/trading_system/training/financial_loss.py), [entraînement et artifacts](../../src/trading_system/experiments/position_objectives.py), [CLI de comparaison](../../src/trading_system/pipelines/compare_losses.py), [CV](purged-cv.md), [trading OHLC](trading-module.md).

Résultats : [losses initiales](../benchmarks/learning/11-loss-objectives.md), [CARA et combinaison](../benchmarks/gru-optim/05-cara-combined.md). Les détails de l'intégration de septembre restent dans [l'archive historique](../papers/old_docs/financial-loss.md), sans modification de l'archive.
