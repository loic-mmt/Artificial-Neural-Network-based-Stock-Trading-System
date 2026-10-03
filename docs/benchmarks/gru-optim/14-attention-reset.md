# 14. Reset GRU par attention sur les canaux

J'ai comparé la cellule PyTorch native, les mêmes équations déroulées manuellement et un reset par attention. 27 résultats, 3 folds et 3 seeds. L'attention agit sur les canaux cachés, pas sur des dates futures.

| Variante | Paramètres | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen | Rendement sous stress défavorable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 7 523 | 0,7879 | +7,67 % | -3,82 % | +5,09 % |
| Manuelle | 7 523 | 0,7895 | +7,69 % | -3,82 % | +5,11 % |
| Attention reset | 8 579 | 0,7765 | +7,63 % | -3,86 % | +5,10 % |

L'attention gagne en Sharpe dans 3/9 paires face au natif et 3/9 face au manuel. En rendement, elle gagne dans 4/9 face au natif et 3/9 face au manuel. Les résultats natifs et manuels proches montrent l'intérêt du contrôle d'implémentation, sans garantir une identité numérique après un long entraînement.

Je garde la cellule native. Le reset attention n'apporte pas de gain moyen et ajoute environ 14 % de paramètres ainsi qu'un déroulement moins optimisé. Cette adaptation locale ne reproduit pas le MCI-GRU complet. Holdout fermé, fills réels non testés.

Sources : [27 résultats](../../../artifacts/comparisons/gru-optim/14-attention-reset-post-open/results.json), [équations et lancement](../../plan/attention-reset-post-open.md).
