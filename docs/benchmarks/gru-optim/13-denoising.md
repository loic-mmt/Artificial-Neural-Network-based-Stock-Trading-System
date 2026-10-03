# 13. Débruitage avant le GRU

J'ai comparé les features brutes, un autoencodeur débruiteur et sa variante avec attention du latent. 27 résultats GRU, 3 folds et 3 seeds. Le débruiteur est appris sur train, sélectionné sur validation intérieure puis gelé. Les prix cibles restent inchangés.

| Variante | Sharpe régularisé moyen | Rendement net moyen | Drawdown moyen | Exposition moyenne | Rendement sous stress défavorable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Brut | 0,7879 | +7,67 % | -3,82 % | 12,39 % | +5,09 % |
| DAE | 0,6537 | +6,08 % | -3,56 % | 10,78 % | +3,65 % |
| Attention DAE | 0,6877 | +6,13 % | -3,59 % | 10,58 % | +3,66 % |

Les deux débruiteurs ne gagnent en Sharpe et rendement que dans 2/9 paires chacun. Leurs Sharpes sous stress défavorable sont 0,2132 et 0,2816, contre 0,5027 pour le brut. La réduction du drawdown va avec une exposition réduite ; elle ne suffit pas à démontrer un meilleur signal.

Je conserve les features brutes. Une meilleure reconstruction ne garantit pas une meilleure prévision financière. C'est une adaptation légère, pas une reproduction complète d'AGRUA. Les stress sont simulés et le holdout reste fermé.

Sources : [27 résultats](../../../artifacts/comparisons/gru-optim/13-denoising-post-open/results.json), [protocole et lancement](../../plan/denoising-post-open.md).
