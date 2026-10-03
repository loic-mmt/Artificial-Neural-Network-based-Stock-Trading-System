# 07. Fusion fixe GRU et branche de nœuds

J'ai rejoué les positions sauvegardées du benchmark 06, sans réentraîner. Ce test mélange les positions continues, pas les logits et pas une fusion apprise. Les poids GRU 0,50 et 0,75 sont fixés à l'avance. Neuf paires fold/seed par combinaison.

| Branche ajoutée | Poids GRU | Sharpe régularisé | Rendement net | Gain rendement face au GRU redimensionné |
| --- | ---: | ---: | ---: | ---: |
| Identité | 0,50 | 0,7285 | +6,02 % | 4/9 paires |
| Identité | 0,75 | 0,7484 | +6,64 % | 4/9 paires |
| Pearson rolling | 0,50 | 0,7308 | +6,10 % | 4/9 paires |
| Pearson rolling | 0,75 | 0,7492 | +6,68 % | 4/9 paires |

GRU seul : Sharpe 0,7535, rendement +7,19 %, drawdown moyen -3,42 %. Les mélanges font moins bien en Sharpe et rendement moyens. Face au GRU ramené à leur exposition, les gains de rendement quotidien moyens sont très petits, de 0,006 à 0,010 bps. Un seul intervalle apparié sur neuf exclut zéro dans chaque combinaison.

Je ne retiens pas cette fusion comme défaut. Le contrôle d'exposition utilise tout le fold : diagnostic ex post, pas règle déployable. Le bootstrap par blocs ne transforme pas les seeds en périodes indépendantes. Le holdout reste fermé.

Source : [rapport de fusion](../../../artifacts/comparisons/gru-optim/07-gnn-fusion-fixed/report.json), avec bootstrap de 1 000 tirages et blocs de 20 séances.
