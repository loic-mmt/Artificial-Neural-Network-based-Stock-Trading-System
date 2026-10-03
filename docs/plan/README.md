# Plans

Je conserve ici les hypothèses, l'ordre d'implémentation et les protocoles de lancement. Les conclusions sont dans [le suivi des benchmarks](../benchmark-summary.md).

| Sujet | Plan | État au 3 octobre 2026 |
| --- | --- | --- |
| Améliorations GRU et idées des papiers | [GRU optim](GRU_optim.md) | Plan conservé. Résultats 01 à 14 documentés séparément. |
| Contrat, branches et fusion multimodale | [Next steps](next_steps.md) | Contrat et branches disponibles. Les fusions prévues ne sont pas toutes validées. |
| Optimisation US multimodale | [P0 et suite](us-multimodal-optimization-plan.md) | P0 disponible ; suite conditionnée aux contrôles. |
| Features et gate marché | [Interaction](us-feature-gate-interaction.md) | Prêt, 12 variantes et 108 entraînements maximum, avec `identity_market` aux caps 32/64. Aucun résultat local trouvé. |
| StockMixer après l'open | [Protocole 12](stock-mixer-post-open.md) | Terminé ; [résultats](../benchmarks/gru-optim/12-stock-mixer.md). |
| Débruitage après l'open | [Protocole 13](denoising-post-open.md) | Terminé ; [résultats](../benchmarks/gru-optim/13-denoising.md). |
| Reset par attention | [Protocole 14](attention-reset-post-open.md) | Terminé ; [résultats](../benchmarks/gru-optim/14-attention-reset.md). |

Les anciens plans dans `papers/old_docs/` restent des archives, pas une feuille de route actuelle. Je conserve leur contenu et leur emplacement.
