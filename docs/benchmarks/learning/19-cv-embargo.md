# 19 - Purged-CV embargo

**But.** Confirmer la sémantique de l'embargo dans une CV expanding past-only.

**Résultat.** Embargo 5 et embargo 0 donnent exactement les mêmes métriques,
checkpoints et états de purging. Nombre d'exclusions supplémentaires : zéro.

**Décision.** Embargo 0. Aucun échantillon d'entraînement n'existe après la
validation dans ce protocole, donc l'embargo post-validation est sans effet.

**Sources.** [Embargo 5](../../../artifacts/comparisons/ohlc-clean/19-cv-embargo-5/report.json)
et contrôle embargo 0 dans `17-cv-folds-3`.

Protocole commun : [conventions et limites](README.md).
