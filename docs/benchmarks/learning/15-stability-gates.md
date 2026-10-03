# 15 - Stability-gate sensitivity

**But.** Vérifier mécaniquement les règles de gouvernance, sans réutiliser les
données de marché ni ouvrir le holdout.

**Résultat.** 2 tests passés, 7 non sélectionnés. Validation de l'écart-type seed
maximal 0,05, du gap train/validation maximal 0,15, de la couverture complète des
seeds et du maintien du holdout fermé pour un candidat instable.

**Décision.** Seuils gelés ; ne pas les optimiser sur le holdout.

Source logicielle : [tests des règles et du maintien du holdout fermé](../../../tests/test_overfitting_control.py).

Protocole commun : [conventions et limites](README.md).
