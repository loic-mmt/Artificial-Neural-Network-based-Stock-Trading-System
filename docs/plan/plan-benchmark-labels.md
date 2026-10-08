# Benchmark GRU : labels et objectifs post-open

Implémentation disponible dans `scripts/run_label_loss_benchmark.py`. Les anciens runners et labels restent inchangés. Le guide des commandes, des exports et de la reprise est dans [la documentation du benchmark](../src/label-loss-benchmark.md).

## 1. Contrat commun

Conserver :

- 143 tickers US, GRU32 attention, contexte 60, cap 32 features.
- Trois folds chronologiques purgés, seeds `1,7,19`.
- Holdout à partir du 22 juin 2023 fermé.
- Décision après ouverture J : features achevées jusqu’à J−1 + gap d’ouverture J. Aucun high/low/close/volume J dans les inputs.
- Même sélection de features, remplissage et normalisation TRAIN pour tous les modèles d’un fold, indépendamment des labels.
- Diagnostics d’apprentissage activés.

L’open sera un **proxy d’exécution**, pas une garantie de fill après observation de cet open. Les nouveaux contrats seront explicites, sans modifier silencieusement les anciens benchmarks.

## 2. Trois labellisations

Toutes exposeront des cibles `Short / Flat / Long`. Pour cette comparaison, le neutre forward sera **Flat**, pas une instruction de conserver la position précédente.

| Méthode | Cible proposée |
| --- | --- |
| Intraday | Signe de `close(J) / open(J) − 1` ; égalité → Flat |
| Forward | Rendement depuis open J sur dix séances ; seuils ±0,2 % |
| Volatility | Même rendement futur, normalisé par volatilité historique, avec règles de persistance existantes |

**Convention proposée pour dix séances : J à J+9 inclus**, donc sortie cible à clôture J+9. Cette convention sera enregistrée pour éviter toute ambiguïté avec les anciens labels clôture J → clôture J+10.

Pour volatility :

- Volatilité calculée uniquement jusqu’à J−1.
- Paramètres actuels conservés : fenêtre 20, seuils Long/Short 1/1,5, sortie 0,25, maintien minimum 5.
- État persistant et dépendances futures contrôlés aux frontières des partitions.

Les labels décrivent une cible d’apprentissage. **L’horizon 10 n’impose pas dix jours de détention à la stratégie.**

## 3. Deux protocoles financiers

| Protocole | Exécution | Coûts |
| --- | --- | --- |
| Intraday | Entrée après open J, liquidation à close J | Entrée + sortie chaque séance active |
| Overnight | Rééquilibrage après open J, détention jusqu’à open J+1 | Changements de position, entrée initiale, liquidation finale |

Même protocole pour tous les modèles comparés. Pas de fermeture quotidienne cachée dans le scénario overnight.

Référence d’allocation : slots égaux par ticker, position `q/N`, sans levier ni nouvelle optimisation de sizing. Exposition et cash seront enregistrés, avec contrôle comparatif à exposition commune.

Coûts proposés : **5 bps par sens**. Références distinctes :

- Intraday : toujours Long pendant la séance.
- Overnight : buy-and-hold.
- Cash dans les deux cas.

## 4. Trois objectifs d’apprentissage

### Cross-entropy

Un GRU par labellisation. Mesure l’apprentissage des classes, avec pondération des classes calculée uniquement sur TRAIN et enregistrée.

### Financière combinée

Conserver la formule actuelle :

```text
L_fin = −[0,75 × Sharpe régularisé
          + 0,25 × rendement net quotidien moyen / 0,0001]
```

Un témoin par protocole. **Pas trois témoins identiques par label** : cette loss ne consomme pas les classes.

### Hybride

Premier poids proposé, configurable mais figé avant lancement :

```text
L_hybride = 0,5 × L_CE + 0,5 × L_fin
```

Le poids PnL **0,25 reste interne à L_fin**. Le coefficient hybride 0,5 est un autre paramètre.

Enregistrer les composantes et leurs gradients : coefficients égaux ne signifient pas influence égale. Pas de grille de poids hybride au premier passage.

Point de contrôle important : le trainer CE historique fait des updates par minibatch, le financier par parcours chronologique complet. Le trainer dédié à ce benchmark fait une seule update AdamW globale par époque pour les trois familles. Les blocs limitent la mémoire, pas la fréquence des updates.

Le plafond commun est porté à **300 époques** par défaut (`--epochs 100 --epoch-multiplier 3`). L'early stopping conserve le meilleur checkpoint de validation intérieure, avec une patience configurable (`--patience`, 20 par défaut). Les nombres d'updates réellement effectuées sont enregistrés : l'arrêt anticipé peut les rendre différents. Une convergence insuffisante au plafond est signalée, pas assimilée à une labellisation inapprenable.

## 5. Grille et volume

| Famille | Configurations uniques | Entraînements |
| --- | ---: | ---: |
| CE : trois labels | 3 | 27 |
| Financière : deux protocoles | 2 | 18 |
| Hybride : trois labels × deux protocoles | 6 | 54 |
| Total | 11 | **99** |

Les checkpoints CE seront évalués dans les deux protocoles sans réentraînement.

Trois décodages post-entraînement :

- Continu : `P(Long) − P(Short)`.
- Signe : ±1 ; zéro exact → Flat.
- Argmax : Short / Flat / Long.

Cela produit **378 trajectoires d’évaluation uniques**, sans entraînement supplémentaire pour les décodages.

## 6. Comparaisons produites

**Apprentissage**

Accuracy, balanced accuracy, macro-F1, matrice de confusion, référence classe majoritaire, CE TRAIN/validation, courbes d’apprentissage et gradients.

Les scores de classification des modèles financiers resteront diagnostiques : leur loss optimise la position, pas directement les probabilités de chaque classe.

**Stratégie**

PnL/rendement net, Sharpe, drawdown, exposition, turnover, coûts, durées des positions, contributions Long/Short, résultats par ticker et courbes de portefeuille.

Inclure l’exécution parfaite des labels sous chaque protocole, clairement identifiée comme référence rétrospective non réalisable. Ce n'est pas nécessairement un plafond financier : un label à horizon dix séances est aussi évalué sous des protocoles ayant un autre horizon de détention.

**Positions opposées**

Comparer les positions exécutées à mêmes ticker, séance, fold, seed, protocole et décodage :

- Fréquence Long contre Short.
- Séquences de désaccord.
- Exposition engagée pendant ces désaccords.
- Contributions au rendement.
- Graphiques prix + positions des modèles.

## 7. Ordre d’implémentation et validation

1. Contrats post-open des labels et rendements, purge intraday incluse : implémentés.
2. Prétraitement commun ; masques de supervision séparés des masques financiers : implémentés. Aucun saut artificiel de séance. Un trou de cotation interne est rejeté ; les absences initiales/finales restent des slots cash.
3. Trainer commun CE / financière / hybride : implémenté dans un module dédié.
4. Décodages, comparaisons et graphiques : implémentés, sans réentraînement par décodage.
5. Exports, `--resume`, progression et diagnostics : implémentés. La reprise réutilise les fits terminés après contrôle des hashes ; un fit interrompu redémarre depuis son initialisation, pas depuis un optimizer partiel.
6. Tests de causalité, frontières, coûts, gradients et réutilisation des checkpoints : ajoutés.
7. Smoke test, reprise et petit pilote : validés. Le pilote AAPL/MSFT a terminé 11 fits et 42 évaluations avec plafond 300, sans atteindre ce plafond. Les 86 nouveaux tests et deux tests d'architecture passent ; la suite complète conserve 25 échecs FNSPID/news hors des modules modifiés ici. Détails dans [la documentation du benchmark](../src/label-loss-benchmark.md#validation-technique).
8. Grille complète : à lancer séparément, pas pendant l'implémentation.

Sélection des checkpoints uniquement sur validation intérieure. Un modèle encore en progression au plafond d’époques sera signalé comme **budget insuffisant**, pas comme preuve que ses labels sont inapprenables.
