# Benchmark US : interaction features et gate marché

Statut au 8 octobre 2026 : **108/108 entraînements terminés et vérifiés**, rapports
appariés et contrôles d'exposition disponibles. La [fiche de résultats](../benchmarks/us-feature-gate-interaction.md)
fixe la conclusion exploratoire : cap32, gate désactivé par défaut, identité
obligatoire ; `identity_market32` reste un challenger fragile. Les commandes
ci-dessous décrivent une reproduction, pas un run restant à terminer.

## Ce que l'on teste

Je teste si le gate guidé par le Transformer aide à exploiter davantage de features, indépendamment pour le GRU, le GNN `rolling_residual_topk` et le GNN identité sans relations. Ce graphe résiduel reste un challenger exploratoire, pas un vainqueur confirmé.

| Branche | Cap 32 | Cap 64 |
| --- | --- | --- |
| GRU sans gate | Oui | Oui |
| GRU avec gate marché | Oui | Oui |
| GNN résiduel sans gate | Oui | Oui |
| GNN résiduel avec gate marché | Oui | Oui |
| GNN identité, sans relations ni gate (`identity`) | Contrôle | Contrôle |
| GNN identité, sans relations avec gate marché (`identity_market`) | Contrôle | Contrôle |

Je conserve six candidats par cap : `gru`, `gru_market`, `rolling_residual_topk`, `rolling_residual_topk_market`, `identity` et `identity_market`. Budget maximal : **12 variantes × 3 folds × 3 seeds = 108 entraînements**. L'ancienne grille de dix variantes prévoyait 90 entraînements ; ce budget historique est remplacé par 108. Les checkpoints historiques de schéma 1 ne sont pas réutilisés : le replay précédent valide leurs métriques, pas une provenance absente. Une référence de schéma 2 n'est réutilisée que si sa signature et ses fichiers correspondent exactement.

Les 54 tâches par cap sont complètes, y compris les 18 tâches `identity_market`.
Aucun export requis ne manque et le holdout reste fermé. Les diagnostics détaillés
des poids du gate ne sont pas encore produits ; ils sont séparés de la complétude
des entraînements et de la comparaison financière.

Le même pool `technical,market,sector` est utilisé dans toutes les variantes. Le benchmark historique utilisait effectivement `technical,sector`. Les deux baselines sont donc réentraînées sur le pool prospectif commun : ne pas comparer le nouveau cap 64 à l'ancien cap 32 pour attribuer un effet aux seules features supplémentaires.

Le cap est un maximum, pas une garantie du nombre d'entrées. Couverture, variance et corrélation peuvent réduire le compte. Le dry-run vérifie les colonnes effectivement sélectionnées par fold, leur ordre, le même pool, les mêmes calendriers et les mêmes fills/scalers sur les colonnes communes. Les entrées du petit cap doivent être le préfixe ordonné de celles du grand cap. Une incompatibilité bloque l'entraînement ; un cap sous-rempli est signalé. Aucun label externe n'entre dans la sélection.

Vérification locale des données PC fournies : les trois folds retiennent effectivement 32 et 64 colonnes, avec nesting ordonné et prétraitements communs compatibles.

## Paramètres inchangés

143 actions, contexte temporel 60, objectif `combined` à 0,25, coût 5 bps, décodage continu `P(Buy)-P(Sell)`, exécution différée d'une séance. Labels triple barrier ATR20, horizon10, barrières0,75/0,75 et CUSUM0,5. GRU `hidden_size=32`, GNN largeur32/profondeur1, Transformer largeur32/profondeur1/4têtes, température1. Graphe lookback252, top-k5, rééquilibrage20.

CV figée : 3 folds, fractions initiale0,5/interne0,2, gap5, embargo0, seeds1/7/19. Aucun test sur le holdout final, aucune news, aucun téléchargement. Ce benchmark ne change ni le sizing ni la fréquence de réentraînement.

Les features marché des actions et le contexte ETF/VIX du Transformer sont deux entrées distinctes. Le Transformer pondère les features de la branche ; il ne produit pas un vote de trading ajouté au GRU/GNN.

Les colonnes du contexte Transformer portent le préfixe `market_context__`, par exemple `market_context__vix_level`. L'audit `market_audit.source_features` conserve le nom de chaque feature source. Ce renommage préserve les valeurs et les dates de disponibilité ainsi que le pool commun des actions : `vix_level` peut aussi être sélectionné pour le GRU/GNN dans cette grille. Le test mesure donc l'apport de l'encodage et du gate sur ce pool commun, pas l'ajout d'une information VIX jusque-là absente.

## Lancer sur ce Mac

Utiliser les fichiers PC figés déjà copiés, depuis la racine du dépôt :

```bash
.venv/bin/python scripts/run_us_multimodal_study.py \
  --stage features \
  --graph-choice rolling_residual_topk \
  --data data/data_pc/processed/mt5_stocks_us_daily_clean.parquet \
  --market-context-data data/data_pc/processed/us_market_context_daily.parquet \
  --device auto \
  --compare-exposure \
  --output-dir artifacts/comparisons/us-relational-market/04-feature-gate-interaction
```

Ajouter `--dry-run` pour vérifier le budget et les audits sans entraînement ni écriture. Ajouter `--resume` pour reprendre exactement cette étude après interruption. La reprise vérifie données, code, runtime et protocole ; une autre machine n'est pas supposée numériquement identique. Ne pas modifier les sources pendant le run.

Si l'ancienne grille de 90 entraînements a déjà été lancée, je crée un nouveau dossier de sortie pour cette grille de 108. `--resume` ne peut pas ajouter silencieusement les contrôles à une étude déjà figée. Une référence compatible peut être fournie avec `--reference-run`, mais chaque signature et chaque fichier doivent passer les vérifications.

## Lancer sur le PC GPU

Même commande, avec les fichiers figés dans `data/processed` :

```powershell
.venv\Scripts\python.exe scripts/run_us_multimodal_study.py --stage features --graph-choice rolling_residual_topk --data data/processed/mt5_stocks_us_daily_clean.parquet --market-context-data data/processed/us_market_context_daily.parquet --device cuda --compare-exposure --output-dir artifacts/comparisons/us-relational-market/04-feature-gate-interaction
```

## Résultats et lecture

Les checkpoints, logits/probabilités/positions et prix de backtest sont sauvegardés dans `02-features/features-32` et `02-features/features-64`, avec manifests vérifiés. `study.json` conserve le préflight et les comptes effectifs. Une seule barre `tqdm` affiche progression, ETA et pics RAM/VRAM disponibles.

Avec `--compare-exposure`, un rapport est créé automatiquement après les entraînements dans `reports/features-interaction-<date UTC>`. Il contient JSON, Markdown et CSV dédiés. Le calcul est apparié par fold/seed, sans sélection automatique :

```text
gain_features_plain = score64_plain - score32_plain
gain_features_gate  = score64_gate  - score32_gate
gain_gate32         = score32_gate - score32_plain
gain_gate64         = score64_gate - score64_plain
interaction         = gain_features_gate - gain_features_plain
```

Ces effets sont calculés pour le Sharpe régularisé primaire et les métriques financières. **Une interaction positive ne signifie pas que le gate64 bat le plain64**, ni que le cap64 est meilleur. Lire les effets simples, leur variation par fold et par seed, les comptes de features et le coût d'entraînement.

Les rapports d'exposition utilisent les mêmes prédictions externes et les mêmes prix, sans réentraîner :

1. Positions brutes, avec contrôle des métriques sauvegardées.
2. Exposition brute moyenne commune aux douze variantes et au buy-and-hold, par fold/seed : diagnostic ex post, pas règle de déploiement.
3. Exposition brute commune séance par séance : sensibilité avec resizing, turnover et coûts recalculés.

Aucune position manquante ne devient artificiellement FLAT. Égaliser le gross ne neutralise ni l'exposition nette, ni le beta, ni la volatilité. Les seeds partagent les mêmes périodes ; neuf résultats ne constituent pas neuf périodes indépendantes. L'univers survivant et les secteurs non point-in-time restent des limites.

Le Sharpe net ordinaire est invariant à une réduction constante positive des positions lorsque les coûts sont linéaires. Le Sharpe régularisé avec epsilon fixe ne l'est pas : son changement après normalisation ne représente pas nécessairement un changement de qualité du signal. Le score brut signé reste le critère primaire ; les normalisations sont des contrôles descriptifs.

Les comparaisons `identity_market` contre `identity`, puis GNN résiduel contre
identité au même cap et état du gate, sont produites par fold/seed. Les neuf
interactions moyennes de Sharpe, trois branches et trois méthodes d'exposition,
sont négatives. Le gate ne rend pas le passage à64 intéressant dans ce protocole.
Les diagnostics détaillés de ses poids, les nouvelles profondeurs et l'attribution
individuelle des features restent des étapes distinctes. Il ne faut pas confondre
variance TRAIN, poids d'un gate et contribution financière d'une feature.

Pour recalculer uniquement le rapport après un entraînement terminé :

```bash
.venv/bin/python scripts/compare_us_multimodal_study.py \
  --study-dir artifacts/comparisons/us-relational-market/04-feature-gate-interaction \
  --exposure-comparison \
  --output-dir artifacts/comparisons/us-relational-market/04-feature-gate-interaction/reports/reanalysis-01
```

Le dossier de rapport doit être nouveau. Les résultats sources ne sont jamais réécrits.
