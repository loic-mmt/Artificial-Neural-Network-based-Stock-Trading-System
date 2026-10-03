# Benchmark 12 : StockMixer GRU après l'open

Statut : terminé, 45 résultats. Je conserve ici le protocole.
Les [résultats et la décision](../benchmarks/gru-optim/12-stock-mixer.md)
sont séparés du plan.

Le benchmark 11 a établi le protocole de décision juste après l'open. Ce test
garde exactement les mêmes données, folds, seeds, coûts et objectif Sharpe.
Chaque action voit les features journalières terminées à J−1 ainsi que son
`night_pct` de J. Ni high, ni low, ni close de J ne sont des entrées. Le prix
d'open ajusté J est un proxy d'exécution vers l'open J+1, pas un fill observé.
Le holdout final à partir du 10 mai 2022 reste fermé.

Le GRU est partagé entre tickers et encode indépendamment chaque fenêtre de 60
jours. Les embeddings sont ensuite regroupés par date, selon un ordre explicite
de tickers. Le bloc optionnel échange de l'information **uniquement entre les
actions de cette date**. Aucune sortie du GNN n'est utilisée.

| Candidat | Relation inter-actions | Rôle |
| --- | --- | --- |
| `none` | Aucune | Reproduit le GRU `open_gap` du benchmark 11 |
| `mean` | Moyenne des autres actions présentes | Contexte de marché simple |
| `stock_mixer` | Compression `N → m` puis expansion `m → N` | Relation apprise de faible rang inspirée de StockMixer |
| `attention_self` | Chaque action ne voit qu'elle-même | Contrôle de capacité pour l'attention |
| `attention` | Attention entre toutes les actions présentes | Relation apprise et dépendante du jour |

Le mélange StockMixer apprend des paramètres par place de ticker. Il suppose
donc un univers et un ordre fixes, ici les dix tickers sélectionnés. Les places
absentes sont masquées dans la compression et la sortie. L'attention masque
aussi les actifs absents. Le test de `stock_mixer` **ne mesure pas** la capacité
à généraliser à un nouvel univers. L'éventuel mélange d'indicateurs avant GRU
(`S3` dans `GRU_optim.md`) reste une étape séparée, à faire seulement si le
mélange inter-actions apporte déjà un gain.

Le calcul de la loss et de son gradient porte sur la trajectoire financière
complète ; seuls les passages dans le modèle sont découpés par groupes de
dates. La sélection du checkpoint utilise la validation intérieure. Les mêmes
tirages de sensibilité OHLC sont appliqués à tous les candidats appariés.

Lancer les cinq candidats sur trois folds et trois seeds, soit 45 entraînements :

```bash
.venv/bin/python scripts/run_stock_mixer_ablation.py \
  --data data/processed/cac40_daily_clean.parquet \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --reference-cv artifacts/comparisons/gru-optim/04-causal-normalization-60-atr \
  --seeds 1,7,19 \
  --folds 0,1,2 \
  --market-states 3 \
  --attention-heads 1 \
  --draws 250 \
  --device auto \
  --output-dir artifacts/comparisons/gru-optim/12-stock-mixer-post-open
```

Après une interruption, relancer la même commande avec `--resume`. Changer
la liste des candidats, le device, le nombre de tirages ou le budget d'epochs
nécessite un autre dossier de sortie. `--max-epochs 1` sert uniquement à un
essai technique, pas à comparer les performances. Les fichiers de positions,
états de modèles et `results.json` sont sauvegardés pour chaque entraînement.

Lecture recommandée : comparer chaque candidat au contrôle `none` **dans ce
benchmark**, paire par paire. Examiner rendement net, Sharpe, drawdown,
exposition, variation entre folds/seeds et scénarios de prix défavorables.
`attention` doit aussi dépasser `attention_self` avant d'attribuer un gain aux
relations entre actions. Les 250 tirages OHLC sont une sensibilité de prix,
pas des fills observés ni un intervalle de confiance statistique.
