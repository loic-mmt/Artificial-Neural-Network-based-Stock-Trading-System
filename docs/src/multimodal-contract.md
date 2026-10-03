# Contrat multimodal

Je garde GRU, GNN, Transformer marché et sentiment indépendants. Le contrat aligne leurs entrées par date de séance et ticker, sans obliger une branche à lire les sorties d'une autre.

## Dataset et lots

`build_multimodal_dataset` reçoit des frames déjà préparées et un univers ordonné explicitement. Chaque ticker garde sa place, même absent certains jours. Le dataset construit les lots à la demande et réutilise les fenêtres par ticker ; il ne matérialise pas tout le tenseur dates × tickers × temps × features.

| Entrée | Forme | Utilisation |
| --- | --- | --- |
| `temporal` | `[B,N,T,F]` | Séquences propres aux actions pour le GRU |
| `node` | `[B,N,Fn]` | Features de nœuds pour le GNN indépendant |
| `market_sequence` | `[B,Tm,Fm]` | Contexte global réservé au Transformer marché |
| `market` | `[B,Fm]` | Contexte instantané historique du contrat |
| `sentiment` | `[B,N,Fs]` | Export daté optionnel de la librairie externe |
| Labels et clés | Par date/ticker | Cibles connues et retour exact aux lignes du backtest |
| Graphes sparse | Snapshots datés | Arêtes fournies avec provenance et `source_end` |

`B` compte les dates, `N` les places de tickers. Les colonnes et l'ordre des tickers doivent rester explicites dans les artifacts. Le constructeur n'apprend pas de scaler, ne calcule pas les features et ne choisit pas les arêtes.

## Disponibilité et causalité

Une valeur de remplissage n'est pas une observation. Les masques distinguent actif présent, fenêtre temporelle complète, features de nœud valides, contexte marché disponible, séquence marché complète, sentiment couvert, label connu et graphe utilisable. Label inconnu vaut `-1`, jamais Hold. Sentiment neutre observé et absence de source restent distincts.

Les dates sont uniques par ticker. Train, validation et test sont disjoints par date globale, même avec des tickers différents. L'historique antérieur peut compléter une fenêtre, mais ne crée pas de cible hors partition.

Les entrées fondées sur prix peuvent inclure la clôture J uniquement pour une exécution ultérieure. Un graphe ne peut pas lire J+1. Les publications news/fondamentaux exigent `available_at < J minuit UTC`. Les colonnes de marché issues de clôture et les colonnes de publication suivent des règles distinctes, explicites lors de la construction.

Le pilote post-open a son propre contrat plus strict : features terminées à J-1 plus open J, jamais high/low/close J en entrée. Je ne transpose pas automatiquement la convention close-J du benchmark GNN à une décision du matin.

## Sortie des branches et modularité

Chaque branche renvoie un `BranchOutput` : logits `[B,N,3]` en ordre Sell/Hold/Buy, représentation `[B,N,d]`, disponibilité `[B,N]`. Le GNN utilise `node` et les graphes ; il ne consomme pas les embeddings du GRU. Le Transformer de marché lit le contexte global, pas les séquences propres aux actions.

Le routeur et `MultimodalSystem` permettent d'activer séparément branches, gate et fusion. Les modes disponibles dans l'API ne sont pas tous des stratégies entraînées ou validées. Le runner 3D historique reste distinct ; les runners d'ablation raccordent les branches selon leur propre protocole.

Sources : [dataset](../../src/trading_system/data/multimodal.py), [sortie commune](../../src/trading_system/models/multimodal_contract.py), [branches](../../src/trading_system/models/multimodal_branches.py), [routeur](../../src/trading_system/models/multimodal_router.py), [système](../../src/trading_system/models/multimodal_system.py), [intégration FinBERT](news-sentiment-integration.md), [plan complet](../plan/next_steps.md).
