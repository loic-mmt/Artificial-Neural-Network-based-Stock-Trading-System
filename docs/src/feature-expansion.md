# Features étendues et sélection train-only

`--feature-set expanded` ouvre un pool de features supplémentaires, sans imposer une nouvelle architecture. Je choisis les groupes avec `--feature-groups`, puis laisse la sélection ajustée sur train déterminer les colonnes effectivement utilisables.

## Familles

| Groupe | Contenu |
| --- | --- |
| `technical` | Indicateurs OHLCV existants |
| `market` | Tendances, volatilité, liquidité, beta, macro, VIX et calendrier |
| `sector` | Rendements, breadth, rangs relatifs et surprises sectorielles |
| `fundamentals` | Ratios reconstruits à partir de snapshots datés |
| `sentiment` | Scores et nombres d'événements disponibles avant la décision |

Les familles sont indépendantes du labeling, du contexte temporel et de la loss. Le contexte global ETF/VIX du Transformer marché est une entrée séparée, pas simplement le groupe `market` des actions.

## Sans corpus externe

Pour les benchmarks sans news/fondamentaux, j'utilise :

```bash
--feature-set expanded --feature-groups technical,market,sector --no-external-features
```

`--no-external-features` désactive les fondamentaux et sentiments historiques, pas les données de marché déjà présentes. Les snapshots Yahoo actuels ne sont pas répétés dans le passé comme des fondamentaux connus. Les champs externes manquants restent manquants et sont exclus si la couverture train est insuffisante.

Un chemin externe explicite manquant ou invalide doit échouer, pas être ignoré silencieusement. `configs/feature_sources/` contient des schémas d'aide, pas des données réelles. Les dates fiscales ne remplacent pas `available_at`.

## Sélection et remplissage

Couverture, constance, médianes de remplissage et scaler sont ajustés sur train. Le profil overfitting ajoute classement, filtre de corrélation et cap de features. Validation/test gardent colonnes, ordre et paramètres gelés. Walk-forward réajuste ces éléments sur l'historique de chaque bloc.

Un cap de 32 ou 64 est un maximum : je vérifie le nombre réellement retenu. L'ancien benchmark US avait quatre features techniques et 28 sectorielles ; cela ne signifie pas qu'un Transformer avait sélectionné les meilleures 32 parmi 64.

Les agrégats sectoriels utilisent l'univers fourni. Un secteur réduit à une seule action n'apporte pas un contexte indépendant. Le classement de secteur actuel ne corrige pas le biais de survivance ou de membership historique. Les macros prix déjà enrichies et les publications macro PIT ne sont pas équivalentes.

## Sentiment

L'ancien chemin événements/VADER et le nouvel export FinBERT sont distincts. Pour FinBERT, je passe par [l'adaptateur et son manifeste](news-sentiment-integration.md), avec couverture explicite et `available_at < J minuit UTC`. Absence de source n'est pas un sentiment neutre. Ces benchmarks n'ont pas encore validé un gain des news.

Sources : [constructeur](../../src/trading_system/features/expanded.py), [sources externes](../../src/trading_system/data/feature_sources.py), [contrôle du surapprentissage](overfitting-control.md), [contrat multimodal](multimodal-contract.md). Le [détail historique](../papers/old_docs/feature-expansion.md) reste dans papers, inchangé.

Résultats : [familles en classification](../benchmarks/learning/09-feature-families.md), [contexte sous Sharpe](../benchmarks/learning/22-market-sector.md), [interaction US à lancer](../plan/us-feature-gate-interaction.md).
