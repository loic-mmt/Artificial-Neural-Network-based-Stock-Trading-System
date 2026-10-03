# 1 - Reference baseline

**But.** Comparer les cinq architectures avec labels Triple Barrier, features
techniques, cross-entropy et données non nettoyées.

| Modèle | Macro-F1 | Rendement | Sharpe | Runs positifs |
| --- | ---: | ---: | ---: | ---: |
| Manual ANN | 0,467 | −4,36 % | −0,016 | 3/9 |
| RNN | 0,505 | −9,17 % | −0,040 | 2/9 |
| LSTM | 0,504 | −11,79 % | −0,068 | 4/9 |
| **GRU** | **0,524** | −4,79 % | −0,023 | 4/9 |
| Transformer | 0,486 | −7,53 % | −0,076 | 5/9 |

**Décision.** GRU retenu comme architecture principale. Résultat historique non
comparable aux runs nettoyés suivants.

**Source.** [Rapport](../../../artifacts/comparisons/01-baseline/report.json).

Protocole commun : [conventions et limites](README.md).
