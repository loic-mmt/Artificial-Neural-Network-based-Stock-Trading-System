# 02. Têtes après attention, contexte 60/ATR

Quatre candidats avec même pooling `attention`, même loss Sharpe, même protocole
CV : **36/36 entraînements réussis**. Le MLP testé possède 16 neurones cachés
et un dropout de 0,1.

| Tête | Sharpe moyen | Écart-type | Minimum | Écart au témoin |
| --- | ---: | ---: | ---: | ---: |
| **Linear** | **0,7509** | 0,0705 | **0,6554** | référence |
| MLP | 0,7465 | 0,0692 | 0,5996 | −0,0044 |
| LayerNorm + MLP | 0,7327 | 0,0729 | 0,5732 | −0,0182 |
| LayerNorm + Linear | 0,6891 | 0,1361 | 0,4191 | −0,0618 |

Décision : ne pas ajouter de tête normalisée ou non linéaire à cette version du
GRU. Le gain d'un fold isolé ne compense pas la dégradation moyenne et du pire
fold. Ce test concerne l'attention seule, pas `last_attention + LayerNorm + MLP`.

Source : [rapport 02](../../../artifacts/comparisons/gru-optim/02-attention-head-60-atr/report.json).

Protocole commun : [règles de lecture](README.md).
