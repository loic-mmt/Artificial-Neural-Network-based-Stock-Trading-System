# 5 - Triple Barrier between-event policy

**But.** Comparer labels d'action `hold` et cibles de position `flat`/`carry`.

| Politique | Modèle | Macro-F1 | Rendement historique | Runs positifs |
| --- | --- | ---: | ---: | ---: |
| **hold** | **GRU** | **0,529** | **+6,48 %** | **6/9** |
| hold | RNN | 0,500 | −8,13 % | 3/9 |
| hold | Transformer | 0,501 | −16,23 % | 1/9 |
| flat | **GRU** | **0,529** | −0,24 % | **6/9** |
| flat | RNN | 0,500 | −9,32 % | 1/9 |
| flat | Transformer | 0,501 | −11,40 % | 2/9 |
| carry | GRU | 0,173 | **+6,77 %** | **4/9** |
| carry | RNN | 0,174 | −1,95 % | 3/9 |
| carry | Transformer | **0,177** | −15,53 % | 3/9 |

**Décision.** `hold`. `carry` change fortement la sémantique et devient peu
apprenable ; `flat` ne gagne rien face à `hold`.

**Sources.** `artifacts/comparisons/ohlc-clean/05-policy-*`.

Protocole commun : [conventions et limites](README.md).

## Rapports sauvegardés

- [05-policy-carry](../../../artifacts/comparisons/ohlc-clean/05-policy-carry/report.json)
- [05-policy-flat](../../../artifacts/comparisons/ohlc-clean/05-policy-flat/report.json)
- [05-policy-hold](../../../artifacts/comparisons/ohlc-clean/05-policy-hold/report.json)
