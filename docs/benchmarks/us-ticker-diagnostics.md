# Diagnostic des tickers US : 2026-09-29

Cette fiche conserve l'analyse des fréquences 5/10/20 au 29 septembre.
Les runs 40/60 sont maintenant terminés et figurent dans
[la comparaison complète des fréquences](mt5-retraining.md). Je n'étends pas
rétroactivement les comptes de ce diagnostic aux deux nouvelles fréquences.

## Périmètre

Analyse descriptive des prédictions déjà sauvegardées, sans nouvel entraînement.
196 tickers, du 2026-05-13 au 2026-09-25 : 94 dates, 93 rendements open-to-open.
GRU, seed 42, loss combinée PnL/Sharpe, coûts de 5 bps, réentraînement toutes
les 5, 10 ou 20 séances. Comparaison des positions continues et du décodeur
confidence : six variantes par ticker. Les mêmes dates et une seule seed sont
réutilisées : ce ne sont pas six tests indépendants.

Sources principales :

- `artifacts/mt5/benchmarks/retraining-frequency-confidence/every-{005,010,020}/seed-42/positions-confidence.parquet`
- `artifacts/mt5/stocks-us-gru-benchmark11-walkforward-combined-025-seed42/decoder-benchmark/kpis-*.csv`
- `data/processed/mt5_stocks_us_daily_clean.parquet`
- [Diagnostic antérieur des shorts](market-regime-short-filter.md)

Le run every-040 ne contient pas de résultat final comparable et est exclu.
Les anciens résultats CAC40 n'établissent aucune récurrence par ticker US.
Cette période US a déjà été consultée : elle n'est plus un holdout scellé pour
de nouvelles décisions de sélection.

## Récurrence des pertes

15 tickers perdent dans les six variantes, 21 dans cinq, 19 dans quatre.
40 n'ont aucune perte dans les six variantes, ce qui inclut des résultats nuls
quand le décodeur reste flat. Une perte signifie rendement net strictement
négatif, pas simple sous-performance face au buy-and-hold.

Rendements nets du décodeur confidence, non annualisés, en pourcentage :

| Ticker | Every 5 | Every 10 | Every 20 | Diagnostic des positions confidence |
| --- | ---: | ---: | ---: | --- |
| MSTR | −17,90 | −18,24 | −36,28 | Pertes principalement sur les longs |
| MRNA | −33,70 | −17,54 | −10,01 | Longs perdants ; grands sauts haussiers manqués |
| RBLX | −8,49 | −8,49 | −31,78 | Longs perdants |
| COIN | −8,22 | −8,22 | −25,77 | Longs perdants |
| GEV | −13,18 | −9,73 | −23,33 | Longs perdants |
| HUM | −11,79 | −8,55 | −19,00 | Uniquement shorts actifs |
| CAT | −14,62 | −10,99 | −13,59 | Longs perdants |
| PSX | −19,71 | −19,32 | −8,85 | Uniquement shorts actifs |
| AEP | −8,78 | −7,15 | −7,15 | Longs perdants |
| BAC | −7,96 | −7,96 | −5,90 | Uniquement shorts actifs |
| CVX | −13,13 | −5,85 | −5,85 | Uniquement shorts actifs |
| PLTR | −0,73 | −0,73 | −4,00 | Longs perdants |
| KO | −6,26 | −5,68 | −0,95 | Pertes principalement sur les shorts |
| STT | −5,76 | −0,48 | −0,89 | Uniquement shorts actifs |
| TJX | −1,33 | −1,27 | −0,79 | Uniquement shorts actifs |

Le contrôle avec cinq décodeurs à fréquence 20 confirme des pertes pour 13
de ces 15 tickers dans les cinq décodeurs. Exceptions : KO devient flat avec
deadband ; PLTR devient positif (+10,20 %) avec deadband. « Toujours mauvais »
est donc une description du périmètre observé, pas une propriété intrinsèque.

## Ce qui explique effectivement les pertes observées

### 1. Des shorts opposés à la hausse de certains actifs

PSX monte de 45,25 % sur les prix ajustés open-to-open de cette période,
HUM de 30,93 %, BAC de 12,97 % et CVX de 11,66 % (rendements bruts).
Le décodeur confidence ne prend que des shorts sur ces quatre tickers.
Leurs pertes ne viennent donc pas d'une incapacité générale du marché à monter.

Sur les 196 tickers, la corrélation de rang entre fraction de périodes short
moyennée sur les six variantes et rendement net moyen vaut −0,31, contre
+0,35 pour la fraction long. C'est une association descriptive, non causale.

Retirer les shorts des positions confidence déjà enregistrées ramène à zéro
les résultats de BAC, CVX, HUM, PSX, STT et TJX. Il s'agit d'une abstention,
pas d'une stratégie rentable. Cette ablation est faite après observation des
résultats et ne valide pas un filtre de production.

### 2. Le timing et l'asymétrie gains/pertes des longs

Même en supprimant tous les shorts, AEP, CAT, COIN, GEV, MRNA, MSTR, PLTR et
RBLX restent déficitaires aux trois fréquences. Un filtre de régime short ne
résout donc pas leur problème.

À fréquence 20, MSTR n'a que 4 périodes actives gagnantes contre 16 perdantes.
Pour RBLX, les périodes gagnantes et perdantes sont aussi nombreuses (20/20),
mais le gain brut moyen est +2,38 %, contre −4,02 % pour une période perdante.
Ce sont des périodes open-to-open exposées, pas des trades clôturés.

MRNA est un cas extrême à auditer : le parquet contient +84,04 % entre opens
ajustés les 18 et 19 août, puis +29,41 % le lendemain. Le décodeur confidence
every-20 reste flat sur ces deux intervalles, tandis que la position continue
est légèrement short. Le buy-and-hold brut atteint +269,61 %, alors que les
longs sélectionnés perdent. Ces sauts expliquent une partie de l'écart ; leur
validité économique et les corporate actions n'ont pas été vérifiées auprès
d'une source externe.

### 3. Les frais ne sont pas le moteur principal

Sur 588 chemins ticker/fréquence continus, 280 perdent net et 271 de ces 280
perdent déjà avant coûts. Pour confidence : 202 pertes nettes, dont 198 déjà
présentes avant coûts. Les 5 bps aggravent donc les pertes, mais ne les créent
que dans une petite minorité des chemins perdants.

### 4. Volatilité, liquidité et historique : pas de filtre évident

Les statistiques pré-évaluation utilisent au plus 253 clôtures antérieures
au 13 mai 2026 ; volatilité annualisée à 252 périodes. Sur les 196 tickers,
corrélations de Spearman avec le rendement moyen des six variantes :

| Statistique | Corrélation |
| --- | ---: |
| Volatilité avant évaluation | +0,04 |
| Fréquence des variations quotidiennes > 5 % avant évaluation | +0,04 |
| Volume monétaire quotidien médian avant évaluation | −0,08 |
| Longueur d'historique disponible | −0,09 |
| Autocorrélation des rendements avant évaluation | −0,004 |
| Écart initial du prix à sa moyenne mobile 60 séances | −0,20 |

La volatilité pré-évaluation médiane des 15 perdants récurrents est de 33,4 %,
contre 26,5 % pour les 40 tickers sans perte. Mais la relation globale est
quasi nulle et diffère selon le décodeur : corrélation volatilité/rendement
−0,15 en continu, +0,08 avec confidence. Exclure tous les actifs volatils
n'est pas justifié par ces données.

L'écart initial à la moyenne mobile vaut +3,77 % en médiane chez les perdants
récurrents, contre −2,39 % chez les tickers sans perte. C'est une piste de
sur-extension/timing à tester avant décision, pas une règle validée.

L'énergie est le secteur au rendement moyen le plus faible (−1,83 %, moyenne
des tickers et des six variantes), mais les 15 perdants couvrent neuf secteurs.
Les classifications sectorielles sont actuelles, pas point-in-time.

### 5. Qualité des données : contrôles élémentaires rassurants, audit incomplet

Chaque ticker a les mêmes 94 dates dans les fichiers de positions, et 94 lignes
dans le parquet brut nettoyé sur la période : aucun doublon de date ni violation
`low <= close <= high` détectée. Historique avant évaluation : au moins 533
lignes par ticker. Ces contrôles n'excluent ni prix aberrants économiquement,
ni erreurs d'ajustement, ni biais de survivance ; MRNA mérite un contrôle ciblé.

## Suite de recherche

Ne pas construire une blacklist à partir de ces pertes déjà observées. Comparer
sur de nouvelles périodes et plusieurs seeds : (1) référence long/short,
(2) long-only, (3) filtre causal de shorts, puis un diagnostic séparé des longs
(amplitude des pertes, timing, écarts à la moyenne mobile). Vérifier les sauts
de prix avant d'en tirer une règle sur la volatilité ou les événements.

Calcul reproductible local : `tmp/us-ticker-diagnostics/analyze.py` ; exports
`paths.csv`, `tickers.csv`, `decoder_crosscheck.csv` et
`long_only_diagnostic.csv` dans le même dossier. Rendements reconstruits avec
les positions exécutées sans délai supplémentaire, coûts de turnover incluant
la liquidation finale. Concordance à 1e-10 avec les rendements continus des
trois fichiers KPI existants. Les sommes de contributions long/short sont
arithmétiques ; elles ne s'additionnent pas aux rendements composés affichés.
