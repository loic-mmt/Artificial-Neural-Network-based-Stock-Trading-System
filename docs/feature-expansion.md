# Feature expansion

`--feature-set expanded` appends a broader, opt-in input space to the existing
benchmark workflow. `technical` remains the default for comparison presets;
walk-forward keeps its existing `market` default unless explicitly overridden.
The legacy `market` feature set is unchanged and still expects its external columns.

## Run and ablate

```bash
python scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --preset multi_ticker_long_short \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --label-method triple-barrier \
  --feature-set expanded
```

Same feature flags work with `scripts/run_walkforward.py` and
`scripts/run_gridsearch_walkforward.py`. For a controlled family ablation:

```bash
--feature-set expanded --feature-groups technical,market,sector --feature-min-coverage 0.5
```

Families: `technical`, `market`, `sector`, `fundamentals`, `sentiment`.
Default: all five, currently 122 unique candidate columns before optional FracDiff.
Model architecture, labels and execution are independent of feature selection.
Sample weighting and FracDiff can still be enabled separately.

## Features

- Technical: the existing 15 OHLCV indicators.
- Market: longer-horizon returns/trends/volatility, liquidity, market beta and
  correlation, breadth, VIX, rates/spreads, oil/DXY/gold, calendar indicators.
- Sector: existing sector returns/volatility/breadth and relative ranks, plus
  60-observation sector momentum, a daily sector-return z-score relative to the
  **previous** 20 observations, and a surge flag (`z > 2` and breadth ≥ 70%).
- Fundamentals: P/E, P/B, EV/EBITDA, profit margin, return on equity/assets,
  revenue and EPS growth, debt/equity, age since publication, capitalization,
  book-to-market, earnings yield and share turnover.
- Sentiment: average news score/count over 7 calendar days, and earnings-call
  score/count over 30 calendar days. Scores use only already published events.

The ratios use unadjusted close, contemporaneous shares and compatible currency
units. P/E requires positive EPS, P/B and ROE positive equity, EV/EBITDA positive
EV and EBITDA; undefined ratios remain missing. Profit margin and earnings yield
can be negative. Growth uses supplied TTM and prior-year TTM values, not a guessed
row shift over potentially revised quarterly statements. EPS growth divides the
change by the absolute prior EPS; zero prior values remain missing. ROE/ROA use
the latest supplied equity/assets, not an average balance-sheet denominator.

## Do not backfill current fundamentals into history

The current CAC40 parquet has Yahoo `.info` snapshots repeated through history.
`expanded` deliberately ignores those undated firm fields. It requires
publication-dated snapshots to construct fundamental features. This protection
does not retrofit the legacy `market` mode.

Prepare a CSV or parquet with:

| Field | Meaning |
|---|---|
| `ticker` | Exact identifier matching price rows |
| `available_at` | Actual timestamp when the data became public, not fiscal period end |
| `shares_outstanding`, `equity`, `assets`, `debt`, `cash` | Contemporaneously known balance-sheet/share values |
| `revenue_ttm`, `net_income_ttm`, `ebitda_ttm`, `eps_ttm` | Values known at that publication |
| `revenue_ttm_prev_year`, `eps_ttm_prev_year` | Comparable prior-year figures known then |

Only available fields are needed; missing ratios will be reported/excluded.
Each record is a complete snapshot of known fields, not a sparse update. Duplicate
ticker/publication timestamps are rejected. Revisions must keep their actual later
publication dates. Shares must be consistent with corporate actions and raw prices;
no split/currency reconciliation is inferred automatically.

## News and earnings-call inputs

Provide `ticker`, `available_at`, `kind` (`news` or `earnings`) and either:

- `score` already in `[-1, 1]`, produced without future information; or
- `text` and `language=en`, scored locally by VADER 3.3.2.

Optional `event_id` deduplicates each ticker/kind event, retaining its earliest
publication. Without IDs each row counts as a distinct event. VADER scores whole
news texts; earnings-call scores average punctuation-delimited sentence scores.
This is a lightweight English lexical baseline, not a finance-specialized or
multilingual model. Its financial-language accuracy still needs evaluation.
VADER is [MIT-licensed](https://github.com/cjhutto/vaderSentiment) and requires no
paid API, access key, or model-server request. Its lexicon ships with the package.

For a ticker/kind with no source events, sentiment stays missing. Before its
first observed publication, feed coverage is unknown (missing). Thereafter an
empty rolling window produces score/count zero. This assumes supplied event-feed
coverage is complete after that first publication; gaps in the feed are not proof
of no news. These are document averages, not confidence-weighted sentiment.

## Enrich a new dataset

### Work now without downloads; attach sources later

No download or directory scan occurs. With `--feature-set expanded` and no source
arguments, existing embedded source columns are used if present; otherwise the
fundamental/sentiment families remain missing and train-only selection excludes
them. Technical, market and sector features remain available. Optional does not
mean that an explicitly supplied missing, empty or malformed file is ignored:
that is an error, so a typo cannot silently change an experiment.

Run the benchmark now, explicitly ignoring historical fundamental/sentiment
columns even if the dataset was previously enriched:

```bash
python scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --preset multi_ticker_long_short \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --label-method triple-barrier \
  --feature-set expanded --no-external-features
```

Later, replace `--no-external-features` with one or both source arguments:

```bash
  --fundamentals data/external/fundamentals_pit.csv \
  --sentiment data/external/sentiment_events.csv
```

These flags also work with walk-forward and walk-forward grid search. Source
flags require `expanded` in model pipelines; combining paths with
`--no-external-features` is rejected. The latter disables only historical
fundamentals and sentiment, not existing macro/sector inputs. Legacy `market`
mode is unchanged and does not gain point-in-time guarantees.

Header-only templates live in `configs/feature_sources/`. They are schema aids,
not datasets: fill them with real records before use. `data/external/` is ignored
by Git. Fundamentals must use compatible currency units (not a mixture of euros
and millions of euros), actual shares (not ADR counts), and original publication
versions. Keep source URLs and source documents for auditing. A later restatement
must get its later publication timestamp. Unknown publication dates are not
replaced with fiscal year-end or an arbitrary reporting lag.

Each run prints a `feature_sources` report: `absent`, `embedded`, `file`, or
`disabled`, matched row coverage per ticker, plus file path, SHA-256 and source
publication range for supplied files. Comparison reports and per-run manifests
save it; expanded selector diagnostics also carry it into walk-forward retraining
logs. Grid-search JSON includes it. Coverage describes the entire input, not the
training selection criterion; eligibility remains training-only. A source loaded
successfully may still have zero overlap or insufficient training coverage.

Pre-enriched sentiment aggregates have no per-event timestamps to recheck:
their causal construction remains the provider's responsibility. For auditable
new runs, supply dated events directly. The event join still assumes a complete
feed after its first event; do not use a sparse scraped sample as proof of no news.

### Optional persistent enrichment

```bash
python scripts/enrich_feature_data.py \
  --data data/processed/cac40_daily.parquet \
  --fundamentals data/external/fundamentals_pit.csv \
  --sentiment data/external/sentiment_events.csv \
  --output data/processed/cac40_daily_enriched.parquet
```

The external paths are examples: no historical filings or news corpus is bundled
or fabricated. Either source is optional. The script refuses to overwrite an
existing output. Run the benchmark with the new parquet and `--feature-set expanded`.
Enrichment is offline; no paid API or automatic latest-data backfill is involved.
The enrichment script also accepts no sources (an unchanged-price copy with
availability diagnostics) or `--no-external-features` (removing embedded historical
fundamentals/sentiment). It always requires a new output path. The report is
embedded in the parquet's pandas attributes as well as printed.

Fundamentals use a backward as-of join strictly before each price-row timestamp,
with a maximum age of 550 calendar days. Sentiment windows also exclude events
exactly at the row timestamp. For daily dates represented at midnight, same-day
publications become available only at the following bar. Timezone-naive source
timestamps are interpreted as UTC; callers must provide the correct timezone.

Macro inputs already present in price data are reused. Conflicting non-null global
macro values for one date raise instead of choosing an arbitrary ticker. Missing
values may be shared from another ticker **on that same date**, never a future
date. Publication/revision vintages, actual market closing times and historical
sector membership remain caller responsibilities. Sector aggregates use only the
input universe, including the subject stock; they are not full external sector
indices. A sector with one selected ticker has no independent peer signal.

## Coverage, training, and reproducibility

After the 20-return warmup (and optional FracDiff warmup), each candidate is
screened using **training data only**: coverage ≥ `feature-min-coverage` and more
than one distinct observed value. This is coverage/constant-column filtering, not
supervised importance selection. Remaining missing values are filled with train
medians; the train scaler then sees the selected columns. Test and validation use
the exact frozen columns and medians, even if a previously missing field appears
later. Walk-forward repeats selection inside each historical training partition.

Artifacts save `experiment_parameters.feature_selection`: selected features,
train coverage, per-ticker coverage where available, and exclusion reasons.
Comparison CSVs and walk-forward retraining logs also contain the selection.
The notebook displays the first loaded artifact's coverage report. Configuration
hashes include feature set/groups/coverage thresholds; artifact hashes include
the fitted selection. Older artifacts without this metadata still load.

On a read-only check of AIR.PA/BNP.PA (first 2,000 observations each), 87 of 122
candidates passed the default filter. Undated fundamentals and absent sentiment
were excluded explicitly; this was a coverage diagnostic, not a model benchmark.

Next: source genuine historical publication-dated inputs, inspect dropped fields
and feed completeness, then run matched family ablations across seeds/folds.
No improvement in learning or out-of-sample P&L is established by implementation.
