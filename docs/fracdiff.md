# Fractional differentiation

FracDiff is an opt-in **feature**, not a labeling method. It appends
`fracdiff_log_price` to either the technical or market feature set and leaves
prices, labels, and backtest execution unchanged.

## Usage

Append these options to the model-comparison, walk-forward, or walk-forward
grid-search command:

```bash
--fracdiff --fracdiff-order auto
```

For an explicitly chosen fractional order:

```bash
--fracdiff --fracdiff-order 0.5 --fracdiff-threshold 0.001
```

Other options: `--fracdiff-max-terms 1000`, `--fracdiff-min-samples 64`,
`--fracdiff-adf-pvalue 0.05`. Feature parameters without `--fracdiff` are rejected
unless the preset already enables the feature. Default: disabled.

Python configuration uses `ExperimentConfig(fracdiff=FracDiffConfig(...))` from
`trading_system.features.fracdiff`. Custom sorted unique candidates in `(0, 1)`
can be supplied through `FracDiffConfig(candidates=(0.2, 0.4, 0.6, 0.8))`.

## Definition and causal boundaries

The input is `x[t] = log(price[t])`, using the configured label-price column.
Coefficients are `w[0] = 1`, `w[k] = -w[k-1] * (d-k+1) / k`.
The fixed-width output is `sum(w[k] * x[t-k])`. The first coefficient smaller
than the absolute threshold is excluded, along with later coefficients. This is
a coefficient threshold, not a bound on total discarded weight. The maximum
term count is a safety bound: exceeding it raises rather than silently truncating.

The first `len(weights)-1` rows are unknown, not backfilled. Training drops
incomplete feature rows before scaling. Validation/test use the available raw
past, so the filter does not restart at split boundaries. All operations include
the current close but never a future close; execution delay must remain suitable
for close-derived features. Prices must be finite and positive, dates unique per
ticker, and ticker identifiers non-missing strings. No implicit gap filling.

## Train-only automatic order selection

1. Use only the raw training partition, separately for each ticker.
2. Build weights for each candidate `0.1` through `0.9`; discard candidates whose
   weights exceed the cap or leave fewer than `min_samples` training observations.
3. Compare remaining candidates on the same trailing training dates, starting
   after the largest feasible warmup.
4. Compute ADF with a constant and AIC-selected lag count. Choose the smallest
   candidate with p-value at or below the configured threshold. Also record
   correlation with the original log price over exactly those dates.
5. Freeze the order and coefficients for validation/test. At each walk-forward
   retraining, repeat only on the new training subpartition, excluding its
   internal validation partition and its prediction chunk.

Automatic mode fails if no candidate qualifies. A fixed order bypasses the ADF
acceptance criterion, but still records diagnostics and enforces sufficient
training history. Constant series have an undefined ADF diagnostic. There is no
fallback to `d=1`. ADF selection is a heuristic with unadjusted candidate tests;
it does not establish stationarity in future regimes or better model performance.

## Reproduction and comparison

Manifest `experiment_parameters.fracdiff_state` stores coefficients, chosen
orders, full candidate diagnostics and fit dates. This state is included in the
artifact hash. Comparison `runs.csv` also contains a JSON `fracdiff_state` column,
including when model artifacts are disabled. Walk-forward logs contain one state
per retraining; grid-search exports record configuration and retraining states.
The existing benchmark notebook displays the first loaded artifact's diagnostics.

`FracDiffTransformer.from_state_dict(state)` restores the frozen transform.
To reproduce features, supply raw history including at least `len(weights)-1`
preceding observations per ticker. The transform never refits during inference;
unseen tickers fail explicitly. Bundle prediction still expects feature windows,
not raw OHLC: apply the stored transform before window creation.

Compare feature-off versus feature-on with identical datasets, labels, models,
seeds, splits and execution settings. Watch training-row loss from warmup and
per-ticker feature distributions. Tune only on validation; keep final test sealed.
`statsmodels` already exists in project dependencies; no new dependency is added.

## Local verification status

The regression suite and FracDiff integration tests were exercised, with an ADF
test double for the latter. The real statsmodels ADF smoke test currently blocks
while importing local SciPy files, before running any statistical calculation.
No dependency was installed or altered. After the environment is available, run
`python -m pytest tests/test_fracdiff.py -q` to validate the complete path with the
real ADF implementation. No trading benchmark has been run for this feature.

Implementation references: [fixed-width fractional differentiation](https://mlfinpy.readthedocs.io/en/latest/_modules/mlfinpy/util/frac_diff.html)
and [statsmodels ADF API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html).
