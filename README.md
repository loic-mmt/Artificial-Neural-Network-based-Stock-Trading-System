# Artificial-Neural-Network-based-Stock-Trading-System

## Layout

- `src/trading_system/data/`: loading, chronological splits, context windows, scaling
- `src/trading_system/features/`: technical and full-market feature builders
- `src/trading_system/labels/`: shared label schema, breakout, forward-return, and oracle labels
- `src/trading_system/models/`: 3D model contract, registry, NumPy ANN, RNN, LSTM, GRU, Transformer
- `src/trading_system/artifacts/`: safe checksummed model and run serialization
- `src/trading_system/experiments/`: model-neutral static, walk-forward, and search runners
- `src/trading_system/pipelines/`: thin CLI/configuration wrappers
- `src/trading_system/backtest/`: positions, timing, fees, benchmarks, and advanced backtests
- `scripts/`: runnable entrypoints
- `data/processed/`: local market datasets
- `data/derived/`: generated labels and derived tables
- `artifacts/`: grid-search outputs and persisted backtest runs
- `notebooks/`: exploratory notebooks

## Quick Start

Install deps from [requirements.txt](requirements.txt), then run scripts from repo root:

```bash
python scripts/download_market_data.py
python scripts/run_single_ticker.py
python scripts/run_walkforward.py
python scripts/run_gridsearch_walkforward.py
python scripts/run_model_comparison.py --device cpu
```

Run tests with development dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Validation and final test

Grid searches select parameters using validation only. After selection, only
the frozen winner is evaluated on final test. Static search retains the fitted
model and decision policy; walk-forward search retains the chosen retraining
configuration and records each deterministic ANN chunk seed.

Walk-forward search CSVs contain `val_*` metrics and a `selected` flag. Its JSON
report separates the validation ranking (`top`) from the winner's `final_test`.
Oracle labels are diagnostic-only and are not accepted by grid search.

Forward-return labels crossing split boundaries are excluded from training and
classification scoring. Their feature rows remain available as context, and
backtests retain the complete prediction interval. These integrity fixes mean
older scores are not directly comparable. Static and comparison runs can now
persist reloadable, checksummed artifacts and immutable manifests.

Current/frozen CAC40 and market-universe datasets are survivorship-biased for
historical evaluation. Their CLIs and manifests display this limitation.

## Adding another model

Register a factory returning `ProbabilisticSequenceClassifier`. It receives a
`ModelBuildContext`, consumes `(samples, context, features)` arrays, and returns
probabilities in `[Sell, Hold, Buy]` order. No runner branch is needed; feature
preparation, train-only scaling, thresholds, metrics, and backtest stay shared.

PyTorch architectures are optional:

```bash
python -m pip install -e ".[neural]"
python scripts/run_walkforward.py --model gru --device cpu \
  --model-config '{"hidden_size":64,"epochs":100}'
```

Fair multi-model, multi-seed comparison (five seeds by default):

```bash
python scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --models manual_ann,rnn,lstm,gru,transformer \
  --device cpu
```

Optional overfitting controls add compact regularized defaults, train-only feature
selection, learning-gap diagnostics, and a validation-only multi-seed stability
gate before final test:

```bash
python scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --models manual_ann,rnn,lstm,gru,transformer \
  --overfitting-control
```

See [docs/overfitting-control.md](docs/overfitting-control.md) for thresholds,
saved fields, and walk-forward usage.

The controlled breakout baseline can be configured without changing a preset:

```bash
python scripts/run_model_comparison.py \
  --label-method breakout \
  --label-window 20 \
  --label-buy-buffer 0.001 \
  --label-sell-buffer 0.001 \
  --label-alternating
```

Use `--no-label-alternating` to retain repeated same-side breakout actions.

The fixed-horizon forward-return baseline supports asymmetric thresholds:

```bash
python scripts/run_model_comparison.py \
  --label-method forward-return \
  --label-horizon 5 \
  --label-buy-threshold 0.005 \
  --label-sell-threshold 0.0075
```

Targets whose horizon crosses a split boundary are excluded from fitting and
classification metrics.

The volatility-adjusted M2 labeler emits persistent `Short`/`Flat`/`Long`
target positions. Its default `long-flat` mode is suitable for comparison with
buy and hold:

```bash
python scripts/run_model_comparison.py \
  --label-method volatility-position \
  --label-horizon 10 \
  --label-vol-window 20 \
  --label-long-threshold 1.0 \
  --label-short-threshold 1.5 \
  --label-exit-threshold 0.25 \
  --label-min-hold 5 \
  --label-cooldown 0 \
  --label-cost-bps 5 \
  --label-position-mode long-flat
```

Use `--label-position-mode long-short` to enable Short targets. Warm-up rows and
targets crossing a temporal split boundary are excluded from fitting and metrics.

The path-dependent triple-barrier labeler supports either every eligible row or
a causal symmetric CUSUM event filter:

```bash
python scripts/run_model_comparison.py \
  --label-method triple-barrier \
  --label-max-holding 10 \
  --label-vol-window 20 \
  --label-profit-barrier 1.0 \
  --label-stop-barrier 0.75 \
  --label-event-filter cusum \
  --label-cusum-threshold 0.5 \
  --label-cost-bps 5 \
  --label-between-events hold
```

`hold` emits action labels (`Sell/Hold/Buy`). `flat` and `carry` emit explicit
target positions (`Short/Flat/Long`). Each event records its first barrier,
realized return, event ID, and end date.

Choose the horizontal barrier scale with
`--label-volatility-estimator rolling_std|atr|bollinger` (comparison,
walk-forward, and walk-forward grid search):

- `rolling_std` (default): population standard deviation of daily percentage
  returns. Preserves the previous labeling behavior.
- `atr`: simple rolling mean of true range, divided by the current label price.
  Requires positive finite `high`, `low`, and `close` with `low <= close <= high`.
  OHLC is aligned to the label price using `price / close`, so raw OHLC can be
  used with `adj_close` without creating artificial split gaps.
- `bollinger`: normalized half-width of ±2σ price bands,
  `2 * rolling_std(price, ddof=0) / rolling_mean(price)`. Requires only the
  label price, not OHLC. These are event-centered symmetric barriers scaled
  by band width, not touches of moving Bollinger levels.

All three use `--label-vol-window`, historical data through the event's close,
and the same warmup. The scale is frozen at event entry; barrier touches remain
**close-based**, not intraday high/low touches. CUSUM continues to use daily
return volatility, independent of the barrier estimator. The estimated cost is
added after scaling. `barrier_scale` records the pre-multiplier, pre-cost width.
The estimator is saved in run configuration/label metadata and included in
configuration hashes; the comparison notebook reconstructs it automatically.
Old artifacts without the setting retain `rolling_std`.

For an estimator comparison, repeat the same benchmark with only
`--label-volatility-estimator` changed. Multiples of ATR, return volatility,
and Bollinger width are not numerically equivalent: first compare fixed settings,
then tune each estimator on validation only. Wider barriers are not a guarantee
of better learning or out-of-sample P&L.

See [restructuring plan](docs/restructuring-plan.md) for ownership rules and migration record.

Optional fractional-differentiation feature: append `--fracdiff` to model
comparison, walk-forward, or walk-forward grid search. Order selection uses
training history only; `--fracdiff-order 0.5` chooses a fixed order instead.
Disabled by default. See [FracDiff configuration and diagnostics](docs/fracdiff.md).

Optional event sample weighting with Triple Barrier: append
`--sample-weighting net_return` (also `volatility` or `uniqueness`). Applies to
Manual ANN and the shared PyTorch trainer; disabled by default. See
[sample weighting and validation status](docs/sample-weighting.md).

Direct net-P&L/Sharpe position training is available for static loss comparison,
single-ticker walk-forward and walk-forward grid search. Cross-entropy remains
the default. Use `--loss-objective pnl|sharpe`; financial grid objectives include
`net_return` and `regularized_sharpe`. See [financial losses](docs/financial-loss.md).

Expanded inputs: `--feature-set expanded`, optionally
`--feature-groups technical,market,sector` for family ablations. Includes dated
fundamental ratios, local news/earnings sentiment, market context and sector-surge
signals, with train-only coverage selection. Historical sources are not bundled;
undated Yahoo firm snapshots are excluded. See [feature expansion](docs/feature-expansion.md)
and `scripts/enrich_feature_data.py --help`.

## Optional financial objectives

Optional position-objective comparison: `scripts/run_loss_comparison.py` supports
`--losses cross_entropy pnl sharpe`, with proportional transaction costs and
validation-only selection by default. All five architectures are supported.
The previous cross-entropy trainer and discrete backtest remain available unchanged;
the new report also compares every loss using the same continuous position decoder.
See [financial loss definitions, comparison and final-test protocol](docs/financial-loss.md).

## Optional purged cross-validation

Optional nested purged cross-validation: add `--cv-folds 3 --cv-gap-bars 5` to
`scripts/run_model_comparison.py` or `scripts/run_loss_comparison.py`. Selection
uses all outer folds/seeds; final test stays sealed unless `--cv-final-test` is
supplied. Omit these flags to retain the previous workflow. See
[purging, embargo semantics and comparison protocol](docs/purged-cv.md).

## Inspiration

This repository contains my personal implementation inspired by the paper  
[An Artificial Neural Network-based Stock Trading System Using Technical Analysis and Big Data Framework](https://arxiv.org/abs/1712.09592).

This is not an official reproduction of the original paper, but an independent implementation with my own design choices, code structure, and experiments.
