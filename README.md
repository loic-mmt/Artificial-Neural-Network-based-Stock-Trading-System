# Artificial-Neural-Network-based-Stock-Trading-System

## Layout

- `src/trading_system/data/`: loading, chronological splits, context windows, scaling
- `src/trading_system/features/`: technical and full-market feature builders
- `src/trading_system/labels/`: shared label schema, breakout, forward-return, and oracle labels
- `src/trading_system/models/`: common classifier contract, manual ANN, estimator adapters
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
older scores are not directly comparable; the new repeated-seed baseline and
immutable run manifests are still pending.

## Adding another model

Implement `fit(...)` and `predict_proba(...)` from `ProbabilisticClassifier`, or wrap a scikit-learn-style estimator with `SklearnClassifierAdapter`. Pass model into `run_experiment`; feature preparation, thresholds, metrics, and backtest stay unchanged.

See [restructuring plan](docs/restructuring-plan.md) for ownership rules and migration record.

## Inspiration

This repository contains my personal implementation inspired by the paper  
[An Artificial Neural Network-based Stock Trading System Using Technical Analysis and Big Data Framework](https://arxiv.org/abs/1712.09592).

This is not an official reproduction of the original paper, but an independent implementation with my own design choices, code structure, and experiments.
