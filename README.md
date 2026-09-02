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

See [restructuring plan](docs/restructuring-plan.md) for ownership rules and migration record.

## Inspiration

This repository contains my personal implementation inspired by the paper  
[An Artificial Neural Network-based Stock Trading System Using Technical Analysis and Big Data Framework](https://arxiv.org/abs/1712.09592).

This is not an official reproduction of the original paper, but an independent implementation with my own design choices, code structure, and experiments.
