# Artificial-Neural-Network-based-Stock-Trading-System

## Layout

- `src/trading_system/`: importable package with data, features, labels, pipelines, and backtest code
- `scripts/`: runnable entrypoints
- `data/processed/`: local market datasets
- `data/derived/`: generated labels and derived tables
- `artifacts/`: grid-search outputs and persisted backtest runs
- `notebooks/`: exploratory notebooks

## Quick Start

Install deps from [requirements.txt](/Users/loic/Documents/Code/DL/Artificial-Neural-Network-based-Stock-Trading-System/requirements.txt), then run scripts from repo root:

```bash
python scripts/download_market_data.py
python scripts/run_single_ticker.py
python scripts/run_walkforward.py
python scripts/run_gridsearch_walkforward.py
```

## Inspiration

This repository contains my personal implementation inspired by the paper  
[An Artificial Neural Network-based Stock Trading System Using Technical Analysis and Big Data Framework](https://arxiv.org/abs/1712.09592).

This is not an official reproduction of the original paper, but an independent implementation with my own design choices, code structure, and experiments.
