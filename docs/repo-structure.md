# Repository Structure

- `src/trading_system/`: importable source package
- `scripts/`: CLI entrypoints that add `src/` to `sys.path`
- `data/processed/`: primary local datasets
- `data/derived/`: generated label tables and other derived datasets
- `artifacts/runs/`: backtest run outputs
- `artifacts/gridsearch/`: grid-search result files
- `notebooks/`: exploratory notebooks
