# Repository structure

## Dependency direction

```text
scripts
  -> pipelines
    -> experiments
      -> data / features / labels / models / evaluation / backtest / reporting
```

Lower layers never import `trading_system.pipelines`.

## Source packages

| Package | Ownership |
|---|---|
| `data` | Parquet loading, chronological splits, context windows, fitted scaling. |
| `features` | Compact technical features and full market feature set. |
| `labels` | Sell/Hold/Buy schema and all label generators. |
| `models` | Common probabilistic-classifier contract and model-specific fit/inference. |
| `training` | Cross-model training policies such as class weights. |
| `evaluation` | Classification metrics and probability threshold calibration. |
| `backtest` | Position decoding, next-bar execution, turnover, fees, benchmarks. |
| `reporting` | Plots and formatted experiment summaries. |
| `experiments` | Static runner, expanding walk-forward runner, grid search. |
| `pipelines` | CLI arguments and predefined experiment configurations only. |

## Model contract

Every model returns probabilities in its declared `classes_` order:

```python
class ProbabilisticClassifier(Protocol):
    classes_: np.ndarray

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> FitResult: ...
    def predict_proba(self, X) -> np.ndarray: ...
```

Experiment runner realigns columns to fixed `[Sell, Hold, Buy]` order. Manual NumPy ANN lives only in `models/manual_ann/manual_nn.py`. Scikit-learn-style estimators use `SklearnClassifierAdapter`.

## Runtime directories

- `data/processed/`: source parquet datasets.
- `data/derived/`: generated labels and derived tables.
- `artifacts/runs/`: persisted advanced-backtest runs.
- `artifacts/gridsearch/`: search CSV/JSON outputs.
- `notebooks/`: research notebooks.
