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
| `models` | Registry, common 3D sequence contract, NumPy ANN and optional PyTorch architectures. |
| `training` | Cross-model training policies such as class weights. |
| `evaluation` | Classification metrics and probability threshold calibration. |
| `backtest` | Position decoding, next-bar execution, turnover, fees, benchmarks. |
| `reporting` | Plots and formatted experiment summaries. |
| `experiments` | Static/walk-forward runners, leakage-safe search and fair comparison. |
| `pipelines` | CLI arguments and predefined experiment configurations only. |
| `artifacts` | Safe NPZ/JSON states, checksums, manifests and compatibility validation. |

## Model contract

Every model returns probabilities in its declared `classes_` order:

```python
class ProbabilisticSequenceClassifier(Protocol):
    model_name: str
    classes_: np.ndarray

    def fit(self, X_train_3d, y_train, *, X_val=None, y_val=None) -> FitResult: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def state_dict(self) -> dict[str, object]: ...
```

The runner owns canonical `(N, T, F)` windows and realigns columns to fixed
`[Sell, Hold, Buy]` order. The NumPy ANN flattens only inside its sequence
adapter. PyTorch models share one trainer, early stopping, device handling and
deterministic seed policy.

## Reproducible outputs

- Static pipelines can save model state, scaler, ordered features, decision
  policy, metrics, history and immutable run metadata.
- `scripts/run_model_comparison.py` writes CSV/JSON reports and reloadable
  per-run artifacts for identical architecture/seed coverage.
- Model search ranks validation only and opens final test once for the frozen
  winner.

## Runtime directories

- `data/processed/`: source parquet datasets.
- `data/derived/`: generated labels and derived tables.
- `artifacts/runs/`: persisted advanced-backtest runs.
- `artifacts/gridsearch/`: search CSV/JSON outputs.
- `artifacts/comparisons/`: multi-model CSV/JSON reports and run artifacts.
- `notebooks/`: research notebooks.
