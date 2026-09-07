# Overfitting Control

The controls are opt-in. Existing experiments remain byte-for-byte configured as
before unless `--overfitting-control` is passed or `overfitting_control` is set on
`ExperimentConfig`.

## What the profile does

- applies compact, regularized defaults to parameters that were not explicitly set;
- selects features using training rows only;
- removes near-constant features, ranks remaining features by training-label
  relevance, greedily removes highly correlated duplicates, and caps feature count;
- records train/validation loss at the restored best checkpoint;
- in multi-seed model comparison, completes every validation run first and opens
  final test only for candidates passing seed coverage, validation-dispersion, and
  train/validation-gap thresholds.

Direct PnL/Sharpe walk-forward training uses unsupervised training variance for
feature ranking because its objective intentionally does not consume labels.
Walk-forward refits the selector inside every historical training window.

Compact defaults are:

| Model | Capacity defaults | Regularization defaults |
|---|---|---|
| Manual ANN | `hidden_size=32` | dropout `0.30`, weight decay `1e-4`, patience `15` |
| RNN/LSTM/GRU | `hidden_size=32` | AdamW weight decay `1e-4`, patience `15` |
| Transformer | `d_model=32`, 1 layer, FFN 64 | dropout `0.30`, AdamW weight decay `1e-4`, patience `15` |

Explicit model JSON/grid parameters win over these defaults. Manual ANN weight
decay is decoupled and applies only to weight matrices, matching AdamW semantics;
biases and reported validation loss are not regularized.

## Model comparison

```bash
python scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --models manual_ann,rnn,lstm,gru,transformer \
  --seeds 1,7,19,42,1337 \
  --overfitting-control \
  --overfitting-max-features 32 \
  --overfitting-max-feature-correlation 0.98 \
  --overfitting-max-seed-std 0.05 \
  --overfitting-max-train-val-gap 0.15
```

`summary.csv` contains `stable` and `train_validation_macro_f1_gap`.
`runs.csv` contains `stable_candidate`, `final_test_evaluated`, best-checkpoint
loss gaps, and selected-feature state. An unstable candidate has no `test_*` or
final `backtest_*` values: test remained sealed.

## Walk-forward and grid search

Add `--overfitting-control` to either command. Feature selection is refitted on
each expanding training history; the resulting state and learning gap are written
to each retrain log. Grid search still ranks trials exclusively on validation.

These controls reduce capacity and reject unstable validation evidence. They do
not prove higher future PnL; thresholds and model defaults still require matched
multi-seed static and walk-forward benchmarks.
