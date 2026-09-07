# Sharpe and net P&L objectives

`scripts/run_loss_comparison.py` runs a separate, opt-in position experiment for
Manual ANN, RNN, LSTM, GRU and Transformer. Existing scripts, model configurations,
cross-entropy trainers, discrete execution and old artifacts are unchanged.
This implements the roadmap's separate position-output experiment; static
chronological splits are supported. Walk-forward/grid-search integration is not
part of this experiment and is not enabled by its flags.

## Compare with the previous version

```bash
python scripts/run_loss_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --preset multi_ticker_long_short \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --label-method triple-barrier \
  --feature-set expanded --no-external-features \
  --models manual_ann --seeds 1,7,19 \
  --losses cross_entropy pnl sharpe \
  --loss-cost-bps 5 \
  --output-dir artifacts/comparisons/losses-validation
```

This runs **validation only**. Architecture parameters are supplied through the
same `--model-parameter-sets` JSON as the model comparator. For a small smoke run,
add `--model-parameter-sets '{"manual_ann":[{"epochs":3,"hidden_size":8}]}'` and
`--seeds 1`. Epochs, learning rates, batch sizes and early-stopping settings remain
the model's explicit configuration; full-path optimization may need different
learning rates selected on validation.

The comparison has two views of the unchanged cross-entropy control:

| Output | Interpretation |
| --- | --- |
| `legacy_val_*`, final-test `legacy` | Original discrete policy and cash-fee backtest, including its original metric conventions. |
| `val_*`, final-test `continuous` | Shared continuous position decoder and proportional costs, identical for CE, P&L and Sharpe. Use this view to compare objectives. |

The CE control invokes the existing `run_validation_experiment` unchanged,
including class weights, label masks, threshold calibration, early stopping and
checkpoint restoration. Regression tests compare its weights, fitted scaler,
loss history and old backtest against that runner exactly. Old workflows remain
available through `scripts/run_model_comparison.py`; no flag is needed to retain
the earlier behavior.

Financial objectives do not attempt to reproduce labels. Their three softmax
channels parameterize positions; they are not calibrated class probabilities.
CE label accuracy is therefore not an objective-selection criterion here.

## Objective and execution definitions

For softmax output `p`, the fixed decoder is:

- long/short: `q[t] = p[Buy] - p[Sell]`, within `[-1, 1]`;
- long-only: `q[t] = p[Buy]`, within `[0, 1]`.

The resolved experiment position mode is used, including label methods that
specify a target-position mode. No threshold is optimized for this decoder.

For each asset, `e[t] = q[t-delay]` and
`r[t] = price[t+1] / price[t] - 1`. Signals use features through close `t`;
the default delay of one executes at the **next close**, earning the subsequent
close-to-close return. Delays below one are rejected. An asset's turnover is
`abs(e[t] - e[t-1])`, with an initial flat position. The last held exposure is
liquidated at the final observed close, adding `abs(e[last])` to the last
turnover. No new position is entered at a terminal row without a future return.
Positions and costs never cross train/validation/test boundaries.

`net[t]` is the equal-weight asset mean of
`e[t] * r[t] - (cost_bps / 10000) * turnover[t]`.

- **P&L loss**: `-mean(net)`. This is arithmetic, capital-normalized net P&L per
  period, not negative compounded terminal wealth. Reported `net_pnl` compounds
  `1 + net[t]` and multiplies by initial capital.
- **Sharpe loss**: `-sqrt(annualization) * mean(net) /
  sqrt(mean((net - mean(net))**2) + epsilon**2)`. Defaults: annualization 252,
  epsilon `1e-4` in per-period return units. Both training and early stopping use
  this regularized denominator. Reports also expose unregularized `net_sharpe`;
  an all-flat path returns zero, and nonzero constant returns yield JSON null.
- `abs` turnover uses its standard zero subgradient at equality.

Use `--loss-cost-bps`, `--loss-sharpe-epsilon`, and `--loss-annualization` to set
these quantities. Loss costs are distinct from estimated `--label-cost-bps`
used to construct labels and from legacy `fee_per_trade` cash fees.

This is a fixed equal-weight exposure return model. Costs charge changes in
target exposure, including flips and liquidation; share drift, financing,
borrow fees and market impact are not simulated. Identical asset calendars are
required before splitting and after feature/window construction. Missing or
asynchronous calendars fail explicitly rather than inventing returns or silently
changing portfolio composition. Annualization assumes the supplied bar frequency.

## Stable training and memory

P&L/Sharpe are computed over the **entire chronological training portfolio**,
never independently averaged over shuffled mini-batches. Asset boundaries reset
positions; activation blocks do not. One optimizer update is made per epoch:

1. At frozen weights, compute positions in bounded blocks without retaining
   activation graphs.
2. Compute the exact full-path objective and analytic gradient with respect to
   every position, including turnover links across blocks and final liquidation.
3. Replay the same dropout randomness and recompute each block's activations.
   Chain its position gradient through the model, accumulating parameter gradients.
4. Apply SGD for Manual ANN or AdamW for PyTorch once; PyTorch retains its configured
   weight decay and gradient clipping.

`batch_size` bounds activation memory. Sequence tensors are still materialized,
so feature count, context and ticker count remain relevant on an 8 GB machine.
There are two model passes per gradient update; this trades compute for bounded
activation memory. With dropout disabled, changing block size preserves the
global update up to floating-point reductions. With dropout enabled, masks depend
on block layout but are replayed identically within the two passes.

Train/validation losses are measured again with dropout disabled after each
update. Early stopping and best-checkpoint restoration use the selected objective
on the complete validation path. No sample/class weights are applied to financial
returns. Event weighting is rejected for mixed financial comparisons.

Features, coverage filtering, imputation, FracDiff and scaling reuse existing
train-only preparation. The financial scaler uses the same known-label training
windows as CE. Financial losses additionally use observed within-split returns
from rows whose longer label horizon is unknown; they do not use an unobserved
return across the split. This target difference and the optimizer update frequency
are deliberate differences from label classification, not a bit-identical training
protocol with a scalar substitution.

## Selection, final test and artifacts

`validation.csv` contains each model/parameter/loss/seed run. `selection.json`
records the data hash, experiment and loss parameters, cost/decoder conventions,
and frozen candidate selection **before** final-test evaluation. `report.json`
includes validation outcomes, failed runs, selection and optional final results.
Choose `--selection-metric regularized_sharpe` (default) or `net_return`; candidates
must succeed on every declared seed to be eligible. Selection uses the mean
validation metric across seeds, separately for the CE control and financial
challenger. Failed candidates cannot win through a subset of successful seeds.

Adding `--final-test` to a run evaluates only the selected CE configuration and
selected financial configuration, across their predeclared seeds. Checkpoints are
reloaded, never refitted or recalibrated. Freeze the comparison protocol before
using that flag; repeated final-test inspection is not a hyperparameter search.
Output directories cannot be overwritten. Without that flag, no test-return
panel or test prediction is constructed.

Each candidate saves a checksummed NPZ/JSON artifact using the existing safe
serializer. Metadata includes the objective, costs, epsilon, annualization,
decoder, legacy policy, config hash, input data hash, feature-source provenance,
fitted preprocessing, loss histories and runtime versions. These artifacts are
tagged `position_objective_v1`; use their dedicated loader rather than the old
classification notebook's reconstruction:

```python
from trading_system.experiments.position_objectives import load_position_artifact

result = load_position_artifact("artifacts/comparisons/losses-validation/runs/<run>")
positions = result.predict_positions(raw_feature_windows)
```

Windows must already contain the frozen feature columns and causal history, just
as with classifier artifacts. `--no-run-artifacts` retains reports only and cannot
be combined with `--final-test`. Existing artifacts remain readable unchanged.

## Walk-forward and grid-search integration

The same position objectives are available in the expanding-window runner. Each
chunk creates a fresh model from the registry, splits its strictly historical
data into inner train/validation partitions, fits preprocessing on that history,
selects the financial checkpoint on the complete validation return path, then
predicts continuous positions for the next chunk. Final reporting evaluates one
continuous path across all predicted chunks: chunks do not force liquidation or
reset positions. Train/validation panels do reset because they are independent
checkpoint-selection partitions. The evaluation split starts flat and includes
terminal liquidation.

```bash
python scripts/run_walkforward.py \
  --data-dir data/processed/cac40_daily.parquet \
  --ticker EN.PA \
  --model manual_ann \
  --loss-objective sharpe \
  --loss-cost-bps 5
```

`--loss-objective cross_entropy` is the default and executes the unchanged
classifier path. `pnl` and `sharpe` reject sample weighting and Oracle labels.
Financial walk-forward currently requires one explicitly filtered ticker;
multi-asset synchronized-panel walk-forward remains unsupported rather than
silently allowing cross-ticker sequence windows.

Walk-forward grid search supports `net_pnl`, `net_return`,
`regularized_sharpe`, `model_pnl`, or `outperformance` as financial validation
objectives. It ranks validation walks and evaluates only the frozen winner on the
final test, preserving the existing grid-search protocol:

```bash
python scripts/run_gridsearch_walkforward.py \
  --data-dir data/processed/cac40_daily.parquet \
  --ticker EN.PA \
  --models manual_ann,rnn,lstm,gru,transformer \
  --loss-objective pnl \
  --objective net_return \
  --loss-cost-bps 5
```

Direct objectives do not generate or use labels, label thresholds, or discrete
decision calibration for training. Grid search therefore retains one inert label
configuration for CLI/report compatibility and removes duplicate
label-threshold/decision axes; model parameters, context length and
walk-forward step remain searchable. Results export the full loss config,
per-chunk financial validation metrics, seeds, fitted preprocessing diagnostics,
continuous-position coverage, costs, turnover, drawdown and buy-and-hold
comparison. Cross-entropy CSV/JSON shapes and defaults remain compatible.

## Verification and limits

Verified after walk-forward integration on 2026-09-07 in isolated local Python
3.12: full repository suite passed **416 tests** (25 warnings). Financial-loss
coverage contains 47 tests across static and walk-forward modules.
No historical market/fundamental/news data were downloaded.

Tests exercise finite-difference gradients for both objectives, delayed execution,
portfolio aggregation, entry/flip/liquidation costs, flat-path stability, all five
model families, dropout replay reproducibility, block-size equivalence without
dropout, unchanged CE behavior, final-test perturbation isolation, selection,
artifact reconstruction and CLI operation.
Walk-forward tests additionally cover both objectives, continuous bounded
positions, causal final-test isolation, absence of label consumption, proportional
cost reporting, CE-path equivalence, CLI parsing, validation-only grid ranking,
objective/loss mismatch rejection and winner-only final evaluation.

These checks establish implementation behavior, not profitable trading. Matched
multi-seed benchmarks and validation-selected learning-rate/regularization studies
remain necessary. Current CAC40 constituent/sector survivorship limitations and
historical source publication requirements still apply.
