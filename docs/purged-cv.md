# Purged expanding cross-validation

The model and loss comparators support opt-in nested expanding cross-validation.
Omitting `--cv-folds` preserves their existing behavior. This is chronological
walk-forward CV, not shuffled K-fold or combinatorial purged CV (CPCV).

## Run and compare

Add the CV flags to an existing classifier benchmark:

```bash
python scripts/run_model_comparison.py \
  --data data/processed/cac40_daily.parquet \
  --preset multi_ticker_long_short \
  --ticker-selection configs/benchmark/cac40_diversified_10.json \
  --label-method triple-barrier \
  --feature-set expanded --no-external-features \
  --models manual_ann --seeds 1,7,19 \
  --cv-folds 3 --cv-gap-bars 5 --cv-score macro_f1 \
  --output-dir artifacts/comparisons/purged-cv
```

Without `--cv-final-test`, the final holdout stays sealed. Start with small
explicit model parameters to verify runtime on local hardware, for example
`--model-parameter-sets '{"manual_ann":[{"epochs":3,"hidden_size":8}]}'`.
Lists of parameter objects produce a model hyperparameter search; every candidate
receives the same outer periods and seeds. All five model architectures use the
same shared runners.

For a matched continuous-position comparison, the same flags work with
`scripts/run_loss_comparison.py --losses cross_entropy pnl sharpe`; choose
`--cv-score net_return` or `regularized_sharpe`. Its existing `--final-test`
also enables the CV-selected final evaluation. **In CV mode only one winning
configuration is tested**, across all predeclared seeds. The ordinary non-CV loss
comparator still has its separate CE-control/financial-challenger protocol.

Run the prior command without CV flags into a different output directory to keep
the original baseline. A CV outer-fold score and the old single-validation score
cover different periods/protocols and should not be compared as identical tests.
Changing this validation protocol does not establish improved profitability.

## Nested temporal structure

1. Reserve the final fraction `1 - train_ratio - val_ratio` of **unique dates**.
   Defaults inherited from the preset typically reserve 15%.
2. The initial historical window occupies `--cv-initial-train-fraction` of the
   development dates (default 0.5). Divide subsequent development dates into
   `--cv-folds` consecutive, non-overlapping outer evaluation blocks.
3. For each block, all prior dates form expanding history. Its last
   `--cv-inner-val-fraction` (default 0.2) is internal validation, used for early
   stopping and classifier threshold calibration. The earlier history trains the
   model. The outer evaluation block is not visible during fitting or calibration.
4. Rank candidates using the mean outer score across **all folds and seeds**.
   Report population standard deviation and minimum score as diagnostics, not
   confidence intervals; folds share training history and are not independent.
   Any failed or non-finite fold makes that candidate ineligible.
5. Persist selection before final evaluation. When explicitly enabled, refit the
   selected configuration on the development history, reserving the same inner
   validation fraction for checkpoint selection, then evaluate its frozen model
   on final holdout once per declared seed. No runner-up is tried after failure.

Calendar boundaries are global across tickers, preventing one asset's later
training rows from entering another asset's validation period. Classification
can accommodate differing asset histories when each split retains enough rows.
Financial portfolios retain their stricter identical-calendar requirement.

Every fold refits features/coverage selection, imputation, scaling, optional
FracDiff and overfitting feature selection inside its own historical window.
Outer scores are never used for early stopping within that fold. Older outer
periods may enter subsequent folds' history, as in an expanding live retraining
protocol. Final-test dates never enter any development fold.

## Purging actual information intervals

Training sample intervals are closed, `[entry, information_end]`. A target ending
exactly at the validation boundary is removed, as is any target extending beyond
it. Unknown ends are ineligible. This is applied between inner training and inner
validation and between inner validation and outer evaluation.

- **Triple Barrier**: training labels inspect the available inner-validation
  history to determine actual first-touch ends. Targets with ends before the
  validation boundary are retained; overlapping targets are masked before any
  supervised fitting. Thus an early resolved event can survive where a blanket
  `max_holding` tail cut would discard it. Unknown events remain excluded; no
  unobserved first touch is invented. Hold/flat non-events have same-date ends.
  Carry rows inherit their event's information end, bounded below by their own
  timestamp, within the same split.
- **Forward return**: end is the per-ticker date at the configured horizon.
- **Volatility position**: the configured forward horizon is a conservative
  information-end bound for its persistent-state labels.
- **Breakout**: the label uses information through its current close.
- **Oracle labels**: rejected for CV selection.

Inner-validation labels remain independently segmented by the existing labeler;
its unknown tail may conservatively exclude more than the actual first-touch
minimum. Feature windows retain past observations: overlapping historical
context is not automatically future leakage. Purged targets cannot contribute
to classifier gradients, class/sample weights, supervised feature selection or
normalizer fitting. Coverage selection and imputation use eligible training
rows. FracDiff's ADF uses consecutive raw training observations before the gap,
without compressing time by removing arbitrary event-label rows.

Financial losses have observed one-period return paths rather than event-label
targets. Their paths end within each split and exclude the configured gap; no
return crosses into validation. They retain their existing flat start and terminal
liquidation rules. Label-based purging still determines the common CE-compatible
scaler support, but does not delete arbitrary interior rows from a financial path.

## Gap versus embargo

These controls are intentionally distinct:

| Flag | Meaning |
| --- | --- |
| `--cv-gap-bars N` | Exclude the last N unique dates before inner validation and before outer evaluation from fitting/early-stopping targets, plus earlier targets whose information ends reach that gap. Rows stay available as past feature context. |
| `--cv-embargo-bars N` | Exclude training entries on N unique dates **after** the latest validation information end. A strict expanding fold has no such future training rows, so this adds zero exclusions there. |

Defaults are zero. An embargo is implemented by the reusable
`data.purged_cv.purge_intervals` utility and tested with training observations on
both sides of validation. It is not falsely represented as improving a
past-only split. Use the gap if the research protocol calls for temporal spacing
before validation. Tune these settings on development data, not final results.
The embargo counts calendar observations, not ticker rows or calendar days.

`purge_intervals` merges validation intervals and searches the merged boundaries;
it does not allocate an N-by-M overlap matrix. It purges across tickers when
given a pooled panel. Folds run sequentially, keeping one model at a time.
Compute scales with candidates × seeds × folds, plus selected final fits.

## Results and reproducibility

- `folds.json`: incremental per-fold metrics, boundaries, seed, errors and purge
  counts, retained after interruption.
- `folds.csv`: flat run identifiers, status and outer score.
- `selection.json`: protocol, global final boundary, data hash, feature-source
  provenance, candidate summaries and selected configuration, written before
  opening final test.
- `report.json`: complete fold/selection/final results.
- `folds/` and `final/`: optional existing-format model artifacts with explicit
  `config.purged_split` dates and `purging` diagnostics. `--no-run-artifacts`
  suppresses them; reports remain. Final fitting in CV does not require restoring
  a fold checkpoint because the selected configuration is refitted on development.

Counts distinguish overlap, unknown end, embargo and additional gap exclusions;
they describe raw labeled rows before feature warmup/window alignment. Config
hashes include explicit split boundaries and controls. Legacy artifacts missing
`purged_split` load with the disabled default. Existing notebook reconstructions
that assume ratio-only boundaries should not be used to reconstruct CV folds;
use the recorded calendar splits and shared runners.

The sequential `run_walkforward.py` streaming-chunk evaluator and its legacy
grid-search API are unchanged. The new comparator mode provides expanding-fold
hyperparameter selection with independently scored outer periods. CPCV, arbitrary
future-trained folds and automatic embargo-length optimization are not enabled.

## Verification

On 2026-09-08, the 25 CV tests and 425 existing regression tests passed in separate
batches. After the final gap/provenance changes, the relevant 68 CV, runner,
financial-loss and artifact tests passed again. Coverage includes brute-force
interval overlap equivalence, exact-boundary exclusion, calendar-based embargo,
actual first-touch retention, gaps extending to event ends, multi-asset boundaries,
final-holdout perturbation isolation, failed-candidate exclusion, both comparator
CLIs, and real PyTorch/expanded/FracDiff artifact restoration. Compilation and
`git diff --check` passed. Tests used synthetic fixtures; no market data was
downloaded and no real final-test benchmark was opened. Test dependencies were
installed in an isolated temporary Python environment.
