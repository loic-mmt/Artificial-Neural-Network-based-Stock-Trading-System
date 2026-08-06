# Pipeline restructuring plan

Status: implemented on 2026-08-06.

Implementation result:

- pipeline code reduced from 8,468 to 519 lines;
- six static pipeline implementations replaced by configuration wrappers;
- manual ANN centralized in `models/manual_ann/manual_nn.py`;
- static and walk-forward runners use common classifier interface;
- data, labels, metrics, thresholds, positions, and backtests have one owner each;
- lower layers no longer import pipelines;
- advanced `backtest/lib.py` kept as lazy-loaded legacy implementation while lightweight engine stays independent;
- 17 tests cover single/multi, long-only/long-short, static/walk-forward, leakage guards, ANN determinism, adapter alignment, and oracle/backtest agreement.

Original findings and unchecked migration checklist below remain as before-state/history. Implementation summary above is authoritative.

## Goal

Make data preparation, labels, training, evaluation, and backtesting reusable by manual ANN and future models. Keep pipeline modules as thin orchestration only.

Desired rule:

```text
scripts -> pipelines/experiments -> data/features/labels/models/evaluation/backtest/reporting
```

Lower layers must never import `trading_system.pipelines`.

## Current findings

- `src/trading_system/pipelines/`: 8,468 lines.
- Six static pipeline variants: 6,903 lines, mostly copies.
- 268 top-level pipeline functions.
- 41 function names repeated across pipelines.
- 26 repeated groups have identical AST bodies.
- 239 definitions belong to repeated-name groups.
- `models/manual_ann/manual_nn.py`: empty.
- `analysis/compare_labels_vs_buyhold.py` and `labels/breakout_gridsearch.py` import helpers from pipelines. Dependency direction wrong.
- `walkforward.py` duplicates oracle-DP code already present in `labels/oracle_dp.py`.
- `train_model` performs too many jobs: feature engineering, split, windowing, scaling, ANN optimization, threshold calibration, metrics, backtest, printing, plotting.

High-risk behavior differences must remain explicit:

- single-ticker split versus per-ticker grouped split;
- long-only versus long/short position mapping;
- single-asset versus per-ticker portfolio aggregation;
- static pipeline currently has both same-bar and next-bar execution behavior; canonical anti-look-ahead behavior must be selected and tested;
- simple technical features versus full market features;
- standard split versus walk-forward split;
- fixed fee and long/short flip semantics.

## Ownership rules

- `data`: loading, chronological splitting, window construction, scaling.
- `features`: feature calculations and feature-column definitions.
- `labels`: label schema and label generators.
- `models`: model-specific fit and inference only.
- `evaluation`: classifier metrics and probability-to-class calibration.
- `backtest`: class-to-position conversion, execution delay, fees, equity curves, benchmarks.
- `reporting`: plots and console/table formatting.
- `experiments`: model-neutral workflow and search orchestration.
- `pipelines`: configuration, CLI arguments, calls into experiment runner.

Important ANN boundary: `manual_nn.py` owns ANN math and optimization. It must not own parquet loading, DataFrame splitting, market features, backtests, or plots. Those operations must remain reusable by every model.

## Proposed target tree

```text
src/trading_system/
├── data/
│   ├── io.py
│   ├── splits.py
│   ├── windows.py
│   └── scaling.py
├── features/
│   ├── technical.py
│   └── market.py
├── labels/
│   ├── schema.py
│   ├── breakout.py
│   ├── forward_return.py
│   └── oracle_dp.py
├── models/
│   ├── base.py
│   └── manual_ann/
│       └── manual_nn.py
├── training/
│   └── weights.py
├── evaluation/
│   ├── classification.py
│   └── thresholds.py
├── backtest/
│   ├── positions.py
│   ├── adapters.py
│   ├── benchmarks.py
│   └── engine.py
├── reporting/
│   └── plots.py
├── experiments/
│   ├── config.py
│   ├── runner.py
│   └── search.py
└── pipelines/
    ├── single_ticker.py
    ├── multi_ticker.py
    ├── walkforward.py
    └── gridsearch_walkforward.py
```

Final pipeline count may be smaller. Existing script names can remain compatibility entrypoints while calling one shared runner.

## Move inventory

### Data and feature preparation

| Current functions | Target | Action |
|---|---|---|
| `read_parquet_dataset` in six static pipelines, `walkforward.py`, and `oracle_dp.py` | `data/io.py` | Keep one implementation with path checks, column projection, Arrow filter, and clear fallback behavior. |
| `chronological_train_val_test_split`, `chronological_train_val_split`, `_split_by_calendar_boundaries`, `_prepare_feature_split_frames` | `data/splits.py` | Expose separate single-series, grouped, and calendar-boundary APIs. Do not hide semantics in one ambiguous function. |
| `to_train_test` | delete after verification | No callers found. Current float slice behavior is suspect. |
| `build_context_dataset`, `build_context_dataset_with_history`, `build_context_features` | `data/windows.py` | One supervised window builder plus one inference window builder. `group_col` prevents cross-ticker windows. Return aligned row indices explicitly. |
| `standardize_features` | `data/scaling.py` | Replace tuple convention with fitted `Standardizer` state. Fit on train only; transform validation/test. |
| `compute_returns`, `normalize_prices`, six `compute_features` copies | `features/technical.py` | One builder supporting optional `group_col`. |
| repeated hard-coded 15-column list | `features/technical.py` as `TECHNICAL_FEATURE_COLUMNS` | Single source of truth. |
| lowercase `features` in `features/market.py` | keep module, rename public constant to `MARKET_FEATURE_COLUMNS` | Preserve temporary `features` alias during migration. |
| `compute_market_features` | keep in `features/market.py` | Pass as feature-builder strategy to experiment runner. |

### Labels

| Current functions/constants | Target | Action |
|---|---|---|
| repeated label IDs/maps (`Sell=0`, `Hold=1`, `Buy=2`) | `labels/schema.py` | Define one enum or constants plus ID/name maps. |
| `enforce_alternating_signals`, `add_labels`, `labelling`, `labelling_all` copied six times | `labels/breakout.py` | Canonical names: `generate_breakout_labels` and `generate_breakout_labels_by_ticker`. Keep old aliases temporarily. |
| `build_forward_return_labels` | `labels/forward_return.py` | Remove from `walkforward.py`. |
| `_allowed_next_executed_positions`, `_compute_forward_returns`, `solve_oracle_executed_positions_dp`, `executed_to_target_positions`, `target_positions_to_labels`, `build_oracle_labels_train_only` | `labels/oracle_dp.py` | Keep one canonical implementation; delete walk-forward copy. |
| `merge_oracle_labels_on_train_only`, `apply_oracle_labels_on_all_data` | `labels/oracle_dp.py` | Make oracle-label application reusable outside one pipeline. |

### Manual ANN and common model contract

| Current code | Target | Action |
|---|---|---|
| `relu`, `relu_derivative`, `softmax`, `dropout_mask`, `one_hot`, `forward_pass` | `models/manual_ann/manual_nn.py` | Make internal helpers unless tests or research use require public API. |
| weight initialization, minibatch loop, weighted loss, gradients, early stopping, best-weight restore | `ManualANNClassifier.fit()` in `manual_nn.py` | Extract from every `train_model`, `train_one_trial`, and `train_ann_on_labeled_history`. |
| ANN probability inference | `ManualANNClassifier.predict_proba()` | Replace raw `W0`, `b0`, `W1`, `b1` access outside model. |
| ANN hyperparameters | `ManualANNConfig` dataclass | Include hidden size, learning rate, epochs, batch size, dropout, patience, minimum delta, and seed. |
| training diagnostics | `TrainingHistory`/`FitResult` dataclass | Return losses, validation metrics, best epoch, and stop reason. No printing inside core math. |
| `compute_class_weights` | `training/weights.py` | Shared target-weight policy. Manual ANN consumes returned weights. |
| module-level `np.random.seed(1)` | model/runner-owned `np.random.Generator` | Deterministic tests without global RNG side effects. |

Proposed common interface:

```python
class ProbabilisticClassifier(Protocol):
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
```

All models must return probability columns in fixed `[Sell, Hold, Buy]` order. Adapters for scikit-learn models must realign `classes_`, including missing classes.

Model artifact must separate concerns:

```text
TrainedModelBundle
├── estimator
├── feature_columns
├── fitted_standardizer
├── context_len
├── label schema
└── calibrated decision thresholds
```

Estimator generates probabilities. Threshold policy converts probabilities into labels. Backtest converts labels into positions. Keep these three stages separate.

### Evaluation, decisions, and reporting

| Current functions | Target | Action |
|---|---|---|
| `recall_for_label`, `balanced_accuracy`, `precision_recall_f1_for_label`, `macro_f1`, `evaluate_predictions`, `compute_confusion_matrix` | `evaluation/classification.py` | One implementation, stable result schema. |
| `predict_with_thresholds`, `predict_from_probs`, `threshold_gridsearch` | `evaluation/thresholds.py` | One decision policy with explicit `argmax`/threshold mode and minimum action rate. |
| `plot_confusion_matrix`, `plot_signals` | `reporting/plots.py` | Plot only; no training side effects. |
| training/test `print()` blocks | reporter in `reporting` or CLI layer | Core functions return structured results. |

### Positions and backtesting

| Current functions | Target | Action |
|---|---|---|
| `signals_to_positions`, `labels_to_positions`, `pred_labels_to_target_positions`, `target_positions_to_actions` | `backtest/positions.py` | One label decoder with explicit `position_mode="long_only" | "long_short"`. |
| `_coerce_utc_datetime_index`, `prepare_advanced_backtest_inputs` | `backtest/adapters.py` | Keep conversion into existing advanced backtest schema. |
| all `evaluate_strategy_vs_buy_hold` variants | `backtest/engine.py` and `benchmarks.py` | One single-asset engine plus grouped portfolio wrapper. Explicit signal delay, fees, turnover, and position mode. |
| `compute_benchmark`, `evaluate_buy_hold_only` | `backtest/benchmarks.py` | One buy-and-hold implementation. |

Do not merge backtest variants until characterization tests establish current behavior. Static long-only code applies positions on same bar, while walk-forward and long/short code delay execution by one bar. Preferred canonical rule: prediction at bar `t` executes at `t+1`.

### Experiment orchestration and search

| Current code | Target | Action |
|---|---|---|
| feature/split/window/scale/evaluate parts of six `train_model` copies | `experiments/runner.py` | Build one model-neutral `run_experiment(config, model)` workflow. |
| `train_ann_on_labeled_history`, `predict_chunk_with_model` orchestration | `experiments/runner.py` / walk-forward runner | Delegate only fit/predict math to model object. |
| `train_one_trial`, `model_grid_search`, `TrialConfig`, trial grid, sampling, objective selection | `experiments/search.py` | Search any classifier factory, not ANN-specific functions. |
| argparse parsing and `main()` | pipelines | Keep CLI concerns near entrypoint. |

Recommended configuration axes:

| Axis | Values |
|---|---|
| universe | `single`, `multi` |
| feature set | `technical`, `market` |
| position mode | `long_only`, `long_short` |
| evaluation mode | `static`, `walk_forward` |
| labeler | `breakout`, `forward_return`, `oracle_train_only` |
| model | `manual_ann`, future model names |

Current six static modules then become configurations, not separate implementations:

| Current module | Configuration |
|---|---|
| `single_ticker.py` | single + technical + long-only |
| `single_ticker_features.py` | single + market + long-only |
| `single_ticker_long_short.py` | single + technical + long/short |
| `single_ticker_long_short_features.py` | single + market + long/short |
| `multi_ticker.py` | multi + technical + long-only |
| `multi_ticker_long_short.py` | multi + technical + long/short |

## Dead or dormant code audit

Verify, then remove instead of moving blindly:

- `to_train_test`: zero callers.
- `compute_benchmark`: zero callers.
- `plot_signals`: zero callers.
- `train_model_features` in `multi_ticker.py`: zero callers.
- `model_grid_search`: definitions exist, current `main()` calls are commented.
- stale commented training/debug blocks inside pipelines.

Public callers needing import repair:

- `labels/breakout_gridsearch.py` currently imports data, split, labels, and backtest helpers from `pipelines.single_ticker`.
- `analysis/compare_labels_vs_buyhold.py` currently imports loading, labels, and backtest helpers from `pipelines.multi_ticker`.

Both must import owning modules after extraction.

## Migration phases

### Phase 0 — Characterize behavior

- [ ] Add synthetic fixtures for one ticker and multiple tickers.
- [ ] Capture outputs for labels, split sizes, windows, scaler values, probability decisions, metrics, positions, fees, and PnL.
- [ ] Add regression test proving no window crosses ticker boundaries.
- [ ] Add regression test proving scaler fits train only.
- [ ] Add regression test for signal timing (`t` prediction executes at `t+1`).
- [ ] Add fee test: long/short flip has turnover `2`.
- [ ] Add deterministic ANN smoke test with fixed seed.
- [ ] Record intentional differences between current pipelines.

Exit: behavior baseline exists before any move.

### Phase 1 — Extract exact shared helpers

- [ ] Create `data/io.py`, `data/splits.py`, `data/windows.py`, `data/scaling.py`.
- [ ] Create `features/technical.py` and feature constants.
- [ ] Create `labels/schema.py` and `labels/breakout.py`.
- [ ] Create `evaluation/classification.py` and `evaluation/thresholds.py`.
- [ ] Replace copied functions with imports, one family at a time.
- [ ] Update analysis and label-grid-search imports.
- [ ] Keep temporary re-exports only where external import compatibility matters.

Exit: exact duplicate helper bodies removed; pipeline behavior unchanged.

### Phase 2 — Consolidate position and backtest semantics

- [ ] Introduce explicit `position_mode` and `execution_delay` configuration.
- [ ] Implement canonical position decoder.
- [ ] Implement single-asset evaluation and grouped multi-asset wrapper.
- [ ] Compare old/new equity curves row by row.
- [ ] Select and document next-bar execution as canonical behavior.

Exit: one tested PnL path serves static and walk-forward workflows.

### Phase 3 — Extract manual ANN

- [ ] Add `ManualANNConfig`, `ManualANNClassifier`, `FitResult`, and `TrainingHistory`.
- [ ] Move primitives, forward pass, gradient updates, dropout, weighted loss, early stopping, and inference into `manual_nn.py`.
- [ ] Make feature arrays and validation arrays explicit inputs.
- [ ] Replace raw weight dictionaries in pipeline code with model object.
- [ ] Keep threshold calibration, metrics, backtest, and plots outside model.
- [ ] Test probabilities and one deterministic training run against baseline.

Exit: manual ANN train/inference has one implementation and no pipeline dependency.

### Phase 4 — Add model-neutral experiment runner

- [ ] Add typed `ExperimentConfig` and result dataclasses.
- [ ] Build shared load -> label -> features -> split -> window -> scale workflow.
- [ ] Call `model.fit()` and `model.predict_proba()` through common interface.
- [ ] Calibrate thresholds on validation only.
- [ ] Evaluate test only once after model selection.
- [ ] Return structured metrics, predictions, aligned rows, and backtest outputs.

Exit: swapping model factory requires no data/evaluation code copy.

### Phase 5 — Thin pipelines

- [ ] Convert six static modules into small config/CLI wrappers.
- [ ] Preserve current scripts as compatibility entrypoints.
- [ ] Optionally replace six pipeline modules with one `train.py` entrypoint and flags.
- [ ] Keep `walkforward.py` focused on rolling schedule only.
- [ ] Keep `gridsearch_walkforward.py` focused on CLI/search configuration only.

Exit: pipeline modules contain orchestration, not model math or reusable helpers.

### Phase 6 — Cleanup and add second model

- [ ] Remove compatibility aliases after imports are migrated.
- [ ] Remove confirmed dead code and commented blocks.
- [ ] Export intentional public APIs through package `__init__.py` files.
- [ ] Add first alternative classifier through adapter/factory.
- [ ] Run same experiment config with manual ANN and alternative model.

Exit: second model uses identical data, thresholds, metrics, and backtest path.

### Phase 7 — Separate follow-up: backtest monolith

`backtest/lib.py` is 3,996 lines. Current `engine.py`, `analytics.py`, `artifacts.py`, and `reporting.py` mostly re-export from it. Split implementations later, after pipeline extraction. Mixing this large change into early phases raises regression risk.

## Test layout

```text
tests/
├── data/
│   ├── test_splits.py
│   ├── test_windows.py
│   └── test_scaling.py
├── features/
│   └── test_technical.py
├── labels/
│   ├── test_breakout.py
│   └── test_oracle_dp.py
├── models/
│   └── test_manual_ann.py
├── evaluation/
│   ├── test_classification.py
│   └── test_thresholds.py
├── backtest/
│   ├── test_positions.py
│   └── test_engine.py
└── experiments/
    ├── test_runner.py
    └── test_walkforward.py
```

Required invariants:

- chronological order always preserved;
- no future rows enter features, scaling, thresholds, or training;
- no context window mixes tickers;
- validation chooses epoch/thresholds; test never chooses parameters;
- probability columns always match label schema;
- position mode explicit;
- signal execution timing explicit;
- fees and turnover consistent across all workflows;
- fixed seed gives reproducible manual ANN output;
- old and new outputs match unless behavior change is documented.

## Definition of done

- No reusable function defined in more than one pipeline.
- No lower-level package imports `trading_system.pipelines`.
- `manual_nn.py` contains sole manual ANN training and inference implementation.
- Pipelines contain configuration and orchestration only.
- Alternative model runs through same experiment runner.
- Shared regression suite passes for single/multi, technical/market, long-only/long-short, and static/walk-forward modes.
