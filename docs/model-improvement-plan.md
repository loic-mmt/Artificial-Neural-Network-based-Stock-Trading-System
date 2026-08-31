# Model Improvement Plan

## Purpose

This document defines the controlled path for improving the **CAC40 daily,
forward-return, walk-forward** system. Primary target remains robust
out-of-sample trading performance versus buy-and-hold.

This plan reflects the shared architecture imported in commit `bc8c6f0`. It
supersedes the pre-refactor implementation notes kept under
`tmp/model-improvement-tracker/`.

Canonical experiment path:

- CLI: `python scripts/run_walkforward.py`
- CLI orchestration: `src/trading_system/pipelines/walkforward.py`
- Walk-forward implementation: `src/trading_system/experiments/walkforward.py`
- Shared static runner: `src/trading_system/experiments/runner.py`
- Search: `src/trading_system/experiments/search.py`
- Labels: `src/trading_system/labels/`
- Decision policy: `src/trading_system/evaluation/thresholds.py`
- Core model/benchmark evaluation: `src/trading_system/backtest/engine.py`
- Advanced diagnostics and acceptance checks: `src/trading_system/backtest/lib.py`

Oracle labels remain an **upper-bound diagnostic only**. They must never select
a model, define a production decision rule, or appear as headline learnable
performance.

## Current State

### Architecture now in place

- Data loading, splitting, scaling, and window construction live under
  `src/trading_system/data/`.
- Feature construction lives under `src/trading_system/features/`.
- Label generation lives under `src/trading_system/labels/`.
- Classification metrics and decision calibration live under
  `src/trading_system/evaluation/`.
- Shared experiment orchestration lives under
  `src/trading_system/experiments/`.
- Pipelines are thin CLI and compatibility wrappers; new training logic must not
  return to pipeline modules.
- Manual ANN has one canonical implementation under
  `src/trading_system/models/manual_ann/`.
- Core execution uses next-bar timing through
  `evaluate_strategy_vs_buy_hold`.
- Static single-ticker and multi-ticker experiments share `run_experiment`.

### Verified baseline of the codebase

- `136` core automated tests pass as of 2026-08-31, including `31`
  search-integrity tests and `52` memory-optimization tests. The separate
  untracked local MT5 smoke test is not part
  of this verification: the test interpreter lacks TA-Lib, while the project
  `.venv` lacks pytest and stalls while importing pandas.
- Tests cover chronology, grouped windows, scaling, labels, next-bar execution,
  grouped capital allocation, shared experiment paths, deterministic manual ANN,
  state round-trips, and architecture boundaries.
- `DecisionPolicy` supports `argmax` and validation-calibrated `thresholds`.
- Walk-forward retraining uses expanding history and creates a fresh manual ANN
  per chunk.
- Seed support exists in the walk-forward CLI and manual ANN configuration.
- Static and walk-forward search now rank on validation only, then evaluate the
  frozen winner once on final test per search invocation.
- Static selection exposes a `ValidationResult` without test metrics or prices.
- Forward-return targets crossing a split boundary are excluded from training,
  calibration and classification scoring; their feature context is retained.
- Manual-ANN retrain seeds derive from `(run_seed, chunk_id)` and are recorded.
- Window construction, scaling, manual ANN training/inference and experiment
  buffer lifetimes now use fewer allocations. Synthetic comparisons against
  `38a2412` preserve all measured outputs, with lower peak allocations and runtime
  across single-ticker, multi-ticker and walk-forward paths. See
  [memory optimization measurements](memory-optimization.md) for methodology,
  reproduction commands and limitations; these are not a trading baseline.

### Important incomplete work

The refactor created interfaces for the target architecture, but several are
still scaffolds:

- `ModelBuildContext` and `ModelSelection` are not implemented.
- `ModelRegistry` and `create_default_model_registry` are not implemented.
- RNN, LSTM, GRU, Transformer, and shared PyTorch trainer modules contain
  `NotImplementedError` placeholders.
- Multi-model comparison orchestration is not implemented.
- Walk-forward still uses the manual-ANN compatibility path and flattened
  context representation rather than the final model-neutral 3D sequence path.
- Static and walk-forward paths do not yet use one common model registry.
- Search selection leakage is addressed, but immutable run manifests, integrated
  diagnostics and a current-code repeated-seed baseline still block promotion.
- Artifact manifest validation, hashing, save/load and compatibility checks are
  still scaffolds in `artifacts/serialization.py`.
- Shared backtest output currently exposes capital, PnL, and outperformance.
  Sharpe, drawdown, turnover, execution stress, drift, and overfit diagnostics
  still need one integrated experiment result.

### Historical baseline

Historical reference artifact:
`artifacts/runs/20260412T201542Z_saf.pa_1d`.

- Final capital: `15315`
- Total PnL: `5315`
- Sharpe ratio: `3.77`
- Max drawdown: `-34.7%`
- Win rate: `77.1%`
- Total trades: `48`
- Overfit verdict: `caution`
- Probability of backtest overfitting: `0.5`
- Robustness score: approximately `0.48`

This artifact predates the shared architecture. It is a historical reference,
not the promotable baseline. Phase 0 must rerun and freeze a new baseline using
the current code and an immutable configuration.

### Historical local experiment notes

`tmp/model-improvement-tracker/` records useful June 2026 experiments, including
seed sweeps and threshold/hysteresis trials. Those runs reference implementation
options that are absent after the refactor, including
`selection_objective`, `trading_combo`, and
`long_only_hysteresis`.

Keep these files as local research history. Do not use their metrics as current
baseline evidence and do not promote their configurations until the behavior is
reimplemented in shared modules and rerun.

## Goal And Success Rule

Target: improve **median OOS outperformance versus buy-and-hold** across repeated
seeds, walk-forward windows, and a declared ticker set.

### Promotable candidate

A candidate is promotable only if it:

- improves median OOS outperformance across repeated runs;
- or improves Sharpe while keeping drawdown equal or better;
- does not win from one seed, one ticker, one market interval, or oracle leakage;
- keeps fees, execution delay, capital, benchmark, and position mode identical
  to baseline;
- passes leakage, overfit, drift, execution, and acceptance checks;
- was selected using training and validation data only;
- touches final test data once, after configuration is frozen.

### Minimum promotion gate

Every baseline/candidate comparison must use the same:

- immutable dataset snapshot and hash;
- ticker set and date range;
- chronological split policy;
- feature and label definitions;
- fees, slippage, execution delay, and initial capital;
- benchmark and position mode;
- seed list, with at least five seeds.

Report individual runs plus median, mean, dispersion, worst case, and failure
count. Higher raw PnL alone never overrides a failed guardrail.

## Non-Negotiable Guardrails

### No lookahead

- Fit scalers and fill values on training history only.
- Build features only from information available at prediction time.
- Align forward-return labels so the prediction bar never sees its future target.
- End walk-forward training history strictly before each prediction chunk.
- Execute predictions with the declared delay, normally at `t+1`.
- Keep oracle computations isolated and explicitly marked as upper bound.

### No survivorship bias

`src/trading_system/data/download.py` uses a current/frozen CAC40 universe for
historical data. Until historical membership snapshots exist, every CAC40 result
must be marked **survivor-biased and provisional**.

### No selection leakage

- Never rank candidates on final-test metrics.
- Never tune thresholds, labels, features, or model parameters on test data.
- Use validation metrics or validation-only trading simulation for selection.
- Freeze the chosen configuration before one final test evaluation.
- Never promote a one-seed or best-ticker result.

### No benchmark drift

- Use the same buy-and-hold definition for baseline and candidate.
- Keep execution delay, fees, capital allocation, and position conversion fixed.
- Record every comparison input in the run manifest.

### No silent implementation regressions

- All existing tests must pass.
- New behavior requires focused unit tests plus one shared-runner integration
  test.
- Core execution invariants belong in `backtest/engine.py`.
- Advanced acceptance, overfit, drift, and execution diagnostics remain in
  `backtest/lib.py` until intentionally migrated.
- Compatibility wrappers may call shared logic but must not duplicate it.

## Roadmap

Work proceeds in order. Performance exploration stops when an earlier integrity
phase is incomplete.

### Phase 0: Make evaluation trustworthy

Implementation progress — 2026-08-31:

- [x] Static trial ranking uses only `val_metrics` / `val_backtest`.
- [x] Walk-forward validation runs stop before final test, before labeling or
  retraining. Their results have no test-result fields.
- [x] Final test runs only after all candidates have been scored and the winner
  is fixed. Static search reuses the exact fitted weights, scaler and policy;
  walk-forward search evaluates one frozen retraining strategy.
- [x] Failed validation trials remain visible; ties preserve trial order.
- [x] Oracle label modes are rejected for search, but remain available as
  standalone diagnostics.
- [x] Raw static split boundaries are fixed before feature/label construction.
  Forward-return boundary targets and unknown trailing targets are excluded from
  supervised fitting/scoring in static and walk-forward paths.
- [x] Manual ANN uses deterministic derived chunk seeds, shared across trial
  configurations and recorded in retrain logs. Reusing an estimator instance
  across trials/chunks is rejected.
- [ ] Extend the chunk seed contract to registry-based custom model factories.
- [ ] Complete immutable manifests and integrate advanced diagnostics.
- [ ] Rerun and freeze the current-code baseline across at least five seeds.
- [ ] Add survivor-bias warnings to generated current-universe reports.

Verification: `tests/test_search_integrity.py` covers held-out-price poisoning,
train/validation target purging, fixed winner evaluation, failure handling,
stable ties, seed reproduction, grouped static data and CLI JSON output.
Core suite: `python -m pytest -q --ignore=tests/test_mt5_walkforward_strategies.py`.

Result/API notes:

- `run_validation_experiment` returns train/validation diagnostics without
  building test windows. `evaluate_experiment_test` evaluates its frozen bundle;
  `run_experiment` remains the convenience wrapper for both stages.
- `val_label_mask` / `test_label_mask` identify observed classification targets.
  Backtests still use every predicted row, so horizons do not shorten their
  price/benchmark intervals. `label_stats` describes eligible train/validation
  labels only; it does not expose the held-out label distribution.
- Walk-forward search retains its DataFrame return type. Rows/CSV contain
  `val_*` metrics, validation objective scores and a `selected` flag. The separate
  winner report is in `results.attrs["final_test"]`; frozen trial parameters are
  in `results.attrs["best_parameters"]`. Per-trial validation retrain logs are in
  `results.attrs["validation_retrain_logs"]`. The CLI persists all three in JSON.
- Once-only evaluation is enforced by search orchestration within one run, not
  by a durable cross-run lock. Repeatedly tuning from the final-test report is
  still prohibited.
- Forward-return metrics and seeded ANN results intentionally differ from older
  runs. Historical numbers are not a new comparable baseline.

Required work:

1. Change `experiments/search.py` so all trial ranking uses validation-only
   objectives.
2. Prevent walk-forward grid search from reading final-test metrics during
   selection.
3. Evaluate final test once, only for the frozen winner.
4. Complete deterministic seed derivation from `run_seed` and `chunk_id` and
   record each retrain seed.
5. Add a run manifest containing:
   - dataset path and hash;
   - ticker set and date range;
   - feature and label configuration;
   - split boundaries;
   - seed and model configuration;
   - fees, execution delay, capital, and position mode;
   - code commit and dependency versions.
6. Integrate core and advanced reports into one persisted experiment artifact.
7. Rerun a current-code baseline across at least five seeds.
8. Mark all current-universe CAC40 results as survivor-biased.

Primary touchpoints:

- `src/trading_system/experiments/search.py`
- `src/trading_system/experiments/walkforward.py`
- `src/trading_system/artifacts/serialization.py`
- `src/trading_system/backtest/engine.py`
- `src/trading_system/backtest/lib.py`
- `tests/test_experiment_runner.py`
- new search and walk-forward leakage tests

Exit criteria:

- search cannot access test results while selecting;
- repeated fixed-seed runs reproduce metrics within tolerance;
- baseline artifact is generated from current code;
- every result has enough metadata for exact rerun;
- final test remains untouched until winner freeze.

### Phase 1: Complete model-neutral execution

Required work:

1. Implement and validate `ModelBuildContext` and `ModelSelection`.
2. Implement `ModelRegistry` with lazy model factories.
3. Move ANN-specific configuration out of `ExperimentConfig`.
4. Make static and walk-forward runners consume 3D `(N, T, F)` sequences through
   the same classifier contract.
5. Replace compatibility factory hooks with registry-based model construction.
6. Implement structured comparison runs with equal model/seed budgets.
7. Persist model name, parameter count, duration, seed, and failures.

Manual ANN remains the only production candidate until this phase passes. RNN,
LSTM, GRU, and Transformer scaffolds must not be presented as implemented
models.

Exit criteria:

- one registry creates every supported model;
- static and walk-forward paths share model contracts and sequence semantics;
- model comparisons enforce equal data, seeds, and budgets;
- failed trials remain visible rather than silently disappearing.

### Phase 2: Align validation objective and decision policy

Required experiments:

- classification baseline: `macro_f1` and balanced accuracy;
- validation-only outperformance;
- trading-aware composite objective with drawdown, turnover, and action-rate
  penalties;
- calibrated thresholds and no-trade zones;
- optional long-only hysteresis implemented inside
  `evaluation/thresholds.py`, not a pipeline;
- probability calibration measured on validation only;
- joint threshold and `min_action_rate` scans.

Validation-period backtest output is now available as `val_backtest`. Never
derive a selection objective from `result.test_metrics` or `result.backtest`.

Promotion condition:

- candidate beats classification-only selection on median OOS metrics;
- result remains stable across seeds and ticker slices;
- drawdown, turnover, execution stress, and failure rate stay within gate.

### Phase 3: Improve label policy

Primary learnable family remains `forward_return`.

Experiment backlog:

- horizons `1`, `3`, `5`, and `10`;
- asymmetric buy/sell thresholds;
- volatility-scaled thresholds;
- fixed versus rolling volatility-normalized thresholds;
- breakout labels as controlled alternative.

Reject any label policy that:

- improves only one ticker;
- collapses action coverage;
- depends on oracle information;
- cannot clear repeated-seed promotion gates;
- changes benchmark or execution assumptions.

### Phase 4: Compare features and models

Feature ablations:

- price-only;
- technical;
- full market;
- full market without unstable or high-drift features.

Model order:

1. majority/action-frequency sanity baseline;
2. logistic regression;
3. shallow tree-based baseline;
4. manual ANN;
5. RNN, LSTM, GRU, and Transformer only after Phase 1 implementation.

ANN/neural search may include hidden size, dropout, learning rate, context length,
depth, and architecture-specific parameters. Every model receives equal search
budget and the same seed list.

Complexity rule: if ANN or neural models do not beat simpler baselines under the
same OOS protocol, keep the simpler model.

### Phase 5: Robustness and trading filters

Only candidates surviving Phases 0–4 may test:

- regime-gated trading;
- drift or anomaly filters;
- confidence/meta-label gates;
- execution-cost stress;
- cross-ticker and subperiod stability.

Filters must be learned or calibrated without final-test information.

## Reporting Contract

Each batch must persist raw run rows and an aggregated summary.

Required run-level fields:

- `run_id`, `config_id`, and code commit;
- dataset hash, ticker, and date range;
- feature set, label family, horizon, and thresholds;
- model name, parameters, parameter count, and seed;
- split boundaries and walk-forward step;
- decision mode and calibrated policy;
- fees, execution delay, capital, and position mode;
- validation objective and validation score;
- OOS classification metrics;
- model and buy-and-hold PnL;
- outperformance, Sharpe, max drawdown, turnover, exposure, and trade count;
- stressed execution PnL;
- overfit verdict and drift severity;
- duration, status, and failure reason;
- promotion-gate pass/fail.

Aggregated summaries must include count, mean, median, standard deviation,
minimum, maximum, worst drawdown, and failed-run count by candidate.

## Test Plan

Required tests before performance promotion:

- feature pipeline never uses future rows;
- forward-return label alignment for every horizon;
- grouped windows never cross tickers;
- walk-forward chunk trains only on earlier rows;
- every prediction maps to its intended chunk row;
- execution occurs at declared delay;
- train-only scaling and missing-value statistics;
- fixed run/chunk seed reproduces predictions;
- different seeds produce a distribution;
- decision calibration reads validation only;
- grid search cannot read test metrics while ranking;
- final test executes only after winner freeze;
- baseline and candidate use identical benchmark assumptions;
- manifests and model state round-trip;
- survivor-bias warning remains until historical membership lands;
- purged CV, overfit, drift, and execution reports still generate.

## Decision Rules

- Stop an experiment branch after three consecutive batches fail the promotion
  gate.
- Stop any branch that raises raw PnL while materially worsening drawdown,
  execution realism, overfit verdict, or failure rate.
- Stop using feature groups with high drift and no stable OOS contribution.
- Treat missing or failed runs as evidence, not rows to discard.
- Do not merge performance claims while Phase 0 remains incomplete.

## Defaults

- Scope: current CAC40 daily setup.
- Primary label: `forward_return`.
- Primary evaluation: expanding walk-forward.
- Primary objective: robust median OOS outperformance, not in-sample fit.
- Current production model: manual ANN only.
- Oracle policy: diagnostic upper bound only.
- Current CAC40 history: survivor-biased until historical membership is added.
- `tmp/model-improvement-tracker/`: local historical evidence, not source of
  current truth.
