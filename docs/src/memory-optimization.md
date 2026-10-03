# Memory Optimization

## Scope

Allocation reductions verified on 2026-08-31 against commit
`38a2412b4bb213bf75fe09c1af593c71f4fd22eb`:

- Construct rolling windows with one contiguous allocation instead of a Python
  list of per-row arrays. Returned windows remain writable and independent;
  flattening their time/feature axes for the ANN needs no additional copy.
- Normalize into one private output buffer. Compute sequence variance with a
  bounded float64 workspace: 8 MiB, or one sequence if that is larger. The input
  finiteness check and other small arrays are additional allocations.
- Reuse ANN mini-batch buffers for activation, softmax and gradients. Update
  weights only after all gradients are computed; keep independent copies of the
  best checkpoint. Inference retains one hidden activation buffer.
- Reuse slices for contiguous valid-label prefixes; retain boolean selection
  for genuinely sparse masks, including grouped data.
- Release raw/normalized windows and intermediate frames after their last use
  in static and walk-forward experiments.

Float32 model/input precision, float64 sequence-statistic accumulation, model
configuration, mini-batches, seeds, dropout, labels, split boundaries, decision
policies, fees and execution rules are unchanged. Caller-owned inputs and fitted
scaler state are not modified by transforms or inference.

## Measurements

Synthetic data, 30,000 rows, context length 20, 15 features; ANN hidden size 64,
3 epochs, mini-batch size 512, seed 9. Multi-ticker data uses 4 tickers with 7,500
rows each. Walk-forward uses expanding-history retraining with 3,000-row steps.

Local macOS measurements use Python 3.10, NumPy 2.2.6, pandas 2.2.2 and the
existing feature fallback without TA-Lib. Each case/variant runs in a fresh
single-threaded worker, with one warm-up and a median of five timed runs.
Allocation tracing is performed in a separate run so its overhead does not
distort the reported timings.

| Case | Before peak (MiB) | After peak (MiB) | Before time (ms) | After time (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sequence windows | 45.66 | 36.95 | 21.07 | 8.05 |
| Flat windows | 78.58 | 36.49 | 34.87 | 7.79 |
| Sequence scaler fit | 68.79 | 8.59 | 39.73 | 25.25 |
| Sequence scaler transform | 68.70 | 42.92 | 18.09 | 15.75 |
| ANN inference | 16.37 | 8.59 | 8.35 | 7.09 |
| ANN fit | 17.84 | 8.68 | 84.80 | 74.11 |
| Single-ticker pipeline | 113.58 | 64.52 | 279.98 | 236.23 |
| Multi-ticker pipeline | 113.15 | 75.19 | 408.38 | 367.67 |
| Walk-forward | 146.58 | 105.83 | 336.18 | 249.85 |

Every case produced exactly equal compared arrays. Comparisons include window
values, scaler state/transforms, ANN weights and loss histories, probabilities,
predictions, classification metrics and backtest outputs where available.

These are **peak traced allocations during each workload**, not total process
RAM/RSS. Input construction is excluded; returned arrays are included. NumPy
buffers are traced, but not all native BLAS allocations. Walk-forward feature
construction is outside the measured region; static pipelines include it.
ANN fit includes a small final prediction used to compare the trained models.
Timings and savings depend on dataset size, memory layout, backend and hardware.

Blocked float64 variance accumulation can change reduction order; universal
bit-for-bit equality is not guaranteed across every input/backend. The final
float32 statistics and all downstream outputs were identical in the measured
cases and focused equivalence tests. These synthetic results do not establish
out-of-sample trading performance or replace the model-promotion baseline.

## Reproduce

From the repository root, with the project's NumPy/pandas dependencies installed:

```sh
python scripts/benchmark_memory.py \
  --baseline-ref 38a2412 \
  --rows 30000 \
  --repeat 5 \
  --output artifacts/benchmarks/ann-memory.json

python -m pytest -q --ignore=tests/test_mt5_walkforward_strategies.py
```

The benchmark reads only the relevant source modules from the specified local
Git revision. It does not check out a branch or download market data. The
reference should remain the pre-optimization commit; `HEAD` is only a useful
default while the patches are uncommitted. Larger unrelated changes to the
module interfaces may require updating the comparison harness.

The benchmark asserts exact equality for predictions, metrics, backtests and
best epochs, and numerical closeness for other floating-point outputs
(`rtol=1e-6`, `atol=1e-7`). Its `exact_outputs` field reports whether every
compared output is also exactly equal.

## Regression Coverage

`tests/test_memory_optimizations.py` adds 52 tests, including:

- Exact SGD equivalence against the original equations with five seeds, three
  dropout settings, validation/no-validation and early stopping.
- Missing classes, explicit class weights and read-only/non-contiguous inputs.
- Window values, indices, writable independence and zero-copy ANN flattening.
- Scaler state and transforms for C-order, F-order and strided arrays, constant
  and near-constant features, large values and multiple variance blocks.
- Allocation bounds for sequence-statistic fitting and ANN inference.
- Input immutability, label-mask selection and raw-window lifetime checks in the
  shared runner.

The core suite passes 136 tests. The separate pre-existing untracked local MT5
test is excluded because this verification interpreter lacks TA-Lib; the
project `.venv` lacks pytest and stalled on importing pandas. No environment
packages or market data were downloaded for this work.
