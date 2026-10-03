# Sample weighting

Optional event importance weighting for Manual ANN, RNN, LSTM, GRU and Transformer.
This changes training/early-stopping losses, not input features, label generation,
decision-policy calibration, classification metrics, or backtest execution.
Current automatic weight sources require **Triple Barrier** diagnostics.

## CLI

Append to a comparison, walk-forward, or walk-forward grid-search command:

```bash
--label-method triple-barrier --sample-weighting net_return
```

Available modes:

- `none` (default): existing class-weighted training, without sample weighting.
- `net_return`: `abs(net_event_return)` for observed events.
- `volatility`: `abs(label_score)`, the existing net event return / barrier scale.
  The scale follows the chosen Triple Barrier estimator (return std, ATR or
  Bollinger), not necessarily daily return volatility.
- `uniqueness`: `abs(net_event_return)` multiplied by average inverse event
  concurrency. A sweep over `(entry, end]` intervals computes calendar-duration
  averages independently per ticker and temporal split, among retained observed
  samples. This is an explicit overlap proxy, not return-attribution weighting.

Limits: `--sample-weight-clip-quantile 0.99`, `--sample-weight-min 0.1`,
`--sample-weight-max 10`. Parameters without an enabled mode fail explicitly.
Python: `ExperimentConfig(sample_weighting=SampleWeightConfig(...))`, where
`SampleWeightConfig` is in `trading_system.training.sample_weighting`.

## Normalization and safeguards

1. Construct sequences, remove unknown target rows, and retain the aligned event
   diagnostics in the same order as each model's training/validation labels.
2. Fit the upper quantile on **positive training-event magnitudes** only. Clip
   all training-event magnitudes, including zeros, and compute their mean scale.
3. Divide event magnitudes by that scale. Non-event Hold/Flat/Carry rows receive
   a neutral base weight of 1; inherited Carry positions do not invent returns.
4. Bound base weights to `[min_weight, max_weight]`, then divide all weights by
   their training mean. Final train mean is 1; the final bounds are likewise
   divided by that mean. Zero-return events retain a positive floor.
5. Validation uses the same frozen clip, scale and normalizer, without fitting
   to its distribution. Validation overlap is measured on validation events only.
   Early stopping uses weighted validation loss. Test never computes/fits weights.

No training events: explicit error. All event magnitudes zero: uniform weights
with an explicit `uniform_zero_magnitude` diagnostic. Missing/infinite magnitudes
for observed events: error. Unknown labels and mixed temporal splits: rejected.
Only the observed samples remaining after feature warmup, sequence alignment and
purging are used. Walk-forward recomputes these train-only statistics at each fit.

## Loss and model API

`fit(..., sample_weight=..., sample_weight_val=...)` accepts aligned 1D arrays.
Existing inverse-frequency class weights are still derived from training labels.
For a weighted batch, both engines minimize:

```text
effective_i = class_weight[label_i] * sample_weight_i
loss = sum(effective_i * cross_entropy_i) / sum(effective_i)
```

Manual ANN uses the same weights in its output gradient and recorded losses.
PyTorch carries weights inside each shuffled dataset item and uses unreduced
cross-entropy before reduction. Weighted validation aggregation uses effective
mass, so changing validation batch size does not change its loss. Disabling sample
weighting retains the legacy reduction path. Weight normalization cancels within
the weighted ratio; its benefits are reproducible diagnostics and bounded scale.

Direct API weights may contain zeros but must be finite, non-negative, aligned,
and have positive total mass. Zero-mass mini-batches skip the optimizer step.
Validation weights require validation data. With batch size 1, normalized weights
cancel for each nonzero sample; use multi-sample batches to change relative gradient
contributions. Custom model adapters must explicitly support the weighted fit API;
unsupported adapters fail rather than silently discarding weights.

## Results and verification

Configuration and fitted normalization summaries are saved in artifact
`experiment_parameters.sample_weight_state`, comparison `runs.csv` (JSON column),
and walk-forward retraining logs. Grid search exports the selected configuration
and logs. The comparison notebook displays the first loaded artifact's summary.
No extra dependencies and no new data downloads.

Tests cover exact ANN loss, unit-weight compatibility, changed optimization,
zero batches, event overlap/ticker isolation, clipping, alignment through sequence
filtering, static training, frozen final evaluation, CLI and walk-forward.
Dedicated PyTorch tests cover loss/gradient, shuffled alignment and validation
batch invariance. Their execution is currently blocked by the local `torch` import
reading dependency files; no successful PyTorch runtime validation is claimed.
After the local environment is available:

```bash
python -m pytest tests/test_sample_weighting.py tests/test_sample_weighting_torch.py -q
```

Benchmark `none`, `net_return`, `volatility`, and `uniqueness` with identical data,
labels, seeds, models and execution costs. Compare validation before final test.
No profitability or learnability improvement is established by implementation.
