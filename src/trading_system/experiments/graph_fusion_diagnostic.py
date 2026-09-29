"""Fixed-weight GRU/GNN fusion on saved outer-fold positions, without fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.experiments.graph_information import moving_block_interval
from trading_system.training.financial_loss import ReturnPanel


FUSION_GRU_WEIGHTS = (0.5, 0.75)
GRAPH_CANDIDATES = ("identity", "rolling_pearson")


def _daily_interval(values, *, block_length, samples, seed):
    interval = moving_block_interval(values, block_length=block_length,
                                     samples=samples, seed=seed)
    return {
        "mean_bps_per_day": interval["mean"] * 1e4,
        "ci_95_bps_per_day": [value * 1e4 for value in interval["ci_95"]],
    }


def evaluate_fixed_fusions(
    paired: pd.DataFrame,
    loss,
    *,
    execution_delay=1,
    initial_capital=10000.,
    block_length=20,
    samples=1000,
    seed=1,
):
    """Evaluate predeclared blends and an ex-post exposure-matched GRU control.

    `paired` contains one outer-fold row per date/ticker and position columns
    `gru`, `identity`, `rolling_pearson`. The exposure factor uses the *whole*
    outer fold's positions, never realized returns. It is a descriptive control,
    not an executable or validation-selected sizing rule.
    """
    required = {"date", "ticker", "adj_close", "gru", *GRAPH_CANDIDATES}
    if required - set(paired):
        raise ValueError(f"Missing fusion columns: {sorted(required - set(paired))}")
    work = paired.sort_values(["date", "ticker"]).reset_index(drop=True)
    if work.empty or work.duplicated(["date", "ticker"]).any():
        raise ValueError("Fusion requires non-empty unique ticker/date rows.")
    positions = {name: work[name].to_numpy(dtype=np.float64)
                 for name in ("gru", *GRAPH_CANDIDATES)}
    if any(not np.isfinite(values).all() or (np.abs(values) > 1 + 1e-6).any()
           for values in positions.values()):
        raise ValueError("Fusion positions must be finite and bounded by [-1, 1].")
    panel = ReturnPanel(work, price_col="adj_close", date_col="date",
                        group_col="ticker", execution_delay=execution_delay)
    baseline = panel.metrics(positions["gru"], loss, initial_capital)
    baseline_daily = panel.path(positions["gru"], loss)[0]
    if baseline["mean_abs_position"] <= 0:
        raise ValueError("GRU exposure is zero; exposure matching is undefined.")
    results, daily = [], []
    for graph_index, graph in enumerate(GRAPH_CANDIDATES):
        for weight_index, gru_weight in enumerate(FUSION_GRU_WEIGHTS):
            blend = gru_weight * positions["gru"] + (1 - gru_weight) * positions[graph]
            blend_metrics = panel.metrics(blend, loss, initial_capital)
            blend_daily = panel.path(blend, loss)[0]
            scale = blend_metrics["mean_abs_position"] / baseline["mean_abs_position"]
            scaled_gru = scale * positions["gru"]
            if (np.abs(scaled_gru) > 1 + 1e-6).any():
                raise ValueError("Exposure-matched GRU would exceed position bounds.")
            control_metrics = panel.metrics(scaled_gru, loss, initial_capital)
            control_daily = panel.path(scaled_gru, loss)[0]
            if not np.isclose(control_metrics["mean_abs_position"],
                              blend_metrics["mean_abs_position"], rtol=1e-10, atol=1e-12):
                raise ValueError("Exposure control failed to match the fusion.")
            offset = graph_index * len(FUSION_GRU_WEIGHTS) + weight_index
            results.append({
                "gnn": graph, "gru_weight": gru_weight,
                "ex_post_gru_scale": float(scale),
                "fusion": blend_metrics,
                "exposure_matched_gru": control_metrics,
                "fusion_minus_gru": _daily_interval(
                    blend_daily - baseline_daily, block_length=block_length,
                    samples=samples, seed=seed + 10 * offset,
                ),
                "fusion_minus_exposure_matched_gru": _daily_interval(
                    blend_daily - control_daily, block_length=block_length,
                    samples=samples, seed=seed + 10 * offset + 1,
                ),
            })
            daily.append(pd.DataFrame({
                "date": pd.to_datetime(panel.dates, utc=True),
                "gnn": graph, "gru_weight": gru_weight,
                "gru_net": baseline_daily, "fusion_net": blend_daily,
                "exposure_matched_gru_net": control_daily,
            }))
    # Compare actual graph mixing against the second-model identity control
    # at the same predeclared weight and on exactly the same daily path.
    for weight in FUSION_GRU_WEIGHTS:
        identity = next(item for item in daily if item.gnn.iat[0] == "identity"
                        and item.gru_weight.iat[0] == weight)
        rolling = next(item for item in daily if item.gnn.iat[0] == "rolling_pearson"
                       and item.gru_weight.iat[0] == weight)
        interval = _daily_interval(
            rolling.fusion_net.to_numpy() - identity.fusion_net.to_numpy(),
            block_length=block_length, samples=samples,
            seed=seed + 100 + int(weight * 100),
        )
        for result in results:
            if result["gnn"] == "rolling_pearson" and result["gru_weight"] == weight:
                result["rolling_minus_identity_fusion"] = interval
    return {"gru": baseline, "fusions": results}, pd.concat(daily, ignore_index=True)


__all__ = ["FUSION_GRU_WEIGHTS", "GRAPH_CANDIDATES", "evaluate_fixed_fusions"]
