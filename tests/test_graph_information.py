"""Date-blocked GRU/GNN information diagnostics."""

import numpy as np
import pandas as pd
import pytest

from trading_system.experiments.graph_information import (
    moving_block_interval, paired_graph_information,
)
from trading_system.training.financial_loss import FinancialLossConfig


def _paired():
    dates = pd.date_range("2020-01-01", periods=35, freq="B", tz="UTC")
    rows = []
    for index, day in enumerate(dates):
        for asset in ("A", "B"):
            rows.append({
                "date": day, "ticker": asset,
                "adj_close": 100 + index + (2 if asset == "B" else 0),
                "gru_position": .3, "gnn_position": .3,
                "gnn_degree": 1 if asset == "A" else 0,
                "high_volatility": index % 2 == 0,
            })
    return pd.DataFrame(rows)


def test_identical_positions_have_zero_incremental_effect():
    result = paired_graph_information(
        _paired(), FinancialLossConfig("sharpe"),
        block_length=5, samples=100, seed=7,
    )
    assert result["position_correlation"] is None
    assert result["opposite_signal_rows"] == 0
    assert result["net_gnn_minus_gru"]["mean_bps_per_day"] == 0
    assert result["net_gnn_minus_gru"]["ci_95_bps_per_day"] == [0, 0]
    assert result["connected_signal_fraction"] == .5


def test_opposed_signal_attribution_and_reproducible_intervals():
    paired = _paired()
    paired.loc[paired.ticker.eq("A"), "gnn_position"] = -.3
    kwargs = dict(block_length=5, samples=100, seed=11)
    first = paired_graph_information(paired, FinancialLossConfig("sharpe"), **kwargs)
    second = paired_graph_information(paired, FinancialLossConfig("sharpe"), **kwargs)
    assert first == second
    assert first["opposite_signal_fraction"] == .5
    assert first["opposite_signal_rows"] > 0
    assert first["connected_node_gross_contribution"]["mean_bps_per_day"] < 0


def test_block_interval_validates_inputs():
    with pytest.raises(ValueError, match="finite"):
        moving_block_interval(np.array([1., np.nan]))
    with pytest.raises(ValueError, match="block_length"):
        moving_block_interval(np.ones(5), block_length=0)
