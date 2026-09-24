"""Matched market-context CV, causal source checks, and safe resume."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from trading_system.experiments.market_gru_ablation import (
    MarketAblationConfig, build_close_market_frame, run_market_gru_ablation,
)
from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
from trading_system.training.financial_loss import FinancialLossConfig


def _frame():
    days = pd.date_range("2020-01-01", periods=160, freq="B", tz="UTC")
    rows = []
    for number, ticker in enumerate(("A", "B")):
        time = np.arange(len(days))
        close = 80 + number * 10 + .1 * time + np.sin(time / 5 + number)
        for index, day in enumerate(days):
            rows.append({
                "date": day, "ticker": ticker, "sector": "Finance",
                "open": close[index] * .999, "high": close[index] * 1.01,
                "low": close[index] * .99, "close": close[index],
                "adj_close": close[index], "volume": 1000 + index,
                "market_close": 300 + .2 * index + np.sin(index / 9),
                "vix_close": 20 + np.sin(index / 7),
            })
    return pd.DataFrame(rows)


def test_market_frame_rejects_conflicts_and_unverified_macro():
    frame = _frame()
    result, audit = build_close_market_frame(frame, "date", ("market_close", "vix_close"))
    assert audit["feature_coverage"]["vix_level"] == 1
    assert len(result) == 160
    changed = frame.copy()
    changed.loc[0, "vix_close"] += 1
    with pytest.raises(ValueError, match="Conflicting global market"):
        build_close_market_frame(changed, "date", ("vix_close",))
    with pytest.raises(ValueError, match="Only audited close"):
        build_close_market_frame(frame, "date", ("ust10y",))


def test_market_ablation_matches_folds_and_resumes(tmp_path):
    frame = _frame()
    config = replace(DEFAULT_CONFIG, context_len=5, device="cpu")
    loss = FinancialLossConfig("sharpe")
    params = {"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1}
    market = MarketAblationConfig(date_batch_size=16, transformer_width=8,
                                  transformer_heads=2)
    args = (frame, config, loss, params, [1], tmp_path / "market")
    options = {"ablation": market, "n_splits": 2, "gap_bars": 2}
    report = run_market_gru_ablation(*args, **options)
    assert len(report["folds"]) == 12
    assert len(report["summary"]) == 6
    assert report["final_test"] == []
    for fold in (0, 1):
        rows = [row for row in report["folds"] if row["fold"] == fold]
        assert len({row["train_dates"] for row in rows}) == 1
        assert len({row["outer_dates"] for row in rows}) == 1
        assert all(row["outer_metrics"]["assets"] == 2 for row in rows)
    progress = tmp_path / "market" / "folds.json"
    progress.write_text(json.dumps(json.loads(progress.read_text())[:-1]))
    resumed = run_market_gru_ablation(*args, **options, resume=True)
    assert len(resumed["folds"]) == 12
    with pytest.raises(ValueError, match="Resume metadata"):
        run_market_gru_ablation(*args, **{**options, "ablation": replace(market, transformer_width=16)}, resume=True)
