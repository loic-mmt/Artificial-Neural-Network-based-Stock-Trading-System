"""Fixed late fusion must not fit on outer returns or silently shrink exposure."""

from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.experiments.graph_fusion_diagnostic import evaluate_fixed_fusions
from trading_system.pipelines.diagnose_gnn_fusion import run_fusion_diagnostic
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


def _paired():
    dates = pd.date_range("2020-01-01", periods=35, freq="B", tz="UTC")
    rows = []
    for index, day in enumerate(dates):
        for ticker in ("A", "B"):
            rows.append({
                "date": day, "ticker": ticker,
                "adj_close": 100 + index + (2 if ticker == "B" else 0),
                "gru": .2 + .02 * np.sin(index),
                "identity": .2 + .02 * np.sin(index),
                "rolling_pearson": .2 + .02 * np.sin(index),
            })
    return pd.DataFrame(rows)


def test_identical_branches_give_identical_fusions_and_intervals():
    report, daily = evaluate_fixed_fusions(
        _paired(), FinancialLossConfig("sharpe"), block_length=5, samples=100,
    )
    assert len(report["fusions"]) == 4
    assert len(daily) == 4 * 34
    for item in report["fusions"]:
        assert item["ex_post_gru_scale"] == pytest.approx(1)
        assert item["fusion"]["net_return"] == pytest.approx(report["gru"]["net_return"])
        assert item["fusion_minus_gru"]["ci_95_bps_per_day"] == [0, 0]
        assert item["fusion_minus_exposure_matched_gru"]["ci_95_bps_per_day"] == [0, 0]


def test_exposure_control_matches_diluted_gru_without_returns_fitting():
    paired = _paired()
    paired["identity"] = 0.
    paired["rolling_pearson"] = 0.
    report, _ = evaluate_fixed_fusions(
        paired, FinancialLossConfig("sharpe"), block_length=5, samples=100,
    )
    for item in report["fusions"]:
        assert item["ex_post_gru_scale"] == pytest.approx(item["gru_weight"])
        assert item["fusion"]["mean_abs_position"] == pytest.approx(
            item["exposure_matched_gru"]["mean_abs_position"])
        assert item["fusion_minus_exposure_matched_gru"]["mean_bps_per_day"] == pytest.approx(0)


def test_pipeline_rejects_unmatched_predictions_and_preserves_holdout(tmp_path):
    paired = _paired()
    loss = FinancialLossConfig("sharpe")
    panel = ReturnPanel(paired, group_col="ticker")
    baseline = panel.metrics(paired.gru.to_numpy(), loss)
    run = tmp_path / "run"
    information = tmp_path / "information"
    run.mkdir()
    information.mkdir()
    source = {
        "metadata": {
            "dataset_sha256": "matching-data", "final_holdout_opened": False,
            "final_split": {"test_start": "2021-01-01T00:00:00+00:00"},
            "n_splits": 1, "seeds": [1], "loss_config": asdict(loss),
            "config": {"execution_delay": 1, "initial_capital": 10000.},
        },
        "final_test": [],
        "folds": [{"candidate": "gru", "fold": 0, "seed": 1, "status": "ok",
                   "outer_dates": 35, "outer_metrics": baseline}],
    }
    (run / "report.json").write_text(json.dumps(source))
    (information / "statistics.json").write_text(json.dumps({
        "source_dataset_sha256": "matching-data",
        "replay_dataset_sha256": "matching-data",
        "hash_mismatch_override": False, "final_holdout_opened": False,
    }))
    predictions = pd.concat([
        paired[["date", "ticker", "adj_close"]].assign(
            position=paired[name], candidate=name, fold=0, seed=1)
        for name in ("gru", "identity", "rolling_pearson")
    ], ignore_index=True)
    predictions.to_parquet(information / "predictions.parquet", index=False)
    result = run_fusion_diagnostic(
        information, run, tmp_path / "fusion", block_length=5, bootstrap_samples=100,
    )
    assert result["final_holdout_opened"] is False
    assert result["exposure_control"].startswith("ex_post")
    assert len(result["summary"]) == 4
    assert (tmp_path / "fusion" / "daily_returns.parquet").is_file()
    with pytest.raises(FileExistsError):
        run_fusion_diagnostic(information, run, tmp_path / "fusion")
    predictions = predictions.iloc[:-1]
    predictions.to_parquet(information / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        run_fusion_diagnostic(
            information, run, tmp_path / "incomplete", block_length=5,
            bootstrap_samples=100,
        )
