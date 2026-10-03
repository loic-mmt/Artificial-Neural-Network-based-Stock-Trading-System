"""Small end-to-end graph ablation and safe resume."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from trading_system.experiments.graph_ablation import GraphAblationConfig, run_graph_ablation
from trading_system.pipelines.diagnose_gnn_information import run_information_tests
from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
from trading_system.training.financial_loss import FinancialLossConfig


def _frame():
    dates = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    rows = []
    for ticker_index, ticker in enumerate(("A", "B", "C")):
        t = np.arange(len(dates))
        close = 80 + 20 * ticker_index + .12 * t + 2 * np.sin(t / (5 + ticker_index))
        for index, day in enumerate(dates):
            rows.append({
                "date": day, "ticker": ticker,
                "sector": "Finance" if ticker != "C" else "Energy",
                "open": close[index] * 0.999, "high": close[index] * 1.01,
                "low": close[index] * .99, "close": close[index],
                "adj_close": close[index], "volume": 1000 + 10 * index + ticker_index,
            })
    return pd.DataFrame(rows)


def test_graph_ablation_uses_matched_folds_and_resumes_without_retraining(tmp_path):
    config = replace(DEFAULT_CONFIG, context_len=5, device="cpu")
    loss = FinancialLossConfig("sharpe")
    parameters = {"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1}
    ablation = GraphAblationConfig(graph_lookback=10, graph_threshold=.3,
                                    gnn_hidden_size=4, date_batch_size=16)
    data_path = tmp_path / "prices.parquet"
    _frame().to_parquet(data_path, index=False)
    selection_path = tmp_path / "tickers.json"
    selection_path.write_text(json.dumps({"tickers": ["A", "B", "C"]}))
    args = (pd.read_parquet(data_path), config, loss, parameters, [1], tmp_path / "graphs")
    options = dict(ablation=ablation, n_splits=2, gap_bars=2)
    report = run_graph_ablation(*args, **options)
    assert len(report["folds"]) == 10
    assert len(report["summary"]) == 5
    assert len(report["paired_vs_gru"]) == 4
    assert report["final_test"] == []
    information = run_information_tests(
        tmp_path / "graphs", tmp_path / "information", data_path, selection_path,
        modes=("identity", "sector"), folds=[0], seeds=[1], device="cpu",
        block_length=5, bootstrap_samples=100,
    )
    assert information["final_holdout_opened"] is False
    assert information["hash_mismatch_override"] is False
    assert len(information["statistics"]) == 2
    assert (tmp_path / "information" / "predictions.parquet").is_file()
    for fold in (0, 1):
        rows = [row for row in report["folds"] if row["fold"] == fold]
        assert len({row["outer_dates"] for row in rows}) == 1
        assert len({row["train_dates"] for row in rows}) == 1
        assert all(row["outer_metrics"]["assets"] == 3 for row in rows)
        assert all(0 <= row["classification"]["macro_f1"] <= 1 for row in rows)
        assert all(row["classification"]["nll"] >= 0 for row in rows)
        assert next(row for row in rows if row["candidate"] == "sector")["graph_train"]["mean_density"] > 0
        assert (tmp_path / "graphs" / f"fold-{fold}-sector-graphs.json.gz").is_file()
    progress = tmp_path / "graphs" / "folds.json"
    progress.write_text(json.dumps(json.loads(progress.read_text())[:-1]))
    resumed = run_graph_ablation(*args, **options, resume=True)
    assert len(resumed["folds"]) == 10
    assert len({(row["candidate"], row["seed"], row["fold"]) for row in resumed["folds"]}) == 10
    with pytest.raises(ValueError, match="Resume metadata"):
        run_graph_ablation(*args, **{**options, "ablation": replace(ablation, graph_threshold=.4)}, resume=True)


def test_graph_ablation_supports_topk_residuals_and_market_transformer_gate(tmp_path):
    frame = _frame()
    dates = pd.DatetimeIndex(frame["date"].unique())
    context = pd.DataFrame({
        "date": dates,
        "spy_close": 300 + .2 * np.arange(len(dates)) + np.sin(np.arange(len(dates)) / 9),
        "xlf_close": 100 + .1 * np.arange(len(dates)) + np.sin(np.arange(len(dates)) / 7),
        "xle_close": 80 + .08 * np.arange(len(dates)) + np.cos(np.arange(len(dates)) / 8),
    })
    context["source_end"] = context["date"] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    market = pd.DataFrame({
        "date": dates,
        "spy_return": context["spy_close"].pct_change(fill_method=None),
        "source_end": context["source_end"],
    })
    config = replace(DEFAULT_CONFIG, context_len=5, device="cpu")
    ablation = GraphAblationConfig(
        graph_lookback=10,
        graph_weight_mode="absolute",
        graph_neighbors=1,
        graph_rebalance_bars=3,
        gnn_hidden_size=4,
        date_batch_size=16,
        candidates=("gru", "rolling_topk_market", "rolling_residual_topk_market", "identity", "identity_market"),
        market_transformer_width=8,
        market_transformer_heads=2,
    )
    report = run_graph_ablation(
        frame,
        config,
        FinancialLossConfig("sharpe"),
        {"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1},
        [1],
        tmp_path / "relational-market",
        ablation=ablation,
        n_splits=2,
        gap_bars=2,
        graph_context=context,
        sector_context_columns={"Finance": "xlf_close", "Energy": "xle_close"},
        market_frame=market,
        market_columns=("spy_return",),
    )
    assert len(report["folds"]) == 10
    assert {row["candidate"] for row in report["summary"]} == set(ablation.candidates)
    residual = next(row for row in report["folds"]
                    if row["candidate"] == "rolling_residual_topk_market")
    assert residual["graph_outer"]["mean_isolated"] == 0
    for row in report["folds"]:
        if row["candidate"] in ("identity", "identity_market"):
            for partition in ("train", "inner", "outer"):
                assert row[f"graph_{partition}"]["mean_density"] == 0
                assert row[f"graph_{partition}"]["mean_isolated"] == 3
