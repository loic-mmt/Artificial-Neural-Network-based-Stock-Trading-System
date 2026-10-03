"""P0 end-to-end provenance, exported keys, recovery and verified reuse."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from trading_system.artifacts.multimodal_study import validate_completion
from trading_system.experiments import graph_ablation as graph
from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
from trading_system.training.financial_loss import FinancialLossConfig


def prices():
    dates = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    result = []
    for number, ticker in enumerate(("A", "B")):
        for index, day in enumerate(dates):
            close = 80 + number * 20 + index * .12 + np.sin(index / (5 + number))
            result.append(dict(date=day, ticker=ticker, sector="Finance", open=close,
                               high=close * 1.01, low=close * .99, close=close,
                               adj_close=close, volume=1000 + index * 10))
    return pd.DataFrame(result)


def inputs(destination, candidates=("gru",)):
    return (prices(), replace(DEFAULT_CONFIG, context_len=5, device="cpu"),
            FinancialLossConfig("sharpe"),
            {"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1},
            [1], destination), dict(
                ablation=graph.GraphAblationConfig(graph_lookback=10, gnn_hidden_size=4,
                                                 date_batch_size=16, candidates=candidates),
                n_splits=2, gap_bars=2,
            )


def test_exports_provenance_and_prediction_backtest_alignment(tmp_path):
    args, options = inputs(tmp_path / "source")
    report = graph.run_graph_ablation(*args, **options)
    metadata = report["metadata"]
    assert metadata["schema_version"] == 2
    assert metadata["cv_spec"]["initial_train_fraction"] == .5
    assert len(metadata["cv_folds"]) == 2
    assert metadata["calendar"]["tickers"] == ["A", "B"]
    assert metadata["class_names"] == ["Sell", "Hold", "Buy"]
    assert metadata["export_contract"]["unknown_label"] == -1
    assert metadata["export_contract"]["timezone"] == "UTC"
    for row in report["folds"]:
        validate_completion(row["completion_manifest"], args[-1], row["task_signature"])
        for partition in ("inner", "outer"):
            predictions = pd.read_parquet(args[-1] / row["prediction_artifacts"][partition])
            assert not predictions.duplicated(["date", "ticker"]).any()
            assert predictions["row_position"].tolist() == list(range(len(predictions)))
            expected_keys = predictions["date"].map(lambda value: value.isoformat()) + "|" + predictions["ticker"]
            assert predictions["backtest_key"].equals(expected_keys.rename("backtest_key"))
            assert np.allclose(predictions[["p_sell", "p_hold", "p_buy"]].sum(axis=1), 1)
            assert np.allclose(predictions["position"], predictions["p_buy"] - predictions["p_sell"])
            expected_prices = prices().set_index(["date", "ticker"])["adj_close"]
            keys = pd.MultiIndex.from_frame(predictions[["date", "ticker"]])
            assert np.array_equal(predictions.adj_close.to_numpy(), expected_prices.loc[keys].to_numpy())
            assert (predictions.loc[~predictions.label_known, "label"] == -1).all()
            daily = pd.read_parquet(args[-1] / row["daily_path_artifacts"][partition])
            assert np.allclose(daily.net_return, daily.gross_return - daily.cost)
            assert daily.date.min() > predictions.date.min()


def test_planner_and_reuse_share_signatures_without_retraining(tmp_path, monkeypatch):
    args, options = inputs(tmp_path / "source")
    planned = graph.plan_run_graph_ablation(*args[:-1], **options)
    original = graph.run_graph_ablation(*args, **options)
    assert {item["signature"] for item in planned["task_specs"]} == {
        row["task_signature"] for row in original["folds"]}
    monkeypatch.setattr(graph, "_fit", lambda *a, **k: pytest.fail("Compatible control retrained"))
    reused = graph.run_graph_ablation(
        *(*args[:-1], tmp_path / "imported"),
        **{**options, "ablation": replace(options["ablation"], gnn_layers=2)},
        reuse_from=[args[-1]],
    )
    assert all(row["reuse"]["verified"] for row in reused["folds"])
    assert reused["summary"] == original["summary"]
    assert not list((tmp_path / "imported").glob("*.pt"))
    graph.run_graph_ablation(
        *(*args[:-1], tmp_path / "imported"),
        **{**options, "ablation": replace(options["ablation"], gnn_layers=2)}, resume=True,
    )


def test_resume_binds_all_cv_boundaries_and_immutable_metrics(tmp_path, monkeypatch):
    args, options = inputs(tmp_path / "source")
    original = graph.run_graph_ablation(*args, **options)
    path = args[-1] / "folds.json"
    editable = json.loads(path.read_text())
    editable[0]["score"] = 12345
    editable[0]["outer_metrics"]["net_return"] = 12345
    path.write_text(json.dumps(editable))
    monkeypatch.setattr(graph, "_fit", lambda *a, **k: pytest.fail("Completed task retrained"))
    resumed = graph.run_graph_ablation(*args, **options, resume=True)
    assert resumed["summary"] == original["summary"]
    with pytest.raises(ValueError, match="Resume metadata"):
        graph.run_graph_ablation(*args, **options, initial_train_fraction=.55, resume=True)
    with pytest.raises(ValueError, match="Resume metadata"):
        graph.run_graph_ablation(*args, **options, inner_val_fraction=.25, resume=True)


def test_corrupt_or_missing_task_retrains_only_that_task(tmp_path, monkeypatch):
    args, options = inputs(tmp_path / "source")
    report = graph.run_graph_ablation(*args, **options)
    row = report["folds"][0]
    (args[-1] / row["prediction_artifacts"]["inner"]).write_bytes(b"corrupted")
    original_fit, calls = graph._fit, []
    def fit(*a, **k):
        calls.append(1)
        return original_fit(*a, **k)
    monkeypatch.setattr(graph, "_fit", fit)
    recovered = graph.run_graph_ablation(*args, **options, resume=True)
    assert len(calls) == 1
    assert len(recovered["folds"]) == 2
    assert (args[-1] / "recovery.json").is_file()
    for saved in recovered["folds"]:
        validate_completion(saved["completion_manifest"], args[-1], saved["task_signature"])


def test_partial_tasks_do_not_select_a_winner(tmp_path):
    args, options = inputs(tmp_path / "partial", candidates=("identity",))
    report = graph.run_graph_ablation(*args, **options, task_filter={("identity", 1, 0)})
    assert len(report["folds"]) == 1
    assert report["selected"] is None
    assert report["paired_vs_gru"] == []
    assert report["final_test"] == []


def test_one_corrupt_reference_does_not_block_other_verified_imports(tmp_path, monkeypatch):
    args, options = inputs(tmp_path / "source")
    source = graph.run_graph_ablation(*args, **options)
    broken = source["folds"][0]
    (args[-1] / broken["prediction_artifacts"]["inner"]).write_bytes(b"corrupted")
    fit_original, calls = graph._fit, []
    def fit(*a, **k):
        calls.append(1)
        return fit_original(*a, **k)
    monkeypatch.setattr(graph, "_fit", fit)
    imported = graph.run_graph_ablation(*(*args[:-1], tmp_path / "imported"),
                                        **options, reuse_from=[args[-1]])
    assert len(calls) == 1
    assert len([row for row in imported["folds"] if row.get("reuse")]) == 1


def test_triple_barrier_unknown_labels_export_as_unknown_not_hold(tmp_path):
    args, options = inputs(tmp_path / "triple-barrier")
    config = replace(args[1], label_mode="triple_barrier", triple_barrier_max_holding=3,
                     triple_barrier_volatility_window=5)
    report = graph.run_graph_ablation(args[0], config, *args[2:], **options)
    outer = pd.read_parquet(args[-1] / report["folds"][0]["prediction_artifacts"]["outer"])
    assert (~outer.label_known).any()
    assert (outer.loc[~outer.label_known, "label"] == -1).all()


def test_interrupted_or_invalid_references_do_not_block_reuse_scan(tmp_path):
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "metadata.json").write_text('{"schema_version": 2}')
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "metadata.json").write_text("not json")
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "metadata.json").write_text('{"schema_version": 2}')
    (malformed / "folds.json").write_text('[null, {"status": "pending"}]')
    assert graph._reuse_rows([interrupted, corrupt, malformed, tmp_path / "missing"]) == {}
