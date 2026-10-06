"""Persisted-format fixtures exercise identity isolation and provenance."""

from pathlib import Path
import json
import hashlib

import pandas as pd
import pytest

from trading_system.visualization import (
    RunRecord, RunData, catalog_frame, catalog_signature, comparison_issues,
    discover_runs, load_run, asset_names, asset_frame, position_events,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_advanced_index_timestamps_and_missing_tables(tmp_path):
    path = tmp_path / "legacy"
    write_json(path / "manifest.json", {"schema_version": "v3.x", "hashes": {"raw_market_hash": "a"}})
    write_json(path / "config.json", {"run_train_only": True, "symbol": "ABC", "annualization_factor": 8760})
    write_json(path / "metrics/core_metrics.json", {"sharpe_ratio": 1.5, "cumulative_return": .2})
    write_parquet(path / "backtest/equity_curve.parquet", pd.DataFrame({"equity": [100, 110, 90]}, index=pd.date_range("2020-01-01", periods=3, tz="UTC", name="timestamp")))
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 1
    record = catalog.records[0]
    assert record.family == "backtest" and record.status == "partial"
    assert record.metadata["partition"] == "train"
    assert record.metadata["annualization_factor"] == 8760
    data = load_run(record, start="2020-01-02", include_details=True)
    assert data.equity.date.dt.tz is not None
    assert data.equity.equity.tolist() == [110, 90]
    assert data.equity.drawdown.iloc[-1] == pytest.approx(90 / 110 - 1)
    assert data.trades.empty
    assert catalog.issues


def test_graph_identity_compounding_and_projection(tmp_path, monkeypatch):
    path = tmp_path / "graph"
    rows = []
    for fold in (0, 1):
        for seed in (1, 7):
            for partition in ("inner", "outer"):
                for date, ret in zip(pd.date_range("2020-01-01", periods=2, tz="UTC"), (.1, -.1)):
                    rows.append({"date": date, "candidate": "gru", "fold": fold, "seed": seed, "partition": partition, "net_return": ret, "buy_hold_return": .05, "large_unused_feature": "ignore"})
    write_parquet(path / "daily_paths.parquet", pd.DataFrame(rows))
    write_json(path / "report.json", {"metadata": {"initial_capital": 100}, "folds": [{"candidate": "gru", "fold": 0, "seed": 1, "outer_metrics": {"net_sharpe": 2, "regularized_sharpe": 1.7}, "inner_metrics": {"net_sharpe": 1.5}}]})
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 8
    record = next(record for record in catalog.records if record.selectors == {"candidate": "gru", "fold": 0, "seed": 1, "partition": "outer"})
    assert record.metrics["net_sharpe"] == 2
    assert record.metrics["regularized_sharpe"] == 1.7
    data = load_run(record, start="2020-01-02")
    assert data.equity.equity.tolist() == pytest.approx([99])
    assert data.equity.benchmark_equity.tolist() == pytest.approx([110.25])
    assert "large_unused_feature" not in data.equity
    assert len(data.equity) == 1
    inner = next(record for record in catalog.records if record.selectors == {"candidate": "gru", "fold": 0, "seed": 1, "partition": "inner"})
    assert inner.metrics["net_sharpe"] == 1.5
    assert "regularized_sharpe" not in inner.metrics


def test_unselected_identity_and_duplicate_dates_rejected(tmp_path):
    path = tmp_path / "paths.parquet"
    write_parquet(path, pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "equity": [100., 110.], "seed": [1, 7]}))
    record = RunRecord("x", "x", "graph", tmp_path, tables={"equity": path})
    data = load_run(record)
    assert data.equity.empty
    assert any("mixes seed" in warning for warning in data.warnings)
    write_parquet(path, pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "equity": [100., 110.]}))
    data = load_run(record)
    assert data.equity.empty
    assert any("duplicate dates" in warning for warning in data.warnings)


def test_metrics_only_and_incremental_results_history(tmp_path):
    complete = tmp_path / "benchmark"
    write_json(complete / "report.json", {"runs": [{"run_id": 1, "model_name": "gru", "seed": 1, "status": "ok", "backtest_sharpe_ratio": 1.2}, {"run_id": 2, "model_name": "gru", "seed": 7, "status": "error", "error": "OOM"}]})
    incremental = tmp_path / "post-open"
    write_json(incremental / "metadata.json", {"protocol": "post_open"})
    write_json(incremental / "results.json", [{"candidate": "native", "fold": 0, "seed": 1, "metrics": {"net_sharpe": .8}, "train_loss": [.5, .4], "inner_loss": [.7, .6]}])
    checkpoint = complete / "folds/00001"
    write_json(checkpoint / "report.json", {"runs": [{"run_id": 999}]})
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 3
    records = catalog_frame(catalog.records)
    assert not records.has_equity.any()
    native = next(record for record in catalog.records if record.metadata.get("candidate") == "native")
    assert native.status == "partial"
    assert load_run(native, include_details=True).history["val_loss"] == [.7, .6]
    error = next(record for record in catalog.records if record.status == "error")
    assert "OOM" in error.warnings


def test_mt5_decoder_and_retrain_have_distinct_series(tmp_path):
    dates = pd.date_range("2020-01-01", periods=2)
    decoders = tmp_path / "decoder"
    write_parquet(decoders / "equity-curves.parquet", pd.DataFrame({"date": list(dates) * 2, "model_equity": [100., 120., 100., 90.], "buy_hold_equity": [100., 110.] * 2, "decoder": ["sign"] * 2 + ["confidence"] * 2}))
    write_json(decoders / "report.json", {"rows": [{"decoder": "sign", "net_sharpe": 1., "regularized_sharpe": .8}, {"decoder": "confidence", "net_sharpe": .5, "regularized_sharpe": .3}]})
    retrain = tmp_path / "retrain"
    write_parquet(retrain / "equity-curves.parquet", pd.DataFrame({"date": list(dates) * 2, "model_equity": [100., 110., 100., 115.], "buy_hold_equity": [100., 120.] * 2, "retrain_every_sessions": [5] * 2 + [20] * 2, "seed": [42] * 4}))
    write_json(retrain / "report.json", {"decoder": "confidence", "runs": [{"retrain_every_sessions": 5, "seed": 42, "net_sharpe": 2., "regularized_sharpe": 1.9}, {"retrain_every_sessions": 20, "seed": 42, "net_sharpe": 1.}]})
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 4
    assert len({record.run_id for record in catalog.records}) == 4
    selected = next(record for record in catalog.records if record.selectors.get("retrain_every_sessions") == 5)
    assert selected.metadata["decoder"] == "confidence"
    assert load_run(selected).equity.equity.tolist() == [100., 110.]
    assert selected.metrics["regularized_sharpe"] == 1.9


def test_trading_arms_not_duplicated_and_trades_overlap(tmp_path):
    path = tmp_path / "replay"
    write_json(path / "report.json", {"config": {"initial_capital": 100, "fees_bps": 5}, "comparisons": [{"directory": "group-000", "group": {"fold": 0, "seed": 7}, "without_rules": {"sharpe": 1.}, "with_rules": {"sharpe": 2.}}]})
    for arm in ("with_rules", "without_rules"):
        folder = path / "group-000" / arm
        write_json(folder / "metadata.json", {"config": {"initial_capital": 100}})
        write_parquet(folder / "equity.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3, tz="UTC"), "equity": [100., 110., 105.], "drawdown": [0., 0., -5 / 110]}))
        write_parquet(folder / "trades.parquet", pd.DataFrame({"entry_time": pd.to_datetime(["2019-12-31", "2020-01-04"], utc=True), "exit_time": pd.to_datetime(["2020-01-03", "2020-01-05"], utc=True), "net_pnl": [5., 10.]}))
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 2
    assert {record.metadata["arm"] for record in catalog.records} == {"with_rules", "without_rules"}
    data = load_run(catalog.records[0], start="2020-01-02", end="2020-01-03", include_details=True)
    assert len(data.trades) == 1 and data.trades.net_pnl.iloc[0] == 5


def test_comparability_and_signature_ignore_checkpoints(tmp_path):
    common = {"dataset_sha256": "abc", "tickers": ["B", "A"], "evaluation_start": "2020-01-01", "evaluation_end": "2020-01-03", "partition": "outer", "price_basis": "adjusted", "execution": "open_proxy", "signal_timing": "next_bar", "execution_delay": 1, "fees_bps": 5, "slippage_bps": 0, "annualization": 252, "initial_capital": 100}
    first = RunRecord("a", "a", "graph", tmp_path, metadata=common)
    second = RunRecord("b", "b", "graph", tmp_path, metadata={**common, "tickers": ["A", "B"]})
    assert comparison_issues([first, second]) == []
    second.metadata["fees_bps"] = 10
    second.metadata["partition"] = "inner"
    issues = comparison_issues([first, second])
    assert "Comparison mismatch: costs." in issues
    assert "Comparison mismatch: partition." in issues
    write_json(tmp_path / "report.json", {"runs": [{"status": "ok"}]})
    before = catalog_signature(tmp_path)
    write_json(tmp_path / "folds/0001/report.json", {"runs": [{"status": "error"}]})
    assert catalog_signature(tmp_path) == before
    write_json(tmp_path / "report.json", {"runs": [{"status": "error"}]})
    assert catalog_signature(tmp_path) != before


def test_invalid_json_does_not_hide_other_runs(tmp_path):
    write_json(tmp_path / "ok/report.json", {"runs": [{"model_name": "gru", "status": "ok"}]})
    corrupt = tmp_path / "corrupt/report.json"
    corrupt.parent.mkdir()
    corrupt.write_text("{partial")
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 1
    assert len(catalog.issues) == 1
    with pytest.raises(ValueError, match="start"):
        load_run(catalog.records[0], start="2020-01-03", end="2020-01-01")


def test_graph_replay_partition_metrics_and_existing_equity_benchmark(tmp_path):
    write_json(tmp_path / "replay.json", {"tasks": [{"candidate": "gru", "fold": 0, "seed": 1, "metrics": {"inner": {"net_sharpe": 1}, "outer": {"net_sharpe": 2}}}]})
    write_parquet(tmp_path / "daily_paths.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "equity": [110., 99.], "net_return": [.1, -.1], "buy_hold_return": [.05, .05], "candidate": ["gru"] * 2, "fold": [0] * 2, "seed": [1] * 2, "partition": ["outer"] * 2}))
    record = discover_runs(tmp_path).records[0]
    assert record.metrics["net_sharpe"] == 2
    assert load_run(record).equity.benchmark_equity.tolist() == pytest.approx([105, 110.25])


def test_graph_per_partition_artifacts_and_nested_mt5_metrics(tmp_path):
    graph = tmp_path / "graph"
    write_json(graph / "report.json", {"metadata": {"initial_capital": 100}, "folds": [{"candidate": "gru", "fold": 0, "seed": 1, "daily_path_artifacts": {"inner": "inner.parquet", "outer": "outer.parquet"}, "inner_metrics": {"net_sharpe": 1}, "outer_metrics": {"net_sharpe": 2}}]})
    for partition in ("inner", "outer"):
        write_parquet(graph / f"{partition}.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "net_return": [.1, -.1]}))
    mt5 = tmp_path / "mt5"
    write_json(mt5 / "metadata.json", {"seed": 42, "initial_capital": 100})
    write_json(mt5 / "portfolio_metrics.json", {"model": {"net_return": .1, "net_sharpe": 1.7}, "buy_hold": {"net_return": .2}, "vs_buy_hold_pnl": -10})
    write_parquet(mt5 / "equity_curve.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "model_equity": [100., 110.], "buy_hold_equity": [100., 120.]}))
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 3
    outer = next(record for record in catalog.records if record.selectors.get("partition") == "outer")
    assert outer.metrics["net_sharpe"] == 2
    assert load_run(outer).equity.equity.tolist() == pytest.approx([110, 99])
    direct = next(record for record in catalog.records if record.family == "mt5")
    assert direct.metrics["net_sharpe"] == 1.7
    assert direct.metrics["benchmark_net_return"] == .2


def test_signature_tracks_only_referenced_history_and_fold_paths(tmp_path):
    write_json(tmp_path / "report.json", {"folds": [{"candidate": "gru", "fold": 0, "seed": 1, "artifact_path": "folds/00001", "daily_path_artifacts": {"outer": "outer-path.parquet"}}]})
    write_parquet(tmp_path / "outer-path.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "net_return": [.1, -.1]}))
    write_json(tmp_path / "folds/00001/training_history.json", {"train_loss": [.5]})
    before = catalog_signature(tmp_path)
    write_json(tmp_path / "folds/00002/training_history.json", {"train_loss": [.4]})
    write_json(tmp_path / "models/unreferenced/metrics.json", {"a": 1})
    assert catalog_signature(tmp_path) == before
    write_json(tmp_path / "folds/00001/training_history.json", {"train_loss": [.5, .4]})
    after_history = catalog_signature(tmp_path)
    assert after_history != before
    write_parquet(tmp_path / "outer-path.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "net_return": [.1, -.1, .05]}))
    assert catalog_signature(tmp_path) != after_history


def test_gridsearch_keeps_validation_selection_and_final_test_separate(tmp_path):
    stem = "gridsearch_walkforward_20261003_123000"
    pd.DataFrame([{"trial_id": 1, "model_name": "gru", "seed": 7, "status": "ok", "selection_split": "validation", "val_net_return": .1, "selected": True}, {"trial_id": 2, "model_name": "gru", "seed": 7, "status": "ok", "selection_split": "validation", "val_net_return": .3, "selected": False}]).to_csv(tmp_path / f"{stem}.csv", index=False)
    write_json(tmp_path / f"{stem}.json", {"ticker": "ABC", "selection_split": "validation", "best_parameters": {"model_name": "gru", "seed": 7}, "final_test": {"test_metrics": {"macro_f1": .5}, "benchmark_comparison": {"model_return": .02}, "n_test_rows": 10}})
    catalog = discover_runs(tmp_path)
    assert len(catalog.records) == 3
    assert len({record.run_id for record in catalog.records}) == 3
    validation = [record for record in catalog.records if record.metadata["partition"] == "validation"]
    final = next(record for record in catalog.records if record.metadata["partition"] == "final_test")
    assert sum(record.metadata["selected"] for record in validation) == 1
    assert final.metrics["model_return"] == .02
    assert final.metrics["macro_f1"] == .5
    assert all("model_return" not in record.metrics for record in validation)
    assert "selected" in catalog_frame(catalog.records)
    signature = catalog_signature(tmp_path)
    write_json(tmp_path / f"{stem}.json", {"selection_split": "validation"})
    assert signature != catalog_signature(tmp_path)


def test_legacy_gridsearch_partition_unknown_and_partial_mt5(tmp_path):
    legacy = tmp_path / "gridsearch"
    legacy.mkdir()
    pd.DataFrame([{"trial_id": 1, "status": "ok", "model_pnl": 10, "n_test_rows": 5}]).to_csv(legacy / "gridsearch_walkforward_old.csv", index=False)
    write_json(legacy / "gridsearch_walkforward_old.json", {"meta": {"ticker": "ABC", "seed": 42}, "top": []})
    mt5 = tmp_path / "mt5"
    write_json(mt5 / "metadata.json", {"protocol": "expanding_walk_forward_v1"})
    write_parquet(mt5 / "positions.parquet", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "position": [0, 1]}))
    catalog = discover_runs(tmp_path)
    legacy_record = next(record for record in catalog.records if record.family == "benchmark")
    assert legacy_record.metadata["partition"] == "unknown"
    assert any("Legacy grid search" in warning for warning in legacy_record.warnings)
    mt5_record = next(record for record in catalog.records if record.family == "mt5")
    assert mt5_record.status == "partial"
    assert "equity" not in mt5_record.tables
    assert "positions" in mt5_record.tables


def test_comparison_flags_capital_and_annualization_conventions(tmp_path):
    first = RunRecord("a", "a", "backtest", tmp_path, metadata={"annualization_factor": 8760})
    second = RunRecord("b", "b", "trading", tmp_path, metadata={"annualization": 252})
    issues = comparison_issues([first, second])
    assert "Comparison mismatch: annualization." in issues
    assert "Comparison mismatch: capital_basis." in issues


def test_position_signals_ignore_initial_state_and_size_changes(tmp_path):
    dates = pd.date_range("2020-01-01", periods=7, tz="UTC")
    record = RunRecord("x", "x", "mt5", tmp_path)
    data = RunData(record, positions=pd.DataFrame({"date": dates, "ticker": ["A"] * 7, "adj_close": range(100, 107), "position": [1., .4, 0., -.2, -.7, 0., .5]}))
    events = position_events(data, "A")
    assert events.date.tolist() == list(dates[[2, 3, 5, 6]])
    assert events.side.tolist() == ["sell", "sell", "buy", "buy"]
    assert events.kind.tolist() == ["exit_long", "entry_short", "exit_short", "entry_long"]
    assert set(events.source) == {"position_signals"}
    assert position_events(data, "B").empty
    assert asset_names(data) == ["A"]


def test_position_flips_close_and_open_without_unknown_bridging(tmp_path):
    dates = pd.date_range("2020-01-01", periods=4, tz="UTC")
    record = RunRecord("x", "x", "graph", tmp_path)
    data = RunData(record, positions=pd.DataFrame({"date": dates, "ticker": ["A"] * 4, "adj_close": [100.] * 4, "position": [1., -1., 0., 1.]}))
    events = position_events(data, "A")
    assert events.iloc[:2].kind.tolist() == ["exit_long", "entry_short"]
    assert events.iloc[:2].side.tolist() == ["sell", "sell"]
    data.positions["available"] = [True, False, True, True]
    events = position_events(data, "A")
    assert events.kind.tolist() == ["entry_long"]
    assert events.date.tolist() == [dates[3]]


def test_actual_orders_take_precedence_and_keep_fill_time_price(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3, tz="UTC")
    record = RunRecord("x", "x", "trading", tmp_path, metadata={"price_basis": "split_adjusted"})
    data = RunData(record, positions=pd.DataFrame({"date": dates, "ticker": ["A"] * 3, "adj_close": [50.] * 3, "close": [100.] * 3, "position": [0, 1, -1]}), orders=pd.DataFrame({"date": [dates[2]], "timestamp": [dates[2] + pd.Timedelta(hours=9)], "ticker": ["A"], "quantity": [-2.], "quantity_before": [1.], "quantity_after": [-1.], "execution_price": [101.]}))
    events = position_events(data, "A")
    assert len(events) == 1
    assert events.source.iloc[0] == "executed_orders"
    assert events.kind.iloc[0] == "flip_long_to_short"
    assert events.quantity_delta.iloc[0] == -2.
    assert events.price.iloc[0] == 101.
    assert events.date.iloc[0] == dates[2] + pd.Timedelta(hours=9)
    assert asset_frame(data, "A").price.tolist() == [100.] * 3


def test_trade_markers_available_without_market_line(tmp_path):
    record = RunRecord("x", "x", "trading", tmp_path)
    data = RunData(record, trades=pd.DataFrame({"ticker": ["A"], "side": ["short"], "entry_time": ["2020-01-01"], "exit_time": ["2020-01-02"], "entry_price": [100.], "exit_price": [95.]}))
    assert asset_frame(data, "A").empty
    events = position_events(data, "A")
    assert events.side.tolist() == ["sell", "buy"]
    assert events.price.tolist() == [100., 95.]
    assert set(events.source) == {"executed_trades"}


def test_market_projection_tickers_dates_and_transition_before_slice(tmp_path):
    dates = pd.date_range("2020-01-01", periods=4, tz="UTC")
    write_parquet(tmp_path / "equity.parquet", pd.DataFrame({"date": dates, "equity": [100.] * 4}))
    write_parquet(tmp_path / "positions.parquet", pd.DataFrame({"date": list(dates) * 2, "ticker": ["A"] * 4 + ["B"] * 4, "position": [0, 1, 1, 0] * 2, "close": [100.] * 8}))
    market = tmp_path / "source.parquet"
    source_dates = pd.date_range("2019-12-31", periods=6, tz="UTC")
    write_parquet(market, pd.DataFrame({"date": list(source_dates) * 2, "ticker": ["A"] * 6 + ["B"] * 6, "open": [99.] * 12, "close": [100.] * 12, "adj_close": [50.] * 12, "secret_holdout_feature": ["unused"] * 12}))
    record = RunRecord("x", "x", "mt5", tmp_path, metadata={"tickers": ["A", "B"]}, tables={"equity": tmp_path / "equity.parquet", "positions": tmp_path / "positions.parquet", "market": market})
    data = load_run(record, include_details=True, tickers=["A"], start="2020-01-03")
    assert set(data.market.ticker) == {"A"}
    assert set(data.positions.ticker) == {"A"}
    assert data.market.date.tolist() == list(dates[2:])
    assert "secret_holdout_feature" not in data.market
    events = position_events(data, "A")
    assert events.kind.tolist() == ["exit_long"]
    assert events.date.tolist() == [dates[3]]


def test_light_asset_inventory_projects_one_run_without_loading_prices(tmp_path):
    write_parquet(tmp_path / "predictions.parquet", pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 3), "ticker": ["A", "B", "OTHER"], "candidate": ["gru"] * 3, "fold": [0, 0, 1], "seed": [7] * 3, "partition": ["outer"] * 3, "adj_close": [100.] * 3, "large_unused_feature": ["ignore"] * 3}))
    record = RunRecord("x", "x", "graph", tmp_path, tables={"positions": tmp_path / "predictions.parquet"}, selectors={"candidate": "gru", "fold": 0, "seed": 7, "partition": "outer"})
    data = RunData(record)
    assert asset_names(data) == ["A", "B"]
    assert data.positions.empty and data.market.empty


def test_trading_price_source_hash_and_variant_records(tmp_path):
    source = tmp_path / "source.parquet"
    dates = pd.date_range("2020-01-01", periods=2, tz="UTC")
    write_parquet(source, pd.DataFrame({"date": dates, "ticker": ["A"] * 2, "close": [100., 110.], "adj_close": [50., 55.]}))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    replay = tmp_path / "replay"
    write_json(replay / "report.json", {"prices_path": str(source), "inputs_sha256": {"data": digest}, "variants": {"base": {"price_basis": "split_adjusted"}}, "comparisons": [{"variant": "base", "metrics": {"net_return": .1}}]})
    write_parquet(replay / "base/equity.parquet", pd.DataFrame({"date": dates, "equity": [100., 110.]}))
    write_parquet(replay / "base/positions.parquet", pd.DataFrame({"date": dates, "ticker": ["A"] * 2, "quantity": [0., 1.]}))
    write_parquet(replay / "base/orders.parquet", pd.DataFrame({"date": [dates[1]], "ticker": ["A"], "execution_price": [109.], "quantity": [1.]}))
    catalog = discover_runs(replay)
    assert len(catalog.records) == 1
    record = catalog.records[0]
    data = load_run(record, include_details=True)
    assert asset_frame(data, "A").price.tolist() == [100., 110.]
    assert position_events(data, "A").source.tolist() == ["executed_orders"]
    write_parquet(source, pd.DataFrame({"date": dates, "ticker": ["A"] * 2, "close": [100., 999.]}))
    invalid = load_run(record, include_details=True)
    assert invalid.market.empty
    assert any("SHA256 mismatch" in warning for warning in invalid.warnings)
    assert not position_events(invalid, "A").empty


def test_legacy_benchmark_and_mt5_sign_decoder_are_coherent(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3, tz="UTC")
    legacy = tmp_path / "legacy"
    write_json(legacy / "manifest.json", {})
    write_json(legacy / "config.json", {"symbol": "A", "initial_capital": 100, "run_train_only": True})
    write_parquet(legacy / "backtest/equity_curve.parquet", pd.DataFrame({"equity": [99., 105., 98.]}, index=pd.DatetimeIndex(dates, name="timestamp")))
    write_parquet(legacy / "data/raw_market.parquet", pd.DataFrame({"date": dates, "ticker": ["A"] * 3, "close": [100., 110., 90.], "adj_close": [50., 52., 49.]}))
    mt5 = tmp_path / "mt5"
    write_json(mt5 / "metadata.json", {"tickers": ["A"]})
    for suffix, values in (("", [0., .3, -.2]), ("_sign", [0., 1., -1.])):
        write_parquet(mt5 / f"equity_curve{suffix}.parquet", pd.DataFrame({"date": dates, "model_equity": [100., 110., 105.], "buy_hold_equity": [100., 120., 100.]}))
        write_parquet(mt5 / f"positions{suffix}.parquet", pd.DataFrame({"date": dates, "ticker": ["A"] * 3, "close": [100.] * 3, "position": values, "sign_position": values if suffix else [0, 1, -1]}))
    catalog = discover_runs(tmp_path)
    backtest = next(record for record in catalog.records if record.family == "backtest")
    assert load_run(backtest).equity.benchmark_equity.tolist() == [100., 110., 90.]
    sign = next(record for record in catalog.records if record.metadata.get("decoder") == "sign")
    assert sign.tables["positions"].name == "positions_sign.parquet"
    assert load_run(sign, include_details=True).positions.position.tolist() == [0., 1., -1.]
